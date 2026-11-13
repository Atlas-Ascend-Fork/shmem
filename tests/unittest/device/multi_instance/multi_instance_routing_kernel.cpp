/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#define MULTI_INSTANCE
#include "shmem.h"

// One AIV per rank, one ring successor, QP 0: exactly one queue writer.
// Source [0,4096), receive [4096,8192), guard [8192,12288), in int32 elements.
// PIPE_S publishes WQEs through scalar stores; no UB scratch is used.
// Completion: UDMA quiet -> stream sync -> host control barrier -> D2H.
// No device collective and no AIV-to-AIV shared storage in this routing test.
constexpr uint32_t SLOT = 4096;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void instance_udma_kernel(
    GM_ADDR heap, uint32_t count, int peer, uint64_t instance_id, GM_ADDR identity)
{
    aclshmemx_instance_ctx_set(instance_id);
    __gm__ int32_t* ids = (__gm__ int32_t*)identity;
    ids[0] = aclshmem_my_pe();
    ids[1] = aclshmem_n_pes();
    aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(
        heap + SLOT * sizeof(int32_t), heap, nullptr, count * sizeof(int32_t), peer, 0);
    aclshmemx_udma_quiet(peer);
}

// No metadata is dereferenced here. Invalid IDs exercise the same predicate as
// the production guards without intentionally trapping the device.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void instance_id_boundary_kernel(GM_ADDR result)
{
    const uint64_t ids[] = {0, 254, 255, 256, UINT64_MAX};
    __gm__ uint64_t* output = reinterpret_cast<__gm__ uint64_t*>(result);
    for (uint32_t i = 0; i < 5; ++i) {
        output[2 * i] = aclshmemi_is_valid_hbm_instance_id(ids[i]);
        output[2 * i + 1] = UINT64_MAX;
        if (aclshmemi_is_valid_hbm_instance_id(ids[i])) {
            aclshmemx_instance_ctx_set(ids[i]);
            output[2 * i + 1] = aclshmemi_get_current_entity_id();
        }
    }
}

extern "C" void multi_instance_id_boundary(void* stream, void* result)
{
    instance_id_boundary_kernel<<<1, nullptr, stream>>>(static_cast<uint8_t*>(result));
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void instance_mte_kernel(
    GM_ADDR heap, uint32_t count, int peer, uint64_t instance_id, GM_ADDR identity)
{
    aclshmemx_instance_ctx_set(instance_id);
    __gm__ int32_t* ids = reinterpret_cast<__gm__ int32_t*>(identity);
    ids[0] = aclshmem_my_pe();
    ids[1] = aclshmem_n_pes();
    // Only this MTE test needs staging. One AIV owns all 4096 UB bytes;
    // the MTE helper orders MTE2 -> MTE3 and drains each reused chunk.
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buffer;
    pipe.InitBuffer(buffer, 4096);
    auto ub = buffer.Get<uint8_t>();
    aclshmemx_mte_put_nbi<uint8_t>(
        heap + SLOT * sizeof(int32_t), heap, reinterpret_cast<__ubuf__ uint8_t*>(ub.GetPhyAddr()), 4096,
        count * sizeof(int32_t), peer, 0);
    aclshmemx_mte_quiet();
}

extern "C" void multi_instance_mte_transfer(
    void* stream, void* heap, uint32_t count, int peer, uint64_t id, void* identity)
{
    instance_mte_kernel<<<1, nullptr, stream>>>(
        static_cast<uint8_t*>(heap), count, peer, id, static_cast<uint8_t*>(identity));
}

extern "C" void multi_instance_routing_transfer(
    void* stream, void* heap, uint32_t count, int peer, uint64_t id, void* identity)
{
    instance_udma_kernel<<<1, nullptr, stream>>>((uint8_t*)heap, count, peer, id, (uint8_t*)identity);
}

// Each QP writes a disjoint half of the receive span. Deliberately do not quiet
// here: Host quiet drains QP0 for instance 0 and both QPs for nonzero instances.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void instance_udma_host_quiet_producer(
    GM_ADDR heap, uint32_t count, int peer, uint64_t id, GM_ADDR identity)
{
    aclshmemx_instance_ctx_set(id);
    auto result = reinterpret_cast<__gm__ int32_t*>(identity);
    result[0] = aclshmem_my_pe();
    result[1] = aclshmem_n_pes();
    for (uint32_t qp = 0; qp < 2; ++qp) {
        const uint32_t offset = qp * (count / 2) * sizeof(int32_t);
        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S>(
            heap + SLOT * sizeof(int32_t) + offset, heap + offset, nullptr, (count / 2) * sizeof(int32_t), peer, qp);
        auto wq = aclshmemi_udma_get_qp_ctx(aclshmemi_udma_qp_info_fetch(), peer, qp);
        auto state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(wq->state_addr);
        result[2 + qp * 2] = state->cq_tail;
        result[3 + qp * 2] = state->cqe_cnt;
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void instance_udma_host_quiet_snapshot(
    int peer, uint64_t id, GM_ADDR identity)
{
    aclshmemx_instance_ctx_set(id);
    auto result = reinterpret_cast<__gm__ int32_t*>(identity);
    for (uint32_t qp = 0; qp < 2; ++qp) {
        auto wq = aclshmemi_udma_get_qp_ctx(aclshmemi_udma_qp_info_fetch(), peer, qp);
        auto state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(wq->state_addr);
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(state), sizeof(*state));
        result[6 + qp * 2] = state->cq_tail;
        result[7 + qp * 2] = state->cqe_cnt;
    }
    // Capture the untouched QP1 consumer before explicit completion for instance 0.
    // This keeps the receive buffers safe without attributing QP1 to Host quiet.
    if (id == 0) {
        aclshmemx_udma_qp_quiet(peer, 1);
    }
}

extern "C" void multi_instance_host_quiet_produce(
    void* stream, void* heap, uint32_t count, int peer, uint64_t id, void* identity)
{
    instance_udma_host_quiet_producer<<<1, nullptr, stream>>>(
        static_cast<uint8_t*>(heap), count, peer, id, static_cast<uint8_t*>(identity));
}

extern "C" void multi_instance_host_quiet_observe(void* stream, int peer, uint64_t id, void* identity)
{
    instance_udma_host_quiet_snapshot<<<1, nullptr, stream>>>(peer, id, static_cast<uint8_t*>(identity));
}
