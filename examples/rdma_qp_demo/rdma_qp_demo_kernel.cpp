/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "shmem.h"

namespace {
constexpr uint32_t AGGREGATE_OP_COUNT = 8;
constexpr uint32_t AGGREGATE_UB_BYTES = 64 + 128 * AGGREGATE_OP_COUNT;

enum class DemoOperation : int32_t {
    PUT = 0,
    GET = 1,
    AGGREGATE_PUT = 2,
    AGGREGATE_GET = 3,
};

template <typename T>
__aicore__ inline void AggregatePut(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t slice_elements, int peer, uint32_t qp_idx, uint32_t sync_id)
{
    const uint32_t chunk_elements = slice_elements / AGGREGATE_OP_COUNT;
    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);
    for (uint32_t op = 0; op < AGGREGATE_OP_COUNT; ++op) {
        __gm__ T* chunk_dst = dst + op * chunk_elements;
        __gm__ T* chunk_src = src + op * chunk_elements;
        if (op + 1 == AGGREGATE_OP_COUNT) {
            aclshmemx_roce_qp_put_nbi(chunk_dst, chunk_src, buf, chunk_elements, peer, qp_idx, sync_id, submit_action);
        } else {
            aclshmemx_roce_qp_put_nbi(chunk_dst, chunk_src, buf, chunk_elements, peer, qp_idx, sync_id, defer_action);
        }
    }
}

template <typename T>
__aicore__ inline void AggregateGet(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t slice_elements, int peer, uint32_t qp_idx, uint32_t sync_id)
{
    const uint32_t chunk_elements = slice_elements / AGGREGATE_OP_COUNT;
    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);
    for (uint32_t op = 0; op < AGGREGATE_OP_COUNT; ++op) {
        __gm__ T* chunk_dst = dst + op * chunk_elements;
        __gm__ T* chunk_src = src + op * chunk_elements;
        if (op + 1 == AGGREGATE_OP_COUNT) {
            aclshmemx_roce_qp_get_nbi(chunk_dst, chunk_src, buf, chunk_elements, peer, qp_idx, sync_id, submit_action);
        } else {
            aclshmemx_roce_qp_get_nbi(chunk_dst, chunk_src, buf, chunk_elements, peer, qp_idx, sync_id, defer_action);
        }
    }
}
} // namespace

/**
 * @brief Execute a single-QP or multi-QP RDMA transfer demonstration.
 * @note @p elements must be divisible by the QP count. Aggregate operations additionally require each QP
 * slice to be divisible by AGGREGATE_OP_COUNT.
 */
extern "C" __global__ __vector__ void rdma_qp_demo_kernel(
    uint64_t ffts_addr, GM_ADDR symmetric, uint64_t elements, int32_t peer, int32_t operation, uint32_t sync_id)
{
    util_set_ffts_config(ffts_addr);
    const uint32_t core_idx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t qp_idx = core_idx;
    const uint32_t qp_num = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t slice_elements = static_cast<uint32_t>(elements / qp_num);
    const uint64_t slice_offset = static_cast<uint64_t>(qp_idx) * slice_elements;

    __gm__ uint8_t* src = symmetric + slice_offset;
    __gm__ uint8_t* dst = symmetric + elements + slice_offset;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> scratch;
    pipe.InitBuffer(scratch, AGGREGATE_UB_BYTES);
    __ubuf__ uint8_t* ub = reinterpret_cast<__ubuf__ uint8_t*>(scratch.Get<uint8_t>().GetPhyAddr());

    if (core_idx == 0) {
        aclshmemx_roce_sync_all();
    }
    AscendC::SyncAll<true>();

    const auto op = static_cast<DemoOperation>(operation);
    if (op == DemoOperation::PUT) {
        aclshmemx_roce_qp_put_nbi(dst, src, ub, slice_elements, peer, qp_idx, sync_id);
    } else if (op == DemoOperation::GET) {
        aclshmemx_roce_qp_get_nbi(dst, src, ub, slice_elements, peer, qp_idx, sync_id);
    } else if (op == DemoOperation::AGGREGATE_PUT) {
        AggregatePut(dst, src, ub, slice_elements, peer, qp_idx, sync_id);
    } else {
        AggregateGet(dst, src, ub, slice_elements, peer, qp_idx, sync_id);
    }
    aclshmemx_roce_qp_quiet(peer, qp_idx, ub, sync_id);
    AscendC::SyncAll<true>();
    if (core_idx == 0) {
        aclshmemx_roce_sync_all();
    }
    AscendC::SyncAll<true>();
}

void launch_rdma_qp_demo(
    uint32_t block_dim, void* stream, uint64_t ffts_addr, uint8_t* symmetric, uint64_t elements, int32_t peer,
    int32_t operation, uint32_t sync_id)
{
    rdma_qp_demo_kernel<<<block_dim, nullptr, stream>>>(ffts_addr, symmetric, elements, peer, operation, sync_id);
}
