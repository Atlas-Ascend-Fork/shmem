/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

#include "shmem.h"
constexpr uint64_t MESSAGE_SIZE = 64;

extern "C" ACLSHMEM_GLOBAL_VECTOR void RDMAGetTestLowLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR dest_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmemx_roce_get_nbi(dest_addr, dest_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer, 0);
    }
}

void test_rdma_get_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAGetTestLowLevel<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void RDMAPutTestLowLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        src_addr = gva + rank * MESSAGE_SIZE;
        aclshmemx_roce_put_nbi(src_addr, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer, 0);
    }
}

void test_rdma_put_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAPutTestLowLevel<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void RDMAGetTestHighLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR dest_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmem_uint8_get_nbi(dest_addr, dest_addr, MESSAGE_SIZE, peer);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void test_rdma_get_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAGetTestHighLevel<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void RDMAPutTestHighLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        src_addr = gva + rank * MESSAGE_SIZE;
        aclshmem_uint8_put_nbi(src_addr, src_addr, MESSAGE_SIZE, peer);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void test_rdma_put_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAPutTestHighLevel<<<block_dim, nullptr, stream>>>(gva, config);
}

// Aggregate stage/commit entries (non-QP aclshmemx_roce_get_nbi/put_nbi with defer/submit actions) are
// dispatched to the XSCALE and IN_DIE backends; HNS_1825 has no aggregate support. IN_DIE is the default
// backend when no backend macro is defined.
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)

namespace {

constexpr uint32_t RDMA_AGGREGATE_ELEMENT_COUNT = 16;
constexpr uint32_t RDMA_AGGREGATE_UB_BYTES = 64 + 128 * 5;
constexpr uint32_t RDMA_AGGREGATE_SYNC_ID = 0;

constexpr uint32_t RDMA_AGGREGATE_GET_POINTER_SOURCE0 = 0;
constexpr uint32_t RDMA_AGGREGATE_GET_POINTER_DESTINATION0 = 1;
constexpr uint32_t RDMA_AGGREGATE_GET_POINTER_SOURCE1 = 2;
constexpr uint32_t RDMA_AGGREGATE_GET_POINTER_DESTINATION1 = 3;
constexpr uint32_t RDMA_AGGREGATE_GET_TENSOR_SOURCE0 = 4;
constexpr uint32_t RDMA_AGGREGATE_GET_TENSOR_DESTINATION0 = 5;
// Slots 6 and 7 remain reserved to keep the existing slot layout stable.
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_SOURCE0 = 8;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_DESTINATION0 = 9;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_SOURCE1 = 10;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_DESTINATION1 = 11;
constexpr uint32_t RDMA_AGGREGATE_PUT_TENSOR_SOURCE0 = 12;
constexpr uint32_t RDMA_AGGREGATE_PUT_TENSOR_DESTINATION0 = 13;
constexpr uint32_t RDMA_AGGREGATE_PUT_TENSOR_SOURCE1 = 14;
constexpr uint32_t RDMA_AGGREGATE_PUT_TENSOR_DESTINATION1 = 15;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE0 = 16;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION0 = 17;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE1 = 18;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION1 = 19;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE2 = 20;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION2 = 21;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE3 = 22;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION3 = 23;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE4 = 24;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION4 = 25;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE0 = 26;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION0 = 27;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE1 = 28;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION1 = 29;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE2 = 30;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION2 = 31;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE3 = 32;
constexpr uint32_t RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION3 = 33;

ACLSHMEM_DEVICE void rdma_aggregate_pointer_get(__gm__ uint32_t* data, __ubuf__ uint32_t* ub, int peer)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);

    aclshmemx_roce_get_nbi<uint32_t>(
        data + RDMA_AGGREGATE_GET_POINTER_DESTINATION0 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_GET_POINTER_SOURCE0 * RDMA_AGGREGATE_ELEMENT_COUNT, ub, RDMA_AGGREGATE_ELEMENT_COUNT,
        peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_get_nbi<uint32_t>(
        data + RDMA_AGGREGATE_GET_POINTER_DESTINATION1 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_GET_POINTER_SOURCE1 * RDMA_AGGREGATE_ELEMENT_COUNT, ub, RDMA_AGGREGATE_ELEMENT_COUNT,
        peer, RDMA_AGGREGATE_SYNC_ID, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void rdma_aggregate_tensor_get(__gm__ uint32_t* data, AscendC::LocalTensor<uint32_t> ub_local, int peer)
{
    AscendC::GlobalTensor<uint32_t> dst0;
    AscendC::GlobalTensor<uint32_t> src0;
    dst0.SetGlobalBuffer(
        data + RDMA_AGGREGATE_GET_TENSOR_DESTINATION0 * RDMA_AGGREGATE_ELEMENT_COUNT, RDMA_AGGREGATE_ELEMENT_COUNT);
    src0.SetGlobalBuffer(
        data + RDMA_AGGREGATE_GET_TENSOR_SOURCE0 * RDMA_AGGREGATE_ELEMENT_COUNT, RDMA_AGGREGATE_ELEMENT_COUNT);

    aclshmemx_submit_state_t submit_state{};
    aclshmemx_submit_t submit_action(submit_state);

    aclshmemx_roce_get_nbi<uint32_t>(
        dst0, src0, ub_local, RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void rdma_aggregate_pointer_put(__gm__ uint32_t* data, __ubuf__ uint32_t* ub, int peer)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);

    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_DESTINATION0 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_SOURCE0 * RDMA_AGGREGATE_ELEMENT_COUNT, ub, RDMA_AGGREGATE_ELEMENT_COUNT,
        peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_DESTINATION1 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_SOURCE1 * RDMA_AGGREGATE_ELEMENT_COUNT, ub, RDMA_AGGREGATE_ELEMENT_COUNT,
        peer, RDMA_AGGREGATE_SYNC_ID, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void rdma_aggregate_pointer_put_loop_defer(__gm__ uint32_t* data, __ubuf__ uint32_t* ub, int peer)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);

    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION0 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE0 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION1 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE1 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION2 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE2 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION3 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE3 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_DESTINATION4 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_LOOP_DEFER_SOURCE4 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void rdma_aggregate_pointer_put_reuse_actions(__gm__ uint32_t* data, __ubuf__ uint32_t* ub, int peer)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);

    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION0 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE0 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION1 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE1 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, submit_action);
    if (submit_state.pending_count != 0) {
        aclshmemi_kernel_abort(
            "RDMA aggregate action reuse: pending_count after first submit is %u, expected 0\n",
            submit_state.pending_count);
        return;
    }

    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION2 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE2 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_DESTINATION3 * RDMA_AGGREGATE_ELEMENT_COUNT,
        data + RDMA_AGGREGATE_PUT_POINTER_REUSE_SOURCE3 * RDMA_AGGREGATE_ELEMENT_COUNT, ub,
        RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, submit_action);
    if (submit_state.pending_count != 0) {
        aclshmemi_kernel_abort(
            "RDMA aggregate action reuse: pending_count after second submit is %u, expected 0\n",
            submit_state.pending_count);
        return;
    }

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void rdma_aggregate_tensor_put(__gm__ uint32_t* data, AscendC::LocalTensor<uint32_t> ub_local, int peer)
{
    AscendC::GlobalTensor<uint32_t> dst0;
    AscendC::GlobalTensor<uint32_t> src0;
    AscendC::GlobalTensor<uint32_t> dst1;
    AscendC::GlobalTensor<uint32_t> src1;
    dst0.SetGlobalBuffer(
        data + RDMA_AGGREGATE_PUT_TENSOR_DESTINATION0 * RDMA_AGGREGATE_ELEMENT_COUNT, RDMA_AGGREGATE_ELEMENT_COUNT);
    src0.SetGlobalBuffer(
        data + RDMA_AGGREGATE_PUT_TENSOR_SOURCE0 * RDMA_AGGREGATE_ELEMENT_COUNT, RDMA_AGGREGATE_ELEMENT_COUNT);
    dst1.SetGlobalBuffer(
        data + RDMA_AGGREGATE_PUT_TENSOR_DESTINATION1 * RDMA_AGGREGATE_ELEMENT_COUNT, RDMA_AGGREGATE_ELEMENT_COUNT);
    src1.SetGlobalBuffer(
        data + RDMA_AGGREGATE_PUT_TENSOR_SOURCE1 * RDMA_AGGREGATE_ELEMENT_COUNT, RDMA_AGGREGATE_ELEMENT_COUNT);

    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);

    aclshmemx_roce_put_nbi<uint32_t>(
        dst0, src0, ub_local, RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        dst1, src1, ub_local, RDMA_AGGREGATE_ELEMENT_COUNT, peer, RDMA_AGGREGATE_SYNC_ID, submit_action);

    aclshmemx_roce_barrier_all();
}

} // namespace

extern "C" ACLSHMEM_GLOBAL_VECTOR void RDMAAggregateTest(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, RDMA_AGGREGATE_UB_BYTES);
    AscendC::LocalTensor<uint32_t> ub_local =
        buf.GetWithOffset<uint32_t>(RDMA_AGGREGATE_UB_BYTES / sizeof(uint32_t), 0);
    __ubuf__ uint32_t* ub = (__ubuf__ uint32_t*)ub_local.GetPhyAddr();

    const int my_pe = static_cast<int>(aclshmem_my_pe());
    const int n_pes = static_cast<int>(aclshmem_n_pes());
    if (n_pes < 2) {
        return;
    }

    const int peer = (my_pe + 1) % n_pes;
    __gm__ uint32_t* data = (__gm__ uint32_t*)gva;

    aclshmemx_roce_barrier_all();
    rdma_aggregate_pointer_get(data, ub, peer);
    rdma_aggregate_tensor_get(data, ub_local, peer);
    rdma_aggregate_pointer_put(data, ub, peer);
    rdma_aggregate_pointer_put_loop_defer(data, ub, peer);
    rdma_aggregate_pointer_put_reuse_actions(data, ub, peer);
    rdma_aggregate_tensor_put(data, ub_local, peer);
}

void test_rdma_aggregate(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAAggregateTest<<<block_dim, nullptr, stream>>>(gva, config);
}

#endif // !ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
