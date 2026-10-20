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

#include "rdma_aggregate_demo_common.h"

using namespace rdma_aggregate_demo;

namespace {

ACLSHMEM_DEVICE void run_pointer_get(__gm__ uint32_t* data, __ubuf__ uint32_t* ub, int peer)
{
    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);

    aclshmemx_roce_get_nbi<uint32_t>(
        data + kGetPointerDestination0 * kElementCount, data + kGetPointerSource0 * kElementCount, ub, kElementCount,
        peer, kSyncId, defer_action);
    aclshmemx_roce_get_nbi<uint32_t>(
        data + kGetPointerDestination1 * kElementCount, data + kGetPointerSource1 * kElementCount, ub, kElementCount,
        peer, kSyncId, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void run_tensor_get(__gm__ uint32_t* data, AscendC::LocalTensor<uint32_t> ub_local, int peer)
{
    AscendC::GlobalTensor<uint32_t> dst0;
    AscendC::GlobalTensor<uint32_t> src0;
    AscendC::GlobalTensor<uint32_t> dst1;
    AscendC::GlobalTensor<uint32_t> src1;
    dst0.SetGlobalBuffer(data + kGetTensorDestination0 * kElementCount, kElementCount);
    src0.SetGlobalBuffer(data + kGetTensorSource0 * kElementCount, kElementCount);
    dst1.SetGlobalBuffer(data + kGetTensorDestination1 * kElementCount, kElementCount);
    src1.SetGlobalBuffer(data + kGetTensorSource1 * kElementCount, kElementCount);

    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);

    aclshmemx_roce_get_nbi<uint32_t>(dst0, src0, ub_local, kElementCount, peer, kSyncId, defer_action);
    aclshmemx_roce_get_nbi<uint32_t>(dst1, src1, ub_local, kElementCount, peer, kSyncId, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void run_pointer_put(__gm__ uint32_t* data, __ubuf__ uint32_t* ub, int peer)
{
    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);

    aclshmemx_roce_put_nbi<uint32_t>(
        data + kPutPointerDestination0 * kElementCount, data + kPutPointerSource0 * kElementCount, ub, kElementCount,
        peer, kSyncId, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + kPutPointerDestination1 * kElementCount, data + kPutPointerSource1 * kElementCount, ub, kElementCount,
        peer, kSyncId, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void run_pointer_put_loop_defer(__gm__ uint32_t* data, __ubuf__ uint32_t* ub, int peer)
{
    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);

    aclshmemx_roce_put_nbi<uint32_t>(
        data + kPutPointerLoopDeferDestination0 * kElementCount, data + kPutPointerLoopDeferSource0 * kElementCount, ub,
        kElementCount, peer, kSyncId, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + kPutPointerLoopDeferDestination1 * kElementCount, data + kPutPointerLoopDeferSource1 * kElementCount, ub,
        kElementCount, peer, kSyncId, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + kPutPointerLoopDeferDestination2 * kElementCount, data + kPutPointerLoopDeferSource2 * kElementCount, ub,
        kElementCount, peer, kSyncId, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + kPutPointerLoopDeferDestination3 * kElementCount, data + kPutPointerLoopDeferSource3 * kElementCount, ub,
        kElementCount, peer, kSyncId, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(
        data + kPutPointerLoopDeferDestination4 * kElementCount, data + kPutPointerLoopDeferSource4 * kElementCount, ub,
        kElementCount, peer, kSyncId, submit_action);

    aclshmemx_roce_barrier_all();
}

ACLSHMEM_DEVICE void run_tensor_put(__gm__ uint32_t* data, AscendC::LocalTensor<uint32_t> ub_local, int peer)
{
    AscendC::GlobalTensor<uint32_t> dst0;
    AscendC::GlobalTensor<uint32_t> src0;
    AscendC::GlobalTensor<uint32_t> dst1;
    AscendC::GlobalTensor<uint32_t> src1;
    dst0.SetGlobalBuffer(data + kPutTensorDestination0 * kElementCount, kElementCount);
    src0.SetGlobalBuffer(data + kPutTensorSource0 * kElementCount, kElementCount);
    dst1.SetGlobalBuffer(data + kPutTensorDestination1 * kElementCount, kElementCount);
    src1.SetGlobalBuffer(data + kPutTensorSource1 * kElementCount, kElementCount);

    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);

    aclshmemx_roce_put_nbi<uint32_t>(dst0, src0, ub_local, kElementCount, peer, kSyncId, defer_action);
    aclshmemx_roce_put_nbi<uint32_t>(dst1, src1, ub_local, kElementCount, peer, kSyncId, submit_action);

    aclshmemx_roce_barrier_all();
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_rdma_aggregate_demo(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> ub_buffer;
    pipe.InitBuffer(ub_buffer, kAggregateMaxUbBytes);
    AscendC::LocalTensor<uint32_t> ub_local =
        ub_buffer.GetWithOffset<uint32_t>(kAggregateMaxUbBytes / sizeof(uint32_t), 0);
    __ubuf__ uint32_t* ub = (__ubuf__ uint32_t*)ub_local.GetPhyAddr();

    const int my_pe = static_cast<int>(aclshmem_my_pe());
    const int n_pes = static_cast<int>(aclshmem_n_pes());
    if (n_pes < 2 || my_pe < 0 || my_pe >= n_pes) {
        return;
    }
    // Exchange with the next PE so the example scales to multi-PE and multi-node runs.
    const int peer = (my_pe + 1) % n_pes;
    __gm__ uint32_t* data = (__gm__ uint32_t*)gva;

    // Wait until both PEs have initialized their symmetric buffers before starting the first batch.
    aclshmemx_roce_barrier_all();

    run_pointer_get(data, ub, peer);
    run_tensor_get(data, ub_local, peer);
    run_pointer_put(data, ub, peer);
    run_pointer_put_loop_defer(data, ub, peer);
    run_tensor_put(data, ub_local, peer);
}

void launch_rdma_aggregate_demo(uint32_t block_dim, void* stream, uint8_t* gva)
{
    device_rdma_aggregate_demo<<<block_dim, nullptr, stream>>>(gva);
}
