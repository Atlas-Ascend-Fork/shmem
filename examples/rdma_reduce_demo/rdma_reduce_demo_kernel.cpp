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
#include "rdma_reduce_demo_kernel.h"

#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
#error "rdma_reduce_demo requires the HNS_1825 backend."
#endif

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void rdma_reduce_demo_kernel(GM_ADDR data)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buffer;
    // The first UB allocation matches the barrier workspace configured on the Host at offset 0.
    pipe.InitBuffer(buffer, UB_BYTES);
    auto ub = buffer.Get<int32_t>();
    auto* workspace = reinterpret_cast<__ubuf__ int32_t*>(ub.GetPhyAddr());

    const int my_pe = aclshmem_my_pe();
    const int n_pes = aclshmem_n_pes();
    auto* source = reinterpret_cast<__gm__ int32_t*>(data);
    auto* add_dst = source + n_pes * ELEMENT_COUNT;
    auto* max_dst = add_dst + ELEMENT_COUNT;
    auto* min_dst = max_dst + ELEMENT_COUNT;
    auto* sat_src = reinterpret_cast<__gm__ half*>(min_dst + ELEMENT_COUNT);
    auto* sat_dst = sat_src + 1;
    auto* sat_workspace = reinterpret_cast<__ubuf__ half*>(workspace);

    AscendC::PipeBarrier<PIPE_ALL>();
    // All targets must be initialized before any PE starts its reduce operations.
    aclshmemx_roce_barrier_all();

    for (int peer = 0; peer < n_pes; ++peer) {
        // The local contribution is already included in the target arrays.
        if (peer == my_pe) {
            continue;
        }
        auto* src = source + peer * ELEMENT_COUNT;
        aclshmemx_roce_put_reduce_nbi(add_dst, src, workspace, ELEMENT_COUNT, peer, 0);
        aclshmemx_roce_put_reduce_nbi<ACLSHMEMX_REDUCE_MAX>(max_dst, src, workspace, ELEMENT_COUNT, peer, 0);
        aclshmemx_roce_put_reduce_nbi<ACLSHMEMX_REDUCE_MIN>(min_dst, src, workspace, ELEMENT_COUNT, peer, 0);
        aclshmemx_roce_put_reduce_nbi<ACLSHMEMX_REDUCE_ADD, aclshmemx_half_sat_t>(
            sat_dst, sat_src, sat_workspace, 1, peer, 0);
        // NBI calls only submit work; quiet completes all updates on this peer's QP0.
        aclshmemx_roce_quiet(peer, workspace, 0);
    }

    // Complete incoming updates before the Host reads or frees the local targets.
    aclshmemx_roce_barrier_all();
}

void launch_rdma_reduce_demo(uint32_t block_dim, void* stream, uint8_t* data)
{
    rdma_reduce_demo_kernel<<<block_dim, nullptr, stream>>>(data);
}
