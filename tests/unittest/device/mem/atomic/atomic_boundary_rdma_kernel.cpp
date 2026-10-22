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

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_atomic_high_half_uint32_kernel(
    __gm__ uint32_t* gva, __gm__ uint32_t* result, uint32_t value, uint32_t compare, bool compare_swap, uint64_t config)
{
    util_set_ffts_config(config);
    const int peer = (aclshmem_my_pe() + 1) % aclshmem_n_pes();
    // The second uint32_t is the high half of the aligned 64-bit word.
    uint32_t old = compare_swap ? aclshmemx_roce_atomic_compare_swap(gva + 1, compare, value, peer) :
                                  aclshmemx_roce_atomic_fetch_add(gva + 1, value, peer);
    AscendC::GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(result);
    output.SetValue(0, old);
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(result), sizeof(uint32_t));
}

void test_rdma_atomic_high_half_uint32_do(
    uint32_t block_dim, void* stream, uint32_t* gva, uint32_t* result, uint32_t value, uint32_t compare,
    bool compare_swap, uint64_t config)
{
    test_rdma_atomic_high_half_uint32_kernel<<<block_dim, nullptr, stream>>>(
        gva, result, value, compare, compare_swap, config);
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_atomic_contended_fetch_add_uint32_kernel(
    __gm__ uint32_t* gva, __gm__ uint32_t* result, uint32_t iterations, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(result);
    // All sources update PE0's same high half; returned values stay local to each source.
    for (uint32_t i = 0; i < iterations; ++i) {
        output.SetValue(i, aclshmemx_roce_atomic_fetch_add(gva + 1, 1U, 0));
    }
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(result), iterations * sizeof(uint32_t));
}

void test_rdma_atomic_contended_fetch_add_uint32_do(
    uint32_t block_dim, void* stream, uint32_t* gva, uint32_t* result, uint32_t iterations, uint64_t config)
{
    test_rdma_atomic_contended_fetch_add_uint32_kernel<<<block_dim, nullptr, stream>>>(gva, result, iterations, config);
}

#endif // ACLSHMEMI_RDMA_K_BACKEND_XSCALE || ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
