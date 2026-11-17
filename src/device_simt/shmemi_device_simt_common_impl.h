/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_SIMT_COMMON_IMPL_H
#define SHMEMI_DEVICE_SIMT_COMMON_IMPL_H

#include "shmemi_def.h"

#include "shmemi_device_meta.h"
#include "shmemi_device_simt_meta.h"
#include "utils/shmemi_simt_kernel_debug.h"

namespace simt {

__simt_callee__ inline bool aclshmemi_is_valid_hbm_instance_id(uint64_t instance_id)
{
    // An HBM-only instance may use the last even entity without a DRAM companion.
    return instance_id <= (SMEM_OBJECT_NUM_MAX - 1) / 2;
}

__simt_callee__ inline void aclshmemi_debug_check_instance_id(uint64_t instance_id)
{
    if (!aclshmemi_is_valid_hbm_instance_id(instance_id)) {
        simt::aclshmemi_kernel_abort("invalid instance_id=%lu\n", instance_id);
    }
}

#if defined(MULTI_INSTANCE)
// Compiler-allocated UB shared by this block. Set it in every VF before use.
__inline__ __ubuf__ uint64_t g_simt_instance_addr;

__simt_callee__ inline void aclshmemx_instance_ctx_set(uint64_t instance_id)
{
    if (!aclshmemi_is_valid_hbm_instance_id(instance_id)) {
        ACLSHMEM_DEBUG_FUNC(simt::aclshmemi_debug_check_instance_id, instance_id);
        __trap();
        return;
    }
    asc_syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        g_simt_instance_addr =
            SMEM_SHM_DEVICE_USER_CONTEXT_ADDR + (instance_id << 1) * SMEM_SHM_DEVICE_USER_CONTEXT_PRE_SIZE;
    }
    asc_syncthreads();
}

__simt_callee__ inline uint64_t aclshmemx_instance_ctx_get() { return g_simt_instance_addr; }
#else
__attribute__((unavailable("requires macro MULTI_INSTANCE to be defined!"))) __simt_callee__ inline void
aclshmemx_instance_ctx_set(uint64_t instance_id);

__attribute__((unavailable("requires macro MULTI_INSTANCE to be defined!"))) __simt_callee__ inline uint64_t
aclshmemx_instance_ctx_get();
#endif

__simt_callee__ inline __gm__ aclshmem_device_host_state_simt_t* aclshmemi_get_state()
{
#if defined(MULTI_INSTANCE)
    return reinterpret_cast<__gm__ aclshmem_device_host_state_simt_t*>(simt::aclshmemx_instance_ctx_get());
#else
    return reinterpret_cast<__gm__ aclshmem_device_host_state_simt_t*>(aclshmemi_get_extra_context_addr(0));
#endif
}

__simt_callee__ inline int aclshmemi_get_my_pe() { return aclshmemi_get_state()->mype; }

__simt_callee__ inline int aclshmemi_get_total_pe() { return aclshmemi_get_state()->npes; }

__simt_callee__ inline uint64_t aclshmemi_get_heap_size() { return aclshmemi_get_state()->heap_size; }

template <typename T>
__simt_callee__ inline void aclshmemi_store(__gm__ T* addr, T val)
{
    *((__gm__ T*)addr) = val;
}

template <typename T>
__simt_callee__ inline T aclshmemi_load(__gm__ T* cache)
{
    return *((__gm__ T*)cache);
}

typedef enum {
    ACLSHMEMI_THREADGROUP_THREAD = 0,
    aclshmemi_threadgroup_thread = 0,
    ACLSHMEMI_THREADGROUP_WARP = 1,
    aclshmemi_threadgroup_warp = 1,
    ACLSHMEMI_THREADGROUP_BLOCK = 2,
    aclshmemi_threadgroup_block = 2
} thread_group_t;

template <thread_group_t scope>
__simt_callee__ inline void aclshmemi_threadgroup_sync()
{
    if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_THREAD) {
        return;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_WARP) {
        // CANN 9.2: warp RMA requires uniform whole-block participation.
        asc_syncthreads();
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_BLOCK) {
        asc_syncthreads();
    }
}

template <thread_group_t scope>
__simt_callee__ inline size_t aclshmemi_thread_id_in_threadgroup()
{
    if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_THREAD) {
        return 0;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_WARP) {
        return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) % warpSize;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_BLOCK) {
        return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    }
}

template <thread_group_t scope>
__simt_callee__ inline size_t aclshmemi_threadgroup_size()
{
    if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_THREAD) {
        return 1;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_WARP) {
        const size_t threads = blockDim.x * blockDim.y * blockDim.z;
        const size_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        const size_t remaining = threads - (tid / warpSize) * warpSize;
        return remaining < warpSize ? remaining : warpSize;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_BLOCK) {
        return blockDim.x * blockDim.y * blockDim.z;
    }
}

} // namespace simt

#endif
