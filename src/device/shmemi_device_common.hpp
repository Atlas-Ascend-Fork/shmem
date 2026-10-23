/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_COMMON_HPP
#define SHMEM_DEVICE_COMMON_HPP

#include "shmemi_def.h"

#include "shmemi_device_meta.h"

#if defined(MULTI_INSTANCE)
// Kernel-Level Global Variables
[[block_local]] __inline__ uint64_t g_instance_addr;

ACLSHMEM_DEVICE uint64_t aclshmemx_instance_ctx_get() { return g_instance_addr; }

ACLSHMEM_DEVICE void aclshmemx_instance_ctx_set(uint64_t instance_id)
{
    // Set aclshmem_instance Ctx, one instance have dram and hbm entity, so need << 1.
    g_instance_addr = SMEM_SHM_DEVICE_USER_CONTEXT_ADDR + (instance_id << 1) * SMEM_SHM_DEVICE_USER_CONTEXT_PRE_SIZE;
    return;
}

#else // no MULTI_INSTANCE

__attribute__((unavailable("requires macro MULTI_INSTANCE to be defined!"))) ACLSHMEM_DEVICE uint64_t
aclshmemx_instance_ctx_get();

__attribute__((unavailable("requires macro MULTI_INSTANCE to be defined!"))) ACLSHMEM_DEVICE void
aclshmemx_instance_ctx_set(uint64_t instance_id);

#endif // MULTI_INSTANCE

ACLSHMEM_DEVICE __gm__ aclshmem_device_host_state_t* aclshmemi_get_state()
{
#if defined(MULTI_INSTANCE)
    return reinterpret_cast<__gm__ aclshmem_device_host_state_t*>(g_instance_addr);
#else
    return reinterpret_cast<__gm__ aclshmem_device_host_state_t*>(aclshmemi_get_extra_context_addr(0));
#endif
}

ACLSHMEM_DEVICE int aclshmemi_get_my_pe() { return aclshmemi_get_state()->mype; }

ACLSHMEM_DEVICE int aclshmemi_get_total_pe() { return aclshmemi_get_state()->npes; }

ACLSHMEM_DEVICE uint64_t aclshmemi_get_heap_size() { return aclshmemi_get_state()->heap_size; }

ACLSHMEM_DEVICE void aclshmemi_debug_assert_remote_mr_addr(
    uint64_t local_ptr, uint64_t heap_base, uint64_t remote_mr_base)
{
    const uint64_t heap_size = aclshmemi_get_state()->heap_size;
    if (local_ptr < heap_base || local_ptr - heap_base >= heap_size ||
        remote_mr_base > UINT64_MAX - (local_ptr - heap_base)) {
        aclshmemi_kernel_abort("Invalid symmetric address for remote MR conversion\n");
    }
}

ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_remote_mr_addr(
    __gm__ const void* local_symmetric_ptr, const void* local_heap_base, uint64_t remote_mr_base)
{
    const uint64_t local_ptr = reinterpret_cast<uint64_t>(local_symmetric_ptr);
    const uint64_t heap_base = reinterpret_cast<uint64_t>(local_heap_base);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_debug_assert_remote_mr_addr, local_ptr, heap_base, remote_mr_base);
    const uint64_t offset = local_ptr - heap_base;
    return reinterpret_cast<__gm__ uint8_t*>(remote_mr_base + offset);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_store(__gm__ T* addr, T val)
{
    *((__gm__ T*)addr) = val;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_load(__gm__ T* cache)
{
    return *((__gm__ T*)cache);
}

#endif
