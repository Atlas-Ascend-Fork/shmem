/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define MULTI_INSTANCE
#include "kernel_operator.h"
#include "shmem.h"
#include "unittest/utils/sync_timeout_kernel.h"

// Shared matcher vectors for both the timeout-enabled and the default contexts. The last
// vector advances from INT32_MAX to INT32_MIN, which is the signed-overflow boundary.
template <typename Context>
ACLSHMEM_DEVICE void ExerciseSignalMatcher(__gm__ int32_t* signal, Context& context)
{
    // Current generation.
    aclshmemi_signal_set(signal, 41);
    aclshmemi_sync_wait(signal, 41, context);
    // Next generation.
    aclshmemi_signal_set(signal, 42);
    aclshmemi_sync_wait(signal, 41, context);
    // Signed wraparound -1 -> 0.
    aclshmemi_signal_set(signal, 0);
    aclshmemi_sync_wait(signal, -1, context);
    // Signed-overflow boundary INT32_MAX -> INT32_MIN.
    aclshmemi_signal_set(signal, (-2147483647 - 1));
    aclshmemi_sync_wait(signal, 2147483647, context);
}

template <SyncTimeoutMode Mode>
ACLSHMEM_DEVICE void SyncTimeoutExercise(
    uint64_t config, SyncTimeoutFault fault, uint32_t iterations, GM_ADDR output, uint64_t instance_id)
{
    util_set_ffts_config(config);
    aclshmemx_instance_ctx_set(instance_id);
    const uint32_t block = AscendC::GetBlockIdx();
    if ((fault == SyncTimeoutFault::MISSING_PE && aclshmem_my_pe() == 1) ||
        (fault == SyncTimeoutFault::MISSING_CORE && block == 1)) {
        return;
    }
    auto result = reinterpret_cast<__gm__ uint64_t*>(output + block * SYNC_TIMEOUT_TEST_SLOT_BYTES);
    for (uint32_t i = 0; i < iterations; ++i) {
        if constexpr (Mode == SyncTimeoutMode::V3) {
            aclshmemi_sync_npu_v3<true>(ACLSHMEM_TEAM_WORLD);
        } else if constexpr (Mode == SyncTimeoutMode::V4) {
            aclshmemi_sync_npu_v4<true>(ACLSHMEM_TEAM_WORLD);
        } else if constexpr (Mode == SyncTimeoutMode::VEC) {
            aclshmemx_sync_vec(ACLSHMEM_TEAM_WORLD);
        } else if constexpr (Mode == SyncTimeoutMode::SOFT) {
            aclshmemi_sync_core_soft();
        } else {
            // Exercise the real signal matcher, including next-generation and wraparound acceptance.
            auto signal = reinterpret_cast<__gm__ int32_t*>(result + 1);
            const uint32_t timeout_us = aclshmemi_get_sync_timeout_us();
            if (timeout_us == 0) {
                aclshmemi_sync_context<false> context(aclshmemi_sync_algorithm::LOCAL, ACLSHMEM_TEAM_WORLD, 0);
                ExerciseSignalMatcher(signal, context);
            } else {
                aclshmemi_sync_timeout context(aclshmemi_sync_algorithm::LOCAL, ACLSHMEM_TEAM_WORLD, timeout_us);
                ExerciseSignalMatcher(signal, context);
            }
        }
    }
    result[0] = iterations;
    result[1] = aclshmemi_get_sync_timeout_us();
    dcci_entire_cache();
}

#define DEFINE_SYNC_TIMEOUT_KERNEL(name, mode)                                                              \
    extern "C" ACLSHMEM_GLOBAL_VECTOR void name(                                                            \
        uint64_t config, SyncTimeoutFault fault, uint32_t iterations, GM_ADDR output, uint64_t instance_id) \
    {                                                                                                       \
        SyncTimeoutExercise<mode>(config, fault, iterations, output, instance_id);                          \
    }
DEFINE_SYNC_TIMEOUT_KERNEL(sync_timeout_v3_kernel, SyncTimeoutMode::V3)
DEFINE_SYNC_TIMEOUT_KERNEL(sync_timeout_v4_kernel, SyncTimeoutMode::V4)
DEFINE_SYNC_TIMEOUT_KERNEL(sync_timeout_vec_kernel, SyncTimeoutMode::VEC)
DEFINE_SYNC_TIMEOUT_KERNEL(sync_timeout_soft_kernel, SyncTimeoutMode::SOFT)
DEFINE_SYNC_TIMEOUT_KERNEL(sync_timeout_signal_kernel, SyncTimeoutMode::SIGNAL)
#undef DEFINE_SYNC_TIMEOUT_KERNEL

__attribute__((visibility("default"))) void sync_timeout_do(
    uint32_t blocks, void* stream, uint64_t config, SyncTimeoutMode mode, SyncTimeoutFault fault, uint32_t iterations,
    uint8_t* output, uint64_t instance_id)
{
    switch (mode) {
        case SyncTimeoutMode::V3:
            sync_timeout_v3_kernel<<<blocks, nullptr, stream>>>(config, fault, iterations, output, instance_id);
            break;
        case SyncTimeoutMode::V4:
            sync_timeout_v4_kernel<<<blocks, nullptr, stream>>>(config, fault, iterations, output, instance_id);
            break;
        case SyncTimeoutMode::VEC:
            sync_timeout_vec_kernel<<<blocks, nullptr, stream>>>(config, fault, iterations, output, instance_id);
            break;
        case SyncTimeoutMode::SOFT:
            sync_timeout_soft_kernel<<<blocks, nullptr, stream>>>(config, fault, iterations, output, instance_id);
            break;
        case SyncTimeoutMode::SIGNAL:
            sync_timeout_signal_kernel<<<blocks, nullptr, stream>>>(config, fault, iterations, output, instance_id);
            break;
    }
}
