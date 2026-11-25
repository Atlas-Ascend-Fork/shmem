/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEMI_SYNC_TIMEOUT_H
#define SHMEMI_SYNC_TIMEOUT_H

#include "kernel_operator.h"
#include "shmemi_device_common.h"
#include "utils/shmemi_kernel_debug.h"
#include "../../host_device/shmemi_common_types.h"

constexpr uint32_t ACLSHMEMI_SYNC_TIMEOUT_K = 1024;
#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
constexpr uint64_t ACLSHMEMI_SYNC_CYCLES_PER_US = 50;
#else
constexpr uint64_t ACLSHMEMI_SYNC_CYCLES_PER_US = 1000;
#endif

enum class aclshmemi_sync_algorithm : int32_t { LOCAL = 0, V3 = 3, V4 = 4 };
enum class aclshmemi_sync_phase : int32_t { LOCAL_ENTER = 0, LOCAL_PUBLISH = 1, LOCAL_EXIT = 2, SEND = 3, WAIT = 4 };

ACLSHMEM_DEVICE uint64_t aclshmemi_sync_cycle()
{
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
    // CANN exposes GetSystemCycle on AIV. AIC executes hardware barriers only;
    // it never enters the software polling paths protected by this timer.
    return 0;
#else
    return static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
}

ACLSHMEM_DEVICE uint32_t aclshmemi_get_sync_timeout_us()
{
    auto config = reinterpret_cast<__gm__ aclshmemi_device_runtime_config_t*>(
        reinterpret_cast<__gm__ uint8_t*>(aclshmemi_get_state()) + sizeof(aclshmem_device_host_state_t));
    return config->sync_timeout_us;
}

// Advance a barrier generation without signed overflow. The wait matchers accept both the
// current and the next generation, so wrapping INT32_MAX -> INT32_MIN keeps the sequence
// continuous on implementations where the value is stored as int32_t.
ACLSHMEM_DEVICE int32_t aclshmemi_next_generation(int32_t current)
{
    return current == 2147483647 ? (-2147483647 - 1) : current + 1;
}

// The disabled form is the primary definition: it carries no state and performs no
// timing. The enabled specialization below adds the timer fields and checks.
template <bool EnableTimeout>
struct aclshmemi_sync_context {
    ACLSHMEM_DEVICE aclshmemi_sync_context(aclshmemi_sync_algorithm, int32_t, uint32_t) {}
};

template <>
struct aclshmemi_sync_context<true> {
    uint64_t start = aclshmemi_sync_cycle();
    uint64_t timeout_us;
    aclshmemi_sync_algorithm algorithm;
    int32_t team;
    aclshmemi_sync_phase phase = aclshmemi_sync_phase::LOCAL_ENTER;
    int32_t generation = -1;
    int32_t round = -1;
    int32_t slot = -1;
    int32_t peer = -1; // World PE; local waits use the local PE and wait_core.
    int32_t local_round = -1;
    int32_t wait_core = -1;

    ACLSHMEM_DEVICE aclshmemi_sync_context(
        aclshmemi_sync_algorithm algorithm_, int32_t team_, uint32_t timeout_us_ = aclshmemi_get_sync_timeout_us())
        : timeout_us(timeout_us_), algorithm(algorithm_), team(team_)
    {}

    ACLSHMEM_DEVICE void check(int64_t expected, int64_t observed)
    {
        const uint64_t elapsed = aclshmemi_sync_cycle() - start;
        if (elapsed < timeout_us * ACLSHMEMI_SYNC_CYCLES_PER_US) {
            return;
        }
        int32_t core_type = 0;
        if ASCEND_IS_AIV {
            core_type = 1;
        }
        if ASCEND_IS_AIC {
            core_type = 2;
        }
        aclshmemi_kernel_abort(
            "SHMEM_SYNC_TIMEOUT instance=%u team=%d pe=%d core_type=%d core=%u algorithm=%d phase=%d "
            "generation=%d round=%d slot=%d peer=%d wait_core=%d local_round=%d expected=%lld observed=%lld "
            "elapsed_us=%llu timeout_us=%llu\n",
            static_cast<uint32_t>(aclshmemi_get_current_entity_id() / 2), team, aclshmemi_get_state()->mype, core_type,
            AscendC::GetBlockIdx(), static_cast<int32_t>(algorithm), static_cast<int32_t>(phase), generation, round,
            slot, peer, wait_core, local_round, static_cast<long long>(expected), static_cast<long long>(observed),
            static_cast<unsigned long long>(elapsed / ACLSHMEMI_SYNC_CYCLES_PER_US),
            static_cast<unsigned long long>(timeout_us));
    }
};

using aclshmemi_sync_timeout = aclshmemi_sync_context<true>;

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t* address, int32_t expected);

ACLSHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t* addr);

template <bool EnableTimeout>
ACLSHMEM_DEVICE void aclshmemi_sync_wait(
    __gm__ int32_t* address, int32_t expected, aclshmemi_sync_context<EnableTimeout>& context)
{
    if constexpr (!EnableTimeout) {
        (void)aclshmemi_signal_wait_until_eq_for_barrier(address, expected);
    } else {
        uint32_t failures = 0;
        while (true) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(address));
            const int32_t observed = *address;
            if (observed == expected || static_cast<uint32_t>(observed) == static_cast<uint32_t>(expected) + 1U) {
                return;
            }
            if ((++failures & (ACLSHMEMI_SYNC_TIMEOUT_K - 1U)) == 0) {
                context.check(expected, observed);
            }
        }
    }
}

#endif
