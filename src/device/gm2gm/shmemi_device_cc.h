/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
This file provides device-side collective synchronization implementations, ensuring that:
1. ALL VEC CORES of all ranks of a team reach a sychonization point before doing subsequent operations.
2. All operations of ALL VEC CORES of all ranks of the team before the synchronization point are visible
   to ALL VEC CORES of all ranks of the team after the synchronization point.

*/

#ifndef _DEVICE_GM2GM_ACLSHMEMI_DEVICE_CC_H_
#define _DEVICE_GM2GM_ACLSHMEMI_DEVICE_CC_H_

#include <cstdint>

#include "kernel_operator.h"

#include "shmemi_device_common.h"
#include "host_device/shmem_common_types.h"
#include "shmemi_device_mo.h"
#include "shmemi_device_p2p_sync.h"
#include "gm2gm/engine/shmemi_device_rdma.h"
#include "device/team/shmem_device_team.h"
#include "utils/mstx/shmemi_mstx_report.h"

constexpr int32_t SYNC_DISSEM_KVAL = 2;
constexpr int32_t SYNC_DISSEM_KVAL_MIN = 2;
constexpr int32_t SYNC_V3_SMALL_TEAM_SIZE = 32;
constexpr int32_t SHIFT_MULTIPLIER = 2;
constexpr uint64_t ACLSHMEM_DATA_CACHE_LINE_SIZE = 64;

constexpr int32_t SYNC_V4_SLOT_CAPACITY = SYNC_ARRAY_SIZE / ACLSHMEM_SYNCBIT_SIZE;

// A full phase at radix k consumes k - 1 slots. If the slot capacity contains q full phases and r remaining slots,
// those r slots cover r additional radix multiples in the next phase, so max_team_size = (r + 1) * k^q.
// With 40 slots, k=32 gives (9 + 1) * 32 = 320, and k=16 gives (10 + 1) * 16^2 = 2816.
constexpr int64_t aclshmemi_sync_v4_max_pes_for_radix(int32_t radix)
{
    int32_t full_phases = SYNC_V4_SLOT_CAPACITY / (radix - 1);
    int32_t remaining_slots = SYNC_V4_SLOT_CAPACITY % (radix - 1);
    int64_t max_pes = remaining_slots + 1;
    for (int32_t phase = 0; phase < full_phases; ++phase) {
        max_pes *= radix;
    }
    return max_pes;
}

constexpr int64_t SYNC_V4_K32_MAX_PES = aclshmemi_sync_v4_max_pes_for_radix(32);
constexpr int64_t SYNC_V4_K16_MAX_PES = aclshmemi_sync_v4_max_pes_for_radix(16);

static_assert(SYNC_V4_SLOT_CAPACITY > 0, "V4 requires at least one per-team sync slot");
static_assert(
    aclshmemi_sync_v4_max_pes_for_radix(8) >= ACLSHMEM_MAX_PES,
    "The V4 per-team sync-slot capacity is insufficient for ACLSHMEM_MAX_PES");

ACLSHMEM_DEVICE int32_t aclshmemi_sync_v4_k_cap(int32_t team_size)
{
    if (team_size <= 1) {
        return 1;
    }
    if (team_size < 32) {
        int32_t k = 1;
        while (k <= team_size / 2) {
            k <<= 1;
        }
        return k;
    }
    if (team_size <= SYNC_V4_K32_MAX_PES) {
        return 32;
    }
    if (team_size <= SYNC_V4_K16_MAX_PES) {
        return 16;
    }
    return 8;
}

ACLSHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t* addr);

ACLSHMEM_DEVICE __gm__ aclshmemx_sync_bit* aclshmemi_get_core_sync_pool()
{
    return (__gm__ aclshmemx_sync_bit*)aclshmemi_get_state()->core_sync_pool;
}

ACLSHMEM_DEVICE __gm__ aclshmemx_sync_bit* aclshmemi_get_core_sync_counter()
{
    return (__gm__ aclshmemx_sync_bit*)aclshmemi_get_state()->core_sync_counter;
}

ACLSHMEM_DEVICE void aclshmemi_sync_core_soft()
{
    if ASCEND_IS_AIC {
        return;
    }
    MSTX_FUSE_SCOPE_START();
    int32_t cur_block_idx = AscendC::GetBlockIdx();
    int32_t block_dim = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    auto sync_pool = aclshmemi_get_core_sync_pool();
    auto sync_counter = aclshmemi_get_core_sync_counter();
    int32_t core_sync_stride = 0;
    int32_t covered_aivs = 1;
    while (covered_aivs < block_dim) {
        covered_aivs <<= 1;
        core_sync_stride++;
    }
    core_sync_stride = core_sync_stride > 0 ? core_sync_stride : 1;
    int32_t count = aclshmemi_load((__gm__ int32_t*)(sync_counter)) + 1;

    int32_t shift = 1;
    int32_t offset = 0;
    while (shift < block_dim) {
        int32_t next = (cur_block_idx + shift) % block_dim;

        aclshmemi_signal_set((__gm__ int32_t*)(sync_pool + next * core_sync_stride + offset), count);
        aclshmemi_signal_wait_until_eq_for_barrier(
            (__gm__ int32_t*)(sync_pool + cur_block_idx * core_sync_stride + offset), count);

        shift *= SHIFT_MULTIPLIER;
        offset++;
    }

    aclshmemi_store((__gm__ int32_t*)(sync_counter), count);
    MSTX_FUSE_SCOPE_END();
    MSTX_CROSS_CORE_BARRIER_REPORT(block_dim);
}

/* Level 1: sync between vec cores (within a device) */
template <bool IS_AIV_ONLY>
ACLSHMEM_DEVICE void aclshmemi_sync_core()
{
#ifdef __CCE_AICORE_ENABLE_MIX__
    AscendC::SyncAll<IS_AIV_ONLY>();
#else
    aclshmemi_sync_core_soft();
#endif
}

ACLSHMEM_DEVICE __gm__ aclshmemx_sync_bit* aclshmemi_get_team_sync_pool(aclshmem_team_t team_idx)
{
    uint64_t addr = (uint64_t)aclshmemi_get_state()->sync_pool;
    addr += team_idx * SYNC_ARRAY_SIZE;
    return (__gm__ aclshmemx_sync_bit*)addr;
}

ACLSHMEM_DEVICE __gm__ aclshmemx_sync_bit* aclshmemi_get_team_sync_counter(aclshmem_team_t team_idx)
{
    uint64_t addr = (uint64_t)aclshmemi_get_state()->sync_counter;
    addr += team_idx * SYNC_COUNTER_SIZE;
    return (__gm__ aclshmemx_sync_bit*)addr;
}

/* Level 2: sync between devices (within a host)

Dissemination Sync

1. Algorithm process

The algorithm process could be separated into multiple rounds.
In each round, every participating rank signals its succeeding rank and waits its preceding rank's signal.
The distance of a rank and its successor increases exponentially with the round.

An 8-rank example is shown below:

           round 1            round 2            round 3
  rank 0  --------→  rank 1  --------→  rank 3  --------→  rank 7
  rank 1  --------→  rank 2  --------→  rank 4  --------→  rank 0
  rank 2  --------→  rank 3  --------→  rank 5  --------→  rank 1
  rank 3  --------→  rank 4  --------→  rank 6  --------→  rank 2
  rank 4  --------→  rank 5  --------→  rank 7  --------→  rank 3
  rank 5  --------→  rank 6  --------→  rank 0  --------→  rank 4
  rank 6  --------→  rank 7  --------→  rank 1  --------→  rank 5
  rank 7  --------→  rank 0  --------→  rank 2  --------→  rank 6

Refer to https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf for more details.

2. Implementation details

Current implementation maintains an array of MAX_RANK_SIZE for each rank, with element of pos i indicating whether the
rank has received signal of rank i.
In each round, every rank writes remote array and check local array to decide whether this round has finished.
Once all rounds finish, sync ends.

Theoretically, each element is writen by only 1 rank and read by self, involving only p2p synchronization.
However, separate elements may exist on the same cacheline, so that concurrent write
acctually happens and may cause wrong result.

For example:
a. rank n is waiting for rank n-1's signal (in round 1).
             ↑   n
--------------------------------------------
      ...  | 0 | 0 | ...
--------------------------------------------

b. rank n-1 reads rank n's array, and write the element at position n-1 (in round 1).
             ↓   n
--------------------------------------------
      ...  | 1 | 0 | ...
--------------------------------------------

c. rank n-2 reads staled rank n's array (no cache consistency ensurance), and write
   the element at position n-2 (in round 2).
         ↓       n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

d. rank n-2 overwrites rank n-1，so rank n may miss rank n-1's signal and wait forever.
             ↑   n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

To avoid this issue, separate elements must exist on different cachelines. See aclshmemx_sync_bit for detailed
definition.

Additionly, instead of simply write a flag, each rank writes a 64-bit number into the array,
indicating how many times this team has performed sync.

The temporal and spatial complexity of this implementation are O(logN) and O(N), respectively.

3. Futher development
  a. Hierarchical synchronization.
    Sync within the host first, then sync between host. May achieve better
    performance by utilizing low-latency in-host network better.

  b. Group dissemination.
    Group the ranks so that each rank could issue multiple signals and waits concurrently,
    instead of 1 signal and 1 wait as above.

  c. Optimize spatial complexity to O(logN).
*/
template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_sync_npu_v1(aclshmem_team_t team_idx)
{
    if (AscendC::GetBlockIdx() != 0) {
        aclshmemi_sync_core<IS_AIV_ONLY>();
        return;
    }
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_idx];
    int my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t*)sync_counter) + 1;
    if ASCEND_IS_AIV {
        int32_t offset = 0;
        const int multiplier = 2;
        while (shift < size) {
            int next_pe_in_team = (my_pe_in_team + shift) % size;
            int next_pe = start + next_pe_in_team * stride;

            // signal next pe
            aclshmemi_signal_set((__gm__ int32_t*)(sync_pool + offset), next_pe, count);

            // wait pre pe
            aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t*)(sync_pool + offset), count);

            shift *= multiplier;
            offset++;
        }

        aclshmemi_store((__gm__ int32_t*)sync_counter, count);
    }

    aclshmemi_sync_core<IS_AIV_ONLY>();
}

/** Group Dissemination Sync.
 *
 *  An optimized version of aclshmemi_sync_npu_v1, with temporal complexity reduced to
 * O(log_{k}^{N}).
 */
template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_sync_npu_v2(aclshmem_team_t team)
{
    int32_t cur_block_idx = AscendC::GetBlockIdx();
    int32_t block_dim = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    auto my_pe = aclshmem_team_my_pe(team);
    auto team_size = aclshmem_team_n_pes(team);

    int32_t k = SYNC_DISSEM_KVAL < team_size ? SYNC_DISSEM_KVAL : team_size;
    k = SYNC_DISSEM_KVAL_MIN < k ? k : SYNC_DISSEM_KVAL_MIN;
    int32_t temp = team_size - 1;
    int32_t phase_num = 0;
    int32_t pow_k = 1;

    auto sync_pool = aclshmemi_get_team_sync_pool(team);
    auto sync_counter = aclshmemi_get_team_sync_counter(team);
    int32_t count = aclshmemi_load((__gm__ int32_t*)sync_counter) + 1;
    aclshmemi_sync_core<IS_AIV_ONLY>();

    if ASCEND_IS_AIV {
        while (temp) {
            // notify neighbors
            for (int32_t i = cur_block_idx + 1; i <= k - 1; i += block_dim) {
                int32_t shift = i * pow_k;
                if (shift >= team_size) {
                    break;
                }
                int32_t to_nbr_idx = (my_pe + shift) % team_size;
                int32_t to_nbr = aclshmemi_get_state()->team_pools[team]->pe_mapping[to_nbr_idx];
                aclshmemi_signal_set((__gm__ int32_t*)(sync_pool + my_pe), to_nbr, count);
            }

            // wait for neighbors notification
            for (int32_t i = cur_block_idx + 1; i <= k - 1; i += block_dim) {
                int32_t shift = i * pow_k;
                if (shift >= team_size) {
                    break;
                }
                int32_t from_nbr_idx = my_pe - shift;
                if (from_nbr_idx < 0) {
                    from_nbr_idx += team_size;
                }
                aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t*)(sync_pool + from_nbr_idx), count);
            }

            // update params
            temp /= k;
            phase_num++;
            pow_k *= k;
        }

        if (!cur_block_idx) {
            aclshmemi_store((__gm__ int32_t*)sync_counter, count);
        }
    }
    aclshmemi_sync_core<IS_AIV_ONLY>();
}

/** Centralized Sync (pull mode).
 *
 *  The temporal and spatial complexity of this implementation are O(N/K) and O(1),
 * respectively.
 *  Performs better than Group Dissemination Sync at small scale (eg. 8 ranks).
 */
template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_sync_npu_v3(aclshmem_team_t team_idx)
{
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_idx];
    int32_t vec_id = AscendC::GetBlockIdx();
    int32_t vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t size = team->size;

    MSTX_FUSE_SCOPE_START();

    int32_t my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int32_t start = team->start;
    int32_t stride = team->stride;
    auto sync_pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int32_t k = vec_size < size ? vec_size : size;
    int32_t my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t*)sync_counter) + 1;

    aclshmemi_sync_core<IS_AIV_ONLY>();

    if ASCEND_IS_AIV {
        // Only the first k AIVs participate in remote polling. Without this
        // guard, vec_id >= k revisits slots already assigned to vec_id % k.
        if (vec_id < k) {
            for (int32_t i = vec_id; i < size; i += k) {
                if (i == my_pe_in_team) {
                    // write local
                    aclshmemi_signal_set((__gm__ int32_t*)sync_pool, count);
                } else {
                    // read remote
                    int32_t remote_pe = start + i * stride;
                    aclshmemi_signal_wait_until_eq_for_barrier(
                        (__gm__ int32_t*)aclshmem_ptr(sync_pool, remote_pe), count);
                }
            }
        }

#if defined(ACLSHMEM_MSSANITIZER_BUILD)
        dcci_cacheline((__gm__ uint8_t*)sync_counter);
        if (vec_id == 0) {
            aclshmemi_signal_set((__gm__ int32_t*)sync_counter, count);
        }
#else
        aclshmemi_store((__gm__ int32_t*)sync_counter, count);
#endif
    }

    aclshmemi_sync_core<IS_AIV_ONLY>();
    MSTX_FUSE_SCOPE_END();
    MSTX_BARRIER_NPU_REPORT(size, vec_size);
}

/** K-ary dissemination sync (push mode).
 *
 *  Each PE writes its generation to the destination PE's symmetric sync slot and
 *  polls only local GM. It reuses the v3 per-team sync pool and counter, so no
 *  additional symmetric memory is required.
 */
template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_sync_npu_v4(aclshmem_team_t team_idx)
{
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_idx];
    int32_t vec_id = AscendC::GetBlockIdx();
    int32_t vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    MSTX_FUSE_SCOPE_START();

    int32_t size = team->size;
    int32_t my_pe_in_team = team->mype;
    auto sync_pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);
    int32_t count = 0;

    int32_t k_max = aclshmemi_sync_v4_k_cap(size);
    int32_t runtime_k_cap = vec_size < size ? vec_size : size;
    if (size > 1) {
        runtime_k_cap = runtime_k_cap < SYNC_DISSEM_KVAL_MIN ? SYNC_DISSEM_KVAL_MIN : runtime_k_cap;
    }
    while (k_max > runtime_k_cap && k_max > 1) {
        k_max >>= 1;
    }

    // Minimize the number of dissemination rounds first. Among radices with
    // the same round count, use the smallest one to reduce remote signals and
    // local slot polling. Keep all selection work inside the V4 path so V3 has
    // no additional runtime overhead.
    int32_t phases = 0;
    int32_t temp = size > 1 ? size - 1 : 0;
    while (temp) {
        phases++;
        temp /= k_max;
    }

    int32_t k = k_max;
    for (int32_t candidate_k = SYNC_DISSEM_KVAL_MIN; candidate_k < k_max; candidate_k <<= 1) {
        int32_t candidate_phases = 0;
        temp = size > 1 ? size - 1 : 0;
        while (temp) {
            candidate_phases++;
            temp /= candidate_k;
        }
        if (candidate_phases == phases) {
            k = candidate_k;
            break;
        }
    }
    int32_t slot_stride = k - 1;

    // AIV0 advances one shared generation. Every AIV invalidates its stale
    // scalar-cache line before the generation is published.
    if ASCEND_IS_AIV {
        dcci_cacheline((__gm__ uint8_t*)sync_counter);
    }
    aclshmemi_sync_core<IS_AIV_ONLY>();
    if ASCEND_IS_AIV {
        if (vec_id == 0) {
            count = aclshmemi_load((__gm__ int32_t*)sync_counter) + 1;
            aclshmemi_signal_set((__gm__ int32_t*)sync_counter, count);
        }
    }
    aclshmemi_sync_core<IS_AIV_ONLY>();
    if ASCEND_IS_AIV {
        count = aclshmemi_load((__gm__ int32_t*)sync_counter);
    }

    int32_t pow_k = 1;
    for (int32_t round = 0; round < phases; round++) {
        if ASCEND_IS_AIV {
            for (int32_t i = vec_id + 1; i <= slot_stride; i += vec_size) {
                int32_t shift = i * pow_k;
                if (shift >= size) {
                    break;
                }
                int32_t to_nbr_idx = (my_pe_in_team + shift) % size;
                int32_t to_pe = team->pe_mapping[to_nbr_idx];
                int32_t slot = round * slot_stride + (i - 1);
                aclshmemi_signal_set((__gm__ int32_t*)aclshmem_ptr(sync_pool + slot, to_pe), count);
            }
            for (int32_t i = vec_id + 1; i <= slot_stride; i += vec_size) {
                int32_t shift = i * pow_k;
                if (shift >= size) {
                    break;
                }
                int32_t slot = round * slot_stride + (i - 1);
                aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t*)(sync_pool + slot), count);
            }
        }
        aclshmemi_sync_core<IS_AIV_ONLY>();
        pow_k *= k;
    }

    MSTX_FUSE_SCOPE_END();
    MSTX_BARRIER_NPU_REPORT(size, vec_size);
}

template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_sync_algo(aclshmem_team_t team_idx)
{
    aclshmemi_sync_core<IS_AIV_ONLY>();

    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    aclshmemx_team_t* team_ptr = device_state->team_pools[team_idx];

    bool intra_node_reaches_all = true;
    bool roce_reaches = false;

    if (team_idx == ACLSHMEM_TEAM_WORLD) {
        uint8_t quiet_transport_mask =
            aclshmemi_filter_unsupported_quiet_transports(aclshmemi_get_quiet_transport_mask());
        if (quiet_transport_mask & ACLSHMEM_TRANSPORT_ROCE) {
            roce_reaches = true;
        }
    } else {
        constexpr uint8_t intra_node_transport_mask =
            ACLSHMEM_TRANSPORT_MTE | ACLSHMEM_TRANSPORT_SDMA | ACLSHMEM_TRANSPORT_UDMA;

        for (int32_t team_pe = 0; team_pe < team_ptr->size; ++team_pe) {
            int32_t global_pe = team_ptr->pe_mapping[team_pe];
            uint8_t transport_mask = device_state->topo_list[global_pe];
            intra_node_reaches_all = intra_node_reaches_all && ((transport_mask & intra_node_transport_mask) != 0);
            roce_reaches = roce_reaches || ((transport_mask & ACLSHMEM_TRANSPORT_ROCE) != 0);
        }
    }

    if (roce_reaches) {
        if ASCEND_IS_AIV {
            if (AscendC::GetBlockIdx() == 0) {
                uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
                uint32_t sync_id = device_state->rdma_config.sync_id;
                aclshmemi_roce_team_sync(team_ptr, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
            }
        }
        aclshmemi_sync_core<IS_AIV_ONLY>();
    } else if (intra_node_reaches_all) {
        int32_t vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        if (team_ptr->size <= SYNC_V3_SMALL_TEAM_SIZE || vec_size >= team_ptr->size) {
            aclshmemi_sync_npu_v3<IS_AIV_ONLY>(team_idx);
        } else {
            aclshmemi_sync_npu_v4<IS_AIV_ONLY>(team_idx);
        }
    }
}
template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_barrier_algo(aclshmem_team_t team_idx)
{
    aclshmemi_sync_core<IS_AIV_ONLY>();
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            aclshmemi_quiet();
        }
    }
    aclshmemi_sync_algo<IS_AIV_ONLY>(team_idx);
}

template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_sync(aclshmem_team_t team)
{
    if (team == -1) {
        return; // not in team
    }
    aclshmemi_sync_algo<IS_AIV_ONLY>(team);
}

template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_barrier(aclshmem_team_t team)
{
    if (team == -1) {
        return; // not in team
    }
    aclshmemi_barrier_algo<IS_AIV_ONLY>(team);
}

ACLSHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t* addr)
{
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        global);
    __asm__ __volatile__("");
}

ACLSHMEM_DEVICE void dcci_cachelines(__gm__ uint8_t* addr, uint64_t length)
{
    __gm__ uint8_t* start =
        (__gm__ uint8_t*)((uint64_t)addr / ACLSHMEM_DATA_CACHE_LINE_SIZE * ACLSHMEM_DATA_CACHE_LINE_SIZE);
    __gm__ uint8_t* end =
        (__gm__ uint8_t*)(((uint64_t)addr + length) / ACLSHMEM_DATA_CACHE_LINE_SIZE * ACLSHMEM_DATA_CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint64_t i = 0; i <= end - start; i += ACLSHMEM_DATA_CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[i]);
        __asm__ __volatile__("");
    }
}

ACLSHMEM_DEVICE void dcci_entire_cache()
{
    AscendC::GlobalTensor<uint8_t> global;

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        global);
    __asm__ __volatile__("");
}

ACLSHMEM_DEVICE void dcci_atomic()
{
    AscendC::GlobalTensor<uint8_t> global;

    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_ATOMIC>(global);
    __asm__ __volatile__("");
}

ACLSHMEM_DEVICE void dsb_all() { AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>(); }

ACLSHMEM_DEVICE void aclshmemi_barrier_cross_host(aclshmemx_team_t* team)
{
    int my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int round = 0;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t*)sync_counter) + 1;
    aclshmemi_store((__gm__ int32_t*)sync_counter, count);
    dcci_cacheline((__gm__ uint8_t*)sync_counter);
    while (shift < size) {
        int next_pe_in_team = (my_pe_in_team + shift) % size;
        int next_pe = start + next_pe_in_team * stride;

        // The destination PE receives this round's signal in its own team slot.
        aclshmemi_highlevel_signal_set((__gm__ int32_t*)(sync_pool + round), (__gm__ int32_t*)sync_counter, next_pe);

        // Wait for this round's signal from the preceding PE.
        aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t*)(sync_pool + round), count);

        shift *= SHIFT_MULTIPLIER;
        ++round;
    }
}

ACLSHMEM_DEVICE void aclshmemi_handle(aclshmem_team_t tid)
{
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[tid];

    int mype = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    if ((mype - start) % stride != 0) {
        // not in this team
        return;
    }

    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == mype) {
            continue;
        }
        aclshmemi_roce_quiet(peer, ub_tensor_64, ub_tensor_32, 0);
    }

    if ASCEND_IS_AIV {
        aclshmemi_barrier_cross_host(team);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
{
#if !defined(__DAV_C310_VEC__) && !defined(__DAV_C310_CUBE__)
    AscendC::SetSyncBaseAddr(config);
#endif
}

ACLSHMEM_DEVICE void aclshmem_sync(aclshmem_team_t tid) { aclshmemi_sync<false>(tid); }

ACLSHMEM_DEVICE int aclshmem_team_sync(aclshmem_team_t tid)
{
    aclshmemi_sync<false>(tid);
    return 0;
}

ACLSHMEM_DEVICE void aclshmem_sync_all() { aclshmem_sync(ACLSHMEM_TEAM_WORLD); }

ACLSHMEM_DEVICE int aclshmem_barrier(aclshmem_team_t tid)
{
    aclshmemi_barrier<false>(tid);
    return 0;
}

ACLSHMEM_DEVICE void aclshmem_barrier_all() { aclshmem_barrier(ACLSHMEM_TEAM_WORLD); }

ACLSHMEM_DEVICE void aclshmemx_sync_vec(aclshmem_team_t tid) { aclshmemi_sync<true>(tid); }

ACLSHMEM_DEVICE void aclshmemx_sync_vec_all() { aclshmemx_sync_vec(ACLSHMEM_TEAM_WORLD); }

ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t tid) { aclshmemx_sync_vec(tid); }

ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec() { aclshmemx_sync_vec(ACLSHMEM_TEAM_WORLD); }

#ifdef __cplusplus
}
#endif

#endif // _DEVICE_GM2GM_ACLSHMEMI_DEVICE_CC_H_
