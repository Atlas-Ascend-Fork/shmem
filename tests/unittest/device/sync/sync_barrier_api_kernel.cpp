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
constexpr uint32_t kSyncBarrierApiBlockNum = 16;

ACLSHMEM_DEVICE void put_local_segment_to_world(GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    for (int peer = 0; peer < rank_size; ++peer) {
        if (peer == rank_id) {
            continue;
        }
        aclshmem_putmem_nbi(addr + rank_id * message_length, addr + rank_id * message_length, message_length, peer);
    }
}

ACLSHMEM_DEVICE bool is_team_member(int rank_id, int team_start, int team_stride, int team_size)
{
    if (team_stride <= 0 || team_size <= 0) {
        return false;
    }
    const int index = rank_id - team_start;
    return index >= 0 && (index % team_stride == 0) && (index / team_stride < team_size);
}

ACLSHMEM_DEVICE void put_local_segment_to_team(
    GM_ADDR addr, int rank_id, int rank_size, int message_length, int team_start, int team_stride, int team_size)
{
    for (int i = 0; i < team_size; ++i) {
        const int peer = team_start + i * team_stride;
        if (peer >= rank_size || peer == rank_id) {
            continue;
        }
        aclshmem_putmem_nbi(addr + rank_id * message_length, addr + rank_id * message_length, message_length, peer);
    }
}
} // namespace

extern "C" ACLSHMEM_GLOBAL void device_sync_all_api(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    util_set_ffts_config(config);
    aclshmem_barrier_all();

    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            put_local_segment_to_world(addr, rank_id, rank_size, message_length);
            aclshmem_quiet();
        }
    }

    aclshmem_sync_all();
}

extern "C" ACLSHMEM_GLOBAL void device_barrier_all_api(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    util_set_ffts_config(config);
    aclshmem_barrier_all();

    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            put_local_segment_to_world(addr, rank_id, rank_size, message_length);
        }
    }

    // The barrier must complete the NBI operations before it returns.
    aclshmem_barrier_all();
}

extern "C" ACLSHMEM_GLOBAL void device_sync_team_api(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length, aclshmem_team_t team_id,
    int team_pe_start, int team_pe_stride, int team_pe_size)
{
    util_set_ffts_config(config);
    aclshmem_barrier(team_id);

    if ASCEND_IS_AIV {
        if (is_team_member(rank_id, team_pe_start, team_pe_stride, team_pe_size) && AscendC::GetBlockIdx() == 0) {
            put_local_segment_to_team(
                addr, rank_id, rank_size, message_length, team_pe_start, team_pe_stride, team_pe_size);
            aclshmem_quiet();
        }
    }

    aclshmem_sync(team_id);
}

extern "C" ACLSHMEM_GLOBAL void device_barrier_team_api(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length, aclshmem_team_t team_id,
    int team_pe_start, int team_pe_stride, int team_pe_size)
{
    util_set_ffts_config(config);
    aclshmem_barrier(team_id);

    if ASCEND_IS_AIV {
        if (is_team_member(rank_id, team_pe_start, team_pe_stride, team_pe_size) && AscendC::GetBlockIdx() == 0) {
            put_local_segment_to_team(
                addr, rank_id, rank_size, message_length, team_pe_start, team_pe_stride, team_pe_size);
        }
    }

    // The team barrier must complete the NBI operations from every team member.
    aclshmem_barrier(team_id);
}

void device_sync_all_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length)
{
    device_sync_all_api<<<kSyncBarrierApiBlockNum, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length);
}

void device_barrier_all_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length)
{
    device_barrier_all_api<<<kSyncBarrierApiBlockNum, nullptr, stream>>>(
        config, addr, rank_id, rank_size, message_length);
}

void device_sync_team_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id, int team_pe_start, int team_pe_stride, int team_pe_size)
{
    device_sync_team_api<<<kSyncBarrierApiBlockNum, nullptr, stream>>>(
        config, addr, rank_id, rank_size, message_length, team_id, team_pe_start, team_pe_stride, team_pe_size);
}

void device_barrier_team_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id, int team_pe_start, int team_pe_stride, int team_pe_size)
{
    device_barrier_team_api<<<kSyncBarrierApiBlockNum, nullptr, stream>>>(
        config, addr, rank_id, rank_size, message_length, team_id, team_pe_start, team_pe_stride, team_pe_size);
}
