/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_SIMT_TEAM_H
#define SHMEMI_DEVICE_SIMT_TEAM_H

#include "host_device/shmem_common_types.h"
#include "device_simt/shmem_simt_common_types.h"
#include "utils/shmemi_simt_kernel_debug.h"

namespace simt {

#ifndef MULTI_INSTANCE
__simt_callee__ inline void aclshmemi_check_team_pe(int pe, int team_size)
{
    if (pe < 0 || pe >= team_size) {
        simt::aclshmemi_kernel_abort("invalid PE index: pe=%d, team_size=%d\n", pe, team_size);
    }
}
#endif

__simt_callee__ inline int aclshmem_my_pe()
{
    __gm__ aclshmem_device_host_state_simt_t* device_state = simt::aclshmemi_get_state();
    __gm__ aclshmemx_team_t* team_ptr = (__gm__ aclshmemx_team_t*)device_state->team_pools[ACLSHMEM_TEAM_WORLD];
    return team_ptr->mype;
}

__simt_callee__ inline int aclshmem_n_pes()
{
    __gm__ aclshmem_device_host_state_simt_t* device_state = simt::aclshmemi_get_state();
    __gm__ aclshmemx_team_t* team_ptr = (__gm__ aclshmemx_team_t*)device_state->team_pools[ACLSHMEM_TEAM_WORLD];
    return team_ptr->size;
}

__simt_callee__ inline int aclshmem_team_my_pe(aclshmem_team_t team)
{
#ifdef MULTI_INSTANCE
    if (team < 0 || team >= ACLSHMEM_MAX_TEAMS) {
#else
    if (team == ACLSHMEM_TEAM_INVALID) {
#endif
        return -1;
    } else {
        __gm__ aclshmemx_team_t* src_team_ptr = (__gm__ aclshmemx_team_t*)aclshmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->mype : -1);
    }
}

__simt_callee__ inline int aclshmem_team_n_pes(aclshmem_team_t team)
{
#ifdef MULTI_INSTANCE
    if (team < 0 || team >= ACLSHMEM_MAX_TEAMS) {
#else
    if (team == ACLSHMEM_TEAM_INVALID) {
#endif
        return -1;
    } else {
        __gm__ aclshmemx_team_t* src_team_ptr = (__gm__ aclshmemx_team_t*)aclshmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->size : -1);
    }
}

__simt_callee__ inline int aclshmem_team_translate_pe(aclshmem_team_t src_team, int src_pe, aclshmem_team_t dest_team)
{
#ifdef MULTI_INSTANCE
    if (src_team < 0 || src_team >= ACLSHMEM_MAX_TEAMS || dest_team < 0 || dest_team >= ACLSHMEM_MAX_TEAMS) {
#else
    if (src_team == ACLSHMEM_TEAM_INVALID || dest_team == ACLSHMEM_TEAM_INVALID) {
#endif
        return -1;
    }
    __gm__ aclshmem_device_host_state_simt_t* device_state = aclshmemi_get_state();

    __gm__ aclshmemx_team_t* src_team_ptr = (__gm__ aclshmemx_team_t*)device_state->team_pools[src_team];
    __gm__ aclshmemx_team_t* dest_team_ptr = (__gm__ aclshmemx_team_t*)device_state->team_pools[dest_team];

#ifdef MULTI_INSTANCE
    if (src_team_ptr == nullptr || dest_team_ptr == nullptr || src_pe < 0 || src_pe >= src_team_ptr->size) {
        return -1;
    }

#else
    ACLSHMEM_DEBUG_FUNC(simt::aclshmemi_check_team_pe, src_pe, src_team_ptr->size);
#endif

    int global_pe = src_team_ptr->pe_mapping[src_pe];
#ifdef MULTI_INSTANCE
    if (global_pe < 0 || global_pe >= ACLSHMEM_MAX_PES) {
        return -1;
    }
#endif
    return dest_team_ptr->pe_mapping[global_pe + ACLSHMEM_MAX_PES];
}

__simt_callee__ inline int aclshmem_team_pe_mapping(aclshmem_team_t team, int pe)
{
#ifdef MULTI_INSTANCE
    if (team < 0 || team >= ACLSHMEM_MAX_TEAMS) {
#else
    if (team == ACLSHMEM_TEAM_INVALID) {
#endif
        return -1;
    }
    __gm__ aclshmem_device_host_state_simt_t* device_state = aclshmemi_get_state();
    __gm__ aclshmemx_team_t* team_ptr = (__gm__ aclshmemx_team_t*)device_state->team_pools[team];

#ifdef MULTI_INSTANCE
    if (team_ptr == nullptr || pe < 0 || pe >= team_ptr->size) {
        return -1;
    }

#else
    ACLSHMEM_DEBUG_FUNC(simt::aclshmemi_check_team_pe, pe, team_ptr->size);
#endif
    return team_ptr->pe_mapping[pe];
}

} // namespace simt

#endif
