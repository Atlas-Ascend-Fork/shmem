/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>
#include <mutex>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "gm2gm/shmemi_device_cc_kernel.h"
std::bitset<ACLSHMEM_MAX_TEAMS> g_team_mask;
aclshmemx_team_t* g_aclshmem_team_pool = nullptr;
std::recursive_mutex g_team_mutex;

#ifdef ACLSHMEM_UNIT_TEST
bool g_fail_next_device_team_update = false;
std::bitset<ACLSHMEM_MAX_TEAMS> g_saved_team_mask;

void aclshmemi_team_fail_next_device_update_for_test()
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    g_fail_next_device_team_update = true;
}

void aclshmemi_team_fill_mask_for_test(int32_t free_team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    g_saved_team_mask = g_team_mask;
    g_team_mask.set();
    g_team_mask.reset(free_team);
}

void aclshmemi_team_restore_mask_for_test()
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    g_team_mask = g_saved_team_mask;
}
#endif

namespace {

constexpr uint32_t core_sync_stride_for_aiv_count(uint32_t aiv_count)
{
    uint32_t stride = 0;
    uint64_t covered_aivs = 1;
    while (covered_aivs < aiv_count) {
        covered_aivs <<= 1;
        ++stride;
    }
    return stride > 0 ? stride : 1;
}

} // namespace

#define ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(func_name)                                                                 \
    do {                                                                                                               \
        if (g_instance_ctx->id != 0) {                                                                                 \
            SHM_LOG_ERROR(                                                                                             \
                (func_name) << " is not supported in multi-instance mode (instance_id=" << g_instance_ctx->id << "). " \
                            << "Only Instance 0 supports team operations. Please Create new instance instead.");       \
            return ACLSHMEM_NOT_SUPPORTED;                                                                             \
        }                                                                                                              \
    } while (0)

inline std::string team_config2string(aclshmemx_team_t* config)
{
    std::ostringstream oss;
    oss << "[team:" << config->team_idx;
    oss << ",npes:" << config->size;
    oss << ",mype:" << config->mype;
    oss << ",start:" << config->start;
    oss << ",stride:" << config->stride;
    oss << "]";
    return oss.str();
}

inline bool is_valid_team(aclshmem_team_t& team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    bool is_allocated =
        team == ACLSHMEM_TEAM_WORLD || (team >= 0 && team < ACLSHMEM_MAX_TEAMS && g_team_mask.test(team));
    return (
        g_state.is_aclshmem_initialized && g_aclshmem_team_pool != nullptr && team >= 0 && team < ACLSHMEM_MAX_TEAMS &&
        is_allocated && g_state.team_pools[team] != nullptr);
}

inline void release_team_idx(aclshmem_team_t team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    if (team > ACLSHMEM_TEAM_WORLD && team < ACLSHMEM_MAX_TEAMS) {
        g_team_mask.reset(team);
        if (g_aclshmem_team_pool != nullptr) {
            g_aclshmem_team_pool[team] = {};
        }
    }
}

inline void device_team_destroy(int32_t team_idx)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    // device_ptr Free
    aclshmemx_team_t* device_team_ptr = g_state.team_pools[team_idx];
    if (device_team_ptr != nullptr) {
        if (aclrtFree((void*)device_team_ptr) != ACL_SUCCESS) {
            SHM_LOG_ERROR("aclrtFree for device_team_ptr failed for team " << team_idx);
        }
        g_state.team_pools[team_idx] = nullptr;
    }
}

inline int32_t device_team_update(int team_idx, aclshmemx_team_t* host_team_ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
#ifdef ACLSHMEM_UNIT_TEST
    if (g_fail_next_device_team_update) {
        g_fail_next_device_team_update = false;
        return ACLSHMEM_INNER_ERROR;
    }
#endif
    // device_ptr Malloc
    void* team_ptr = nullptr;
    ACLSHMEM_CHECK_RET(aclrtMalloc(&team_ptr, sizeof(aclshmemx_team_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    auto ret = aclrtMemcpy(
        (aclshmemx_team_t*)team_ptr, sizeof(aclshmemx_team_t), host_team_ptr, sizeof(aclshmemx_team_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("memcpy device team info failed, ret: " << ret);
        ACLSHMEM_CHECK_RET(aclrtFree(team_ptr));
        return ACLSHMEM_INNER_ERROR;
    }
    g_state.team_pools[team_idx] = (aclshmemx_team_t*)team_ptr;
    return ACLSHMEM_SUCCESS;
}

inline void rollback_team_creation(aclshmem_team_t team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    device_team_destroy(team);
    release_team_idx(team);
}

int32_t aclshmemi_team_init_sync_pool()
{
    g_state.sync_pool = (uint64_t)aclshmem_malloc(SYNC_POOL_SIZE);
    if (g_state.sync_pool == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void*)g_state.sync_pool, SYNC_POOL_SIZE, 0, SYNC_POOL_SIZE);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset sync pool failed. ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init_sync_counter()
{
    g_state.sync_counter = (uint64_t)aclshmem_malloc(SYNC_COUNTERS_SIZE);
    if (g_state.sync_counter == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync counter failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void*)g_state.sync_counter, SYNC_COUNTERS_SIZE, 0, SYNC_COUNTERS_SIZE);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset sync counter failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init_core_sync_pool(uint32_t aiv_count)
{
    if (aiv_count == 0) {
        SHM_LOG_ERROR("invalid internal core sync config, aiv_count=" << aiv_count);
        return ACLSHMEM_INNER_ERROR;
    }
    const uint32_t core_sync_stride = core_sync_stride_for_aiv_count(aiv_count);
    size_t core_sync_pool_size = static_cast<size_t>(aiv_count) * core_sync_stride * ACLSHMEM_SYNCBIT_SIZE;
    auto ret = aclrtMalloc((void**)&(g_state.core_sync_pool), core_sync_pool_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_pool == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void*)g_state.core_sync_pool, core_sync_pool_size, 0, core_sync_pool_size);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init_core_sync_counter()
{
    auto ret =
        aclrtMalloc((void**)&(g_state.core_sync_counter), ACLSHMEM_CORE_SYNC_COUNTER_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_counter == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync counter failed, ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    ret = aclrtMemset(
        (void*)g_state.core_sync_counter, ACLSHMEM_CORE_SYNC_COUNTER_SIZE, 0, ACLSHMEM_CORE_SYNC_COUNTER_SIZE);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync counter failed, ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init(int32_t rank, int32_t size, uint32_t aiv_count)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    /* Initialize ACLSHMEM_TEAM_WORLD */
    if (g_instance_ctx->id == 0) {
        g_team_mask.reset();
    }
    int team_size = g_instance_ctx->id == 0 ? ACLSHMEM_MAX_TEAMS : 1;
    g_aclshmem_team_pool = (aclshmemx_team_t*)calloc(team_size, sizeof(aclshmemx_team_t));
    if (g_aclshmem_team_pool == nullptr) {
        SHM_LOG_ERROR("malloc host shmem team pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    for (int i = 0; i < team_size; i++) {
        g_aclshmem_team_pool[i] = aclshmemx_team_t{-1, -1, -1, -1, -1, {0, 0, 0, {'0'}}, {-1}};
    }

    aclshmemx_team_t& aclshmem_team_world = g_aclshmem_team_pool[ACLSHMEM_TEAM_WORLD];
    aclshmem_team_world.team_idx = ACLSHMEM_TEAM_WORLD;
    aclshmem_team_world.start = 0;
    aclshmem_team_world.stride = 1;
    aclshmem_team_world.size = size;
    aclshmem_team_world.mype = rank;
    aclshmemi_team_populate_pe_mappings_from_constant_stride(&aclshmem_team_world);
    int ret = device_team_update(ACLSHMEM_TEAM_WORLD, &aclshmem_team_world);
    if (ret != 0) {
        aclshmemi_team_finalize();
        return ret;
    }

    /* Initialize TEAM SYNC */
    ret = aclshmemi_team_init_sync_pool();
    if (ret != 0) {
        return ret;
    }

    ret = aclshmemi_team_init_sync_counter();
    if (ret != 0) {
        return ret;
    }

    ret = aclshmemi_team_init_core_sync_pool(aiv_count);
    if (ret != 0) {
        return ret;
    }

    return aclshmemi_team_init_core_sync_counter();
}

int32_t first_free_idx_fetch()
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    for (int32_t i = ACLSHMEM_TEAM_WORLD + 1; i < ACLSHMEM_MAX_TEAMS; i++) {
        if (!g_team_mask.test(i)) {
            g_team_mask.set(i);
            return i;
        }
    }
    return -1;
}

int32_t aclshmemi_team_finalize()
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    /* Destroy all undestroyed teams */
    int team_size = g_instance_ctx->id == 0 ? ACLSHMEM_MAX_TEAMS : 1;
    for (int32_t i = 0; i < team_size; i++) {
        aclshmem_team_t team = i;
        if (is_valid_team(team)) {
            aclshmem_team_destroy(team);
        }
    }

    if (g_state.sync_counter != 0) {
        aclshmem_free(reinterpret_cast<void*>(g_state.sync_counter));
        g_state.sync_counter = 0;
    }
    if (g_state.sync_pool != 0) {
        aclshmem_free(reinterpret_cast<void*>(g_state.sync_pool));
        g_state.sync_pool = 0;
    }
    if (g_state.core_sync_counter != 0) {
        ACLSHMEM_CHECK_RET(aclrtFree(reinterpret_cast<void*>(g_state.core_sync_counter)));
        g_state.core_sync_counter = 0;
    }
    if (g_state.core_sync_pool != 0) {
        ACLSHMEM_CHECK_RET(aclrtFree(reinterpret_cast<void*>(g_state.core_sync_pool)));
        g_state.core_sync_pool = 0;
    }
    if (g_aclshmem_team_pool != nullptr) {
        free(g_aclshmem_team_pool);
        g_aclshmem_team_pool = nullptr;
    }
    if (g_instance_ctx->id == 0) {
        g_team_mask.reset();
    }
    return 0;
}
void aclshmemi_team_populate_from_world_pe_mapping(aclshmemx_team_t* team)
{
    for (int i = 0; i < team->size; i++) {
        int global_pe_index = team->pe_mapping[i];
        team->pe_mapping[global_pe_index + ACLSHMEM_MAX_PES] = i;
    }
}

void aclshmemi_team_populate_pe_mappings_from_constant_stride(aclshmemx_team_t* team)
{
    for (int i = 0; i < team->size; i++) {
        int global_pe_index = team->start + i * team->stride;
        team->pe_mapping[i] = global_pe_index;
    }
    aclshmemi_team_populate_from_world_pe_mapping(team);
}

int32_t aclshmemi_team_pe_mapping(aclshmem_team_t team, int pe)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    if (!is_valid_team(team)) {
        SHM_LOG_ERROR("input team is invalid!, team: " << team);
        return ACLSHMEM_INVALID_PARAM;
    }
    return g_aclshmem_team_pool[team].pe_mapping[pe];
}

int32_t aclshmem_team_split_strided_precheck(
    aclshmem_team_t parent_team, int32_t pe_start, int32_t pe_stride, int32_t pe_size, aclshmem_team_t* new_team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    if (new_team == nullptr) {
        SHM_LOG_ERROR("output team is null.");
        return ACLSHMEM_INVALID_PARAM;
    }

    *new_team = ACLSHMEM_TEAM_INVALID;
    if (!is_valid_team(parent_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << parent_team);
        return ACLSHMEM_INVALID_PARAM;
    }

    aclshmemx_team_t* src_team = &g_aclshmem_team_pool[parent_team];

    if (pe_start >= ACLSHMEM_MAX_PES || pe_stride >= ACLSHMEM_MAX_PES || pe_size > ACLSHMEM_MAX_PES) {
        SHM_LOG_ERROR(
            "create team failed, input invalid, pe_start:" << pe_start << " pe_size:" << pe_size << " pe_stride:"
                                                           << pe_stride << " parent:" << team_config2string(src_team));
        return ACLSHMEM_INVALID_PARAM;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmem_team_split_strided(
    aclshmem_team_t parent_team, int32_t pe_start, int32_t pe_stride, int32_t pe_size, aclshmem_team_t* new_team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    auto ret = aclshmem_team_split_strided_precheck(parent_team, pe_start, pe_stride, pe_size, new_team);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    aclshmemx_team_t* src_team = &g_aclshmem_team_pool[parent_team];
    if (pe_start < 0 || pe_start >= src_team->size || pe_size <= 0 || pe_size > src_team->size || pe_stride < 1) {
        SHM_LOG_ERROR(
            "create team failed, input invalid, pe_start:" << pe_start << " pe_size:" << pe_size << " pe_stride:"
                                                           << pe_stride << " parent:" << team_config2string(src_team));
        return ACLSHMEM_INVALID_PARAM;
    }

    int32_t global_pe = src_team->mype;
    int32_t global_pe_start = src_team->pe_mapping[pe_start];
    int32_t global_pe_stride = src_team->stride * pe_stride;
    int32_t global_pe_end = global_pe_start + global_pe_stride * (pe_size - 1);

    if (global_pe_start >= aclshmem_n_pes() || global_pe_end >= aclshmem_n_pes()) {
        SHM_LOG_ERROR(
            "create team failed, large than world size, pe_start:"
            << pe_start << " pe_size:" << pe_size << " pe_stride:" << pe_stride << " world_size:" << aclshmem_n_pes()
            << " parent:" << team_config2string(src_team));
        return ACLSHMEM_INVALID_PARAM;
    }

    aclshmemx_team_t my_team{};
    my_team.mype = (global_pe - global_pe_start) / global_pe_stride;

    if (global_pe < global_pe_start || (global_pe - global_pe_start) % global_pe_stride || my_team.mype >= pe_size) {
        SHM_LOG_INFO("This PE is not a member of the new team.");
        return ACLSHMEM_SUCCESS;
    }

    for (int i = 0; i < ACLSHMEM_MAX_PES; i++) {
        my_team.pe_mapping[i + ACLSHMEM_MAX_PES] = -1;
    }

    my_team.start = global_pe_start;
    my_team.stride = global_pe_stride;
    my_team.size = pe_size;

    for (int i = 0; i < pe_size; i++) {
        int temp_global_pe = global_pe_start + i * global_pe_stride;
        my_team.pe_mapping[i] = temp_global_pe;
        my_team.pe_mapping[temp_global_pe + ACLSHMEM_MAX_PES] = i;
    }

    my_team.team_idx = first_free_idx_fetch();
    if (my_team.team_idx == -1) {
        SHM_LOG_ERROR("create team failed, team num is full!");
        return ACLSHMEM_INNER_ERROR;
    }

    g_aclshmem_team_pool[my_team.team_idx] = my_team;
    if (device_team_update(my_team.team_idx, &g_aclshmem_team_pool[my_team.team_idx]) != 0) {
        rollback_team_creation(my_team.team_idx);
        SHM_LOG_ERROR("create team failed, malloc device state failed!");
        return ACLSHMEM_INNER_ERROR;
    }
    if (update_device_state() != 0) {
        rollback_team_creation(my_team.team_idx);
        if (update_device_state() != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("restore device state failed after team creation rollback!");
        }
        SHM_LOG_ERROR("create team failed, update state failed!");
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("create team success:" << team_config2string(&my_team));
    *new_team = my_team.team_idx;
    return 0;
}

int aclshmemi_team_split_2d_precheck(
    aclshmem_team_t p_team, int x_range, aclshmem_team_t* x_team, aclshmem_team_t* y_team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    if (x_team == nullptr || y_team == nullptr) {
        SHM_LOG_ERROR("output team is null.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (x_range <= 0) {
        SHM_LOG_ERROR("input x range must be larger than 0.");
        return ACLSHMEM_INVALID_PARAM;
    }

    *x_team = ACLSHMEM_TEAM_INVALID;
    *y_team = ACLSHMEM_TEAM_INVALID;
    if (!is_valid_team(p_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << p_team);
        return ACLSHMEM_INVALID_PARAM;
    }

    return ACLSHMEM_SUCCESS;
}

int aclshmemi_team_split_2d_x(
    aclshmem_team_t& parent_team, int32_t& x_team_counts, int32_t& src_size, int& x_range, aclshmem_team_t*& x_team)
{
    int start = 0;
    int32_t first_error = ACLSHMEM_SUCCESS;
    aclshmemx_team_t* src_team = &g_aclshmem_team_pool[parent_team];

    for (int i = 0; i < x_team_counts; ++i) {
        aclshmem_team_t my_xteam = ACLSHMEM_TEAM_INVALID;
        int x_stride = 1;
        int x_size = ((i == x_team_counts - 1) && (src_size % x_range != 0)) ? src_size % x_range : x_range;
        int ret = aclshmem_team_split_strided(parent_team, start, x_stride, x_size, &my_xteam);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_WARN("create x-axis team " << (i + 1) << " of " << x_team_counts << " failed");
            if (first_error == ACLSHMEM_SUCCESS) {
                first_error = ret;
            }
        }
        start += x_range;

        if (my_xteam != ACLSHMEM_TEAM_INVALID) {
            if (*x_team == ACLSHMEM_TEAM_INVALID) {
                *x_team = my_xteam;
                SHM_LOG_INFO("Current pe is " << src_team->mype << " , split x-axis succeed for x- " << i);
            } else {
                return ACLSHMEM_INNER_ERROR;
            }
        }
    }
    return first_error;
}

int aclshmemi_team_split_2d_y(
    aclshmem_team_t& parent_team, int32_t& y_team_counts, int32_t& src_size, int& x_range, aclshmem_team_t*& y_team)
{
    int start = 0;
    int32_t first_error = ACLSHMEM_SUCCESS;
    aclshmemx_team_t* src_team = &g_aclshmem_team_pool[parent_team];

    for (int i = 0; i < y_team_counts; ++i) {
        aclshmem_team_t my_yteam = ACLSHMEM_TEAM_INVALID;
        int y_stride = x_range;
        int remainder = src_size % x_range;
        int y_range = src_size / x_range;
        int y_size = (remainder && i < remainder) ? y_range + 1 : y_range;

        int ret = aclshmem_team_split_strided(parent_team, start, y_stride, y_size, &my_yteam);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_WARN("create y-axis team " << (i + 1) << " of " << y_team_counts << " failed");
            if (first_error == ACLSHMEM_SUCCESS) {
                first_error = ret;
            }
        }

        start += 1;
        if (my_yteam != ACLSHMEM_TEAM_INVALID) {
            if (*y_team == ACLSHMEM_TEAM_INVALID) {
                *y_team = my_yteam;
                SHM_LOG_INFO("Current pe is " << src_team->mype << " , split y-axis succeed for y- " << i);
            } else {
                return ACLSHMEM_INNER_ERROR;
            }
        }
    }
    return first_error;
}

namespace {

// signal_addr[0] is reserved by the signal APIs. One word per world PE and a
// final summary word are available for the parent-team failure agreement.
constexpr int32_t TEAM_SPLIT_STATUS_BASE = 1;
int32_t* team_split_status_slot(int32_t global_pe)
{
    if (global_pe < 0 || global_pe >= g_state.npes) {
        return nullptr;
    }
    return reinterpret_cast<int32_t*>(g_state.signal_addr) + TEAM_SPLIT_STATUS_BASE + global_pe;
}

int32_t* team_split_status_summary_slot()
{
    return reinterpret_cast<int32_t*>(g_state.signal_addr) + TEAM_SPLIT_STATUS_BASE + g_state.npes;
}

bool team_split_is_member(const aclshmemx_team_t& team) { return team.mype >= 0 && team.mype < team.size; }

int32_t aclshmemi_team_split_status_prepare(aclshmem_team_t parent_team)
{
    const aclshmemx_team_t& team = g_aclshmem_team_pool[parent_team];
    if (!team_split_is_member(team) || g_state.signal_addr == 0) {
        return team_split_is_member(team) ? ACLSHMEM_INNER_ERROR : ACLSHMEM_SUCCESS;
    }

    int32_t ret = ACLSHMEM_SUCCESS;
    if (team.mype == 0) {
        // The root reads every member's slot from its own symmetric array.
        // Clear those root-local slots here; clearing each PE's local copy
        // leaves stale failures behind in the root's array.
        for (int32_t i = 0; i < team.size; ++i) {
            if (aclrtMemset(team_split_status_slot(team.pe_mapping[i]), sizeof(int32_t), 0, sizeof(int32_t)) !=
                ACL_SUCCESS) {
                SHM_LOG_ERROR("reset team split status failed for PE " << team.pe_mapping[i]);
                ret = ACLSHMEM_INNER_ERROR;
            }
        }
        if (aclrtMemset(team_split_status_summary_slot(), sizeof(int32_t), 0, sizeof(int32_t)) != ACL_SUCCESS) {
            SHM_LOG_ERROR("reset team split status summary failed");
            ret = ACLSHMEM_INNER_ERROR;
        }
    }

    // The parent-team barrier makes the root's status resets visible before
    // any member starts allocating X/Y teams.
    aclshmem_barrier(parent_team);
    return ret;
}

int32_t aclshmemi_team_split_status_gate(aclshmem_team_t parent_team, int32_t local_status)
{
    const aclshmemx_team_t& team = g_aclshmem_team_pool[parent_team];
    if (!team_split_is_member(team) || g_state.signal_addr == 0) {
        return team_split_is_member(team) ? ACLSHMEM_INNER_ERROR : local_status;
    }

    const int32_t root_pe = team.pe_mapping[0];
    int32_t* local_status_slot = team_split_status_slot(g_state.mype);
    if (local_status_slot == nullptr) {
        return ACLSHMEM_INNER_ERROR;
    }

    // Report only failures. Each PE owns one slot in the root's symmetric
    // status array, so SET is sufficient and works for all transports.
    if (local_status != ACLSHMEM_SUCCESS) {
        if (g_state.mype == root_pe) {
            if (aclrtMemset(local_status_slot, sizeof(int32_t), 1, sizeof(int32_t)) != ACL_SUCCESS) {
                SHM_LOG_ERROR("record local team split failure failed");
            }
        } else {
            aclshmemx_signal_op_on_stream(
                local_status_slot, 1, ACLSHMEM_SIGNAL_SET, root_pe, g_state_host.default_stream);
        }
    }
    aclshmem_barrier(parent_team);

    int32_t global_failure = local_status != ACLSHMEM_SUCCESS ? 1 : 0;
    if (team.mype == 0) {
        for (int32_t i = 0; i < team.size; ++i) {
            const int32_t global_pe = team.pe_mapping[i];
            if (team_split_status_slot(global_pe) == nullptr) {
                SHM_LOG_ERROR("invalid team split PE " << global_pe);
                global_failure = 1;
                continue;
            }
            int32_t status = 0;
            if (aclrtMemcpy(
                    &status, sizeof(status), team_split_status_slot(global_pe), sizeof(status),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                SHM_LOG_ERROR("read team split status failed for PE " << global_pe);
                global_failure = 1;
                continue;
            }
            global_failure = global_failure || status != 0;
        }

        int32_t summary = global_failure ? 1 : 0;
        if (aclrtMemset(team_split_status_summary_slot(), sizeof(summary), summary, sizeof(summary)) != ACL_SUCCESS) {
            SHM_LOG_ERROR("write team split status summary failed");
            global_failure = 1;
            summary = 1;
        }
        for (int32_t i = 1; i < team.size; ++i) {
            aclshmemx_signal_op_on_stream(
                team_split_status_summary_slot(), summary, ACLSHMEM_SIGNAL_SET, team.pe_mapping[i],
                g_state_host.default_stream);
        }
    }

    // Ensure every member observes the root's summary before deciding whether
    // to destroy locally-created child teams.
    aclshmem_barrier(parent_team);
    int32_t summary = 0;
    if (aclrtMemcpy(
            &summary, sizeof(summary), team_split_status_summary_slot(), sizeof(summary), ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        SHM_LOG_ERROR("read team split status summary failed");
        summary = 1;
    }
    if (summary != 0) {
        return local_status == ACLSHMEM_SUCCESS ? ACLSHMEM_INNER_ERROR : local_status;
    }
    return local_status;
}

} // namespace

int aclshmem_team_split_2d(aclshmem_team_t parent_team, int x_range, aclshmem_team_t* x_team, aclshmem_team_t* y_team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    auto ret = aclshmemi_team_split_2d_precheck(parent_team, x_range, x_team, y_team);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    int32_t local_ret = aclshmemi_team_split_status_prepare(parent_team);

    aclshmemx_team_t* src_team = &g_aclshmem_team_pool[parent_team];
    int32_t src_start = src_team->start;
    int32_t src_stride = src_team->stride;
    int32_t src_size = src_team->size;
    if (x_range > src_size) {
        x_range = src_size;
    }
    int32_t x_team_counts = static_cast<int32_t>(std::ceil(src_size / static_cast<double>(x_range)));
    int32_t y_team_counts = x_range;

    if (local_ret == ACLSHMEM_SUCCESS) {
        local_ret = aclshmemi_team_split_2d_x(parent_team, x_team_counts, src_size, x_range, x_team);
    }
    if (local_ret == ACLSHMEM_SUCCESS) {
        local_ret = aclshmemi_team_split_2d_y(parent_team, y_team_counts, src_size, x_range, y_team);
    }

    ret = aclshmemi_team_split_status_gate(parent_team, local_ret);
    if (ret != ACLSHMEM_SUCCESS) {
        if (*x_team != ACLSHMEM_TEAM_INVALID) {
            aclshmem_team_destroy(*x_team);
            *x_team = ACLSHMEM_TEAM_INVALID;
        }
        if (*y_team != ACLSHMEM_TEAM_INVALID) {
            aclshmem_team_destroy(*y_team);
            *y_team = ACLSHMEM_TEAM_INVALID;
        }
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmem_team_translate_pe(aclshmem_team_t src_team, int32_t src_pe, aclshmem_team_t dest_team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    if (!is_valid_team(src_team) || !is_valid_team(dest_team)) {
        return -1;
    }

    if (src_pe < 0 || src_pe >= g_aclshmem_team_pool[src_team].size) {
        return -1;
    }
    int global_pe = g_aclshmem_team_pool[src_team].pe_mapping[src_pe];
    return g_aclshmem_team_pool[dest_team].pe_mapping[global_pe + ACLSHMEM_MAX_PES];
}

void aclshmem_team_destroy(aclshmem_team_t team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    if (!is_valid_team(team)) {
        SHM_LOG_WARN("input team is invalid!, team: " << team);
        return;
    }

    device_team_destroy(team);
    release_team_idx(team);
    if (update_device_state() != ACLSHMEM_SUCCESS) {
        SHM_LOG_WARN("update state failed when destroy team!");
    }
}

int32_t aclshmem_my_pe(void) { return g_state.mype; }

int32_t aclshmem_n_pes(void) { return g_state.npes; }

int32_t aclshmem_team_my_pe(aclshmem_team_t team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    if (is_valid_team(team)) {
        return g_aclshmem_team_pool[team].mype;
    } else {
        return -1;
    }
}

int32_t aclshmem_team_n_pes(aclshmem_team_t team)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    if (is_valid_team(team)) {
        return g_aclshmem_team_pool[team].size;
    } else {
        return -1;
    }
}

int aclshmem_team_get_config(aclshmem_team_t team, aclshmem_team_config_t* config)
{
    std::lock_guard<std::recursive_mutex> lock(g_team_mutex);
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    ACLSHMEM_CHECK_RET(config == nullptr);
    if (is_valid_team(team)) {
        *config = g_aclshmem_team_pool[team].config;
        return 0;
    } else {
        return ACLSHMEM_INVALID_PARAM;
    }
}
