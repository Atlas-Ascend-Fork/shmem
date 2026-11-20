/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <set>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"

void aclshmemi_team_fail_next_device_update_for_test();
void aclshmemi_team_fill_mask_for_test(int32_t free_team);
void aclshmemi_team_restore_mask_for_test();

namespace {
void test_aclshmem_team_allocation_state(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    std::vector<aclshmem_team_t> boundary_teams;
    constexpr int team_count = 65;
    for (int i = 0; i < team_count; ++i) {
        aclshmem_team_t team = ACLSHMEM_TEAM_INVALID;
        ASSERT_EQ(aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, 0, 1, n_ranks, &team), ACLSHMEM_SUCCESS);
        boundary_teams.push_back(team);
    }
    ASSERT_GE(boundary_teams.back(), 64);
    EXPECT_EQ(aclshmem_team_n_pes(boundary_teams.back()), n_ranks);

    const aclshmem_team_t first_dynamic_team = boundary_teams.front();
    for (auto team = boundary_teams.rbegin(); team != boundary_teams.rend(); ++team) {
        aclshmem_team_destroy(*team);
    }

    constexpr int concurrent_team_count = 8;
    std::vector<aclshmem_team_t> concurrent_teams(concurrent_team_count, ACLSHMEM_TEAM_INVALID);
    std::vector<int32_t> concurrent_status(concurrent_team_count, ACLSHMEM_INNER_ERROR);
    std::vector<int32_t> set_device_status(concurrent_team_count, ACLSHMEM_INNER_ERROR);
    std::vector<std::thread> workers;
    workers.reserve(concurrent_team_count);
    for (int i = 0; i < concurrent_team_count; ++i) {
        workers.emplace_back([&, i]() {
            // ACL runtime device context is thread-local; bind each worker to this PE's device.
            set_device_status[i] = aclrtSetDevice(device_id);
            if (set_device_status[i] != ACL_SUCCESS) {
                return;
            }
            concurrent_status[i] =
                aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, 0, 1, n_ranks, &concurrent_teams[i]);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    std::set<aclshmem_team_t> unique_teams;
    for (int i = 0; i < concurrent_team_count; ++i) {
        EXPECT_EQ(set_device_status[i], ACL_SUCCESS);
        EXPECT_EQ(concurrent_status[i], ACLSHMEM_SUCCESS);
        EXPECT_GT(concurrent_teams[i], ACLSHMEM_TEAM_WORLD);
        unique_teams.insert(concurrent_teams[i]);
    }
    EXPECT_EQ(unique_teams.size(), concurrent_team_count);
    for (const auto team : concurrent_teams) {
        aclshmem_team_destroy(team);
    }

    aclshmemi_team_fail_next_device_update_for_test();
    aclshmem_team_t failed_team = ACLSHMEM_TEAM_INVALID;
    EXPECT_EQ(aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, 0, 1, n_ranks, &failed_team), ACLSHMEM_INNER_ERROR);
    EXPECT_EQ(failed_team, ACLSHMEM_TEAM_INVALID);
    EXPECT_EQ(g_state.team_pools[first_dynamic_team], nullptr);

    aclshmem_team_t recovered_team = ACLSHMEM_TEAM_INVALID;
    ASSERT_EQ(aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, 0, 1, n_ranks, &recovered_team), ACLSHMEM_SUCCESS);
    EXPECT_EQ(recovered_team, first_dynamic_team);
    aclshmem_team_destroy(recovered_team);

    aclshmemi_team_fail_next_device_update_for_test();
    aclshmem_team_t x_team = ACLSHMEM_TEAM_INVALID;
    aclshmem_team_t y_team = ACLSHMEM_TEAM_INVALID;
    EXPECT_EQ(aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, 1, &x_team, &y_team), ACLSHMEM_INNER_ERROR);
    EXPECT_EQ(x_team, ACLSHMEM_TEAM_INVALID);
    EXPECT_EQ(y_team, ACLSHMEM_TEAM_INVALID);
    EXPECT_EQ(g_state.team_pools[first_dynamic_team], nullptr);
    if (x_team != ACLSHMEM_TEAM_INVALID) {
        aclshmem_team_destroy(x_team);
    }
    if (y_team != ACLSHMEM_TEAM_INVALID) {
        aclshmem_team_destroy(y_team);
    }

    aclshmemi_team_fill_mask_for_test(first_dynamic_team);
    x_team = ACLSHMEM_TEAM_INVALID;
    y_team = ACLSHMEM_TEAM_INVALID;
    EXPECT_EQ(aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, 1, &x_team, &y_team), ACLSHMEM_INNER_ERROR);
    EXPECT_EQ(x_team, ACLSHMEM_TEAM_INVALID);
    EXPECT_EQ(y_team, ACLSHMEM_TEAM_INVALID);
    EXPECT_EQ(g_state.team_pools[first_dynamic_team], nullptr);
    if (x_team != ACLSHMEM_TEAM_INVALID) {
        aclshmem_team_destroy(x_team);
    }
    if (y_team != ACLSHMEM_TEAM_INVALID) {
        aclshmem_team_destroy(y_team);
    }
    aclshmemi_team_restore_mask_for_test();

    // A single PE failing during Y creation must make every parent-team member
    // roll back its locally created X/Y teams before returning.
    if (n_ranks == 2) {
        if (rank_id == 0) {
            aclshmemi_team_fill_mask_for_test(first_dynamic_team);
        }
        x_team = ACLSHMEM_TEAM_INVALID;
        y_team = ACLSHMEM_TEAM_INVALID;
        EXPECT_EQ(aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, 2, &x_team, &y_team), ACLSHMEM_INNER_ERROR);
        if (rank_id == 0) {
            aclshmemi_team_restore_mask_for_test();
        }
        EXPECT_EQ(x_team, ACLSHMEM_TEAM_INVALID);
        EXPECT_EQ(y_team, ACLSHMEM_TEAM_INVALID);
        EXPECT_EQ(g_state.team_pools[first_dynamic_team], nullptr);

        x_team = ACLSHMEM_TEAM_INVALID;
        y_team = ACLSHMEM_TEAM_INVALID;
        EXPECT_EQ(aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, 2, &x_team, &y_team), ACLSHMEM_SUCCESS);
        EXPECT_NE(x_team, ACLSHMEM_TEAM_INVALID);
        EXPECT_NE(y_team, ACLSHMEM_TEAM_INVALID);
        aclshmem_team_destroy(x_team);
        aclshmem_team_destroy(y_team);
    }

    // A failed split must not leave a stale parent-team failure bit that
    // poisons the next collective split.
    x_team = ACLSHMEM_TEAM_INVALID;
    y_team = ACLSHMEM_TEAM_INVALID;
    EXPECT_EQ(aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, 1, &x_team, &y_team), ACLSHMEM_SUCCESS);
    EXPECT_NE(x_team, ACLSHMEM_TEAM_INVALID);
    EXPECT_NE(y_team, ACLSHMEM_TEAM_INVALID);
    aclshmem_team_destroy(x_team);
    aclshmem_team_destroy(y_team);

    test_finalize(stream, device_id);
}
} // namespace

TEST(TestTeamApi, TestShmemTeamAllocationState)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_team_allocation_state, local_mem_size, process_count);
}
