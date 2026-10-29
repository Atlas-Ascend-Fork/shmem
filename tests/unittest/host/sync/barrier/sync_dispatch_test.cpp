/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <gtest/gtest.h>

namespace {
using aclshmem_team_t = int32_t;
struct aclshmemx_team_t {
    int32_t size;
};
struct SyncState {
    aclshmemx_team_t* team_pools[2];
};
enum class Path { NONE, V3, V4 };

aclshmemx_team_t test_team{};
SyncState state{{nullptr, &test_team}};
int32_t block_num = 0;
int32_t task_ratio = 1;
int32_t state_reads = 0;
int32_t calls = 0;
Path called_path = Path::NONE;
aclshmem_team_t called_team = -1;
bool called_aiv_only = true;

namespace AscendC {
inline int32_t GetBlockNum() { return block_num; }
// The old selector reads this value; the fixed selector must be independent of it.
inline int32_t GetTaskRation() { return task_ratio; }
} // namespace AscendC

SyncState* aclshmemi_get_state()
{
    ++state_reads;
    return &state;
}

template <bool IS_AIV_ONLY>
void aclshmemi_sync_npu_v3(aclshmem_team_t team)
{
    ++calls;
    called_path = Path::V3;
    called_team = team;
    called_aiv_only = IS_AIV_ONLY;
}

template <bool IS_AIV_ONLY>
void aclshmemi_sync_npu_v4(aclshmem_team_t team)
{
    ++calls;
    called_path = Path::V4;
    called_team = team;
    called_aiv_only = IS_AIV_ONLY;
}

// CMake extracts the production dispatcher from shmemi_device_cc.h into the
// build directory. Compile it against these CPU recorders without duplicating
// the selection expression or changing the production header layout.
#define ACLSHMEM_DEVICE inline
#include "sync_dispatch_under_test.inc"
#undef ACLSHMEM_DEVICE

void Reset(int32_t team_size, int32_t blocks, int32_t ratio)
{
    test_team.size = team_size;
    block_num = blocks;
    task_ratio = ratio;
    state_reads = 0;
    calls = 0;
    called_path = Path::NONE;
    called_team = -1;
    called_aiv_only = true;
}

void ExpectMixedPath(int32_t team_size, int32_t blocks, Path expected)
{
    for (int32_t ratio : {1, 2}) {
        SCOPED_TRACE(::testing::Message() << "team_size=" << team_size << " blocks=" << blocks << " ratio=" << ratio);
        Reset(team_size, blocks, ratio);
        aclshmemi_sync<false>(1);
        EXPECT_EQ(called_path, expected);
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(state_reads, 1);
        EXPECT_EQ(called_team, 1);
        EXPECT_FALSE(called_aiv_only);
    }
}

TEST(TEST_SYNC_DISPATCH_CPU, Mixed64Peers32BlocksSelectV4OnBothSides)
{
    // Restoring GetBlockNum() * GetTaskRation() makes the AIV side choose V3.
    ExpectMixedPath(64, 32, Path::V4);
}

TEST(TEST_SYNC_DISPATCH_CPU, MixedSelectionBoundaries)
{
    ExpectMixedPath(1, 1, Path::V3);
    ExpectMixedPath(31, 1, Path::V3);
    ExpectMixedPath(32, 1, Path::V3);
    ExpectMixedPath(33, 32, Path::V4);
    ExpectMixedPath(63, 32, Path::V4);
    ExpectMixedPath(65, 32, Path::V4);
    ExpectMixedPath(64, 64, Path::V3);
}

TEST(TEST_SYNC_DISPATCH_CPU, ForwardsAivOnlyTemplateAndTeam)
{
    Reset(64, 32, 1);
    aclshmemi_sync<true>(1);
    EXPECT_EQ(called_path, Path::V4);
    EXPECT_EQ(called_team, 1);
    EXPECT_TRUE(called_aiv_only);
    EXPECT_EQ(calls, 1);
}

TEST(TEST_SYNC_DISPATCH_CPU, NonMemberSkipsStateAndBothPaths)
{
    Reset(64, 32, 2);
    aclshmemi_sync<false>(-1);
    EXPECT_EQ(called_path, Path::NONE);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(state_reads, 0);
}
} // namespace
