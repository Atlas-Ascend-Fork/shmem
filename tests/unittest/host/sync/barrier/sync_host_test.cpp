/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "sync_kernel.h"

constexpr int32_t ACLSHMEM_SYNC_TEST_NUM = 3;
constexpr unsigned int ACLSHMEM_SYNC_TIMEOUT_SECONDS = 30;

static void test_sync_black_box(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    alarm(ACLSHMEM_SYNC_TIMEOUT_SECONDS);

    uint64_t* addr_dev = static_cast<uint64_t*>(aclshmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t* addr_host;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&addr_host), sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= ACLSHMEM_SYNC_TEST_NUM; i++) {
        std::cout << "[TEST] sync test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
        sync_increase_do(stream, util_get_ffts_config(), (uint8_t*)addr_dev, rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ((*addr_host), i);
        aclshmemi_control_barrier_all();
    }

    uint64_t* addr_dev_vec = static_cast<uint64_t*>(aclshmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev_vec, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t* addr_host_vec;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&addr_host_vec), sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= ACLSHMEM_SYNC_TEST_NUM; i++) {
        std::cout << "[TEST] vec sync test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
        sync_increase_vec_do(stream, util_get_ffts_config(), (uint8_t*)addr_dev_vec, rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(
            aclrtMemcpy(addr_host_vec, sizeof(uint64_t), addr_dev_vec, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ((*addr_host_vec), i);
        aclshmemi_control_barrier_all();
    }

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);
    aclshmem_free(addr_dev);
    ASSERT_EQ(aclrtFreeHost(addr_host_vec), 0);
    aclshmem_free(addr_dev_vec);

    test_finalize(stream, device_id);
    alarm(0);
}

static void test_sync_black_box_odd_team(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    alarm(ACLSHMEM_SYNC_TIMEOUT_SECONDS);

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    uint64_t* addr_dev = static_cast<uint64_t*>(aclshmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t* addr_host;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&addr_host), sizeof(uint64_t)), 0);

    uint64_t* addr_dev_vec = static_cast<uint64_t*>(aclshmem_malloc(sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev_vec, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t* addr_host_vec;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&addr_host_vec), sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= ACLSHMEM_SYNC_TEST_NUM; i++) {
        if (rank_id & 1) {
            std::cout << "[TEST] sync test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
            sync_increase_do_odd_team(stream, util_get_ffts_config(), (uint8_t*)addr_dev, rank_id, n_ranks, team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
            ASSERT_EQ(
                aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
            ASSERT_EQ((*addr_host), i);
        }
        aclshmemi_control_barrier_all();
    }

    for (int32_t i = 1; i <= ACLSHMEM_SYNC_TEST_NUM; i++) {
        if (rank_id & 1) {
            std::cout << "[TEST] vec sync test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
            sync_increase_vec_do_odd_team(
                stream, util_get_ffts_config(), (uint8_t*)addr_dev_vec, rank_id, n_ranks, team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
            ASSERT_EQ(
                aclrtMemcpy(addr_host_vec, sizeof(uint64_t), addr_dev_vec, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
                0);
            ASSERT_EQ((*addr_host_vec), i);
        }
        aclshmemi_control_barrier_all();
    }

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);
    aclshmem_free(addr_dev);
    ASSERT_EQ(aclrtFreeHost(addr_host_vec), 0);
    aclshmem_free(addr_dev_vec);

    aclshmem_team_destroy(team_odd);

    test_finalize(stream, device_id);
    alarm(0);
}

static void test_sync_edge_cases(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    alarm(ACLSHMEM_SYNC_TIMEOUT_SECONDS);

    uint64_t* addr_dev = static_cast<uint64_t*>(aclshmem_malloc(sizeof(uint64_t)));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t* addr_host;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&addr_host), sizeof(uint64_t)), 0);

    // Exercise a world sync with exactly one AIV per rank.
    for (int32_t i = 1; i <= ACLSHMEM_SYNC_TEST_NUM; ++i) {
        sync_increase_single_aiv_do(
            stream, util_get_ffts_config(), reinterpret_cast<uint8_t*>(addr_dev), rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ(*addr_host, static_cast<uint64_t>(i));
        ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    }

    // Exercise v4 directly with one AIV. Normal small-rank UT configurations
    // select v3 through the public dispatcher and otherwise leave v4 untested.
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    for (int32_t i = 1; i <= ACLSHMEM_SYNC_TEST_NUM; ++i) {
        sync_v4_single_aiv_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t*>(addr_dev), rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ(*addr_host, static_cast<uint64_t>(i));
        ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    }

    // Repeated generations with short, rank/core-dependent arrival skew.
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    sync_stress_vec_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t*>(addr_dev), rank_id, n_ranks);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(*addr_host, ACLSHMEM_SYNC_STRESS_ITERATIONS);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    // Exercise repeated v4 generations with all 32 AIVs even when n_ranks is
    // below the production switch threshold.
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    sync_v4_stress_vec_do(
        ACLSHMEM_SYNC_STRESS_AIV_COUNT, stream, util_get_ffts_config(), reinterpret_cast<uint8_t*>(addr_dev), rank_id,
        n_ranks);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(*addr_host, ACLSHMEM_SYNC_STRESS_ITERATIONS);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    // Cover the public Host sync entry points.
    for (int32_t i = 0; i < ACLSHMEM_SYNC_TEST_NUM; ++i) {
        aclshmem_sync_all();
        aclshmem_sync(ACLSHMEM_TEAM_WORLD);
        ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    }

    // A singleton team covers the size == 1 boundary. Non-members receive an
    // invalid team handle and the public Host APIs must return without blocking.
    aclshmem_team_t singleton_team = ACLSHMEM_TEAM_INVALID;
    ASSERT_EQ(aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, 0, 1, 1, &singleton_team), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    if (rank_id == 0) {
        ASSERT_NE(singleton_team, ACLSHMEM_TEAM_INVALID);
        sync_singleton_team_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t*>(addr_dev), singleton_team);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ(*addr_host, static_cast<uint64_t>(1));
    } else {
        ASSERT_EQ(singleton_team, ACLSHMEM_TEAM_INVALID);
    }
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    aclshmem_sync(singleton_team);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    aclshmem_team_destroy(singleton_team);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);
    aclshmem_free(addr_dev);
    test_finalize(stream, device_id);
    alarm(0);
}

static void test_sync_core_soft_mixed(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    alarm(ACLSHMEM_SYNC_TIMEOUT_SECONDS);

    int32_t active_device_id = -1;
    ASSERT_EQ(aclrtGetDevice(&active_device_id), ACL_SUCCESS);

    int64_t physical_aiv_count = 0;
    auto ret = aclrtGetDeviceInfo(active_device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &physical_aiv_count);
    if (ret != ACL_SUCCESS || physical_aiv_count <= 0) {
        ret = aclGetDeviceCapability(active_device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &physical_aiv_count);
    }
    ASSERT_EQ(ret, ACL_SUCCESS) << "failed to query vector core count for device " << active_device_id;
    ASSERT_GT(physical_aiv_count, 0);
    ASSERT_LE(physical_aiv_count, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));

    int64_t physical_aic_count = 0;
    ret = aclGetDeviceCapability(active_device_id, ACL_DEVICE_INFO_AI_CORE_NUM, &physical_aic_count);
    ASSERT_EQ(ret, ACL_SUCCESS) << "failed to query AI Core count for device " << active_device_id;
    ASSERT_GT(physical_aic_count, 0);
    ASSERT_LE(physical_aic_count, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));

    const uint32_t stress_aiv_count = static_cast<uint32_t>(physical_aiv_count);
    const uint32_t mixed_block_count = static_cast<uint32_t>(physical_aic_count);
    const size_t completion_size = static_cast<size_t>(stress_aiv_count) * ACLSHMEM_SYNCBIT_SIZE;

    uint32_t* observed_aiv_count_dev = nullptr;
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&observed_aiv_count_dev), sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
        ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(observed_aiv_count_dev, sizeof(uint32_t), 0, sizeof(uint32_t)), ACL_SUCCESS);

    uint8_t* completion_dev = nullptr;
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&completion_dev), completion_size, ACL_MEM_MALLOC_HUGE_FIRST),
        ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(completion_dev, completion_size, 0, completion_size), ACL_SUCCESS);
    uint8_t* completion_host = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&completion_host), completion_size), ACL_SUCCESS);

    sync_core_soft_mixed_stress_do(
        mixed_block_count, stream, util_get_ffts_config(), reinterpret_cast<uint8_t*>(observed_aiv_count_dev),
        completion_dev, rank_id);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    uint32_t observed_aiv_count = 0;
    ASSERT_EQ(
        aclrtMemcpy(
            &observed_aiv_count, sizeof(uint32_t), observed_aiv_count_dev, sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    ASSERT_EQ(observed_aiv_count, stress_aiv_count)
        << "mixed kernel GetBlockNum() * GetTaskRation() does not match the runtime AIV count";

    ASSERT_EQ(
        aclrtMemcpy(completion_host, completion_size, completion_dev, completion_size, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    for (uint32_t aiv_id = 0; aiv_id < stress_aiv_count; ++aiv_id) {
        const auto* completion_slot =
            reinterpret_cast<const uint32_t*>(completion_host + static_cast<size_t>(aiv_id) * ACLSHMEM_SYNCBIT_SIZE);
        ASSERT_EQ(*completion_slot, 1U) << "rank " << rank_id << " AIV " << aiv_id
                                        << " did not finish aclshmemi_sync_core_soft";
    }
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    ASSERT_EQ(aclrtFreeHost(completion_host), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(completion_dev), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(observed_aiv_count_dev), ACL_SUCCESS);
    test_finalize(stream, device_id);
    alarm(0);
}

TEST(TEST_SYNC_API, test_sync_black_box)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_sync_black_box, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_sync_black_box_odd_team)
{
    if (test_gnpu_num < 4) {
        GTEST_SKIP() << "odd-team sync needs at least 4 ranks";
    }
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_sync_black_box_odd_team, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_sync_edge_cases)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_sync_edge_cases, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_sync_core_soft_mixed)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_sync_core_soft_mixed, local_mem_size, process_count);
}

static void test_sync_mixed_stress(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    alarm(ACLSHMEM_SYNC_TIMEOUT_SECONDS);

    constexpr size_t report_bytes = 2 * ACLSHMEM_SYNC_MIXED_SLOTS_PER_SIDE * ACLSHMEM_SYNCBIT_SIZE;
    uint8_t* reports = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&reports), report_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
    std::vector<int32_t> host(report_bytes / sizeof(int32_t));
    // Change the worker budget across launches while reusing the same team's
    // generations. Check completion on every AIC/AIV, including non-polling AIVs.
    // With four PEs, two logical blocks expose the old MIX 1:2 V4 round mismatch.
    for (uint32_t blocks : {1U, 2U, 8U, 1U}) {
        ASSERT_EQ(aclrtMemset(reports, report_bytes, 0xff, report_bytes), ACL_SUCCESS);
        sync_mixed_stress_do(
            blocks, stream, util_get_ffts_config(), reports, ACLSHMEM_TEAM_WORLD, ACLSHMEM_SYNC_STRESS_ITERATIONS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(host.data(), report_bytes, reports, report_bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        for (uint32_t side = 0; side < 2; ++side) {
            const uint32_t cores = side == 0 ? blocks : blocks * 2;
            for (uint32_t core = 0; core < cores; ++core) {
                const size_t offset =
                    (side * ACLSHMEM_SYNC_MIXED_SLOTS_PER_SIDE + core) * ACLSHMEM_SYNCBIT_SIZE / sizeof(int32_t);
                EXPECT_EQ(host[offset], ACLSHMEM_SYNC_STRESS_ITERATIONS) << "side=" << side << " core=" << core;
            }
        }
        ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    }
    ASSERT_EQ(aclrtFree(reports), ACL_SUCCESS);
    test_finalize(stream, device_id);
    alarm(0);
}

TEST(TEST_SYNC_API, test_sync_mixed_stress)
{
    test_mutil_task(test_sync_mixed_stress, 16UL * 1024UL * 1024UL, test_gnpu_num);
}
