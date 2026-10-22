/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "atomic.h"

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)

extern void test_rdma_atomic_high_half_uint32_do(
    uint32_t block_dim, void* stream, uint32_t* gva, uint32_t* result, uint32_t value, uint32_t compare,
    bool compare_swap, uint64_t config);
extern void test_rdma_atomic_contended_fetch_add_uint32_do(
    uint32_t block_dim, void* stream, uint32_t* gva, uint32_t* result, uint32_t iterations, uint64_t config);

namespace {

constexpr uint32_t LOW_HALF_SENTINEL = 0xa5a55a5aU;
constexpr size_t RESULT_BYTES = 512;

void test_aclshmem_rdma_atomic_high_half_uint32_mem(int pe_id, int n_pes, uint64_t local_mem_size)
{
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_rdma_init(pe_id, n_pes, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    uint32_t pair[2] = {LOW_HALF_SENTINEL, 0xfffffffeU};
    auto* ptr = static_cast<uint32_t*>(aclshmem_malloc(sizeof(pair)));
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(aclrtMemcpy(ptr, sizeof(pair), pair, sizeof(pair), ACL_MEMCPY_HOST_TO_DEVICE), 0);
    uint32_t* result = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&result), RESULT_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    ASSERT_EQ(aclrtMemset(result, RESULT_BYTES, 0xff, RESULT_BYTES), 0);

    struct TestCase {
        bool compare_swap;
        uint32_t value;
        uint32_t compare;
        uint32_t expected_old;
        uint32_t expected_high;
    };
    const std::array<TestCase, 5> cases{{
        {false, 3U, 0U, 0xfffffffeU, 1U},                           // FAA wraparound
        {true, 0x89abcdefU, 1U, 1U, 0x89abcdefU},                   // CAS match
        {true, 0x12345678U, 0x89abcdeeU, 0x89abcdefU, 0x89abcdefU}, // CAS mismatch
        {false, 0U, 0U, 0x89abcdefU, 0x89abcdefU},                  // FAA no-op
        {true, 0x89abcdefU, 0x89abcdefU, 0x89abcdefU, 0x89abcdefU}, // CAS no-op
    }};
    for (size_t step = 0; step < cases.size(); ++step) {
        SCOPED_TRACE(step);
        const auto& test_case = cases[step];
        // Finish initialization or the preceding step's observations on every PE.
        ASSERT_EQ(aclshmemi_control_barrier_all(), 0);
        test_rdma_atomic_high_half_uint32_do(
            1, stream, ptr, result, test_case.value, test_case.compare, test_case.compare_swap, util_get_ffts_config());
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclshmemi_control_barrier_all(), 0);

        uint32_t old = 0;
        ASSERT_EQ(aclrtMemcpy(&old, sizeof(old), result, sizeof(old), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ(aclrtMemcpy(pair, sizeof(pair), ptr, sizeof(pair), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        EXPECT_EQ(old, test_case.expected_old);
        EXPECT_EQ(pair[0], LOW_HALF_SENTINEL);
        EXPECT_EQ(pair[1], test_case.expected_high);
    }

    aclshmem_free(ptr);
    EXPECT_EQ(aclrtFree(result), 0);
    test_finalize(stream, pe_id % test_gnpu_num + test_first_npu);
}

void test_aclshmem_rdma_atomic_contended_fetch_add_uint32_mem(int pe_id, int n_pes, uint64_t local_mem_size)
{
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_rdma_init(pe_id, n_pes, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    constexpr uint32_t iterations = 64;
    uint32_t pair[2] = {LOW_HALF_SENTINEL, 0U};
    auto* ptr = static_cast<uint32_t*>(aclshmem_malloc(sizeof(pair)));
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(aclrtMemcpy(ptr, sizeof(pair), pair, sizeof(pair), ACL_MEMCPY_HOST_TO_DEVICE), 0);
    uint32_t* result = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&result), RESULT_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    ASSERT_EQ(aclrtMemset(result, RESULT_BYTES, 0xff, RESULT_BYTES), 0);
    std::vector<uint32_t> old_values(iterations);

    ASSERT_EQ(aclshmemi_control_barrier_all(), 0);
    if (pe_id != 0) {
        test_rdma_atomic_contended_fetch_add_uint32_do(1, stream, ptr, result, iterations, util_get_ffts_config());
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(
            aclrtMemcpy(
                old_values.data(), iterations * sizeof(uint32_t), result, iterations * sizeof(uint32_t),
                ACL_MEMCPY_DEVICE_TO_HOST),
            0);
    }

    // Gather only after each source has completed its atomics and copied the returned values.
    std::vector<uint32_t> all_old_values(n_pes * iterations);
    ASSERT_EQ(
        g_boot_handle.allgather(
            old_values.data(), all_old_values.data(), iterations * sizeof(uint32_t), &g_boot_handle),
        0);
    ASSERT_EQ(aclrtMemcpy(pair, sizeof(pair), ptr, sizeof(pair), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    const uint32_t total_updates = (n_pes - 1) * iterations;
    EXPECT_EQ(pair[0], LOW_HALF_SENTINEL);
    EXPECT_EQ(pair[1], pe_id == 0 ? total_updates : 0U);
    if (pe_id == 0) {
        // PE0 did not issue atomics. The sources' old values must form [0, total_updates).
        auto first = all_old_values.begin() + iterations;
        std::sort(first, all_old_values.end());
        for (uint32_t i = 0; i < total_updates; ++i) {
            EXPECT_EQ(first[i], i) << "old-value index " << i;
        }
    }

    aclshmem_free(ptr);
    EXPECT_EQ(aclrtFree(result), 0);
    test_finalize(stream, pe_id % test_gnpu_num + test_first_npu);
}

} // namespace

TEST(TestMemApi, TestShmemRdmaAtomicHighHalfuint32Mem)
{
    SKIP_IF_UNSUPPORTED_ATOMIC();
    if (test_global_ranks < 2 || (test_global_ranks & (test_global_ranks - 1)) != 0) {
        GTEST_SKIP() << "RDMA atomic high-half test requires a power-of-two PE count of at least 2";
    }
    const uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_rdma_atomic_high_half_uint32_mem, local_mem_size, test_gnpu_num);
}

TEST(TestMemApi, TestShmemRdmaAtomicContendedFetchAdduint32Mem)
{
    SKIP_IF_UNSUPPORTED_ATOMIC();
    if (test_global_ranks < 4 || (test_global_ranks & (test_global_ranks - 1)) != 0) {
        GTEST_SKIP() << "RDMA atomic contention test requires a power-of-two PE count of at least 4";
    }
    const uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_rdma_atomic_contended_fetch_add_uint32_mem, local_mem_size, test_gnpu_num);
}

#endif // ACLSHMEMI_RDMA_K_BACKEND_XSCALE || ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
