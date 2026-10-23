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
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <string.h>

#include "acl/acl.h"
#include "init/shmemi_init.h"
#include "shmemi_host_common.h"
#include "rdma_mem_kernel.h"
#include "utils/exception/shmemi_device_rdma_exception_report_kernel.h"
#include "utils/exception/shmem_exception_report.h"
#include "utils/under_api/dl_hccp_def.h"
#include "unittest_main_test.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern int32_t test_rdma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st);
extern void test_finalize(aclrtStream stream, int device_id);

namespace {
constexpr int TIMEOUT = 30;
constexpr size_t RDMA_TEST_VA_GAP_GRANULARITY = 2UL * 1024UL * 1024UL;

// Aggregate tests are available on the XSCALE and IN_DIE (default) backends.
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
constexpr uint32_t RDMA_AGGREGATE_ELEMENT_COUNT = 16;
constexpr uint32_t RDMA_AGGREGATE_SLOT_COUNT = 34;

constexpr uint32_t RDMA_AGGREGATE_GET_DESTINATION_LAST_SLOT = 5;
constexpr uint32_t RDMA_AGGREGATE_PUT_DESTINATION_FIRST_SLOT = 9;
constexpr uint32_t RDMA_AGGREGATE_RESERVED_SLOT = 7;

uint32_t rdma_aggregate_expected_value(uint32_t pe, uint32_t slot, uint32_t element)
{
    return (pe + 1) * 100000U + slot * 1000U + element;
}
#endif
} // namespace

static void test_rdma_put_get(aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size)
{
    size_t messageSize = 64;
    uint32_t rankOffset = 10;
    uint32_t* inHost;
    uint32_t* outHost;
    size_t totalSize = messageSize * rank_size;
    uint32_t block_dim = 1;
    aclshmem_handle_t handle;
    handle.team_id = ACLSHMEM_TEAM_WORLD;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    bzero(inHost, totalSize);
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        inHost[i + rank_id * messageSize / sizeof(uint32_t)] = rank_id + rankOffset;
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_put_low_level(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, TIMEOUT), 0);
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_get_low_level(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, TIMEOUT), 0);
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_put_high_level(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, TIMEOUT), 0);
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_get_high_level(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, TIMEOUT), 0);
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

void test_aclshmem_rdma_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    void* ptr = aclshmem_malloc(1024);
    test_rdma_put_get(stream, (uint8_t*)ptr, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

void test_aclshmem_rdma_mem_with_different_va_base(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    void* va_gap = nullptr;
    const size_t va_gap_size = RDMA_TEST_VA_GAP_GRANULARITY * (static_cast<uint32_t>(rank_id) + 1U);
    aclrtStream stream;
    ASSERT_EQ(
        test_rdma_init_with_va_gap(rank_id, n_ranks, local_mem_size, va_gap_size, &stream, &va_gap), ACLSHMEM_SUCCESS);
    ASSERT_NE(stream, nullptr);
    void* ptr = aclshmem_malloc(1024);
    ASSERT_NE(ptr, nullptr);
    test_rdma_put_get(stream, static_cast<uint8_t*>(ptr), rank_id, n_ranks);
    aclshmem_free(ptr);
    EXPECT_EQ(aclrtReleaseMemAddress(va_gap), ACL_SUCCESS);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemRDMAMem)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_rdma_mem, local_mem_size, processCount);
}

TEST(TestMemApi, TestShmemRDMAMemWithDifferentVirtualBases)
{
    if (test_gnpu_num < 2) {
        GTEST_SKIP() << "RDMA different-base test requires at least two ranks/devices";
    }
    test_mutil_task(test_aclshmem_rdma_mem_with_different_va_base, 1024UL * 1024UL * 1024UL, test_gnpu_num);
}

namespace {
struct RdmaQueueWatermarks {
    uint32_t sq_head{0};
    uint32_t scq_tail{0};
};

bool ReadRdmaQueueWatermarks(uint32_t peer, aclrtStream stream, RdmaQueueWatermarks& watermarks)
{
    if (init_manager == nullptr) {
        return false;
    }
    const uint32_t rank_count = g_state.npes;
    const uint64_t qp_info_address = reinterpret_cast<uint64_t>(g_state.qp_info);
    if (qp_info_address == 0 || peer >= rank_count) {
        return false;
    }
    void* device_entry = nullptr;
    if (aclrtMalloc(&device_entry, sizeof(aclshmemi_rdma_exception_report_entry_t), ACL_MEM_MALLOC_NORMAL_ONLY) !=
        ACL_SUCCESS) {
        return false;
    }

    auto read_entry = [&](uint32_t type, uint64_t address, size_t size,
                          aclshmemi_rdma_exception_report_entry_t& entry) {
        entry = {};
        if (aclrtMemset(device_entry, sizeof(entry), 0, sizeof(entry)) != ACL_SUCCESS ||
            aclshmemi_rdma_exception_report_read_entry_on_stream(
                type, address, size, static_cast<aclshmemi_rdma_exception_report_entry_t*>(device_entry), stream) !=
                ACLSHMEM_SUCCESS ||
            aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
            aclrtMemcpy(&entry, sizeof(entry), device_entry, sizeof(entry), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return false;
        }
        return entry.ret == ACLSHMEMI_RDMA_EXCEPTION_REPORT_SUCCESS && entry.entry_type == type;
    };

    aclshmemi_rdma_exception_report_entry_t entry{};
    shm::AiQpRMAQueueInfo info{};
    bool success = read_entry(ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_RAW, qp_info_address, sizeof(info), entry);
    if (success) {
        std::memcpy(&info, entry.raw.data, sizeof(info));
        success = info.count != 0 && info.sq != nullptr && info.scq != nullptr;
    }
    uint64_t sq_addr = 0;
    uint64_t scq_addr = 0;
    if (success) {
        const size_t index = static_cast<size_t>(peer) * info.count;
        sq_addr = reinterpret_cast<uint64_t>(info.sq + index);
        scq_addr = reinterpret_cast<uint64_t>(info.scq + index);
    }
    if (success) {
        success = read_entry(ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_WQ, sq_addr, sizeof(shm::AiQpRMAWQ), entry);
    }
    const uint64_t sq_head_addr = entry.wq.head_addr;
    if (success) {
        success = read_entry(ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQ, scq_addr, sizeof(shm::AiQpRMACQ), entry);
    }
    const uint64_t scq_tail_addr = entry.cq.tail_addr;
    if (success) {
        success =
            sq_head_addr != 0 && scq_tail_addr != 0 &&
            read_entry(ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_RAW, sq_head_addr, sizeof(watermarks.sq_head), entry);
    }
    if (success) {
        std::memcpy(&watermarks.sq_head, entry.raw.data, sizeof(watermarks.sq_head));
        success =
            read_entry(ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_RAW, scq_tail_addr, sizeof(watermarks.scq_tail), entry);
    }
    if (success) {
        std::memcpy(&watermarks.scq_tail, entry.raw.data, sizeof(watermarks.scq_tail));
    }
    const bool free_success = aclrtFree(device_entry) == ACL_SUCCESS;
    return success && free_success;
}

} // namespace

void test_aclshmem_rdma_exception_report(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    ASSERT_EQ(aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG), ACLSHMEM_SUCCESS);

    aclrtStream stream = nullptr;
    ASSERT_EQ(test_rdma_init(rank_id, n_ranks, local_mem_size, &stream), ACLSHMEM_SUCCESS);
    ASSERT_NE(stream, nullptr);

    const uint32_t peer = static_cast<uint32_t>((rank_id + 1) % n_ranks);
    RdmaQueueWatermarks before{};
    RdmaQueueWatermarks after{};
    ASSERT_TRUE(ReadRdmaQueueWatermarks(peer, stream, before));
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    ASSERT_TRUE(aclshmemi_exception_report_record_snapshot(1U, 2U, 3U, static_cast<uint32_t>(device_id), 4U));
    ASSERT_TRUE(aclshmemi_exception_report_pending());
    ASSERT_EQ(aclshmemx_report_exception(), ACLSHMEM_SUCCESS);
    ASSERT_FALSE(aclshmemi_exception_report_pending());
    ASSERT_TRUE(ReadRdmaQueueWatermarks(peer, stream, after));
    EXPECT_EQ(after.sq_head, before.sq_head);
    EXPECT_EQ(after.scq_tail, before.scq_tail);

    ASSERT_EQ(aclshmemx_report_exception(), ACLSHMEM_SUCCESS);
    RdmaQueueWatermarks after_repeat{};
    ASSERT_TRUE(ReadRdmaQueueWatermarks(peer, stream, after_repeat));
    EXPECT_EQ(after_repeat.sq_head, before.sq_head);
    EXPECT_EQ(after_repeat.scq_tail, before.scq_tail);

    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemRDMAExceptionReport)
{
    if (test_gnpu_num < 2) {
        GTEST_SKIP() << "RDMA exception report integration test requires at least 2 PEs";
    }
    constexpr uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_rdma_exception_report, local_mem_size, test_gnpu_num);
}

// Aggregate tests are available on the XSCALE and IN_DIE (default) backends.
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)

static void test_rdma_aggregate_mem_func(aclrtStream stream, uint8_t* gva, uint32_t pe_id, uint32_t pe_size)
{
    ASSERT_GE(pe_size, 2U);

    constexpr uint32_t block_dim = 1;
    const size_t total_elements = RDMA_AGGREGATE_SLOT_COUNT * RDMA_AGGREGATE_ELEMENT_COUNT;
    const size_t total_size = total_elements * sizeof(uint32_t);
    std::vector<uint32_t> in_host(total_elements);
    std::vector<uint32_t> out_host(total_elements);

    for (uint32_t slot = 0; slot < RDMA_AGGREGATE_SLOT_COUNT; ++slot) {
        for (uint32_t element = 0; element < RDMA_AGGREGATE_ELEMENT_COUNT; ++element) {
            in_host[slot * RDMA_AGGREGATE_ELEMENT_COUNT + element] =
                rdma_aggregate_expected_value(pe_id, slot, element);
        }
    }

    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host.data(), total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    test_rdma_aggregate(block_dim, stream, gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, TIMEOUT), 0);

    ASSERT_EQ(aclrtMemcpy(out_host.data(), total_size, gva, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    const uint32_t next_peer = (pe_id + 1) % pe_size;
    const uint32_t previous_peer = (pe_id + pe_size - 1) % pe_size;
    for (uint32_t slot = 0; slot < RDMA_AGGREGATE_SLOT_COUNT; ++slot) {
        const bool is_destination = (slot % 2) != 0 && slot != RDMA_AGGREGATE_RESERVED_SLOT;
        const bool is_get_destination = is_destination && slot <= RDMA_AGGREGATE_GET_DESTINATION_LAST_SLOT;
        const uint32_t expected_pe =
            is_get_destination ?
                next_peer :
                (is_destination && slot >= RDMA_AGGREGATE_PUT_DESTINATION_FIRST_SLOT ? previous_peer : pe_id);
        const uint32_t expected_source_slot = is_destination ? slot - 1 : slot;

        for (uint32_t element = 0; element < RDMA_AGGREGATE_ELEMENT_COUNT; ++element) {
            const uint32_t actual = out_host[slot * RDMA_AGGREGATE_ELEMENT_COUNT + element];
            const uint32_t expected = rdma_aggregate_expected_value(expected_pe, expected_source_slot, element);
            ASSERT_EQ(actual, expected) << "slot=" << slot << ", element=" << element << ", pe=" << pe_id;
        }
    }
}

void test_aclshmem_rdma_aggregate(int pe_id, int n_pes, uint64_t local_mem_size)
{
    if (n_pes < 2) {
        return;
    }

    const int32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    const auto status = test_rdma_init(pe_id, n_pes, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    const size_t total_size =
        static_cast<size_t>(RDMA_AGGREGATE_SLOT_COUNT) * RDMA_AGGREGATE_ELEMENT_COUNT * sizeof(uint32_t);
    void* ptr = aclshmem_malloc(total_size);
    ASSERT_NE(ptr, nullptr);

    test_rdma_aggregate_mem_func(
        stream, static_cast<uint8_t*>(ptr), static_cast<uint32_t>(pe_id), static_cast<uint32_t>(n_pes));
    std::cout << "[TEST] RDMA aggregate pointer/tensor defer+submit, including loop_defer pointer put and "
              << "action reuse, finished, pe_id: " << pe_id << std::endl;

    aclshmem_free(ptr);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemRDMAAggregateNbi)
{
    const int processCount = test_gnpu_num;
    if (processCount < 2) {
        GTEST_SKIP() << "RDMA aggregate UT requires at least 2 PEs";
    }
    const uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_rdma_aggregate, local_mem_size, processCount);
}

#endif // !ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
