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
#include <unistd.h>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "sync_barrier_api_kernel.h"
#include "unittest_main_test.h"

namespace {
constexpr int32_t kSyncBarrierApiTestIterations = 3;
constexpr int32_t kSyncBarrierMessageLength = 256 * 1024;
constexpr unsigned int kSyncBarrierTimeoutSeconds = 60;

constexpr uint32_t kSyncBarrierEngineMte = static_cast<uint32_t>(ACLSHMEM_DATA_OP_MTE);
constexpr uint32_t kSyncBarrierEngineUdmaRdma =
    static_cast<uint32_t>(ACLSHMEM_DATA_OP_UDMA) | static_cast<uint32_t>(ACLSHMEM_DATA_OP_ROCE);
constexpr uint32_t kSyncBarrierEngineSdmaRdma =
    static_cast<uint32_t>(ACLSHMEM_DATA_OP_SDMA) | static_cast<uint32_t>(ACLSHMEM_DATA_OP_ROCE);

int32_t test_engine_init(
    uint32_t engine, int32_t rank_id, int32_t rank_size, uint64_t local_mem_size, aclrtStream* stream)
{
    *stream = nullptr;
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclrtSetDevice(device_id), ACL_SUCCESS);
    aclrtStream st = nullptr;
    EXPECT_EQ(aclrtCreateStream(&st), ACL_SUCCESS);
    EXPECT_EQ(aclshmemx_set_conf_store_tls(false, nullptr, 0), ACL_SUCCESS);

    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, rank_size, local_mem_size, test_global_ipport, &attributes);
    attributes.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(engine);
    const int32_t status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACL_SUCCESS);
    *stream = st;
    return status;
}

void fill_segment_to_device(uint8_t* addr_dev, int32_t offset, int32_t message_length, uint64_t pattern)
{
    uint8_t* host_buf = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_buf), message_length), ACL_SUCCESS);
    auto* data = reinterpret_cast<uint64_t*>(host_buf);
    const int32_t num_elements = message_length / static_cast<int32_t>(sizeof(uint64_t));
    for (int32_t i = 0; i < num_elements; ++i) {
        data[i] = pattern;
    }
    ASSERT_EQ(
        aclrtMemcpy(addr_dev + offset, message_length, host_buf, message_length, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(host_buf), ACL_SUCCESS);
}

void prepare_test_buffer(uint8_t* addr_dev, size_t total_size, int32_t rank_id)
{
    // Finish verification on every PE before any PE resets its buffer.
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), ACL_SUCCESS);
    fill_segment_to_device(
        addr_dev, rank_id * kSyncBarrierMessageLength, kSyncBarrierMessageLength, static_cast<uint64_t>(rank_id + 1));
    // Keep remote puts from racing with another PE's initialization.
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
}

void verify_segment_from_device(uint8_t* addr_dev, int32_t offset, int32_t message_length, uint64_t expected)
{
    uint8_t* host_buf = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_buf), message_length), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(host_buf, message_length, addr_dev + offset, message_length, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    auto* data = reinterpret_cast<uint64_t*>(host_buf);
    const int32_t num_elements = message_length / static_cast<int32_t>(sizeof(uint64_t));
    for (int32_t i = 0; i < num_elements; ++i) {
        ASSERT_EQ(data[i], expected);
    }
    ASSERT_EQ(aclrtFreeHost(host_buf), ACL_SUCCESS);
}

void verify_world_segments(uint8_t* addr_dev, int32_t rank_size)
{
    for (int32_t peer = 0; peer < rank_size; ++peer) {
        verify_segment_from_device(
            addr_dev, peer * kSyncBarrierMessageLength, kSyncBarrierMessageLength, static_cast<uint64_t>(peer + 1));
    }
}

bool is_team_member(int32_t rank_id, int32_t team_start, int32_t team_stride, int32_t team_size)
{
    if (team_stride <= 0 || team_size <= 0) {
        return false;
    }
    const int32_t index = rank_id - team_start;
    return index >= 0 && (index % team_stride == 0) && (index / team_stride < team_size);
}

void verify_team_segments(
    uint8_t* addr_dev, int32_t rank_size, int32_t team_start, int32_t team_stride, int32_t team_size)
{
    for (int32_t i = 0; i < team_size; ++i) {
        const int32_t peer = team_start + i * team_stride;
        if (peer >= rank_size) {
            break;
        }
        verify_segment_from_device(
            addr_dev, peer * kSyncBarrierMessageLength, kSyncBarrierMessageLength, static_cast<uint64_t>(peer + 1));
    }
}

void issue_world_puts(uint8_t* addr_dev, int32_t rank_id, int32_t rank_size, bool nonblocking)
{
    for (int32_t peer = 0; peer < rank_size; ++peer) {
        if (peer == rank_id) {
            continue;
        }
        if (nonblocking) {
            aclshmem_putmem_nbi(
                addr_dev + rank_id * kSyncBarrierMessageLength, addr_dev + rank_id * kSyncBarrierMessageLength,
                kSyncBarrierMessageLength, peer);
        } else {
            aclshmem_putmem(
                addr_dev + rank_id * kSyncBarrierMessageLength, addr_dev + rank_id * kSyncBarrierMessageLength,
                kSyncBarrierMessageLength, peer);
        }
    }
}

void issue_team_puts(
    uint8_t* addr_dev, int32_t rank_id, int32_t rank_size, bool nonblocking, int32_t team_start, int32_t team_stride,
    int32_t team_size)
{
    if (!is_team_member(rank_id, team_start, team_stride, team_size)) {
        return;
    }
    for (int32_t i = 0; i < team_size; ++i) {
        const int32_t peer = team_start + i * team_stride;
        if (peer >= rank_size || peer == rank_id) {
            continue;
        }
        if (nonblocking) {
            aclshmem_putmem_nbi(
                addr_dev + rank_id * kSyncBarrierMessageLength, addr_dev + rank_id * kSyncBarrierMessageLength,
                kSyncBarrierMessageLength, peer);
        } else {
            aclshmem_putmem(
                addr_dev + rank_id * kSyncBarrierMessageLength, addr_dev + rank_id * kSyncBarrierMessageLength,
                kSyncBarrierMessageLength, peer);
        }
    }
}

void issue_world_puts_on_stream(uint8_t* addr_dev, int32_t rank_id, int32_t rank_size, aclrtStream stream)
{
    for (int32_t peer = 0; peer < rank_size; ++peer) {
        if (peer == rank_id) {
            continue;
        }
        aclshmemx_putmem_on_stream(
            addr_dev + rank_id * kSyncBarrierMessageLength, addr_dev + rank_id * kSyncBarrierMessageLength,
            kSyncBarrierMessageLength, peer, stream);
    }
}

void issue_team_puts_on_stream(
    uint8_t* addr_dev, int32_t rank_id, int32_t rank_size, aclrtStream stream, int32_t team_start, int32_t team_stride,
    int32_t team_size)
{
    if (!is_team_member(rank_id, team_start, team_stride, team_size)) {
        return;
    }
    for (int32_t i = 0; i < team_size; ++i) {
        const int32_t peer = team_start + i * team_stride;
        if (peer >= rank_size || peer == rank_id) {
            continue;
        }
        aclshmemx_putmem_on_stream(
            addr_dev + rank_id * kSyncBarrierMessageLength, addr_dev + rank_id * kSyncBarrierMessageLength,
            kSyncBarrierMessageLength, peer, stream);
    }
}

void test_device_sync_barrier_world_api(uint32_t engine, int32_t rank_id, int32_t rank_size, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_engine_init(engine, rank_id, rank_size, local_mem_size, &stream) != ACLSHMEM_SUCCESS) {
        return;
    }
    ASSERT_NE(stream, nullptr);
    alarm(kSyncBarrierTimeoutSeconds);

    const size_t total_size = static_cast<size_t>(rank_size) * kSyncBarrierMessageLength;
    auto* addr_dev = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);

    for (int32_t iteration = 0; iteration < kSyncBarrierApiTestIterations; ++iteration) {
        prepare_test_buffer(addr_dev, total_size, rank_id);
        device_sync_all_api_do(stream, util_get_ffts_config(), addr_dev, rank_id, rank_size, kSyncBarrierMessageLength);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        verify_world_segments(addr_dev, rank_size);

        prepare_test_buffer(addr_dev, total_size, rank_id);
        device_barrier_all_api_do(
            stream, util_get_ffts_config(), addr_dev, rank_id, rank_size, kSyncBarrierMessageLength);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        verify_world_segments(addr_dev, rank_size);
    }

    aclshmem_free(addr_dev);
    test_finalize(stream, device_id);
    alarm(0);
}

void test_device_sync_barrier_team_api(
    uint32_t engine, int32_t rank_id, int32_t rank_size, uint64_t local_mem_size, int32_t team_pe_start,
    int32_t team_pe_stride, int32_t team_pe_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_engine_init(engine, rank_id, rank_size, local_mem_size, &stream) != ACLSHMEM_SUCCESS) {
        return;
    }
    ASSERT_NE(stream, nullptr);
    alarm(kSyncBarrierTimeoutSeconds);

    aclshmem_team_t team = ACLSHMEM_TEAM_INVALID;
    ASSERT_EQ(
        aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, team_pe_start, team_pe_stride, team_pe_size, &team),
        ACLSHMEM_SUCCESS);
    const bool is_member = is_team_member(rank_id, team_pe_start, team_pe_stride, team_pe_size);
    if (is_member) {
        ASSERT_NE(team, ACLSHMEM_TEAM_INVALID);
    } else {
        ASSERT_EQ(team, ACLSHMEM_TEAM_INVALID);
    }

    const size_t total_size = static_cast<size_t>(rank_size) * kSyncBarrierMessageLength;
    auto* addr_dev = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);

    for (int32_t iteration = 0; iteration < kSyncBarrierApiTestIterations; ++iteration) {
        prepare_test_buffer(addr_dev, total_size, rank_id);
        device_sync_team_api_do(
            stream, util_get_ffts_config(), addr_dev, rank_id, rank_size, kSyncBarrierMessageLength, team,
            team_pe_start, team_pe_stride, team_pe_size);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        if (is_member) {
            verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
        }

        prepare_test_buffer(addr_dev, total_size, rank_id);
        device_barrier_team_api_do(
            stream, util_get_ffts_config(), addr_dev, rank_id, rank_size, kSyncBarrierMessageLength, team,
            team_pe_start, team_pe_stride, team_pe_size);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        if (is_member) {
            verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
        }
    }

    aclshmem_free(addr_dev);
    aclshmem_team_destroy(team);
    test_finalize(stream, device_id);
    alarm(0);
}

void test_host_sync_barrier_world_api(uint32_t engine, int32_t rank_id, int32_t rank_size, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_engine_init(engine, rank_id, rank_size, local_mem_size, &stream) != ACLSHMEM_SUCCESS) {
        return;
    }
    ASSERT_NE(stream, nullptr);
    alarm(kSyncBarrierTimeoutSeconds);

    // Only ACLSHMEM_TEAM_INVALID is the documented non-member no-op sentinel.
    ASSERT_EQ(aclshmem_barrier(ACLSHMEM_TEAM_INVALID), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclshmem_team_sync(ACLSHMEM_TEAM_INVALID), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_INVALID, stream), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclshmemx_sync_on_stream(ACLSHMEM_TEAM_INVALID, stream), ACLSHMEM_SUCCESS);

    const size_t total_size = static_cast<size_t>(rank_size) * kSyncBarrierMessageLength;
    auto* addr_dev = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts(addr_dev, rank_id, rank_size, false);
    aclshmem_quiet();
    aclshmem_sync_all();
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts(addr_dev, rank_id, rank_size, false);
    aclshmem_quiet();
    aclshmem_sync(ACLSHMEM_TEAM_WORLD);
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts(addr_dev, rank_id, rank_size, true);
    ASSERT_EQ(aclshmem_barrier(ACLSHMEM_TEAM_WORLD), ACLSHMEM_SUCCESS);
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts(addr_dev, rank_id, rank_size, true);
    aclshmem_barrier_all();
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts_on_stream(addr_dev, rank_id, rank_size, stream);
    aclshmemx_quiet_on_stream(stream);
    ASSERT_EQ(aclshmemx_sync_on_stream(ACLSHMEM_TEAM_WORLD, stream), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts_on_stream(addr_dev, rank_id, rank_size, stream);
    aclshmemx_quiet_on_stream(stream);
    aclshmemx_sync_all_on_stream(stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts_on_stream(addr_dev, rank_id, rank_size, stream);
    ASSERT_EQ(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts_on_stream(addr_dev, rank_id, rank_size, stream);
    aclshmemx_barrier_all_on_stream(stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    verify_world_segments(addr_dev, rank_size);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts(addr_dev, rank_id, rank_size, false);
    aclshmem_quiet();
    ASSERT_EQ(aclshmem_team_sync(ACLSHMEM_TEAM_WORLD), ACLSHMEM_SUCCESS);
    verify_world_segments(addr_dev, rank_size);

    aclrtStream producer_stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&producer_stream), ACL_SUCCESS);
    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_world_puts_on_stream(addr_dev, rank_id, rank_size, producer_stream);
    ASSERT_EQ(aclrtSynchronizeStream(producer_stream), ACL_SUCCESS);
    ASSERT_EQ(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    verify_world_segments(addr_dev, rank_size);
    ASSERT_EQ(aclrtDestroyStream(producer_stream), ACL_SUCCESS);

    aclshmem_free(addr_dev);
    test_finalize(stream, device_id);
    alarm(0);
}

void test_host_sync_barrier_team_api(
    uint32_t engine, int32_t rank_id, int32_t rank_size, uint64_t local_mem_size, int32_t team_pe_start,
    int32_t team_pe_stride, int32_t team_pe_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_engine_init(engine, rank_id, rank_size, local_mem_size, &stream) != ACLSHMEM_SUCCESS) {
        return;
    }
    ASSERT_NE(stream, nullptr);
    alarm(kSyncBarrierTimeoutSeconds);

    aclshmem_team_t team = ACLSHMEM_TEAM_INVALID;
    ASSERT_EQ(
        aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, team_pe_start, team_pe_stride, team_pe_size, &team),
        ACLSHMEM_SUCCESS);
    const bool is_member = is_team_member(rank_id, team_pe_start, team_pe_stride, team_pe_size);
    if (is_member) {
        ASSERT_NE(team, ACLSHMEM_TEAM_INVALID);
    } else {
        ASSERT_EQ(team, ACLSHMEM_TEAM_INVALID);
    }

    const size_t total_size = static_cast<size_t>(rank_size) * kSyncBarrierMessageLength;
    auto* addr_dev = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_team_puts(addr_dev, rank_id, rank_size, false, team_pe_start, team_pe_stride, team_pe_size);
    aclshmem_sync(team);
    if (is_member) {
        verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
    }

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_team_puts(addr_dev, rank_id, rank_size, false, team_pe_start, team_pe_stride, team_pe_size);
    ASSERT_EQ(aclshmem_team_sync(team), ACLSHMEM_SUCCESS);
    if (is_member) {
        verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
    }

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_team_puts(addr_dev, rank_id, rank_size, true, team_pe_start, team_pe_stride, team_pe_size);
    aclshmem_quiet();
    ASSERT_EQ(aclshmem_team_sync(team), ACLSHMEM_SUCCESS);
    if (is_member) {
        verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
    }

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_team_puts(addr_dev, rank_id, rank_size, true, team_pe_start, team_pe_stride, team_pe_size);
    ASSERT_EQ(aclshmem_barrier(team), ACLSHMEM_SUCCESS);
    if (is_member) {
        verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
    }

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_team_puts_on_stream(addr_dev, rank_id, rank_size, stream, team_pe_start, team_pe_stride, team_pe_size);
    aclshmemx_quiet_on_stream(stream);
    ASSERT_EQ(aclshmemx_sync_on_stream(team, stream), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    if (is_member) {
        verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
    }

    prepare_test_buffer(addr_dev, total_size, rank_id);
    issue_team_puts_on_stream(addr_dev, rank_id, rank_size, stream, team_pe_start, team_pe_stride, team_pe_size);
    ASSERT_EQ(aclshmemx_barrier_on_stream(team, stream), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    if (is_member) {
        verify_team_segments(addr_dev, rank_size, team_pe_start, team_pe_stride, team_pe_size);
    }

    aclshmem_free(addr_dev);
    aclshmem_team_destroy(team);
    test_finalize(stream, device_id);
    alarm(0);
}
} // namespace

// 场景1：单机，仅 MTE，world sync/barrier 精度验证。
TEST(TEST_SYNC_BARRIER_PUBLIC_API, device_world_mte_api)
{
    if (test_gnpu_num < test_global_ranks) {
        GTEST_SKIP() << "single-machine scenario requires all ranks on one machine";
    }
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [](int32_t rank_id, int32_t rank_size, uint64_t size) {
            test_device_sync_barrier_world_api(kSyncBarrierEngineMte, rank_id, rank_size, size);
        },
        local_mem_size, test_gnpu_num);
}

TEST(TEST_SYNC_BARRIER_PUBLIC_API, host_world_mte_api)
{
    if (test_gnpu_num < test_global_ranks) {
        GTEST_SKIP() << "single-machine scenario requires all ranks on one machine";
    }
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [](int32_t rank_id, int32_t rank_size, uint64_t size) {
            test_host_sync_barrier_world_api(kSyncBarrierEngineMte, rank_id, rank_size, size);
        },
        local_mem_size, test_gnpu_num);
}

// 场景2：双机，Ascend950 走 UDMA+RDMA，其余平台走 SDMA+RDMA，world sync/barrier 精度验证。
TEST(TEST_SYNC_BARRIER_PUBLIC_API, device_world_dual_machine_api)
{
#if !defined(ACLSHMEM_UT_RDMA_ENABLED)
    GTEST_SKIP() << "dual-machine scenario requires an RDMA-enabled build";
#else
    if (test_gnpu_num >= test_global_ranks) {
        GTEST_SKIP() << "dual-machine scenario requires ranks on multiple machines";
    }
#if defined(ACLSHMEM_UT_ASCEND950)
    constexpr uint32_t engine = kSyncBarrierEngineUdmaRdma;
#else
    constexpr uint32_t engine = kSyncBarrierEngineSdmaRdma;
#endif
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [engine](int32_t rank_id, int32_t rank_size, uint64_t size) {
            test_device_sync_barrier_world_api(engine, rank_id, rank_size, size);
        },
        local_mem_size, test_gnpu_num);
#endif
}

TEST(TEST_SYNC_BARRIER_PUBLIC_API, host_world_dual_machine_api)
{
#if !defined(ACLSHMEM_UT_RDMA_ENABLED)
    GTEST_SKIP() << "dual-machine scenario requires an RDMA-enabled build";
#else
    if (test_gnpu_num >= test_global_ranks) {
        GTEST_SKIP() << "dual-machine scenario requires ranks on multiple machines";
    }
#if defined(ACLSHMEM_UT_ASCEND950)
    constexpr uint32_t engine = kSyncBarrierEngineUdmaRdma;
#else
    constexpr uint32_t engine = kSyncBarrierEngineSdmaRdma;
#endif
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [engine](int32_t rank_id, int32_t rank_size, uint64_t size) {
            test_host_sync_barrier_world_api(engine, rank_id, rank_size, size);
        },
        local_mem_size, test_gnpu_num);
#endif
}

// 场景3：双机子team，子team = 单机内所有PE（如 2机16卡时子team为其中一机的8卡），sync/barrier 精度验证。
TEST(TEST_SYNC_BARRIER_PUBLIC_API, device_team_dual_machine_api)
{
#if !defined(ACLSHMEM_UT_RDMA_ENABLED)
    GTEST_SKIP() << "dual-machine team scenario requires an RDMA-enabled build";
#else
    if (test_gnpu_num >= test_global_ranks) {
        GTEST_SKIP() << "dual-machine scenario requires ranks on multiple machines";
    }
    if (test_gnpu_num < 2) {
        GTEST_SKIP() << "intra-machine team scenario needs at least 2 ranks per machine";
    }
#if defined(ACLSHMEM_UT_ASCEND950)
    constexpr uint32_t engine = kSyncBarrierEngineUdmaRdma;
#else
    constexpr uint32_t engine = kSyncBarrierEngineSdmaRdma;
#endif
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [engine](int32_t rank_id, int32_t rank_size, uint64_t size) {
            const int32_t machine_size = test_gnpu_num;
            const int32_t machine_start = (rank_id / machine_size) * machine_size;
            test_device_sync_barrier_team_api(engine, rank_id, rank_size, size, machine_start, 1, machine_size);
        },
        local_mem_size, test_gnpu_num);
#endif
}

TEST(TEST_SYNC_BARRIER_PUBLIC_API, host_team_dual_machine_api)
{
#if !defined(ACLSHMEM_UT_RDMA_ENABLED)
    GTEST_SKIP() << "dual-machine team scenario requires an RDMA-enabled build";
#else
    if (test_gnpu_num >= test_global_ranks) {
        GTEST_SKIP() << "dual-machine scenario requires ranks on multiple machines";
    }
    if (test_gnpu_num < 2) {
        GTEST_SKIP() << "intra-machine team scenario needs at least 2 ranks per machine";
    }
#if defined(ACLSHMEM_UT_ASCEND950)
    constexpr uint32_t engine = kSyncBarrierEngineUdmaRdma;
#else
    constexpr uint32_t engine = kSyncBarrierEngineSdmaRdma;
#endif
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [engine](int32_t rank_id, int32_t rank_size, uint64_t size) {
            const int32_t machine_size = test_gnpu_num;
            const int32_t machine_start = (rank_id / machine_size) * machine_size;
            test_host_sync_barrier_team_api(engine, rank_id, rank_size, size, machine_start, 1, machine_size);
        },
        local_mem_size, test_gnpu_num);
#endif
}

TEST(TEST_SYNC_BARRIER_PUBLIC_API, device_team_cross_machine_api)
{
#if !defined(ACLSHMEM_UT_RDMA_ENABLED)
    GTEST_SKIP() << "cross-machine team requires an RDMA-enabled build";
#else
    if (test_gnpu_num == 0 || test_global_ranks <= test_gnpu_num || test_global_ranks % test_gnpu_num != 0) {
        GTEST_SKIP() << "requires an equal number of PEs per machine";
    }
#if defined(ACLSHMEM_UT_ASCEND950)
    constexpr uint32_t engine = kSyncBarrierEngineUdmaRdma;
#else
    constexpr uint32_t engine = kSyncBarrierEngineSdmaRdma;
#endif
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [engine](int32_t rank_id, int32_t rank_size, uint64_t size) {
            // The first PE on each machine is a team member; all other PEs are non-members.
            test_device_sync_barrier_team_api(
                engine, rank_id, rank_size, size, 0, test_gnpu_num, rank_size / test_gnpu_num);
        },
        local_mem_size, test_gnpu_num);
#endif
}

TEST(TEST_SYNC_BARRIER_PUBLIC_API, host_team_cross_machine_api)
{
#if !defined(ACLSHMEM_UT_RDMA_ENABLED)
    GTEST_SKIP() << "cross-machine team requires an RDMA-enabled build";
#else
    if (test_gnpu_num == 0 || test_global_ranks <= test_gnpu_num || test_global_ranks % test_gnpu_num != 0) {
        GTEST_SKIP() << "requires an equal number of PEs per machine";
    }
#if defined(ACLSHMEM_UT_ASCEND950)
    constexpr uint32_t engine = kSyncBarrierEngineUdmaRdma;
#else
    constexpr uint32_t engine = kSyncBarrierEngineSdmaRdma;
#endif
    constexpr uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    test_mutil_task(
        [engine](int32_t rank_id, int32_t rank_size, uint64_t size) {
            test_host_sync_barrier_team_api(
                engine, rank_id, rank_size, size, 0, test_gnpu_num, rank_size / test_gnpu_num);
        },
        local_mem_size, test_gnpu_num);
#endif
}
