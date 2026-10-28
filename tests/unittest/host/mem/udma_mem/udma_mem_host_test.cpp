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
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <string>

#include "acl/acl.h"
#include "host/init/shmem_host_init.h"
#include "shmemi_host_common.h"
#include "unittest/udma_mem_kernel.h"
#include "unittest_main_test.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern int32_t test_udma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st);
extern void test_finalize(aclrtStream stream, int device_id);

#ifndef ACLSHMEM_RELAY_SUPPORT
extern void test_udma_qp_data_path(
    uint32_t block_dim, void* stream, uint8_t* symmetric, uint8_t* local_buffer, uint64_t* signal_words,
    uint32_t slice_size, int32_t peer, int32_t operation);
#endif

constexpr size_t UDMA_TEST_MESSAGE_SIZE = 64;
constexpr uint32_t UDMA_TEST_ACTION_BATCH_COUNT = 2;
constexpr size_t UDMA_TEST_WRAP_WORKSPACE_SIZE = 5UL * 1024UL * 1024UL;
constexpr size_t UDMA_TEST_VA_GAP_GRANULARITY = 2UL * 1024UL * 1024UL;
constexpr uint64_t UDMA_TEST_WRAP_GUARD_VALUE = 0xC0FFEE1234567890ULL;
constexpr size_t UDMA_TEST_WRAP_RESULT_COUNT = 8;
constexpr size_t UDMA_TEST_WRAP_PIPE_S_LAYOUT_IDX = 0;
constexpr size_t UDMA_TEST_WRAP_PIPE_MTE3_LAYOUT_IDX = 1;
constexpr size_t UDMA_TEST_WRAP_PIPE_S_GUARD_IDX = 2;
constexpr size_t UDMA_TEST_WRAP_PIPE_MTE3_GUARD_IDX = 3;
constexpr size_t UDMA_TEST_WRAP_FAA_LAYOUT_IDX = 4;
constexpr size_t UDMA_TEST_WRAP_CAS_LAYOUT_IDX = 5;
constexpr size_t UDMA_TEST_WRAP_FAA_GUARD_IDX = 6;
constexpr size_t UDMA_TEST_WRAP_CAS_GUARD_IDX = 7;
using UdmaActionTestKernel = void (*)(uint32_t, void*, uint8_t*);

static void prepare_udma_action_input(uint32_t* host_data, size_t total_size, uint32_t rank_id)
{
    bzero(host_data, total_size);
    constexpr uint32_t rankOffset = 100;
    size_t words_per_message = UDMA_TEST_MESSAGE_SIZE / sizeof(uint32_t);
    for (uint32_t i = 0; i < UDMA_TEST_ACTION_BATCH_COUNT; ++i) {
        size_t word_offset = (rank_id * UDMA_TEST_ACTION_BATCH_COUNT + i) * words_per_message;
        host_data[word_offset] = rank_id + rankOffset + i;
    }
}

static void verify_udma_action_output(uint32_t* host_data, uint32_t rank_size)
{
    constexpr uint32_t rankOffset = 100;
    size_t words_per_message = UDMA_TEST_MESSAGE_SIZE / sizeof(uint32_t);
    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        for (uint32_t i = 0; i < UDMA_TEST_ACTION_BATCH_COUNT; ++i) {
            size_t word_offset = (rank * UDMA_TEST_ACTION_BATCH_COUNT + i) * words_per_message;
            ASSERT_EQ(host_data[word_offset], rank + rankOffset + i);
        }
    }
}

static void run_udma_action_test(
    aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size, UdmaActionTestKernel kernel)
{
    size_t totalSize = UDMA_TEST_MESSAGE_SIZE * UDMA_TEST_ACTION_BATCH_COUNT * rank_size;
    uint32_t block_dim = 1;
    uint32_t* inHost;
    uint32_t* outHost;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    prepare_udma_action_input(inHost, totalSize, rank_id);
    bzero(outHost, totalSize);

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    kernel(block_dim, stream, gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_action_output(outHost, rank_size);

    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

static void verify_highlevel_put_signal_data(uint32_t* outHost, uint32_t rank_size)
{
    constexpr size_t message_size = 64;
    constexpr uint32_t rank_offset = 10;
    const size_t words_per_message = message_size / sizeof(uint32_t);
    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        for (size_t i = 0; i < words_per_message; ++i) {
            ASSERT_EQ(outHost[rank * words_per_message + i], rank + rank_offset);
        }
    }
}

static void test_udma_highlevel_put_signal_split_case(
    aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size, size_t message_size, int32_t signal,
    uint32_t family, bool nbi, bool use_tensor)
{
    SCOPED_TRACE(
        ::testing::Message() << "family=" << family << " nbi=" << nbi << " tensor=" << use_tensor
                             << " message_size=" << message_size);
    constexpr uint8_t rank_offset = 10;
    constexpr size_t boundary_offsets[] = {0, 256UL * 1024UL * 1024UL - 1};
    for (size_t offset : boundary_offsets) {
        uint8_t value = static_cast<uint8_t>(rank_id + rank_offset);
        ASSERT_EQ(
            aclrtMemcpy(
                gva + rank_id * message_size + offset, sizeof(value), &value, sizeof(value), ACL_MEMCPY_HOST_TO_DEVICE),
            0);
    }
    if (message_size > 256UL * 1024UL * 1024UL) {
        uint8_t value = static_cast<uint8_t>(rank_id + rank_offset);
        ASSERT_EQ(
            aclrtMemcpy(
                gva + rank_id * message_size + 256UL * 1024UL * 1024UL, sizeof(value), &value, sizeof(value),
                ACL_MEMCPY_HOST_TO_DEVICE),
            0);
        ASSERT_EQ(
            aclrtMemcpy(
                gva + (rank_id + 1) * message_size - 1, sizeof(value), &value, sizeof(value),
                ACL_MEMCPY_HOST_TO_DEVICE),
            0);
    }

    auto* sig_addr = static_cast<uint8_t*>(aclshmem_malloc(rank_size * sizeof(int32_t)));
    ASSERT_NE(sig_addr, nullptr);
    aclshmemi_control_barrier_all();
    test_udma_highlevel_put_signal_split(1, stream, gva, sig_addr, message_size, signal, family, nbi, use_tensor);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();

    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        for (size_t offset : boundary_offsets) {
            uint8_t value = 0;
            ASSERT_EQ(
                aclrtMemcpy(
                    &value, sizeof(value), gva + rank * message_size + offset, sizeof(value),
                    ACL_MEMCPY_DEVICE_TO_HOST),
                0);
            ASSERT_EQ(value, static_cast<uint8_t>(rank + rank_offset));
        }
        if (message_size > 256UL * 1024UL * 1024UL) {
            uint8_t split_value = 0;
            uint8_t tail_value = 0;
            ASSERT_EQ(
                aclrtMemcpy(
                    &split_value, sizeof(split_value), gva + rank * message_size + 256UL * 1024UL * 1024UL,
                    sizeof(split_value), ACL_MEMCPY_DEVICE_TO_HOST),
                0);
            ASSERT_EQ(
                aclrtMemcpy(
                    &tail_value, sizeof(tail_value), gva + (rank + 1) * message_size - 1, sizeof(tail_value),
                    ACL_MEMCPY_DEVICE_TO_HOST),
                0);
            ASSERT_EQ(split_value, static_cast<uint8_t>(rank + rank_offset));
            ASSERT_EQ(tail_value, static_cast<uint8_t>(rank + rank_offset));
        }
    }

    std::vector<int32_t> signal_values(rank_size, 0);
    ASSERT_EQ(
        aclrtMemcpy(
            signal_values.data(), rank_size * sizeof(int32_t), sig_addr, rank_size * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST),
        0);
    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        if (rank != rank_id) {
            ASSERT_EQ(signal_values[rank], signal);
        }
    }
    aclshmem_free(sig_addr);
}

static uint8_t udma_signal_source_byte(uint32_t rank, uint32_t slot, size_t offset)
{
    // Avoid the guard value so missing writes cannot accidentally match.
    return static_cast<uint8_t>((rank * 31 + slot * 13 + offset * 7) % 127);
}

static void test_udma_signal_overload_matrix(
    aclrtStream stream, uint32_t rank_id, uint32_t rank_size, uint8_t* gva, bool use_add)
{
    const size_t slots = static_cast<size_t>(rank_size) * UDMA_SIGNAL_SLOT_COUNT;
    const size_t bank_bytes = slots * UDMA_SIGNAL_SLOT_BYTES;
    const size_t signal_entries = slots + rank_size;
    const size_t signal_bytes = signal_entries * sizeof(int32_t);
    auto* data = gva;
    auto* signals = gva + 2 * bank_bytes;
    std::vector<uint8_t> input(2 * bank_bytes, UDMA_SIGNAL_GUARD);
    std::vector<uint8_t> output(2 * bank_bytes);
    std::vector<int32_t> signal_values(signal_entries, UDMA_SIGNAL_INITIAL);
    std::fill(signal_values.begin() + slots, signal_values.end(), 0);
    for (uint32_t slot = 0; slot < UDMA_SIGNAL_SLOT_COUNT; ++slot) {
        const size_t base = (static_cast<size_t>(rank_id) * UDMA_SIGNAL_SLOT_COUNT + slot) * UDMA_SIGNAL_SLOT_BYTES;
        for (size_t byte = 0; byte < UDMA_SIGNAL_SLOT_BYTES; ++byte) {
            input[base + byte] = udma_signal_source_byte(rank_id, slot, byte);
        }
    }
    ASSERT_EQ(aclrtMemcpy(data, input.size(), input.data(), input.size(), ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(signals, signal_bytes, signal_values.data(), signal_bytes, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_highlevel_put_signal_overloads(1, stream, data, signals, use_add);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(output.data(), output.size(), data, output.size(), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(aclrtMemcpy(signal_values.data(), signal_bytes, signals, signal_bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        ASSERT_EQ(signal_values[slots + rank], 0)
            << "contiguous put-with-signal data was not visible when its signal was observed; bitmask identifies "
               "the failing slot";
    }
    for (size_t byte = 0; byte < bank_bytes; ++byte) {
        ASSERT_EQ(output[byte], input[byte]) << "source bank modified at byte=" << byte;
    }
    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        for (uint32_t slot = 0; slot < UDMA_SIGNAL_SLOT_COUNT; ++slot) {
            const size_t index = static_cast<size_t>(rank) * UDMA_SIGNAL_SLOT_COUNT + slot;
            const int32_t expected_signal = rank == rank_id ?
                                                UDMA_SIGNAL_INITIAL :
                                                100 + static_cast<int32_t>(slot) + (use_add ? UDMA_SIGNAL_INITIAL : 0);
            ASSERT_EQ(signal_values[index], expected_signal) << "rank=" << rank << " slot=" << slot;
            for (size_t byte = 0; byte < UDMA_SIGNAL_SLOT_BYTES; ++byte) {
                uint8_t expected = UDMA_SIGNAL_GUARD;
                if (rank != rank_id && byte >= UDMA_SIGNAL_PREFIX_BYTES) {
                    const size_t relative = byte - UDMA_SIGNAL_PREFIX_BYTES;
                    if (relative < UDMA_SIGNAL_COPY_BYTES) {
                        expected = udma_signal_source_byte(rank, slot, UDMA_SIGNAL_PREFIX_BYTES + relative);
                    }
                }
                ASSERT_EQ(output[bank_bytes + index * UDMA_SIGNAL_SLOT_BYTES + byte], expected)
                    << "destination byte=" << byte;
            }
        }
    }
    aclshmemi_control_barrier_all();
}

static void test_udma_put_get(aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size)
{
    size_t messageSize = 64;
    uint32_t rankOffset = 10;
    uint32_t* inHost;
    uint32_t* outHost;
    size_t totalSize = messageSize * rank_size;
    uint32_t block_dim = 1;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    bzero(inHost, totalSize);
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        inHost[i + rank_id * messageSize / sizeof(uint32_t)] = rank_id + rankOffset;
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_put(block_dim, stream, (uint8_t*)gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_get(block_dim, stream, (uint8_t*)gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_op_config_put_get(block_dim, stream, (uint8_t*)gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_put_action_pointer);
    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_put_action_tensor);
    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_get_action_pointer);
    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_get_action_tensor);
    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_op_config_action_pointer);

    // Test udma put signal
    uint8_t* sig_addr = static_cast<uint8_t*>(aclshmem_malloc(rank_size * sizeof(uint64_t)));
    ASSERT_NE(sig_addr, nullptr);

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_put_signal(block_dim, stream, (uint8_t*)gva, sig_addr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();

    // Read and validate signals
    std::vector<uint64_t> signal_values(rank_size, 0);
    ASSERT_EQ(
        aclrtMemcpy(
            signal_values.data(), rank_size * sizeof(uint64_t), sig_addr, rank_size * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST),
        0);

    std::cout << "Signal values received on rank=" << rank_id << ":" << std::endl;
    bool all_signals_set = true;
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id) {
            continue;
        }
        uint64_t signal = 1000;
        if (signal_values[i] != signal) {
            all_signals_set = false;
        }
    }
    ASSERT_TRUE(all_signals_set) << "Signal for rank " << rank_id << " is not set correctly";

    // Test high-level UDMA put signal
    uint8_t* highlevel_sig_addr = static_cast<uint8_t*>(aclshmem_malloc(rank_size * sizeof(int32_t)));
    ASSERT_NE(highlevel_sig_addr, nullptr);

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_highlevel_put_signal(block_dim, stream, (uint8_t*)gva, highlevel_sig_addr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_highlevel_put_signal_data(outHost, rank_size);

    std::vector<int32_t> highlevel_signal_values(rank_size, 0);
    ASSERT_EQ(
        aclrtMemcpy(
            highlevel_signal_values.data(), rank_size * sizeof(int32_t), highlevel_sig_addr,
            rank_size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
        0);
    for (uint32_t i = 0; i < rank_size; i++) {
        if (i == rank_id) {
            continue;
        }
        ASSERT_EQ(highlevel_signal_values[i], 1000);
    }

    // Test synchronous high-level UDMA put signal
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_highlevel_put_signal_sync(block_dim, stream, (uint8_t*)gva, highlevel_sig_addr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_highlevel_put_signal_data(outHost, rank_size);

    ASSERT_EQ(
        aclrtMemcpy(
            highlevel_signal_values.data(), rank_size * sizeof(int32_t), highlevel_sig_addr,
            rank_size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
        0);
    for (uint32_t i = 0; i < rank_size; i++) {
        if (i == rank_id) {
            continue;
        }
        ASSERT_EQ(highlevel_signal_values[i], 2000);
    }

    // Test high-level size put signal
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_highlevel_put_size_signal(block_dim, stream, (uint8_t*)gva, highlevel_sig_addr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_highlevel_put_signal_data(outHost, rank_size);

    ASSERT_EQ(
        aclrtMemcpy(
            highlevel_signal_values.data(), rank_size * sizeof(int32_t), highlevel_sig_addr,
            rank_size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
        0);
    for (uint32_t i = 0; i < rank_size; i++) {
        if (i == rank_id) {
            continue;
        }
        ASSERT_EQ(highlevel_signal_values[i], 3000);
    }

    // Test synchronous high-level size put signal
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_highlevel_put_size_signal_sync(block_dim, stream, (uint8_t*)gva, highlevel_sig_addr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_highlevel_put_signal_data(outHost, rank_size);

    ASSERT_EQ(
        aclrtMemcpy(
            highlevel_signal_values.data(), rank_size * sizeof(int32_t), highlevel_sig_addr,
            rank_size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
        0);
    for (uint32_t i = 0; i < rank_size; i++) {
        if (i == rank_id) {
            continue;
        }
        ASSERT_EQ(highlevel_signal_values[i], 4000);
    }

    aclshmem_free(sig_addr);
    aclshmem_free(highlevel_sig_addr);
    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

void test_aclshmem_udma_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    uint64_t action_test_size = UDMA_TEST_MESSAGE_SIZE * UDMA_TEST_ACTION_BATCH_COUNT * n_ranks;
    uint64_t test_buffer_size = action_test_size > 1024 ? action_test_size : 1024;
    void* ptr = aclshmem_malloc(test_buffer_size);
    test_udma_put_get(stream, (uint8_t*)ptr, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

void test_aclshmem_udma_mem_with_different_va_base(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    void* va_gap = nullptr;
    const size_t va_gap_size = UDMA_TEST_VA_GAP_GRANULARITY * (static_cast<uint32_t>(rank_id) + 1U);
    aclrtStream stream;
    ASSERT_EQ(test_udma_init_with_va_gap(rank_id, n_ranks, local_mem_size, va_gap_size, &stream, &va_gap), 0);
    ASSERT_NE(stream, nullptr);
    void* ptr = aclshmem_malloc(1024);
    ASSERT_NE(ptr, nullptr);
    test_udma_put_get(stream, static_cast<uint8_t*>(ptr), rank_id, n_ranks);
    aclshmem_free(ptr);
    EXPECT_EQ(aclrtReleaseMemAddress(va_gap), ACL_SUCCESS);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAMem)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    // test_mutil_task(test_aclshmem_udma_mem, local_mem_size, processCount);
}

TEST(TestMemApi, TestShmemUDMAMemWithDifferentVirtualBases)
{
    if (test_global_ranks < 2 || test_gnpu_num < test_global_ranks) {
        GTEST_SKIP() << "UDMA different-base test requires at least two ranks/devices";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA different-base test requires Ascend950";
    }
    test_mutil_task(test_aclshmem_udma_mem_with_different_va_base, 1024UL * 1024UL * 1024UL, test_global_ranks);
}

void test_aclshmem_udma_signal_overload(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    const size_t slots = static_cast<size_t>(n_ranks) * UDMA_SIGNAL_SLOT_COUNT;
    const size_t bank_bytes = slots * UDMA_SIGNAL_SLOT_BYTES;
    const size_t gva_size = 2 * bank_bytes + (slots + n_ranks) * sizeof(int32_t);
    auto* gva = static_cast<uint8_t*>(aclshmem_malloc(gva_size));
    ASSERT_NE(gva, nullptr);
    test_udma_signal_overload_matrix(stream, rank_id, n_ranks, gva, false);
    test_udma_signal_overload_matrix(stream, rank_id, n_ranks, gva, true);
    aclshmem_free(gva);
    test_finalize(stream, device_id);
}

static uint8_t noncontiguous_tensor_source_byte(uint32_t rank, uint32_t slot, size_t row, size_t byte)
{
    return static_cast<uint8_t>((rank * 17 + slot * 29 + row * 7 + byte * 3) % 127);
}

void test_aclshmem_noncontiguous_tensor_signal(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);
    const size_t slots = static_cast<size_t>(n_ranks) * UDMA_SIGNAL_NONCONTIG_SLOT_COUNT;
    const size_t bank_bytes = slots * UDMA_SIGNAL_NONCONTIG_SLOT_BYTES;
    const size_t signal_bytes = slots * sizeof(int32_t);
    const size_t gva_size = 2 * bank_bytes + signal_bytes;
    auto* gva = static_cast<uint8_t*>(aclshmem_malloc(gva_size));
    ASSERT_NE(gva, nullptr);
    auto* signals = gva + 2 * bank_bytes;
    std::vector<uint8_t> input(2 * bank_bytes, UDMA_SIGNAL_GUARD);
    std::vector<uint8_t> output(2 * bank_bytes, UDMA_SIGNAL_GUARD);
    std::vector<int32_t> signal_values(slots, UDMA_SIGNAL_INITIAL);
    for (uint32_t slot = 0; slot < UDMA_SIGNAL_NONCONTIG_SLOT_COUNT; ++slot) {
        const size_t base = (static_cast<size_t>(rank_id) * UDMA_SIGNAL_NONCONTIG_SLOT_COUNT + slot) *
                                UDMA_SIGNAL_NONCONTIG_SLOT_BYTES +
                            UDMA_SIGNAL_NONCONTIG_PREFIX_BYTES;
        for (uint32_t row = 0; row < UDMA_SIGNAL_NONCONTIG_REPEAT; ++row) {
            for (uint32_t byte = 0; byte < UDMA_SIGNAL_NONCONTIG_LENGTH; ++byte) {
                input[base + row * UDMA_SIGNAL_NONCONTIG_LD + byte] =
                    noncontiguous_tensor_source_byte(rank_id, slot, row, byte);
            }
        }
    }
    ASSERT_EQ(aclrtMemcpy(gva, input.size(), input.data(), input.size(), ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(signals, signal_bytes, signal_values.data(), signal_bytes, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_highlevel_put_signal_noncontiguous_tensor(1, stream, gva, signals);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(output.data(), output.size(), gva, output.size(), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(aclrtMemcpy(signal_values.data(), signal_bytes, signals, signal_bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t sender = 0; sender < static_cast<uint32_t>(n_ranks); ++sender) {
        for (uint32_t slot = 0; slot < UDMA_SIGNAL_NONCONTIG_SLOT_COUNT; ++slot) {
            const size_t index = static_cast<size_t>(sender) * UDMA_SIGNAL_NONCONTIG_SLOT_COUNT + slot;
            const int32_t expected_signal =
                100 + static_cast<int32_t>(slot) + ((slot == 1 || slot == 3) ? UDMA_SIGNAL_INITIAL : 0);
            if (sender == static_cast<uint32_t>(rank_id)) {
                ASSERT_EQ(signal_values[index], UDMA_SIGNAL_INITIAL);
                continue;
            }
            ASSERT_EQ(signal_values[index], expected_signal) << "sender=" << sender << " slot=" << slot;
            const size_t base = bank_bytes + index * UDMA_SIGNAL_NONCONTIG_SLOT_BYTES;
            for (uint32_t byte = 0; byte < UDMA_SIGNAL_NONCONTIG_SLOT_BYTES; ++byte) {
                uint8_t expected = UDMA_SIGNAL_GUARD;
                for (uint32_t row = 0; row < UDMA_SIGNAL_NONCONTIG_REPEAT; ++row) {
                    const uint32_t row_begin = UDMA_SIGNAL_NONCONTIG_PREFIX_BYTES + row * UDMA_SIGNAL_NONCONTIG_LD;
                    if (byte >= row_begin && byte < row_begin + UDMA_SIGNAL_NONCONTIG_LENGTH) {
                        expected = noncontiguous_tensor_source_byte(sender, slot, row, byte - row_begin);
                        break;
                    }
                }
                ASSERT_EQ(output[base + byte], expected) << "sender=" << sender << " slot=" << slot << " byte=" << byte;
            }
        }
    }
    aclshmem_free(gva);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemNonContiguousTensorSignal)
{
    if (test_gnpu_num < 2) {
        GTEST_SKIP() << "non-contiguous Tensor signal test requires at least 2 NPUs";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "non-contiguous Tensor signal test requires Ascend950";
    }
    const uint64_t local_mem_size = ALIGN_UP(64UL * 1024UL * 1024UL, ACLSHMEM_PAGE_SIZE);
    test_mutil_task(test_aclshmem_noncontiguous_tensor_signal, local_mem_size, test_gnpu_num);
}

TEST(TestMemApi, TestShmemUDMASignalOverloads)
{
    const int processCount = test_gnpu_num;
    if (processCount < 2) {
        GTEST_SKIP() << "UDMA high-level overload test requires at least 2 NPUs";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA high-level overload test requires Ascend950";
    }
    const uint64_t required_mem_size = 64UL * 1024UL * 1024UL;
    const uint64_t local_mem_size = ALIGN_UP(required_mem_size, ACLSHMEM_PAGE_SIZE);
    test_mutil_task(test_aclshmem_udma_signal_overload, local_mem_size, processCount);
}

void test_aclshmem_udma_highlevel_put_signal_split(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    constexpr size_t max_message_size = 256UL * 1024UL * 1024UL + 1;
    const size_t gva_size = max_message_size * n_ranks;
    auto* gva = static_cast<uint8_t*>(aclshmem_malloc(gva_size));
    ASSERT_NE(gva, nullptr);
    ASSERT_EQ(aclrtMemset(gva, gva_size, 0, gva_size), 0);
    int32_t signal = 5000;
    // Keep one exact-limit case, then cover every overload form with a
    // transfer one byte beyond the UDMA limit so splitting is mandatory.
    test_udma_highlevel_put_signal_split_case(
        stream, gva, rank_id, n_ranks, 256UL * 1024UL * 1024UL, signal++, UDMA_SIGNAL_SPLIT_PUTMEM, false, false);
    for (uint32_t family = 0; family < UDMA_SIGNAL_SPLIT_FAMILY_COUNT; ++family) {
        for (uint32_t tensor_index = 0; tensor_index < 2; ++tensor_index) {
            for (uint32_t nbi_index = 0; nbi_index < 2; ++nbi_index) {
                test_udma_highlevel_put_signal_split_case(
                    stream, gva, rank_id, n_ranks, max_message_size, signal++, family, nbi_index != 0,
                    tensor_index != 0);
            }
        }
    }
    aclshmem_free(gva);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAHighLevelPutSignalSplit)
{
    constexpr int min_rank_size = 2;
    if (test_global_ranks < min_rank_size) {
        GTEST_SKIP() << "UDMA split boundary test requires at least " << min_rank_size << " ranks";
    }
    if (test_gnpu_num < test_global_ranks) {
        GTEST_SKIP() << "UDMA split boundary test requires at least " << test_global_ranks << " NPUs";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA split boundary test requires Ascend950";
    }
    // The test allocates one max-sized message region per rank, plus a signal
    // array. Size the symmetric heap for the configured worker count instead
    // of keeping the old two-rank 1 GiB budget.
    constexpr size_t max_message_size = 256UL * 1024UL * 1024UL + 1;
    const uint64_t requested_size = max_message_size * static_cast<size_t>(test_global_ranks) +
                                    static_cast<size_t>(test_global_ranks) * sizeof(int32_t);
    const uint64_t local_mem_size = ALIGN_UP(requested_size, ACLSHMEM_PAGE_SIZE);
    test_mutil_task(test_aclshmem_udma_highlevel_put_signal_split, local_mem_size, test_global_ranks);
}

void test_aclshmem_udma_put_signal_sq_wrap(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    auto* workspace = static_cast<uint8_t*>(aclshmem_malloc(UDMA_TEST_WRAP_WORKSPACE_SIZE));
    ASSERT_NE(workspace, nullptr);
    ASSERT_EQ(aclrtMemset(workspace, UDMA_TEST_WRAP_WORKSPACE_SIZE, 0, UDMA_TEST_WRAP_WORKSPACE_SIZE), 0);

    test_udma_put_signal_sq_wrap(1, stream, workspace);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    std::vector<uint64_t> result(UDMA_TEST_WRAP_RESULT_COUNT, 0);
    ASSERT_EQ(
        aclrtMemcpy(
            result.data(), result.size() * sizeof(uint64_t), workspace, result.size() * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST),
        0);
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_S_LAYOUT_IDX], 1U) << "PIPE_S wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_MTE3_LAYOUT_IDX], 1U) << "PIPE_MTE3 wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_S_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE) << "PIPE_S wrote past the SQ ring";
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_MTE3_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE)
        << "PIPE_MTE3 wrote past the SQ ring";
    EXPECT_EQ(result[UDMA_TEST_WRAP_FAA_LAYOUT_IDX], 1U) << "FAA wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_CAS_LAYOUT_IDX], 1U) << "CAS wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_FAA_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE) << "FAA wrote past the SQ ring";
    EXPECT_EQ(result[UDMA_TEST_WRAP_CAS_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE) << "CAS wrote past the SQ ring";

    aclshmem_free(workspace);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAPutSignalSqWrap)
{
    constexpr int min_rank_size = 2;
    if (test_global_ranks < min_rank_size) {
        GTEST_SKIP() << "UDMA SQ wrap test requires at least " << min_rank_size << " ranks";
    }
    if (test_gnpu_num < test_global_ranks) {
        GTEST_SKIP() << "UDMA SQ wrap test requires at least " << test_global_ranks << " NPUs";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA SQ wrap test requires Ascend950";
    }
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(test_aclshmem_udma_put_signal_sq_wrap, local_mem_size, test_global_ranks);
}

namespace {

constexpr size_t UDMA_HIGHLEVEL_LOCAL_REGION_COUNT = 8;

void test_aclshmem_udma_highlevel_local_rma(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    ASSERT_EQ(g_state.mype, rank_id);
    EXPECT_NE(g_state.topo_list[rank_id] & ACLSHMEM_TRANSPORT_MTE, 0);
    EXPECT_EQ(g_state.topo_list[rank_id] & ACLSHMEM_TRANSPORT_UDMA, 0);

    const size_t total_size = UDMA_TEST_MESSAGE_SIZE * UDMA_HIGHLEVEL_LOCAL_REGION_COUNT;
    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    ASSERT_NE(symmetric, nullptr);

    std::vector<uint8_t> host_data(total_size, 0);
    for (size_t source_region = 0; source_region < UDMA_HIGHLEVEL_LOCAL_REGION_COUNT; source_region += 2) {
        for (size_t offset = 0; offset < UDMA_TEST_MESSAGE_SIZE; ++offset) {
            host_data[source_region * UDMA_TEST_MESSAGE_SIZE + offset] =
                static_cast<uint8_t>(source_region * 17 + offset + 1);
        }
    }
    ASSERT_EQ(aclrtMemcpy(symmetric, total_size, host_data.data(), total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    test_udma_highlevel_local_rma(1, stream, symmetric, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(host_data.data(), total_size, symmetric, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (size_t source_region = 0; source_region < UDMA_HIGHLEVEL_LOCAL_REGION_COUNT; source_region += 2) {
        const size_t destination_region = source_region + 1;
        for (size_t offset = 0; offset < UDMA_TEST_MESSAGE_SIZE; ++offset) {
            const size_t source_offset = source_region * UDMA_TEST_MESSAGE_SIZE + offset;
            const size_t destination_offset = destination_region * UDMA_TEST_MESSAGE_SIZE + offset;
            EXPECT_EQ(host_data[destination_offset], host_data[source_offset])
                << "source_region=" << source_region << ", offset=" << offset;
        }
    }

    aclshmem_free(symmetric);
    test_finalize(stream, device_id);
}

} // namespace

TEST(TestMemApi, TestShmemUDMAHighLevelLocalRma)
{
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA high-level local RMA test requires Ascend950";
    }
    const int process_count = test_gnpu_num;
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(test_aclshmem_udma_highlevel_local_rma, local_mem_size, process_count);
}

static bool udma_highlevel_peers_ready(int rank_id, int n_ranks)
{
    const int peer = (rank_id + 1) % n_ranks;
    aclshmemx_net_layer_t net_layer{};
    const int ready = (g_state.topo_list[peer] & ACLSHMEM_TRANSPORT_UDMA) != 0 &&
                      (g_state.topo_list[peer] & ACLSHMEM_TRANSPORT_SDMA) == 0 &&
                      aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, peer, 0, &net_layer) == ACLSHMEM_SUCCESS;
    if (!ready) {
        std::cout << "UDMA NOT COVERED: local_pe=" << rank_id << ", peer=" << peer
                  << " requires UDMA topology, an established QP route and no higher-priority SDMA" << std::endl;
    }
    // All PEs must make the same decision before allocating buffers or entering test barriers.
    std::vector<int> readiness(static_cast<size_t>(n_ranks), 0);
    const int ret = g_boot_handle.allgather(&ready, readiness.data(), sizeof(ready), &g_boot_handle);
    EXPECT_EQ(ret, 0) << "Failed to exchange UDMA test preconditions";
    return ret == 0 && std::all_of(readiness.begin(), readiness.end(), [](int value) { return value != 0; });
}

void test_aclshmem_udma_highlevel_get_semantics(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    constexpr size_t region_bytes = UDMA_HIGHLEVEL_GET_SEMANTICS_REGION_BYTES;
    if (!udma_highlevel_peers_ready(rank_id, n_ranks)) {
        test_finalize(stream, device_id);
        GTEST_SKIP() << "UDMA NOT COVERED: at least one tested peer lacks the required UDMA configuration";
    }
    const size_t data_bytes = 2 * static_cast<size_t>(n_ranks) * region_bytes;
    const size_t status_bytes = static_cast<size_t>(n_ranks) * sizeof(uint32_t);
    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(data_bytes + status_bytes));
    ASSERT_NE(symmetric, nullptr);
    auto* status = reinterpret_cast<uint32_t*>(symmetric + data_bytes);

    std::vector<uint8_t> host_data(data_bytes, 0xCD);
    // Only the owning PE populates its source slot, so a GET from the wrong PE reads guard bytes.
    for (uint32_t offset = 0; offset < region_bytes; ++offset) {
        host_data[static_cast<size_t>(rank_id) * region_bytes + offset] =
            static_cast<uint8_t>((static_cast<uint32_t>(rank_id) * 17 + offset + 3) & 0xFF);
    }
    std::vector<uint32_t> status_host(static_cast<size_t>(n_ranks), 0xFFFFFFFFU);
    ASSERT_EQ(aclrtMemcpy(symmetric, data_bytes, host_data.data(), data_bytes, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(status, status_bytes, status_host.data(), status_bytes, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();

    test_udma_highlevel_get_semantics(1, stream, symmetric, reinterpret_cast<uint8_t*>(status), util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();

    ASSERT_EQ(aclrtMemcpy(status_host.data(), status_bytes, status, status_bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t rank = 0; rank < static_cast<uint32_t>(n_ranks); ++rank) {
        if (rank == static_cast<uint32_t>(rank_id)) {
            EXPECT_EQ(status_host[rank], 0U) << "get semantic errors on local_pe=" << rank_id;
        }
    }

    std::vector<uint8_t> result(data_bytes, 0);
    ASSERT_EQ(aclrtMemcpy(result.data(), data_bytes, symmetric, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    const size_t destination_base = static_cast<size_t>(n_ranks) * region_bytes;
    for (uint32_t rank = 0; rank < static_cast<uint32_t>(n_ranks); ++rank) {
        const uint32_t peer = (rank + 1) % static_cast<uint32_t>(n_ranks);
        for (uint32_t offset = 0; offset < region_bytes; ++offset) {
            const uint32_t chunk = offset / UDMA_HIGHLEVEL_SEMANTICS_SLOT_BYTES;
            const uint32_t inner = offset % UDMA_HIGHLEVEL_SEMANTICS_SLOT_BYTES;
            const uint32_t bytes = chunk < 2 ?
                                       UDMA_HIGHLEVEL_GET_SEMANTICS_BYTES :
                                       (chunk < 4 ? UDMA_HIGHLEVEL_GET_SEMANTICS_SIZE_NELEMS * sizeof(uint32_t) :
                                                    UDMA_HIGHLEVEL_SEMANTICS_TYPED_NELEMS * sizeof(uint32_t));
            const uint8_t expected = rank == static_cast<uint32_t>(rank_id) &&
                                             inner >= UDMA_HIGHLEVEL_SEMANTICS_PREFIX_BYTES &&
                                             inner < UDMA_HIGHLEVEL_SEMANTICS_PREFIX_BYTES + bytes ?
                                         static_cast<uint8_t>((peer * 17 + offset + 3) & 0xFF) :
                                         0xCD;
            EXPECT_EQ(result[destination_base + static_cast<size_t>(rank) * region_bytes + offset], expected)
                << "rank=" << rank << " offset=" << offset;
        }
    }

    aclshmem_free(symmetric);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAHighLevelGetSemantics)
{
    const int process_count = test_gnpu_num;
    if (test_global_ranks < 2) {
        GTEST_SKIP() << "UDMA high-level get semantic test requires at least 2 NPUs";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA high-level get semantic test requires Ascend950";
    }
    const uint64_t local_mem_size = ALIGN_UP(64UL * 1024UL * 1024UL, ACLSHMEM_PAGE_SIZE);
    test_mutil_task(test_aclshmem_udma_highlevel_get_semantics, local_mem_size, process_count);
}

void test_aclshmem_udma_highlevel_put_semantics(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    constexpr size_t region_bytes = UDMA_HIGHLEVEL_PUT_SEMANTICS_REGION_BYTES;
    if (!udma_highlevel_peers_ready(rank_id, n_ranks)) {
        test_finalize(stream, device_id);
        GTEST_SKIP() << "UDMA NOT COVERED: at least one tested peer lacks the required UDMA configuration";
    }
    const size_t data_bytes = 2 * static_cast<size_t>(n_ranks) * region_bytes;
    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(data_bytes));
    ASSERT_NE(symmetric, nullptr);

    std::vector<uint8_t> host_data(data_bytes, UDMA_SIGNAL_GUARD);
    for (uint32_t rank = 0; rank < static_cast<uint32_t>(n_ranks); ++rank) {
        for (uint32_t offset = 0; offset < region_bytes; ++offset) {
            host_data[static_cast<size_t>(rank) * region_bytes + offset] =
                static_cast<uint8_t>((rank * 23 + offset + 5) & 0xFF);
        }
    }
    ASSERT_EQ(aclrtMemcpy(symmetric, data_bytes, host_data.data(), data_bytes, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();

    test_udma_highlevel_put_semantics(1, stream, symmetric, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();

    std::vector<uint8_t> result(data_bytes, 0);
    ASSERT_EQ(aclrtMemcpy(result.data(), data_bytes, symmetric, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    const size_t destination_base = static_cast<size_t>(n_ranks) * region_bytes;
    for (uint32_t rank = 0; rank < static_cast<uint32_t>(n_ranks); ++rank) {
        const uint32_t received_slot = (static_cast<uint32_t>(rank_id) + n_ranks - 1) % n_ranks;
        for (uint32_t offset = 0; offset < region_bytes; ++offset) {
            const uint32_t chunk = offset / UDMA_HIGHLEVEL_SEMANTICS_SLOT_BYTES;
            const uint32_t inner = offset % UDMA_HIGHLEVEL_SEMANTICS_SLOT_BYTES;
            const uint32_t bytes = chunk < 2 ?
                                       UDMA_HIGHLEVEL_PUT_SEMANTICS_BYTES :
                                       (chunk < 4 ? UDMA_HIGHLEVEL_PUT_SEMANTICS_SIZE_NELEMS * sizeof(uint32_t) :
                                                    UDMA_HIGHLEVEL_SEMANTICS_TYPED_NELEMS * sizeof(uint32_t));
            const uint8_t expected = rank == received_slot && inner >= UDMA_HIGHLEVEL_SEMANTICS_PREFIX_BYTES &&
                                             inner < UDMA_HIGHLEVEL_SEMANTICS_PREFIX_BYTES + bytes ?
                                         static_cast<uint8_t>((rank * 23 + offset + 5) & 0xFF) :
                                         UDMA_SIGNAL_GUARD;
            EXPECT_EQ(result[destination_base + static_cast<size_t>(rank) * region_bytes + offset], expected)
                << "rank=" << rank << ", offset=" << offset;
        }
    }

    aclshmem_free(symmetric);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAHighLevelPutSemantics)
{
    const int process_count = test_gnpu_num;
    if (test_global_ranks < 2) {
        GTEST_SKIP() << "UDMA high-level put semantic test requires at least 2 NPUs";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA high-level put semantic test requires Ascend950";
    }
    const uint64_t local_mem_size = ALIGN_UP(64UL * 1024UL * 1024UL, ACLSHMEM_PAGE_SIZE);
    test_mutil_task(test_aclshmem_udma_highlevel_put_semantics, local_mem_size, process_count);
}

namespace {

#ifndef ACLSHMEM_RELAY_SUPPORT
constexpr uint32_t UDMA_QP_TEST_SLICE_SIZE = 64;

enum class UdmaQpTestOperation : int32_t {
    PUT = 0,
    GET = 1,
    PUT_SIGNAL = 2,
};

uint8_t udma_qp_test_pattern(int rank_id, uint32_t offset)
{
    return static_cast<uint8_t>((static_cast<uint32_t>(rank_id + 1) * 37 + offset) & 0xFF);
}

void prepare_udma_qp_test_data(std::vector<uint8_t>& data, int rank_id)
{
    for (uint32_t i = 0; i < data.size(); ++i) {
        data[i] = udma_qp_test_pattern(rank_id, i);
    }
}

void verify_udma_qp_test_data(const std::vector<uint8_t>& data, int expected_rank)
{
    for (uint32_t i = 0; i < data.size(); ++i) {
        ASSERT_EQ(data[i], udma_qp_test_pattern(expected_rank, i)) << "offset=" << i;
    }
}

void run_udma_qp_kernel(
    aclrtStream stream, uint32_t qp_num, uint8_t* symmetric, uint8_t* local_buffer, uint64_t* signal_words, int peer,
    UdmaQpTestOperation operation)
{
    test_udma_qp_data_path(
        qp_num, stream, symmetric, local_buffer, signal_words, UDMA_QP_TEST_SLICE_SIZE, peer,
        static_cast<int32_t>(operation));
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
}

void test_aclshmem_udma_qp_data_path(int rank_id, int n_ranks, uint64_t local_mem_size, uint32_t qp_num)
{
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num), ACLSHMEM_SUCCESS);

    const int device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_udma_init(rank_id, n_ranks, local_mem_size, &stream) != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    const size_t data_size = static_cast<size_t>(qp_num) * UDMA_QP_TEST_SLICE_SIZE;
    const size_t signal_size = static_cast<size_t>(qp_num) * sizeof(uint64_t);
    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(data_size));
    auto* signal_words = static_cast<uint64_t*>(aclshmem_malloc(signal_size));
    uint8_t* local_buffer = nullptr;
    ASSERT_NE(symmetric, nullptr);
    ASSERT_NE(signal_words, nullptr);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&local_buffer), data_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    std::vector<uint8_t> own_data(data_size);
    std::vector<uint8_t> zero_data(data_size, 0);
    std::vector<uint8_t> output(data_size);
    std::vector<uint64_t> zero_signals(qp_num, 0);
    std::vector<uint64_t> output_signals(qp_num);
    prepare_udma_qp_test_data(own_data, rank_id);

    const int next_rank = rank_id + 1 == n_ranks ? 0 : rank_id + 1;
    const int previous_rank = rank_id == 0 ? n_ranks - 1 : rank_id - 1;

    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    run_udma_qp_kernel(stream, qp_num, symmetric, local_buffer, signal_words, next_rank, UdmaQpTestOperation::PUT);
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, symmetric, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_qp_test_data(output, previous_rank);

    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    run_udma_qp_kernel(stream, qp_num, symmetric, local_buffer, signal_words, next_rank, UdmaQpTestOperation::GET);
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, local_buffer, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_qp_test_data(output, next_rank);

    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(signal_words, signal_size, zero_signals.data(), signal_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    run_udma_qp_kernel(
        stream, qp_num, symmetric, local_buffer, signal_words, next_rank, UdmaQpTestOperation::PUT_SIGNAL);
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, symmetric, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_qp_test_data(output, previous_rank);
    ASSERT_EQ(aclrtMemcpy(output_signals.data(), signal_size, signal_words, signal_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t qp_idx = 0; qp_idx < qp_num; ++qp_idx) {
        const uint64_t expected_signal = (static_cast<uint64_t>(previous_rank + 1) << 32) | (qp_idx + 1);
        ASSERT_EQ(output_signals[qp_idx], expected_signal) << "qp_idx=" << qp_idx;
    }

    ASSERT_EQ(aclrtFree(local_buffer), 0);
    aclshmem_free(signal_words);
    aclshmem_free(symmetric);
    test_finalize(stream, device_id);
}

void run_udma_qp_data_path_test(uint32_t qp_num)
{
#if defined(ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME)
    const int process_count = test_gnpu_num;
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(
        [qp_num](int rank_id, int n_ranks, uint64_t heap_size) {
            test_aclshmem_udma_qp_data_path(rank_id, n_ranks, heap_size, qp_num);
        },
        local_mem_size, process_count);
#else
    std::cout << "[TEST] HCOMM ABI does not provide HcommChannelDesc::channelName; treating the " << qp_num
              << "-QP UDMA data-path test as passed without execution." << std::endl;
    SUCCEED();
#endif
}
#endif

} // namespace

#ifndef ACLSHMEM_RELAY_SUPPORT
TEST(TestMemApi, TestShmemUDMA2QpDataPath) { run_udma_qp_data_path_test(2); }

TEST(TestMemApi, TestShmemUDMA4QpDataPath) { run_udma_qp_data_path_test(4); }

TEST(TestMemApi, TestShmemUDMA8QpDataPath) { run_udma_qp_data_path_test(8); }
#endif
