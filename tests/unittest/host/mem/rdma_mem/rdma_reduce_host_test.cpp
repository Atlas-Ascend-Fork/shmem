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
#include <limits>
#include <iostream>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "opdev/bfloat16.h"
#include "opdev/fp16_t.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "unittest/utils/rdma_reduce_kernel.h"

#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)

namespace {
constexpr int TARGET_PE = 1;
constexpr uint32_t QP_NUM = 2;
constexpr uint32_t MAX_ELEMENTS = 65;
constexpr uint32_t PARTIAL_OFFSET = 8;
constexpr uint64_t HEAP_SIZE = 64UL * 1024UL * 1024UL;
constexpr std::array<aclshmemx_reduce_op_t, 3> OPS{ACLSHMEMX_REDUCE_ADD, ACLSHMEMX_REDUCE_MAX, ACLSHMEMX_REDUCE_MIN};
using ReduceKernel = void (*)(
    uint32_t, void*, uint8_t*, uint8_t*, uint32_t, aclshmemx_reduce_op_t, RdmaReduceApi, uint32_t, uint32_t, uint64_t);

template <typename T>
T source_value(uint32_t i, int rank, aclshmemx_reduce_op_t op)
{
    if constexpr (std::is_same_v<T, uint32_t>) {
        if (op != ACLSHMEMX_REDUCE_ADD) {
            return i % 3 == 0 ? 1U : (rank == 0 ? 0x80000001U : 0x80000008U);
        }
        return i % 5 + 1;
    } else {
        const int value = static_cast<int>(i % 5) - 2 + (rank == 2 ? 1 : 0);
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(value);
        } else {
            return static_cast<T>(static_cast<float>(value) * 0.5F + 0.25F);
        }
    }
}

template <typename T>
T initial_value(uint32_t i, aclshmemx_reduce_op_t op)
{
    if constexpr (std::is_same_v<T, uint32_t>) {
        return op == ACLSHMEMX_REDUCE_ADD ? 7U : (i % 2 == 0 ? 0x80000002U : 0x7ffffffeU);
    } else {
        return static_cast<T>(static_cast<int>(i % 3) * 4 - 4);
    }
}

template <typename T>
T reference_reduce(T dst, T src, aclshmemx_reduce_op_t op)
{
    if constexpr (std::is_integral_v<T>) {
        if (op == ACLSHMEMX_REDUCE_ADD) {
            return static_cast<T>(dst + src);
        }
        return op == ACLSHMEMX_REDUCE_MAX ? (dst > src ? dst : src) : (dst < src ? dst : src);
    } else {
        const float left = static_cast<float>(dst);
        const float right = static_cast<float>(src);
        const float result = op == ACLSHMEMX_REDUCE_ADD ? left + right :
                             op == ACLSHMEMX_REDUCE_MAX ? (left > right ? left : right) :
                                                          (left < right ? left : right);
        return static_cast<T>(result);
    }
}

struct ReduceCase {
    uint32_t elements;
    aclshmemx_reduce_op_t op;
    RdmaReduceApi api;
    uint32_t iterations = 3;
    bool concurrent = false;
    uint32_t second_offset = 0;
};

template <typename T>
void run_reduce_case(aclrtStream stream, int rank, T* dst, T* src, ReduceKernel kernel, const ReduceCase& test_case)
{
    SCOPED_TRACE(
        ::testing::Message() << "PE=" << rank << " op=" << static_cast<uint32_t>(test_case.op)
                             << " api=" << static_cast<uint32_t>(test_case.api) << " elements=" << test_case.elements
                             << " iterations=" << test_case.iterations << " concurrent=" << test_case.concurrent
                             << " offset=" << test_case.second_offset);
    constexpr uint32_t guard_elements = 64 / sizeof(T);
    constexpr uint32_t total_elements = 2 * guard_elements + MAX_ELEMENTS + PARTIAL_OFFSET;
    constexpr size_t bytes = total_elements * sizeof(T);
    std::vector<T> input(total_elements, static_cast<T>(19));
    std::vector<T> expected(total_elements, static_cast<T>(19));
    std::vector<T> output(total_elements);
    for (uint32_t i = 0; i < MAX_ELEMENTS + PARTIAL_OFFSET; ++i) {
        expected[guard_elements + i] = initial_value<T>(i, test_case.op);
        input[guard_elements + i] = source_value<T>(i, rank, test_case.op);
    }
    ASSERT_EQ(aclrtMemcpy(dst, bytes, expected.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(src, bytes, input.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    const bool explicit_qp = test_case.api == RdmaReduceApi::POINTER_QP || test_case.api == RdmaReduceApi::TENSOR_QP;
    const uint32_t qp_idx = explicit_qp ? 1U : 0U;
    if (rank == 0 || (test_case.concurrent && rank == 2)) {
        const uint32_t offset = rank == 2 ? test_case.second_offset : 0;
        kernel(
            1, stream, reinterpret_cast<uint8_t*>(dst + guard_elements + offset),
            reinterpret_cast<uint8_t*>(src + guard_elements), test_case.elements, test_case.op, test_case.api, qp_idx,
            test_case.iterations, util_get_ffts_config());
        ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, 30), ACL_SUCCESS);
    }
    // Every source has completed before the target is copied back or any allocation is reused.
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    if (rank == TARGET_PE) {
        const uint32_t updates = test_case.iterations;
        for (int sender : {0, 2}) {
            if (sender == 2 && !test_case.concurrent) {
                continue;
            }
            const uint32_t offset = sender == 2 ? test_case.second_offset : 0;
            for (uint32_t repeat = 0; repeat < updates; ++repeat) {
                for (uint32_t i = 0; i < test_case.elements; ++i) {
                    auto& value = expected[guard_elements + offset + i];
                    value = reference_reduce(value, source_value<T>(i, sender, test_case.op), test_case.op);
                }
            }
        }
    }
    ASSERT_EQ(aclrtMemcpy(output.data(), bytes, dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (uint32_t i = 0; i < total_elements; ++i) {
        if constexpr (std::is_integral_v<T>) {
            EXPECT_EQ(output[i], expected[i]) << "destination element " << i;
        } else {
            EXPECT_EQ(static_cast<float>(output[i]), static_cast<float>(expected[i])) << "destination element " << i;
        }
    }
    ASSERT_EQ(aclrtMemcpy(output.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (uint32_t i = 0; i < total_elements; ++i) {
        EXPECT_EQ(output[i], input[i]) << "source element " << i;
    }
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
}

template <typename T>
void test_reduce_rank(int rank, int ranks, uint64_t heap_size, ReduceKernel kernel, bool concurrent)
{
    aclrtStream stream = nullptr;
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, QP_NUM), ACLSHMEM_SUCCESS);
    ASSERT_EQ(test_rdma_init(rank, ranks, heap_size, &stream), ACLSHMEM_SUCCESS);
    constexpr size_t bytes = (128 / sizeof(T) + MAX_ELEMENTS + PARTIAL_OFFSET) * sizeof(T);
    auto* dst = static_cast<T*>(aclshmem_malloc(bytes));
    auto* src = static_cast<T*>(aclshmem_malloc(bytes));
    ASSERT_NE(dst, nullptr);
    ASSERT_NE(src, nullptr);
    if (concurrent) {
        for (auto op : OPS) {
            for (uint32_t offset : {0U, PARTIAL_OFFSET}) {
                run_reduce_case(stream, rank, dst, src, kernel, {17, op, RdmaReduceApi::TENSOR_QP, 3, true, offset});
            }
        }
    } else {
        const std::array<uint32_t, 3> lengths{1, 17, MAX_ELEMENTS};
        for (uint32_t form = 0; form < 6; ++form) {
            const auto api = static_cast<RdmaReduceApi>(form);
            for (auto op : OPS) {
                run_reduce_case(stream, rank, dst, src, kernel, {lengths[form % lengths.size()], op, api});
            }
        }
    }
    aclshmem_free(src);
    aclshmem_free(dst);
    test_finalize(stream, rank % test_gnpu_num + test_first_npu);
}

bool reduce_supported()
{
    const char* soc = aclrtGetSocName();
    return soc != nullptr && std::string(soc).find("Ascend950") != std::string::npos;
}
} // namespace

#define DEFINE_RDMA_REDUCE_TEST(NAME, HOST_TYPE, DEVICE_TYPE, REDUCE_TYPE)                                  \
    TEST(TestMemApi, TestShmemRdmaPutReduce##NAME)                                                          \
    {                                                                                                       \
        if (!reduce_supported() || test_global_ranks < 2) {                                                 \
            GTEST_SKIP() << "RoCE Put Reduce requires Ascend950 HNS_1825 and at least two PEs";             \
        }                                                                                                   \
        test_mutil_task(                                                                                    \
            [](int rank, int ranks, uint64_t heap) {                                                        \
                test_reduce_rank<HOST_TYPE>(rank, ranks, heap, test_rdma_put_reduce_##NAME##_do, false);    \
            },                                                                                              \
            HEAP_SIZE, test_gnpu_num);                                                                      \
    }                                                                                                       \
    TEST(TestMemApi, TestShmemRdmaPutReduceConcurrent##NAME)                                                \
    {                                                                                                       \
        if (!reduce_supported() || test_global_ranks < 4) {                                                 \
            GTEST_SKIP() << "Concurrent RoCE Put Reduce requires Ascend950 HNS_1825 and at least four PEs"; \
        }                                                                                                   \
        test_mutil_task(                                                                                    \
            [](int rank, int ranks, uint64_t heap) {                                                        \
                test_reduce_rank<HOST_TYPE>(rank, ranks, heap, test_rdma_put_reduce_##NAME##_do, true);     \
            },                                                                                              \
            HEAP_SIZE, test_gnpu_num);                                                                      \
    }
ACLSHMEM_RDMA_REDUCE_TEST_TYPES(DEFINE_RDMA_REDUCE_TEST);
#undef DEFINE_RDMA_REDUCE_TEST

namespace {
void test_fp16_saturation_rank(int rank, int ranks, uint64_t heap_size)
{
    struct SaturationCase {
        uint16_t destination;
        uint16_t source;
        uint16_t add_normal;
        uint16_t add_sat;
        uint16_t maximum;
        uint16_t minimum;
    };
    constexpr std::array<SaturationCase, 8> cases{{
        {0x3c00, 0x4000, 0x4200, 0x4200, 0x4000, 0x3c00},
        {0x3a00, 0x3800, 0x3d00, 0x3d00, 0x3a00, 0x3800},
        {0xb800, 0x3400, 0xb400, 0xb400, 0x3400, 0xb800},
        {0x7bff, 0x7bff, 0x7c00, 0x7bff, 0x7bff, 0x7bff},
        {0xfbff, 0xfbff, 0xfc00, 0xfbff, 0xfbff, 0xfbff},
        {0x7bff, 0x5000, 0x7c00, 0x7bff, 0x7bff, 0x5000},
        {0xfbff, 0xd000, 0xfc00, 0xfbff, 0xd000, 0xfbff},
        {0x7bff, 0xfbff, 0x0000, 0x0000, 0x7bff, 0xfbff},
    }};
    constexpr size_t guard = 32;
    constexpr size_t count = cases.size();
    constexpr size_t total = guard * 2 + count * 2;
    constexpr size_t bytes = total * sizeof(uint16_t);
    aclrtStream stream = nullptr;
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, QP_NUM), ACLSHMEM_SUCCESS);
    ASSERT_EQ(test_rdma_init(rank, ranks, heap_size, &stream), ACLSHMEM_SUCCESS);
    auto* dst = static_cast<uint16_t*>(aclshmem_malloc(bytes));
    auto* src = static_cast<uint16_t*>(aclshmem_malloc(bytes));
    ASSERT_NE(dst, nullptr);
    ASSERT_NE(src, nullptr);
    std::array<uint16_t, total> initial;
    std::array<uint16_t, total> source;
    std::array<uint16_t, total> output;
    initial.fill(0x3555);
    source.fill(0x3555);
    for (size_t i = 0; i < count; ++i) {
        initial[guard + i] = initial[guard + count + i] = cases[i].destination;
        source[guard + i] = cases[i].source;
    }
    for (uint32_t form = 0; form < 6; ++form) {
        const auto api = static_cast<RdmaReduceApi>(form);
        const uint32_t qp_idx = api == RdmaReduceApi::POINTER_QP || api == RdmaReduceApi::TENSOR_QP ? 1 : 0;
        for (auto op : OPS) {
            SCOPED_TRACE(
                ::testing::Message() << "PE=" << rank << " api=" << form << " op=" << static_cast<uint32_t>(op));
            auto expected = initial;
            ASSERT_EQ(aclrtMemcpy(dst, bytes, initial.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            ASSERT_EQ(aclrtMemcpy(src, bytes, source.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
            if (rank == 0) {
                test_rdma_put_reduce_half_do(
                    1, stream, reinterpret_cast<uint8_t*>(dst + guard), reinterpret_cast<uint8_t*>(src + guard), count,
                    op, api, qp_idx, 1, util_get_ffts_config());
                test_rdma_put_reduce_half_sat_do(
                    1, stream, reinterpret_cast<uint8_t*>(dst + guard + count), reinterpret_cast<uint8_t*>(src + guard),
                    count, op, api, qp_idx, 1, util_get_ffts_config());
                ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, 30), ACL_SUCCESS);
            }
            ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
            if (rank == TARGET_PE) {
                for (size_t i = 0; i < count; ++i) {
                    const auto& test_case = cases[i];
                    const uint16_t comparison = op == ACLSHMEMX_REDUCE_MAX ? test_case.maximum : test_case.minimum;
                    expected[guard + i] = op == ACLSHMEMX_REDUCE_ADD ? test_case.add_normal : comparison;
                    expected[guard + count + i] = op == ACLSHMEMX_REDUCE_ADD ? test_case.add_sat : comparison;
                }
            }
            ASSERT_EQ(aclrtMemcpy(output.data(), bytes, dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
            for (size_t i = 0; i < total; ++i) {
                EXPECT_EQ(output[i], expected[i]) << "destination element " << i;
            }
            ASSERT_EQ(aclrtMemcpy(output.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
            EXPECT_EQ(output, source);
            ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
        }
    }
    aclshmem_free(src);
    aclshmem_free(dst);
    test_finalize(stream, rank % test_gnpu_num + test_first_npu);
}

void test_reduce_queue_rank(int rank, int ranks, uint64_t heap_size, bool wrap)
{
    constexpr uint32_t batch_size = 8;
    constexpr uint32_t guard = 16;
    using QueueState = std::array<uint32_t, REDUCE_QUEUE_FIELDS>;
    QueueState state{};
    std::vector<QueueState> all_states(ranks);
    aclrtStream stream = nullptr;
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, QP_NUM), ACLSHMEM_SUCCESS);
    ASSERT_EQ(test_rdma_init(rank, ranks, heap_size, &stream), ACLSHMEM_SUCCESS);
    auto* device_state = static_cast<uint32_t*>(aclshmem_malloc(sizeof(state)));
    ASSERT_NE(device_state, nullptr);
    if (rank == 0) {
        test_rdma_put_reduce_queue_do(1, stream, nullptr, nullptr, 0, false, device_state, util_get_ffts_config());
        ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, 30), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(state.data(), sizeof(state), device_state, sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
    }
    ASSERT_EQ(g_boot_handle.allgather(state.data(), all_states.data(), sizeof(state), &g_boot_handle), 0);
    const auto initial = all_states[0];
    const uint32_t sq_depth = initial[REDUCE_SQ_DEPTH];
    const uint32_t cq_depth = initial[REDUCE_CQ_DEPTH];
    ASSERT_GT(sq_depth, 0U);
    ASSERT_GT(cq_depth, 0U);
    ASSERT_EQ(initial[REDUCE_SQ_HEAD], initial[REDUCE_SQ_TAIL]);
    const uint64_t requests = wrap ? 2ULL * std::max(sq_depth, cq_depth) + 3 : 1;
    ASSERT_LT(requests, static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
    const size_t elements = guard * 2 + requests + 1;
    const size_t bytes = elements * sizeof(int32_t);
    ASSERT_LT(bytes * 2 + sizeof(state), heap_size);
    auto* dst = static_cast<int32_t*>(aclshmem_malloc(bytes));
    auto* src = static_cast<int32_t*>(aclshmem_malloc(bytes));
    ASSERT_NE(dst, nullptr);
    ASSERT_NE(src, nullptr);
    std::vector<int32_t> expected(elements, -7);
    std::vector<int32_t> source(elements, 19);
    std::vector<int32_t> output(elements);
    for (uint32_t i = 0; i <= requests; ++i) {
        source[guard + i] = static_cast<int32_t>(i + 1);
    }
    ASSERT_EQ(aclrtMemcpy(dst, bytes, expected.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(src, bytes, source.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    uint32_t verified = 0;
    bool recovered = false;
    auto previous = initial;
    for (uint32_t posted = 0; posted < requests;) {
        const uint32_t count = std::min<uint64_t>(batch_size, requests - posted);
        const bool complete = posted + count == requests;
        SCOPED_TRACE(::testing::Message() << "PE=" << rank << " posted=" << posted << " count=" << count);
        if (rank == 0) {
            // Pending requests survive kernel boundaries so submission still exercises automatic CQ recovery.
            test_rdma_put_reduce_queue_do(
                1, stream, dst + guard + posted, src + guard + posted, count, complete, device_state,
                util_get_ffts_config());
            ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, 30), ACL_SUCCESS);
            ASSERT_EQ(
                aclrtMemcpy(state.data(), sizeof(state), device_state, sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
        }
        ASSERT_EQ(g_boot_handle.allgather(state.data(), all_states.data(), sizeof(state), &g_boot_handle), 0);
        const auto current = all_states[0];
        posted += count;
        ASSERT_EQ(current[REDUCE_SQ_HEAD] - initial[REDUCE_SQ_HEAD], posted + (complete ? 1U : 0U));
        const uint32_t consumed = current[REDUCE_SQ_TAIL] - initial[REDUCE_SQ_TAIL];
        ASSERT_LE(consumed, posted + (complete ? 1U : 0U));
        ASSERT_EQ(current[REDUCE_CQ_TAIL] - initial[REDUCE_CQ_TAIL], consumed);
        recovered = recovered || (!complete && current[REDUCE_SQ_TAIL] != previous[REDUCE_SQ_TAIL]);
        const uint32_t completed = std::min<uint64_t>(requests, consumed);
        if (rank == TARGET_PE && completed > verified) {
            const size_t completed_bytes = (completed - verified) * sizeof(int32_t);
            ASSERT_EQ(
                aclrtMemcpy(
                    output.data() + guard + verified, completed_bytes, dst + guard + verified, completed_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            for (uint32_t i = verified; i < completed; ++i) {
                expected[guard + i] += source[guard + i];
                EXPECT_EQ(output[guard + i], expected[guard + i]) << "request " << i;
            }
        }
        verified = completed;
        if (rank == 0 && (posted == count || complete || current[REDUCE_SQ_TAIL] != previous[REDUCE_SQ_TAIL] ||
                          current[REDUCE_SQ_HEAD] / sq_depth != previous[REDUCE_SQ_HEAD] / sq_depth)) {
            std::cout << "[PUT_REDUCE_QUEUE] posted=" << posted << "/" << requests << " SQ=" << current[REDUCE_SQ_HEAD]
                      << "/" << current[REDUCE_SQ_TAIL] << " CQ=" << current[REDUCE_CQ_TAIL] << " depths=" << sq_depth
                      << "/" << cq_depth << std::endl;
        }
        previous = current;
    }
    EXPECT_EQ(previous[REDUCE_SQ_HEAD], previous[REDUCE_SQ_TAIL]);
    EXPECT_EQ(verified, requests);
    if (wrap) {
        EXPECT_TRUE(recovered) << "automatic CQ recovery was not exercised";
        EXPECT_GE(previous[REDUCE_SQ_HEAD] / sq_depth - initial[REDUCE_SQ_HEAD] / sq_depth, 2U);
        EXPECT_GE(previous[REDUCE_CQ_TAIL] / cq_depth - initial[REDUCE_CQ_TAIL] / cq_depth, 2U);
    }
    if (rank == TARGET_PE) {
        expected[guard + requests] = source[guard + requests];
    }
    ASSERT_EQ(aclrtMemcpy(output.data(), bytes, dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < elements; ++i) {
        EXPECT_EQ(output[i], expected[i]) << "destination element " << i;
    }
    ASSERT_EQ(aclrtMemcpy(output.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(output, source);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    aclshmem_free(src);
    aclshmem_free(dst);
    aclshmem_free(device_state);
    test_finalize(stream, rank % test_gnpu_num + test_first_npu);
}
} // namespace

TEST(TestMemApi, TestShmemRdmaPutReduceFp16Saturation)
{
    if (!reduce_supported() || test_global_ranks < 2) {
        GTEST_SKIP() << "RoCE Put Reduce requires Ascend950 HNS_1825 and at least two PEs";
    }
    test_mutil_task(test_fp16_saturation_rank, HEAP_SIZE, test_gnpu_num);
}

TEST(TestMemApi, TestShmemRdmaPutReduceThenPut)
{
    if (!reduce_supported() || test_global_ranks < 2) {
        GTEST_SKIP() << "RoCE Put Reduce requires Ascend950 HNS_1825 and at least two PEs";
    }
    test_mutil_task(
        [](int rank, int ranks, uint64_t heap) { test_reduce_queue_rank(rank, ranks, heap, false); }, HEAP_SIZE,
        test_gnpu_num);
}

TEST(TestMemApi, TestShmemRdmaPutReduceQueueWrap)
{
    if (!reduce_supported() || test_global_ranks < 2) {
        GTEST_SKIP() << "RoCE Put Reduce requires Ascend950 HNS_1825 and at least two PEs";
    }
    test_mutil_task(
        [](int rank, int ranks, uint64_t heap) { test_reduce_queue_rank(rank, ranks, heap, true); }, HEAP_SIZE,
        test_gnpu_num);
}

#endif // ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
