/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file simt_udma_qp_test.cpp
 * @brief SIMT UDMA QP-specific unit tests (host side).
 *
 * Test matrix (dual-process = 2 PE):
 *
 * Group 1 - Basic correctness (Scenario 1: single put+quiet / get+quiet per QP):
 *   warp/block x put/get, parameterized qp_num (2/4/8) and slice (2B/1001B/1024B/4096B).
 *   Covers correctness and length boundaries (2B minimum on all driver paths / odd length).
 *   Note: 1B is excluded and rejected by the implementation's length check (L1 debug abort) - DWQE dual-SGE
 *   split makes first_len=0 (zero-length SGE#1), triggering HW error 355.
 *
 * Group 2 - Multi-QP parallel (Scenario 2: warp i binds QP i, single kernel launch):
 *   Validates legal concurrent multi-driver under single-driver constraint.
 *   This is the primary recommended usage pattern (multiple warps each driving a distinct QP).
 *
 * Convention: single-driver constraint satisfied (each kernel = one driver per QP).
 */

#if defined(USE_SIMT)

#include <cstring>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "host/init/shmem_host_init.h"
#include "shmemi_host_common.h"
#include "simt_udma_qp_kernels.h"
#include "unittest_main_test.h"

using SimtUdmaQpKernel =
    void (*)(void* stream, uint8_t* dst, uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx);

using SimtUdmaMqpKernel =
    void (*)(void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe);

namespace {

// Pattern helpers (consistent with details/ examples for cross-validation).
uint8_t TestPattern(int pe, size_t offset)
{
    return static_cast<uint8_t>((static_cast<uint64_t>(pe + 1) * 37 + offset % 251) & 0xFF);
}

void FillPattern(std::vector<uint8_t>& data, int pe, size_t base)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = TestPattern(pe, base + i);
    }
}

bool CheckPattern(const std::vector<uint8_t>& data, int expected_pe, size_t base)
{
    for (size_t i = 0; i < data.size(); ++i) {
        const uint8_t expected = TestPattern(expected_pe, base + i);
        if (data[i] != expected) {
            ADD_FAILURE() << "data[" << i << "] = " << static_cast<uint32_t>(data[i]) << ", expected "
                          << static_cast<uint32_t>(expected) << " (pe=" << expected_pe << ", base=" << base << ")";
            return false;
        }
    }
    return true;
}

// ===========================================================================
// Group 1: Basic correctness (Scenario 1)
// qp_num QPs, each QP handles one slice. Host loops over QPs (single driver per QP).
// PUT: local -> peer symmetric; GET: peer symmetric -> local.
// ===========================================================================

void RunSimtUdmaQpCase(
    int rank_id, int n_ranks, uint64_t local_mem_size, SimtUdmaQpKernel put_kernel, SimtUdmaQpKernel get_kernel,
    size_t slice, uint32_t qp_num, size_t dtype_size)
{
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num), ACLSHMEM_SUCCESS);

    const int device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_udma_init(rank_id, n_ranks, local_mem_size, &stream) != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    // slice is in the kernel's elem_size unit (elements); buffers are byte-sized -> x dtype_size.
    const size_t data_size = static_cast<size_t>(qp_num) * slice * dtype_size;
    const int next_rank = rank_id + 1 == n_ranks ? 0 : rank_id + 1;

    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(data_size));
    uint8_t* local_buffer = nullptr;
    ASSERT_NE(symmetric, nullptr);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&local_buffer), data_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    std::vector<uint8_t> own_data(data_size);
    std::vector<uint8_t> zero_data(data_size, 0);
    std::vector<uint8_t> output(data_size);
    FillPattern(own_data, rank_id, 0);

    // ---- PUT: local -> peer symmetric (per-QP sequential submit, single driver) ----
    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    for (uint32_t qp_idx = 0; qp_idx < qp_num; ++qp_idx) {
        put_kernel(
            stream, symmetric + static_cast<size_t>(qp_idx) * slice * dtype_size,
            local_buffer + static_cast<size_t>(qp_idx) * slice * dtype_size, slice, next_rank, qp_idx);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0) << "PUT kernel failed, errMsg: " << aclGetRecentErrMsg();
    aclshmemi_control_barrier_all();
    // Verify: local symmetric should contain previous_rank's pattern.
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, symmetric, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    const int previous_rank = rank_id == 0 ? n_ranks - 1 : rank_id - 1;
    CheckPattern(output, previous_rank, 0);

    // ---- GET: peer symmetric -> local (per-QP sequential submit) ----
    // Re-fill symmetric with own pattern so GET pulls peer's (next_rank) pattern, not our pushed data.
    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    for (uint32_t qp_idx = 0; qp_idx < qp_num; ++qp_idx) {
        get_kernel(
            stream, local_buffer + static_cast<size_t>(qp_idx) * slice * dtype_size,
            symmetric + static_cast<size_t>(qp_idx) * slice * dtype_size, slice, next_rank, qp_idx);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0) << "GET kernel failed, errMsg: " << aclGetRecentErrMsg();
    aclshmemi_control_barrier_all();
    // Verify: local should contain next_rank's pattern.
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, local_buffer, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    CheckPattern(output, next_rank, 0);

    ASSERT_EQ(aclrtFree(local_buffer), 0);
    aclshmem_free(symmetric);
    test_finalize(stream, device_id);
}

void RunTestCase(SimtUdmaQpKernel put_kernel, SimtUdmaQpKernel get_kernel, size_t slice, uint32_t qp_num)
{
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(
        [put_kernel, get_kernel, slice, qp_num](int rank_id, int n_ranks, uint64_t heap_size) {
            RunSimtUdmaQpCase(rank_id, n_ranks, heap_size, put_kernel, get_kernel, slice, qp_num, 1);
        },
        local_mem_size, 2);
}

// ===========================================================================
// Group 1b: Different symmetric-heap VA bases across ranks.
// Each rank reserves a rank-dependent VA gap before aclshmemx_init_attr, so the
// symmetric heap VA differs across PEs. This validates that the SIMT UDMA
// submission path uses the peer's registered MR base (mem_info->addr), not the
// local peer-mapping address. The regression would otherwise go undetected when
// all PEs share the same heap VA. This mirrors the SIMD-side
// TestShmemUDMAMemWithDifferentVirtualBases.
// ===========================================================================

constexpr size_t SIMT_UDMA_VA_GAP_GRANULARITY = 1ULL << 30; // 1 GiB

void RunSimtUdmaQpCaseWithVaGap(
    int rank_id, int n_ranks, uint64_t local_mem_size, SimtUdmaQpKernel put_kernel, SimtUdmaQpKernel get_kernel,
    size_t slice, uint32_t qp_num, size_t dtype_size)
{
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num), ACLSHMEM_SUCCESS);

    const int device_id = rank_id % test_gnpu_num + test_first_npu;
    const size_t va_gap_size = SIMT_UDMA_VA_GAP_GRANULARITY * (static_cast<size_t>(rank_id) + 1U);
    aclrtStream stream = nullptr;
    void* va_gap = nullptr;
    ASSERT_EQ(test_udma_init_with_va_gap(rank_id, n_ranks, local_mem_size, va_gap_size, &stream, &va_gap), 0);
    ASSERT_NE(stream, nullptr);

    const size_t data_size = static_cast<size_t>(qp_num) * slice * dtype_size;
    const int next_rank = rank_id + 1 == n_ranks ? 0 : rank_id + 1;

    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(data_size));
    uint8_t* local_buffer = nullptr;
    ASSERT_NE(symmetric, nullptr);

    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&local_buffer), data_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    std::vector<uint8_t> own_data(data_size);
    std::vector<uint8_t> zero_data(data_size, 0);
    std::vector<uint8_t> output(data_size);
    FillPattern(own_data, rank_id, 0);

    // ---- PUT: local -> peer symmetric (per-QP sequential submit) ----
    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    for (uint32_t qp_idx = 0; qp_idx < qp_num; ++qp_idx) {
        put_kernel(
            stream, symmetric + static_cast<size_t>(qp_idx) * slice * dtype_size,
            local_buffer + static_cast<size_t>(qp_idx) * slice * dtype_size, slice, next_rank, qp_idx);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0) << "PUT kernel failed, errMsg: " << aclGetRecentErrMsg();
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, symmetric, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    const int previous_rank = rank_id == 0 ? n_ranks - 1 : rank_id - 1;
    CheckPattern(output, previous_rank, 0);

    // ---- GET: peer symmetric -> local (per-QP sequential submit) ----
    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    for (uint32_t qp_idx = 0; qp_idx < qp_num; ++qp_idx) {
        get_kernel(
            stream, local_buffer + static_cast<size_t>(qp_idx) * slice * dtype_size,
            symmetric + static_cast<size_t>(qp_idx) * slice * dtype_size, slice, next_rank, qp_idx);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0) << "GET kernel failed, errMsg: " << aclGetRecentErrMsg();
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, local_buffer, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    CheckPattern(output, next_rank, 0);

    ASSERT_EQ(aclrtFree(local_buffer), 0);
    aclshmem_free(symmetric);
    EXPECT_EQ(aclrtReleaseMemAddress(va_gap), ACL_SUCCESS);
    test_finalize(stream, device_id);
}

void RunTestCaseWithVaGap(SimtUdmaQpKernel put_kernel, SimtUdmaQpKernel get_kernel, size_t slice, uint32_t qp_num)
{
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(
        [put_kernel, get_kernel, slice, qp_num](int rank_id, int n_ranks, uint64_t heap_size) {
            RunSimtUdmaQpCaseWithVaGap(rank_id, n_ranks, heap_size, put_kernel, get_kernel, slice, qp_num, 1);
        },
        local_mem_size, 2);
}

// ===========================================================================
// Group 2: Multi-QP parallel (Scenario 2)
// Single kernel launch: warp i drives QP i. Validates parallel multi-QP correctness.
// This is the primary recommended usage (multiple warps each driving a distinct QP).
// ===========================================================================

void RunMqpParallelCase(
    int rank_id, int n_ranks, uint64_t local_mem_size, size_t slice, uint32_t qp_num, SimtUdmaMqpKernel put_kernel,
    SimtUdmaMqpKernel get_kernel)
{
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num), ACLSHMEM_SUCCESS);

    const int device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_udma_init(rank_id, n_ranks, local_mem_size, &stream) != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    const size_t data_size = static_cast<size_t>(qp_num) * slice;
    const int next_rank = rank_id + 1 == n_ranks ? 0 : rank_id + 1;

    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(data_size));
    uint8_t* local_buffer = nullptr;
    ASSERT_NE(symmetric, nullptr);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&local_buffer), data_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    std::vector<uint8_t> own_data(data_size);
    std::vector<uint8_t> zero_data(data_size, 0);
    std::vector<uint8_t> output(data_size);
    FillPattern(own_data, rank_id, 0);

    // ---- PUT: local -> peer symmetric (single kernel, multi-QP) ----
    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    put_kernel(stream, symmetric, local_buffer, static_cast<uint32_t>(slice), qp_num, next_rank);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0) << "MQP PUT kernel failed, errMsg: " << aclGetRecentErrMsg();
    aclshmemi_control_barrier_all();
    // Verify: local symmetric should contain previous_rank's pattern.
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, symmetric, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    const int previous_rank = rank_id == 0 ? n_ranks - 1 : rank_id - 1;
    CheckPattern(output, previous_rank, 0);

    // ---- GET: peer symmetric -> local (single kernel, multi-QP) ----
    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    get_kernel(stream, local_buffer, symmetric, static_cast<uint32_t>(slice), qp_num, next_rank);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0) << "MQP GET kernel failed, errMsg: " << aclGetRecentErrMsg();
    aclshmemi_control_barrier_all();
    // Verify: local should contain next_rank's pattern.
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, local_buffer, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    CheckPattern(output, next_rank, 0);

    ASSERT_EQ(aclrtFree(local_buffer), 0);
    aclshmem_free(symmetric);
    test_finalize(stream, device_id);
}

void RunMqpTestCase(SimtUdmaMqpKernel put_kernel, SimtUdmaMqpKernel get_kernel, size_t slice, uint32_t qp_num)
{
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(
        [put_kernel, get_kernel, slice, qp_num](int rank_id, int n_ranks, uint64_t heap_size) {
            RunMqpParallelCase(rank_id, n_ranks, heap_size, slice, qp_num, put_kernel, get_kernel);
        },
        local_mem_size, 2);
}

} // namespace

// ===========================================================================
// Group 1: Basic correctness (warp/block x qp_num x slice)
// ===========================================================================

// warp mode: qp_num=2, even + odd (existing baseline preserved)
TEST(SimtUdmaQpTest, SimtUdmaQpWarpQp2Slice1024)
{
    RunTestCase(test_simt_udma_qp_put_warp, test_simt_udma_qp_get_warp, 1024, 2);
}

TEST(SimtUdmaQpTest, SimtUdmaQpWarpQp2OddLength)
{
    RunTestCase(test_simt_udma_qp_put_warp, test_simt_udma_qp_get_warp, 1001, 2);
}

// warp mode: 2B minimum valid length (coop submit path; len=1 would make first_len=0, HW error 355)
TEST(SimtUdmaQpTest, SimtUdmaQpWarpQp2Slice2)
{
    RunTestCase(test_simt_udma_qp_put_warp, test_simt_udma_qp_get_warp, 2, 2);
}

// warp mode: qp_num=4 (expand QP count coverage)
TEST(SimtUdmaQpTest, SimtUdmaQpWarpQp4Slice4096)
{
    RunTestCase(test_simt_udma_qp_put_warp, test_simt_udma_qp_get_warp, 4096, 4);
}

// warp mode: qp_num=8 (match SIMD QP data path coverage)
TEST(SimtUdmaQpTest, SimtUdmaQpWarpQp8Slice4096)
{
    RunTestCase(test_simt_udma_qp_put_warp, test_simt_udma_qp_get_warp, 4096, 8);
}

// block mode: qp_num=2, even + odd (existing baseline preserved)
TEST(SimtUdmaQpTest, SimtUdmaQpBlockQp2Slice1024)
{
    RunTestCase(test_simt_udma_qp_put_block, test_simt_udma_qp_get_block, 1024, 2);
}

TEST(SimtUdmaQpTest, SimtUdmaQpBlockQp2OddLength)
{
    RunTestCase(test_simt_udma_qp_put_block, test_simt_udma_qp_get_block, 1001, 2);
}

// block mode: 2B minimum valid length (coop submit path; len=1 would make first_len=0, HW error 355)
TEST(SimtUdmaQpTest, SimtUdmaQpBlockQp2Slice2)
{
    RunTestCase(test_simt_udma_qp_put_block, test_simt_udma_qp_get_block, 2, 2);
}

// ===========================================================================
// Group 1b: Different symmetric-heap VA bases (regression for MR-base fix)
// Validates that SIMT UDMA uses peer's registered MR VA, not local peer-mapping.
// Would silently pass with identical VAs; the VA gap makes the addresses differ.
// ===========================================================================

TEST(SimtUdmaQpTest, SimtUdmaQpWarpQp2DifferentVaBases)
{
    RunTestCaseWithVaGap(test_simt_udma_qp_put_warp, test_simt_udma_qp_get_warp, 1024, 2);
}

TEST(SimtUdmaQpTest, SimtUdmaQpBlockQp2DifferentVaBases)
{
    RunTestCaseWithVaGap(test_simt_udma_qp_put_block, test_simt_udma_qp_get_block, 1024, 2);
}

TEST(SimtUdmaQpTest, SimtUdmaQpWarpQp4DifferentVaBases)
{
    RunTestCaseWithVaGap(test_simt_udma_qp_put_warp, test_simt_udma_qp_get_warp, 4096, 4);
}

// ===========================================================================
// Group 2: Multi-QP parallel (warp i -> QP i, single kernel launch)
// ===========================================================================

// warp mode: warp i binds QP i (primary recommended usage)
TEST(SimtUdmaQpTest, SimtUdmaQpMqpWarpQp4)
{
    RunMqpTestCase(test_simt_udma_qp_mqp_put_warp, test_simt_udma_qp_mqp_get_warp, 4096, 4);
}

TEST(SimtUdmaQpTest, SimtUdmaQpMqpWarpQp8)
{
    RunMqpTestCase(test_simt_udma_qp_mqp_put_warp, test_simt_udma_qp_mqp_get_warp, 4096, 8);
}

// Note: block multi-QP is excluded - block interface only warp 0 processes,
// cannot assign different QPs to different warps within a block.

// ===========================================================================
// Group 3: Block multi-QP via grid (block i drives QP i)
// Kernel grid = qp_count; each block drives QP blockIdx.x via the block API.
// This is the grid-level QP mapping used by the simt_udma_qp_demo block mode.
// ===========================================================================

TEST(SimtUdmaQpTest, SimtUdmaQpMqpBlockQp4)
{
    RunMqpTestCase(test_simt_udma_qp_mqp_put_block, test_simt_udma_qp_mqp_get_block, 4096, 4);
}

TEST(SimtUdmaQpTest, SimtUdmaQpMqpBlockQp8)
{
    RunMqpTestCase(test_simt_udma_qp_mqp_put_block, test_simt_udma_qp_mqp_get_block, 4096, 8);
}

#endif // defined(USE_SIMT)
