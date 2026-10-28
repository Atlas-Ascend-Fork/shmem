/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * @file simt_udma_qp_kernel.cpp
 * @brief Device implementation of SIMT UDMA QP-specific unit tests.
 *
 * Test scenarios:
 *
 * Scenario 1 - Single put/get + quiet (warp/block):
 *   Covers correctness and length boundaries.
 *   Signature: (stream, dst, src, elem_size, pe, qp_idx)
 *   put: dst=remote symmetric, src=local; get: dst=local, src=remote symmetric.
 *
 * Scenario 2 - Multi-QP parallel (warp i binds QP i, warp mode):
 *   Validates legal concurrent multi-driver under single-driver constraint.
 *   Signature: (stream, dst, src, slice, qp_count, pe)
 *   VF launches qp_count*32 threads; warp_id < qp_count each drives QP warp_id.
 *
 * Scenario 3 - Block multi-QP via grid (block i drives QP i):
 *   Kernel grid = qp_count; each block drives QP blockIdx.x via the block API.
 *   Tests the grid-level QP mapping that the simt_udma_qp_demo block mode uses.
 *   Signature: (stream, dst, src, slice, qp_count, pe)
 */

#if defined(USE_SIMT)

#include "kernel_operator.h"
#include "shmem.h"
#include "utils/debug/asc_printf.h"

// ============================================================================
// Scenario 1: Single put_nbi+quiet / get_nbi+quiet (warp/block)
// Signature: (void* stream, uint8_t* dst, uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx)
// ============================================================================

// Warning: DO NOT REMOVE the printf invocation in simt_call_##FUNCNAME.
// Doing so may cause mysterious compiler errors (same workaround as simt_rma kernels).
#define SIMT_UDMA_QP_TEST(FUNCNAME, API_CALL, VF_BOUNDS)                                                          \
    __simt_vf__ __launch_bounds__(VF_BOUNDS) inline void simt_call_##FUNCNAME(                                    \
        __gm__ uint8_t* dst, __gm__ uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx)                  \
    {                                                                                                             \
        if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                \
            printf("Testing " #FUNCNAME "\n");                                                                    \
        }                                                                                                         \
        asc_syncthreads();                                                                                        \
        API_CALL;                                                                                                 \
    }                                                                                                             \
    __global__ __vector__ void kernel_call_##FUNCNAME(                                                            \
        __gm__ uint8_t* dst, __gm__ uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx)                  \
    {                                                                                                             \
        asc_vf_call<simt_call_##FUNCNAME>(dim3(VF_BOUNDS), dst, src, elem_size, pe, qp_idx);                      \
    }                                                                                                             \
    void test_##FUNCNAME(void* stream, uint8_t* dst, uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx) \
    {                                                                                                             \
        kernel_call_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe, qp_idx);                                \
    }

// warp mode: VF launched with 32 threads (one warp), block-collective call.
SIMT_UDMA_QP_TEST(simt_udma_qp_put_warp,
                  simt::aclshmemx_udma_qp_put_nbi_warp<uint8_t>(dst, src, static_cast<uint32_t>(elem_size), pe, qp_idx);
                  simt::aclshmemx_udma_qp_quiet_warp(pe, qp_idx), 32)

SIMT_UDMA_QP_TEST(simt_udma_qp_get_warp,
                  simt::aclshmemx_udma_qp_get_nbi_warp<uint8_t>(dst, src, static_cast<uint32_t>(elem_size), pe, qp_idx);
                  simt::aclshmemx_udma_qp_quiet_warp(pe, qp_idx), 32)

// block mode: VF launched with 128 threads (4 warps), only warp 0 processes, rest pass barriers.
SIMT_UDMA_QP_TEST(simt_udma_qp_put_block, simt::aclshmemx_udma_qp_put_nbi_block<uint8_t>(
                                              dst, src, static_cast<uint32_t>(elem_size), pe, qp_idx);
                  simt::aclshmemx_udma_qp_quiet_block(pe, qp_idx), 128)

SIMT_UDMA_QP_TEST(simt_udma_qp_get_block, simt::aclshmemx_udma_qp_get_nbi_block<uint8_t>(
                                              dst, src, static_cast<uint32_t>(elem_size), pe, qp_idx);
                  simt::aclshmemx_udma_qp_quiet_block(pe, qp_idx), 128)

#undef SIMT_UDMA_QP_TEST

// ============================================================================
// Scenario 2: Multi-QP parallel (warp i binds QP i, warp mode)
// Validates: legal concurrent multi-driver under single-driver constraint.
// Signature: (void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)
// VF launches qp_count*32 threads; warp_id < qp_count each drives QP warp_id.
// All threads participate in asc_syncthreads (block-collective constraint).
// ============================================================================

constexpr uint32_t SIMT_UDMA_QP_MAX_VF_THREADS = ACLSHMEM_MAX_QP_NUM * 32U; // one warp per QP, warpSize(32)

#define SIMT_UDMA_QP_MQP_TEST(FUNCNAME, API_CALL)                                                                 \
    __simt_vf__ __launch_bounds__(SIMT_UDMA_QP_MAX_VF_THREADS) inline void simt_call_##FUNCNAME(                  \
        __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)                  \
    {                                                                                                             \
        if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                \
            printf("Testing " #FUNCNAME "\n");                                                                    \
        }                                                                                                         \
        asc_syncthreads();                                                                                        \
        const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;      \
        const uint32_t warp_id = tid / warpSize;                                                                  \
        if (warp_id < qp_count) {                                                                                 \
            API_CALL;                                                                                             \
        }                                                                                                         \
    }                                                                                                             \
    __global__ __vector__ void kernel_call_##FUNCNAME(                                                            \
        __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)                  \
    {                                                                                                             \
        asc_vf_call<simt_call_##FUNCNAME>(dim3(qp_count * 32U), dst, src, slice, qp_count, pe);                   \
    }                                                                                                             \
    void test_##FUNCNAME(void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe) \
    {                                                                                                             \
        kernel_call_##FUNCNAME<<<1, 0, stream>>>(dst, src, slice, qp_count, pe);                                  \
    }

SIMT_UDMA_QP_MQP_TEST(simt_udma_qp_mqp_put_warp, simt::aclshmemx_udma_qp_put_nbi_warp<uint8_t>(
                                                     dst + static_cast<uint64_t>(warp_id) * slice,
                                                     src + static_cast<uint64_t>(warp_id) * slice, slice, pe, warp_id);
                      simt::aclshmemx_udma_qp_quiet_warp(pe, warp_id))

SIMT_UDMA_QP_MQP_TEST(simt_udma_qp_mqp_get_warp, simt::aclshmemx_udma_qp_get_nbi_warp<uint8_t>(
                                                     dst + static_cast<uint64_t>(warp_id) * slice,
                                                     src + static_cast<uint64_t>(warp_id) * slice, slice, pe, warp_id);
                      simt::aclshmemx_udma_qp_quiet_warp(pe, warp_id))

#undef SIMT_UDMA_QP_MQP_TEST

// ============================================================================
// Scenario 3: Block multi-QP via grid (block i drives QP i)
// Kernel grid = qp_count; each block drives QP blockIdx.x via the block API.
// This tests the grid-level QP mapping that the simt_udma_qp_demo block mode
// uses. All 128 threads per block participate (block-collective constraint).
// Signature: (void* stream, uint8_t* dst, uint8_t* src, uint32_t slice,
//             uint32_t qp_count, int32_t pe)
// ============================================================================

__simt_vf__ __launch_bounds__(128) inline void simt_call_simt_udma_qp_mqp_put_block(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)
{
    (void)qp_count;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Testing simt_udma_qp_mqp_put_block\n");
    }
    asc_syncthreads();
    const uint32_t qp_idx = blockIdx.x;
    simt::aclshmemx_udma_qp_put_nbi_block<uint8_t>(
        dst + static_cast<uint64_t>(qp_idx) * slice, src + static_cast<uint64_t>(qp_idx) * slice, slice, pe, qp_idx);
    simt::aclshmemx_udma_qp_quiet_block(pe, qp_idx);
}

__global__ __vector__ void kernel_call_simt_udma_qp_mqp_put_block(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)
{
    asc_vf_call<simt_call_simt_udma_qp_mqp_put_block>(dim3(128), dst, src, slice, qp_count, pe);
}

void test_simt_udma_qp_mqp_put_block(
    void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)
{
    kernel_call_simt_udma_qp_mqp_put_block<<<qp_count, 0, stream>>>(dst, src, slice, qp_count, pe);
}

__simt_vf__ __launch_bounds__(128) inline void simt_call_simt_udma_qp_mqp_get_block(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)
{
    (void)qp_count;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Testing simt_udma_qp_mqp_get_block\n");
    }
    asc_syncthreads();
    const uint32_t qp_idx = blockIdx.x;
    simt::aclshmemx_udma_qp_get_nbi_block<uint8_t>(
        dst + static_cast<uint64_t>(qp_idx) * slice, src + static_cast<uint64_t>(qp_idx) * slice, slice, pe, qp_idx);
    simt::aclshmemx_udma_qp_quiet_block(pe, qp_idx);
}

__global__ __vector__ void kernel_call_simt_udma_qp_mqp_get_block(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)
{
    asc_vf_call<simt_call_simt_udma_qp_mqp_get_block>(dim3(128), dst, src, slice, qp_count, pe);
}

void test_simt_udma_qp_mqp_get_block(
    void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe)
{
    kernel_call_simt_udma_qp_mqp_get_block<<<qp_count, 0, stream>>>(dst, src, slice, qp_count, pe);
}

#endif // defined(USE_SIMT)
