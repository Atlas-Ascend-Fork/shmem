/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_DEVICE_RDMA_HPP
#define ACLSHMEM_DEVICE_RDMA_HPP

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "shmemi_device_rdma.h"
#include "rdma_backends/rdma_device_backend_base.h"
#include "rdma_backends/rdma_device_backend_base.hpp"

// Decide Current RDMA Backend
#include "rdma_backends/rdma_device_backend_in_die.hpp"
#include "rdma_backends/rdma_device_backend_xscale.hpp"
#include "rdma_backends/rdma_device_backend_hns_1825.hpp"

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
#define ACLSHMEMI_K_RDMA_BACKEND (aclshmemi_rdma_backend_t::XSCALE)
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
#define ACLSHMEMI_K_RDMA_BACKEND (aclshmemi_rdma_backend_t::HNS_1825)
#else
#define ACLSHMEMI_K_RDMA_BACKEND (aclshmemi_rdma_backend_t::IN_DIE)
#endif

ACLSHMEM_DEVICE __gm__ aclshmemi_rdma_info* aclshmemi_qp_info_fetch()
{
    __gm__ aclshmemi_rdma_info* rdma_info = (__gm__ aclshmemi_rdma_info*)(aclshmemi_get_qp_info_address(0));
    return rdma_info;
}

ACLSHMEM_DEVICE void aclshmemi_rdma_debug_assert_not_self_send(uint32_t pe)
{
    if (pe == aclshmemi_get_my_pe()) {
        aclshmemi_kernel_abort("RDMA self send is invalid: pe=%u\n", pe);
    }
}

ACLSHMEM_DEVICE void aclshmemi_rdma_debug_assert_qp_params_valid(__gm__ aclshmemi_rdma_sq_ctx* sq_context)
{
    if (sq_context == nullptr || sq_context->buf_addr == 0U || sq_context->head_addr == 0U ||
        sq_context->tail_addr == 0U || sq_context->depth == 0U || sq_context->wqe_size == 0U ||
        (sq_context->depth & (sq_context->depth - 1U)) != 0U) {
        aclshmemi_kernel_abort("RDMA invalid SQ parameters\n");
    }
}

ACLSHMEM_DEVICE void aclshmemi_rdma_debug_ensure_sq_capacity(
    __gm__ aclshmemi_rdma_sq_ctx* sq_context, uint32_t pe, uint32_t qp_idx, uint32_t wqe_count)
{
    AscendC::GlobalTensor<uint32_t> sq_head;
    sq_head.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sq_context->head_addr));
    AscendC::GlobalTensor<uint32_t> sq_tail;
    sq_tail.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sq_context->tail_addr));
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(sq_context->head_addr), sizeof(uint32_t));
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(sq_context->tail_addr), sizeof(uint32_t));
    const uint32_t head = sq_head.GetValue(0);
    const uint32_t tail = sq_tail.GetValue(0);
    const uint32_t used = head - tail;
    if (wqe_count == 0U || used > sq_context->depth || wqe_count > sq_context->depth - used) {
        aclshmemi_kernel_abort(
            "RDMA SQ lacks capacity: pe=%u qp_idx=%u need=%u head=%u tail=%u depth=%u\n", pe, qp_idx, wqe_count, head,
            tail, sq_context->depth);
    }
}

ACLSHMEM_DEVICE void aclshmemi_rdma_debug_check_aggregate_batch_size(uint32_t pending_wqe_count, uint32_t depth)
{
    if (depth == 0U || pending_wqe_count == 0U || pending_wqe_count >= depth) {
        aclshmemi_kernel_abort("RDMA aggregate batch too large: pending=%u depth=%u\n", pending_wqe_count, depth);
    }
}

ACLSHMEM_DEVICE void aclshmemi_rdma_dump_sq_wqe(
    __gm__ aclshmemi_rdma_sq_ctx* sq_context, uint32_t posted_head, uint32_t wqe_size, uint32_t wqe_count)
{
    if (sq_context == nullptr || sq_context->buf_addr == 0U || sq_context->depth == 0U || wqe_size == 0U) {
        AscendC::printf("RDMA SQ WQE: invalid context\n");
        return;
    }
    const uint64_t slot = posted_head & (sq_context->depth - 1U);
    if (slot > (UINT64_MAX - sq_context->buf_addr) / wqe_size || wqe_count > UINT32_MAX / wqe_size) {
        AscendC::printf("RDMA SQ WQE: address overflow\n");
        return;
    }
    const uint64_t address = sq_context->buf_addr + slot * wqe_size;
    const uint32_t bytes = wqe_size * wqe_count;
    const uint32_t words = bytes > 32U ? 8U : (bytes + 3U) / 4U;
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(address), words * sizeof(uint32_t));
    AscendC::GlobalTensor<uint32_t> raw;
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    AscendC::printf(
        "RDMA SQ WQE: wqn=%u head=%u slot=%u size=%u count=%u\n", sq_context->wqn, posted_head,
        static_cast<uint32_t>(slot), wqe_size, wqe_count);
    AscendC::printf(
        "RDMA SQ WQE raw DW0-DW7: [0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x]\n", raw.GetValue(0),
        words > 1U ? raw.GetValue(1) : 0U, words > 2U ? raw.GetValue(2) : 0U, words > 3U ? raw.GetValue(3) : 0U,
        words > 4U ? raw.GetValue(4) : 0U, words > 5U ? raw.GetValue(5) : 0U, words > 6U ? raw.GetValue(6) : 0U,
        words > 7U ? raw.GetValue(7) : 0U);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id)
{
    aclshmemi_roce_write<T, ACLSHMEMI_K_RDMA_BACKEND>(
        dst, src, pe, qp_idx, message_len, ub_local64, ub_local32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_read(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id)
{
    aclshmemi_roce_read<T, ACLSHMEMI_K_RDMA_BACKEND>(
        dst, src, pe, qp_idx, message_len, ub_local64, ub_local32, sync_id);
}

ACLSHMEM_DEVICE void aclshmemi_roce_quiet(
    uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32,
    uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;

    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    ACLSHMEM_DEBUG_FUNC(aclshmemi_rdma_debug_assert_not_self_send, pe);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_rdma_debug_assert_qp_params_valid, sq_context);
    auto sq_pi_addr = sq_context->head_addr;
    AscendC::GlobalTensor<uint32_t> sq_head;
    sq_head.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sq_pi_addr));
    dcci_cachelines((__gm__ uint8_t*)sq_pi_addr, 8);
    uint32_t cur_head = sq_head.GetValue(0);
    uint32_t status =
        aclshmemi_roce_poll_cq<ACLSHMEMI_K_RDMA_BACKEND>(pe, qp_idx, cur_head, ub_local64, ub_local32, sync_id);
    if (status != 0U) {
        ACLSHMEM_DEBUG_FUNC(
            aclshmemi_kernel_printf, "RDMA quiet failed: pe=%u, qp_idx=%u, status=%u, backend=%u\n", pe, qp_idx, status,
            static_cast<uint32_t>(ACLSHMEMI_K_RDMA_BACKEND));
    }
}

ACLSHMEM_DEVICE void aclshmemi_roce_quiet(
    uint32_t pe, AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;
    for (uint32_t qp_idx = 0; qp_idx < qp_num; qp_idx++) {
        __gm__ aclshmemi_rdma_sq_ctx* sq_context =
            (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
        auto sq_pi_addr = sq_context->head_addr;
        AscendC::GlobalTensor<uint32_t> sq_head;
        sq_head.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sq_pi_addr));
        dcci_cachelines((__gm__ uint8_t*)sq_pi_addr, 8);
        uint32_t cur_head = sq_head.GetValue(0);
        uint32_t status =
            aclshmemi_roce_poll_cq<ACLSHMEMI_K_RDMA_BACKEND>(pe, qp_idx, cur_head, ub_local64, ub_local32, sync_id);
        if (status != 0U) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_printf, "RDMA quiet failed: pe=%u, qp_idx=%u, status=%u, backend=%u\n", pe, qp_idx,
                status, static_cast<uint32_t>(ACLSHMEMI_K_RDMA_BACKEND));
        }
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_team_sync(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id)
{
    int my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t*)sync_counter) + 1;
    aclshmemi_store((__gm__ int32_t*)sync_counter, count);
    dcci_cacheline((__gm__ uint8_t*)sync_counter);
    while (shift < size) {
        int pre_pe_in_team = (my_pe_in_team - shift + size) % size;
        int next_pe_in_team = (my_pe_in_team + shift) % size;
        int next_pe = start + next_pe_in_team * stride;

        aclshmemi_highlevel_signal_set(
            (__gm__ int32_t*)(sync_pool + my_pe_in_team), (__gm__ int32_t*)sync_counter, next_pe, buf, sync_id);
        aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t*)(sync_pool + pre_pe_in_team), count);

        shift *= SHIFT_MULTIPLIER;
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_barrier(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    int mype = device_state->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    if ((mype - start) % stride != 0) {
        return;
    }

    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == mype) {
            continue;
        }
        aclshmemi_roce_quiet(peer, ub_tensor_64, ub_tensor_32, sync_id);
    }

    aclshmemi_roce_team_sync(team, buf, sync_id);
}

template <typename T, bool IS_MASKED>
ACLSHMEM_DEVICE T aclshmemi_roce_amo_add(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t add_val, uint64_t boundary,
    AscendC::LocalTensor<uint64_t> ub_local64, AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id)
{
    if constexpr (
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825) {
        return aclshmemi_roce_atomic_fetch_and_add<T, IS_MASKED, ACLSHMEMI_K_RDMA_BACKEND>(
            dst, src, pe, qp_idx, add_val, boundary, ub_local64, ub_local32, sync_id);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "ROCE atomic add requires XSCALE or HNS_1825 backend.\n");
        return T(0);
    }
}

template <typename T, bool IS_MASKED>
ACLSHMEM_DEVICE T aclshmemi_roce_amo_cas(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t swap_val, uint64_t comp_val,
    uint64_t swap_mask, uint64_t comp_mask, AscendC::LocalTensor<uint64_t> ub_local64,
    AscendC::LocalTensor<uint32_t> ub_local32, uint32_t sync_id)
{
    if constexpr (
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825) {
        return aclshmemi_roce_atomic_compare_and_swap<T, IS_MASKED, ACLSHMEMI_K_RDMA_BACKEND>(
            dst, src, pe, qp_idx, swap_val, comp_val, swap_mask, comp_mask, ub_local64, ub_local32, sync_id);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "ROCE atomic cas requires XSCALE or HNS_1825 backend.\n");
        return T(0);
    }
}

ACLSHMEM_DEVICE __gm__ void* aclshmem_roce_ptr(__gm__ void* ptr, int pe)
{
    // Get Global State
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->p2p_device_heap_base[pe]) + offset;

    return reinterpret_cast<__gm__ void*>(remote_ptr);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read(
        (__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read(
        (__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
            ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_qp_get_nbi only supports XSCALE and HNS_1825 backends");
    auto ptr = aclshmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read(
        (__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, qp_idx, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32,
        sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
            ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_qp_get_nbi only supports XSCALE and HNS_1825 backends");
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read(
        (__gm__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)ptr, pe, qp_idx, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_get_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr(src, pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, pe, qp_idx, elem_size * sizeof(T), wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_get_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, pe, qp_idx, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32,
        wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_get_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst.GetPhyAddr(), pe, qp_idx, elem_size * sizeof(T), wqes_buf, sync_id,
        action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_get_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst.GetPhyAddr(), pe, qp_idx, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read(
        (__gm__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read(
        (__gm__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_get_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr(src, pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, pe, 0, elem_size * sizeof(T), wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_get_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, pe, 0, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32, wqes_buf,
        sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_get_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst.GetPhyAddr(), pe, 0, elem_size * sizeof(T), wqes_buf, sync_id,
        action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_get_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst.GetPhyAddr(), pe, 0, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    aclshmemi_roce_write(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    aclshmemi_roce_write(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
            ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_qp_put_nbi only supports XSCALE and HNS_1825 backends");
    auto ptr = aclshmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, qp_idx, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32,
        sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
            ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_qp_put_nbi only supports XSCALE and HNS_1825 backends");
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src.GetPhyAddr(), pe, qp_idx, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_put_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr(dst, pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, qp_idx, elem_size * sizeof(T), wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_put_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, qp_idx, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32,
        wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_put_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src.GetPhyAddr(), pe, qp_idx, elem_size * sizeof(T), wqes_buf, sync_id,
        action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE || sizeof(T) == 0,
        "aclshmemx_roce_qp_put_nbi with aggregate stage/commit requires XSCALE backend");
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src.GetPhyAddr(), pe, qp_idx, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    aclshmemi_roce_write(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src.GetPhyAddr(), pe, 0, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    aclshmemi_roce_write(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src.GetPhyAddr(), pe, 0, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_put_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src.GetPhyAddr(), pe, 0, elem_size * sizeof(T), wqes_buf, sync_id,
        action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_put_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf.GetPhyAddr() + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src.GetPhyAddr(), pe, 0, elem_size * sizeof(T), ub_tensor_64,
        ub_tensor_32, wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_put_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr(dst, pe);
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_stage_rma_wqe_ub<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T), wqes_buf, sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND != aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_put_nbi aggregate stage/commit requires a non-HNS_1825 backend");
    auto ptr = aclshmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    __ubuf__ uint8_t* wqes_buf = (__ubuf__ uint8_t*)buf + UB_ALIGN_SIZE * 2;
    aclshmemi_roce_commit_rma_wqes<ACLSHMEMI_K_RDMA_BACKEND, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T), ub_tensor_64, ub_tensor_32, wqes_buf,
        sync_id, action);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_quiet(uint32_t pe, __ubuf__ T* buf, uint32_t sync_id)
{
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    aclshmemi_roce_quiet(pe, ub_tensor_64, ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_quiet(uint32_t pe, uint32_t qp_idx, __ubuf__ T* buf, uint32_t sync_id)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
            ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_qp_quiet only supports XSCALE and HNS_1825 backends");
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    aclshmemi_roce_quiet(pe, qp_idx, ub_tensor_64, ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_quiet(
    uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<T> buf, uint32_t sync_id)
{
    static_assert(
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
            ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825 || sizeof(T) == 0,
        "aclshmemx_roce_qp_quiet only supports XSCALE and HNS_1825 backends");
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    aclshmemi_roce_quiet(pe, qp_idx, ub_tensor_64, ub_tensor_32, sync_id);
}

ACLSHMEM_DEVICE int aclshmemx_roce_team_sync(aclshmemx_team_t* team)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    aclshmemi_roce_team_sync(team, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE int aclshmemx_roce_team_sync(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id)
{
    aclshmemi_roce_team_sync(team, buf, sync_id);
    return 0;
}

ACLSHMEM_DEVICE int aclshmemx_roce_barrier(aclshmemx_team_t* team)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    aclshmemi_roce_barrier(team, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE int aclshmemx_roce_barrier(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id)
{
    aclshmemi_roce_barrier(team, buf, sync_id);
    return 0;
}
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_sync_all(__ubuf__ T* buf, uint32_t sync_id)
{
    aclshmemx_team_t* world_team = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD];
    aclshmemx_roce_team_sync(world_team, buf, sync_id);
}

ACLSHMEM_DEVICE void aclshmemx_roce_sync_all()
{
    aclshmemx_team_t* world_team = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD];
    aclshmemx_roce_team_sync(world_team);
}

ACLSHMEM_DEVICE void aclshmemx_roce_barrier_all()
{
    aclshmemx_team_t* world_team = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD];
    aclshmemx_roce_barrier(world_team);
}
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_barrier_all(__ubuf__ T* buf, uint32_t sync_id)
{
    aclshmemx_team_t* world_team = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD];
    aclshmemx_roce_barrier(world_team, buf, sync_id);
}

ACLSHMEM_DEVICE uint64_t aclshmemi_roce_get_atomic_fetch_addr(uint32_t pe, uint32_t qp_idx)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* qp_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    auto amo_addr = qp_context->amo_addr;
    return amo_addr;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_roce_get_atomic_fetch_data(uint32_t pe, uint32_t qp_idx)
{
    auto amo_addr = aclshmemi_roce_get_atomic_fetch_addr(pe, qp_idx);
    AscendC::GlobalTensor<T> fetch_tensor;
    fetch_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(amo_addr));
    dcci_cachelines((__gm__ uint8_t*)amo_addr, sizeof(T));
    T value = fetch_tensor.GetValue(0);
    if constexpr (sizeof(T) == 4 && ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE) {
        // When the XSCALE backend performs a fetch or swap operation on 4B size data, it will get data in little-endian
        // order, which needs to be converted
        uint32_t fetch_bytes = value;
        return (T)aclshmemi_htobe32(fetch_bytes);
    } else {
        return value;
    }
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch(__gm__ T* src, int32_t pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(src, pe);
    aclshmemi_roce_amo_add<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, 0, 0, ub_tensor_64, ub_tensor_32, sync_id);
    aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return aclshmemi_roce_get_atomic_fetch_data<T>(pe, 0);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_set(__gm__ T* dst, T value, int32_t pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_cas<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, value, 0, UINT64_MAX, 0, ub_tensor_64, ub_tensor_32,
        sync_id);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    if constexpr (sizeof(T) == 4) {
        aclshmemi_roce_amo_cas<T, true>(
            reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, (uint64_t)cond, UINT64_MAX,
            UINT64_MAX, ub_tensor_64, ub_tensor_32, sync_id);
    } else {
        aclshmemi_roce_amo_cas<T, false>(
            reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, (uint64_t)cond, UINT64_MAX,
            UINT64_MAX, ub_tensor_64, ub_tensor_32, sync_id);
    }
    aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return aclshmemi_roce_get_atomic_fetch_data<T>(pe, 0);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_swap(__gm__ T* dst, T value, int32_t pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_cas<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, 0, UINT64_MAX, 0, ub_tensor_64,
        ub_tensor_32, sync_id);
    aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return aclshmemi_roce_get_atomic_fetch_data<T>(pe, 0);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_add(__gm__ T* dst, T value, int32_t pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_add<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, value, 0, ub_tensor_64, ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_inc(__gm__ T* dst, int32_t pe)
{
    aclshmemx_roce_atomic_add(dst, (T)1, pe);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    if constexpr (sizeof(T) == 4) {
        aclshmemi_roce_amo_add<T, true>(
            reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, 0, ub_tensor_64, ub_tensor_32,
            sync_id);
    } else {
        aclshmemi_roce_amo_add<T, false>(
            reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, 0, ub_tensor_64, ub_tensor_32,
            sync_id);
    }
    aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return aclshmemi_roce_get_atomic_fetch_data<T>(pe, 0);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_inc(__gm__ T* dst, int32_t pe)
{
    return aclshmemx_roce_atomic_fetch_add(dst, (T)1, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_and(__gm__ T* dst, T value, int32_t pe)
{
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
            std::is_same<T, uint64_t>::value,
        "aclshmemx_roce_atomic_and only supports int32, uint32, int64, uint64 types");
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    uint64_t swap_mask = ~(uint64_t)value;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_cas<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, 0, swap_mask, 0, ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_or(__gm__ T* dst, T value, int32_t pe)
{
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
            std::is_same<T, uint64_t>::value,
        "aclshmemx_roce_atomic_or only supports int32, uint32, int64, uint64 types");
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_cas<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, 0, (uint64_t)value, 0, ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_xor(__gm__ T* dst, T value, int32_t pe)
{
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
            std::is_same<T, uint64_t>::value,
        "aclshmemx_roce_atomic_xor only supports int32, uint32, int64, uint64 types");
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_add<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, UINT64_MAX, ub_tensor_64,
        ub_tensor_32, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe)
{
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
            std::is_same<T, uint64_t>::value,
        "aclshmemx_roce_atomic_fetch_and only supports int32, uint32, int64, uint64 types");
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    uint64_t swap_mask = ~(uint64_t)value;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_cas<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, 0, swap_mask, 0, ub_tensor_64,
        ub_tensor_32, sync_id);
    aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return aclshmemi_roce_get_atomic_fetch_data<T>(pe, 0);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe)
{
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
            std::is_same<T, uint64_t>::value,
        "aclshmemx_roce_atomic_fetch_or only supports int32, uint32, int64, uint64 types");
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_cas<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, 0, (uint64_t)value, 0, ub_tensor_64,
        ub_tensor_32, sync_id);
    aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return aclshmemi_roce_get_atomic_fetch_data<T>(pe, 0);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe)
{
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value ||
            std::is_same<T, uint64_t>::value,
        "aclshmemx_roce_atomic_fetch_xor only supports int32, uint32, int64, uint64 types");
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto remote_ptr = aclshmem_ptr(dst, pe);
    aclshmemi_roce_amo_add<T, true>(
        reinterpret_cast<__gm__ T*>(remote_ptr), nullptr, pe, 0, (uint64_t)value, UINT64_MAX, ub_tensor_64,
        ub_tensor_32, sync_id);
    aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);
    return aclshmemi_roce_get_atomic_fetch_data<T>(pe, 0);
}

#endif // ACLSHMEM_DEVICE_RDMA_HPP
