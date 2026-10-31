/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP
#define ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP

#include "kernel_operator.h"
#include "rdma_device_backend_base.h"

/**
 * @brief Write data from UB local buffer to GM global memory with synchronization
 *
 * This function writes data from local UB (Uniform Buffer) to specified GM (Global Memory) address
 * with complete hardware synchronization sequence to ensure data write completion.
 *
 * @tparam T Data type template parameter
 * @param addr GM target address where data will be written
 * @param ub_local Local UB buffer containing data to write; the caller sets address_.bufferAddr
 *                 (and optionally address_.dataLen) before the call
 * @param size Data size in bytes to write
 * @param sync_id Synchronization event ID for hardware event synchronization
 *
 * @note Synchronization sequence:
 *   1. S_MTE3: Scalar to MTE3 synchronization before data copy
 *   2. DataCopyPad: Execute data copy from UB to GM
 *   3. PIPE_MTE3: MTE3 pipeline barrier to ensure all MTE3 operations complete
 *   4. MTE3_S: MTE3 to Scalar synchronization after data copy
 *
 * @note Usage: Used in RDMA operations to update critical data structures in global memory
 *       (e.g., CQ tail pointer, SQ head pointer, doorbell registers) with strict synchronization.
 *       Backend-agnostic: shared by all RDMA backends (bulk WQE copy, doorbells, index writeback).
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write_ub_to_gm_with_sync(
    uint64_t addr, AscendC::LocalTensor<T>& ub_local, uint32_t size, uint32_t sync_id)
{
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<T> tmp_global_tensor;
    tmp_global_tensor.SetGlobalBuffer((__gm__ T*)addr);
    AscendC::DataCopyExtParams copy_params{1, size, 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(tmp_global_tensor, ub_local, copy_params);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

ACLSHMEM_DEVICE __ubuf__ aclshmemi_roce_aggregate_cache_t* aclshmemi_roce_aggregate_cache(__ubuf__ uint8_t* buf)
{
    return (__ubuf__ aclshmemi_roce_aggregate_cache_t*)(buf - UB_ALIGN_SIZE * 2);
}

template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_aggregate_wqe_size(__ubuf__ aclshmemi_roce_aggregate_cache_t* cache)
{
    if constexpr (B == aclshmemi_rdma_backend_t::IN_DIE) {
        return cache->backend.in_die.sq_wqe_size;
    } else {
        return cache->backend.xscale.sq_wqe_size;
    }
}

template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_prepare_aggregate_commit(
    __ubuf__ aclshmemi_roce_aggregate_cache_t* cache, uint32_t next_head)
{
    (void)cache;
    (void)next_head;
}

template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_fill_aggregate_cache(
    __ubuf__ aclshmemi_roce_aggregate_cache_t* cache, __gm__ aclshmemi_rdma_info* rdma_info, uint32_t pe,
    uint32_t qp_idx)
{
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr +
                                        ((uint64_t)pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));

    cache->sq_base_addr = sq_context->buf_addr;
    cache->sq_head_addr = sq_context->head_addr;
    cache->sq_tail_addr = sq_context->tail_addr;
    cache->sq_db_addr = sq_context->db_addr;
    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(rdma_info->mem_ptr + sizeof(aclshmemi_rdma_mem_info) * pe);
    cache->remote_mr_base = remote_mem_info->addr;
    cache->remote_rkey = remote_mem_info->rkey;
    cache->sq_depth = sq_context->depth;
    if constexpr (B == aclshmemi_rdma_backend_t::HNS_1825) {
        cache->backend.hns_1825.sq_wqn = sq_context->wqn;
        cache->backend.hns_1825.sq_mtu_shift = sq_context->mtu_shift;
        cache->backend.hns_1825.sq_db_cos = sq_context->db_cos;
    } else if constexpr (B == aclshmemi_rdma_backend_t::IN_DIE) {
        cache->backend.in_die.sq_wqn = sq_context->wqn;
        cache->backend.in_die.sq_wqe_size = sq_context->wqe_size;
        cache->backend.in_die.sq_sl = sq_context->sl;
    } else {
        cache->backend.xscale.sq_wqn = sq_context->wqn;
        cache->backend.xscale.sq_wqe_size = sq_context->wqe_size;
    }

    dcci_cachelines((__gm__ uint8_t*)cache->sq_head_addr, 8);
    cache->base_head = *(__gm__ uint32_t*)(cache->sq_head_addr);

    auto mem_info_table = rdma_info->mem_ptr;
    __gm__ aclshmemi_rdma_mem_info* local_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * aclshmemi_get_my_pe());
    cache->local_lkey = local_mem_info->lkey;
}

template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_stage_rma_wqe_ub(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* buf, uint32_t sync_id, aclshmemx_defer_t action)
{
    static_assert(
        B == aclshmemi_rdma_backend_t::IN_DIE || B == aclshmemi_rdma_backend_t::XSCALE ||
            B == aclshmemi_rdma_backend_t::HNS_1825,
        "RDMA aggregation requires XSCALE, IN_DIE or HNS_1825 backend");
    auto* cache = aclshmemi_roce_aggregate_cache(buf);
    uint32_t pending_count = action.state.pending_count;
    if (pending_count == 0) {
        aclshmemi_roce_fill_aggregate_cache<B>(cache, aclshmemi_qp_info_fetch(), pe, qp_idx);
    }
    const uint32_t wqe_size = aclshmemi_roce_aggregate_wqe_size<B>(cache);
    remote_addr = aclshmemi_remote_mr_addr(remote_addr, aclshmemi_get_state()->heap_base, cache->remote_mr_base);

    // Build one WQE directly in the UB staging buffer using the backend stride.
    aclshmemi_roce_fill_wqe_ub<B, OP_CODE>(
        buf + static_cast<uint64_t>(pending_count) * wqe_size, cache->base_head + pending_count,
        static_cast<uint32_t>(message_len), cache->remote_rkey, remote_addr, cache->local_lkey, local_addr,
        cache->sq_depth, false);

    action.state.pending_count = pending_count + 1;
}

template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_commit_rma_wqes(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, __ubuf__ uint8_t* buf,
    uint32_t sync_id, aclshmemx_submit_t action)
{
    constexpr uint32_t POLL_CQ_THRESHOLD = B == aclshmemi_rdma_backend_t::XSCALE ? 16 : 10;
    static_assert(
        B == aclshmemi_rdma_backend_t::IN_DIE || B == aclshmemi_rdma_backend_t::XSCALE ||
            B == aclshmemi_rdma_backend_t::HNS_1825,
        "RDMA aggregation requires XSCALE, IN_DIE or HNS_1825 backend");
    auto* cache = aclshmemi_roce_aggregate_cache(buf);
    uint32_t pending_count = action.state.pending_count;
    if (pending_count == 0) {
        aclshmemi_roce_fill_aggregate_cache<B>(cache, aclshmemi_qp_info_fetch(), pe, qp_idx);
    }
    const uint32_t wqe_size = aclshmemi_roce_aggregate_wqe_size<B>(cache);

    uint32_t cur_head = cache->base_head;
    uint32_t total_wqe_count = pending_count + 1;
    uint32_t depth = cache->sq_depth;
    ACLSHMEM_DEBUG_FUNC(aclshmemi_rdma_debug_check_aggregate_batch_size, total_wqe_count, depth);

    dcci_cachelines((__gm__ uint8_t*)cache->sq_tail_addr, 8);
    uint32_t cur_tail = *(__gm__ uint32_t*)cache->sq_tail_addr;
    if (cur_head + total_wqe_count - cur_tail + POLL_CQ_THRESHOLD >= depth) {
        uint32_t ret = aclshmemi_roce_poll_cq<B>(pe, qp_idx, cur_head, ub_local64, ub_local32, sync_id);
        if (ret != 0) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort,
                "RDMA aggregate commit poll CQ failed: pe=%u, qp_idx=%u, cur_head=%u, cur_tail=%u, depth=%u, "
                "total_wqe_count=%u, ret=%u\n",
                pe, qp_idx, cur_head, cur_tail, depth, total_wqe_count, ret);
            trap();
            return;
        }
        // poll_cq uses ub_local32/ub_local64 as workspace, which alias the cache region (buf-64 ~ buf).
        // Refill the cache to restore sq_base_addr etc. overwritten during polling.
        aclshmemi_roce_fill_aggregate_cache<B>(cache, aclshmemi_qp_info_fetch(), pe, qp_idx);
        cache->base_head = cur_head;
    }

    remote_addr = aclshmemi_remote_mr_addr(remote_addr, aclshmemi_get_state()->heap_base, cache->remote_mr_base);

    // Assemble the commit call's own WQE at position pending_count (0-indexed).
    // Earlier WQEs were already assembled in stage.
    aclshmemi_roce_fill_wqe_ub<B, OP_CODE>(
        buf + static_cast<uint64_t>(pending_count) * wqe_size, cur_head + pending_count,
        static_cast<uint32_t>(message_len), cache->remote_rkey, remote_addr, cache->local_lkey, local_addr, depth,
        true);

    uint32_t next_head = cur_head + total_wqe_count;
    aclshmemi_roce_prepare_aggregate_commit<B>(cache, next_head);

    // Bulk copy all WQEs from UB to GM SQ, splitting at the ring boundary when needed.
    // Each copy is fully synchronized (S_MTE3 -> DataCopyPad ->
    // PIPE_MTE3 barrier -> MTE3_S), so the doorbell below only rings after every WQE landed in GM.
    AscendC::LocalTensor<uint8_t> ub_wqe;
    ub_wqe.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    uint32_t ring_offset = cur_head % depth;
    uint32_t first_count = depth - ring_offset;
    uint32_t first_bytes = (total_wqe_count < first_count ? total_wqe_count : first_count) * wqe_size;
    ub_wqe.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_wqe.address_.dataLen = first_bytes;
    aclshmemi_roce_write_ub_to_gm_with_sync(
        cache->sq_base_addr + static_cast<uint64_t>(ring_offset) * wqe_size, ub_wqe, first_bytes, sync_id);
    if (first_count < total_wqe_count) {
        uint32_t second_bytes = (total_wqe_count - first_count) * wqe_size;
        ub_wqe.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + first_bytes;
        ub_wqe.address_.dataLen = second_bytes;
        aclshmemi_roce_write_ub_to_gm_with_sync(cache->sq_base_addr, ub_wqe, second_bytes, sync_id);
    }

    aclshmemi_roce_ring_sq_doorbell_from_cache<B>(cache, pe, qp_idx, next_head, ub_local64, ub_local32, sync_id);
    action.state.pending_count = 0;
}

template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_fetch_and_add(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t add_val, uint64_t boundary,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    auto mem_info_table = rdma_info->mem_ptr;
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);

    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = aclshmemi_remote_mr_addr(dst, aclshmemi_get_state()->heap_base, remote_mem_info->addr);
    wr.local_addr = (__gm__ uint8_t*)sq_context->amo_addr;
    wr.message_len = 0;
    wr.atomic.masked_common.swap_add_data = add_val;
    wr.atomic.masked_common.swap_add_mask = boundary;
    wr.rkey = remote_mem_info->rkey;
    wr.lkey = sq_context->amo_lkey;

    aclshmemi_backend_traits<B>::template atomic_op_traits<T, IS_MASKED>::template post_send<
        aclshmemi_rdma_atomic_op_t::OP_ATOMIC_FA>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);

    return T(0);
}

template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_compare_and_swap(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t swap_val, uint64_t comp_val,
    uint64_t swap_mask, uint64_t comp_mask, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    __gm__ aclshmemi_rdma_info* rdma_info = aclshmemi_qp_info_fetch();
    auto mem_info_table = rdma_info->mem_ptr;
    uint32_t qp_num = rdma_info->qp_num;
    __gm__ aclshmemi_rdma_sq_ctx* sq_context =
        (__gm__ aclshmemi_rdma_sq_ctx*)(rdma_info->sq_ptr + (pe * qp_num + qp_idx) * sizeof(aclshmemi_rdma_sq_ctx));
    __gm__ aclshmemi_rdma_mem_info* remote_mem_info =
        (__gm__ aclshmemi_rdma_mem_info*)(mem_info_table + sizeof(aclshmemi_rdma_mem_info) * pe);

    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = aclshmemi_remote_mr_addr(dst, aclshmemi_get_state()->heap_base, remote_mem_info->addr);
    wr.local_addr = (__gm__ uint8_t*)sq_context->amo_addr;
    wr.message_len = 0;
    wr.atomic.masked_common.swap_add_data = swap_val;
    wr.atomic.masked_common.compare_data = comp_val;
    wr.atomic.masked_common.swap_add_mask = swap_mask;
    wr.atomic.masked_common.compare_mask = comp_mask;
    wr.rkey = remote_mem_info->rkey;
    wr.lkey = sq_context->amo_lkey;

    aclshmemi_backend_traits<B>::template atomic_op_traits<T, IS_MASKED>::template post_send<
        aclshmemi_rdma_atomic_op_t::OP_ATOMIC_CAS>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);

    return T(0);
}

#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP
