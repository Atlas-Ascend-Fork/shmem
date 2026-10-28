/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_SIMT_UDMA_HPP
#define SHMEM_DEVICE_SIMT_UDMA_HPP

/*
 * SIMT UDMA QP-specific data-plane implementation.
 *
 * The SIMT execution space lacks the scalar doorbell write (st_dev) that the AICore (SIMD) path
 * uses. Submission therefore uses the DWQE (Direct WQE) mechanism: the assembled 128 B WQE
 * (two WQEBBs) is copied to the 128 B window immediately below the doorbell register
 * (db_addr - 0x80). The producer index (PI) is encoded in WQE word0[15:0].
 *
 * Thread-scope interfaces are intentionally not provided. The thread path would have to write the
 * 128 B DWQE window with asc_stwt (write-through) stores at 8 B or 16 B granularity, but a single
 * thread cannot complete that write in one shot: partial or reordered stores may let the NIC
 * observe the producer index before the WQE body lands. Only the cooperative warp/block path is
 * exposed, where 16 lanes issue concurrent asc_stcg (NON_CACHEABLE) stores that coalesce into a
 * single 128 B transaction and bypass the SIMT Data Cache, so no dcci is needed.
 *
 * Warp/block mode: 16 lanes cooperatively assemble the WQE (one u64 per lane), then each lane
 * writes its 8 B to both the SQ ring and the DWQE window. A per-thread asc_threadfence orders
 * each lane's SQ store before its window store; no block-level barrier is required between
 * assembly and the two writes.
 *
 * Concurrency model -- single-driver constraint: at any moment, one (pe, qp_idx) may be driven
 * by a single thread group executing one collective call (warp / block mode). Use distinct
 * qp_idx values to run multiple drivers in parallel.
 *
 * Cross-programming-model boundary: the SIMD and SIMT UDMA data planes share one
 * aclshmemi_udma_queue_state_t (sq_head / sq_tail / cqe_cnt) and one SQ ring per (pe, qp_idx),
 * but differ in WQE BB count (SIMD normal R/W = 1 BB, SIMT DWQE = 2 BB) and in sq_tail recovery
 * (aclshmemi_udma_unwrap_wq_tail in shmem_device_udma.hpp vs the ring-distance accumulation in
 * aclshmemi_simt_udma_poll_cq). The Host-side transport manager is programming-model agnostic,
 * so driving the same (pe, qp_idx) from both a SIMD kernel and a SIMT VF within one process is
 * unsupported; use distinct qp_idx values per programming model.
 */

// Include paths: this file may be compiled by library, example, and unit-test targets whose
// common roots are src/device/ and src/device_simt/, so cross-directory headers are resolved
// from those two roots (same convention as the MTE SIMT implementation).
#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_udma.h" // UDMA internal POD types / opcode, shared with SIMD
#include "../host_device/shmemi_host_device_constant.h"
#include "shmemi_device_simt_common.hpp" // thread_group_t utilities and SIMT device state
#include "utils/shmemi_simt_kernel_debug.h"
#include "simt_api/device_functions.h" // asc_stcg / asc_threadfence / asc_ldcg

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define ACLSHMEMI_SIMT_UDMA_SUPPORTED 1
#else
#define ACLSHMEMI_SIMT_UDMA_SUPPORTED 0
#endif

// Wrap ACLSHMEM_RELAY_SUPPORT into a compile-time constant for use with if constexpr.
#if defined(ACLSHMEM_RELAY_SUPPORT)
#define ACLSHMEM_RELAY_SUPPORTED 1
#else
#define ACLSHMEM_RELAY_SUPPORTED 0
#endif

namespace simt {

// ---------------- Constants ----------------

// The DWQE window sits 128 B below the doorbell register.
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_DWQE_OFFSET = 0x80U;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_WQE_BB_CNT = 2U;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_MAX_RETRY = 1000000U;                       // CQ poll limit, matches SIMD
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_CQE_STATUS_SHIFT = 8U;                      // status<<8 | substatus packing
constexpr uint64_t ACLSHMEMI_SIMT_UDMA_MAX_MSG_BYTES = 256ULL * 1024ULL * 1024ULL; // API contract upper bound
// Lower bound: the DWQE path always splits the payload across two SGEs (first_len = len >> 1).
// A 1 B transfer would produce a zero-length SGE #1, which the NIC rejects. The minimum
// legal transfer is therefore 2 B.
constexpr uint64_t ACLSHMEMI_SIMT_UDMA_MIN_MSG_BYTES = 2ULL;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_REMOTE_JETTY_TYPE = 1U;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_FLAG_ODR_MASK = 0x7U;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_FLAG_FENCE_SHIFT = 3U;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_FLAG_SE_SHIFT = 4U;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_FLAG_CQE_SHIFT = 5U;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_CQE_ENTRY_IDX_MOD = 1U << 16;
constexpr uint32_t ACLSHMEMI_SIMT_UDMA_CREDIT_GUARD = 10U; // SQ capacity reserve, matches SIMD

static_assert(
    shm::UDMA_SQ_BASKBLK_CNT <= ACLSHMEMI_SIMT_UDMA_CQE_ENTRY_IDX_MOD,
    "UDMA CQE entry_idx is 16-bit and must cover the SQ BB count");

// ---------------- Common utilities (build-gate independent) ----------------

// Linear thread index within the Thread Block.
__simt_callee__ inline size_t aclshmemi_simt_udma_linear_tid()
{
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

#if ACLSHMEMI_SIMT_UDMA_SUPPORTED && !defined(ACLSHMEM_RELAY_SUPPORT)

// ---------------- SQE assembly ----------------

// Owner bit: flips each lap of the SQ ring (depth must be a power of two -- UDMA_SQ_BASKBLK_CNT
// satisfies this; debug_check_sq_depth verifies it at runtime).
__simt_callee__ inline uint32_t aclshmemi_simt_udma_owner_bit(uint32_t cur_head)
{
    return ((cur_head & shm::UDMA_SQ_BASKBLK_CNT) == 0U) ? 1U : 0U;
}

// Build the SQE flag byte. Matches the SIMD aclshmemi_udma_build_flag. SIMT has no defer/submit,
// so cqe == 0 would leave completions un-reaped and SQ space never released -- rejected at
// compile time.
template <const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline constexpr uint32_t aclshmemi_simt_udma_build_flag()
{
    static_assert(CONFIG.cqe == 1U, "SIMT UDMA requires CONFIG.cqe == 1");
    static_assert(CONFIG.se == 0U, "UDMA op config se is not supported yet");
    static_assert(CONFIG.fence <= 1U, "UDMA op config fence must be 0 or 1");
    static_assert(CONFIG.odr <= ACLSHMEMI_SIMT_UDMA_FLAG_ODR_MASK, "UDMA op config odr must be in range 0..7");
    return CONFIG.odr | (CONFIG.fence << ACLSHMEMI_SIMT_UDMA_FLAG_FENCE_SHIFT) |
           (CONFIG.se << ACLSHMEMI_SIMT_UDMA_FLAG_SE_SHIFT) | (CONFIG.cqe << ACLSHMEMI_SIMT_UDMA_FLAG_CQE_SHIFT);
}

// Assemble the SQE header (dw0~dw3, i.e. the first 16 B of the WQE) into two u64 values:
//   w01 = dw0 | dw1 << 32,  w23 = dw2 | dw3 << 32.
// Field layout matches the SIMD fill_sqe_base_ctx. Two additions for the DWQE path:
//   word0[15:0] = PI (next_head low 16 bits, NIC uses it to publish the WQE),
//   word0[31]   = owner (lap parity, same convention as asc-comm).
// The SQE header assembly is shared by the cooperative path: the 128 B WQE is built in
// lane-local registers.
template <aclshmemi_udma_opcode_t OP_CODE, const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemi_simt_udma_build_sqe_header(
    __gm__ aclshmemi_ubmem_info_t* mem_info, uint32_t cur_head, uint32_t next_head, uint64_t& w01, uint64_t& w23)
{
    // SIMT UDMA does not support WRITE_WITH_REDUCE: the reduce attributes (udf_type /
    // reduce_data_type / reduce_opcode) and the amo_addr SGE va are not assembled by this
    // cooperative 16-lane path. Reject at compile time to avoid silent data corruption.
    static_assert(
        OP_CODE != aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE,
        "SIMT UDMA does not support UDMA_OP_WRITE_WITH_REDUCE");
    constexpr uint32_t flag = aclshmemi_simt_udma_build_flag<CONFIG>();
    constexpr uint32_t opcode = static_cast<uint32_t>(OP_CODE);
    uint32_t dw0 = (next_head & 0xFFFFU) | (flag << 16U) | (static_cast<uint32_t>(mem_info->token_value_valid) << 28U) |
                   (ACLSHMEMI_SIMT_UDMA_REMOTE_JETTY_TYPE << 29U) | (aclshmemi_simt_udma_owner_bit(cur_head) << 31U);
    uint32_t dw1 = (mem_info->target_hint & 0xFFU) | (opcode << 8U);
    uint32_t dw2 = (mem_info->tpn & 0xFFFFFFU) | (2U << 24U); // sge_num = 2 (split to fill the window)
    uint32_t dw3 = mem_info->tid & 0xFFFFFU;
    w01 = static_cast<uint64_t>(dw0) | (static_cast<uint64_t>(dw1) << 32U);
    w23 = static_cast<uint64_t>(dw2) | (static_cast<uint64_t>(dw3) << 32U);
}

// ---------------- Metadata and context fetch ----------------

// Fetch the device-side UDMA metadata (QP count and SQ/CQ/mem context table base addresses).
// The data source is the same as the SIMD version (GLOBAL META slot). On Ascend950, a
// register-indirect GM load through a runtime-computed address triggers a vector core
// exception, so a compile-time constant absolute address is dereferenced directly.
__simt_callee__ inline __gm__ aclshmemi_aiv_udma_info_t* aclshmemi_simt_udma_info_fetch()
{
    constexpr uint64_t UDMA_INFO_ADDR =
        SMEM_SHM_DEVICE_META_ADDR + SMEM_SHM_DEVICE_GLOBAL_META_SIZE + SMEM_SHM_DEVICE_META_UDMA_INFO_OFFSET;
    return reinterpret_cast<__gm__ aclshmemi_aiv_udma_info_t*>(
        static_cast<uintptr_t>(*reinterpret_cast<__gm__ uint64_t*>(UDMA_INFO_ADDR)));
}

// In direct mode, slot == pe. QP-specific APIs do not support relay builds.
__simt_callee__ inline __gm__ aclshmemi_udma_wq_ctx_t* aclshmemi_simt_udma_get_wq_ctx(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t slot, uint32_t qp_idx)
{
    uint32_t qp_num = udma_info->qp_num;
    return reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(
        udma_info->direct.sq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_wq_ctx_t));
}

__simt_callee__ inline __gm__ aclshmemi_udma_cq_ctx_t* aclshmemi_simt_udma_get_cq_ctx(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t slot, uint32_t qp_idx)
{
    uint32_t qp_num = udma_info->qp_num;
    return reinterpret_cast<__gm__ aclshmemi_udma_cq_ctx_t*>(
        udma_info->direct.scq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_cq_ctx_t));
}

__simt_callee__ inline __gm__ aclshmemi_ubmem_info_t* aclshmemi_simt_udma_get_mem_info(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t slot, uint32_t qp_idx)
{
    uint32_t qp_num = udma_info->qp_num;
    return reinterpret_cast<__gm__ aclshmemi_ubmem_info_t*>(
        udma_info->direct.mem_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_ubmem_info_t));
}

// The mutable queue counters (sq_head / sq_tail / cqe_cnt / cq_head / cq_tail) now live in a
// standalone aclshmemi_udma_queue_state_t buffer referenced by state_addr. Slots sharing a jetty
// share one instance, so the wq and cq of the same (slot, qp_idx) resolve to the same state.
__simt_callee__ inline __gm__ aclshmemi_udma_queue_state_t* aclshmemi_simt_udma_get_wq_state(
    __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx)
{
    return reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(wq_ctx->state_addr);
}

// ---------------- Validation (L1 debug builds) ----------------

// Validate qp_idx range, target PE, and message length.
__simt_callee__ inline void aclshmemi_simt_udma_debug_check_params(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, int pe, uint32_t qp_idx, uint64_t message_len)
{
    if (qp_idx >= udma_info->qp_num) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: qp_idx=%u out of range, qp_num=%u\n", qp_idx, udma_info->qp_num);
    }
    __gm__ aclshmem_device_host_state_simt_t* state = simt::aclshmemi_get_state();
    if (pe < 0 || pe >= state->npes || pe == state->mype) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: invalid pe=%d (npes=%d mype=%d)\n", pe, state->npes, state->mype);
    }
    if (message_len < ACLSHMEMI_SIMT_UDMA_MIN_MSG_BYTES || message_len > ACLSHMEMI_SIMT_UDMA_MAX_MSG_BYTES) {
        simt::aclshmemi_kernel_abort(
            "SIMT UDMA: message_len=%llu out of range [2B, 256MB]\n", static_cast<unsigned long long>(message_len));
    }
}

// Verify wqe_size == 64: two WQEBBs (128 B) fill the DWQE window only when the basic block is 64 B.
__simt_callee__ inline void aclshmemi_simt_udma_debug_check_wqe_size(uint32_t wqe_size)
{
    if (wqe_size != 64U) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: unsupported wqe_size=%u (expect 64)\n", wqe_size);
    }
}

// Verify SQ depth matches UDMA_SQ_BASKBLK_CNT; the owner-bit protocol relies on the depth
// being a power of two.
__simt_callee__ inline void aclshmemi_simt_udma_debug_check_sq_depth(__gm__ aclshmemi_udma_wq_ctx_t* wq_ctx)
{
    if (wq_ctx->depth != shm::UDMA_SQ_BASKBLK_CNT) {
        simt::aclshmemi_kernel_abort(
            "SIMT UDMA: sq depth=%u != UDMA_SQ_BASKBLK_CNT=%u\n", wq_ctx->depth, shm::UDMA_SQ_BASKBLK_CNT);
    }
}

// Verify remote jetty type matches the expected value.
__simt_callee__ inline void aclshmemi_simt_udma_debug_check_remote_jetty_type(__gm__ aclshmemi_ubmem_info_t* mem_info)
{
    uint32_t rmt_jetty_type = mem_info->rmt_jetty_type;
    if (rmt_jetty_type != ACLSHMEMI_SIMT_UDMA_REMOTE_JETTY_TYPE) {
        simt::aclshmemi_kernel_abort(
            "SIMT UDMA: remote jetty type=%u != %u\n", rmt_jetty_type, ACLSHMEMI_SIMT_UDMA_REMOTE_JETTY_TYPE);
    }
}

// Validate the symmetric address and remote MR base for the remote MR conversion,
// mirroring the SIMD-side aclshmemi_debug_assert_remote_mr_addr.
__simt_callee__ inline void aclshmemi_simt_udma_debug_assert_remote_mr_addr(
    uint64_t local_ptr, uint64_t heap_base, uint64_t heap_size, uint64_t remote_mr_base)
{
    if (local_ptr < heap_base || local_ptr - heap_base >= heap_size ||
        remote_mr_base > UINT64_MAX - (local_ptr - heap_base)) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: invalid symmetric address for remote MR conversion\n");
    }
}

// Check SQ capacity (credit semantics match the SIMD version; GUARD prevents wrap ambiguity).
// head / tail live in the shared queue state, so the caller resolves it once and passes it in.
__simt_callee__ inline bool aclshmemi_simt_udma_sq_has_credit(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t wqe_bb_cnt)
{
    uint32_t used_after_post = state->sq_head + wqe_bb_cnt - state->sq_tail;
    return used_after_post + ACLSHMEMI_SIMT_UDMA_CREDIT_GUARD < shm::UDMA_SQ_BASKBLK_CNT;
}

// Best-effort single-driver conflict detection: after accounting, read back head; if another
// driver advanced it concurrently, abort.
__simt_callee__ inline void aclshmemi_simt_udma_debug_check_driver(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t next_head, uint32_t slot, uint32_t qp_idx)
{
    if (state->sq_head != next_head) {
        simt::aclshmemi_kernel_abort(
            "SIMT UDMA: concurrent drivers on (slot=%u qp=%u), head moved by others\n", slot, qp_idx);
    }
}

// Validate qp_idx range for the quiet path (post_send_coop already checks via debug_check_params).
__simt_callee__ inline void aclshmemi_simt_udma_debug_check_qp_index(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t qp_idx)
{
    if (qp_idx >= udma_info->qp_num) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: qp_idx=%u out of range, qp_num=%u\n", qp_idx, udma_info->qp_num);
    }
}

// The WQE is cooperatively assembled by 16 lanes. Each driving warp must therefore have at least
// 16 active lanes; a shorter final warp would leave SGE #2 and the upper DWQE window unwritten
// while lane 0 still advances the head.
__simt_callee__ inline void aclshmemi_simt_udma_debug_check_active_lanes(bool active, size_t tid)
{
    if (!active) {
        return;
    }
    uint32_t warp_base = static_cast<uint32_t>(tid - (tid % warpSize));
    uint32_t total_threads = static_cast<uint32_t>(blockDim.x * blockDim.y * blockDim.z);
    uint32_t lanes_in_warp = (warp_base + warpSize <= total_threads) ? warpSize : (total_threads - warp_base);
    if (lanes_in_warp < 16U) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: driving warp has %u active lanes, need at least 16\n", lanes_in_warp);
    }
}

// Post accounting: advance head (NIC and subsequent submissions use it to locate the ring slot)
// and increment cqe_cnt (quiet waits for this many completions).
template <const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemi_simt_udma_post_accounting(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t next_head)
{
    state->sq_head = next_head;
    if constexpr (CONFIG.cqe != 0U) {
        state->cqe_cnt = state->cqe_cnt + 1U;
    }
}

// ---------------- WQE / CQE dump (matches SIMD aclshmemi_dump_* style) ----------------
// SIMT uses simt::aclshmemi_kernel_printf instead of AscendC::printf.
// WQ/CQ context (dump_wq_ctx / dump_cq_ctx) is host-written before the kernel starts, so
// members are read directly through the Data Cache. WQE/CQE entries are written to GM by the
// NIC or by NON_CACHEABLE asc_stcg stores, so the SIMT Data Cache may hold a stale copy on a
// ring reuse; those entries are loaded via aclshmemi_simt_udma_ldcg below, and NIC-written
// entries (CQEs) are preceded by asc_threadfence for the cross-domain visibility order.

// Load a GM struct into a local copy via asc_ldcg, bypassing the SIMT Data Cache (reads from
// L2; L2-GM coherence is hardware-maintained). asc_ldcg is available in all CANN versions that
// support SIMT, unlike asc_dcci_single (CANN >= 9.1.0-beta.2), so this avoids version gates.
// For data written asynchronously by the NIC, the caller must issue asc_threadfence before
// this load (see poll_cq / dump_cqe): it is a memory fence that establishes a release/acquire
// visibility order, not a cache invalidation.
template <typename T>
__simt_callee__ inline T aclshmemi_simt_udma_ldcg(__gm__ T* addr)
{
    T local;
    static_assert(sizeof(T) % sizeof(uint32_t) == 0, "T size must be a multiple of 4 bytes");
    __gm__ uint32_t* src = reinterpret_cast<__gm__ uint32_t*>(addr);
    uint32_t* dst = reinterpret_cast<uint32_t*>(&local);
    constexpr uint32_t n = sizeof(T) / sizeof(uint32_t);
#pragma unroll
    for (uint32_t i = 0; i < n; ++i) {
        dst[i] = asc_ldcg(src + i);
    }
    return local;
}

__simt_callee__ inline void aclshmemi_simt_udma_dump_sqe_ctx(
    uint32_t slot, uint32_t qp_idx, aclshmemi_sqe_ctx_t* sqe_ctx)
{
    if (sqe_ctx == nullptr) {
        simt::aclshmemi_kernel_printf("WQE[slot=%u qp=%u]: nullptr pointer\n", slot, qp_idx);
        return;
    }
    // Bit-field members must be materialized into local uint32_t before binding to variadic args.
    uint32_t sqe_bb_idx = sqe_ctx->sqe_bb_idx;
    uint32_t flag = sqe_ctx->flag;
    uint32_t rsv0 = sqe_ctx->rsv0;
    uint32_t nf = sqe_ctx->nf;
    uint32_t token_en = sqe_ctx->token_en;
    uint32_t rmt_jetty_type = sqe_ctx->rmt_jetty_type;
    simt::aclshmemi_kernel_printf(
        "WQE[slot=%u qp=%u]: sqe_bb_idx: %x flag: %x rsv0: %x nf: %x token_en: %x rmt_jetty_type: %x\n", slot, qp_idx,
        sqe_bb_idx, flag, rsv0, nf, token_en, rmt_jetty_type);
    uint32_t owner = sqe_ctx->owner;
    uint32_t target_hint = sqe_ctx->target_hint;
    uint32_t opcode = sqe_ctx->opcode;
    uint32_t rsv1 = sqe_ctx->rsv1;
    uint32_t inline_msg_len = sqe_ctx->inline_msg_len;
    uint32_t tp_id = sqe_ctx->tp_id;
    simt::aclshmemi_kernel_printf(
        "WQE[slot=%u qp=%u]: owner: %x target_hint: %x opcode: %x rsv1: %x inline_msg_len: %x tp_id: %x\n", slot,
        qp_idx, owner, target_hint, opcode, rsv1, inline_msg_len, tp_id);
    uint32_t sge_num = sqe_ctx->sge_num;
    uint32_t rmt_jetty_or_seg_id = sqe_ctx->rmt_jetty_or_seg_id;
    uint32_t rsv2 = sqe_ctx->rsv2;
    simt::aclshmemi_kernel_printf(
        "WQE[slot=%u qp=%u]: sge_num: %x rmt_jetty_or_seg_id: %x rsv2: %x\n", slot, qp_idx, sge_num,
        rmt_jetty_or_seg_id, rsv2);
    simt::aclshmemi_kernel_printf(
        "WQE[slot=%u qp=%u]: rmt_eid: %x, %x\n", slot, qp_idx, static_cast<uint32_t>(sqe_ctx->rmt_eid_l),
        static_cast<uint32_t>(sqe_ctx->rmt_eid_h));
    uint32_t udf_type = sqe_ctx->udf_type;
    uint32_t reduce_data_type = sqe_ctx->reduce_data_type;
    uint32_t reduce_opcode = sqe_ctx->reduce_opcode;
    simt::aclshmemi_kernel_printf(
        "WQE[slot=%u qp=%u]: rmt_token_value: %x udf_type: %x reduce_data_type: %x reduce_opcode: %x\n", slot, qp_idx,
        sqe_ctx->rmt_token_value, udf_type, reduce_data_type, reduce_opcode);
    simt::aclshmemi_kernel_printf(
        "WQE[slot=%u qp=%u]: rmt_addr_l_or_token_id: %x rmt_addr_h_or_token_value: %x\n", slot, qp_idx,
        sqe_ctx->rmt_addr_l_or_token_id, sqe_ctx->rmt_addr_h_or_token_value);
}

__simt_callee__ inline void aclshmemi_simt_udma_dump_sge_ctx(
    uint32_t slot, uint32_t qp_idx, __gm__ aclshmemi_sge_ctx_t* sge_ctx, uint32_t sge_num)
{
    if (sge_ctx == nullptr) {
        simt::aclshmemi_kernel_printf("SGE[slot=%u qp=%u]: nullptr pointer\n", slot, qp_idx);
        return;
    }
    for (uint32_t i = 0; i < sge_num; ++i) {
        aclshmemi_sge_ctx_t sge = aclshmemi_simt_udma_ldcg(&sge_ctx[i]);
        simt::aclshmemi_kernel_printf(
            "SGE[slot=%u qp=%u]: sge idx: %u, va: %llx sge_len: %u\n", slot, qp_idx, i,
            static_cast<unsigned long long>(sge.va), sge.len);
    }
}

// Locate the WQE in the SQ ring, then dump the SQE header and its SGE array.
__simt_callee__ inline void aclshmemi_simt_udma_dump_sq_wqe(
    uint32_t slot, uint32_t qp_idx, __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size)
{
    __gm__ uint8_t* wqe_addr = reinterpret_cast<__gm__ uint8_t*>(
        qp_ctx_entry->buf_addr + static_cast<uint64_t>(wqe_size) * (cur_head % shm::UDMA_SQ_BASKBLK_CNT));
    if (wqe_addr == nullptr) {
        simt::aclshmemi_kernel_printf("WQE[slot=%u qp=%u]: nullptr pointer\n", slot, qp_idx);
        return;
    }
    // The producer's asc_stcg + asc_threadfence in post_send already published this thread's
    // own WQE write, so no extra fence is needed. sge_num is needed before dumping SGEs.
    __gm__ aclshmemi_sqe_ctx_t* sqe_gm = reinterpret_cast<__gm__ aclshmemi_sqe_ctx_t*>(wqe_addr);
    aclshmemi_sqe_ctx_t sqe = aclshmemi_simt_udma_ldcg(sqe_gm);
    aclshmemi_simt_udma_dump_sqe_ctx(slot, qp_idx, &sqe);
    __gm__ aclshmemi_sge_ctx_t* sge_ctx =
        reinterpret_cast<__gm__ aclshmemi_sge_ctx_t*>(wqe_addr + sizeof(aclshmemi_sqe_ctx_t));
    aclshmemi_simt_udma_dump_sge_ctx(slot, qp_idx, sge_ctx, sqe.sge_num);
}

__simt_callee__ inline void aclshmemi_simt_udma_dump_cqe(
    uint32_t slot, uint32_t qp_idx, __gm__ aclshmemi_jfc_cqe_ctx_t* cqe_addr)
{
    if (cqe_addr == nullptr) {
        simt::aclshmemi_kernel_printf("CQE[slot=%u qp=%u]: nullptr pointer\n", slot, qp_idx);
        return;
    }
    // CQEs are written asynchronously by the NIC (cross-domain producer): precede the load
    // with asc_threadfence for the visibility order (see aclshmemi_simt_udma_ldcg).
    asc_threadfence();
    aclshmemi_jfc_cqe_ctx_t cqe = aclshmemi_simt_udma_ldcg(cqe_addr);
    // Bit-field members are materialized into local uint32_t before binding to variadic args.
    uint32_t s_r = cqe.s_r;
    uint32_t is_jetty = cqe.is_jetty;
    uint32_t owner = cqe.owner;
    uint32_t inline_en = cqe.inline_en;
    uint32_t opcode = cqe.opcode;
    uint32_t fd = cqe.fd;
    uint32_t substatus = cqe.substatus;
    uint32_t status = cqe.status;
    simt::aclshmemi_kernel_printf(
        "CQE[slot=%u qp=%u] DW0 - s_r: %u, is_jetty: %u, owner: %u, inline_en: %u, opcode: %u, "
        "fd: %u, substatus: %u, status: %u\n",
        slot, qp_idx, s_r, is_jetty, owner, inline_en, opcode, fd, substatus, status);
    uint32_t entry_idx = cqe.entry_idx;
    uint32_t local_num_l = cqe.local_num_l;
    simt::aclshmemi_kernel_printf(
        "CQE[slot=%u qp=%u] DW1 - entry_idx: %u, local_num_l: %u\n", slot, qp_idx, entry_idx, local_num_l);
    uint32_t local_num_h = cqe.local_num_h;
    uint32_t rmt_idx = cqe.rmt_idx;
    simt::aclshmemi_kernel_printf(
        "CQE[slot=%u qp=%u] DW2 - local_num_h: %u, rmt_idx: %u\n", slot, qp_idx, local_num_h, rmt_idx);
    uint32_t tpn = cqe.tpn;
    simt::aclshmemi_kernel_printf("CQE[slot=%u qp=%u] DW3 - tpn: %u\n", slot, qp_idx, tpn);
    simt::aclshmemi_kernel_printf("CQE[slot=%u qp=%u] DW4 - byte_cnt: %u\n", slot, qp_idx, cqe.byte_cnt);
    simt::aclshmemi_kernel_printf(
        "CQE[slot=%u qp=%u] DW5-DW6 - userData: 0x%x%x\n", slot, qp_idx, cqe.user_data_h, cqe.user_data_l);
    simt::aclshmemi_kernel_printf(
        "CQE[slot=%u qp=%u] DW7-DW10 - rmt_eid: [0x%x, 0x%x, 0x%x, 0x%x]\n", slot, qp_idx, cqe.rmt_eid[0],
        cqe.rmt_eid[1], cqe.rmt_eid[2], cqe.rmt_eid[3]);
    simt::aclshmemi_kernel_printf(
        "CQE[slot=%u qp=%u] DW11-DW12 - data: 0x%x%x\n", slot, qp_idx, cqe.data_h, cqe.data_l);
    simt::aclshmemi_kernel_printf(
        "CQE[slot=%u qp=%u] DW13-DW15 - inline_data: [0x%x, 0x%x, 0x%x]\n", slot, qp_idx, cqe.inline_data[0],
        cqe.inline_data[1], cqe.inline_data[2]);
}

__simt_callee__ inline void aclshmemi_simt_udma_dump_wq_ctx(
    uint32_t slot, uint32_t qp_idx, __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx, __gm__ aclshmemi_udma_queue_state_t* state)
{
    if (wq_ctx == nullptr || state == nullptr) {
        simt::aclshmemi_kernel_printf("WQ[slot=%u qp=%u]: nullptr pointer\n", slot, qp_idx);
        return;
    }
    uint32_t wqn = wq_ctx->wqn;
    uint32_t wqe_size = wq_ctx->wqe_size;
    uint32_t depth = wq_ctx->depth;
    uint32_t head = state->sq_head;
    uint32_t tail = state->sq_tail;
    uint32_t cqe_cnt = state->cqe_cnt;
    uint32_t sl = wq_ctx->sl;
    simt::aclshmemi_kernel_printf(
        "WQ[slot=%u qp=%u] DW0 - wqn: %x, wqe_size: %x, depth: %x, head: %x, tail: %x, cqe_cnt: %x, sl: %x\n", slot,
        qp_idx, wqn, wqe_size, depth, head, tail, cqe_cnt, sl);
    int32_t db_mode = static_cast<int32_t>(wq_ctx->db_mode);
    simt::aclshmemi_kernel_printf(
        "WQ[slot=%u qp=%u] DW1 - buf_addr: %llx, db_addr: %llx, amo_addr: %llx, db_mode: %d\n", slot, qp_idx,
        static_cast<unsigned long long>(wq_ctx->buf_addr), static_cast<unsigned long long>(wq_ctx->db_addr),
        static_cast<unsigned long long>(wq_ctx->amo_addr), db_mode);
}

__simt_callee__ inline void aclshmemi_simt_udma_dump_cq_ctx(
    uint32_t slot, uint32_t qp_idx, __gm__ aclshmemi_udma_cq_ctx_t* cq_ctx, __gm__ aclshmemi_udma_queue_state_t* state)
{
    if (cq_ctx == nullptr || state == nullptr) {
        simt::aclshmemi_kernel_printf("CQ[slot=%u qp=%u]: nullptr pointer\n", slot, qp_idx);
        return;
    }
    uint32_t cqn = cq_ctx->cqn;
    uint32_t cqe_size = cq_ctx->cqe_size;
    uint32_t depth = cq_ctx->depth;
    uint32_t head = state->cq_head;
    uint32_t tail = state->cq_tail;
    simt::aclshmemi_kernel_printf(
        "CQ[slot=%u qp=%u] DW0 - cqn: %x, cqe_size: %x, depth: %x, head: %x, tail: %x\n", slot, qp_idx, cqn, cqe_size,
        depth, head, tail);
    int32_t db_mode = static_cast<int32_t>(cq_ctx->db_mode);
    simt::aclshmemi_kernel_printf(
        "CQ[slot=%u qp=%u] DW1 - buf_addr: %llx, db_addr: %llx, db_mode: %d\n", slot, qp_idx,
        static_cast<unsigned long long>(cq_ctx->buf_addr), static_cast<unsigned long long>(cq_ctx->db_addr), db_mode);
}

// ---------------- CQ polling (quiet) ----------------

// Poll until the CQ tail reaches expect. On completion, write back the CQ tail, ring the CQ
// doorbell (plain u32 store, matching asc-comm PollCq), and restore the monotonically
// increasing SQ tail from the CQE entry_idx.
// Returns 0 on success; timeout or error status terminates via kernel_abort (L0 diagnostic).
__simt_callee__ inline uint32_t aclshmemi_simt_udma_poll_cq(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t slot, uint32_t qp_idx, uint32_t expect)
{
    __gm__ aclshmemi_udma_cq_ctx_t* cq_ctx = aclshmemi_simt_udma_get_cq_ctx(udma_info, slot, qp_idx);
    __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx = aclshmemi_simt_udma_get_wq_ctx(udma_info, slot, qp_idx);
    __gm__ aclshmemi_udma_queue_state_t* state = aclshmemi_simt_udma_get_wq_state(wq_ctx);
    uint32_t cur_tail = state->cq_tail;
    // Compare cursors modularly: a cqe_cnt that has wrapped to zero while a request is still
    // outstanding must not be mistaken for an empty queue.
    if (cur_tail == expect) {
        return 0;
    }
    // SQ consumer tail (monotonic, in 64 B WQEBB units). Each CQE's entry_idx is the last WQEBB
    // index of the completed WQE; sqTail advances by (ring-distance + 1) per CQE. sqDepth is a
    // power of two (UDMA_SQ_BASKBLK_CNT), so sqMask isolates the physical slot. The general
    // ring-distance form is used instead of hardcoding +2 so the accumulation stays robust to
    // entry_idx anomalies; for the in-order 2-BB WQE it always evaluates to +2.
    //
    // SIMD (aclshmemi_udma_unwrap_wq_tail) reads the same entry_idx as a next-free counter,
    // differing from the last-WQEBB reading here by one BB; either reading leaves sq_tail within
    // +/-1 BB of the true next-free index, absorbed by the credit guard on both planes.
    uint32_t sq_tail = state->sq_tail;
    const uint32_t sq_depth = shm::UDMA_SQ_BASKBLK_CNT;
    const uint32_t sq_mask = sq_depth - 1U;
    bool has_polled = false;

    while (cur_tail != expect) {
        // Read raw 32-bit words (the compiler generates tighter code than through the bit-field
        // struct) via asc_ldcg. The NIC is a cross-domain producer, so asc_threadfence between
        // polls establishes the visibility order needed to observe its GM write.
        __gm__ uint32_t* cqe = reinterpret_cast<__gm__ uint32_t*>(
            cq_ctx->buf_addr + cq_ctx->cqe_size * (cur_tail & (shm::UDMA_CQ_DEPTH_DEFAULT - 1U)));
        // Owner-bit flip protocol: expected owner = consumption-lap parity; CQE DW0 bit 2.
        uint32_t valid_owner = (cur_tail / shm::UDMA_CQ_DEPTH_DEFAULT) & 1U;
        uint32_t retry = 0;
        uint32_t cqe0 = asc_ldcg(cqe);
        while (((cqe0 >> 2U) & 1U) == valid_owner && retry < ACLSHMEMI_SIMT_UDMA_MAX_RETRY) {
            asc_threadfence();
            cqe0 = asc_ldcg(cqe);
            retry++;
        }
        if (retry >= ACLSHMEMI_SIMT_UDMA_MAX_RETRY) {
            uint32_t cqe_owner = (cqe0 >> 2U) & 1U;
            ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_dump_wq_ctx, slot, qp_idx, wq_ctx, state);
            ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_dump_cq_ctx, slot, qp_idx, cq_ctx, state);
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort,
                "SIMT UDMA cq timeout: slot=%u qp=%u cur_tail=%u expect=%u valid_owner=%u cqe_owner=%u cq_depth=%u\n",
                slot, qp_idx, cur_tail, expect, valid_owner, cqe_owner,
                static_cast<unsigned>(shm::UDMA_CQ_DEPTH_DEFAULT));
            __trap();
            return 0xFF;
        }
        uint32_t status = (cqe0 >> 24U) & 0xFFU;
        uint32_t substatus = (cqe0 >> 16U) & 0xFFU;
        uint32_t cqe_entry_idx = asc_ldcg(cqe + 1) & 0xFFFFU;
        if (status != 0 || substatus != 0) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_dump_wq_ctx, slot, qp_idx, wq_ctx, state);
            ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_dump_cq_ctx, slot, qp_idx, cq_ctx, state);
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_simt_udma_dump_cqe, slot, qp_idx, reinterpret_cast<__gm__ aclshmemi_jfc_cqe_ctx_t*>(cqe));
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort,
                "SIMT UDMA cqe error: slot=%u qp=%u cur_tail=%u expect=%u valid_owner=%u "
                "status=%u substatus=%u entry_idx=%u cq_depth=%u\n",
                slot, qp_idx, cur_tail, expect, valid_owner, status, substatus, cqe_entry_idx,
                static_cast<unsigned>(shm::UDMA_CQ_DEPTH_DEFAULT));
            __trap();
            return (status << ACLSHMEMI_SIMT_UDMA_CQE_STATUS_SHIFT) | substatus;
        }
        // entry_idx is the last WQEBB index of the completed WQE. Advance sqTail by the ring
        // distance from the current tail slot to the completed slot, plus one for the completed
        // BB itself. For an in-order 2-BB WQE this always yields +2.
        uint32_t sq_tail_slot = sq_tail & sq_mask;
        uint32_t completed_slot = cqe_entry_idx & sq_mask;
        sq_tail += ((completed_slot + sq_depth - sq_tail_slot) & sq_mask) + 1U;
        has_polled = true;
        cur_tail++;
    }
    if (!has_polled) {
        return 0;
    }

    // Epilogue: write back CQ tail, ring CQ doorbell (consumer index, 24-bit), write back SQ tail.
    state->cq_tail = cur_tail;
    *reinterpret_cast<__gm__ uint32_t*>(cq_ctx->db_addr) = cur_tail & 0xFFFFFFU;
    state->sq_tail = sq_tail;
    return 0;
}

// Quiet: wait for all requests submitted to (pe, qp_idx) to complete (cqe_cnt tracks the count).
__simt_callee__ inline void aclshmemi_simt_udma_quiet_impl(int pe, uint32_t qp_idx)
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_simt_udma_info_fetch();
    ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_debug_check_qp_index, udma_info, qp_idx);
    const uint32_t slot = static_cast<uint32_t>(pe);
    __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx = aclshmemi_simt_udma_get_wq_ctx(udma_info, slot, qp_idx);
    __gm__ aclshmemi_udma_queue_state_t* state = aclshmemi_simt_udma_get_wq_state(wq_ctx);
    (void)aclshmemi_simt_udma_poll_cq(udma_info, slot, qp_idx, state->cqe_cnt);
}

// ---------------- Warp / block submission (cooperative model) ----------------

// 16-lane cooperative submission (warp / block scope). In block mode only warp 0 is active; the
// remaining warps have no work. Submission needs no asc_syncthreads because per-thread
// asc_threadfence orders each lane's SQ and DWQE writes; the block quiet variant adds the
// barrier separately.
//
// remote_symm_addr is the symmetric address, translated to the peer's registered MR VA
// (mem_info->addr + offset) inside the body. The public wrappers place the remote side first
// for both put and get, matching the SIMD convention.
template <aclshmemi_udma_opcode_t OP_CODE, thread_group_t SCOPE, const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemi_simt_udma_post_send_coop(
    __gm__ uint8_t* remote_symm_addr, __gm__ uint8_t* local_addr, int pe, uint32_t qp_idx, uint64_t message_len)
{
    size_t tid = aclshmemi_simt_udma_linear_tid();
    uint32_t lane = static_cast<uint32_t>(tid % warpSize);
    // In block mode only warp 0 processes; in warp mode every warp drives its own QP.
    bool active = (SCOPE == ACLSHMEMI_THREADGROUP_WARP) || (tid < warpSize);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_debug_check_active_lanes, active, tid);

    // Uniform GM reads for the credit check.
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_simt_udma_info_fetch();
    ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_debug_check_params, udma_info, pe, qp_idx, message_len);
    uint32_t slot = static_cast<uint32_t>(pe);
    __gm__ aclshmemi_udma_wq_ctx_t* wq_ctx = aclshmemi_simt_udma_get_wq_ctx(udma_info, slot, qp_idx);
    // The mutable counters (head / tail / cqe_cnt) live in the shared queue state; resolve it once.
    __gm__ aclshmemi_udma_queue_state_t* state = aclshmemi_simt_udma_get_wq_state(wq_ctx);
    uint32_t wqe_size = wq_ctx->wqe_size;
    ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_debug_check_wqe_size, wqe_size);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_debug_check_sq_depth, wq_ctx);

    // Reclaim SQ credit before submission. A single poll_cq call blocks until all outstanding
    // CQEs are consumed, advancing sq_tail so the next credit check succeeds; no loop is needed.
    //
    // active is uniform within each warp (WARP scope: all lanes; BLOCK scope: warp 0 only), so
    // there is no warp divergence. The 32 active lanes read the same cur_tail, observe the same
    // CQE and advance in lockstep, consuming one CQE cooperatively rather than racing for it.
    // The epilogue stores (cq_tail / doorbell / sq_tail) reach the same addresses with identical
    // values from every lane; GM writes are published immediately to other cores, so equal-value
    // concurrent writes settle to that value without an intra-warp barrier or atomic store.
    if (active && !aclshmemi_simt_udma_sq_has_credit(state, ACLSHMEMI_SIMT_UDMA_WQE_BB_CNT)) {
        (void)aclshmemi_simt_udma_poll_cq(udma_info, slot, qp_idx, state->cqe_cnt);
    }

    // Read assembly inputs after the credit check (poll_cq advances tail only, never head).
    __gm__ aclshmemi_ubmem_info_t* mem_info = aclshmemi_simt_udma_get_mem_info(udma_info, slot, qp_idx);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_debug_check_remote_jetty_type, mem_info);
    // Translate the symmetric address to the peer's registered MR VA. The NIC operates on the
    // peer's MR base (mem_info->addr), not the local peer-mapping address that aclshmem_ptr
    // returns. This matches the SIMD path's aclshmemi_remote_mr_addr: the offset from the local
    // heap base is added to the remote MR base. In SIMT, heap_base is already uint64_t (unlike
    // SIMD's void*), so no cast is needed.
    __gm__ aclshmem_device_host_state_simt_t* device_state = simt::aclshmemi_get_state();
    uint64_t local_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(remote_symm_addr));
    uint64_t heap_base = device_state->heap_base;
    ACLSHMEM_DEBUG_FUNC(
        aclshmemi_simt_udma_debug_assert_remote_mr_addr, local_ptr, heap_base, device_state->heap_size, mem_info->addr);
    uint64_t remote_mr_offset = local_ptr - heap_base;
    __gm__ uint8_t* remote_addr = reinterpret_cast<__gm__ uint8_t*>(mem_info->addr + remote_mr_offset);
    uint32_t cur_head = state->sq_head;
    uint32_t next_head = cur_head + ACLSHMEMI_SIMT_UDMA_WQE_BB_CNT;
    uint64_t buf_addr = wq_ctx->buf_addr;
    uint64_t db_addr = wq_ctx->db_addr;
    ulonglong2 eid_pair = *reinterpret_cast<__gm__ ulonglong2*>(mem_info->eid_addr);
    uint64_t token = static_cast<uint64_t>(mem_info->rmt_token_value);
    uint64_t w01 = 0;
    uint64_t w23 = 0;
    aclshmemi_simt_udma_build_sqe_header<OP_CODE, CONFIG>(mem_info, cur_head, next_head, w01, w23);

    uint64_t first_len = message_len >> 1U;

    // SQ ring base addresses and DWQE window address. Typed as __gm__ uint64_t* so 8 B asc_stcg
    // stores can index by element (base[lane] = offset lane*8 B), eliminating per-store casts.
    __gm__ uint64_t* bb0 = reinterpret_cast<__gm__ uint64_t*>(
        buf_addr + static_cast<uint64_t>(wqe_size) * (cur_head % shm::UDMA_SQ_BASKBLK_CNT));
    __gm__ uint64_t* bb1 = ((cur_head % shm::UDMA_SQ_BASKBLK_CNT) == (shm::UDMA_SQ_BASKBLK_CNT - 1U)) ?
                               reinterpret_cast<__gm__ uint64_t*>(buf_addr) :
                               (bb0 + (wqe_size / sizeof(uint64_t)));
    __gm__ uint64_t* dwqe = reinterpret_cast<__gm__ uint64_t*>(db_addr - ACLSHMEMI_SIMT_UDMA_DWQE_OFFSET);

    // 16-lane split assembly + SQ write + fence + DWQE write. Each lane assembles one u64, writes
    // it to the SQ ring via asc_stcg (NON_CACHEABLE), fences, then writes the same u64 to the
    // DWQE window via asc_stcg. No asc_syncthreads between assembly/SQ/DWQE (per-thread
    // asc_threadfence); NON_CACHEABLE stores bypass the SIMT Data Cache, so no cache
    // invalidation is needed.
    if (active && lane < 16U) {
        uint64_t eid_l = eid_pair.x;
        uint64_t eid_h = eid_pair.y;
        uint64_t last_len = message_len - first_len;
        uint64_t remote = reinterpret_cast<uint64_t>(remote_addr);
        uint64_t local = reinterpret_cast<uint64_t>(local_addr);
        uint64_t my_u64 = 0;
        if (lane == 0) {
            my_u64 = w01;
        } else if (lane == 1) {
            my_u64 = w23;
        } else if (lane == 2) {
            my_u64 = eid_l;
        } else if (lane == 3) {
            my_u64 = eid_h;
        } else if (lane == 4) {
            my_u64 = token;
        } else if (lane == 5) {
            my_u64 = remote;
        } else if (lane == 6) {
            my_u64 = first_len;
        } else if (lane == 7) {
            my_u64 = local;
        } else if (lane == 8) {
            my_u64 = last_len;
        } else if (lane == 9) {
            my_u64 = local + first_len;
        }
        // lanes 10~15: reserved (my_u64 stays 0)

        // SQ ring write: 16 lanes x 8 B asc_stcg (NON_CACHEABLE) = 128 B.
        __gm__ uint64_t* base = (lane < 8U) ? bb0 : bb1;
        uint32_t off = lane % 8U;
        asc_stcg(base + off, my_u64);

        // Order this lane's own SQ store before its window store (per-thread; no block barrier).
        asc_threadfence();

        // DWQE window write: 16 lanes x 8 B asc_stcg (NON_CACHEABLE) = 128 B.
        asc_stcg(dwqe + lane, my_u64);
    }

    // Account (advance head, increment cqe_cnt for quiet). Single-lane tail operation (lane 0).
    if (active && lane == 0) {
        aclshmemi_simt_udma_post_accounting<CONFIG>(state, next_head);
        ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_dump_wq_ctx, slot, qp_idx, wq_ctx, state);
        ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_dump_sq_wqe, slot, qp_idx, wq_ctx, cur_head, wqe_size);
        ACLSHMEM_DEBUG_FUNC(aclshmemi_simt_udma_debug_check_driver, state, next_head, slot, qp_idx);
    }
}

#else // !ACLSHMEMI_SIMT_UDMA_SUPPORTED || ACLSHMEM_RELAY_SUPPORT

// Unsupported-build stubs: keep all template interfaces compilable; calling any of them
// produces a self-explanatory abort.
template <aclshmemi_udma_opcode_t OP_CODE, thread_group_t SCOPE, const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemi_simt_udma_post_send_coop(
    __gm__ uint8_t* remote_symm_addr, __gm__ uint8_t* local_addr, int pe, uint32_t qp_idx, uint64_t message_len)
{
    (void)remote_symm_addr;
    (void)local_addr;
    (void)pe;
    (void)qp_idx;
    (void)message_len;
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: QP-specific APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else {
        simt::aclshmemi_kernel_abort("SIMT UDMA: QP-specific APIs require Ascend950 direct mode\n");
    }
}

__simt_callee__ inline void aclshmemi_simt_udma_quiet_impl(int pe, uint32_t qp_idx)
{
    (void)pe;
    (void)qp_idx;
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        simt::aclshmemi_kernel_abort("SIMT UDMA: QP-specific APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else {
        simt::aclshmemi_kernel_abort("SIMT UDMA: QP-specific APIs require Ascend950 direct mode\n");
    }
}

#endif // ACLSHMEMI_SIMT_UDMA_SUPPORTED

// ---------------- Public interface (6 functions) ----------------

// put: dst is the remote target, src is the local source.
// get: src is the remote source, dst is the local destination (swapped relative to put).
// The public wrappers only reorder parameters: the remote side is placed first. The
// symmetric-address-to-GM translation is performed inside post_send_coop (the unsupported-build
// stub aborts before reaching the translation).
template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemx_udma_qp_put_nbi_warp(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx)
{
    aclshmemi_simt_udma_post_send_coop<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, ACLSHMEMI_THREADGROUP_WARP, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(dst), reinterpret_cast<__gm__ uint8_t*>(src), pe, qp_idx,
        static_cast<uint64_t>(elem_size) * sizeof(T));
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemx_udma_qp_get_nbi_warp(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx)
{
    // get swaps src/dst (src=remote, dst=local).
    aclshmemi_simt_udma_post_send_coop<aclshmemi_udma_opcode_t::UDMA_OP_READ, ACLSHMEMI_THREADGROUP_WARP, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(src), reinterpret_cast<__gm__ uint8_t*>(dst), pe, qp_idx,
        static_cast<uint64_t>(elem_size) * sizeof(T));
}

__simt_callee__ inline void aclshmemx_udma_qp_quiet_warp(int pe, uint32_t qp_idx)
{
    // Lane 0 of each warp polls its own QP and the remaining lanes observe completion through
    // warp lockstep. No barrier is needed before the poll: post_send_coop already ordered every
    // lane's SQ and DWQE stores with a per-thread asc_threadfence.
    if ((aclshmemi_simt_udma_linear_tid() % warpSize) == 0) {
        aclshmemi_simt_udma_quiet_impl(pe, qp_idx);
    }
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemx_udma_qp_put_nbi_block(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx)
{
    aclshmemi_simt_udma_post_send_coop<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, ACLSHMEMI_THREADGROUP_BLOCK, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(dst), reinterpret_cast<__gm__ uint8_t*>(src), pe, qp_idx,
        static_cast<uint64_t>(elem_size) * sizeof(T));
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
__simt_callee__ inline void aclshmemx_udma_qp_get_nbi_block(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx)
{
    // get swaps src/dst (src=remote, dst=local).
    aclshmemi_simt_udma_post_send_coop<aclshmemi_udma_opcode_t::UDMA_OP_READ, ACLSHMEMI_THREADGROUP_BLOCK, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(src), reinterpret_cast<__gm__ uint8_t*>(dst), pe, qp_idx,
        static_cast<uint64_t>(elem_size) * sizeof(T));
}

__simt_callee__ inline void aclshmemx_udma_qp_quiet_block(int pe, uint32_t qp_idx)
{
    // Thread 0 polls the CQ for all completions; the remaining threads wait at the
    // asc_syncthreads barrier. The barrier is required because SIMT does not guarantee
    // inter-thread memory visibility: without it, warp 1+ threads could return and access
    // transfer data before thread 0's poll confirms completion. asc_syncthreads blocks until
    // all threads arrive, and guarantees that memory operations before the barrier (thread 0's
    // poll writes) are visible to all threads after the barrier.
    if (aclshmemi_simt_udma_linear_tid() == 0) {
        aclshmemi_simt_udma_quiet_impl(pe, qp_idx);
    }
    asc_syncthreads();
}

} // namespace simt

#endif
