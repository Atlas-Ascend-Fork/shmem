/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H
#define ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H

#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_rdma.h"
#include "shmemi_device_common.h"

ACLSHMEM_DEVICE void aclshmemi_rdma_debug_assert_not_self_send(uint32_t pe);
ACLSHMEM_DEVICE void aclshmemi_rdma_debug_assert_qp_params_valid(__gm__ aclshmemi_rdma_sq_ctx* sq_context);
ACLSHMEM_DEVICE void aclshmemi_rdma_debug_ensure_sq_capacity(
    __gm__ aclshmemi_rdma_sq_ctx* sq_context, uint32_t pe, uint32_t qp_idx, uint32_t wqe_count);
ACLSHMEM_DEVICE void aclshmemi_rdma_debug_check_aggregate_batch_size(uint32_t pending_wqe_count, uint32_t depth);
ACLSHMEM_DEVICE void aclshmemi_rdma_dump_sq_wqe(
    __gm__ aclshmemi_rdma_sq_ctx* sq_context, uint32_t posted_head, uint32_t wqe_size, uint32_t wqe_count);

/*
 *  =====================================================================================================
 *  SHMEM RDMA Architecture Structure —— Base Header
 *  =====================================================================================================
 *  We provide two categories of APIs:
 *  1. Fine-grained primitives (ibverbs-style)
 *      - aclshmemi_roce_fill_wqe
 *      - aclshmemi_roce_ring_sq_doorbell
 *      - aclshmemi_roce_post_send
 *      - aclshmemi_roce_ring_cq_doorbell
 *      - aclshmemi_roce_poll_cq
 *
 *  2. Coarse-grained operations
 *      - aclshmemi_roce_write
 *      - aclshmemi_roce_read
 *      - aclshmemi_roce_atomic_fetch_and_add
 *      - aclshmemi_roce_atomic_compare_and_swap
 * =====================================================================================================
 *  Internal call chain (base -> backend-specialized):
 *      aclshmemi_roce_post_send<B, OP_CODE>()                  // compile-time dispatched entry
 *          -> aclshmemi_roce_post_send_read_write()            // shared post-send helper for READ/WRITE
 *              -> aclshmemi_roce_fill_wqe<B, OP_CODE>()        // compile-time dispatched fill_wqe
 *                  -> aclshmemi_rdma_fill_wqe_write_read()     // shared fill_wqe helper for READ/WRITE
 *          -> aclshmemi_roce_ring_sq_doorbell<B>()             // backend-dispatched doorbell
 * =====================================================================================================
 */

/** @brief Copy UB data to GM and wait for the copy to complete. */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write_ub_to_gm_with_sync(
    uint64_t addr, AscendC::LocalTensor<T>& ub_local, uint32_t size, uint32_t sync_id);

/**
 * @brief Fill WQE for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in aclshmemi_rdma_opcode_t enum class
 * @param wr                     [in] Work request describing the RDMA operation, details in aclshmemi_rdma_send_wr
 * @param sq_context             [in] Current QP's Send queue Context
 * @param wqe_addr               [in] WQE address to fill
 * @param cur_head               [in] Current head of WQE queue
 * @return uint32_t              [out] total size of WQE in bytes
 */
template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe(
    aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr, uint32_t cur_head);

/**
 * @brief Fill one WQE in the UB aggregate staging buffer (WRITE/READ opcodes)
 *
 * Compile-time dispatched on B; every backend with aggregate support must specialize
 * this entry for OP_RDMA_WRITE and OP_RDMA_READ (see the XSCALE / IN_DIE backends).
 * Shared shape: callers stage WQEs with isSubmit=false and submit the last one with
 * isSubmit=true; backends without CQE compression may ignore the flag.
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in aclshmemi_rdma_opcode_t enum class
 * @param wqe_addr               [in] WQE address to fill in the UB staging buffer
 * @param wqe_head               [in] Head index of this WQE (used for WQE id / owner bit)
 * @param message_len            [in] Message length in bytes
 * @param rkey                   [in] Remote memory key
 * @param remote_addr            [in] Remote GM address
 * @param lkey                   [in] Local memory key
 * @param local_addr             [in] Local GM address
 * @param depth                  [in] SQ depth in WQEs (for wraparound-aware backends)
 * @param isSubmit               [in] Whether this WQE should generate a CQE (submit position)
 * @return uint32_t              [out] total size of WQE in bytes
 */
template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe_ub(
    __ubuf__ uint8_t* wqe_addr, uint32_t wqe_head, uint32_t message_len, uint32_t rkey, __gm__ uint8_t* remote_addr,
    uint32_t lkey, __gm__ uint8_t* local_addr, uint32_t depth, bool isSubmit = true);

/**
 * @brief Ring SQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param sq_context             [in] Current QP's Send queue Context
 * @param cur_head               [in] Current head of SQ WQE
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell(
    __gm__ aclshmemi_rdma_sq_ctx*& sq_context, uint32_t cur_head, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief AIV direct RDMA helper function for post send, prepare WQE and ring doorbell.
 * Directly calls the underlying implementation through template parameters
 * without using switch, resulting in higher performance.
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in aclshmemi_rdma_opcode_t enum class
 * @param wr                     [in] Work request describing the RDMA operation, details in aclshmemi_rdma_send_wr
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send(
    aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief Ring CQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param cur_tail               [in] Current tail of CQ WQE
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell(
    uint32_t pe, uint32_t qp_idx, uint32_t cur_tail, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief RDMA Poll Completion Queue (CQ) function. Return status: 0 means success, non-zero means error.
 *
 * @tparam B                     RDMA Backend type
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param target_idx             [in] target progress index after polling; XSCALE use the target SQ
 *                                    producer index because compressed CQEs make CQ and SQ indices different domains
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq(
    uint32_t pe, uint32_t qp_idx, uint32_t target_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/*
 *  =====================================================================================================
 *  Shared RDMA Aggregate Cache (XSCALE / IN_DIE)
 *  =====================================================================================================
 *  Aggregate-capable backends stage WQEs in a UB buffer and snapshot the SQ context into a common 64B cache
 *  placed right before the staging area (buf - UB_ALIGN_SIZE * 2). The cache is filled once per batch
 *  (pending_count == 0) and read by both the stage and commit paths. Each backend keeps its own WQE format
 *  and submit logic. Shared fields occupy 56B; backend-specific fields share the remaining space
 *  through a union. XSCALE needs no extra fields; IN_DIE stores its service level.
 */
struct aclshmemi_roce_aggregate_cache_t {
    uint64_t sq_base_addr;
    uint64_t sq_head_addr;
    uint64_t sq_tail_addr;
    uint64_t sq_db_addr;
    uint32_t sq_depth;
    uint32_t sq_wqn;
    uint32_t sq_wqe_size;
    uint32_t base_head;
    uint32_t remote_rkey;
    uint32_t local_lkey;
    union {
        struct {
            uint32_t sq_sl;
            uint32_t reserved;
        } in_die;
        // Add backend-specific members here so their storage overlaps.
    } backend;
};

static_assert(sizeof(aclshmemi_roce_aggregate_cache_t) <= UB_ALIGN_SIZE * 2);

/**
 * @brief Ring the backend SQ doorbell using the aggregate cache.
 * @note Snapshot required cache fields before writing the aliased UB workspace.
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell_from_cache(
    __ubuf__ aclshmemi_roce_aggregate_cache_t* cache, uint32_t cur_head, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

ACLSHMEM_DEVICE __ubuf__ aclshmemi_roce_aggregate_cache_t* aclshmemi_roce_aggregate_cache(__ubuf__ uint8_t* buf);

/**
 * @brief Snapshot the target QP's SQ context into the UB aggregate cache.
 *
 * @param cache                 [in] UB aggregate cache, obtained via aclshmemi_roce_aggregate_cache()
 * @param rdma_info             [in] RDMA info fetched via aclshmemi_qp_info_fetch()
 * @param pe                    [in] PE number of the remote PE
 * @param qp_idx                [in] QP index in multi-QP scenario
 */
template <aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_fill_aggregate_cache(
    __ubuf__ aclshmemi_roce_aggregate_cache_t* cache, __gm__ aclshmemi_rdma_info* rdma_info, uint32_t pe,
    uint32_t qp_idx);

/** @brief Stage one WQE in UB and increment the pending count. */
template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_stage_rma_wqe_ub(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* buf, uint32_t sync_id, aclshmemx_defer_t action);

/** @brief Append the submit WQE, copy the batch to SQ, and ring the backend doorbell. */
template <aclshmemi_rdma_backend_t B, aclshmemi_rdma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_commit_rma_wqes(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, __ubuf__ uint8_t* buf,
    uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief Asynchronous RDMA Write function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] source address in local HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_write(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = (__gm__ uint8_t*)dst;
    wr.local_addr = (__gm__ uint8_t*)src;
    wr.message_len = message_len;

    aclshmemi_roce_post_send<B, aclshmemi_rdma_opcode_t::OP_RDMA_WRITE>(
        wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

/**
 * @brief Asynchronous RDMA READ function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in local HBM
 * @param src                    [in] source address in remote HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param message_len            [in] message length in Bytes
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE void aclshmemi_roce_read(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    // Attention: Read need src to be wr.remote_addr.
    aclshmemi_rdma_send_wr wr = {};
    wr.remote_addr = (__gm__ uint8_t*)src;
    wr.local_addr = (__gm__ uint8_t*)dst;
    wr.message_len = message_len;

    aclshmemi_roce_post_send<B, aclshmemi_rdma_opcode_t::OP_RDMA_READ>(wr, pe, qp_idx, ub_local64, ub_local32, sync_id);
}

/**
 * @brief Asynchronous RDMA Atomic Fetch and Add function.
 *
 * @tparam T                     Data type for the atomic operation.
 * @tparam IS_MASKED             Flag indicating whether this is a masked atomic operation.
 * @tparam B                     RDMA Backend type
 *
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] reserved field, input could be invalid address, implementation should not use it
 * @param pe                     [in] PE number of the remote PE.
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param add_val                [in] value to be added
 * @param boundary               [in] boundary value for masked FA operation
 * @param ub_local64             [in] temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_fetch_and_add(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t add_val, uint64_t boundary,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA Atomic Compare and Swap function.
 *
 * @tparam T                     Data type for the atomic operation.
 * @tparam IS_MASKED             Flag indicating whether this is a masked atomic operation.
 * @tparam B                     RDMA Backend type
 *
 * @param dst                    [in] Destination address in remote HBM
 * @param src                    [in] reserved field, input could be invalid address, implementation should not use it
 * @param pe                     [in] PE number of the remote PE
 * @param qp_idx                 [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param swap_val               [in] Value to be swapped
 * @param comp_val               [in] Value to be compared
 * @param swap_mask              [in] Mask to apply to swap_val
 * @param comp_mask              [in] Mask to apply to comp_val
 * @param ub_local64             [in] Temporary UB local tensor of uint64_t used as workspace
 * @param ub_local32             [in] Temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event
 */
template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_compare_and_swap(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t swap_val, uint64_t comp_val,
    uint64_t swap_mask, uint64_t comp_mask, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);

/*
 * Backend Atomic Operations Traits System
 *
 * Static polymorphism design for RDMA atomic operations across different backends.
 *
 * Naming Convention:
 *   - aclshmemi_backend_traits<B>: Primary template (undefined, forces specialization)
 *   - atomic_op_traits<T, IS_MASKED>: Nested traits for type-specific operations
 *
 * Structure:
 *   template <aclshmemi_rdma_backend_t B>
 *   struct aclshmemi_backend_traits {
 *       template <typename T, bool IS_MASKED>
 *       struct atomic_op_traits {
 *           static uint32_t fill_wqe(...);   // Construct WQE
 *           static void post_send(...);      // Submit and ring doorbell
 *       };
 *   };
 *
 * Usage:
 *   Each backend (e.g., IN_DIE, XSCALE) must provide a full specialization:
 *
 *   template <>
 *   struct aclshmemi_backend_traits<aclshmemi_rdma_backend_t::XSCALE> {
 *       template <typename T, bool IS_MASKED>
 *       struct atomic_op_traits {
 *           template <aclshmemi_rdma_atomic_op_t OP>
 *           static uint32_t fill_wqe(...) { ... }
 *           template <aclshmemi_rdma_atomic_op_t OP>
 *           static void post_send(...) { ... }
 *       };
 *   };
 *
 * Features:
 *   - Zero runtime overhead through compile-time dispatch
 *   - Each backend implements ONE specialization for all T and IS_MASKED combinations
 */

template <aclshmemi_rdma_backend_t B>
struct aclshmemi_backend_traits {
    template <typename T, bool IS_MASKED>
    struct atomic_op_traits {
        /**
         * @brief Construct Work Queue Entry (WQE) for atomic operation.
         *
         * @tparam ATOMIC_OP_CODE Atomic operation code (OP_ATOMIC_FA or OP_ATOMIC_CAS)
         *
         * @param wr Work request with atomic parameters. Must include:
         *           - remote_addr: Target address in remote memory
         *           - local_addr: Local buffer for result data
         *           - lkey: Local memory key
         *           - rkey: Remote memory key
         *           - atomic.masked_common: Atomic operation parameters
         * @param sq_context Send Queue context for backend configuration
         * @param wqe_addr HBM address for WQE
         * @param cur_head Current SQ producer index
         *
         * @return uint32_t WQE size in bytes for cache flush operations
         */
        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE uint32_t fill_wqe(
            aclshmemi_rdma_send_wr& wr, __gm__ aclshmemi_rdma_sq_ctx*& sq_context, __gm__ uint8_t* wqe_addr,
            uint32_t cur_head);

        /**
         * @brief Submit atomic operation WQE and ring doorbell.
         *
         * Main entry point for atomic operations. Handles QP context retrieval,
         * queue capacity management, WQE construction, cache flush, and doorbell ring.
         *
         * @tparam ATOMIC_OP_CODE Atomic operation code determining operation type
         *
         * @param wr Work request with atomic parameters. Must include:
         *           - remote_addr: Target address in remote memory
         *           - local_addr: Local buffer for result data
         *           - lkey: Local memory key
         *           - rkey: Remote memory key
         *           - atomic.masked_common: Atomic operation parameters
         * @param pe Remote PE number
         * @param qp_idx Queue Pair index (0 for single-QP configurations)
         * @param ub_local64 64-bit workspace for doorbell operations
         * @param ub_local32 32-bit workspace for index updates
         * @param sync_id Sync ID for S/MTE3 pipeline coordination
         *
         */
        template <aclshmemi_rdma_atomic_op_t ATOMIC_OP_CODE>
        static ACLSHMEM_DEVICE void post_send(
            aclshmemi_rdma_send_wr& wr, uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<uint64_t>& ub_local64,
            AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id);
    };
};

// The following structure is used to provide compile-time errors when calling unimplemented atomic operations
template <aclshmemi_rdma_atomic_op_t>
struct aclshmemi_atomic_op_dependent_false : std::false_type {};

// The following structure is used to provide compile-time errors when calling an unimplemented backend
template <aclshmemi_rdma_backend_t>
struct aclshmemi_rdma_backend_dependent_false : std::false_type {};

#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_H
