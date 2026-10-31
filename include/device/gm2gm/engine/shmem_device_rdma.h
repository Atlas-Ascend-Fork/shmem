/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_RDMA_H
#define SHMEM_DEVICE_RDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmem_device_rdma.hpp"

/** @brief Maximum operations in one XSCALE QP-specific aggregate batch, including the submit operation. */
constexpr uint32_t ACLSHMEM_ROCE_QP_AGGREGATE_MAX_OPS = 1023;

/**
 * @anchor roce_qp_aggregate_contract
 * @par QP-specific aggregate RoCE contract
 * QP-specific aggregate operations are supported on XSCALE and HNS_1825.
 * - A batch contains deferred operations followed by exactly one submit operation. The submit operation publishes the
 *   complete batch and resets its state after success.
 * - A batch must use one operation kind, PE, QP index, submit state, UB workspace base, and synchronization ID.
 * - The total number of operations, including the submit operation, must not exceed
 *   @ref ACLSHMEM_ROCE_QP_AGGREGATE_MAX_OPS. Split larger work into multiple batches.
 * - The UB workspace capacity must be at least `64 + 128 * n` bytes, where n is the total batch size.
 * - Different QP indices may concurrently access the same PE when each execution unit owns its QP index, submit state,
 *   and UB workspace. A single `(pe, qp_idx)` supports one producer and one active batch at a time.
 * - After submit, call @ref aclshmemx_roce_qp_quiet before consuming Get destinations, reusing Put sources, or
 *   reusing the batch workspace.
 */

/**
 * @brief Return the target PE's native RDMA address for a local DEVICE_SIDE symmetric address.
 *        RDMA and the target PE's MR metadata must be initialized, @p pe must be a valid PE number,
 *        and @p ptr must belong to the local symmetric DEVICE_SIDE heap.
 *        Invalid inputs are not reported by returning nullptr; diagnostic builds may abort on invalid addresses.
 * @deprecated Low-level RDMA operations now accept local symmetric pointers and derive the target MR address
 *             internally. Use those operations instead of converting the pointer explicitly.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] Target PE number.
 * @return The target PE's registered RDMA address at the same heap offset.
 */
[[deprecated("aclshmem_roce_ptr is deprecated; pass local symmetric pointers to RDMA operations instead.")]]
ACLSHMEM_DEVICE __gm__ void* aclshmem_roce_ptr(__gm__ void* ptr, int pe);
#define shmem_roce_ptr aclshmem_roce_ptr

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data on the local PE.
 * @param src               [in] Symmetric address of the source data.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @note Both operands are local symmetric addresses; the RDMA engine derives the target network address internally.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data on the local PE.
 * @param src               [in] Symmetric address of the source data.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID for MTE3 pipeline synchronization.
 * @note Both operands are local symmetric addresses; the RDMA engine derives the target network address internally.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory using an
 *        explicitly selected QP. This QP-specific ROCE interface is supported on XSCALE and HNS_1825 backends.
 *        See @ref roce_qp_aggregate_contract for QP-specific concurrency rules.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data on the local PE.
 * @param src               [in] Symmetric address of the source data.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index for the target PE. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Hardware event ID for MTE3 pipeline synchronization.
 * @note Both operands are local symmetric addresses; the RDMA engine derives the target network address internally.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory using an
 *        explicitly selected QP. This QP-specific ROCE interface is supported on XSCALE and HNS_1825 backends.
 *        See @ref roce_qp_aggregate_contract for QP-specific concurrency rules.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data on the local PE.
 * @param src               [in] GlobalTensor at the symmetric address of the source data.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index for the target PE. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Hardware event ID for MTE3 pipeline synchronization.
 * @note Both operands are local symmetric addresses; the RDMA engine derives the target network address internally.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Adds one asynchronous RoCE Get operation to an aggregate batch on an explicitly selected QP.
 *
 * @details Adds this operation to the active batch without submitting it. See @ref roce_qp_aggregate_contract.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Local symmetric destination address.
 * @param src               [in] Symmetric source address on the target PE specified by @p pe.
 * @param buf               [in] Local UB workspace for this batch. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p src.
 * @param qp_idx            [in] QP index for @p pe. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Defer action referencing the batch's initialized submit state.
 *
 * @note Call aclshmemx_roce_qp_quiet(pe, qp_idx, buf, sync_id) after the matching submit operation and before
 *       reading @p dst.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds one asynchronous RoCE Get operation to an aggregate batch and submits the batch on an explicitly
 *        selected QP.
 *
 * @details Adds this operation, submits the active batch, and resets the submit state on success. See
 *          @ref roce_qp_aggregate_contract.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Local symmetric destination address.
 * @param src               [in] Symmetric source address on the target PE specified by @p pe.
 * @param buf               [in] Local UB workspace for this batch. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p src.
 * @param qp_idx            [in] QP index for @p pe. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Submit action referencing the same submit state used by the staged operations.
 *
 * @note A normal return means the batch was submitted, not that the transfer completed. Call
 *       aclshmemx_roce_qp_quiet(pe, qp_idx, buf, sync_id) before reading @p dst.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief GlobalTensor/LocalTensor overload of the deferred QP-specific aggregate RoCE Get interface.
 *
 * @details Has the same batch, backend, buffer-capacity, address, and completion requirements as the pointer
 *          overload. The tensor objects must refer to symmetric GM allocations and local UB storage, respectively.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Local symmetric destination GlobalTensor.
 * @param src               [in] Remote symmetric source GlobalTensor.
 * @param buf               [in] Local UB workspace LocalTensor. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p src.
 * @param qp_idx            [in] QP index for @p pe.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Defer action referencing the batch submit state.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief GlobalTensor/LocalTensor overload of the submitting QP-specific aggregate RoCE Get interface.
 *
 * @details Has the same batch, backend, buffer-capacity, address, and completion requirements as the pointer
 *          overload. This call adds the final operation, submits the batch, and resets the submit state on success.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Local symmetric destination GlobalTensor.
 * @param src               [in] Remote symmetric source GlobalTensor.
 * @param buf               [in] Local UB workspace LocalTensor. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p src.
 * @param qp_idx            [in] QP index for @p pe.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Submit action referencing the same submit state used by the staged operations.
 *
 * @note Call aclshmemx_roce_qp_quiet(pe, qp_idx, buf, sync_id) before reading the destination tensor.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data on the local PE.
 * @param src               [in] GlobalTensor at the symmetric address of the source data.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: src is translated to the corresponding address on pe; dst is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size,
    int pe);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations
 *        to the same PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data on the local PE.
 * @param src               [in] GlobalTensor at the symmetric address of the source data.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID for MTE3 pipeline synchronization.
 * @note Address requirements: src is translated to the corresponding address on pe; dst is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);
#define shmem_roce_get_mem_nbi aclshmemx_roce_get_nbi

/**
 * @brief Adds the current nonblocking RoCE Get operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the local PE.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Get operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the local PE.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief Adds the current nonblocking RoCE Get operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Local GlobalTensor for the destination data on the local PE.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Get operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Get operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id) before reading or reusing dst.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Local GlobalTensor for the destination data on the local PE.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Get operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data.
 * @param src               [in] Symmetric address of the source data on the local PE.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data.
 * @param src               [in] Symmetric address of the source data on the local PE.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE using
 *        an explicitly selected QP. This QP-specific ROCE interface is supported on XSCALE and HNS_1825 backends.
 *        See @ref roce_qp_aggregate_contract for QP-specific concurrency rules.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric address of the destination data.
 * @param src               [in] Symmetric address of the source data on the local PE.
 * @param buf               [in] Pointer on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index for the target PE. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE using
 *        an explicitly selected QP. This QP-specific ROCE interface is supported on XSCALE and HNS_1825 backends.
 *        See @ref roce_qp_aggregate_contract for QP-specific concurrency rules.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data.
 * @param src               [in] GlobalTensor at the symmetric address of the source data on the local PE.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index for the target PE. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Adds one asynchronous RoCE Put operation to an aggregate batch on an explicitly selected QP.
 *
 * @details Adds this operation to the active batch without submitting it. See @ref roce_qp_aggregate_contract.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the target PE specified by @p pe.
 * @param src               [in] Local symmetric source address.
 * @param buf               [in] Local UB workspace for this batch. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p dst.
 * @param qp_idx            [in] QP index for @p pe. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Defer action referencing the batch submit state.
 *
 * @note Keep @p src unchanged until aclshmemx_roce_qp_quiet(pe, qp_idx, buf, sync_id) returns.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds one asynchronous RoCE Put operation to an aggregate batch and submits the batch on an explicitly
 *        selected QP.
 *
 * @details Adds this operation, submits the active batch, and resets the submit state on success. See
 *          @ref roce_qp_aggregate_contract.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the target PE specified by @p pe.
 * @param src               [in] Local symmetric source address.
 * @param buf               [in] Local UB workspace for this batch. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p dst.
 * @param qp_idx            [in] QP index for @p pe. Must be less than the configured RDMA QP count.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Submit action referencing the same submit state as the staged operations.
 *
 * @note A normal return means the batch was submitted, not that the transfer completed. Keep @p src unchanged until
 *       aclshmemx_roce_qp_quiet(pe, qp_idx, buf, sync_id) returns.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief GlobalTensor/LocalTensor overload of the deferred QP-specific aggregate RoCE Put interface.
 *
 * @details Has the same batch, backend, buffer-capacity, address, and completion requirements as the pointer
 *          overload.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Remote symmetric destination GlobalTensor.
 * @param src               [in] Local symmetric source GlobalTensor.
 * @param buf               [in] Local UB workspace LocalTensor. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p dst.
 * @param qp_idx            [in] QP index for @p pe.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Defer action referencing the batch submit state.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief GlobalTensor/LocalTensor overload of the submitting QP-specific aggregate RoCE Put interface.
 *
 * @details Has the same batch, backend, buffer-capacity, address, and completion requirements as the pointer
 *          overload. This call adds the final operation, submits the batch, and resets the submit state on success.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Remote symmetric destination GlobalTensor.
 * @param src               [in] Local symmetric source GlobalTensor.
 * @param buf               [in] Local UB workspace LocalTensor. See @ref roce_qp_aggregate_contract.
 * @param elem_size         [in] Number of T elements transferred by this operation.
 * @param pe                [in] Target PE that owns @p dst.
 * @param qp_idx            [in] QP index for @p pe.
 * @param sync_id           [in] Synchronization ID for this batch.
 * @param action             [in] Submit action referencing the same submit state as the staged operations.
 *
 * @note Keep @p src unchanged until aclshmemx_roce_qp_quiet(pe, qp_idx, buf, sync_id) returns.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t qp_idx, uint32_t sync_id, aclshmemx_submit_t action);
/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data.
 * @param src               [in] GlobalTensor at the symmetric address of the source data on the local PE.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size,
    int pe);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] GlobalTensor at the symmetric address of the destination data.
 * @param src               [in] GlobalTensor at the symmetric address of the source data on the local PE.
 * @param buf               [in] LocalTensor on local UB. Must be at least 128 bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @note Address requirements: dst is translated to the corresponding address on pe; src is the local RDMA operand.
 *       Both operands must point to symmetric memory, and each complete transfer range must remain within its
 *       corresponding allocation.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);
#define shmem_roce_put_mem_nbi aclshmemx_roce_put_nbi

/**
 * @brief Adds the current nonblocking RoCE Put operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Symmetric source address on the local PE.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Put operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it
 *       returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Symmetric source address on the local PE.
 * @param buf               [in] Local UB workspace shared by all calls in this batch. For n total operations, its
 *                          capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief Adds the current nonblocking RoCE Put operation to a batch and keeps the batch pending.
 *
 * @warning n is the total number of operations in the batch, including the final submit call,
 *          and must be less than the SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note Finish the batch with an aclshmemx_submit_t action. After submit, call
 *       aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data on the local PE.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking RoCE Put operation and submits all operations in the batch.
 *
 * @warning n is the total number of operations in the batch, including this submit call, and must be less than the
 *          SQ ring depth.
 * @note Aggregate ROCE NBI is supported on IN_DIE, XSCALE and HNS_1825, by one AI Core, one active batch,
 *       and QP0. Every call in a batch must use the same state, pe, Put operation, buf base, and sync_id.
 * @note The submit call contributes one operation to n. On a submit failure, the calling device kernel is aborted
 *       and the submit state is not reset.
 * @note After submit, call aclshmemx_roce_quiet(pe, buf, sync_id); src must remain valid and unchanged until it
 *       returns.
 *
 * @tparam T                  Element type of the transfer.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data on the local PE.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch. For n total
 *                          operations, its capacity must be at least 64 + 128 * n bytes.
 * @param elem_size         [in] Number of elements transferred by this Put operation. The total transfer size
 *                          (`elem_size * sizeof(T)`) must not exceed `UINT32_MAX` bytes.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief RDMA Quiet function. This synchronous function ensures all previous RDMA WQEs on default QP are completed
 * (data has arrived at the destination NIC).
 *
 * @param pe                [in] PE number of the remote PE.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_quiet(uint32_t pe, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief Wait for completion of previously submitted RDMA operations on one explicitly selected QP.
 *        This QP-specific ROCE interface is supported on XSCALE and HNS_1825 backends.
 *        The caller must own the selected QP while its completion is being waited for.
 *
 * @tparam T                  Element type used for the UB pointer.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index for the target PE. Must be less than the configured RDMA QP count.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_quiet(uint32_t pe, uint32_t qp_idx, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief Wait for completion of previously submitted RDMA operations on one explicitly selected QP.
 *        This QP-specific ROCE interface is supported on XSCALE and HNS_1825 backends.
 *        The caller must own the selected QP while its completion is being waited for.
 *
 * @tparam T                  Element type used for the UB tensor.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index for the target PE. Must be less than the configured RDMA QP count.
 * @param buf               [in] LocalTensor on local UB, available space larger than 64 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_qp_quiet(
    uint32_t pe, uint32_t qp_idx, AscendC::LocalTensor<T> buf, uint32_t sync_id);

/**
 * @brief RDMA Team Sync function. Performs a synchronization operation on the specified team,
 * ensuring all PEs in the team reach the sync point before proceeding.
 * This is a collective operation that uses RDMA-based dissemination algorithm.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 *
 * @param team              [in] Pointer to the team on which to perform synchronization.
 */
ACLSHMEM_DEVICE int aclshmemx_roce_team_sync(aclshmemx_team_t* team);
#define aclshmemx_roce_sync aclshmemx_roce_team_sync

/**
 * @brief RDMA Team Sync function with explicit UB buffer and sync_id. Performs a synchronization
 * operation on the specified team, ensuring all PEs in the team reach the sync point before proceeding.
 * This is a collective operation that uses RDMA-based dissemination algorithm.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param team              [in] Pointer to the team on which to perform synchronization.
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 * @return 0 on success, non-zero on failure.
 */
template <typename T>
ACLSHMEM_DEVICE int aclshmemx_roce_team_sync(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Sync All function. Performs a synchronization operation on all PEs
 * (ACLSHMEM_TEAM_WORLD), ensuring all PEs reach the sync point before proceeding.
 * Equivalent to aclshmemx_roce_team_sync with ACLSHMEM_TEAM_WORLD team.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 */
ACLSHMEM_DEVICE void aclshmemx_roce_sync_all();

/**
 * @brief RDMA Sync All function with explicit UB buffer and sync_id. Performs a synchronization
 * operation on all PEs (ACLSHMEM_TEAM_WORLD), ensuring all PEs reach the sync point before proceeding.
 * Equivalent to aclshmemx_roce_team_sync with ACLSHMEM_TEAM_WORLD team.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_sync_all(__ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Barrier function. Performs a barrier operation on the specified team,
 * ensuring all previous RDMA operations are completed and all PEs in the team
 * reach the barrier point before proceeding. This function first performs a quiet
 * operation on all QPs for all PEs in the team, then performs a sync operation.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 *
 * @param team              [in] Pointer to the team on which to perform barrier.
 */
ACLSHMEM_DEVICE int aclshmemx_roce_barrier(aclshmemx_team_t* team);

/**
 * @brief RDMA Barrier function with explicit UB buffer and sync_id. Performs a barrier operation
 * on the specified team, ensuring all previous RDMA operations are completed and all PEs in the team
 * reach the barrier point before proceeding. This function first performs a quiet operation on all QPs
 * for all PEs in the team, then performs a sync operation.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param team              [in] Pointer to the team on which to perform barrier.
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE int aclshmemx_roce_barrier(aclshmemx_team_t* team, __ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief RDMA Barrier All function. Performs a barrier operation on all PEs
 * (ACLSHMEM_TEAM_WORLD), ensuring all previous RDMA operations are completed and
 * all PEs reach the barrier point before proceeding.
 * Equivalent to aclshmemx_roce_barrier with ACLSHMEM_TEAM_WORLD team.
 *        WARNING: This interface reads UB buffer and sync_id from device_state->rdma_config.
 *        If aclshmemx_set_rdma_config is not called to configure device_state's buf and sync_id,
 *        or if multiple concurrent operations share the same rdma_config, resource conflicts
 *        may occur. Use the overload with explicit buf and sync_id parameters for concurrent scenarios.
 */
ACLSHMEM_DEVICE void aclshmemx_roce_barrier_all();

/**
 * @brief RDMA Barrier All function with explicit UB buffer and sync_id. Performs a barrier operation
 * on all PEs (ACLSHMEM_TEAM_WORLD), ensuring all previous RDMA operations are completed and
 * all PEs reach the barrier point before proceeding.
 * Equivalent to aclshmemx_roce_barrier with ACLSHMEM_TEAM_WORLD team.
 *        This version allows the caller to explicitly provide UB buffer and sync_id, avoiding resource
 *        conflicts with the default rdma_config in device_state.
 *
 * @param buf               [in] Pointer on local UB, available space larger than 128 Bytes.
 * @param sync_id           [in] ID used to Sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_barrier_all(__ubuf__ T* buf, uint32_t sync_id);

/**
 * @brief Synchronous interface. Returns the value at the source address on the specified PE.
 * Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit data types.
 *
 * @param src               [in] Symmetric address of the source data.
 * @param pe                [in] PE number of the remote PE.
 * @return The value at the source address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch(__gm__ T* src, int32_t pe);

/**
 * @brief Asynchronous interface. Sets the value at the destination address on the specified PE.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit data types.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be set.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_set(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Conditionally updates the value at the destination address.
 * Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param cond              [in] Value to compare against.
 * @param value             [in] Value to be written if comparison succeeds.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);

/**
 * @brief Synchronous interface. Swaps the value at the destination address. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be swapped.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_swap(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Increments the value at the destination address by 1.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Asynchronous interface. Adds the value to the destination address. Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be added.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types: int32, uint32, int64, uint64.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms results in undefined behavior.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types: int32, uint32, int64, uint64.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms will result in a compile-time
 * error.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Asynchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types: int32, uint32, int64, uint64.
 * Supported hardware platform: Ascend950.
 *        This is an asynchronous operation. The caller must invoke aclshmemx_roce_quiet to ensure
 *        the operation has completed and the data is visible on the remote PE.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms results in undefined behavior.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_atomic_xor(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Increments the value at the destination address by 1 and returns the old
 * value. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address before increment.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_inc(__gm__ T* dst, int32_t pe);
/**
 * @brief Synchronous interface. Adds the value to the destination address and returns the old value.
 * Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be added.
 * @param pe                [in] PE number of the remote PE.
 * @return The original value at the destination address before addition.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms will result in a compile-time
 * error.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms results in undefined behavior.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64. Supported hardware platform: Ascend950.
 *        The function returns after the remote atomic operation has completed and is visible on the remote PE.
 *        An internal quiet operation is performed before returning.
 *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported. Use sync_id in device_state.rdma_config for pipeline synchronization.
 * @note T only supports 32-bit and 64-bit integers. Using unsupported types or platforms will result in a compile-time
 * error.
 *
 * @param dst               [in] Symmetric address of the destination data. Must be a valid symmetric address.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE. Must be a valid PE number within the active set.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_roce_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe);

#endif
