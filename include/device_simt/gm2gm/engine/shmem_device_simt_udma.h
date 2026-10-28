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
#ifndef SHMEM_DEVICE_SIMT_UDMA_H
#define SHMEM_DEVICE_SIMT_UDMA_H

#include "device_simt/shmem_simt_common_types.h"
// Reuse the UDMA op-config type and preset constants shared by both AICore and SIMT.
#include "device/gm2gm/engine/shmem_device_udma.h"

namespace simt {

/**
 * @anchor simt_udma_qp_contract
 * @par SIMT UDMA QP-specific contract
 * - Interfaces are available on Ascend950 direct mode only (release build aborts otherwise).
 * - The number of QPs is configured by the Host via aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num)
 *   before initialization and must be identical on every PE. @p qp_idx must be smaller than the
 *   configured QP count (checked in debug builds; caller-guaranteed in release builds).
 * - A single request transfers at least 2 B and at most 256 MB. A normal return only means the
 *   request was submitted; call the matching quiet before reading @p dst (get) or reusing @p src
 *   (put). Quiet guarantees the request has completed. In SIMT the Data Cache is not coherent
 *   with Global Memory, so a thread that cached @p dst before the transfer may still read the
 *   stale copy after quiet. Before reading a Get destination, invalidate its cache line with
 *   asc_dcci_single, or load it with asc_ldcg (bypassing the Data Cache).
 * - Single-driver constraint: at any moment, one (pe, qp_idx) may be driven by a single thread
 *   group executing one collective call (warp/block mode). Use distinct qp_idx values to run
 *   multiple drivers in parallel. A given (pe, qp_idx) must not be driven across the SIMD and
 *   SIMT programming models within the same process; use distinct qp_idx values per programming
 *   model.
 * - @p CONFIG must have static storage duration because it is passed as a C++17 non-type template
 *   parameter by reference. SIMT UDMA requires CONFIG.cqe == 1 (enforced at compile time);
 *   CONFIG.se is reserved and must be 0. See @ref udma_op_config_contract for the full field
 *   specification.
 * - Warp/block variants are collective calls: every thread of the VF must invoke them. Each
 *   participating warp must have at least 16 active lanes. The block quiet variant synchronizes
 *   all threads before returning; the warp quiet variant relies on warp lockstep and needs no
 *   block-level synchronization.
 */

/**
 * @brief Direct-mode asynchronous UDMA Get on an explicitly selected QP, executed in warp scope.
 *
 * @note A normal return only means the request was submitted. Call @ref aclshmemx_udma_qp_quiet_warp
 *       before reading @p dst.
 *
 * @details This is a warp-collective call: all threads of the VF must invoke it, but only one
 *          warp per QP participates in submission; other warps drive their own QPs.
 *          See @ref simt_udma_qp_contract for the single-driver constraint.
 *
 * @tparam T                Element type of the transfer.
 * @tparam CONFIG           UDMA op config. Must have static storage duration; SIMT requires
 *                          CONFIG.cqe == 1. See @ref udma_op_config_contract.
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on symmetric memory of the source data (remote PE).
 * @param elem_size         [in] Number of T elements. One request transfers at least 2 B and at most 256 MB.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index, must be smaller than the configured QP count.
 */
template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
__simt_callee__ inline void aclshmemx_udma_qp_get_nbi_warp(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx);

/**
 * @brief Direct-mode asynchronous UDMA Put on an explicitly selected QP, executed in warp scope.
 *
 * @note A normal return only means the request was submitted. Call @ref aclshmemx_udma_qp_quiet_warp
 *       before reusing @p src.
 *
 * @details This is a warp-collective call: all threads of the VF must invoke it, but only one
 *          warp per QP participates in submission; other warps drive their own QPs.
 *          See @ref simt_udma_qp_contract for the single-driver constraint.
 *
 * @tparam T                Element type of the transfer.
 * @tparam CONFIG           UDMA op config. Must have static storage duration; SIMT requires
 *                          CONFIG.cqe == 1. See @ref udma_op_config_contract.
 * @param dst               [in] Pointer on symmetric memory of the destination data (remote PE).
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of T elements. One request transfers at least 2 B and at most 256 MB.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index, must be smaller than the configured QP count.
 */
template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
__simt_callee__ inline void aclshmemx_udma_qp_put_nbi_warp(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx);

/**
 * @brief Wait for all requests submitted to the specified direct-mode QP to complete,
 *        executed in warp scope.
 *
 * @details This is a warp-collective call: every thread of the VF must invoke it. Each warp
 *          waits for its own QP via warp lockstep; no block-level synchronization is required.
 *          See @ref simt_udma_qp_contract.
 *
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index whose submissions are drained.
 */
__simt_callee__ inline void aclshmemx_udma_qp_quiet_warp(int pe, uint32_t qp_idx);

/**
 * @brief Direct-mode asynchronous UDMA Get on an explicitly selected QP, executed in block scope.
 *
 * @note A normal return only means the request was submitted. Call @ref aclshmemx_udma_qp_quiet_block
 *       before reading @p dst.
 *
 * @details This is a block-collective call: every thread of the VF must invoke it. Only one
 *          warp participates in submission; the remaining warps are inactive but must still
 *          invoke the call. See @ref simt_udma_qp_contract.
 *
 * @tparam T                Element type of the transfer.
 * @tparam CONFIG           UDMA op config. Must have static storage duration; SIMT requires
 *                          CONFIG.cqe == 1. See @ref udma_op_config_contract.
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on symmetric memory of the source data (remote PE).
 * @param elem_size         [in] Number of T elements. One request transfers at least 2 B and at most 256 MB.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index, must be smaller than the configured QP count.
 */
template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
__simt_callee__ inline void aclshmemx_udma_qp_get_nbi_block(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx);

/**
 * @brief Direct-mode asynchronous UDMA Put on an explicitly selected QP, executed in block scope.
 *
 * @note A normal return only means the request was submitted. Call @ref aclshmemx_udma_qp_quiet_block
 *       before reusing @p src.
 *
 * @details This is a block-collective call: every thread of the VF must invoke it. Only one
 *          warp participates in submission; the remaining warps are inactive but must still
 *          invoke the call. See @ref simt_udma_qp_contract.
 *
 * @tparam T                Element type of the transfer.
 * @tparam CONFIG           UDMA op config. Must have static storage duration; SIMT requires
 *                          CONFIG.cqe == 1. See @ref udma_op_config_contract.
 * @param dst               [in] Pointer on symmetric memory of the destination data (remote PE).
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of T elements. One request transfers at least 2 B and at most 256 MB.
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index, must be smaller than the configured QP count.
 */
template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
__simt_callee__ inline void aclshmemx_udma_qp_put_nbi_block(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe, uint32_t qp_idx);

/**
 * @brief Wait for all requests submitted to the specified direct-mode QP to complete,
 *        executed in block scope.
 *
 * @details This is a block-collective call: every thread of the VF must invoke it. All threads
 *          synchronize at a block-level barrier until completion is confirmed, ensuring every
 *          thread observes the completed state before returning. See @ref simt_udma_qp_contract.
 *
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index whose submissions are drained.
 */
__simt_callee__ inline void aclshmemx_udma_qp_quiet_block(int pe, uint32_t qp_idx);

} // namespace simt

#include "gm2gm/engine/shmem_device_simt_udma.hpp"

#endif
