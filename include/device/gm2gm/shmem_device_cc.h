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

// clang-format off
/*
 * Collective programming contract:
 * - The all-core barrier and sync APIs require a MIX kernel; all participating local AIC and AIV
 *   cores must enter. Vector-only sync APIs involve only AIV cores. The _all forms select
 *   ACLSHMEM_TEAM_WORLD without changing the local core participation scope.
 * - Every PE in a team must call its collectives in the same order. Do not overlap collective
 *   calls on the same team. ACLSHMEM_TEAM_INVALID (-1) is a no-op for team-taking APIs;
 *   other invalid handles are not covered by this rule.
 * - Ordinary symmetric-memory stores from participating cores are synchronized on supported
 *   HCCS/UB paths. The Cube scalar unit is outside this visibility guarantee.
 * - Device collectives cover Device-issued work; Host-issued work needs the corresponding
 *   Host-side completion and ordering.
 */
// clang-format on

/*!
 * \file shmem_device_cc.h
 * \brief shmem device Collective Communication APIs
 */
#ifndef _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_
#define _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_

#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
 * @brief Set runtime ffts address. Call this at MIX Kernel entry point (if the kernel contains barrier calls).
 *
 * @param config              [config] ffts config, acquired by util_get_ffts_config()
 */
ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config);
#define shmemx_set_ffts_config util_set_ffts_config

/**
 * @brief Complete prior communication and synchronize the PEs in a team.
 * @note All local cores in the declared participation scope must enter, and all team PEs
 *       must call collectives in the same order. On return, prior communication covered
 *       by this Device API is complete and its updates are visible to the destination PE.
 *       The source buffer of a completed transfer may then be reused. QP-specific transfers
 *       require their matching QP-specific quiet.
 *
 * @param[in] team Team to synchronize.
 * @return 0 after the collective (also for ACLSHMEM_TEAM_INVALID); Device execution errors
 *         are not reported through this return value.
 */
ACLSHMEM_DEVICE int aclshmem_barrier(aclshmem_team_t team);
#define shmem_barrier aclshmem_barrier

/**
 * @brief Perform aclshmem_barrier over ACLSHMEM_TEAM_WORLD with the same local core scope.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all(void);
#define shmem_barrier_all aclshmem_barrier_all

/**
 * @brief Collective synchronization over a team in which only vector cores participate.
 * @note Cross-card synchronization is supported only within a UB/HCCS domain inside one machine or supernode.
 *       All PEs in the team must belong to that domain.
 * @note This routine does not ensure completion of remote memory updates issued through ACLSHMEM communication APIs.
 *       Cube cores may call the API, but take no effect.
 *
 * @param team              [in] team to synchronize
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec(aclshmem_team_t team);

/**
 * @brief aclshmemx_sync_vec over ACLSHMEM_TEAM_WORLD.
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec_all(void);

/**
 * @brief Compatibility wrapper for aclshmemx_sync_vec.
 *
 * @param team              [in] team to synchronize
 */
[[deprecated("aclshmemx_barrier_vec is deprecated, please use aclshmemx_sync_vec instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t team);
#define shmemx_barrier_vec aclshmemx_barrier_vec

/**
 * @brief Compatibility wrapper for aclshmemx_sync_vec_all.
 */
[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmemx_sync_vec_all instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec(void);
#define shmemx_barrier_all_vec aclshmemx_barrier_all_vec

/**
 * @brief Wait for all team PEs to reach this synchronization point.
 * @note On supported HCCS/UB paths, ordinary symmetric-memory stores from participating
 *       cores are synchronized within the applicable visibility scope. This call does not
 *       complete outstanding NBI communication. Call the matching quiet before sync when
 *       communication completion or source-buffer reuse is required, or use barrier instead.
 *       All cores in the declared participation scope must enter in the same order.
 *
 * @param[in] team Team to synchronize.
 */
ACLSHMEM_DEVICE void aclshmem_sync(aclshmem_team_t team);

/**
 * @brief Wait for all team PEs to reach this synchronization point.
 * @note On supported HCCS/UB paths, ordinary symmetric-memory stores from participating
 *       cores are synchronized within the applicable visibility scope. This call does not
 *       complete outstanding NBI communication. Call the matching quiet before sync when
 *       communication completion or source-buffer reuse is required, or use barrier instead.
 *       All cores in the declared participation scope must enter in the same order.
 *
 * @param[in] team Team to synchronize.
 * @return 0 after the collective (also for ACLSHMEM_TEAM_INVALID); Device execution errors
 *         are not reported through this return value.
 */
ACLSHMEM_DEVICE int aclshmem_team_sync(aclshmem_team_t team);

/**
 * @brief Perform aclshmem_sync over ACLSHMEM_TEAM_WORLD with the same local core scope.
 */
ACLSHMEM_DEVICE void aclshmem_sync_all(void);

#ifdef __cplusplus
}
#endif

#include "gm2gm/shmemi_device_cc.h"

#endif // _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_
