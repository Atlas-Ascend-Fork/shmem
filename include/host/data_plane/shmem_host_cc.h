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

/*!
 * \file shmem_host_cc.h
 * \brief shmem host Collective Communication APIs
 */

#ifndef _HOST_DATA_PLANE_ACLSHMEM_HOST_CC_H_
#define _HOST_DATA_PLANE_ACLSHMEM_HOST_CC_H_

#include "acl/acl.h"
#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Host collective contract: every PE in the team calls collectives in the same order,
 * with no overlapping calls on that team. ACLSHMEM_TEAM_INVALID (-1) is a no-op for
 * team-taking APIs; other invalid handles are not covered by this rule.
 * The _all forms select ACLSHMEM_TEAM_WORLD and do not change the execution scope.
 * Host collectives cover Host-issued work ordered before them; Device-issued asynchronous
 * communication needs completion in its Device execution scope.
 */

/**
 * @brief Complete preceding Host communication and wait for every team PE to arrive.
 * @note Once this blocking call returns successfully, covered updates are visible to their
 *       destination PE. The source buffer of a completed transfer may be reused; a destination
 *       may be read after the relevant participating PEs have reached the barrier.
 *       Work on another stream must be ordered before this collective explicitly.
 *
 * @param[in] team Team to synchronize.
 * @return ACLSHMEM_SUCCESS on completion (also for ACLSHMEM_TEAM_INVALID);
 *         ACLSHMEM_NOT_INITED if not initialized, ACLSHMEM_INNER_ERROR if stream wait fails,
 *         or another nonzero error returned by immediate setup.
 */
ACLSHMEM_HOST_API int aclshmem_barrier(aclshmem_team_t team);

/**
 * @brief Perform a blocking Host barrier over ACLSHMEM_TEAM_WORLD with Host execution scope.
 */
ACLSHMEM_HOST_API void aclshmem_barrier_all(void);

/**
 * @brief Submit a team barrier to the selected stream.
 * @note A successful Host return only reports submission, not collective completion.
 *       The completion guarantee applies after this stream executes the barrier.
 *       Wait for this stream (including the library-managed default stream when stream is NULL),
 *       or establish an explicit dependency before work on another stream observes the result.
 *       Earlier work on another stream must be ordered before this barrier explicitly.
 *
 * @param[in] team Team to synchronize.
 * @param[in] stream ACL stream; NULL selects the library-managed default stream.
 * @return ACLSHMEM_SUCCESS after the immediate call (also for ACLSHMEM_TEAM_INVALID);
 *         ACLSHMEM_NOT_INITED if not initialized, or an immediate nonzero error.
 *         Later execution errors are reported by stream synchronization.
 */
ACLSHMEM_HOST_API int aclshmemx_barrier_on_stream(aclshmem_team_t team, aclrtStream stream);
#define shmemx_barrier_on_stream aclshmemx_barrier_on_stream

/**
 * @brief Submit a WORLD barrier with the same Host stream and completion rules.
 *
 * @param[in] stream ACL stream for the barrier. NULL selects the library-managed default stream.
 */
ACLSHMEM_HOST_API void aclshmemx_barrier_all_on_stream(aclrtStream stream);
#define shmemx_barrier_all_on_stream aclshmemx_barrier_all_on_stream

/**
 * @brief Wait for team PE arrival and ordinary-store visibility in the applicable scope.
 * @note Do not rely on sync for NBI completion; complete communication first or use barrier.
 *
 * @param[in] team Team to synchronize.
 */
ACLSHMEM_HOST_API void aclshmem_sync(aclshmem_team_t team);

/**
 * @brief Wait for team PE arrival and ordinary-store visibility in the applicable scope.
 * @note Do not rely on sync for NBI completion; complete communication first or use barrier.
 *
 * @param[in] team Team to synchronize.
 * @return 0 on success, nonzero on failure.
 */
ACLSHMEM_HOST_API int aclshmem_team_sync(aclshmem_team_t team);

/**
 * @brief Perform a blocking WORLD sync with the same Host execution scope.
 */
ACLSHMEM_HOST_API void aclshmem_sync_all(void);

/**
 * @brief Submit a team sync to the selected stream without completing prior NBI.
 * @note A successful Host return reports submission only. Wait for this stream, or order a
 *       consumer on another stream after it; prior work on another stream also needs an
 *       explicit dependency. Sync's arrival guarantee applies when the selected stream
 *       finishes this collective, not when this function returns.
 *
 * @param[in] team Team to synchronize.
 * @param[in] stream ACL stream; NULL selects the library-managed default stream.
 * @return ACLSHMEM_SUCCESS after the immediate call (also for ACLSHMEM_TEAM_INVALID);
 *         ACLSHMEM_NOT_INITED if not initialized, or an immediate nonzero error.
 *         Later execution errors are reported by stream synchronization.
 */
ACLSHMEM_HOST_API int aclshmemx_sync_on_stream(aclshmem_team_t team, aclrtStream stream);
#define shmemx_sync_on_stream aclshmemx_sync_on_stream

/**
 * @brief Submit a WORLD sync with the same Host stream and completion rules.
 *
 * @param[in] stream ACL stream for the synchronization. NULL selects the library-managed default stream.
 */
ACLSHMEM_HOST_API void aclshmemx_sync_all_on_stream(aclrtStream stream);
#define shmemx_sync_all_on_stream aclshmemx_sync_all_on_stream

#ifdef __cplusplus
}
#endif

#endif // _HOST_DATA_PLANE_ACLSHMEM_HOST_CC_H_
