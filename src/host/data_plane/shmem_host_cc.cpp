/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host/data_plane/shmem_host_cc.h"

#include "acl/acl.h"

#include "shmemi_host_common.h"
#include "gm2gm/shmemi_device_cc_kernel.h"

namespace {
int aclshmemi_sync_on_stream(aclshmem_team_t team, aclrtStream stream)
{
    if (team == -1) { // not in team
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(
        aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED, "shmem is not initialized, please initialize first.",
        ACLSHMEM_NOT_INITED);
    if (stream == nullptr) {
        SHM_LOG_INFO("sync_on_stream received a NULL stream; using the default stream.");
        stream = g_state_host.default_stream;
    }
    ACLSHMEM_CHECK_RET(
        aclshmemi_call_sync_on_stream_kernel(team, stream), "aclshmemi_call_sync_on_stream_kernel failed.");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_barrier_on_stream(aclshmem_team_t team, aclrtStream stream)
{
    if (team == -1) { // not in team
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(
        aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED, "shmem is not initialized, please initialize first.",
        ACLSHMEM_NOT_INITED);
    if (stream == nullptr) {
        SHM_LOG_INFO("barrier_on_stream received a NULL stream; using the default stream.");
        stream = g_state_host.default_stream;
    }
    ACLSHMEM_CHECK_RET(
        aclshmemi_call_barrier_on_stream_kernel(team, stream), "aclshmemi_call_barrier_on_stream_kernel failed.");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_sync(aclshmem_team_t team)
{
    if (team == -1) {
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(
        aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED, "shmem is not initialized, please initialize first.",
        ACLSHMEM_NOT_INITED);
    ACLSHMEM_CHECK_RET(
        aclshmemi_call_sync_on_stream_kernel(team, g_state_host.default_stream),
        "aclshmemi_call_sync_on_stream_kernel failed.");
    aclError sync_status = aclrtSynchronizeStream(g_state_host.default_stream);
    if (sync_status != ACL_SUCCESS) {
        SHM_LOG_ERROR("sync stream after aclshmemi_call_sync_on_stream_kernel failed, aclError: " << sync_status);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_barrier(aclshmem_team_t team)
{
    if (team == -1) {
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(
        aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED, "shmem is not initialized, please initialize first.",
        ACLSHMEM_NOT_INITED);
    ACLSHMEM_CHECK_RET(
        aclshmemi_call_barrier_on_stream_kernel(team, g_state_host.default_stream),
        "aclshmemi_call_barrier_on_stream_kernel failed.");
    aclError sync_status = aclrtSynchronizeStream(g_state_host.default_stream);
    if (sync_status != ACL_SUCCESS) {
        SHM_LOG_ERROR("sync stream after aclshmemi_call_barrier_on_stream_kernel failed, aclError: " << sync_status);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}
} // namespace

int aclshmem_barrier(aclshmem_team_t team) { return aclshmemi_barrier(team); }

void aclshmem_barrier_all() { aclshmem_barrier(ACLSHMEM_TEAM_WORLD); }

int aclshmemx_barrier_on_stream(aclshmem_team_t team, aclrtStream stream)
{
    return aclshmemi_barrier_on_stream(team, stream);
}

void aclshmemx_barrier_all_on_stream(aclrtStream stream) { aclshmemi_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream); }

void aclshmem_sync(aclshmem_team_t team) { aclshmem_barrier(team); }

int aclshmem_team_sync(aclshmem_team_t team) { return aclshmemi_sync(team); }

void aclshmem_sync_all() { aclshmem_sync(ACLSHMEM_TEAM_WORLD); }

int aclshmemx_sync_on_stream(aclshmem_team_t team, aclrtStream stream)
{
    return aclshmemi_sync_on_stream(team, stream);
}

void aclshmemx_sync_all_on_stream(aclrtStream stream) { aclshmemi_sync_on_stream(ACLSHMEM_TEAM_WORLD, stream); }
