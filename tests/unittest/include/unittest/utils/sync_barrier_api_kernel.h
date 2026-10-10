/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SYNC_BARRIER_API_KERNEL_H
#define SYNC_BARRIER_API_KERNEL_H

#include <cstdint>

#include "host/shmem_host_def.h"

void device_sync_all_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length);
void device_barrier_all_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length);
void device_sync_team_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id, int team_pe_start, int team_pe_stride, int team_pe_size);
void device_barrier_team_api_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id, int team_pe_start, int team_pe_stride, int team_pe_size);

#endif // SYNC_BARRIER_API_KERNEL_H
