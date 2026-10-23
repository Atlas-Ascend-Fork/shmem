/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SYNC_KERNEL_H
#define SYNC_KERNEL_H

constexpr uint32_t ACLSHMEM_SYNC_STRESS_AIV_COUNT = 32;
constexpr uint32_t ACLSHMEM_SYNC_STRESS_ITERATIONS = 64;
constexpr uint32_t ACLSHMEM_SYNC_MIXED_SLOTS_PER_SIDE = 128;

void sync_mixed_stress_do(
    uint32_t block_num, void* stream, uint64_t config, uint8_t* reports, aclshmem_team_t team, uint32_t iterations);

void sync_increase_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size);
void sync_increase_vec_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size);
void sync_increase_single_aiv_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size);
void sync_v4_single_aiv_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size);
void sync_stress_vec_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size);
void sync_core_soft_mixed_stress_do(
    uint32_t block_num, void* stream, uint64_t config, uint8_t* observed_aiv_count, uint8_t* completion, int rank_id);
void sync_v4_stress_vec_do(
    uint32_t block_num, void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size);
void sync_singleton_team_do(void* stream, uint64_t config, uint8_t* addr, aclshmem_team_t team_id);
void sync_increase_do_odd_team(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, aclshmem_team_t team_id);
void sync_increase_vec_do_odd_team(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, aclshmem_team_t team_id);

#endif // SYNC_KERNEL_H
