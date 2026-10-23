/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem.h"
#include "gm2gm/shmemi_device_cc.h"
#include "unittest/utils/sync_kernel.h"

extern "C" [[bisheng::core_ratio(1, 2)]] ACLSHMEM_GLOBAL void sync_mixed_stress(
    uint64_t config, GM_ADDR reports, aclshmem_team_t team, uint32_t iterations)
{
    util_set_ffts_config(config);
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        // Even AIVs outside the communication worker budget must reach every
        // local barrier in both the public dispatcher and direct V4 path.
        aclshmem_sync(team);
        aclshmemi_sync_npu_v4<false>(team);
    }
    uint32_t slot = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        slot += ACLSHMEM_SYNC_MIXED_SLOTS_PER_SIDE;
    }
    auto report = reinterpret_cast<__gm__ int32_t*>(reports + slot * ACLSHMEM_SYNCBIT_SIZE);
    aclshmemi_store(report, static_cast<int32_t>(iterations));
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(report));
}

void sync_mixed_stress_do(
    uint32_t block_num, void* stream, uint64_t config, uint8_t* reports, aclshmem_team_t team, uint32_t iterations)
{
    sync_mixed_stress<<<block_num, nullptr, stream>>>(config, reports, team, iterations);
}
