/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_MEGA_MOE_ROUTING_REPORT_H
#define SHMEM_MEGA_MOE_ROUTING_REPORT_H

#include "common/shmem_mega_moe_math.h"

namespace ShmemMegaMoeKernel {
constexpr uint32_t ROUTING_REPORT_TILE_N = 256;
// Separate software report IDs from the kernel's hardware flag IDs. Each
// (global routing row, output column tile) has exactly one SET and one WAIT.
constexpr uint64_t ROUTING_REPORT_EVENT_BASE = 1ULL << 20;
constexpr uint64_t ROUTING_REPORT_EVENT_LIMIT = 0x7fffffffULL;

#if defined(__CCE__)
#define SHMEM_ROUTING_REPORT_INLINE __forceinline__[host, aicore]
#else
#define SHMEM_ROUTING_REPORT_INLINE inline
#endif

SHMEM_ROUTING_REPORT_INLINE uint64_t RoutingReportColumnCount(uint32_t modelDim)
{
    return CeilDiv(static_cast<uint64_t>(modelDim), ROUTING_REPORT_TILE_N);
}

SHMEM_ROUTING_REPORT_INLINE bool RoutingReportFits(uint32_t maxRows, uint32_t modelDim)
{
    const uint64_t eventCount = static_cast<uint64_t>(maxRows) * RoutingReportColumnCount(modelDim);
    return eventCount > 0 && eventCount <= ROUTING_REPORT_EVENT_LIMIT - ROUTING_REPORT_EVENT_BASE + 1;
}

SHMEM_ROUTING_REPORT_INLINE int32_t RoutingReportEvent(uint32_t row, uint32_t column, uint32_t modelDim)
{
    return static_cast<int32_t>(
        ROUTING_REPORT_EVENT_BASE + static_cast<uint64_t>(row) * RoutingReportColumnCount(modelDim) + column);
}

#undef SHMEM_ROUTING_REPORT_INLINE
} // namespace ShmemMegaMoeKernel
#endif
