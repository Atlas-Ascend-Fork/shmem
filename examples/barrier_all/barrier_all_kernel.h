/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BARRIER_ALL_DEMO_KERNEL_H
#define BARRIER_ALL_DEMO_KERNEL_H

#include <cstddef>
#include <cstdint>

namespace barrier_all_demo {
// One segment per PE; each segment holds kElementCount uint32_t values.
constexpr uint32_t kElementCount = 1024;
constexpr size_t kSlotBytes = kElementCount * sizeof(uint32_t);

void launch_barrier_all(void* stream, uint64_t fftsAddr, uint32_t* buffer);
} // namespace barrier_all_demo

#endif // BARRIER_ALL_DEMO_KERNEL_H
