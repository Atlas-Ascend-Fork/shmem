/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_SYNC_TIMEOUT_KERNEL_H
#define ACLSHMEM_SYNC_TIMEOUT_KERNEL_H
#include <cstdint>

enum class SyncTimeoutMode : uint32_t { V3 = 3, V4 = 4, SOFT = 6, VEC = 8, SIGNAL = 9 };
enum class SyncTimeoutFault : uint32_t { NONE = 0, MISSING_PE = 1, MISSING_CORE = 2 };
constexpr uint32_t SYNC_TIMEOUT_TEST_SLOT_BYTES = 128;
constexpr uint32_t SYNC_TIMEOUT_TEST_US = 100000;

void sync_timeout_do(
    uint32_t blocks, void* stream, uint64_t config, SyncTimeoutMode mode, SyncTimeoutFault fault, uint32_t iterations,
    uint8_t* output, uint64_t instance_id = 0);
#endif
