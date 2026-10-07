/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_UDMA_CONNECT_TIMEOUT_H
#define SHMEM_UDMA_CONNECT_TIMEOUT_H
#include <cstdint>

namespace shm::transport::device {
constexpr uint32_t kDefaultUdmaConnectTimeoutMs = 120000;
constexpr uint32_t kMaxUdmaConnectTimeoutMs = 3600000;

// Decimal milliseconds only. Keep the historical default unless explicitly configured.
inline bool ParseUdmaConnectTimeout(const char* value, uint32_t& timeout)
{
    timeout = kDefaultUdmaConnectTimeoutMs;
    if (value == nullptr) return true;
    if (*value == '\0') return false;
    uint32_t parsed = 0;
    for (const char* p = value; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        const uint32_t digit = static_cast<uint32_t>(*p - '0');
        if (parsed > (kMaxUdmaConnectTimeoutMs - digit) / 10U) return false;
        parsed = parsed * 10U + digit;
    }
    if (parsed < 1000U || parsed > kMaxUdmaConnectTimeoutMs) return false;
    timeout = parsed;
    return true;
}
} // namespace shm::transport::device
#endif
