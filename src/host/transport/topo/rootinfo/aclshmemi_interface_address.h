/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_INTERFACE_ADDRESS_H
#define ACLSHMEMI_INTERFACE_ADDRESS_H

#include <cstddef>

namespace shm {
namespace topo {
namespace detail {

// Internal rootinfo helper: require UP/RUNNING, prefer IPv4 in automatic mode,
// and fall back to the first non-link-local IPv6. Explicit families do not cross-fallback.
// Return 0 on success; otherwise return -1 with the getifaddrs errno or ENODEV
// when no matching address fits the caller's buffer (including the terminating null).
int get_interface_address(const char* ifname, int family, char* ip, size_t addr_len);

} // namespace detail
} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_INTERFACE_ADDRESS_H
