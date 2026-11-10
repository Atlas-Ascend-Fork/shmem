/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclshmemi_interface_address.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ifaddrs.h>
#include <net/if.h>
#include <securec.h>

namespace shm {
namespace topo {
namespace detail {

int get_interface_address(const char* ifname, int family, char* ip, size_t addr_len)
{
    struct ifaddrs* ifaddr = NULL;
    if (getifaddrs(&ifaddr) == -1) {
        return -1;
    }
    char ipv6_fallback[INET6_ADDRSTRLEN] = {0};
    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || !ifa->ifa_name) {
            continue;
        }
        if (strcmp(ifname, ifa->ifa_name) != 0 || (ifa->ifa_flags & (IFF_UP | IFF_RUNNING)) != (IFF_UP | IFF_RUNNING)) {
            continue;
        }
        if (ifa->ifa_addr->sa_family == AF_INET) {
            if (family == AF_INET6) {
                continue;
            }
            struct sockaddr_in* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
            if (inet_ntop(AF_INET, &sin->sin_addr, ip, addr_len) != NULL) {
                freeifaddrs(ifaddr);
                return 0;
            }
        } else if (ifa->ifa_addr->sa_family == AF_INET6) {
            const auto* sin6 = reinterpret_cast<const struct sockaddr_in6*>(ifa->ifa_addr);
            const auto& ipv6 = sin6->sin6_addr;
            if (IN6_IS_ADDR_LINKLOCAL(&ipv6)) {
                continue;
            }
            if (IN6_IS_ADDR_V4MAPPED(&ipv6)) {
                if (family == AF_INET6) {
                    if (inet_ntop(AF_INET6, &ipv6, ip, addr_len) != NULL) {
                        freeifaddrs(ifaddr);
                        return 0;
                    }
                    continue;
                }
                in_addr ipv4{};
                const auto* bytes = reinterpret_cast<const uint8_t*>(&ipv6);
                std::memcpy(&ipv4.s_addr, bytes + 12, sizeof(ipv4.s_addr));
                if (inet_ntop(AF_INET, &ipv4, ip, addr_len) != NULL) {
                    freeifaddrs(ifaddr);
                    return 0;
                }
                continue;
            }
            if (family == AF_INET) {
                continue;
            }
            if (ipv6_fallback[0] == '\0') {
                (void)inet_ntop(AF_INET6, &ipv6, ipv6_fallback, sizeof(ipv6_fallback));
            }
        }
    }
    freeifaddrs(ifaddr);
    if (ipv6_fallback[0] != '\0' && strcpy_s(ip, addr_len, ipv6_fallback) == EOK) {
        return 0;
    }
    errno = ENODEV;
    return -1;
}

} // namespace detail
} // namespace topo
} // namespace shm
