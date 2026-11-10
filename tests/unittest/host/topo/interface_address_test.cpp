/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <dlfcn.h>
#include <ifaddrs.h>
#include <memory>
#include <net/if.h>
#include <optional>
#include <string>

#include "aclshmemi_npu_nic_affinity.h"
#include "pcie_nic.h"

namespace {

// Only replace the OS interface snapshot during these tests. Other topo tests use libc normally.
thread_local bool inject_interfaces = false;
thread_local ifaddrs* interface_head = nullptr;
thread_local int interface_error = 0;
thread_local int free_count = 0;

} // namespace

// The declarations in <ifaddrs.h> supply the platform linkage. Only this test
// executable replaces these functions; calls outside the fixture use libc.
int getifaddrs(struct ifaddrs** interfaces) noexcept
{
    if (!inject_interfaces) {
        using get_interfaces_t = int (*)(struct ifaddrs**);
        static const auto system_get = reinterpret_cast<get_interfaces_t>(dlsym(RTLD_NEXT, "getifaddrs"));
        if (system_get == nullptr) {
            errno = ENOSYS;
            return -1;
        }
        return system_get(interfaces);
    }
    if (interface_error != 0) {
        errno = interface_error;
        return -1;
    }
    *interfaces = interface_head;
    return 0;
}

void freeifaddrs(struct ifaddrs* interfaces) noexcept
{
    if (inject_interfaces && interfaces == interface_head) {
        ++free_count;
        return;
    }
    using free_interfaces_t = void (*)(struct ifaddrs*);
    static const auto system_free = reinterpret_cast<free_interfaces_t>(dlsym(RTLD_NEXT, "freeifaddrs"));
    if (system_free == nullptr) {
        std::abort();
    }
    system_free(interfaces);
}

namespace shm {
namespace topo {
namespace {

TEST(InterfaceAddressSystemTest, CallsOutsideTheFixtureUseLibc)
{
    ifaddrs* interfaces = nullptr;
    ASSERT_EQ(getifaddrs(&interfaces), 0);
    std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> snapshot(interfaces, freeifaddrs);
    for (const ifaddrs* current = snapshot.get(); current != nullptr; current = current->ifa_next) {
        if (current->ifa_name == nullptr || current->ifa_addr == nullptr || std::string(current->ifa_name) != "lo" ||
            current->ifa_addr->sa_family != AF_INET ||
            (current->ifa_flags & (IFF_UP | IFF_RUNNING)) != (IFF_UP | IFF_RUNNING)) {
            continue;
        }
        char expected[INET_ADDRSTRLEN] = {0};
        const auto* address = reinterpret_cast<const sockaddr_in*>(current->ifa_addr);
        ASSERT_NE(inet_ntop(AF_INET, &address->sin_addr, expected, sizeof(expected)), nullptr);
        aclshmemi_system_nic_ip_resolver_t resolver;
        EXPECT_EQ(resolver.resolve("lo", AF_INET), std::optional<std::string>(expected));
        char buffer[INET_ADDRSTRLEN] = {0};
        EXPECT_EQ(get_ip_by_if_name("lo", AF_INET, buffer, sizeof(buffer)), 0);
        EXPECT_STREQ(buffer, expected);
        return;
    }
    GTEST_SKIP() << "No active IPv4 loopback interface";
}

class InterfaceAddressTest : public testing::Test {
protected:
    void SetUp() override
    {
        interface_head = nullptr;
        interface_error = 0;
        free_count = 0;
        inject_interfaces = true;
    }

    void TearDown() override
    {
        inject_interfaces = false;
        interface_head = nullptr;
    }

    void AddAddress(const char* address, unsigned int flags = IFF_UP | IFF_RUNNING, const char* name = "eth_test")
    {
        ASSERT_LT(count_, entries_.size());
        auto& entry = entries_[count_];
        entry.info.ifa_name = const_cast<char*>(name);
        entry.info.ifa_flags = flags;
        if (std::string(address).find(':') == std::string::npos) {
            entry.ipv4.sin_family = AF_INET;
            ASSERT_EQ(inet_pton(AF_INET, address, &entry.ipv4.sin_addr), 1);
            entry.info.ifa_addr = reinterpret_cast<sockaddr*>(&entry.ipv4);
        } else {
            entry.ipv6.sin6_family = AF_INET6;
            ASSERT_EQ(inet_pton(AF_INET6, address, &entry.ipv6.sin6_addr), 1);
            entry.info.ifa_addr = reinterpret_cast<sockaddr*>(&entry.ipv6);
        }
        if (count_ == 0) {
            interface_head = &entry.info;
        } else {
            entries_[count_ - 1].info.ifa_next = &entry.info;
        }
        ++count_;
    }

    void ExpectBoth(int family, const std::optional<std::string>& expected)
    {
        aclshmemi_system_nic_ip_resolver_t resolver;
        EXPECT_EQ(resolver.resolve("eth_test", family), expected);
        char buffer[INET6_ADDRSTRLEN] = {0};
        errno = 0;
        const int result = get_ip_by_if_name("eth_test", family, buffer, sizeof(buffer));
        if (expected) {
            EXPECT_EQ(result, 0);
            EXPECT_EQ(std::string(buffer), *expected);
        } else {
            EXPECT_EQ(result, -1);
            EXPECT_EQ(errno, interface_error != 0 ? interface_error : ENODEV);
        }
    }

    struct entry_t {
        ifaddrs info{};
        sockaddr_in ipv4{};
        sockaddr_in6 ipv6{};
    };
    std::array<entry_t, 8> entries_{};
    size_t count_ = 0;
};

TEST_F(InterfaceAddressTest, DualStackPrioritizesIpv4AndHonorsExplicitFamily)
{
    AddAddress("2001:db8::1");
    AddAddress("192.0.2.1");
    ExpectBoth(AF_UNSPEC, "192.0.2.1");
    ExpectBoth(AF_INET, "192.0.2.1");
    ExpectBoth(AF_INET6, "2001:db8::1");
    EXPECT_EQ(free_count, 6);
}

TEST_F(InterfaceAddressTest, Ipv4FirstStillHonorsExplicitIpv6)
{
    AddAddress("192.0.2.1");
    AddAddress("2001:db8::1");
    ExpectBoth(AF_UNSPEC, "192.0.2.1");
    ExpectBoth(AF_INET6, "2001:db8::1");
}

TEST_F(InterfaceAddressTest, Ipv6FallbackUsesFirstAddressAndSupportsLongText)
{
    const char* address = "2001:db8:abcd:ef01:2345:6789:abcd:ef01";
    AddAddress(address);
    AddAddress("2001:db8::2");
    ExpectBoth(AF_UNSPEC, address);
    ExpectBoth(AF_INET6, address);
    ExpectBoth(AF_INET, std::nullopt);
}

TEST_F(InterfaceAddressTest, MappedAddressConvertsToIpv4ExceptForExplicitIpv6)
{
    AddAddress("2001:db8::1");
    AddAddress("::ffff:192.0.2.1");
    ExpectBoth(AF_UNSPEC, "192.0.2.1");
    ExpectBoth(AF_INET, "192.0.2.1");
}

TEST_F(InterfaceAddressTest, ExplicitIpv6PreservesMappedAddress)
{
    AddAddress("::ffff:192.0.2.1");
    ExpectBoth(AF_INET6, "::ffff:192.0.2.1");
}

TEST_F(InterfaceAddressTest, SkipsLinkLocalAndOtherInterfaces)
{
    AddAddress("fe80::1");
    AddAddress("192.0.2.2", IFF_UP | IFF_RUNNING, "eth_other");
    ExpectBoth(AF_UNSPEC, std::nullopt);
    ExpectBoth(AF_INET6, std::nullopt);
    AddAddress("2001:db8::1");
    ExpectBoth(AF_UNSPEC, "2001:db8::1");
}

TEST_F(InterfaceAddressTest, RequiresBothUpAndRunningForBothFamilies)
{
    AddAddress("192.0.2.1", IFF_UP);
    AddAddress("2001:db8::1", IFF_RUNNING);
    ExpectBoth(AF_UNSPEC, std::nullopt);
    ExpectBoth(AF_INET, std::nullopt);
    ExpectBoth(AF_INET6, std::nullopt);
    AddAddress("192.0.2.2");
    ExpectBoth(AF_UNSPEC, "192.0.2.2");
}

TEST_F(InterfaceAddressTest, HcaSearchUsesRealInterfaceSelectionAcrossPorts)
{
    class hca_resolver_t : public aclshmemi_system_nic_ip_resolver_t {
    protected:
        std::vector<std::string> get_hca_interfaces(const std::string& name) const override
        {
            return name == "hca_test" ? std::vector<std::string>{"eth0_test", "eth1_test"} : std::vector<std::string>{};
        }
    } resolver;
    AddAddress("2001:db8::1", IFF_UP | IFF_RUNNING, "eth0_test");
    AddAddress("192.0.2.1", IFF_UP | IFF_RUNNING, "eth1_test");
    EXPECT_EQ(resolver.resolve("hca_test", AF_UNSPEC), std::optional<std::string>("192.0.2.1"));
    EXPECT_EQ(resolver.resolve("hca_test", AF_INET6), std::optional<std::string>("2001:db8::1"));
}

TEST_F(InterfaceAddressTest, MissingNameOrAddressIsSkipped)
{
    AddAddress("192.0.2.1");
    entries_[0].info.ifa_name = nullptr;
    AddAddress("2001:db8::1");
    entries_[1].info.ifa_addr = nullptr;
    ExpectBoth(AF_UNSPEC, std::nullopt);
}

TEST_F(InterfaceAddressTest, PropagatesInterfaceEnumerationFailure)
{
    interface_error = EACCES;
    ExpectBoth(AF_UNSPEC, std::nullopt);
    EXPECT_EQ(free_count, 0);
}

TEST_F(InterfaceAddressTest, PcieBufferCapacityAndErrorContractIsPreserved)
{
    AddAddress("2001:db8:abcd:ef01:2345:6789:abcd:ef01");
    char buffer[INET_ADDRSTRLEN] = {0};
    errno = 0;
    EXPECT_EQ(get_ip_by_if_name("eth_test", AF_INET6, buffer, sizeof(buffer)), -1);
    EXPECT_EQ(errno, ENODEV);
    ExpectBoth(AF_INET6, "2001:db8:abcd:ef01:2345:6789:abcd:ef01");
}

TEST_F(InterfaceAddressTest, PcieBufferIncludesTheTerminatingNull)
{
    AddAddress("192.0.2.1");
    char exact[sizeof("192.0.2.1")] = {0};
    EXPECT_EQ(get_ip_by_if_name("eth_test", AF_INET, exact, sizeof(exact)), 0);
    EXPECT_STREQ(exact, "192.0.2.1");
    char short_buffer[sizeof(exact) - 1] = {0};
    errno = 0;
    EXPECT_EQ(get_ip_by_if_name("eth_test", AF_INET, short_buffer, sizeof(short_buffer)), -1);
    EXPECT_EQ(errno, ENODEV);
}

TEST_F(InterfaceAddressTest, PcieKeepsScanningWhenAnAddressDoesNotFit)
{
    AddAddress("::1");
    AddAddress("192.0.2.1");
    char buffer[sizeof("::1")] = {0};
    EXPECT_EQ(get_ip_by_if_name("eth_test", AF_UNSPEC, buffer, sizeof(buffer)), 0);
    EXPECT_STREQ(buffer, "::1");
    // The XML entry has a full-size buffer and still prefers IPv4 for the same snapshot.
    aclshmemi_system_nic_ip_resolver_t resolver;
    EXPECT_EQ(resolver.resolve("eth_test", AF_UNSPEC), std::optional<std::string>("192.0.2.1"));
}

} // namespace
} // namespace topo
} // namespace shm
