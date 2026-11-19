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
#include "host_device/shmem_common_macros.h"
#if ACLSHMEM_DYNAMIC_META
#include <cerrno>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
// The fault fixtures compile production code against simulated ACL APIs. Run
// them in isolated processes so their symbols cannot interpose on real NPU UTs.
void RunFixture(const char* name)
{
    const std::string script =
        std::string(ACLSHMEM_TEST_SOURCE_DIR) + "/tests/unittest/scripts/test_dynamic_address_" + name + ".py";
    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        alarm(120);
        execl(ACLSHMEM_TEST_PYTHON, ACLSHMEM_TEST_PYTHON, script.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    pid_t result;
    do {
        result = waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);
    ASSERT_EQ(result, child);
    ASSERT_TRUE(WIFEXITED(status)) << "Fixture terminated by signal: " << status;
    EXPECT_EQ(WEXITSTATUS(status), 0) << script;
}
} // namespace
#endif

#if ACLSHMEM_DYNAMIC_META
#define RUN_ADDRESS_FIXTURE(name) RunFixture(name)
#else
#define RUN_ADDRESS_FIXTURE(name) GTEST_SKIP() << "Build with -enable_device_symbol to run this fixture"
#endif

TEST(DynamicDeviceAddress, RegistryFaultRecovery) { RUN_ADDRESS_FIXTURE("registry"); }
TEST(DynamicDeviceAddress, ArenaOwnerCleanup) { RUN_ADDRESS_FIXTURE("owner"); }
TEST(DynamicDeviceAddress, PackageDeviceArchives) { RUN_ADDRESS_FIXTURE("wheel"); }
#undef RUN_ADDRESS_FIXTURE
