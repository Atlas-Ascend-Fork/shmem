/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "unittest/utils/sync_timeout_kernel.h"
#include "unittest/utils/scope_env.h"

namespace {
constexpr uint64_t kLocalBytes = 64ULL << 20;
constexpr uint32_t kIterations = 1000;
constexpr size_t kResultBytes = 32 * SYNC_TIMEOUT_TEST_SLOT_BYTES;
constexpr unsigned kProcessTimeoutSeconds = 45;

void ExpectDeviceTimeout(uint64_t timeout_us, uint64_t instance_id = 0)
{
    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    uint8_t* output = nullptr;
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&output), SYNC_TIMEOUT_TEST_SLOT_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY),
        ACL_SUCCESS);
    sync_timeout_do(
        1, stream, util_get_ffts_config(), SyncTimeoutMode::SIGNAL, SyncTimeoutFault::NONE, 0, output, instance_id);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    std::array<uint64_t, 2> host{};
    ASSERT_EQ(aclrtMemcpy(host.data(), sizeof(host), output, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(host[1], timeout_us);
    EXPECT_EQ(aclrtFree(output), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
}

void TestNormal(int rank, int ranks, uint64_t bytes, SyncTimeoutMode mode, uint64_t timeout_us)
{
    alarm(kProcessTimeoutSeconds);
    aclrtStream stream = nullptr;
    test_init(rank, ranks, bytes, &stream, timeout_us == ACLSHMEM_SYNC_TIMEOUT_DEFAULT_US ? 0 : timeout_us);
    ASSERT_FALSE(::testing::Test::HasFailure());
    ASSERT_NE(stream, nullptr);
    uint8_t* output = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&output), kResultBytes, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
    std::vector<uint64_t> host(kResultBytes / sizeof(uint64_t));
    for (uint32_t blocks : {2U, 4U, 8U, 16U, 32U}) {
        SCOPED_TRACE(::testing::Message() << "rank=" << rank << " blocks=" << blocks);
        ASSERT_EQ(aclrtMemset(output, kResultBytes, 0, kResultBytes), ACL_SUCCESS);
        sync_timeout_do(blocks, stream, util_get_ffts_config(), mode, SyncTimeoutFault::NONE, kIterations, output);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(host.data(), kResultBytes, output, kResultBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        for (uint32_t core = 0; core < blocks; ++core) {
            const auto slot = core * SYNC_TIMEOUT_TEST_SLOT_BYTES / sizeof(uint64_t);
            EXPECT_EQ(host[slot], kIterations) << "core=" << core;
            EXPECT_EQ(host[slot + 1], timeout_us) << "core=" << core;
        }
        ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);
    }
    if (timeout_us != ACLSHMEM_SYNC_TIMEOUT_DEFAULT_US) {
        // The last finalize resets the process configuration. Verify the next initialization on device too.
        ASSERT_EQ(aclrtFree(output), ACL_SUCCESS);
        test_finalize(stream, rank % test_gnpu_num + test_first_npu);
        test_init(rank, ranks, bytes, &stream);
        ASSERT_FALSE(::testing::Test::HasFailure());
        ASSERT_NE(stream, nullptr);
        ASSERT_EQ(
            aclrtMalloc(reinterpret_cast<void**>(&output), kResultBytes, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(output, kResultBytes, 0, kResultBytes), ACL_SUCCESS);
        sync_timeout_do(2, stream, util_get_ffts_config(), mode, SyncTimeoutFault::NONE, 1, output);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(host.data(), kResultBytes, output, kResultBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        EXPECT_EQ(host[1], ACLSHMEM_SYNC_TIMEOUT_DEFAULT_US);
    }
    ASSERT_EQ(aclrtFree(output), ACL_SUCCESS);
    test_finalize(stream, rank % test_gnpu_num + test_first_npu);
    alarm(0);
}

void RunNormal(SyncTimeoutMode mode, uint64_t timeout_us = ACLSHMEM_SYNC_TIMEOUT_DEFAULT_US)
{
    test_mutil_task(
        [mode, timeout_us](int rank, int ranks, uint64_t bytes) { TestNormal(rank, ranks, bytes, mode, timeout_us); },
        kLocalBytes, test_gnpu_num);
}

struct Child {
    std::unique_ptr<FILE, decltype(&std::fclose)> log{std::tmpfile(), &std::fclose};
    pid_t pid = -1;
    int status = 0;
    bool done = false;
    ~Child()
    {
        if (pid > 0 && !done) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }
};

// A failed device context is deliberately confined to a disposable child.
// Do not finalize or reuse its stream after the expected trap.
void FaultChild(int rank, SyncTimeoutMode mode, SyncTimeoutFault fault, uint64_t timeout_us)
{
    alarm(kProcessTimeoutSeconds - 5);
    aclrtStream stream = nullptr;
    test_init(rank, 2, kLocalBytes, &stream, timeout_us);
    if (::testing::Test::HasFailure() || stream == nullptr) {
        std::fflush(nullptr);
        _exit(2);
    }
    uint8_t* output = nullptr;
    if (aclrtMalloc(reinterpret_cast<void**>(&output), kResultBytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        _exit(3);
    }
    // Verify the same kernel and communication path before injecting the fault.
    sync_timeout_do(2, stream, util_get_ffts_config(), mode, SyncTimeoutFault::NONE, 1, output);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS || aclshmemi_control_barrier_all() != ACLSHMEM_SUCCESS) {
        std::fflush(nullptr);
        _exit(8);
    }
    const auto start = std::chrono::steady_clock::now();
    sync_timeout_do(2, stream, util_get_ffts_config(), mode, fault, 1, output);
    const auto status = aclrtSynchronizeStream(stream);
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    if (fault == SyncTimeoutFault::MISSING_PE && rank == 1) {
        if (status != ACL_SUCCESS) {
            _exit(4);
        }
        // Keep mappings live until the parent has collected PE0's timeout result.
        while (true)
            pause();
    }
    // Device printf delivery after trap is runtime-dependent. Check observable
    // failure and timing instead; captured logs remain available on assertion failure.
    if (elapsed_us < static_cast<int64_t>(timeout_us)) {
        std::fprintf(
            stderr, "Fault returned too early: status=%d elapsed_us=%lld timeout_us=%llu\n", status,
            static_cast<long long>(elapsed_us), static_cast<unsigned long long>(timeout_us));
        std::fflush(nullptr);
        _exit(9);
    }
    std::fflush(nullptr);
    _exit(status != ACL_SUCCESS ? 0 : 5);
}

void RunFault(SyncTimeoutMode mode, SyncTimeoutFault fault, uint64_t timeout_us = SYNC_TIMEOUT_TEST_US)
{
    if (test_first_rank != 0 || test_global_ranks != 2 || test_gnpu_num != 2) {
        GTEST_SKIP() << "fault isolation requires exactly two local PEs";
    }
    std::array<Child, 2> children;
    for (auto& child : children) {
        ASSERT_NE(child.log.get(), nullptr);
    }
    std::fflush(nullptr);
    for (int rank = 0; rank < 2; ++rank) {
        auto& child = children[rank];
        child.pid = fork();
        ASSERT_GE(child.pid, 0);
        if (child.pid == 0) {
            if (dup2(fileno(child.log.get()), STDOUT_FILENO) < 0 || dup2(fileno(child.log.get()), STDERR_FILENO) < 0) {
                _exit(6);
            }
            FaultChild(rank, mode, fault, timeout_us);
            _exit(7);
        }
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kProcessTimeoutSeconds);
    bool all_done = false;
    bool released_peer = false;
    while (!all_done && std::chrono::steady_clock::now() < deadline) {
        all_done = true;
        for (auto& child : children) {
            if (!child.done) {
                const auto pid = waitpid(child.pid, &child.status, WNOHANG);
                ASSERT_TRUE(pid >= 0 || errno == EINTR);
                child.done = pid == child.pid;
            }
            all_done = all_done && child.done;
        }
        if (fault == SyncTimeoutFault::MISSING_PE && children[0].done && !released_peer) {
            kill(children[1].pid, SIGTERM);
            released_peer = true;
        }
        if (!all_done)
            usleep(100000);
    }
    ASSERT_TRUE(all_done) << "fault workers exceeded the process deadline";
    for (int rank = 0; rank < 2; ++rank) {
        auto& child = children[rank];
        std::rewind(child.log.get());
        std::string log;
        std::array<char, 4096> buffer{};
        size_t count;
        while ((count = std::fread(buffer.data(), 1, buffer.size(), child.log.get())) != 0) {
            log.append(buffer.data(), count);
        }
        if (fault == SyncTimeoutFault::MISSING_PE && rank == 1) {
            ASSERT_TRUE(WIFSIGNALED(child.status)) << log;
            EXPECT_EQ(WTERMSIG(child.status), SIGTERM) << log;
            continue;
        }
        ASSERT_TRUE(WIFEXITED(child.status)) << log;
        ASSERT_EQ(WEXITSTATUS(child.status), 0) << log;
    }
}
} // namespace

void TestConfiguration(int rank, int, uint64_t bytes)
{
    alarm(90);
    const int device = rank % test_gnpu_num + test_first_npu;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(device), ACL_SUCCESS);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    // Fail after runtime configuration resolution: no UID means bootstrap must reject this attempt.
    aclshmemx_init_attr_t failed{};
    test_set_attr(rank, 1, bytes, test_global_ipport, &failed);
    failed.comm_args = nullptr;
    failed.runtime_config.sync_timeout_us = 125000;
    EXPECT_EQ(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &failed), ACLSHMEM_INVALID_PARAM);
    // Check conversion boundaries and the unspecified/default value without waiting for a timeout.
    for (uint32_t value : {uint32_t{0}, uint32_t{1}, uint32_t{ACLSHMEM_SYNC_TIMEOUT_MAX_US}}) {
        aclshmemx_init_attr_t attr{};
        test_set_attr(rank, 1, bytes, test_global_ipport, &attr);
        attr.runtime_config.sync_timeout_us = value;
        ASSERT_EQ(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), ACLSHMEM_SUCCESS);
        ExpectDeviceTimeout(value == 0 ? ACLSHMEM_SYNC_TIMEOUT_DEFAULT_US : value);
        ASSERT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    }
    EXPECT_EQ(aclrtResetDevice(device), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    alarm(0);
}

TEST(TEST_SYNC_TIMEOUT, ConfigurationValidation) { test_mutil_task(TestConfiguration, kLocalBytes, 1); }

void TestConfigurationInheritance(int rank, int, uint64_t bytes)
{
    alarm(90);
    const int device = rank % test_gnpu_num + test_first_npu;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(device), ACL_SUCCESS);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    std::array<aclshmemx_init_attr_t, 5> attributes{};
    auto init = [&](uint64_t instance, uint64_t timeout) {
        auto& attr = attributes[instance];
        attr = aclshmemx_init_attr_t{};
        test_set_attr(rank, 1, bytes, "tcp://127.0.0.1:0", &attr);
        attr.instance_id = instance;
        attr.comm_args = nullptr;
        attr.runtime_config.sync_timeout_us = timeout;
        return aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    };
    ASSERT_EQ(init(1, 250000), ACLSHMEM_SUCCESS);
    ExpectDeviceTimeout(250000, 1);
    ASSERT_EQ(init(2, 0), ACLSHMEM_SUCCESS);
    ExpectDeviceTimeout(250000, 2);
    EXPECT_EQ(init(3, 100000), ACLSHMEM_NOT_SUPPORTED);
    ASSERT_EQ(init(3, 250000), ACLSHMEM_SUCCESS);
    ExpectDeviceTimeout(250000, 3);
    ASSERT_EQ(aclshmemx_finalize(1), ACLSHMEM_SUCCESS);
    EXPECT_EQ(init(4, 100000), ACLSHMEM_NOT_SUPPORTED);
    ASSERT_EQ(aclshmemx_finalize(3), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclshmemx_finalize(2), ACLSHMEM_SUCCESS);
    ASSERT_EQ(init(1, 0), ACLSHMEM_SUCCESS);
    ExpectDeviceTimeout(ACLSHMEM_SYNC_TIMEOUT_DEFAULT_US, 1);
    ASSERT_EQ(aclshmemx_finalize(1), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    alarm(0);
}

TEST(TEST_SYNC_TIMEOUT, ConfigurationInheritanceAndConflict)
{
    ScopedEnv ports("SHMEM_INSTANCE_PORT_RANGE", get_multi_instance_port_range().c_str());
    test_mutil_task(TestConfigurationInheritance, kLocalBytes, 1);
}
TEST(TEST_SYNC_TIMEOUT, ConfiguredThresholdAndReset) { RunNormal(SyncTimeoutMode::SIGNAL, 250000); }
TEST(TEST_SYNC_TIMEOUT, V3Normal) { RunNormal(SyncTimeoutMode::V3); }
TEST(TEST_SYNC_TIMEOUT, V4Normal) { RunNormal(SyncTimeoutMode::V4); }
TEST(TEST_SYNC_TIMEOUT, PublicVecNormal) { RunNormal(SyncTimeoutMode::VEC); }
TEST(TEST_SYNC_TIMEOUT, SoftwareLocalNormal) { RunNormal(SyncTimeoutMode::SOFT); }
TEST(TEST_SYNC_TIMEOUT, SignalCurrentNextAndWraparound) { RunNormal(SyncTimeoutMode::SIGNAL); }
TEST(TEST_SYNC_TIMEOUT, V3MissingPe) { RunFault(SyncTimeoutMode::V3, SyncTimeoutFault::MISSING_PE); }
TEST(TEST_SYNC_TIMEOUT, V4MissingPe) { RunFault(SyncTimeoutMode::V4, SyncTimeoutFault::MISSING_PE, 250000); }
TEST(TEST_SYNC_TIMEOUT, SoftwareLocalMissingCore) { RunFault(SyncTimeoutMode::SOFT, SyncTimeoutFault::MISSING_CORE); }
