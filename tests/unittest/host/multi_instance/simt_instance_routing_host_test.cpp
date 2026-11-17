/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <exception>
#include <string>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "shmem.h"
#include "init/shmemi_init.h"
#include "unittest_main_test.h"
#include "unittest/utils/scope_env.h"

#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
extern "C" void simt_instance_routing_transfer(void*, void*, uint32_t, int, uint64_t, void*);
#endif

namespace {
constexpr uint32_t SLOT = 4096;
constexpr int32_t SENTINEL = 0x5a5a5a5a;
constexpr size_t BYTES = 3 * SLOT * sizeof(int32_t);
constexpr int ROUNDS = 20;

int32_t value(int rank, int id, int round, uint32_t index)
{
    return 10000000 * id + 100000 * round + 5000 * rank + static_cast<int>(index);
}

// Independent CPU reference: source, written receive span, unwritten receive
// tail and guard are all checked. Every receive span has one ring predecessor.
bool payload_matches(const std::vector<int32_t>& data, int rank, int ranks, int id, int round, uint32_t count)
{
    if (data.size() != 3 * SLOT)
        return false;
    const int source_rank = (rank + ranks - 1) % ranks;
    for (uint32_t i = 0; i < 3 * SLOT; ++i) {
        const int32_t expected = i < SLOT ? value(rank, id, round, i) :
                                            (i < SLOT + count ? value(source_rank, id, round, i - SLOT) : SENTINEL);
        if (data[i] != expected)
            return false;
    }
    return true;
}

#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
struct test_status_t {
    int rc = 0;
    std::string operation;

    bool ok() const { return rc == 0; }
    bool check(int code, const char* name)
    {
        if (code != 0) {
            rc = code;
            operation = name;
        }
        return code == 0;
    }
};

// Host bootstrap rendezvous, as used by the existing heap tests. This is not
// a DMA fence: both streams must complete before the post-transfer barrier.
test_status_t rendezvous(uint64_t id)
{
    test_status_t status;
    if (!status.check(aclshmemx_instance_ctx_set(id), "select barrier instance")) {
        return status;
    }
    if (!status.check(aclshmemi_control_barrier_all(), "control barrier")) {
        return status;
    }

    return {};
}

struct instance_t {
    uint64_t id;
    int rank;
    int ranks;
    int32_t* heap = nullptr;
    int32_t* identity = nullptr;
    aclrtStream stream = nullptr;
    int round = -1;
    uint32_t count = 0;

    test_status_t create(uint64_t bytes)
    {
        aclshmemx_init_attr_t attr{};
        attr.my_pe = rank;
        attr.n_pes = ranks;
        attr.local_mem_size = bytes;
        attr.instance_id = id;
        std::strcpy(attr.ip_port, "tcp://127.0.0.1:0");
        attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
        attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
        test_status_t status;
        if (!status.check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "init instance")) {
            return status;
        }
        if (!status.check(aclshmemx_instance_ctx_set(id), "select allocation instance")) {
            return status;
        }
        heap = static_cast<int32_t*>(aclshmem_malloc(BYTES));
        if (heap == nullptr)
            return {ACLSHMEM_INNER_ERROR, "SHMEM allocation failed"};
        if (!status.check(
                aclrtMalloc(reinterpret_cast<void**>(&identity), 2 * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY),
                "identity allocation")) {
            return status;
        }
        if (!status.check(aclrtCreateStream(&stream), "create stream")) {
            return status;
        }

        return {};
    }

    test_status_t prepare(int next_round, uint32_t n)
    {
        round = next_round;
        count = n;
        std::vector<int32_t> input(3 * SLOT, SENTINEL);
        for (uint32_t i = 0; i < SLOT; ++i)
            input[i] = value(rank, id, round, i);
        test_status_t status;
        if (!status.check(
                aclrtMemcpy(heap, BYTES, input.data(), BYTES, ACL_MEMCPY_HOST_TO_DEVICE), "prepare payload")) {
            return status;
        }
        if (!status.check(aclrtMemset(identity, 2 * sizeof(int32_t), 0xff, 2 * sizeof(int32_t)), "prepare identity")) {
            return status;
        }

        return {};
    }
    test_status_t launch()
    {
        simt_instance_routing_transfer(stream, heap, count, (rank + 1) % ranks, id, identity);
        return {};
    }
    test_status_t wait()
    {
        test_status_t status;
        if (!status.check(aclrtSynchronizeStream(stream), "stream completion")) {
            return status;
        }
        return {};
    }
    test_status_t read(std::vector<int32_t>& data)
    {
        data.resize(3 * SLOT);
        test_status_t status;
        if (!status.check(aclrtMemcpy(data.data(), BYTES, heap, BYTES, ACL_MEMCPY_DEVICE_TO_HOST), "read payload")) {
            return status;
        }
        return {};
    }
    test_status_t verify()
    {
        std::vector<int32_t> actual;
        test_status_t status = read(actual);
        if (!status.ok()) {
            return status;
        }
        if (!payload_matches(actual, rank, ranks, id, round, count)) {
            return {
                ACLSHMEM_INNER_ERROR,
                "payload/guard mismatch instance=" + std::to_string(id) + " round=" + std::to_string(round)};
        }
        int32_t ids[2];
        if (!status.check(
                aclrtMemcpy(ids, sizeof(ids), identity, sizeof(ids), ACL_MEMCPY_DEVICE_TO_HOST), "read identity")) {
            return status;
        }
        if (ids[0] != rank || ids[1] != ranks)
            return {ACLSHMEM_INNER_ERROR, "instance PE query mismatch"};

        return {};
    }
    test_status_t destroy()
    {
        test_status_t status = wait();
        if (!status.ok()) {
            return status;
        }
        status = rendezvous(id);
        if (!status.ok()) {
            return status;
        }
        aclshmem_free(heap);
        if (!status.check(aclshmemx_finalize(id), "finalize instance")) {
            return status;
        }
        if (!status.check(aclrtDestroyStream(stream), "destroy stream")) {
            return status;
        }
        if (!status.check(aclrtFree(identity), "free identity")) {
            return status;
        }
        heap = nullptr;

        return {};
    }
};

void run_rank(int rank, const std::function<test_status_t()>& task)
{
    // Do not enter collective cleanup after a failed stage. The child runner
    // reports failure and bounds another rank blocked in a collective by alarm.
    std::signal(SIGALRM, SIG_DFL);
    alarm(300);
    test_status_t status;
    try {
        status = task();
    } catch (const std::exception& error) {
        status = {ACLSHMEM_INNER_ERROR, std::string("C++ rank exception: ") + error.what()};
    }
    if (!status.ok()) {
        ADD_FAILURE() << "rank=" << rank << " operation=" << status.operation << " rc=" << status.rc;
        return;
    }
    alarm(0);
}

test_status_t run_lifecycle_impl(int rank, int ranks)
{
    test_status_t status;
    if (!status.check(aclInit(nullptr), "aclInit")) {
        return status;
    }
    if (!status.check(aclrtSetDevice(rank % test_gnpu_num + test_first_npu), "set device")) {
        return status;
    }
    if (!status.check(aclshmemx_set_conf_store_tls(false, nullptr, 0), "disable test TLS")) {
        return status;
    }
    instance_t old{1, rank, ranks}, fresh{2, rank, ranks};
    status = old.create(64ULL << 20);
    if (!status.ok()) {
        return status;
    }
    status = old.prepare(0, 256);
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(1);
    if (!status.ok()) {
        return status;
    }
    status = old.launch();
    if (!status.ok()) {
        return status;
    }
    status = old.wait();
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(1);
    if (!status.ok()) {
        return status;
    }
    status = old.verify();
    if (!status.ok()) {
        return status;
    }
    std::vector<int32_t> retained;
    status = old.read(retained);
    if (!status.ok()) {
        return status;
    }
    status = fresh.create(128ULL << 20);
    if (!status.ok()) {
        return status;
    }
    std::vector<int32_t> current;
    status = old.read(current);
    if (!status.ok()) {
        return status;
    }
    if (current != retained)
        return {ACLSHMEM_INNER_ERROR, "old output changed during new instance creation"};
    if (old.heap == fresh.heap)
        return {ACLSHMEM_INNER_ERROR, "live heaps alias"};
    for (int round = 1; round <= ROUNDS; ++round) {
        const uint32_t count = round % 2 ? 256 : SLOT;
        status = old.prepare(round, count);
        if (!status.ok()) {
            return status;
        }
        status = fresh.prepare(round, count);
        if (!status.ok()) {
            return status;
        }
        status = rendezvous(2);
        if (!status.ok()) {
            return status;
        }
        // Both submissions precede both waits, preserving stream overlap.
        status = old.launch();
        if (!status.ok()) {
            return status;
        }
        status = fresh.launch();
        if (!status.ok()) {
            return status;
        }
        status = old.wait();
        if (!status.ok()) {
            return status;
        }
        status = fresh.wait();
        if (!status.ok()) {
            return status;
        }
        status = rendezvous(2);
        if (!status.ok()) {
            return status;
        }
        status = old.verify();
        if (!status.ok()) {
            return status;
        }
        status = fresh.verify();
        if (!status.ok()) {
            return status;
        }
    }
    status = fresh.read(retained);
    if (!status.ok()) {
        return status;
    }
    status = old.destroy();
    if (!status.ok()) {
        return status;
    }
    status = fresh.read(current);
    if (!status.ok()) {
        return status;
    }
    if (current != retained)
        return {ACLSHMEM_INNER_ERROR, "new output changed during old finalize"};
    status = fresh.prepare(ROUNDS + 1, SLOT);
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(2);
    if (!status.ok()) {
        return status;
    }
    status = fresh.launch();
    if (!status.ok()) {
        return status;
    }
    status = fresh.wait();
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(2);
    if (!status.ok()) {
        return status;
    }
    status = fresh.verify();
    if (!status.ok()) {
        return status;
    }
    status = fresh.read(retained);
    if (!status.ok()) {
        return status;
    }
    status = old.create(96ULL << 20);
    if (!status.ok()) {
        return status;
    }
    status = fresh.read(current);
    if (!status.ok()) {
        return status;
    }
    if (current != retained)
        return {ACLSHMEM_INNER_ERROR, "new output changed during old recreation"};
    status = old.prepare(ROUNDS + 2, SLOT);
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(2);
    if (!status.ok()) {
        return status;
    }
    status = old.launch();
    if (!status.ok()) {
        return status;
    }
    status = old.wait();
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(2);
    if (!status.ok()) {
        return status;
    }
    status = old.verify();
    if (!status.ok()) {
        return status;
    }
    status = fresh.read(current);
    if (!status.ok()) {
        return status;
    }
    if (current != retained)
        return {ACLSHMEM_INNER_ERROR, "new output changed during recreated instance use"};
    status = old.destroy();
    if (!status.ok()) {
        return status;
    }
    status = fresh.destroy();
    if (!status.ok()) {
        return status;
    }
    if (!status.check(aclFinalize(), "aclFinalize")) {
        return status;
    }

    return {};
}

void run_lifecycle(int rank, int ranks)
{
    run_rank(rank, [&] { return run_lifecycle_impl(rank, ranks); });
}

#endif

void run_routing_test()
{
#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
    if (test_first_rank != 0 || test_global_ranks != test_gnpu_num ||
        (test_gnpu_num != 2 && test_gnpu_num != 4 && test_gnpu_num != 8)) {
        GTEST_SKIP() << "Requires one host with 2, 4 or 8 ranks, starting at rank 0";
    }
    // Honor an explicit, launcher-checked range; otherwise use the UT default.
    const char* configured_ports = std::getenv("SHMEM_INSTANCE_PORT_RANGE");
    const std::string ports = configured_ports != nullptr ? configured_ports : get_multi_instance_port_range();
    ScopedEnv port_range("SHMEM_INSTANCE_PORT_RANGE", ports.c_str());
    std::printf("INSTANCE_TEST_PORT_RANGE=%s\n", ports.c_str());
    std::fflush(stdout);
    test_mutil_task([](int rank, int ranks, uint64_t) { run_lifecycle(rank, ranks); }, 64ULL << 20, test_gnpu_num);
#else
    GTEST_SKIP() << "Requires Ascend950 and ACLSHMEM_SIMT_SUPPORT";
#endif
}
} // namespace

TEST(TestSimtInstanceRouting, ReferenceRejectsCorruption)
{
    std::vector<int32_t> data(3 * SLOT, SENTINEL);
    for (uint32_t i = 0; i < SLOT; ++i)
        data[i] = value(0, 1, 3, i);
    for (uint32_t i = 0; i < 256; ++i)
        data[SLOT + i] = value(1, 1, 3, i);
    ASSERT_TRUE(payload_matches(data, 0, 2, 1, 3, 256));
    for (uint32_t index : {0U, SLOT, SLOT + 256, 2 * SLOT, 3 * SLOT - 1}) {
        data[index] ^= 1;
        EXPECT_FALSE(payload_matches(data, 0, 2, 1, 3, 256)) << "undetected corruption at " << index;
        data[index] ^= 1;
    }
}

TEST(TestSimtInstanceRouting, SimtLifecycle) { run_routing_test(); }
