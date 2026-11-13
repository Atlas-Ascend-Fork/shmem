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
#include <stdexcept>
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
void check(int status, const char* operation)
{
    if (status != 0)
        throw std::runtime_error(std::string(operation) + " returned " + std::to_string(status));
}

// Host bootstrap rendezvous, as used by the existing heap tests. This is not
// a DMA fence: both streams must complete before the post-transfer barrier.
void rendezvous(uint64_t id)
{
    check(aclshmemx_instance_ctx_set(id), "select barrier instance");
    check(aclshmemi_control_barrier_all(), "control barrier");
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

    void create(uint64_t bytes)
    {
        aclshmemx_init_attr_t attr{};
        attr.my_pe = rank;
        attr.n_pes = ranks;
        attr.local_mem_size = bytes;
        attr.instance_id = id;
        std::strcpy(attr.ip_port, "tcp://127.0.0.1:0");
        attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
        check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "init instance");
        check(aclshmemx_instance_ctx_set(id), "select allocation instance");
        heap = static_cast<int32_t*>(aclshmem_malloc(BYTES));
        if (heap == nullptr)
            throw std::runtime_error("SHMEM allocation failed");
        check(
            aclrtMalloc(reinterpret_cast<void**>(&identity), 2 * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY),
            "identity allocation");
        check(aclrtCreateStream(&stream), "create stream");
    }

    void prepare(int next_round, uint32_t n)
    {
        round = next_round;
        count = n;
        std::vector<int32_t> input(3 * SLOT, SENTINEL);
        for (uint32_t i = 0; i < SLOT; ++i)
            input[i] = value(rank, id, round, i);
        check(aclrtMemcpy(heap, BYTES, input.data(), BYTES, ACL_MEMCPY_HOST_TO_DEVICE), "prepare payload");
        check(aclrtMemset(identity, 2 * sizeof(int32_t), 0xff, 2 * sizeof(int32_t)), "prepare identity");
    }
    void launch() { simt_instance_routing_transfer(stream, heap, count, (rank + 1) % ranks, id, identity); }
    void wait() { check(aclrtSynchronizeStream(stream), "stream completion"); }
    std::vector<int32_t> read()
    {
        std::vector<int32_t> data(3 * SLOT);
        check(aclrtMemcpy(data.data(), BYTES, heap, BYTES, ACL_MEMCPY_DEVICE_TO_HOST), "read payload");
        return data;
    }
    void verify()
    {
        if (!payload_matches(read(), rank, ranks, id, round, count)) {
            throw std::runtime_error(
                "payload/guard mismatch instance=" + std::to_string(id) + " round=" + std::to_string(round));
        }
        int32_t ids[2];
        check(aclrtMemcpy(ids, sizeof(ids), identity, sizeof(ids), ACL_MEMCPY_DEVICE_TO_HOST), "read identity");
        if (ids[0] != rank || ids[1] != ranks)
            throw std::runtime_error("instance PE query mismatch");
    }
    void destroy()
    {
        wait();
        rendezvous(id);
        aclshmem_free(heap);
        check(aclshmemx_finalize(id), "finalize instance");
        check(aclrtDestroyStream(stream), "destroy stream");
        check(aclrtFree(identity), "free identity");
        heap = nullptr;
    }
};

void run_lifecycle(int rank, int ranks)
{
    // Bound a failed child even if another rank is blocked in initialization
    // or a collective. The existing parent runner reports SIGALRM as failure.
    std::signal(SIGALRM, SIG_DFL);
    alarm(300);
    try {
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(rank % test_gnpu_num + test_first_npu), "set device");
        check(aclshmemx_set_conf_store_tls(false, nullptr, 0), "disable test TLS");
        instance_t old{1, rank, ranks}, fresh{2, rank, ranks};
        old.create(64ULL << 20);
        old.prepare(0, 256);
        rendezvous(1);
        old.launch();
        old.wait();
        rendezvous(1);
        old.verify();
        auto retained = old.read();
        fresh.create(128ULL << 20);
        if (old.read() != retained)
            throw std::runtime_error("old output changed during new instance creation");
        if (old.heap == fresh.heap)
            throw std::runtime_error("live heaps alias");
        for (int round = 1; round <= ROUNDS; ++round) {
            const uint32_t count = round % 2 ? 256 : SLOT;
            old.prepare(round, count);
            fresh.prepare(round, count);
            rendezvous(2);
            // Both submissions precede both waits, preserving stream overlap.
            old.launch();
            fresh.launch();
            old.wait();
            fresh.wait();
            rendezvous(2);
            old.verify();
            fresh.verify();
        }
        retained = fresh.read();
        old.destroy();
        if (fresh.read() != retained)
            throw std::runtime_error("new output changed during old finalize");
        fresh.prepare(ROUNDS + 1, SLOT);
        rendezvous(2);
        fresh.launch();
        fresh.wait();
        rendezvous(2);
        fresh.verify();
        retained = fresh.read();
        old.create(96ULL << 20);
        if (fresh.read() != retained)
            throw std::runtime_error("new output changed during old recreation");
        old.prepare(ROUNDS + 2, SLOT);
        rendezvous(2);
        old.launch();
        old.wait();
        rendezvous(2);
        old.verify();
        if (fresh.read() != retained)
            throw std::runtime_error("new output changed during recreated instance use");
        old.destroy();
        fresh.destroy();
        check(aclFinalize(), "aclFinalize");
        alarm(0);
    } catch (const std::exception& error) {
        // No collective cleanup after one rank fails. test_mutil_task reports
        // the GoogleTest failure to the parent via the child's exit signal.
        ADD_FAILURE() << "rank=" << rank << ": " << error.what();
    }
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
