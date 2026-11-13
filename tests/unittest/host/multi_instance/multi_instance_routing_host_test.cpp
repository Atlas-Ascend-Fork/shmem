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
#include <gtest/gtest-spi.h>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <exception>
#include <system_error>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "shmem.h"
#include "init/shmemi_init.h"
#include "unittest_main_test.h"
#include "unittest/utils/scope_env.h"
#if defined(ACLSHMEM_UT_ASCEND950)
#include "utils/under_api/dl_hcomm_api.h"
#endif

#if defined(ACLSHMEM_UT_ASCEND950)
extern "C" void multi_instance_routing_transfer(void*, void*, uint32_t, int, uint64_t, void*);
extern "C" void multi_instance_mte_transfer(void*, void*, uint32_t, int, uint64_t, void*);
extern "C" void multi_instance_id_boundary(void*, void*);
extern "C" void multi_instance_host_quiet_produce(void*, void*, uint32_t, int, uint64_t, void*);
extern "C" void multi_instance_host_quiet_observe(void*, int, uint64_t, void*);
#endif

namespace {
constexpr uint32_t SLOT = 4096;
constexpr int32_t SENTINEL = 0x5a5a5a5a;
constexpr size_t BYTES = 3 * SLOT * sizeof(int32_t);
constexpr int ROUNDS = 20;

int32_t value(int rank, int id, int round, uint32_t index)
{
    return 1000000 * id + 100000 * round + 5000 * rank + static_cast<int>(index);
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

bool host_quiet_completed(const int32_t* queues, uint64_t instance_id = 1)
{
    for (int qp = 0; qp < 2; ++qp) {
        const int32_t before = queues[2 + qp * 2], target = queues[3 + qp * 2];
        const int32_t expected_tail = instance_id == 0 && qp == 1 ? before : target;
        if (target <= before || queues[6 + qp * 2] != expected_tail || queues[7 + qp * 2] != target)
            return false;
    }
    return true;
}

struct test_status_t {
    int rc = 0;
    std::string operation;

    bool ok() const { return rc == 0; }
    bool check(int status, const char* name)
    {
        if (status != 0) {
            rc = status;
            operation = name;
        }
        return status == 0;
    }
};

// Ordinary API failures travel as values. Standard-library exceptions are
// caught separately, and every worker exit wakes a peer waiting for submission.
test_status_t run_worker_pair(
    const std::function<test_status_t(int)>& produce, const std::function<test_status_t(int)>& complete)
{
    std::mutex mutex;
    std::condition_variable changed;
    bool started = false;
    bool cancelled = false;
    bool submitted[2] = {false, false};
    test_status_t results[2];
    test_status_t creation_status;
    std::vector<std::thread> workers;
    workers.reserve(2);
    auto work = [&](int index) -> test_status_t {
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&] { return started; });
            if (cancelled)
                return {};
        }
        auto status = produce(index);
        if (!status.ok())
            return status;
        {
            std::unique_lock<std::mutex> lock(mutex);
            submitted[index] = true;
            changed.notify_all();
            changed.wait(lock, [&] { return cancelled || submitted[1 - index]; });
            if (cancelled)
                return {}; // The failing worker retains the original operation/rc.
        }
        return complete(index);
    };
    try {
        for (int index = 0; index < 2; ++index) {
            workers.emplace_back([&, index] {
                test_status_t status;
                try {
                    status = work(index);
                } catch (const std::exception& error) {
                    status = {ACLSHMEM_INNER_ERROR, std::string("C++ worker exception: ") + error.what()};
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    results[index] = std::move(status);
                    if (!results[index].ok())
                        cancelled = true;
                }
                changed.notify_all();
            });
        }
    } catch (const std::exception& error) {
        creation_status = {ACLSHMEM_INNER_ERROR, std::string("C++ thread creation exception: ") + error.what()};
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = cancelled || !creation_status.ok();
        started = true;
    }
    changed.notify_all();
    for (auto& worker : workers)
        worker.join();
    if (!creation_status.ok())
        return creation_status;
    for (const auto& status : results) {
        if (!status.ok())
            return status;
    }
    return {};
}

#if defined(ACLSHMEM_UT_ASCEND950)
void run_rank(int rank, const std::function<test_status_t()>& task)
{
    // A failed rank must not enter collective cleanup while another rank may
    // be waiting. The child runner reports failure and bounds peers by alarm.
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

struct instance_fixture_t {
    uint64_t id;
    int rank;
    int ranks;
    int32_t* heap = nullptr;
    int32_t* identity = nullptr;
    aclrtStream stream = nullptr;
    int round = -1;
    uint32_t count = 0;
    bool mte_only = false;
    bool host_quiet = false;
    aclrtEvent ready = nullptr;
    aclrtEvent done = nullptr;

    test_status_t create(uint64_t bytes, bool shared)
    {
        aclshmemx_init_attr_t attr{};
        attr.my_pe = rank;
        attr.n_pes = ranks;
        attr.local_mem_size = bytes;
        attr.instance_id = id;
        // Instance 0 uses the launcher endpoint directly; only nonzero IDs
        // replace port 0 using SHMEM_INSTANCE_PORT_RANGE.
        snprintf(attr.ip_port, sizeof(attr.ip_port), "%s", id == 0 ? test_global_ipport : "tcp://127.0.0.1:0");
        attr.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(
            mte_only ? ACLSHMEM_DATA_OP_MTE : ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA);
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
                aclrtMalloc(reinterpret_cast<void**>(&identity), 10 * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY),
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
        if (!status.check(
                aclrtMemset(identity, 10 * sizeof(int32_t), 0xff, 10 * sizeof(int32_t)), "prepare identity")) {
            return status;
        }

        return {};
    }
    void launch()
    {
        if (host_quiet) {
            multi_instance_host_quiet_produce(stream, heap, count, (rank + 1) % ranks, id, identity);
        } else if (mte_only) {
            multi_instance_mte_transfer(stream, heap, count, (rank + 1) % ranks, id, identity);
        } else {
            multi_instance_routing_transfer(stream, heap, count, (rank + 1) % ranks, id, identity);
        }
    }
    test_status_t enqueue_host_quiet()
    {
        if (!host_quiet)
            return {};
        // All workers share this lock across selection and the original quiet call.
        static std::mutex selection_mutex;
        std::lock_guard<std::mutex> lock(selection_mutex);
        test_status_t status;
        if (!status.check(aclshmemx_instance_ctx_set(id), "select Host quiet instance")) {
            return status;
        }
        aclrtStream completion_stream = stream;
        if (round % 3 != 0) {
            // Cross-stream dependencies are explicit; quiet is not a barrier
            // for unrelated streams. Events live through both stream waits.
            completion_stream = g_state_host.default_stream;
            if (!status.check(aclrtCreateEvent(&ready), "create producer event")) {
                return status;
            }
            if (!status.check(aclrtCreateEvent(&done), "create quiet event")) {
                return status;
            }
            if (!status.check(aclrtRecordEvent(ready, stream), "record producer event")) {
                return status;
            }
            if (!status.check(aclrtStreamWaitEvent(completion_stream, ready), "order default stream")) {
                return status;
            }
        }
        if (round % 3 == 2) {
            aclshmem_quiet();
        } else {
            aclshmemx_quiet_on_stream(round % 3 == 1 ? nullptr : stream);
            aclshmemx_quiet_on_stream(round % 3 == 1 ? nullptr : stream); // Already drained is a no-op.
        }
        multi_instance_host_quiet_observe(completion_stream, (rank + 1) % ranks, id, identity);
        if (done != nullptr) {
            if (!status.check(aclrtRecordEvent(done, completion_stream), "record quiet completion")) {
                return status;
            }
            if (!status.check(aclrtStreamWaitEvent(stream, done), "order result stream")) {
                return status;
            }
        }

        return {};
    }
    test_status_t wait()
    {
        test_status_t status;
        if (!status.check(aclrtSynchronizeStream(stream), "stream completion")) {
            return status;
        }
        if (ready != nullptr) {
            if (!status.check(aclrtDestroyEvent(ready), "destroy producer event")) {
                return status;
            }
            if (!status.check(aclrtDestroyEvent(done), "destroy quiet event")) {
                return status;
            }
            ready = done = nullptr;
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
        std::vector<int32_t> data;
        auto status = read(data);
        if (!status.ok())
            return status;
        if (!payload_matches(data, rank, ranks, id, round, count)) {
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
        if (host_quiet) {
            int32_t queues[10];
            if (!status.check(
                    aclrtMemcpy(queues, sizeof(queues), identity, sizeof(queues), ACL_MEMCPY_DEVICE_TO_HOST),
                    "read Host quiet queue evidence")) {
                return status;
            }
            if (!host_quiet_completed(queues, id))
                return {ACLSHMEM_INNER_ERROR, "Host quiet CQ completion scope mismatch"};
        }

        return {};
    }
    test_status_t destroy()
    {
        auto status = wait();
        if (!status.ok())
            return status;
        status = rendezvous(id);
        if (!status.ok())
            return status;
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

// Both producers are submitted before either worker enqueues quiet.
test_status_t run_threaded_pair(instance_fixture_t& first, instance_fixture_t& second)
{
    aclrtContext context = nullptr;
    test_status_t status;
    if (!status.check(aclrtGetCurrentContext(&context), "capture ACL context"))
        return status;
    instance_fixture_t* instances[] = {&first, &second};
    return run_worker_pair(
        [&](int index) -> test_status_t {
            test_status_t worker_status;
            if (!worker_status.check(aclrtSetCurrentContext(context), "worker ACL context"))
                return worker_status;
            instances[index]->launch();
            return {};
        },
        [&](int index) -> test_status_t {
            auto worker_status = instances[index]->enqueue_host_quiet();
            if (!worker_status.ok())
                return worker_status;
            return instances[index]->wait();
        });
}

test_status_t run_lifecycle_impl(
    int rank, int ranks, bool shared, uint64_t old_id, uint64_t fresh_id, bool host_quiet, bool threaded_quiet)
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
    if (!status.check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, shared), "shared Jetty configuration")) {
        return status;
    }
    if (!status.check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, host_quiet ? 2 : 1), "set QP count")) {
        return status;
    }
    instance_fixture_t old{old_id, rank, ranks}, fresh{fresh_id, rank, ranks};
    old.host_quiet = fresh.host_quiet = host_quiet;
    status = old.create(64ULL << 20, shared);
    if (!status.ok())
        return status;
    if (host_quiet) {
        if (!status.check(aclshmemx_instance_ctx_set(old.id), "select empty quiet instance")) {
            return status;
        }
        aclshmemx_quiet_on_stream(old.stream);
        status = old.wait();
        if (!status.ok())
            return status;
    }
    status = old.prepare(0, 256);
    if (!status.ok())
        return status;
    status = rendezvous(old.id);
    if (!status.ok())
        return status;
    old.launch();
    status = old.enqueue_host_quiet();
    if (!status.ok())
        return status;
    status = old.wait();
    if (!status.ok())
        return status;
    status = rendezvous(old.id);
    if (!status.ok())
        return status;
    status = old.verify();
    if (!status.ok())
        return status;
    std::vector<int32_t> retained, observed;
    status = old.read(retained);
    if (!status.ok())
        return status;
    status = fresh.create(128ULL << 20, shared);
    if (!status.ok())
        return status;
    status = old.read(observed);
    if (!status.ok())
        return status;
    if (observed != retained)
        return {ACLSHMEM_INNER_ERROR, "old output changed during new instance creation"};
    if (old.heap == fresh.heap)
        return {ACLSHMEM_INNER_ERROR, "live heaps alias"};
    for (int round = 1; round <= ROUNDS; ++round) {
        const uint32_t count = round % 2 ? 256 : SLOT;
        status = old.prepare(round, count);
        if (!status.ok())
            return status;
        status = fresh.prepare(round, count);
        if (!status.ok())
            return status;
        status = rendezvous(fresh.id);
        if (!status.ok())
            return status;
        if (threaded_quiet) {
            // Producers overlap; the caller serializes set plus quiet.
            status = run_threaded_pair(old, fresh);
            if (!status.ok())
                return status;
        } else {
            // Both submissions precede both waits, preserving stream overlap.
            old.launch();
            fresh.launch();
            status = old.enqueue_host_quiet();
            if (!status.ok())
                return status;
            status = fresh.enqueue_host_quiet();
            if (!status.ok())
                return status;
            if (host_quiet) {
                if (!status.check(aclshmemx_instance_ctx_set(old.id), "switch Host context after enqueue")) {
                    return status;
                }
            }
            status = old.wait();
            if (!status.ok())
                return status;
            status = fresh.wait();
            if (!status.ok())
                return status;
        }
        status = rendezvous(fresh.id);
        if (!status.ok())
            return status;
        status = old.verify();
        if (!status.ok())
            return status;
        status = fresh.verify();
        if (!status.ok())
            return status;
    }
    status = fresh.read(retained);
    if (!status.ok())
        return status;
    status = old.destroy();
    if (!status.ok())
        return status;
    status = fresh.read(observed);
    if (!status.ok())
        return status;
    if (observed != retained)
        return {ACLSHMEM_INNER_ERROR, "new output changed during old finalize"};
    status = fresh.prepare(ROUNDS + 1, SLOT);
    if (!status.ok())
        return status;
    status = rendezvous(fresh.id);
    if (!status.ok())
        return status;
    fresh.launch();
    status = fresh.enqueue_host_quiet();
    if (!status.ok())
        return status;
    status = fresh.wait();
    if (!status.ok())
        return status;
    status = rendezvous(fresh.id);
    if (!status.ok())
        return status;
    status = fresh.verify();
    if (!status.ok())
        return status;
    status = fresh.read(retained);
    if (!status.ok())
        return status;
    status = old.create(96ULL << 20, shared);
    if (!status.ok())
        return status;
    status = fresh.read(observed);
    if (!status.ok())
        return status;
    if (observed != retained)
        return {ACLSHMEM_INNER_ERROR, "new output changed during old recreation"};
    status = old.prepare(ROUNDS + 2, SLOT);
    if (!status.ok())
        return status;
    status = rendezvous(fresh.id);
    if (!status.ok())
        return status;
    old.launch();
    status = old.enqueue_host_quiet();
    if (!status.ok())
        return status;
    status = old.wait();
    if (!status.ok())
        return status;
    status = rendezvous(fresh.id);
    if (!status.ok())
        return status;
    status = old.verify();
    if (!status.ok())
        return status;
    status = fresh.read(observed);
    if (!status.ok())
        return status;
    if (observed != retained)
        return {ACLSHMEM_INNER_ERROR, "new output changed during recreated instance use"};
    status = old.destroy();
    if (!status.ok())
        return status;
    status = fresh.destroy();
    if (!status.ok())
        return status;
    if (!status.check(aclFinalize(), "aclFinalize")) {
        return status;
    }
    std::printf(
        "UDMA_LIFECYCLE rank=%d shared=%d old_id=%lu new_id=%lu rounds=%d "
        "coexist_retention_recreate=PASS\n",
        rank, shared, old_id, fresh_id, ROUNDS);
    std::fflush(stdout);
    if (host_quiet) {
        std::printf(
            "HOST_QUIET rank=%d shared=%d old_id=%lu new_id=%lu qps=2 rounds=%d "
            "payload_guard_cq_default_stream_switch_recreate=PASS\n",
            rank, shared, old_id, fresh_id, ROUNDS);
        std::fflush(stdout);
    }
    if (threaded_quiet) {
        std::printf(
            "HOST_QUIET_SERIALIZED rank=%d shared=%d old_id=%lu new_id=%lu threads=2 qps=2 rounds=%d "
            "payload_guard_cq_serialized_selection_recreate=PASS\n",
            rank, shared, old_id, fresh_id, ROUNDS);
        std::fflush(stdout);
    }
    return {};
}

void run_lifecycle(
    int rank, int ranks, bool shared, uint64_t old_id, uint64_t fresh_id, bool host_quiet, bool threaded_quiet)
{
    run_rank(
        rank, [&] { return run_lifecycle_impl(rank, ranks, shared, old_id, fresh_id, host_quiet, threaded_quiet); });
}
#endif

void run_routing_test(
    bool shared, uint64_t old_id = 1, uint64_t fresh_id = 2, bool host_quiet = false, bool threaded_quiet = false)
{
#if defined(ACLSHMEM_UT_ASCEND950)
#if defined(ACLSHMEM_RELAY_SUPPORT)
    if (host_quiet)
        GTEST_SKIP() << "Explicit QP producer requires direct UDMA mode";
#endif
    if (test_first_rank != 0 || test_global_ranks != test_gnpu_num ||
        (test_gnpu_num != 2 && test_gnpu_num != 4 && test_gnpu_num != 8)) {
        GTEST_SKIP() << "Requires one host with 2, 4 or 8 ranks, starting at rank 0";
    }
    if (shared) {
#if defined(ACLSHMEM_RELAY_SUPPORT)
        constexpr bool relay = true;
#else
        constexpr bool relay = false;
#endif
        if (relay)
            GTEST_SKIP() << "Shared Jetty is unsupported with relay";
        ASSERT_EQ(shm::DlHcommApi::LoadLibrary(), ACLSHMEM_SUCCESS);
        if (!shm::DlHcommApi::HasChannelConfigApis())
            GTEST_SKIP() << "Shared Jetty requires optional HCOMM channel-config APIs";
    }
    // Honor an explicit, launcher-checked range; otherwise use the UT default.
    const char* configured_ports = std::getenv("SHMEM_INSTANCE_PORT_RANGE");
    const std::string ports = configured_ports != nullptr ? configured_ports : get_multi_instance_port_range();
    ScopedEnv port_range("SHMEM_INSTANCE_PORT_RANGE", ports.c_str());
    std::printf("INSTANCE_TEST_PORT_RANGE=%s\n", ports.c_str());
    std::fflush(stdout);
    test_mutil_task(
        [shared, old_id, fresh_id, host_quiet, threaded_quiet](int rank, int ranks, uint64_t) {
            run_lifecycle(rank, ranks, shared, old_id, fresh_id, host_quiet, threaded_quiet);
        },
        64ULL << 20, test_gnpu_num);
#else
    GTEST_SKIP() << "Requires Ascend950";
#endif
}
} // namespace

TEST(TestMultiInstanceRouting, ApiFailurePreservesOperationAndCode)
{
    test_status_t status;
    EXPECT_TRUE(status.check(0, "successful API"));
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(status.check(12345, "injected ACL API failure"));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.rc, 12345);
    EXPECT_EQ(status.operation, "injected ACL API failure");
}

TEST(TestMultiInstanceRouting, WorkerPairSuccessJoinsBoth)
{
    std::atomic<int> produced{0}, completed{0};
    auto status = run_worker_pair(
        [&](int) -> test_status_t {
            ++produced;
            return {};
        },
        [&](int) -> test_status_t {
            ++completed;
            return {};
        });
    EXPECT_TRUE(status.ok()) << status.operation;
    EXPECT_EQ(produced.load(), 2);
    EXPECT_EQ(completed.load(), 2);
}

TEST(TestMultiInstanceRouting, WorkerProducerFailureUnblocksPeer)
{
    for (int failing_worker : {0, 1}) {
        std::atomic<int> completed{0};
        auto status = run_worker_pair(
            [=](int index) -> test_status_t {
                if (index == failing_worker)
                    return {12345, "injected producer API failure"};
                return {};
            },
            [&](int) -> test_status_t {
                ++completed;
                return {};
            });
        EXPECT_EQ(status.rc, 12345);
        EXPECT_EQ(status.operation, "injected producer API failure");
        EXPECT_EQ(completed.load(), 0);
    }
}

TEST(TestMultiInstanceRouting, WorkerCompletionFailurePreservesStatus)
{
    for (int failing_worker : {0, 1}) {
        auto status = run_worker_pair(
            [](int) -> test_status_t { return {}; },
            [=](int index) -> test_status_t {
                if (index == failing_worker)
                    return {23456, "injected completion API failure"};
                return {};
            });
        EXPECT_EQ(status.rc, 23456);
        EXPECT_EQ(status.operation, "injected completion API failure");
    }
}

TEST(TestMultiInstanceRouting, WorkerStandardExceptionUnblocksPeer)
{
    auto status = run_worker_pair(
        [](int index) -> test_status_t {
            if (index == 0)
                throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
            return {};
        },
        [](int) -> test_status_t { return {}; });
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.operation.find("C++ worker exception:"), std::string::npos);
}

TEST(TestMultiInstanceRouting, RankDriverReportsExplicitApiFailure)
{
#if defined(ACLSHMEM_UT_ASCEND950)
    EXPECT_NONFATAL_FAILURE(
        run_rank(7, []() -> test_status_t { return {12345, "injected rank API failure"}; }),
        "rank=7 operation=injected rank API failure rc=12345");
    alarm(0);
#else
    GTEST_SKIP() << "Rank driver requires Ascend950";
#endif
}

TEST(TestMultiInstanceRouting, ReferenceRejectsCorruption)
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

TEST(TestMultiInstanceRouting, RejectsOutOfRangeInstanceIds)
{
    // Reject before bootstrap, entity creation or any device kernel launch.
    auto* selected = g_instance_ctx;
    const uint64_t selected_id = selected->id;
    uint8_t uid_args = 0;
    const uint64_t invalid_ids[] = {256, 32768, UINT32_MAX, (1ULL << 32), UINT64_MAX};
    for (uint64_t id : invalid_ids) {
        aclshmemx_init_attr_t attr{};
        attr.instance_id = id;
        EXPECT_EQ(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), ACLSHMEM_INVALID_VALUE);
        attr.comm_args = &uid_args;
        EXPECT_EQ(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr), ACLSHMEM_INVALID_VALUE);
        EXPECT_EQ(aclshmemx_instance_ctx_get(), selected);
        EXPECT_EQ(selected->id, selected_id);

        // Exercise the legacy/current-instance entry before its initialized-state check.
        aclshmem_instance_ctx invalid_context{id, nullptr};
        g_instance_ctx = &invalid_context;
        aclrtStream stream = nullptr;
        EXPECT_EQ(aclshmemi_enqueue_current_instance_quiet(stream), ACLSHMEM_INVALID_PARAM);
        g_instance_ctx = selected;
        EXPECT_EQ(stream, nullptr);
    }
}

TEST(TestMultiInstanceRouting, HostQuietReferenceRejectsIncomplete)
{
    const int32_t complete[10] = {0, 8, 3, 4, 5, 6, 4, 4, 6, 6};
    EXPECT_TRUE(host_quiet_completed(complete));
    const int32_t default_scope[10] = {0, 8, 3, 4, 5, 6, 4, 4, 5, 6};
    EXPECT_TRUE(host_quiet_completed(default_scope, 0));
    EXPECT_FALSE(host_quiet_completed(complete, 0)) << "undetected QP1 consumption by default quiet";
    EXPECT_FALSE(host_quiet_completed(default_scope, 1)) << "undetected incomplete nonzero-instance QP1";
    for (int index : {3, 5, 6, 7, 8, 9}) {
        int32_t incomplete[10];
        std::memcpy(incomplete, complete, sizeof(complete));
        --incomplete[index];
        EXPECT_FALSE(host_quiet_completed(incomplete)) << "undetected incomplete CQ at " << index;
    }
}

TEST(TestMultiInstanceRouting, UdmaIndependentLifecycle) { run_routing_test(false); }
TEST(TestMultiInstanceRouting, UdmaSharedLifecycle) { run_routing_test(true); }

TEST(TestMultiInstanceRouting, HostQuietIndependentLifecycle) { run_routing_test(false, 1, 2, true); }
TEST(TestMultiInstanceRouting, HostQuietSharedLifecycle) { run_routing_test(true, 1, 2, true); }
TEST(TestMultiInstanceRouting, HostQuietDefaultInstance) { run_routing_test(false, 0, 1, true); }
TEST(TestMultiInstanceRouting, HostQuietSharedDefaultInstance) { run_routing_test(true, 0, 1, true); }

TEST(TestMultiInstanceRouting, HostQuietSerializedIndependentThreads) { run_routing_test(false, 1, 2, true, true); }
TEST(TestMultiInstanceRouting, HostQuietSerializedSharedThreads) { run_routing_test(true, 1, 2, true, true); }
TEST(TestMultiInstanceRouting, HostQuietSerializedDefaultThreads) { run_routing_test(false, 0, 1, true, true); }
TEST(TestMultiInstanceRouting, HostQuietSerializedSharedDefaultThreads) { run_routing_test(true, 0, 1, true, true); }

TEST(TestMultiInstanceRouting, HostQuietSerializedHbmBoundaryIndependentThreads)
{
#if defined(ACLSHMEM_UT_ASCEND950) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    run_routing_test(false, 255, 1, true, true);
#else
    GTEST_SKIP() << "Requires Ascend950 with ordinary HBM heaps";
#endif
}

TEST(TestMultiInstanceRouting, HostQuietSerializedHbmBoundarySharedThreads)
{
#if defined(ACLSHMEM_UT_ASCEND950) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    run_routing_test(true, 255, 1, true, true);
#else
    GTEST_SKIP() << "Requires Ascend950 with ordinary HBM heaps";
#endif
}

TEST(TestMultiInstanceRouting, HostQuietHbmBoundaryIndependentLifecycle)
{
#if defined(ACLSHMEM_UT_ASCEND950) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    run_routing_test(false, 255, 1, true);
#else
    GTEST_SKIP() << "Requires Ascend950 with ordinary HBM heaps";
#endif
}

TEST(TestMultiInstanceRouting, HostQuietHbmBoundarySharedLifecycle)
{
#if defined(ACLSHMEM_UT_ASCEND950) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    run_routing_test(true, 255, 1, true);
#else
    GTEST_SKIP() << "Requires Ascend950 with ordinary HBM heaps";
#endif
}

TEST(TestMultiInstanceRouting, HbmOnlyUdmaIndependentBoundaryLifecycle)
{
#if defined(ACLSHMEM_UT_ASCEND950) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    run_routing_test(false, 255, 1);
#else
    GTEST_SKIP() << "Requires Ascend950 with HBM-only ordinary heaps; Fabric also needs DRAM entity 511";
#endif
}

TEST(TestMultiInstanceRouting, HbmOnlyUdmaSharedBoundaryLifecycle)
{
#if defined(ACLSHMEM_UT_ASCEND950) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    run_routing_test(true, 255, 1);
#else
    GTEST_SKIP() << "Requires Ascend950 with HBM-only ordinary heaps; Fabric also needs DRAM entity 511";
#endif
}

TEST(TestMultiInstanceRouting, InstanceIdBoundary)
{
#if defined(ACLSHMEM_UT_ASCEND950)
    if (test_first_rank != 0 || test_global_ranks != test_gnpu_num ||
        (test_gnpu_num != 2 && test_gnpu_num != 4 && test_gnpu_num != 8)) {
        GTEST_SKIP() << "Requires one host with 2, 4 or 8 ranks, starting at rank 0";
    }
    test_mutil_task(
        [](int rank, int, uint64_t) {
            run_rank(rank, [&]() -> test_status_t {
                test_status_t status;
                if (!status.check(aclInit(nullptr), "aclInit")) {
                    return status;
                }
                if (!status.check(aclrtSetDevice(rank % test_gnpu_num + test_first_npu), "set device")) {
                    return status;
                }
                aclrtStream stream = nullptr;
                void* result = nullptr;
                if (!status.check(aclrtCreateStream(&stream), "create stream")) {
                    return status;
                }
                if (!status.check(
                        aclrtMalloc(&result, 10 * sizeof(uint64_t), ACL_MEM_MALLOC_NORMAL_ONLY),
                        "boundary allocation")) {
                    return status;
                }
                if (!status.check(
                        aclrtMemset(result, 10 * sizeof(uint64_t), 0xa5, 10 * sizeof(uint64_t)), "boundary sentinel")) {
                    return status;
                }
                multi_instance_id_boundary(stream, result);
                if (!status.check(aclrtSynchronizeStream(stream), "boundary completion")) {
                    return status;
                }
                uint64_t actual[10];
                const uint64_t expected[] = {1, 0, 1, 508, 1, 510, 0, UINT64_MAX, 0, UINT64_MAX};
                if (!status.check(
                        aclrtMemcpy(actual, sizeof(actual), result, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST),
                        "boundary result")) {
                    return status;
                }
                for (size_t i = 0; i < 10; ++i) {
                    if (actual[i] != expected[i])
                        return {ACLSHMEM_INNER_ERROR, "instance boundary mismatch at " + std::to_string(i)};
                }
                std::printf(
                    "INSTANCE_BOUNDARY rank=%d valid=0,254,255 invalid=256,UINT64_MAX entities=0,508,510 PASS\n", rank);
                std::fflush(stdout);
                if (!status.check(aclrtFree(result), "free boundary allocation")) {
                    return status;
                }
                if (!status.check(aclrtDestroyStream(stream), "destroy stream")) {
                    return status;
                }
                if (!status.check(aclFinalize(), "aclFinalize")) {
                    return status;
                }
                return {};
            });
        },
        64ULL << 20, test_gnpu_num);
#else
    GTEST_SKIP() << "Requires Ascend950";
#endif
}

TEST(TestMultiInstanceRouting, HbmOnlyBoundaryLifecycle)
{
#if defined(ACLSHMEM_UT_ASCEND950) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    if (test_first_rank != 0 || test_global_ranks != test_gnpu_num ||
        (test_gnpu_num != 2 && test_gnpu_num != 4 && test_gnpu_num != 8)) {
        GTEST_SKIP() << "Requires one host with 2, 4 or 8 ranks, starting at rank 0";
    }
    const char* configured_ports = std::getenv("SHMEM_INSTANCE_PORT_RANGE");
    const std::string ports = configured_ports != nullptr ? configured_ports : get_multi_instance_port_range();
    ScopedEnv port_range("SHMEM_INSTANCE_PORT_RANGE", ports.c_str());
    std::printf("INSTANCE_TEST_PORT_RANGE=%s\n", ports.c_str());
    std::fflush(stdout);
    // ID 255 is a sparse ID, not 256 simultaneously live instances. Fabric
    // builds also allocate DRAM entity 511 and cannot initialize this ID.
    test_mutil_task(
        [](int rank, int ranks, uint64_t) {
            run_rank(rank, [&]() -> test_status_t {
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
                for (uint64_t id : {0ULL, 254ULL, 255ULL}) {
                    instance_fixture_t instance{id, rank, ranks};
                    instance.mte_only = true;
                    status = instance.create(64ULL << 20, false);
                    if (!status.ok())
                        return status;
                    for (uint32_t count : {256U, SLOT}) {
                        status = instance.prepare(0, count);
                        if (!status.ok())
                            return status;
                        status = rendezvous(id);
                        if (!status.ok())
                            return status;
                        instance.launch();
                        status = instance.wait();
                        if (!status.ok())
                            return status;
                        status = rendezvous(id);
                        if (!status.ok())
                            return status;
                        status = instance.verify();
                        if (!status.ok())
                            return status;
                        std::printf(
                            "HBM_BOUNDARY rank=%d instance=%lu count=%u payload_identity_guard=PASS\n", rank, id,
                            count);
                        std::fflush(stdout);
                    }
                    status = instance.destroy();
                    if (!status.ok())
                        return status;
                }
                if (!status.check(aclFinalize(), "aclFinalize")) {
                    return status;
                }
                return {};
            });
        },
        64ULL << 20, test_gnpu_num);
#else
    GTEST_SKIP() << "Requires Ascend950 with HBM-only ordinary heaps (ENABLE_CANN_BUILD=OFF)";
#endif
}
