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
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <fstream>
#include <condition_variable>
#include <mutex>
#include <cstdlib>
#include <exception>
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
#include "unittest/simt_instance_selection_test.h"

#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
extern "C" void simt_warp_tail_transfer(void*, void*, uint32_t, int, uint64_t, uint32_t, int);
extern "C" void simt_instance_boundary(void*, void*);
extern "C" void simt_instance_selection_transfer(
    void*, void*, uint32_t, int, uint64_t, void*, void*, void*, int, uint32_t);
#endif

namespace {
using namespace simt_instance_selection_test;
constexpr size_t BYTES = 3 * SLOT * sizeof(int32_t);
constexpr size_t ID_BYTES = LANES * ID_FIELDS * sizeof(uint64_t);
constexpr size_t OBS_BYTES = 2 * SLOT * sizeof(int32_t);

int32_t value(int id, int rank, int round, uint32_t index)
{
    return id * 1000000 + rank * 5000 + round * 100000 + static_cast<int32_t>(index);
}
std::vector<int32_t> reference(int id, int rank, int ranks, int round, uint32_t count, int mode)
{
    std::vector<int32_t> expected(3 * SLOT, SENTINEL);
    const int source = is_get(mode) ? (rank + 1) % ranks : (rank + ranks - 1) % ranks;
    for (uint32_t i = 0; i < SLOT; ++i)
        expected[i] = value(id, rank, round, i);
    for (uint32_t i = 0; i < count; ++i)
        expected[SLOT + i] = value(id, source, round, i);
    return expected;
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

test_status_t rendezvous(uint64_t id)
{
    test_status_t status;
    if (!status.check(aclshmemx_instance_ctx_set(id), "barrier instance")) {
        return status;
    }
    if (!status.check(aclshmemi_control_barrier_all(), "host control barrier")) {
        return status;
    }

    return {};
}
struct instance_t {
    uint64_t id;
    int rank, ranks;
    int32_t *heap = nullptr, *observed = nullptr;
    uint64_t *identity = nullptr, *timing = nullptr;
    aclrtStream stream = nullptr;
    uint64_t ticks[2]{};

    test_status_t create(uint64_t capacity)
    {
        aclshmemx_init_attr_t attr{};
        attr.my_pe = rank;
        attr.n_pes = ranks;
        attr.instance_id = id;
        attr.local_mem_size = capacity;
        // instance_t 0 uses the explicit test endpoint; only nonzero IDs use
        // the instance-port-range allocation rule.
        std::strcpy(attr.ip_port, id == 0 ? test_global_ipport : "tcp://127.0.0.1:0");
        attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
        attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
        test_status_t status;
        if (!status.check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "instance init")) {
            return status;
        }
        if (!status.check(aclshmemx_instance_ctx_set(id), "allocation context")) {
            return status;
        }
        heap = static_cast<int32_t*>(aclshmem_malloc(BYTES));
        if (!heap)
            return {ACLSHMEM_INNER_ERROR, "SHMEM allocation failed"};
        if (!status.check(
                aclrtMalloc(reinterpret_cast<void**>(&identity), ID_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY),
                "identity malloc")) {
            return status;
        }
        if (!status.check(
                aclrtMalloc(reinterpret_cast<void**>(&observed), OBS_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY),
                "CAS result malloc")) {
            return status;
        }
        if (!status.check(
                aclrtMalloc(reinterpret_cast<void**>(&timing), sizeof(ticks), ACL_MEM_MALLOC_NORMAL_ONLY),
                "timing malloc")) {
            return status;
        }
        if (!status.check(aclrtCreateStream(&stream), "create stream")) {
            return status;
        }

        return {};
    }
    test_status_t prepare(int round)
    {
        const auto input = reference(id, rank, ranks, round, 0, 0);
        test_status_t status;
        if (!status.check(aclrtMemcpy(heap, BYTES, input.data(), BYTES, ACL_MEMCPY_HOST_TO_DEVICE), "prepare heap")) {
            return status;
        }
        if (!status.check(aclrtMemset(identity, ID_BYTES, 0xff, ID_BYTES), "prepare identity")) {
            return status;
        }
        if (!status.check(aclrtMemset(observed, OBS_BYTES, 0x5a, OBS_BYTES), "prepare CAS result")) {
            return status;
        }
        if (!status.check(aclrtMemset(timing, sizeof(ticks), 0, sizeof(ticks)), "prepare timing")) {
            return status;
        }

        return {};
    }
    test_status_t launch(uint32_t count, int mode, uint32_t repetitions)
    {
        simt_instance_selection_transfer(
            stream, heap, count, (rank + 1) % ranks, id, identity, observed, timing, mode, repetitions);

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
        if (!status.check(aclrtMemcpy(data.data(), BYTES, heap, BYTES, ACL_MEMCPY_DEVICE_TO_HOST), "read heap")) {
            return status;
        }
        return {};
    }
    test_status_t verify(int scenario, int round, uint32_t count, int mode)
    {
        std::vector<int32_t> actual;
        test_status_t status = read(actual);
        if (!status.ok()) {
            return status;
        }
        const auto expected = reference(id, rank, ranks, round, count, mode);
        for (size_t i = 0; i < actual.size(); ++i)
            if (actual[i] != expected[i]) {
                return {
                    ACLSHMEM_INNER_ERROR, "data mismatch instance=" + std::to_string(id) +
                                              " mode=" + std::to_string(mode) + " round=" + std::to_string(round) +
                                              " index=" + std::to_string(i) + " actual=" + std::to_string(actual[i]) +
                                              " expected=" + std::to_string(expected[i])};
            }
        uint64_t ids[LANES * ID_FIELDS];
        if (!status.check(
                aclrtMemcpy(ids, sizeof(ids), identity, sizeof(ids), ACL_MEMCPY_DEVICE_TO_HOST), "read identity")) {
            return status;
        }
        const uint64_t expected_ids[ID_FIELDS] = {
            1,
            static_cast<uint64_t>(rank),
            static_cast<uint64_t>(ranks),
            static_cast<uint64_t>(rank),
            static_cast<uint64_t>(ranks),
            static_cast<uint64_t>((rank + 1) % ranks),
            static_cast<uint64_t>((rank + 1) % ranks),
            1,
            1,
            1,
            1};
        for (uint32_t lane = 0; lane < LANES; ++lane)
            for (uint32_t f = 0; f < ID_FIELDS; ++f) {
                if (ids[ID_FIELDS * lane + f] != expected_ids[f])
                    return {ACLSHMEM_INNER_ERROR, "instance/team/ptr query mismatch field=" + std::to_string(f)};
            }
        std::vector<int32_t> cas(2 * SLOT);
        if (!status.check(
                aclrtMemcpy(cas.data(), OBS_BYTES, observed, OBS_BYTES, ACL_MEMCPY_DEVICE_TO_HOST),
                "read CAS results")) {
            return status;
        }
        for (uint32_t i = 0; i < 2 * SLOT; ++i) {
            const int32_t want =
                mode == CAS && i >= SLOT && i < SLOT + count ? value(id, rank, round, i - SLOT) : SENTINEL;
            if (cas[i] != want)
                return {ACLSHMEM_INNER_ERROR, "CAS returned value/guard mismatch"};
        }
        if (!status.check(
                aclrtMemcpy(ticks, sizeof(ticks), timing, sizeof(ticks), ACL_MEMCPY_DEVICE_TO_HOST),
                "read timestamps")) {
            return status;
        }
        if (!ticks[0] || ticks[1] <= ticks[0])
            return {ACLSHMEM_INNER_ERROR, "invalid device timing interval"};
        // Optional raw evidence for independent offline checking; no extra runtime operations.
        if (const char* directory = std::getenv("SHMEM_SIMT_INSTANCE_SELECTION_EVIDENCE_DIR")) {
            const std::string base = std::string(directory) + "/scenario" + std::to_string(scenario) + "_mode" +
                                     std::to_string(mode) + "_id" + std::to_string(id) + "_rank" +
                                     std::to_string(rank) + "_round" + std::to_string(round);
            auto save = [&](const char* suffix, const void* data, size_t bytes) -> test_status_t {
                std::ofstream file(base + suffix, std::ios::binary);
                file.write(static_cast<const char*>(data), bytes);
                if (!file)
                    return {ACLSHMEM_INNER_ERROR, "raw evidence write failed"};
                return {};
            };
            status = save(".payload.bin", actual.data(), BYTES);
            if (!status.ok()) {
                return status;
            }
            status = save(".identity.bin", ids, sizeof(ids));
            if (!status.ok()) {
                return status;
            }
            status = save(".cas.bin", cas.data(), OBS_BYTES);
            if (!status.ok()) {
                return status;
            }
            status = save(".timing.bin", ticks, sizeof(ticks));
            if (!status.ok()) {
                return status;
            }
        }
        std::printf(
            "SIMT_INSTANCE_SELECTION_PASS id=%lu rank=%d mode=%d round=%d count=%u start=%lu end=%lu\n", id, rank, mode,
            round, count, ticks[0], ticks[1]);
        std::fflush(stdout);

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
        if (!status.check(aclrtFree(observed), "free CAS results")) {
            return status;
        }
        if (!status.check(aclrtFree(timing), "free timing")) {
            return status;
        }

        return {};
    }
};

// Only submission and stream completion run concurrently. Initialization,
// allocation, rank rendezvous and destruction remain on the controlling thread.
// Each worker owns one stream and one instance's disjoint heap/output buffers;
// the existing VF owns its private UB and receives the immutable ID by value.
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

test_status_t launch_from_host_threads(
    instance_t& a, instance_t& b, uint32_t count_a, uint32_t count_b, int mode, uint32_t repetitions)
{
    aclrtContext context = nullptr;
    test_status_t status;
    if (!status.check(aclrtGetCurrentContext(&context), "capture runtime context")) {
        return status;
    }
    if (context == nullptr) {
        return {ACLSHMEM_INNER_ERROR, "missing runtime context"};
    }
    instance_t* instances[] = {&a, &b};
    const uint32_t counts[] = {count_a, count_b};
    std::thread::id threads[2];
    status = run_worker_pair(
        [&](int index) -> test_status_t {
            threads[index] = std::this_thread::get_id();
            test_status_t worker_status;
            if (!worker_status.check(aclrtSetCurrentContext(context), "worker runtime context")) {
                return worker_status;
            }
            // Each VF selects its immutable ID; no Host instance selection here.
            return instances[index]->launch(counts[index], mode, repetitions);
        },
        [&](int index) -> test_status_t { return instances[index]->wait(); });
    if (!status.ok()) {
        return status;
    }
    if (threads[0] == threads[1] || threads[0] == std::this_thread::get_id() ||
        threads[1] == std::this_thread::get_id()) {
        return {ACLSHMEM_INNER_ERROR, "workers are not distinct Host threads"};
    }
    return {};
}

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

test_status_t run_selection_impl(int physical_rank, int ranks, int scenario)
{
    test_status_t status;
    if (!status.check(aclInit(nullptr), "aclInit")) {
        return status;
    }
    if (!status.check(aclrtSetDevice(physical_rank % test_gnpu_num + test_first_npu), "device")) {
        return status;
    }
    if (!status.check(aclshmemx_set_conf_store_tls(false, nullptr, 0), "TLS configuration")) {
        return status;
    }
    instance_t a{scenario == 6 ? 255ULL : (scenario == 0 ? 0ULL : 1ULL), physical_rank, ranks};
    instance_t b{scenario == 0 || scenario == 6 ? 1ULL : 2ULL, ranks - 1 - physical_rank, ranks};
    status = a.create(64ULL << 20);
    if (!status.ok()) {
        return status;
    }
    status = b.create(128ULL << 20);
    if (!status.ok()) {
        return status;
    }
    if (a.heap == b.heap || a.stream == b.stream)
        return {ACLSHMEM_INNER_ERROR, "live resources alias"};
    const uint32_t sizes[] = {0, 1, 129, SLOT};
    const bool host_threads = scenario == 4 || scenario == 5;
    const bool all_modes = scenario == 1 || scenario == 4 || scenario == 6;
    const bool require_overlap = scenario == 2 || scenario == 5;
    const int rounds = all_modes ? LEGACY * LENGTH_COUNT : 20;
    int overlaps = 0;
    for (int round = 0; round < rounds; ++round) {
        const int mode_a = scenario == 0 ? LEGACY : (all_modes ? round / LENGTH_COUNT : BLOCK_NBI);
        const int mode_b = scenario == 0 ? BLOCK_NBI : mode_a;
        const uint32_t count_a = all_modes ? sizes[round % LENGTH_COUNT] : (round % 2 ? 129 : SLOT);
        const uint32_t count_b = all_modes ? sizes[(round + 1) % LENGTH_COUNT] : (round % 2 ? SLOT : 129);
        const uint32_t repetitions = require_overlap ? 2048 : 1;
        status = a.prepare(round);
        if (!status.ok()) {
            return status;
        }
        status = b.prepare(round);
        if (!status.ok()) {
            return status;
        }
        status = rendezvous(b.id);
        if (!status.ok()) {
            return status;
        }
        const auto selected_id = round % 2 ? a.id : b.id;
        if (!status.check(aclshmemx_instance_ctx_set(selected_id), "perturb host selector")) {
            return status;
        }
        if (host_threads) {
            status = launch_from_host_threads(a, b, count_a, count_b, mode_a, repetitions);
            if (!status.ok()) {
                return status;
            }
            if (aclshmemx_instance_ctx_get()->id != selected_id)
                return {ACLSHMEM_INNER_ERROR, "workers changed Host instance selector"};
        } else if (round % 2) {
            status = b.launch(count_b, mode_b, repetitions);
            if (!status.ok()) {
                return status;
            }
            status = a.launch(count_a, mode_a, repetitions);
            if (!status.ok()) {
                return status;
            }
        } else {
            status = a.launch(count_a, mode_a, repetitions);
            if (!status.ok()) {
                return status;
            }
            status = b.launch(count_b, mode_b, repetitions);
            if (!status.ok()) {
                return status;
            }
        }
        status = a.wait();
        if (!status.ok()) {
            return status;
        }
        status = b.wait();
        if (!status.ok()) {
            return status;
        }
        status = rendezvous(b.id);
        if (!status.ok()) {
            return status;
        }
        status = a.verify(scenario, round, count_a, mode_a);
        if (!status.ok()) {
            return status;
        }
        status = b.verify(scenario, round, count_b, mode_b);
        if (!status.ok()) {
            return status;
        }
        const auto start = std::max(a.ticks[0], b.ticks[0]), end = std::min(a.ticks[1], b.ticks[1]);
        overlaps += end > start;
        if (require_overlap) {
            std::printf(
                "SIMT_INSTANCE_SELECTION_OVERLAP rank=%d round=%d ticks=%lu\n", physical_rank, round,
                end > start ? end - start : 0);
            std::fflush(stdout);
        }
    }
    std::vector<int32_t> retained;
    status = b.read(retained);
    if (!status.ok()) {
        return status;
    }
    status = a.destroy();
    if (!status.ok()) {
        return status;
    }
    std::vector<int32_t> current;
    status = b.read(current);
    if (!status.ok()) {
        return status;
    }
    if (current != retained)
        return {ACLSHMEM_INNER_ERROR, "surviving output changed during other finalize"};
    status = b.prepare(rounds);
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(b.id);
    if (!status.ok()) {
        return status;
    }
    status = b.launch(129, BLOCK_NBI, 1);
    if (!status.ok()) {
        return status;
    }
    status = b.wait();
    if (!status.ok()) {
        return status;
    }
    status = rendezvous(b.id);
    if (!status.ok()) {
        return status;
    }
    status = b.verify(scenario, rounds, 129, BLOCK_NBI);
    if (!status.ok()) {
        return status;
    }
    status = b.destroy();
    if (!status.ok()) {
        return status;
    }
    if (!status.check(aclFinalize(), "aclFinalize")) {
        return status;
    }

    if (require_overlap && overlaps == 0)
        return {ACLSHMEM_INNER_ERROR, "no actual dual-stream overlap observed"};

    return {};
}

test_status_t run_warp_tails_impl(int rank, int ranks)
{
    test_status_t status;
    if (!status.check(aclInit(nullptr), "aclInit")) {
        return status;
    }
    if (!status.check(aclrtSetDevice(rank % test_gnpu_num + test_first_npu), "device")) {
        return status;
    }
    if (!status.check(aclshmemx_set_conf_store_tls(false, nullptr, 0), "TLS")) {
        return status;
    }
    instance_t a{0, rank, ranks}, b{1, rank, ranks};
    status = a.create(64ULL << 20);
    if (!status.ok()) {
        return status;
    }
    status = b.create(96ULL << 20);
    if (!status.ok()) {
        return status;
    }
    int round = 0;
    for (uint32_t lanes : {16U, 32U, 33U, 48U, 64U, 80U}) {
        for (uint32_t count : {0U, 1U, 32U, 65U, 129U}) {
            for (int mode :
                 {GM_PUT + WARP, GM_GET + WARP, UB_PUT + WARP, UB_GET + WARP, GM_PUT + WARP_NBI, GM_GET + WARP_NBI,
                  UB_PUT + WARP_NBI, UB_GET + WARP_NBI, int(BYTE_GM_PUT), int(BYTE_GM_GET), int(BYTE_UB_PUT),
                  int(BYTE_UB_GET), int(LEGACY)}) {
                for (instance_t* instance : {&a, &b}) {
                    if (mode == LEGACY && instance->id != 0)
                        continue;
                    auto& v = *instance;
                    status = v.prepare(round);
                    if (!status.ok()) {
                        return status;
                    }
                    status = rendezvous(v.id);
                    if (!status.ok()) {
                        return status;
                    }
                    simt_warp_tail_transfer(v.stream, v.heap, count, (rank + 1) % ranks, v.id, lanes, mode);
                    status = v.wait();
                    if (!status.ok()) {
                        return status;
                    }
                    status = rendezvous(v.id);
                    if (!status.ok()) {
                        return status;
                    }
                    auto expected = reference(v.id, rank, ranks, round, 0, GM_PUT);
                    const int source = is_get(mode) ? (rank + 1) % ranks : (rank + ranks - 1) % ranks;
                    for (uint32_t warp = 0; warp < (lanes + 31) / 32; ++warp)
                        for (uint32_t i = 0; i < count; ++i)
                            expected[SLOT + warp * 256 + i] = value(v.id, source, round, warp * 256 + i);
                    std::vector<int32_t> actual;
                    status = v.read(actual);
                    if (!status.ok()) {
                        return status;
                    }
                    if (const char* directory = std::getenv("SHMEM_SIMT_INSTANCE_SELECTION_EVIDENCE_DIR")) {
                        const std::string path = std::string(directory) + "/warp_id" + std::to_string(v.id) + "_rank" +
                                                 std::to_string(rank) + "_round" + std::to_string(round) +
                                                 ".payload.bin";
                        std::ofstream file(path, std::ios::binary);
                        file.write(reinterpret_cast<const char*>(actual.data()), BYTES);
                        if (!file)
                            return {ACLSHMEM_INNER_ERROR, "warp evidence write failed"};
                    }
                    if (actual != expected)
                        return {
                            ACLSHMEM_INNER_ERROR, "warp tail data/guard mismatch lanes=" + std::to_string(lanes) +
                                                      " count=" + std::to_string(count) +
                                                      " mode=" + std::to_string(mode)};
                    std::printf(
                        "WARP_TAIL_PASS rank=%d id=%lu lanes=%u count=%u mode=%d round=%d\n", rank, v.id, lanes, count,
                        mode, round);
                    std::fflush(stdout);
                }
                ++round;
            }
        }
    }
    status = a.destroy();
    if (!status.ok()) {
        return status;
    }
    status = b.destroy();
    if (!status.ok()) {
        return status;
    }
    if (!status.check(aclFinalize(), "aclFinalize")) {
        return status;
    }

    return {};
}

void run_selection(int physical_rank, int ranks, int scenario)
{
    run_rank(physical_rank, [&] { return run_selection_impl(physical_rank, ranks, scenario); });
}

void run_warp_tails(int rank, int ranks)
{
    run_rank(rank, [&] { return run_warp_tails_impl(rank, ranks); });
}

#endif

void run_case(int scenario)
{
#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
    if (test_first_rank != 0 || test_global_ranks != test_gnpu_num ||
        (test_gnpu_num != 2 && test_gnpu_num != 4 && test_gnpu_num != 8))
        GTEST_SKIP() << "Requires a single host with 2, 4 or 8 ranks";
    // Honor an explicit, launcher-checked range; otherwise use the UT default.
    const char* configured_ports = std::getenv("SHMEM_INSTANCE_PORT_RANGE");
    const std::string ports = configured_ports != nullptr ? configured_ports : get_multi_instance_port_range();
    ScopedEnv port_range("SHMEM_INSTANCE_PORT_RANGE", ports.c_str());
    std::printf("INSTANCE_TEST_PORT_RANGE=%s\n", ports.c_str());
    std::fflush(stdout);
    test_mutil_task(
        [scenario](int rank, int ranks, uint64_t) {
            if (scenario == 3)
                run_warp_tails(rank, ranks);
            else
                run_selection(rank, ranks, scenario);
        },
        64ULL << 20, test_gnpu_num);
#else
    GTEST_SKIP() << "Requires Ascend950 and ACLSHMEM_SIMT_SUPPORT";
#endif
}
} // namespace

TEST(TestSimtInstanceSelection, ReferenceDistinguishesInstanceAndRank)
{
    const auto expected = reference(1, 0, 2, 3, 129, 0);
    EXPECT_NE(expected, reference(2, 0, 2, 3, 129, 0));
    EXPECT_NE(expected, reference(1, 1, 2, 3, 129, 0));
}
TEST(TestSimtInstanceSelection, LegacyDefaultInstance) { run_case(0); }
TEST(TestSimtInstanceSelection, GmUbScalarAtomicAndTeam) { run_case(1); }
TEST(TestSimtInstanceSelection, DualStreamIsolation) { run_case(2); }
TEST(TestSimtInstanceSelection, DualHostThreadGmUbScalarAtomicAndTeam) { run_case(4); }
TEST(TestSimtInstanceSelection, DualHostThreadDualStreamIsolation) { run_case(5); }

TEST(TestSimtInstanceSelection, WarpTailBlockParticipation) { run_case(3); }

TEST(TestSimtInstanceSelection, HbmOnlyBoundaryInstance)
{
#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT) && !defined(USE_ACLRT_MEM_FABRIC_HANDLE)
    run_case(6);
#else
    GTEST_SKIP() << "Requires Ascend950 SIMT with HBM-only ordinary heaps";
#endif
}

TEST(TestSimtInstanceSelection, InstanceIdBoundary)
{
#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
    test_mutil_task(
        [](int rank, int, uint64_t) {
            ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
            ASSERT_EQ(aclrtSetDevice(rank % test_gnpu_num + test_first_npu), ACL_SUCCESS);
            aclrtStream stream = nullptr;
            void* result = nullptr;
            ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&result, 5 * sizeof(uint64_t), ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
            simt_instance_boundary(stream, result);
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
            uint64_t actual[5] = {};
            ASSERT_EQ(
                aclrtMemcpy(actual, sizeof(actual), result, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
            const uint64_t expected[] = {1, 1, 1, 0, 0};
            for (uint32_t i = 0; i < 5; ++i) {
                EXPECT_EQ(actual[i], expected[i]);
            }
            EXPECT_EQ(aclrtFree(result), ACL_SUCCESS);
            EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
            EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
        },
        64ULL << 20, test_gnpu_num);
#else
    GTEST_SKIP() << "Requires Ascend950 SIMT";
#endif
}
