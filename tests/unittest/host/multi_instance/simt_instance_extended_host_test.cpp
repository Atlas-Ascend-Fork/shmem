/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "shmem.h"
#include "init/shmemi_init.h"
#include "unittest_main_test.h"
#include "unittest/utils/scope_env.h"
#include "unittest/simt_instance_extended_test.h"

#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
extern "C" void simt_instance_extended_transfer(
    void*, void*, void*, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, int, int, bool, uint32_t, void*);
#endif

namespace {
using namespace simt_instance_extended_test;
enum class scenario_t { SAME_VF, MULTI_AIV, BYTE_TAIL, MULTI_DTYPE };

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

// CPU-only oracle: finite floating-point bit patterns and full-width integer
// patterns encode instance, logical rank, round and source element. No device
// indexing helper is used to construct the expected receive buffer.
std::vector<uint8_t> extended_input(uint64_t id, int rank, int round, int type)
{
    std::vector<uint8_t> data(HEAP_BYTES, GUARD);
    const uint32_t width = TYPE_BYTES[type];
    for (uint32_t i = 0; i < REGION_BYTES / width; ++i) {
        uint64_t bits = (id + 1) * 0x9e3779b97f4a7c15ULL;
        bits ^= (uint64_t(rank) + 1) * 0xbf58476d1ce4e5b9ULL;
        bits ^= (uint64_t(round) + 1) * 0x94d049bb133111ebULL;
        bits ^= uint64_t(i + 1) * 0xd6e8feb86659fd93ULL;
        bits ^= bits >> 29;
        if (type == FP16)
            bits = 0x3800 | (bits & 0x3ff);
        else if (type == FP32)
            bits = 0x3f000000 | (bits & 0x7fffff);
        else if (type == BF16)
            bits = 0x3f00 | (bits & 0x7f);
        for (uint32_t j = 0; j < width; ++j)
            data[i * width + j] = static_cast<uint8_t>(bits >> (8 * j));
    }
    return data;
}

struct extended_instance_t {
    uint64_t id;
    int rank, ranks;
    uint8_t* heap = nullptr;
    aclrtStream stream = nullptr;

    test_status_t create(uint64_t capacity)
    {
        aclshmemx_init_attr_t attr{};
        attr.my_pe = rank;
        attr.n_pes = ranks;
        attr.instance_id = id;
        attr.local_mem_size = capacity;
        std::strcpy(attr.ip_port, "tcp://127.0.0.1:0");
        attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
        attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
        test_status_t status;
        if (!status.check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "extended init")) {
            return status;
        }
        if (!status.check(aclshmemx_instance_ctx_set(id), "extended allocation context")) {
            return status;
        }
        heap = static_cast<uint8_t*>(aclshmem_malloc(HEAP_BYTES));
        if (heap == nullptr)
            return {ACLSHMEM_INNER_ERROR, "extended symmetric allocation failed"};
        if (!status.check(aclrtCreateStream(&stream), "extended stream")) {
            return status;
        }

        return {};
    }
    test_status_t rendezvous()
    {
        test_status_t status;
        if (!status.check(aclshmemx_instance_ctx_set(id), "extended barrier context")) {
            return status;
        }
        if (!status.check(aclshmemi_control_barrier_all(), "extended control barrier")) {
            return status;
        }

        return {};
    }
    test_status_t prepare(int round, int type)
    {
        const auto data = extended_input(id, rank, round, type);
        test_status_t status;
        if (!status.check(
                aclrtMemcpy(heap, HEAP_BYTES, data.data(), HEAP_BYTES, ACL_MEMCPY_HOST_TO_DEVICE), "extended H2D")) {
            return status;
        }

        return {};
    }
    test_status_t verify(int round, int type, int mode, uint32_t count, uint32_t shift, uint32_t blocks)
    {
        auto expected = extended_input(id, rank, round, type);
        const int source_rank = is_get(mode) ? (rank + 1) % ranks : (rank + ranks - 1) % ranks;
        const auto source = extended_input(id, source_rank, round, type);
        const uint32_t bytes = count * TYPE_BYTES[type];
        for (uint32_t block = 0; block < blocks; ++block)
            for (uint32_t phase = 0; phase < 2; ++phase) {
                const uint32_t offset = block * BLOCK_BYTES + phase * PHASE_BYTES + shift;
                std::memcpy(expected.data() + REGION_BYTES + offset, source.data() + offset, bytes);
            }
        std::vector<uint8_t> actual(HEAP_BYTES);
        test_status_t status;
        if (!status.check(
                aclrtMemcpy(actual.data(), HEAP_BYTES, heap, HEAP_BYTES, ACL_MEMCPY_DEVICE_TO_HOST), "extended D2H")) {
            return status;
        }
        for (uint32_t i = 0; i < HEAP_BYTES; ++i)
            if (actual[i] != expected[i])
                return {
                    ACLSHMEM_INNER_ERROR, "extended mismatch id=" + std::to_string(id) +
                                              " rank=" + std::to_string(rank) + " round=" + std::to_string(round) +
                                              " type=" + std::to_string(type) + " mode=" + std::to_string(mode) +
                                              " count=" + std::to_string(count) + " shift=" + std::to_string(shift) +
                                              " byte=" + std::to_string(i) + " actual=" + std::to_string(actual[i]) +
                                              " expected=" + std::to_string(expected[i])};

        return {};
    }
    test_status_t destroy()
    {
        test_status_t status = rendezvous();
        if (!status.ok()) {
            return status;
        }
        aclshmem_free(heap);
        if (!status.check(aclshmemx_finalize(id), "extended finalize")) {
            return status;
        }
        if (!status.check(aclrtDestroyStream(stream), "extended destroy stream")) {
            return status;
        }

        return {};
    }
};

void run_rank(int rank, const std::function<test_status_t()>& task)
{
    // Do not enter collective cleanup after a failed stage. The child runner
    // reports failure and bounds another rank blocked in a collective by alarm.
    std::signal(SIGALRM, SIG_DFL);
    alarm(600);
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

test_status_t run_extended_rank_impl(int rank, int ranks, scenario_t scenario)
{
    test_status_t status;
    if (!status.check(aclInit(nullptr), "extended aclInit")) {
        return status;
    }
    if (!status.check(aclrtSetDevice(rank % test_gnpu_num + test_first_npu), "extended device")) {
        return status;
    }
    if (!status.check(aclshmemx_set_conf_store_tls(false, nullptr, 0), "extended TLS")) {
        return status;
    }
    extended_instance_t a{1, rank, ranks}, b{2, ranks - 1 - rank, ranks};
    status = a.create(64ULL << 20);
    if (!status.ok()) {
        return status;
    }
    status = b.create(96ULL << 20);
    if (!status.ok()) {
        return status;
    }
    uint64_t* identity = nullptr;
    constexpr size_t identity_bytes = MAX_BLOCKS * ID_FIELDS * sizeof(uint64_t);
    if (!status.check(
            aclrtMalloc(reinterpret_cast<void**>(&identity), identity_bytes, ACL_MEM_MALLOC_NORMAL_ONLY),
            "extended identity allocation")) {
        return status;
    }
    const uint32_t blocks = scenario == scenario_t::MULTI_AIV ? MAX_BLOCKS : 1;
    const bool byte = scenario == scenario_t::BYTE_TAIL;
    const std::vector<uint32_t> counts =
        byte ? std::vector<uint32_t>{0, 1, 3, 31, 32, 33, 127, 129} : std::vector<uint32_t>{0, 1, 33, 129};
    const std::vector<uint32_t> shifts = byte ? std::vector<uint32_t>{1, 3} : std::vector<uint32_t>{8};
    const int first_type = byte ? INT8 : (scenario == scenario_t::MULTI_DTYPE ? INT8 : INT32);
    const int end_type = scenario == scenario_t::MULTI_DTYPE ? TYPE_COUNT : first_type + 1;
    int round = 0;
    for (int type = first_type; type < end_type; ++type)
        for (int mode = 0; mode < MODE_COUNT; ++mode)
            for (uint32_t shift : shifts)
                for (size_t length = 0; length < counts.size(); ++length) {
                    const uint32_t count_a = counts[length], count_b = counts[(length + 1) % counts.size()];
                    status = a.prepare(round, type);
                    if (!status.ok()) {
                        return status;
                    }
                    status = b.prepare(round, type);
                    if (!status.ok()) {
                        return status;
                    }
                    if (!status.check(
                            aclrtMemset(identity, identity_bytes, 0xff, identity_bytes), "extended identity reset")) {
                        return status;
                    }
                    status = b.rendezvous();
                    if (!status.ok()) {
                        return status;
                    }
                    const uint64_t selected = round % 2 ? a.id : b.id;
                    if (!status.check(aclshmemx_instance_ctx_set(selected), "extended perturb Host selector")) {
                        return status;
                    }
                    simt_instance_extended_transfer(
                        a.stream, a.heap, b.heap, a.id, b.id, count_a, count_b, shift, mode, type, byte, blocks,
                        identity);
                    if (!status.check(aclrtSynchronizeStream(a.stream), "extended completion")) {
                        return status;
                    }
                    if (aclshmemx_instance_ctx_get()->id != selected)
                        return {ACLSHMEM_INNER_ERROR, "extended launch changed Host selector"};
                    status = b.rendezvous();
                    if (!status.ok()) {
                        return status;
                    }
                    status = a.verify(round, type, mode, count_a, shift, blocks);
                    if (!status.ok()) {
                        return status;
                    }
                    status = b.verify(round, type, mode, count_b, shift, blocks);
                    if (!status.ok()) {
                        return status;
                    }
                    uint64_t ids[MAX_BLOCKS * ID_FIELDS];
                    if (!status.check(
                            aclrtMemcpy(ids, identity_bytes, identity, identity_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                            "extended read identity")) {
                        return status;
                    }
                    for (uint32_t block = 0; block < blocks; ++block)
                        if (ids[block * ID_FIELDS] == 0 || ids[block * ID_FIELDS + 1] == 0 ||
                            ids[block * ID_FIELDS] == ids[block * ID_FIELDS + 1] ||
                            ids[block * ID_FIELDS + 2] != uint64_t(a.rank) ||
                            ids[block * ID_FIELDS + 3] != uint64_t(b.rank))
                            return {
                                ACLSHMEM_INNER_ERROR,
                                "extended per-AIV identity mismatch block=" + std::to_string(block)};
                    std::printf(
                        "SIMT_EXTENDED_PASS scenario=%d rank=%d round=%d type=%d mode=%d counts=%u,%u shift=%u "
                        "blocks=%u\n",
                        static_cast<int>(scenario), rank, round, type, mode, count_a, count_b, shift, blocks);
                    std::fflush(stdout);
                    ++round;
                }
    status = a.destroy();
    if (!status.ok()) {
        return status;
    }
    status = b.destroy();
    if (!status.ok()) {
        return status;
    }
    if (!status.check(aclrtFree(identity), "extended free identity")) {
        return status;
    }
    if (!status.check(aclFinalize(), "extended aclFinalize")) {
        return status;
    }

    return {};
}

void run_extended_rank(int rank, int ranks, scenario_t scenario)
{
    run_rank(rank, [&] { return run_extended_rank_impl(rank, ranks, scenario); });
}

#endif

void run_extended(scenario_t scenario)
{
#if defined(ACLSHMEM_UT_ASCEND950) && defined(USE_SIMT)
    if (test_first_rank != 0 || test_global_ranks != test_gnpu_num ||
        (test_gnpu_num != 2 && test_gnpu_num != 4 && test_gnpu_num != 8))
        GTEST_SKIP() << "Requires Ascend950 SIMT on one host with 2, 4 or 8 ranks";
    const char* configured_ports = std::getenv("SHMEM_INSTANCE_PORT_RANGE");
    const std::string ports = configured_ports ? configured_ports : get_multi_instance_port_range();
    ScopedEnv port_range("SHMEM_INSTANCE_PORT_RANGE", ports.c_str());
    test_mutil_task(
        [scenario](int rank, int ranks, uint64_t) { run_extended_rank(rank, ranks, scenario); }, 64ULL << 20,
        test_gnpu_num);
#else
    GTEST_SKIP() << "Requires Ascend950 and ACLSHMEM_SIMT_SUPPORT";
#endif
}
} // namespace

TEST(TestSimtInstanceExtended, SameVfInstanceSwitch) { run_extended(scenario_t::SAME_VF); }
TEST(TestSimtInstanceExtended, MultiAivInstanceSwitch) { run_extended(scenario_t::MULTI_AIV); }
TEST(TestSimtInstanceExtended, ByteLengthsAndOffsets) { run_extended(scenario_t::BYTE_TAIL); }
TEST(TestSimtInstanceExtended, RuntimeDtypes) { run_extended(scenario_t::MULTI_DTYPE); }
