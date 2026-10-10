/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "shmem.h"
#include "barrier_all_kernel.h"

namespace {
using namespace barrier_all_demo;

constexpr uint64_t kHeapBytes = 64ULL * 1024 * 1024;

void check(int status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

uint32_t expected_value(int stage, int pe, uint32_t index)
{
    return (stage + 1) * 1000000U + (pe + 1) * 10000U + index;
}

void prepare_data(uint32_t* buffer, int pe, int n_pes, int stage)
{
    const size_t total_bytes = static_cast<size_t>(n_pes) * kSlotBytes;
    std::vector<uint32_t> data(n_pes * kElementCount, 0);
    for (uint32_t i = 0; i < kElementCount; ++i) {
        data[pe * kElementCount + i] = expected_value(stage, pe, i);
    }
    check(aclrtMemcpy(buffer, total_bytes, data.data(), total_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "initialize data");
    // Every destination must be initialized before any PE starts sending.
    aclshmem_barrier_all();
}

bool report(const char* stage, int pe, uint32_t errors)
{
    std::cout << "[" << (errors == 0 ? "PASS" : "FAIL") << "] PE " << pe << " " << stage << ": " << errors
              << " mismatches" << std::endl;
    return errors == 0;
}

uint32_t verify_segments(const std::vector<uint32_t>& received, int pe, int n_pes, int stage)
{
    uint32_t errors = 0;
    for (int peer = 0; peer < n_pes; ++peer) {
        if (peer == pe) {
            continue;
        }
        for (uint32_t i = 0; i < kElementCount; ++i) {
            errors += received[peer * kElementCount + i] != expected_value(stage, peer, i);
        }
    }
    return errors;
}

bool run_device(uint32_t* buffer, aclrtStream stream, int pe, int n_pes)
{
    prepare_data(buffer, pe, n_pes, 0);
    launch_barrier_all(stream, util_get_ffts_config(), buffer);
    check(aclrtSynchronizeStream(stream), "Device example");

    // The device barrier_all() has completed the NBI puts; verify the receive segments on host.
    const size_t total_bytes = static_cast<size_t>(n_pes) * kSlotBytes;
    std::vector<uint32_t> received(n_pes * kElementCount);
    check(
        aclrtMemcpy(received.data(), total_bytes, buffer, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        "read device results");
    return report("Device", pe, verify_segments(received, pe, n_pes, 0));
}

bool run_host(uint32_t* buffer, int pe, int n_pes)
{
    // Wait for every PE to finish Device verification before reusing the same buffers.
    aclshmem_barrier_all();
    prepare_data(buffer, pe, n_pes, 1);

    // The high-level API selects MTE for same-machine PEs and RDMA for cross-machine PEs.
    for (int peer = 0; peer < n_pes; ++peer) {
        if (peer == pe) {
            continue;
        }
        aclshmem_putmem_nbi(buffer + pe * kElementCount, buffer + pe * kElementCount, kSlotBytes, peer);
    }
    aclshmem_barrier_all();

    const size_t total_bytes = static_cast<size_t>(n_pes) * kSlotBytes;
    std::vector<uint32_t> received(n_pes * kElementCount);
    check(
        aclrtMemcpy(received.data(), total_bytes, buffer, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read host results");
    return report("Host", pe, verify_segments(received, pe, n_pes, 1));
}

bool parse_nonnegative_int(const char* text, int& value)
{
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}
} // namespace

int main(int argc, char* argv[])
{
    int pe = 0;
    int device_id = 0;
    int n_pes = 0;
    if ((argc != 5 && argc != 6) || !parse_nonnegative_int(argv[1], pe) || !parse_nonnegative_int(argv[2], device_id) ||
        !parse_nonnegative_int(argv[3], n_pes) || n_pes < 2 || pe >= n_pes || std::strlen(argv[4]) == 0 ||
        std::strlen(argv[4]) >= ACLSHMEM_MAX_IP_PORT_LEN || (argc == 6 && std::strcmp(argv[5], "--roce") != 0)) {
        std::cerr << "Usage: " << argv[0] << " <pe_id> <device_id> <n_pes:>=2> <tcp://rank0_ip:port> [--roce]"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const bool use_roce = argc == 6;
#ifndef BARRIER_ALL_RDMA_SUPPORT
    if (use_roce) {
        std::cerr << "[ERROR] --roce requires an RDMA-enabled build (-enable_rdma)." << std::endl;
        return EXIT_FAILURE;
    }
#endif

    bool acl_initialized = false;
    bool device_selected = false;
    bool shmem_initialized = false;
    aclrtStream stream = nullptr;
    uint32_t* buffer = nullptr;
    int result = EXIT_FAILURE;
    try {
        check(aclInit(nullptr), "aclInit");
        acl_initialized = true;
        check(aclrtSetDevice(device_id), "aclrtSetDevice");
        device_selected = true;
        check(aclrtCreateStream(&stream), "aclrtCreateStream");

        aclshmemx_init_attr_t attributes{};
        attributes.my_pe = pe;
        attributes.n_pes = n_pes;
        attributes.local_mem_size = kHeapBytes;
        std::memcpy(attributes.ip_port, argv[4], std::strlen(argv[4]) + 1);
        attributes.option_attr.data_op_engine_type =
            use_roce ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE) :
                       ACLSHMEM_DATA_OP_MTE;
        check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), "aclshmemx_init_attr");
        shmem_initialized = true;

        buffer = static_cast<uint32_t*>(aclshmem_malloc(static_cast<size_t>(n_pes) * kSlotBytes));
        if (buffer == nullptr) {
            throw std::runtime_error("aclshmem_malloc failed");
        }

        const bool device_ok = run_device(buffer, stream, pe, n_pes);
        const bool host_ok = run_host(buffer, pe, n_pes);
        aclshmem_barrier_all();
        result = device_ok && host_ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] PE " << pe << ": " << error.what() << std::endl;
        if (shmem_initialized) {
            aclshmem_global_exit(EXIT_FAILURE);
            return EXIT_FAILURE;
        }
    }

    auto cleanup = [&result](int status, const char* operation) {
        if (status != 0) {
            std::cerr << "[ERROR] " << operation << " failed: " << status << std::endl;
            result = EXIT_FAILURE;
        }
    };
    if (buffer != nullptr) {
        aclshmem_free(buffer);
    }
    if (shmem_initialized) {
        cleanup(aclshmem_finalize(), "aclshmem_finalize");
    }
    if (stream != nullptr) {
        cleanup(aclrtDestroyStream(stream), "aclrtDestroyStream");
    }
    if (device_selected) {
        cleanup(aclrtResetDevice(device_id), "aclrtResetDevice");
    }
    if (acl_initialized) {
        cleanup(aclFinalize(), "aclFinalize");
    }
    return result;
}
