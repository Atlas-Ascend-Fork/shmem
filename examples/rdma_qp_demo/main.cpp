/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "rdma_qp_demo_kernel.h"
#include "shmem.h"
#include "utils.h"

namespace {
constexpr uint32_t AGGREGATE_OP_COUNT = 8;
constexpr uint32_t SYNC_ID = 0;
constexpr uint64_t ELEMENTS_PER_QP = 512ULL * 1024ULL;
constexpr uint64_t HEAP_BYTES = 1024ULL * 1024ULL * 1024ULL;

enum class DemoOperation : int32_t {
    PUT = 0,
    GET = 1,
    AGGREGATE_PUT = 2,
    AGGREGATE_GET = 3,
};

struct Options {
    int pe{0};
    int pes{2};
    int gnpus{2};
    int fpe{0};
    int fnpu{0};
    uint32_t qp{2};
    std::string ipport{"tcp://127.0.0.1:8899"};
    std::string op{"all"};
};

struct RuntimeResources {
    int device_id{0};
    aclrtStream stream{nullptr};
    uint8_t* symmetric{nullptr};
    bool acl_initialized{false};
    bool device_set{false};
    bool stream_created{false};
    bool shmem_initialized{false};
};

void PrintUsage(const char* program)
{
    std::cerr << "Usage: " << program
              << " -pe ID -pes N -gnpus N -fpe ID -fnpu ID [-qp N]"
                 " [-op put|get"
                 "|aggregate_put|aggregate_get"
                 "|all] [-ipport tcp://IP:PORT]\n";
}

bool ParseUint64(const char* text, uint64_t& value)
{
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    value = std::strtoull(text, &end, 10);
    return errno != ERANGE && end != nullptr && *end == '\0';
}

bool ParseOptions(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            return false;
        }
        const std::string key = argv[i];
        if (key == "-op" || key == "-ipport") {
            (key == "-op" ? options.op : options.ipport) = argv[i + 1];
            continue;
        }
        uint64_t value = 0;
        if (!ParseUint64(argv[i + 1], value)) {
            return false;
        }
        if (key == "-pe") {
            if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                return false;
            options.pe = static_cast<int>(value);
        } else if (key == "-pes") {
            if (value > ACLSHMEM_MAX_PES || value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                return false;
            }
            options.pes = static_cast<int>(value);
        } else if (key == "-gnpus") {
            if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                return false;
            options.gnpus = static_cast<int>(value);
        } else if (key == "-fpe") {
            if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                return false;
            options.fpe = static_cast<int>(value);
        } else if (key == "-fnpu") {
            if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                return false;
            options.fnpu = static_cast<int>(value);
        } else if (key == "-qp") {
            if (value > ACLSHMEM_MAX_QP_NUM)
                return false;
            options.qp = static_cast<uint32_t>(value);
        } else {
            return false;
        }
    }
    return true;
}

uint8_t Pattern(int pe, uint64_t index)
{
    return static_cast<uint8_t>((static_cast<uint64_t>(pe + 1) * 37 + index % 251) & 0xFF);
}

std::vector<uint8_t> MakePattern(int pe, uint64_t elements)
{
    std::vector<uint8_t> data(elements);
    for (uint64_t i = 0; i < elements; ++i) {
        data[i] = Pattern(pe, i);
    }
    return data;
}

bool Validate(const std::vector<uint8_t>& actual, int expected_pe, const std::string& operation, int pe)
{
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const uint8_t expected = Pattern(expected_pe, i);
        if (actual[i] != expected) {
            std::cerr << "[FAIL] op=" << operation << " pe=" << pe << " index=" << i
                      << " actual=" << static_cast<uint32_t>(actual[i])
                      << " expected=" << static_cast<uint32_t>(expected) << std::endl;
            return false;
        }
    }
    std::cout << "[PASS] op=" << operation << " pe=" << pe << " elements=" << actual.size() << std::endl;
    return true;
}

void RecordCleanupError(int cleanup_status, const char* operation, int& status)
{
    if (cleanup_status != 0) {
        std::cerr << operation << " failed during cleanup, status=" << cleanup_status << std::endl;
        if (status == 0) {
            status = cleanup_status;
        }
    }
}

int Cleanup(RuntimeResources& resources, int status)
{
    if (resources.symmetric != nullptr && resources.shmem_initialized) {
        aclshmem_free(resources.symmetric);
        resources.symmetric = nullptr;
    }
    if (resources.shmem_initialized) {
        RecordCleanupError(aclshmem_finalize(), "aclshmem_finalize", status);
        resources.shmem_initialized = false;
    }
    if (resources.stream_created) {
        RecordCleanupError(aclrtDestroyStream(resources.stream), "aclrtDestroyStream", status);
        resources.stream_created = false;
    }
    if (resources.device_set) {
        RecordCleanupError(aclrtResetDevice(resources.device_id), "aclrtResetDevice", status);
        resources.device_set = false;
    }
    if (resources.acl_initialized) {
        RecordCleanupError(aclFinalize(), "aclFinalize", status);
        resources.acl_initialized = false;
    }
    return status;
}

bool ParseOperation(const std::string& operation, DemoOperation& parsed_operation)
{
    if (operation == "put") {
        parsed_operation = DemoOperation::PUT;
        return true;
    }
    if (operation == "get") {
        parsed_operation = DemoOperation::GET;
        return true;
    }
    if (operation == "aggregate_put") {
        parsed_operation = DemoOperation::AGGREGATE_PUT;
        return true;
    }
    if (operation == "aggregate_get") {
        parsed_operation = DemoOperation::AGGREGATE_GET;
        return true;
    }
    return false;
}

bool IsPut(DemoOperation operation)
{
    return operation == DemoOperation::PUT || operation == DemoOperation::AGGREGATE_PUT;
}

bool RunCase(
    const Options& options, RuntimeResources& resources, uint64_t ffts_addr, const std::string& operation_name,
    const std::vector<uint8_t>& own_pattern)
{
    DemoOperation operation{};
    if (!ParseOperation(operation_name, operation)) {
        std::cerr << "unsupported operation: " << operation_name << std::endl;
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(options.qp) * ELEMENTS_PER_QP;
    uint8_t* src = resources.symmetric;
    uint8_t* dst = resources.symmetric + elements;
    int status = aclrtMemcpy(src, elements, own_pattern.data(), elements, ACL_MEMCPY_HOST_TO_DEVICE);
    status |= aclrtMemset(dst, elements, 0, elements);
    if (status != 0) {
        std::cerr << "prepare buffers failed, status=" << status << std::endl;
        aclshmem_global_exit(status);
        return false;
    }

    const int next_pe = options.pe + 1 == options.pes ? 0 : options.pe + 1;
    launch_rdma_qp_demo(
        options.qp, resources.stream, ffts_addr, resources.symmetric, elements, next_pe,
        static_cast<int32_t>(operation), SYNC_ID);
    status = aclrtSynchronizeStream(resources.stream);
    if (status != 0) {
        std::cerr << "aclrtSynchronizeStream failed, status=" << status << std::endl;
        aclshmem_global_exit(status);
        return false;
    }
    std::vector<uint8_t> result(elements);
    status = aclrtMemcpy(result.data(), elements, dst, elements, ACL_MEMCPY_DEVICE_TO_HOST);
    if (status != 0) {
        std::cerr << "copy result failed, status=" << status << std::endl;
        aclshmem_global_exit(status);
        return false;
    }
    const int previous_pe = options.pe == 0 ? options.pes - 1 : options.pe - 1;
    return Validate(result, IsPut(operation) ? previous_pe : next_pe, operation_name, options.pe);
}
} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseOptions(argc, argv, options)) {
        PrintUsage(argv[0]);
        return 1;
    }
    const bool valid_op = options.op == "put" || options.op == "get" || options.op == "all" ||
                          options.op == "aggregate_put" || options.op == "aggregate_get";
    const uint64_t local_pe_end = static_cast<uint64_t>(options.fpe) + static_cast<uint64_t>(options.gnpus);
    if (options.pes < 2 || options.pe < 0 || options.pe >= options.pes || options.gnpus <= 0 || options.fpe < 0 ||
        options.fnpu < 0 || local_pe_end > static_cast<uint64_t>(options.pes) || options.pe < options.fpe ||
        static_cast<uint64_t>(options.pe) >= local_pe_end || options.qp == 0 || options.qp > ACLSHMEM_MAX_QP_NUM ||
        !valid_op) {
        PrintUsage(argv[0]);
        return 1;
    }
    const int local_rank = options.pe - options.fpe;
    if (options.fnpu > std::numeric_limits<int>::max() - local_rank) {
        std::cerr << "-fnpu + local rank exceeds the supported device ID range" << std::endl;
        return 1;
    }

    RuntimeResources resources;
    resources.device_id = options.fnpu + local_rank;
    int status = aclInit(nullptr);
    if (status != 0) {
        std::cerr << "aclInit failed, status=" << status << std::endl;
        return status;
    }
    resources.acl_initialized = true;
    status = aclrtSetDevice(resources.device_id);
    if (status != 0) {
        std::cerr << "aclrtSetDevice failed, device=" << resources.device_id << " status=" << status << std::endl;
        return Cleanup(resources, status);
    }
    resources.device_set = true;
    status = aclrtCreateStream(&resources.stream);
    if (status != 0) {
        std::cerr << "aclrtCreateStream failed, status=" << status << std::endl;
        return Cleanup(resources, status);
    }
    resources.stream_created = true;

    aclshmemx_uniqueid_t uid{};
    aclshmemx_init_attr_t attr{};
    status = test_set_attr(options.pe, options.pes, HEAP_BYTES, options.ipport.c_str(), uid, &attr);
    if (status != 0) {
        std::cerr << "test_set_attr failed, status=" << status << std::endl;
        return Cleanup(resources, status);
    }
    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, options.qp);
    if (status != 0) {
        std::cerr << "aclshmemx_set_qp_num failed, qp=" << options.qp << " status=" << status << std::endl;
        return Cleanup(resources, status);
    }
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    if (status != 0) {
        std::cerr << "aclshmemx_init_attr failed, status=" << status << std::endl;
        return Cleanup(resources, status);
    }
    resources.shmem_initialized = true;

    const uint64_t elements = static_cast<uint64_t>(options.qp) * ELEMENTS_PER_QP;
    resources.symmetric = static_cast<uint8_t*>(aclshmem_malloc(static_cast<size_t>(elements * 2)));
    if (resources.symmetric == nullptr) {
        std::cerr << "aclshmem_malloc failed, bytes=" << elements * 2 << std::endl;
        return Cleanup(resources, 1);
    }

    const uint64_t ffts_addr = util_get_ffts_config();
    const std::vector<uint8_t> own_pattern = MakePattern(options.pe, elements);
    const std::vector<std::string> operations =
        options.op == "all" ? std::vector<std::string>{"put", "get", "aggregate_put", "aggregate_get"} :
                              std::vector<std::string>{options.op};
    bool passed = true;
    for (const auto& operation : operations) {
        passed = RunCase(options, resources, ffts_addr, operation, own_pattern) && passed;
    }

    status = Cleanup(resources, status);
    return status == 0 && passed ? 0 : 1;
}
