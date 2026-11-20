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
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "opdev/fp16_t.h"
#include "shmem.h"
#include "utils.h"
#include "rdma_reduce_demo_kernel.h"

static int parse_int_arg(const char* text, const char* name)
{
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid " << name << ": " << text << std::endl;
        return -1;
    }
    return static_cast<int>(parsed);
}

int test_aclshmem_rdma_reduce(int pe_id, int n_pes, const char* ipport, int device_id)
{
    // Initialize ACL and ACLSHMEM.
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclInit(nullptr);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclInit failed, status=%d", status), return status);
    status = aclrtSetDevice(device_id);
    if (status != 0) {
        ERROR_LOG("aclrtSetDevice failed, status=%d", status);
        (void)aclFinalize();
        return status;
    }
    status = aclrtCreateStream(&stream);
    if (status != 0) {
        ERROR_LOG("aclrtCreateStream failed, status=%d", status);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }

    aclshmemx_uniqueid_t default_flag_uid{};
    aclshmemx_init_attr_t attributes{};
    const uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    status = test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    if (status != 0) {
        ERROR_LOG("test_set_attr failed, status=%d", status);
        (void)aclrtDestroyStream(stream);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    attributes.comm_args = nullptr;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (status != 0) {
        ERROR_LOG("aclshmemx_init_attr failed, status=%d", status);
        (void)aclrtDestroyStream(stream);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }
    // After initialization, fatal errors return without entering unmatched collective cleanup.
    status = aclshmemx_set_rdma_config(0, UB_BYTES, 0);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclshmemx_set_rdma_config failed, status=%d", status), return status);

    // One source segment per target PE, followed by ADD/MAX/MIN results and a half source/target pair.
    const size_t source_elements = static_cast<size_t>(n_pes) * ELEMENT_COUNT;
    const size_t integer_bytes = (source_elements + 3 * ELEMENT_COUNT) * sizeof(int32_t);
    std::array<op::fp16_t, 2> sat_values;
    // Each incoming 40000 contribution saturates the target at the largest finite half value, 65504.
    sat_values.fill(static_cast<op::fp16_t>(40000.0F));
    const size_t buffer_bytes = integer_bytes + sizeof(sat_values);
    auto buffer = static_cast<uint8_t*>(aclshmem_malloc(buffer_bytes));
    if (buffer == nullptr) {
        std::cerr << "[ERROR] pe=" << pe_id << " aclshmem_malloc failed" << std::endl;
        return 1;
    }

    std::vector<int32_t> values(source_elements + 3 * ELEMENT_COUNT);
    for (int peer = 0; peer < n_pes; ++peer) {
        for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
            values[peer * ELEMENT_COUNT + i] = (pe_id + 1) * (peer + 1) * (i + 1);
        }
    }
    // Each target starts with this PE's contribution to its own segment.
    auto* results = values.data() + source_elements;
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        const int32_t value = values[pe_id * ELEMENT_COUNT + i];
        results[i] = value;
        results[ELEMENT_COUNT + i] = value;
        results[2 * ELEMENT_COUNT + i] = value;
    }
    status = aclrtMemcpy(buffer, integer_bytes, values.data(), integer_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("copy input failed, status=%d", status), return status);
    auto* sat_buffer = buffer + integer_bytes;
    status =
        aclrtMemcpy(sat_buffer, sizeof(sat_values), sat_values.data(), sizeof(sat_values), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("copy SAT input failed, status=%d", status), return status);

    launch_rdma_reduce_demo(1, stream, buffer);
    status = aclrtSynchronizeStream(stream);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclrtSynchronizeStream failed, status=%d", status), return status);

    status = aclrtMemcpy(values.data(), integer_bytes, buffer, integer_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("copy result failed, status=%d", status), return status);
    status =
        aclrtMemcpy(sat_values.data(), sizeof(sat_values), sat_buffer, sizeof(sat_values), ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("copy SAT result failed, status=%d", status), return status);
    const char* operations[] = {"ADD", "MAX", "MIN"};
    // Sender s contributes (s + 1) * (pe_id + 1) * (i + 1) to local element i.
    const int64_t expected[3] = {static_cast<int64_t>(n_pes) * (n_pes + 1LL) / 2, n_pes, 1};
    for (uint32_t op = 0; op < 3; ++op) {
        for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
            const int64_t expected_value = expected[op] * (pe_id + 1) * (i + 1);
            if (results[op * ELEMENT_COUNT + i] != expected_value) {
                std::cerr << "[ERROR] pe=" << pe_id << " " << operations[op] << "[" << i
                          << "]=" << results[op * ELEMENT_COUNT + i] << ", expected=" << expected_value << std::endl;
                status = 1;
            }
        }
    }
    const float sat_result = static_cast<float>(sat_values[1]);
    if (sat_result != 65504.0F) {
        std::cerr << "[ERROR] pe=" << pe_id << " ADD FP16_SAT=" << sat_result << ", expected=65504" << std::endl;
        status = 1;
    }
    if (status == 0) {
        std::cout << "check transport result success, relative pe=" << pe_id << std::endl;
    }

    // Release resources after all PEs have completed the kernel.
    aclshmem_free(buffer);
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return status;
}

int main(int argc, char* argv[])
{
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>" << std::endl;
        return 1;
    }
    int argIdx = 1;
    int n_pes = parse_int_arg(argv[argIdx++], "n_pes");
    int pe_id = parse_int_arg(argv[argIdx++], "pe_id");
    const char* ipport = argv[argIdx++];
    int g_npus = parse_int_arg(argv[argIdx++], "g_npus");
    int f_pe = parse_int_arg(argv[argIdx++], "f_pe");
    int f_npu = parse_int_arg(argv[argIdx++], "f_npu");
    if (n_pes < 2 || pe_id < 0 || pe_id >= n_pes) {
        std::cerr << "This demo requires at least two PEs, with pe_id in [0, n_pes)." << std::endl;
        return 1;
    }
    const int64_t device_id = static_cast<int64_t>(pe_id) - f_pe + f_npu;
    if (g_npus <= 0 || g_npus > n_pes || f_pe < 0 || f_npu < 0 || f_pe > pe_id || pe_id - f_pe >= g_npus ||
        device_id > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid PE/device mapping; use one process per NPU." << std::endl;
        return 1;
    }
    const int status = test_aclshmem_rdma_reduce(pe_id, n_pes, ipport, static_cast<int>(device_id));
    if (status != 0) {
        std::cerr << "[ERROR] demo run failed in relative pe " << pe_id << ", status=" << status << std::endl;
    } else {
        std::cout << "[SUCCESS] demo run success in relative pe " << pe_id << std::endl;
    }
    return status;
}
