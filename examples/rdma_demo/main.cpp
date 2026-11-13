/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "shmem.h"
#include "host/utils/shmem_host_exception.h"
#include "shmemi_host_common.h"
#include "utils.h"

int g_npus = 8;
const char* ipport;
int f_pe = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length);

aclshmemx_uniqueid_t default_flag_uid;

int test_aclshmem_team_all_gather(int pe_id, int n_pes, uint64_t local_mem_size, int iterations)
{
    // 初始化ACL和ACLSHMEM
    int32_t device_id = pe_id - f_pe + f_npu;
    int status = 0;
    const int num10 = 10;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    if (status != ACL_SUCCESS) {
        std::cerr << "aclInit failed: " << status << std::endl;
        return status;
    }
    status = aclrtSetDevice(device_id);
    if (status != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed: " << status << std::endl;
        (void)aclFinalize();
        return status;
    }
    status = aclrtCreateStream(&stream);
    if (status != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed: " << status << std::endl;
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "aclshmemx_init_attr failed: " << status << std::endl;
        (void)aclrtDestroyStream(stream);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }

    // Enable the internal Runtime exception snapshot and RDMA queue diagnostics.
    status = aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG);
    if (status == ACLSHMEM_NOT_SUPPORTED) {
        std::cout << "runtime exception report is not supported, skip enabling" << std::endl;
        status = ACLSHMEM_SUCCESS;
    }
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "aclshmemx_enable_exception_report failed: " << status << std::endl;
        (void)aclshmem_finalize();
        (void)aclrtDestroyStream(stream);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }

    uint8_t* ptr = static_cast<uint8_t*>(aclshmem_malloc(1024));
    if (ptr == nullptr) {
        std::cerr << "aclshmem_malloc failed" << std::endl;
        (void)aclshmem_finalize();
        (void)aclrtDestroyStream(stream);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return ACLSHMEM_INNER_ERROR;
    }

    // 初始化数据
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (pe_id + num10);
    }

    status |= aclrtMemcpy(
        ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t), input.data(),
        trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // AllGather
    for (int iteration = 0; iteration < iterations; ++iteration) {
        allgather_demo(1, stream, (uint8_t*)ptr, trans_size * sizeof(int32_t));
    }
    status |= aclrtSynchronizeStream(stream);
    // The stream synchronization is the host safe point for reporting a device trap.
    const int report_status = aclshmemx_report_exception();
    if (report_status != ACLSHMEM_SUCCESS) {
        std::cerr << "aclshmemx_report_exception failed: " << report_status << std::endl;
    }
    status |= report_status;

    // 结果校验打印
    int32_t* y_host;
    size_t input_size = n_pes * trans_size * sizeof(int32_t);
    status |= aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);
    status |= aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    const int block_size = 16;
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < block_size; j++) {
            if (y_host[trans_size * i + trans_size / block_size * j] != num10 + i) {
                std::cout << y_host[trans_size * i + trans_size / block_size * j] << " != " << num10 + i << std::endl;
                status |= aclrtFreeHost(y_host);
                aclshmem_free(ptr);
                status |= aclshmem_finalize();
                status |= aclrtDestroyStream(stream);
                status |= aclrtResetDevice(device_id);
                status |= aclFinalize();
                return -1;
            }
        }
    }
    std::cout << "check transport result success, relative pe=" << pe_id << std::endl;
    // 去初始化
    status |= aclrtFreeHost(y_host);
    aclshmem_free(ptr);
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return status;
}

int main(int argc, char* argv[])
{
    if (argc != 7 && argc != 9) {
        std::cerr << "Usage: " << argv[0] << " PES PE IPPORT GNPUS FPE FNPU [-n N]" << std::endl;
        return 1;
    }
    int argIdx = 1;
    int status = 0;
    int n_pes = atoi(argv[argIdx++]);
    int pe_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_pe = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);
    if (n_pes < 2 || g_npus <= 0 || g_npus > n_pes || f_pe < 0 || f_pe > n_pes - g_npus || pe_id < f_pe ||
        pe_id - f_pe >= g_npus || f_npu < 0 || f_npu > std::numeric_limits<int>::max() - (pe_id - f_pe)) {
        std::cerr << "Invalid PE/device mapping; use a PE in the local range and one process per NPU." << std::endl;
        return 1;
    }
    int iterations = 1;
    if (argc == 9) {
        char* end = nullptr;
        const long parsed = std::strtol(argv[argIdx + 1], &end, 10);
        if (std::string(argv[argIdx]) != "-n" || end == argv[argIdx + 1] || *end != '\0' || parsed < 1 ||
            parsed > std::numeric_limits<int>::max()) {
            std::cerr << "-n must be a positive integer" << std::endl;
            return 1;
        }
        iterations = static_cast<int>(parsed);
    }
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_aclshmem_team_all_gather(pe_id, n_pes, local_mem_size, iterations);
    if (status != 0) {
        std::cout << "[ERROR] demo run failed in relative pe " << pe_id << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] demo run success in relative pe " << pe_id << std::endl;
    return 0;
}
