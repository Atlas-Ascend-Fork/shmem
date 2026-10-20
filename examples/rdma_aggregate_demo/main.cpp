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

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "shmem.h"
#include "utils.h"

#include "rdma_aggregate_demo_common.h"
#include "rdma_aggregate_demo_kernel.h"

namespace {

using namespace rdma_aggregate_demo;

constexpr int kMinPes = 2;
constexpr uint64_t kLocalMemSize = 64UL * 1024UL * 1024UL;

aclshmemx_uniqueid_t g_default_flag_uid{};

struct VerificationCase {
    const char* name;
    uint32_t source_slot;
    uint32_t destination_slot;
    bool is_put;
};

constexpr VerificationCase kVerificationCases[] = {
    {"get_pointer_defer_submit", kGetPointerSource0, kGetPointerDestination0, false},
    {"get_pointer_submit", kGetPointerSource1, kGetPointerDestination1, false},
    {"get_tensor_defer_submit", kGetTensorSource0, kGetTensorDestination0, false},
    {"get_tensor_submit", kGetTensorSource1, kGetTensorDestination1, false},
    {"put_pointer_defer_submit", kPutPointerSource0, kPutPointerDestination0, true},
    {"put_pointer_submit", kPutPointerSource1, kPutPointerDestination1, true},
    {"put_pointer_loop_defer_0", kPutPointerLoopDeferSource0, kPutPointerLoopDeferDestination0, true},
    {"put_pointer_loop_defer_1", kPutPointerLoopDeferSource1, kPutPointerLoopDeferDestination1, true},
    {"put_pointer_loop_defer_2", kPutPointerLoopDeferSource2, kPutPointerLoopDeferDestination2, true},
    {"put_pointer_loop_defer_3", kPutPointerLoopDeferSource3, kPutPointerLoopDeferDestination3, true},
    {"put_pointer_loop_defer_submit", kPutPointerLoopDeferSource4, kPutPointerLoopDeferDestination4, true},
    {"put_tensor_defer_submit", kPutTensorSource0, kPutTensorDestination0, true},
    {"put_tensor_submit", kPutTensorSource1, kPutTensorDestination1, true},
};

int run_demo(int pe_id, int n_pes, const char* ip_port, int g_npus, int f_npu)
{
    if (n_pes < kMinPes) {
        std::cerr << "[ERROR] rdma_aggregate_demo requires at least 2 PEs." << std::endl;
        return -1;
    }

    const int device_id = pe_id % g_npus + f_npu;
    const size_t total_elements = kSlotCount * kElementCount;
    const size_t total_bytes = total_elements * sizeof(uint32_t);
    int status = 0;
    aclrtStream stream = nullptr;
    uint8_t* gva = nullptr;

    status = aclInit(nullptr);
    ACL_CHECK(status);
    if (status != ACL_ERROR_NONE) {
        return status;
    }

    status = aclrtSetDevice(device_id);
    ACL_CHECK(status);
    if (status != ACL_ERROR_NONE) {
        ACL_CHECK(aclFinalize());
        return status;
    }

    status = aclrtCreateStream(&stream);
    ACL_CHECK(status);
    if (status != ACL_ERROR_NONE) {
        ACL_CHECK(aclrtResetDevice(device_id));
        ACL_CHECK(aclFinalize());
        return status;
    }

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, kLocalMemSize, ip_port, g_default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    ACL_CHECK(status);
    if (status != ACL_ERROR_NONE) {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(device_id));
        ACL_CHECK(aclFinalize());
        return status;
    }

    gva = static_cast<uint8_t*>(aclshmem_malloc(total_bytes));
    if (gva == nullptr) {
        std::cerr << "[ERROR] aclshmem_malloc failed." << std::endl;
        ACL_CHECK(aclshmem_finalize());
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(device_id));
        ACL_CHECK(aclFinalize());
        return -1;
    }

    std::vector<uint32_t> host_data(total_elements);
    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        for (uint32_t element = 0; element < kElementCount; ++element) {
            host_data[slot * kElementCount + element] = value_for(static_cast<uint32_t>(pe_id), slot, element);
        }
    }

    status = aclrtMemcpy(gva, total_bytes, host_data.data(), total_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_CHECK(status);
    if (status == ACL_ERROR_NONE) {
        launch_rdma_aggregate_demo(1, stream, gva);
        status = aclrtSynchronizeStream(stream);
        ACL_CHECK(status);
    }

    if (status == ACL_ERROR_NONE) {
        status = aclrtMemcpy(host_data.data(), total_bytes, gva, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        ACL_CHECK(status);
    }

    if (status == ACL_ERROR_NONE) {
        // Get 拉取下一个 PE 的数据，而本 PE 收到的 Put 来自上一个 PE，两者的来源 PE 需要分开校验。
        const uint32_t get_peer = static_cast<uint32_t>((pe_id + 1) % n_pes);
        const uint32_t put_peer = static_cast<uint32_t>((pe_id + n_pes - 1) % n_pes);
        for (const VerificationCase& test_case : kVerificationCases) {
            const uint32_t source_pe = test_case.is_put ? put_peer : get_peer;
            for (uint32_t element = 0; element < kElementCount; ++element) {
                const uint32_t actual = host_data[test_case.destination_slot * kElementCount + element];
                const uint32_t expected = value_for(source_pe, test_case.source_slot, element);
                if (actual != expected) {
                    std::cerr << "[ERROR] " << test_case.name << " failed at element " << element
                              << ": actual=" << actual << ", expected=" << expected << std::endl;
                    status = -1;
                    break;
                }
            }
            if (status != ACL_ERROR_NONE) {
                break;
            }
        }
    }

    if (status == ACL_ERROR_NONE) {
        std::cout << "[SUCCESS] rdma aggregate pointer/tensor defer/submit overloads, including 4-defer pointer put,"
                  << " passed on PE " << pe_id << std::endl;
    }

    aclshmem_free(gva);
    ACL_CHECK(aclshmem_finalize());
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());
    return status;
}

} // namespace

int main(int argc, char* argv[])
{
    constexpr int kExpectedArgc = 7;
    if (argc != kExpectedArgc) {
        std::cerr << "Usage: " << argv[0] << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>" << std::endl;
        return -1;
    }

    const int n_pes = std::atoi(argv[1]);
    const int pe_id = std::atoi(argv[2]);
    const int g_npus = std::atoi(argv[4]);
    const int f_pe = std::atoi(argv[5]);
    const int f_npu = std::atoi(argv[6]);
    if (g_npus <= 0 || f_pe < 0 || pe_id < 0 || pe_id >= n_pes) {
        std::cerr << "[ERROR] Invalid PE or NPU arguments." << std::endl;
        return -1;
    }

    return run_demo(pe_id, n_pes, argv[3], g_npus, f_npu);
}
