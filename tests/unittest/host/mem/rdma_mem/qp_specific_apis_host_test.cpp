/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <iostream>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "host/init/shmem_host_init.h"
#include "qp_specific_apis_test_kernels.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"

namespace {
constexpr size_t MESSAGE_SIZE = 64;
constexpr size_t AGGREGATE_MESSAGE_SIZE = 256 * sizeof(uint32_t);
constexpr uint32_t REQUESTED_QP_NUM = 2;
constexpr uint32_t RANK_OFFSET = 10;
constexpr int TIMEOUT = 30;

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
constexpr bool kQpSpecificBackendSupported = true;
#else
constexpr bool kQpSpecificBackendSupported = false;
#endif

using QpSpecificKernel = void (*)(uint32_t, void*, uint8_t*, uint64_t);

void init_rank_pattern(uint32_t* host_ptr, uint32_t rank_id, uint32_t rank_size, size_t message_size)
{
    std::memset(host_ptr, 0, message_size * rank_size);
    const size_t elem_per_rank = message_size / sizeof(uint32_t);
    for (size_t i = 0; i < elem_per_rank; ++i) {
        host_ptr[rank_id * elem_per_rank + i] = rank_id + RANK_OFFSET;
    }
}

void check_all_rank_pattern(const uint32_t* host_ptr, uint32_t rank_size, size_t message_size)
{
    const size_t elem_per_rank = message_size / sizeof(uint32_t);
    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        for (size_t elem = 0; elem < elem_per_rank; ++elem) {
            ASSERT_EQ(host_ptr[rank * elem_per_rank + elem], rank + RANK_OFFSET);
        }
    }
}

void run_qp_specific_case(
    aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size, uint32_t qp_num, size_t message_size,
    QpSpecificKernel kernel)
{
    const size_t total_size = message_size * rank_size;
    uint32_t* in_host = nullptr;
    uint32_t* out_host = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&in_host), total_size), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&out_host), total_size), 0);

    init_rank_pattern(in_host, rank_id, rank_size, message_size);
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    aclshmemi_control_barrier_all();
    kernel(qp_num, stream, gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(stream, TIMEOUT), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, gva, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    check_all_rank_pattern(out_host, rank_size, message_size);

    ASSERT_EQ(aclrtFreeHost(in_host), 0);
    ASSERT_EQ(aclrtFreeHost(out_host), 0);
}

void test_rdma_qp_specific_apis(int rank_id, int rank_size, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, REQUESTED_QP_NUM), 0);

    aclrtStream stream = nullptr;
    auto status = test_rdma_init(rank_id, rank_size, local_mem_size, &stream);
    ASSERT_EQ(status, 0);
    ASSERT_NE(stream, nullptr);

    const size_t total_size = AGGREGATE_MESSAGE_SIZE * rank_size;
    void* ptr = aclshmem_malloc(total_size);
    ASSERT_NE(ptr, nullptr);

    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, MESSAGE_SIZE,
        test_rdma_roce_qp_put_nbi_raw_do);
    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, MESSAGE_SIZE,
        test_rdma_roce_qp_get_nbi_raw_do);
    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, MESSAGE_SIZE,
        test_rdma_roce_qp_put_nbi_tensor_do);
    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, MESSAGE_SIZE,
        test_rdma_roce_qp_get_nbi_tensor_do);
    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, AGGREGATE_MESSAGE_SIZE,
        test_rdma_roce_qp_put_nbi_raw_aggregate_do);
    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, AGGREGATE_MESSAGE_SIZE,
        test_rdma_roce_qp_get_nbi_raw_aggregate_do);
    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, AGGREGATE_MESSAGE_SIZE,
        test_rdma_roce_qp_put_nbi_tensor_aggregate_do);
    run_qp_specific_case(
        stream, static_cast<uint8_t*>(ptr), rank_id, rank_size, REQUESTED_QP_NUM, AGGREGATE_MESSAGE_SIZE,
        test_rdma_roce_qp_get_nbi_tensor_aggregate_do);

    aclshmem_free(ptr);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}
} // namespace

TEST(TestMemApi, TestShmemRdmaQpSpecificApis)
{
    if (!kQpSpecificBackendSupported) {
        GTEST_SKIP() << "RDMA QP-specific APIs require XSCALE or HNS_1825 backend.";
    }

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_rdma_qp_specific_apis, local_mem_size, process_count);
}
