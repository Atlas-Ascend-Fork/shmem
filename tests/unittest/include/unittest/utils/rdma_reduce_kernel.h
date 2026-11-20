/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UT_RDMA_REDUCE_KERNEL_H
#define UT_RDMA_REDUCE_KERNEL_H

#include <cstdint>
#include "host_device/shmem_common_types.h"

#define ACLSHMEM_RDMA_REDUCE_TEST_TYPES(FUNC)             \
    FUNC(int8, int8_t, int8_t, int8_t);                   \
    FUNC(int16, int16_t, int16_t, int16_t);               \
    FUNC(int32, int32_t, int32_t, int32_t);               \
    FUNC(uint32, uint32_t, uint32_t, uint32_t);           \
    FUNC(half, op::fp16_t, half, half);                   \
    FUNC(float, float, float, float);                     \
    FUNC(bfloat16, op::bfloat16, bfloat16_t, bfloat16_t); \
    FUNC(half_sat, op::fp16_t, half, aclshmemx_half_sat_t)

enum class RdmaReduceApi : uint32_t {
    POINTER_DEFAULT = 0,
    POINTER_EVENT,
    POINTER_QP,
    TENSOR_DEFAULT,
    TENSOR_EVENT,
    TENSOR_QP,
};

#define DECLARE_RDMA_REDUCE_KERNEL(NAME, HOST_TYPE, DEVICE_TYPE, REDUCE_TYPE)                                       \
    void test_rdma_put_reduce_##NAME##_do(                                                                          \
        uint32_t block_dim, void* stream, uint8_t* dst, uint8_t* src, uint32_t elem_size, aclshmemx_reduce_op_t op, \
        RdmaReduceApi api, uint32_t qp_idx, uint32_t iterations, uint64_t config)
ACLSHMEM_RDMA_REDUCE_TEST_TYPES(DECLARE_RDMA_REDUCE_KERNEL);
#undef DECLARE_RDMA_REDUCE_KERNEL

enum RdmaReduceQueueField : uint32_t {
    REDUCE_SQ_DEPTH = 0,
    REDUCE_CQ_DEPTH,
    REDUCE_SQ_HEAD,
    REDUCE_SQ_TAIL,
    REDUCE_CQ_TAIL,
    REDUCE_QUEUE_FIELDS,
};

void test_rdma_put_reduce_queue_do(
    uint32_t block_dim, void* stream, int32_t* dst, int32_t* src, uint32_t count, bool complete, uint32_t* queue_state,
    uint64_t config);

#endif // UT_RDMA_REDUCE_KERNEL_H
