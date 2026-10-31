/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_MEM_KERNEL_H
#define RDMA_MEM_KERNEL_H

#include <cstdint>

void test_rdma_put_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_rdma_get_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_rdma_put_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_rdma_get_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);

void test_rdma_aggregate(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);

#endif // RDMA_MEM_KERNEL_H
