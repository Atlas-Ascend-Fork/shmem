/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UDMA_PUT_VALUE_REDUCE_KERNEL_H
#define UDMA_PUT_VALUE_REDUCE_KERNEL_H

#include <cstdint>

// Shared host/kernel layout contract: one 64-byte slot per region, five regions per PE.
// The UDMA UT (tests/unittest/{device,host}/mem/udma_mem) uses a 4+1 layout for the same feature;
// keep both layouts in sync when the slot semantics change.
constexpr uint32_t SLOT_BYTES = 64;
constexpr uint32_t REGIONS_PER_PE = 5;
constexpr uint32_t ELEMS_PER_SLOT = SLOT_BYTES / sizeof(int32_t);
constexpr uint64_t VALUE_MAGIC = 0x5A5A000000000000ULL;

void launch_udma_put_value_reduce(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dump);

#endif // UDMA_PUT_VALUE_REDUCE_KERNEL_H
