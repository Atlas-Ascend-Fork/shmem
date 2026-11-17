/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SIMT_INSTANCE_EXTENDED_TEST_H
#define SHMEM_SIMT_INSTANCE_EXTENDED_TEST_H
#include <cstdint>
namespace simt_instance_extended_test {
constexpr uint32_t REGION_BYTES = 16384;
constexpr uint32_t HEAP_BYTES = 3 * REGION_BYTES;
constexpr uint32_t BLOCK_BYTES = 4096;
constexpr uint32_t PHASE_BYTES = 2048;
constexpr uint32_t LANES = 128;
constexpr uint32_t ID_FIELDS = 4;
constexpr uint32_t MAX_BLOCKS = 4;
constexpr uint8_t GUARD = 0x5a;
enum dtype_t { INT8, INT16, INT32, INT64, UINT32, UINT64, FP16, FP32, BF16, TYPE_COUNT };
constexpr uint32_t TYPE_BYTES[TYPE_COUNT] = {1, 2, 4, 8, 4, 8, 2, 4, 2};
// Six participation/completion variants for each of GM put/get and UB put/get.
constexpr int MODE_COUNT = 24;
inline bool is_get(int mode) { return mode / 6 == 1 || mode / 6 == 3; }
} // namespace simt_instance_extended_test
#endif
