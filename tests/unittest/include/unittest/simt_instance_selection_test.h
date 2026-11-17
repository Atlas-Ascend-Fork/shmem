/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_SIMT_INSTANCE_SELECTION_TEST_H
#define ACLSHMEM_SIMT_INSTANCE_SELECTION_TEST_H
#include <cstdint>
namespace simt_instance_selection_test {
constexpr uint32_t SLOT = 4096;
constexpr uint32_t LANES = 128;
constexpr uint32_t ID_FIELDS = 11;
constexpr int32_t SENTINEL = 0x5a5a5a5a;
enum mode_t {
    GM_PUT = 0,
    GM_GET = 6,
    UB_PUT = 12,
    UB_GET = 18,
    BYTE_GM_PUT = 24,
    BYTE_GM_GET = 25,
    BYTE_UB_PUT = 26,
    BYTE_UB_GET = 27,
    SCALAR_P = 28,
    SCALAR_G = 29,
    CAS = 30,
    LEGACY = 31,
    MODE_COUNT = 32
};
enum variant_t { THREAD = 0, WARP = 1, BLOCK = 2, THREAD_NBI = 3, WARP_NBI = 4, BLOCK_NBI = 5, VARIANT_COUNT = 6 };
constexpr int LENGTH_COUNT = 4;
inline bool is_get(int mode)
{
    return (mode >= GM_GET && mode < UB_PUT) || (mode >= UB_GET && mode < BYTE_GM_PUT) || mode == BYTE_GM_GET ||
           mode == BYTE_UB_GET || mode == SCALAR_G;
}
} // namespace simt_instance_selection_test
#endif
