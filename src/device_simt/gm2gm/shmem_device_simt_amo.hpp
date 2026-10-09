/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SIMT_GM2GM_AMO_HPP
#define SHMEM_DEVICE_SIMT_GM2GM_AMO_HPP

#include "device/shmem_def.h"
#include "simt_api/device_functions.h"

namespace simt {
#define ACLSHMEM_TYPE_FUNC_SIMT_ATOMIC_CAS(FUNC) \
    FUNC(int32, int32_t);                        \
    FUNC(int64, int64_t);                        \
    FUNC(uint32, uint32_t);                      \
    FUNC(uint64, uint64_t)

// ========================================================

#define ACLSHMEM_ATOMIC_COMPARE_SWAP_SIMT(NAME, TYPE)                  \
    __simt_callee__ inline TYPE aclshmem_##NAME##_atomic_compare_swap( \
        __gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe)          \
    {                                                                  \
        auto ptr = simt::aclshmem_ptr(dest, pe);                       \
        __gm__ TYPE* addr_gm = reinterpret_cast<__gm__ TYPE*>(ptr);    \
        return asc_atomic_cas(addr_gm, cond, value);                   \
    }

ACLSHMEM_TYPE_FUNC_SIMT_ATOMIC_CAS(ACLSHMEM_ATOMIC_COMPARE_SWAP_SIMT);

#undef ACLSHMEM_ATOMIC_COMPARE_SWAP_SIMT
#undef ACLSHMEM_TYPE_FUNC_SIMT_ATOMIC_CAS

} // namespace simt
#endif
