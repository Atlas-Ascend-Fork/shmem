/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_SIMT_GM2GM_AMO_H
#define SHMEMI_DEVICE_SIMT_GM2GM_AMO_H

#include <type_traits>
#include "device/shmem_def.h"
#include "simt_api/device_functions.h"

namespace simt {
template <typename T>
__simt_callee__ inline T aclshmemi_atomic_compare_swap(__gm__ T* dst, T expected, T desired, int pe)
{
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value || std::is_same<T, uint32_t>::value ||
            std::is_same<T, uint64_t>::value,
        "SIMT CAS requires signed/unsigned 32/64-bit integers");
    return asc_atomic_cas(reinterpret_cast<__gm__ T*>(aclshmemi_ptr(dst, pe)), expected, desired);
}

__simt_callee__ inline int32_t aclshmem_int32_atomic_compare_swap(
    __gm__ int32_t* dest, int32_t cond, int32_t value, int32_t pe)
{
    return aclshmemi_atomic_compare_swap(dest, cond, value, pe);
}

__simt_callee__ inline int64_t aclshmem_int64_atomic_compare_swap(
    __gm__ int64_t* dest, int64_t cond, int64_t value, int32_t pe)
{
    return aclshmemi_atomic_compare_swap(dest, cond, value, pe);
}

__simt_callee__ inline uint32_t aclshmem_uint32_atomic_compare_swap(
    __gm__ uint32_t* dest, uint32_t cond, uint32_t value, int32_t pe)
{
    return aclshmemi_atomic_compare_swap(dest, cond, value, pe);
}

__simt_callee__ inline uint64_t aclshmem_uint64_atomic_compare_swap(
    __gm__ uint64_t* dest, uint64_t cond, uint64_t value, int32_t pe)
{
    return aclshmemi_atomic_compare_swap(dest, cond, value, pe);
}

} // namespace simt
#endif
