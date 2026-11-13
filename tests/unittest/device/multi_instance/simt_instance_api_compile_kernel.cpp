/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#define MULTI_INSTANCE
#include "shmem.h"

#define CHECK_VARIANT(PREFIX, NAME, SUFFIX)                  \
    simt::PREFIX##NAME##_put##SUFFIX(dst, src, count, peer); \
    simt::PREFIX##NAME##_get##SUFFIX(dst, src, count, peer); \
    simt::PREFIX##NAME##_put##SUFFIX(dst, ub, count, peer);  \
    simt::PREFIX##NAME##_get##SUFFIX(ub, src, count, peer);  \
    simt::PREFIX##putmem##SUFFIX(dst, src, count, peer);     \
    simt::PREFIX##getmem##SUFFIX(dst, src, count, peer);     \
    simt::PREFIX##putmem##SUFFIX(dst, ub, count, peer);      \
    simt::PREFIX##getmem##SUFFIX(ub, src, count, peer)

template <typename T>
__simt_vf__ __launch_bounds__(128) inline void instance_api_types(
    __gm__ T* dst, __gm__ T* src, size_t count, int peer, uint64_t id)
{
    __ubuf__ T ub[16384 / sizeof(T)];
    simt::aclshmemx_instance_ctx_set(id);
#define CHECK_TYPED(NAME, TYPE)                                                     \
    if constexpr (std::is_same<T, TYPE>::value) {                                   \
        CHECK_VARIANT(aclshmem_, NAME, );                                           \
        CHECK_VARIANT(aclshmemx_, NAME, _warp);                                     \
        CHECK_VARIANT(aclshmemx_, NAME, _block);                                    \
        CHECK_VARIANT(aclshmem_, NAME, _nbi);                                       \
        CHECK_VARIANT(aclshmemx_, NAME, _nbi_warp);                                 \
        CHECK_VARIANT(aclshmemx_, NAME, _nbi_block);                                \
        simt::aclshmem_##NAME##_p(dst, simt::aclshmem_##NAME##_g(src, peer), peer); \
    }
    CHECK_TYPED(half, half)
    CHECK_TYPED(float, float)
    CHECK_TYPED(int8, int8_t)
    CHECK_TYPED(int16, int16_t)
    CHECK_TYPED(int32, int32_t)
    CHECK_TYPED(int64, int64_t)
    CHECK_TYPED(uint8, uint8_t)
    CHECK_TYPED(uint16, uint16_t)
    CHECK_TYPED(uint32, uint32_t)
    CHECK_TYPED(uint64, uint64_t)
    CHECK_TYPED(bfloat16, bfloat16_t)
#undef CHECK_TYPED
#define CHECK_CAS(NAME, TYPE)                                                                                \
    if constexpr (std::is_same<T, TYPE>::value) {                                                            \
        dst[threadIdx.x] = simt::aclshmem_##NAME##_atomic_compare_swap(src + threadIdx.x, T(0), T(1), peer); \
    }
    CHECK_CAS(int32, int32_t)
    CHECK_CAS(int64, int64_t)
    CHECK_CAS(uint32, uint32_t)
    CHECK_CAS(uint64, uint64_t)
#undef CHECK_CAS
}
__global__ __vector__ void simt_instance_api_compile(
    GM_ADDR dst, GM_ADDR src, size_t count, int peer, uint64_t id, int type)
{
#define CHECK_TYPE(N, T)                                                                                \
    case N:                                                                                             \
        asc_vf_call<instance_api_types<T>>(dim3(128), (__gm__ T*)dst, (__gm__ T*)src, count, peer, id); \
        break
    switch (type) {
        CHECK_TYPE(0, half);
        CHECK_TYPE(1, float);
        CHECK_TYPE(2, int8_t);
        CHECK_TYPE(3, int16_t);
        CHECK_TYPE(4, int32_t);
        CHECK_TYPE(5, int64_t);
        CHECK_TYPE(6, uint8_t);
        CHECK_TYPE(7, uint16_t);
        CHECK_TYPE(8, uint32_t);
        CHECK_TYPE(9, uint64_t);
        CHECK_TYPE(10, signed char);
        CHECK_TYPE(11, unsigned char);
        CHECK_TYPE(12, bfloat16_t);
        default:
            return;
    }
#undef CHECK_TYPE
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}
#undef CHECK_VARIANT
