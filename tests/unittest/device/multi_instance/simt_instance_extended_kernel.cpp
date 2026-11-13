/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#define MULTI_INSTANCE
#include "shmem.h"
#include "unittest/simt_instance_extended_test.h"

using namespace simt_instance_extended_test;

// All lanes enter every cooperative call, including empty ranges. Thread calls
// use lane zero with the full length, including zero. Each AIV owns one disjoint
// 4096-byte slice. Each call owns its VF-local UB staging storage.
template <typename T, bool byte>
__simt_callee__ inline void extended_move(__gm__ T* heap, uint32_t offset, uint32_t count, int mode)
{
    __ubuf__ T ub[PHASE_BYTES / sizeof(T)];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = lane / 32;
    const uint32_t warp_count = count / 4 + (warp < count % 4);
    const uint32_t warp_begin = warp * (count / 4) + (warp < count % 4 ? warp : count % 4);
    const int peer = (simt::aclshmem_my_pe() + 1) % simt::aclshmem_n_pes();
    auto src = heap + offset / sizeof(T);
    auto dst = heap + (REGION_BYTES + offset) / sizeof(T);
    if (mode / 6 == 2)
        for (uint32_t i = lane; i < count; i += LANES)
            ub[i] = src[i];
    asc_syncthreads();
#define EXTENDED_VARIANTS(OP, DST, SRC) EXTENDED_VARIANTS_IMPL(OP, DST, SRC)
#define EXTENDED_VARIANTS_IMPL(OP, DST, SRC)                                                               \
    do {                                                                                                   \
        switch (mode % 6) {                                                                                \
            case 0:                                                                                        \
                if (lane == 0)                                                                             \
                    simt::aclshmem_##OP(DST, SRC, count, peer);                                            \
                break;                                                                                     \
            case 1:                                                                                        \
                simt::aclshmemx_##OP##_warp((DST) + warp_begin, (SRC) + warp_begin, warp_count, peer);     \
                break;                                                                                     \
            case 2:                                                                                        \
                simt::aclshmemx_##OP##_block(DST, SRC, count, peer);                                       \
                break;                                                                                     \
            case 3:                                                                                        \
                if (lane == 0)                                                                             \
                    simt::aclshmem_##OP##_nbi(DST, SRC, count, peer);                                      \
                break;                                                                                     \
            case 4:                                                                                        \
                simt::aclshmemx_##OP##_nbi_warp((DST) + warp_begin, (SRC) + warp_begin, warp_count, peer); \
                break;                                                                                     \
            case 5:                                                                                        \
                simt::aclshmemx_##OP##_nbi_block(DST, SRC, count, peer);                                   \
                break;                                                                                     \
            default:                                                                                       \
                break;                                                                                     \
        }                                                                                                  \
    } while (0)
    if constexpr (byte) {
        if (mode / 6 == 0) {
            EXTENDED_VARIANTS(putmem, dst, src);
        } else if (mode / 6 == 1) {
            EXTENDED_VARIANTS(getmem, dst, src);
        } else if (mode / 6 == 2) {
            EXTENDED_VARIANTS(putmem, dst, ub);
        } else {
            EXTENDED_VARIANTS(getmem, ub, src);
        }
    } else {
#define EXTENDED_TYPED(NAME, TYPE)                       \
    do {                                                 \
        if constexpr (__is_same(T, TYPE)) {              \
            if (mode / 6 == 0) {                         \
                EXTENDED_VARIANTS(NAME##_put, dst, src); \
            } else if (mode / 6 == 1) {                  \
                EXTENDED_VARIANTS(NAME##_get, dst, src); \
            } else if (mode / 6 == 2) {                  \
                EXTENDED_VARIANTS(NAME##_put, dst, ub);  \
            } else {                                     \
                EXTENDED_VARIANTS(NAME##_get, ub, src);  \
            }                                            \
        }                                                \
    } while (0)
        EXTENDED_TYPED(int8, int8_t);
        EXTENDED_TYPED(int16, int16_t);
        EXTENDED_TYPED(int32, int32_t);
        EXTENDED_TYPED(int64, int64_t);
        EXTENDED_TYPED(uint32, uint32_t);
        EXTENDED_TYPED(uint64, uint64_t);
        EXTENDED_TYPED(half, half);
        EXTENDED_TYPED(float, float);
        EXTENDED_TYPED(bfloat16, bfloat16_t);
#undef EXTENDED_TYPED
    }
#undef EXTENDED_VARIANTS
#undef EXTENDED_VARIANTS_IMPL
    asc_syncthreads();
    if (mode / 6 == 3)
        for (uint32_t i = lane; i < count; i += LANES)
            dst[i] = ub[i];
    asc_syncthreads();
}

template <typename T, bool byte>
__simt_vf__ __launch_bounds__(LANES) inline void extended_instance_ops(
    __gm__ T* a, __gm__ T* b, uint64_t id_a, uint64_t id_b, uint32_t block, uint32_t count_a, uint32_t count_b,
    uint32_t shift, int mode, __gm__ uint64_t* identity)
{
    const uint32_t offset = block * BLOCK_BYTES + shift;
    // All lanes select the same instance together within this VF.
    simt::aclshmemx_instance_ctx_set(id_a);
    const uint64_t addr_a = simt::aclshmemx_instance_ctx_get();
    const int rank_a = simt::aclshmem_my_pe();
    extended_move<T, byte>(a, offset, count_a, mode);
    simt::aclshmemx_instance_ctx_set(id_b);
    const uint64_t addr_b = simt::aclshmemx_instance_ctx_get();
    const int rank_b = simt::aclshmem_my_pe();
    extended_move<T, byte>(b, offset, count_b, mode);
    simt::aclshmemx_instance_ctx_set(id_a);
    extended_move<T, byte>(a, offset + PHASE_BYTES, count_a, mode);
    simt::aclshmemx_instance_ctx_set(id_b);
    extended_move<T, byte>(b, offset + PHASE_BYTES, count_b, mode);
    if (threadIdx.x == 0) {
        identity[block * ID_FIELDS] = addr_a;
        identity[block * ID_FIELDS + 1] = addr_b;
        identity[block * ID_FIELDS + 2] = rank_a;
        identity[block * ID_FIELDS + 3] = rank_b;
    }
}

__global__ __vector__ void simt_instance_extended_kernel(
    GM_ADDR a, GM_ADDR b, uint64_t id_a, uint64_t id_b, uint32_t count_a, uint32_t count_b, uint32_t shift, int mode,
    int type, bool byte, GM_ADDR identity)
{
    const uint32_t block = AscendC::GetBlockIdx();
    if (byte) {
        asc_vf_call<extended_instance_ops<uint8_t, true>>(
            dim3(LANES), (__gm__ uint8_t*)a, (__gm__ uint8_t*)b, id_a, id_b, block, count_a, count_b, shift, mode,
            (__gm__ uint64_t*)identity);
    } else {
        switch (type) {
            case INT8:
                asc_vf_call<extended_instance_ops<int8_t, false>>(
                    dim3(LANES), (__gm__ int8_t*)a, (__gm__ int8_t*)b, id_a, id_b, block, count_a, count_b, shift, mode,
                    (__gm__ uint64_t*)identity);
                break;
            case INT16:
                asc_vf_call<extended_instance_ops<int16_t, false>>(
                    dim3(LANES), (__gm__ int16_t*)a, (__gm__ int16_t*)b, id_a, id_b, block, count_a, count_b, shift,
                    mode, (__gm__ uint64_t*)identity);
                break;
            case INT32:
                asc_vf_call<extended_instance_ops<int32_t, false>>(
                    dim3(LANES), (__gm__ int32_t*)a, (__gm__ int32_t*)b, id_a, id_b, block, count_a, count_b, shift,
                    mode, (__gm__ uint64_t*)identity);
                break;
            case INT64:
                asc_vf_call<extended_instance_ops<int64_t, false>>(
                    dim3(LANES), (__gm__ int64_t*)a, (__gm__ int64_t*)b, id_a, id_b, block, count_a, count_b, shift,
                    mode, (__gm__ uint64_t*)identity);
                break;
            case UINT32:
                asc_vf_call<extended_instance_ops<uint32_t, false>>(
                    dim3(LANES), (__gm__ uint32_t*)a, (__gm__ uint32_t*)b, id_a, id_b, block, count_a, count_b, shift,
                    mode, (__gm__ uint64_t*)identity);
                break;
            case UINT64:
                asc_vf_call<extended_instance_ops<uint64_t, false>>(
                    dim3(LANES), (__gm__ uint64_t*)a, (__gm__ uint64_t*)b, id_a, id_b, block, count_a, count_b, shift,
                    mode, (__gm__ uint64_t*)identity);
                break;
            case FP16:
                asc_vf_call<extended_instance_ops<half, false>>(
                    dim3(LANES), (__gm__ half*)a, (__gm__ half*)b, id_a, id_b, block, count_a, count_b, shift, mode,
                    (__gm__ uint64_t*)identity);
                break;
            case FP32:
                asc_vf_call<extended_instance_ops<float, false>>(
                    dim3(LANES), (__gm__ float*)a, (__gm__ float*)b, id_a, id_b, block, count_a, count_b, shift, mode,
                    (__gm__ uint64_t*)identity);
                break;
            case BF16:
                asc_vf_call<extended_instance_ops<bfloat16_t, false>>(
                    dim3(LANES), (__gm__ bfloat16_t*)a, (__gm__ bfloat16_t*)b, id_a, id_b, block, count_a, count_b,
                    shift, mode, (__gm__ uint64_t*)identity);
                break;
            default:
                break;
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}

extern "C" void simt_instance_extended_transfer(
    void* stream, void* a, void* b, uint64_t id_a, uint64_t id_b, uint32_t count_a, uint32_t count_b, uint32_t shift,
    int mode, int type, bool byte, uint32_t blocks, void* identity)
{
    simt_instance_extended_kernel<<<blocks, nullptr, stream>>>(
        (GM_ADDR)a, (GM_ADDR)b, id_a, id_b, count_a, count_b, shift, mode, type, byte, (GM_ADDR)identity);
}
