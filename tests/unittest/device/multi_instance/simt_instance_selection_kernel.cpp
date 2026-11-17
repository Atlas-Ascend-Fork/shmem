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
#include "unittest/simt_instance_selection_test.h"

using namespace simt_instance_selection_test;

// Check invalid IDs without intentionally trapping or dereferencing their metadata.
__simt_vf__ __launch_bounds__(1) inline void instance_boundary_ops(__gm__ uint64_t* result)
{
    const uint64_t ids[] = {0, 254, 255, 256, UINT64_MAX};
    for (uint32_t i = 0; i < 5; ++i) {
        result[i] = simt::aclshmemi_is_valid_hbm_instance_id(ids[i]);
    }
}

__global__ __vector__ void simt_instance_boundary_kernel(__gm__ uint64_t* result)
{
    asc_vf_call<instance_boundary_ops>(dim3(1), result);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}

extern "C" void simt_instance_boundary(void* stream, void* result)
{
    simt_instance_boundary_kernel<<<1, nullptr, stream>>>(static_cast<uint64_t*>(result));
}

// Original free-function types must remain exact, including taking addresses.
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__simt_callee__ int legacy_query_signature();
using pe_query_t = decltype(&legacy_query_signature);
static_assert(std::is_same<decltype(&simt::aclshmem_my_pe), pe_query_t>::value, "legacy my_pe type");
static_assert(std::is_same<decltype(&simt::aclshmem_n_pes), pe_query_t>::value, "legacy n_pes type");
__simt_callee__ __gm__ void* legacy_ptr_signature(__gm__ void*, int);
using ptr_query_t = decltype(&legacy_ptr_signature);
static_assert(std::is_same<decltype(&simt::aclshmem_ptr), ptr_query_t>::value, "legacy ptr type");
__simt_callee__ void legacy_gm_put_signature(__gm__ int32_t*, __gm__ int32_t*, size_t, int32_t);
using put32_t = decltype(&legacy_gm_put_signature);
__simt_callee__ void legacy_ub_put_signature(__gm__ int32_t*, __ubuf__ int32_t*, size_t, int32_t);
using ub_put32_t = decltype(&legacy_ub_put_signature);
static_assert(
    std::is_same<decltype(static_cast<put32_t>(&simt::aclshmemx_int32_put_nbi_block)), put32_t>::value,
    "legacy GM put type");
static_assert(
    std::is_same<decltype(static_cast<ub_put32_t>(&simt::aclshmemx_int32_put_nbi_block)), ub_put32_t>::value,
    "legacy UB put type");
__simt_callee__ int32_t legacy_cas_signature(__gm__ int32_t*, int32_t, int32_t, int32_t);
using cas32_t = decltype(&legacy_cas_signature);
static_assert(std::is_same<decltype(&simt::aclshmem_int32_atomic_compare_swap), cas32_t>::value, "legacy CAS type");
#endif

// One AIV per stream/instance. Its 128 lanes own disjoint words or disjoint
// warp slices. Both live heaps and the output/timing buffers are disjoint.
// UB: one compiler-declared 16384-byte SIMT shared region per VF block; barriers
// publish it and protect reuse. No MTE writer or cross-AIV sharing.
// Completion: VF -> V_S -> stream completion -> host rank rendezvous -> D2H.
__simt_vf__ __launch_bounds__(LANES) inline void instance_selection_ops(
    __gm__ int32_t* heap, uint32_t count, int peer, uint64_t id, __gm__ uint64_t* identity, __gm__ int32_t* observed,
    int mode, uint32_t repetitions)
{
    __ubuf__ int32_t ub[SLOT];
    simt::aclshmemx_instance_ctx_set(id);
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = lane / 32;
    const uint32_t warp_count = count / 4 + (warp < count % 4);
    const uint32_t warp_begin = warp * (count / 4) + (warp < count % 4 ? warp : count % 4);
    for (uint32_t rep = 0; rep < repetitions; ++rep) {
        if ((mode >= UB_PUT && mode < UB_GET) || mode == BYTE_UB_PUT) {
            for (uint32_t i = lane; i < count; i += LANES)
                ub[i] = heap[i];
        }
        asc_syncthreads();
#define RUN_VARIANTS(OP, DST, SRC, VARIANT)                                                                  \
    switch (VARIANT) {                                                                                       \
        case THREAD:                                                                                         \
            for (uint32_t i = lane; i < count; i += LANES)                                                   \
                simt::aclshmem_int32_##OP((DST) + i, (SRC) + i, 1, peer);                                    \
            break;                                                                                           \
        case WARP:                                                                                           \
            simt::aclshmemx_int32_##OP##_warp((DST) + warp_begin, (SRC) + warp_begin, warp_count, peer);     \
            break;                                                                                           \
        case BLOCK:                                                                                          \
            simt::aclshmemx_int32_##OP##_block(DST, SRC, count, peer);                                       \
            break;                                                                                           \
        case THREAD_NBI:                                                                                     \
            for (uint32_t i = lane; i < count; i += LANES)                                                   \
                simt::aclshmem_int32_##OP##_nbi((DST) + i, (SRC) + i, 1, peer);                              \
            break;                                                                                           \
        case WARP_NBI:                                                                                       \
            simt::aclshmemx_int32_##OP##_nbi_warp((DST) + warp_begin, (SRC) + warp_begin, warp_count, peer); \
            break;                                                                                           \
        case BLOCK_NBI:                                                                                      \
            simt::aclshmemx_int32_##OP##_nbi_block(DST, SRC, count, peer);                                   \
            break;                                                                                           \
        default:                                                                                             \
            break;                                                                                           \
    }
        if (mode < GM_GET) {
            RUN_VARIANTS(put, heap + SLOT, heap, mode);
        } else if (mode < UB_PUT) {
            RUN_VARIANTS(get, heap + SLOT, heap, mode - GM_GET);
        } else if (mode < UB_GET) {
            RUN_VARIANTS(put, heap + SLOT, ub, mode - UB_PUT);
        } else if (mode < BYTE_GM_PUT) {
            RUN_VARIANTS(get, ub, heap, mode - UB_GET);
        } else if (mode == BYTE_GM_PUT)
            simt::aclshmemx_putmem_nbi_block(heap + SLOT, heap, count * sizeof(int32_t), peer);
        else if (mode == BYTE_GM_GET)
            simt::aclshmemx_getmem_block(heap + SLOT, heap, count * sizeof(int32_t), peer);
        else if (mode == BYTE_UB_PUT)
            simt::aclshmemx_putmem_block(heap + SLOT, ub, count * sizeof(int32_t), peer);
        else if (mode == BYTE_UB_GET)
            simt::aclshmemx_getmem_nbi_block(ub, heap, count * sizeof(int32_t), peer);
        else if (mode == SCALAR_P) {
            for (uint32_t i = lane; i < count; i += LANES)
                simt::aclshmem_int32_p(heap + SLOT + i, heap[i], peer);
        } else if (mode == SCALAR_G) {
            for (uint32_t i = lane; i < count; i += LANES)
                heap[SLOT + i] = simt::aclshmem_int32_g(heap + i, peer);
        } else if (mode == CAS) {
            for (uint32_t i = lane; i < count; i += LANES) {
                observed[i] = simt::aclshmem_int32_atomic_compare_swap(heap + SLOT + i, SENTINEL, heap[i], peer);
                observed[SLOT + i] =
                    simt::aclshmem_int32_atomic_compare_swap(heap + SLOT + i, SENTINEL, int32_t(-1), peer);
            }
        } else if (mode == LEGACY) {
            simt::aclshmemx_int32_put_nbi_block(heap + SLOT, heap, count, peer);
        }
#undef RUN_VARIANTS
        asc_syncthreads();
        if ((mode >= UB_GET && mode < BYTE_GM_PUT) || mode == BYTE_UB_GET) {
            for (uint32_t i = lane; i < count; i += LANES)
                heap[SLOT + i] = ub[i];
        }
        asc_syncthreads();
    }
    identity[ID_FIELDS * lane] = simt::aclshmemx_instance_ctx_get() != 0;
    identity[ID_FIELDS * lane + 1] = simt::aclshmem_my_pe();
    identity[ID_FIELDS * lane + 2] = simt::aclshmem_n_pes();
    identity[ID_FIELDS * lane + 3] = simt::aclshmem_team_my_pe(ACLSHMEM_TEAM_WORLD);
    identity[ID_FIELDS * lane + 4] = simt::aclshmem_team_n_pes(ACLSHMEM_TEAM_WORLD);
    identity[ID_FIELDS * lane + 5] = simt::aclshmem_team_translate_pe(ACLSHMEM_TEAM_WORLD, peer, ACLSHMEM_TEAM_WORLD);
    identity[ID_FIELDS * lane + 6] = simt::aclshmem_team_pe_mapping(ACLSHMEM_TEAM_WORLD, peer);
    identity[ID_FIELDS * lane + 7] = simt::aclshmem_team_my_pe(ACLSHMEM_TEAM_INVALID) == -1;
    identity[ID_FIELDS * lane + 8] = simt::aclshmem_team_n_pes(ACLSHMEM_TEAM_INVALID) == -1;
    identity[ID_FIELDS * lane + 9] = simt::aclshmem_ptr(heap, simt::aclshmem_my_pe()) == heap;
    bool rejected = true;
    const int invalid_teams[] = {-2, ACLSHMEM_TEAM_INVALID, ACLSHMEM_MAX_TEAMS - 1, ACLSHMEM_MAX_TEAMS};
    for (int team : invalid_teams) {
        rejected &= simt::aclshmem_team_my_pe(team) == -1;
        rejected &= simt::aclshmem_team_n_pes(team) == -1;
        rejected &= simt::aclshmem_team_translate_pe(team, 0, ACLSHMEM_TEAM_WORLD) == -1;
        rejected &= simt::aclshmem_team_translate_pe(ACLSHMEM_TEAM_WORLD, 0, team) == -1;
        rejected &= simt::aclshmem_team_pe_mapping(team, 0) == -1;
    }
    const int invalid_pes[] = {-1, simt::aclshmem_n_pes()};
    for (int pe : invalid_pes) {
        rejected &= simt::aclshmem_team_translate_pe(ACLSHMEM_TEAM_WORLD, pe, ACLSHMEM_TEAM_WORLD) == -1;
        rejected &= simt::aclshmem_team_pe_mapping(ACLSHMEM_TEAM_WORLD, pe) == -1;
    }
    identity[ID_FIELDS * lane + 10] = rejected;
}

__global__ __vector__ void simt_instance_selection_kernel(
    __gm__ int32_t* heap, uint32_t count, int peer, uint64_t id, __gm__ uint64_t* identity, __gm__ int32_t* observed,
    __gm__ uint64_t* timing, int mode, uint32_t repetitions)
{
    const uint64_t start = AscendC::GetSystemCycle();
    asc_vf_call<instance_selection_ops>(dim3(LANES), heap, count, peer, id, identity, observed, mode, repetitions);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    timing[0] = start;
    timing[1] = AscendC::GetSystemCycle();
}

extern "C" void simt_instance_selection_transfer(
    void* stream, void* heap, uint32_t count, int peer, uint64_t id, void* identity, void* observed, void* timing,
    int mode, uint32_t repetitions)
{
    simt_instance_selection_kernel<<<1, nullptr, stream>>>(
        (int32_t*)heap, count, peer, id, (uint64_t*)identity, (int32_t*)observed, (uint64_t*)timing, mode, repetitions);
}

// All block threads enter every RMA call. Each warp owns a separate 256-word
// source/receive slice. 3*256 words fit inside each existing 4096-word region.
// UB stores -> block barrier -> cooperative RMA -> block barrier -> local copy.
__simt_vf__ __launch_bounds__(128) inline void warp_tail_ops(
    __gm__ int32_t* heap, uint32_t count, int peer, uint64_t id, int mode)
{
    __ubuf__ int32_t ub[1024];
    simt::aclshmemx_instance_ctx_set(id);
    const uint32_t lane = threadIdx.x;
    const uint32_t offset = (lane / 32) * 256;
    for (uint32_t i = lane; i < 1024; i += blockDim.x)
        ub[i] = heap[i];
    asc_syncthreads();
    switch (mode) {
        case GM_PUT + WARP:
            simt::aclshmemx_int32_put_warp(heap + SLOT + offset, heap + offset, count, peer);
            break;
        case GM_GET + WARP:
            simt::aclshmemx_int32_get_warp(heap + SLOT + offset, heap + offset, count, peer);
            break;
        case UB_PUT + WARP:
            simt::aclshmemx_int32_put_warp(heap + SLOT + offset, ub + offset, count, peer);
            break;
        case UB_GET + WARP:
            simt::aclshmemx_int32_get_warp(ub + offset, heap + offset, count, peer);
            break;
        case GM_PUT + WARP_NBI:
            simt::aclshmemx_int32_put_nbi_warp(heap + SLOT + offset, heap + offset, count, peer);
            break;
        case GM_GET + WARP_NBI:
            simt::aclshmemx_int32_get_nbi_warp(heap + SLOT + offset, heap + offset, count, peer);
            break;
        case UB_PUT + WARP_NBI:
            simt::aclshmemx_int32_put_nbi_warp(heap + SLOT + offset, ub + offset, count, peer);
            break;
        case UB_GET + WARP_NBI:
            simt::aclshmemx_int32_get_nbi_warp(ub + offset, heap + offset, count, peer);
            break;
        case BYTE_GM_PUT:
            simt::aclshmemx_putmem_warp(heap + SLOT + offset, heap + offset, count * sizeof(int32_t), peer);
            break;
        case BYTE_GM_GET:
            simt::aclshmemx_getmem_warp(heap + SLOT + offset, heap + offset, count * sizeof(int32_t), peer);
            break;
        case BYTE_UB_PUT:
            simt::aclshmemx_putmem_nbi_warp(heap + SLOT + offset, ub + offset, count * sizeof(int32_t), peer);
            break;
        case BYTE_UB_GET:
            simt::aclshmemx_getmem_nbi_warp(ub + offset, heap + offset, count * sizeof(int32_t), peer);
            break;
        case LEGACY:
            simt::aclshmemx_int32_put_nbi_warp(heap + SLOT + offset, heap + offset, count, peer);
            break;
        default:
            break;
    }
    asc_syncthreads();
    if (mode == UB_GET + WARP || mode == UB_GET + WARP_NBI || mode == BYTE_UB_GET) {
        const uint32_t start = (lane / 32) * 32;
        const uint32_t threads = blockDim.x - start < 32 ? blockDim.x - start : 32;
        for (uint32_t i = lane % 32; i < count; i += threads)
            heap[SLOT + offset + i] = ub[offset + i];
    }
    asc_syncthreads();
}

__global__ __vector__ void simt_warp_tail_kernel(
    __gm__ int32_t* heap, uint32_t count, int peer, uint64_t id, uint32_t lanes, int mode)
{
    asc_vf_call<warp_tail_ops>(dim3(lanes), heap, count, peer, id, mode);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}
extern "C" void simt_warp_tail_transfer(
    void* stream, void* heap, uint32_t count, int peer, uint64_t id, uint32_t lanes, int mode)
{
    simt_warp_tail_kernel<<<1, nullptr, stream>>>((int32_t*)heap, count, peer, id, lanes, mode);
}
