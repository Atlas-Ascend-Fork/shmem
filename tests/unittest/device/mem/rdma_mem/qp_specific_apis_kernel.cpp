/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "shmem.h"

constexpr uint64_t MESSAGE_SIZE = 64;
constexpr uint32_t MIN_QP_NUM = 2;
constexpr uint32_t AGGREGATE_OP_COUNT = 256;
constexpr uint64_t AGGREGATE_MESSAGE_SIZE = AGGREGATE_OP_COUNT * sizeof(uint32_t);
constexpr uint32_t AGGREGATE_UB_SIZE = 64 + AGGREGATE_OP_COUNT * 128;

namespace {
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
constexpr bool kQpSpecificBackendSupported = true;
#else
constexpr bool kQpSpecificBackendSupported = false;
#endif

__aicore__ inline bool is_active_qp_block(uint32_t qp_num) { return AscendC::GetBlockIdx() < qp_num; }

template <typename T>
__aicore__ inline void roce_qp_barrier_all(AscendC::LocalTensor<T> ub_local, uint32_t sync_id)
{
    AscendC::SyncAll<true>();
    if (AscendC::GetBlockIdx() == 0) {
        aclshmemx_roce_barrier_all(reinterpret_cast<__ubuf__ uint8_t*>(ub_local.GetPhyAddr()), sync_id);
    }
}

template <typename T>
__aicore__ inline uint32_t get_qp_num_or_zero()
{
    if constexpr (kQpSpecificBackendSupported) {
        return aclshmemi_qp_info_fetch()->qp_num;
    }
    return 0;
}
} // namespace

template <typename T>
__aicore__ inline void roce_qp_put_nbi_raw_impl(__gm__ T* gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM) {
        return;
    }
    if (!is_active_qp_block(qp_num)) {
        return;
    }

    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ub_local = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);

    int64_t my_pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    __gm__ T* src_addr = gva + my_pe * MESSAGE_SIZE / sizeof(T);

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe) {
            continue;
        }
        uint32_t qp_idx = static_cast<uint32_t>(peer % qp_num);
        if (qp_idx != AscendC::GetBlockIdx()) {
            continue;
        }
        __gm__ T* dst_addr = gva + my_pe * MESSAGE_SIZE / sizeof(T);
        aclshmemx_roce_qp_put_nbi(
            dst_addr, src_addr, reinterpret_cast<__ubuf__ T*>(ub_local.GetPhyAddr()), MESSAGE_SIZE / sizeof(T),
            static_cast<int>(peer), qp_idx, sync_id);
        aclshmemx_roce_qp_quiet(
            static_cast<uint32_t>(peer), qp_idx, reinterpret_cast<__ubuf__ T*>(ub_local.GetPhyAddr()), sync_id);
    }
    roce_qp_barrier_all(ub_local, sync_id);
#endif
}

template <typename T>
__aicore__ inline void roce_qp_put_nbi_raw_aggregate_impl(__gm__ T* gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM || !is_active_qp_block(qp_num)) {
        return;
    }

    const uint32_t qp_idx = AscendC::GetBlockIdx();
    const uint32_t sync_id = aclshmemi_get_state()->rdma_config.sync_id;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, AGGREGATE_UB_SIZE);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(buf.GetWithOffset<uint8_t>(AGGREGATE_UB_SIZE, 0).GetPhyAddr());
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    __gm__ T* src_addr = gva + my_pe * AGGREGATE_MESSAGE_SIZE / sizeof(T);
    constexpr uint32_t aggregate_elem_size = AGGREGATE_MESSAGE_SIZE / AGGREGATE_OP_COUNT / sizeof(T);

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe || static_cast<uint32_t>(peer % qp_num) != qp_idx) {
            continue;
        }
        __gm__ T* dst_addr = gva + my_pe * AGGREGATE_MESSAGE_SIZE / sizeof(T);
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t op = 0; op < AGGREGATE_OP_COUNT; ++op) {
            auto dst = dst_addr + op * aggregate_elem_size;
            auto src = src_addr + op * aggregate_elem_size;
            if (op + 1 == AGGREGATE_OP_COUNT) {
                aclshmemx_roce_qp_put_nbi(
                    dst, src, ub_ptr, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id, submit_action);
            } else {
                aclshmemx_roce_qp_put_nbi(
                    dst, src, ub_ptr, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id, defer_action);
            }
        }
        aclshmemx_roce_qp_quiet(static_cast<uint32_t>(peer), qp_idx, ub_ptr, sync_id);
    }
    roce_qp_barrier_all(buf.GetWithOffset<uint8_t>(AGGREGATE_UB_SIZE, 0), sync_id);
#endif
}

template <typename T>
__aicore__ inline void roce_qp_get_nbi_raw_impl(__gm__ T* gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM) {
        return;
    }
    if (!is_active_qp_block(qp_num)) {
        return;
    }

    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ub_local = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);

    int64_t my_pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe) {
            continue;
        }
        uint32_t qp_idx = static_cast<uint32_t>(peer % qp_num);
        if (qp_idx != AscendC::GetBlockIdx()) {
            continue;
        }
        __gm__ T* dst_addr = gva + peer * MESSAGE_SIZE / sizeof(T);
        __gm__ T* src_addr = gva + peer * MESSAGE_SIZE / sizeof(T);
        aclshmemx_roce_qp_get_nbi(
            dst_addr, src_addr, reinterpret_cast<__ubuf__ T*>(ub_local.GetPhyAddr()), MESSAGE_SIZE / sizeof(T),
            static_cast<int>(peer), qp_idx, sync_id);
        aclshmemx_roce_qp_quiet(
            static_cast<uint32_t>(peer), qp_idx, reinterpret_cast<__ubuf__ T*>(ub_local.GetPhyAddr()), sync_id);
    }
    roce_qp_barrier_all(ub_local, sync_id);
#endif
}

template <typename T>
__aicore__ inline void roce_qp_get_nbi_raw_aggregate_impl(__gm__ T* gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM || !is_active_qp_block(qp_num)) {
        return;
    }

    const uint32_t qp_idx = AscendC::GetBlockIdx();
    const uint32_t sync_id = aclshmemi_get_state()->rdma_config.sync_id;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, AGGREGATE_UB_SIZE);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(buf.GetWithOffset<uint8_t>(AGGREGATE_UB_SIZE, 0).GetPhyAddr());
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    constexpr uint32_t aggregate_elem_size = AGGREGATE_MESSAGE_SIZE / AGGREGATE_OP_COUNT / sizeof(T);

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe || static_cast<uint32_t>(peer % qp_num) != qp_idx) {
            continue;
        }
        __gm__ T* dst_addr = gva + peer * AGGREGATE_MESSAGE_SIZE / sizeof(T);
        __gm__ T* src_addr = gva + peer * AGGREGATE_MESSAGE_SIZE / sizeof(T);
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t op = 0; op < AGGREGATE_OP_COUNT; ++op) {
            auto dst = dst_addr + op * aggregate_elem_size;
            auto src = src_addr + op * aggregate_elem_size;
            if (op + 1 == AGGREGATE_OP_COUNT) {
                aclshmemx_roce_qp_get_nbi(
                    dst, src, ub_ptr, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id, submit_action);
            } else {
                aclshmemx_roce_qp_get_nbi(
                    dst, src, ub_ptr, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id, defer_action);
            }
        }
        aclshmemx_roce_qp_quiet(static_cast<uint32_t>(peer), qp_idx, ub_ptr, sync_id);
    }
    roce_qp_barrier_all(buf.GetWithOffset<uint8_t>(AGGREGATE_UB_SIZE, 0), sync_id);
#endif
}

template <typename T>
__aicore__ inline void roce_qp_put_nbi_tensor_impl(AscendC::GlobalTensor<T> gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM) {
        return;
    }
    if (!is_active_qp_block(qp_num)) {
        return;
    }

    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<T> ub_local = buf.GetWithOffset<T>(UB_ALIGN_SIZE_64 * 2 / sizeof(T), 0);

    int64_t my_pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    AscendC::GlobalTensor<T> src_tensor;
    src_tensor.SetGlobalBuffer(const_cast<__gm__ T*>(gva.GetPhyAddr()) + my_pe * MESSAGE_SIZE / sizeof(T));

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe) {
            continue;
        }
        uint32_t qp_idx = static_cast<uint32_t>(peer % qp_num);
        if (qp_idx != AscendC::GetBlockIdx()) {
            continue;
        }
        AscendC::GlobalTensor<T> dst_tensor;
        dst_tensor.SetGlobalBuffer(const_cast<__gm__ T*>(gva.GetPhyAddr()) + my_pe * MESSAGE_SIZE / sizeof(T));
        aclshmemx_roce_qp_put_nbi(
            dst_tensor, src_tensor, ub_local, MESSAGE_SIZE / sizeof(T), static_cast<int>(peer), qp_idx, sync_id);
        aclshmemx_roce_qp_quiet(static_cast<uint32_t>(peer), qp_idx, ub_local, sync_id);
    }
    roce_qp_barrier_all(ub_local, sync_id);
#endif
}

template <typename T>
__aicore__ inline void roce_qp_put_nbi_tensor_aggregate_impl(AscendC::GlobalTensor<T> gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM || !is_active_qp_block(qp_num)) {
        return;
    }

    const uint32_t qp_idx = AscendC::GetBlockIdx();
    const uint32_t sync_id = aclshmemi_get_state()->rdma_config.sync_id;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, AGGREGATE_UB_SIZE);
    auto ub_local = buf.GetWithOffset<T>(AGGREGATE_UB_SIZE / sizeof(T), 0);
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    constexpr uint32_t aggregate_elem_size = AGGREGATE_MESSAGE_SIZE / AGGREGATE_OP_COUNT / sizeof(T);
    AscendC::GlobalTensor<T> src_tensor;
    src_tensor.SetGlobalBuffer(const_cast<__gm__ T*>(gva.GetPhyAddr()) + my_pe * AGGREGATE_MESSAGE_SIZE / sizeof(T));

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe || static_cast<uint32_t>(peer % qp_num) != qp_idx) {
            continue;
        }
        AscendC::GlobalTensor<T> dst_tensor;
        dst_tensor.SetGlobalBuffer(
            const_cast<__gm__ T*>(gva.GetPhyAddr()) + my_pe * AGGREGATE_MESSAGE_SIZE / sizeof(T));
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t op = 0; op < AGGREGATE_OP_COUNT; ++op) {
            dst_tensor.SetGlobalBuffer(
                const_cast<__gm__ T*>(gva.GetPhyAddr()) + my_pe * AGGREGATE_MESSAGE_SIZE / sizeof(T) +
                op * aggregate_elem_size);
            src_tensor.SetGlobalBuffer(
                const_cast<__gm__ T*>(gva.GetPhyAddr()) + my_pe * AGGREGATE_MESSAGE_SIZE / sizeof(T) +
                op * aggregate_elem_size);
            if (op + 1 == AGGREGATE_OP_COUNT) {
                aclshmemx_roce_qp_put_nbi(
                    dst_tensor, src_tensor, ub_local, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id,
                    submit_action);
            } else {
                aclshmemx_roce_qp_put_nbi(
                    dst_tensor, src_tensor, ub_local, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id,
                    defer_action);
            }
        }
        aclshmemx_roce_qp_quiet(static_cast<uint32_t>(peer), qp_idx, ub_local, sync_id);
    }
    roce_qp_barrier_all(ub_local, sync_id);
#endif
}

template <typename T>
__aicore__ inline void roce_qp_get_nbi_tensor_impl(AscendC::GlobalTensor<T> gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM) {
        return;
    }
    if (!is_active_qp_block(qp_num)) {
        return;
    }

    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<T> ub_local = buf.GetWithOffset<T>(UB_ALIGN_SIZE_64 * 2 / sizeof(T), 0);

    int64_t my_pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe) {
            continue;
        }
        uint32_t qp_idx = static_cast<uint32_t>(peer % qp_num);
        if (qp_idx != AscendC::GetBlockIdx()) {
            continue;
        }
        AscendC::GlobalTensor<T> dst_tensor;
        dst_tensor.SetGlobalBuffer(const_cast<__gm__ T*>(gva.GetPhyAddr()) + peer * MESSAGE_SIZE / sizeof(T));
        AscendC::GlobalTensor<T> src_tensor;
        src_tensor.SetGlobalBuffer(const_cast<__gm__ T*>(gva.GetPhyAddr()) + peer * MESSAGE_SIZE / sizeof(T));
        aclshmemx_roce_qp_get_nbi(
            dst_tensor, src_tensor, ub_local, MESSAGE_SIZE / sizeof(T), static_cast<int>(peer), qp_idx, sync_id);
        aclshmemx_roce_qp_quiet(static_cast<uint32_t>(peer), qp_idx, ub_local, sync_id);
    }
    roce_qp_barrier_all(ub_local, sync_id);
#endif
}

template <typename T>
__aicore__ inline void roce_qp_get_nbi_tensor_aggregate_impl(AscendC::GlobalTensor<T> gva)
{
#if (defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)) && \
    (defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__))
    if (g_coreType != AscendC::AIV || AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    const uint32_t qp_num = get_qp_num_or_zero<T>();
    if (qp_num < MIN_QP_NUM || !is_active_qp_block(qp_num)) {
        return;
    }

    const uint32_t qp_idx = AscendC::GetBlockIdx();
    const uint32_t sync_id = aclshmemi_get_state()->rdma_config.sync_id;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, AGGREGATE_UB_SIZE);
    auto ub_local = buf.GetWithOffset<T>(AGGREGATE_UB_SIZE / sizeof(T), 0);
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    constexpr uint32_t aggregate_elem_size = AGGREGATE_MESSAGE_SIZE / AGGREGATE_OP_COUNT / sizeof(T);

    for (int64_t peer = 0; peer < pe_size; ++peer) {
        if (peer == my_pe || static_cast<uint32_t>(peer % qp_num) != qp_idx) {
            continue;
        }
        AscendC::GlobalTensor<T> dst_tensor;
        dst_tensor.SetGlobalBuffer(const_cast<__gm__ T*>(gva.GetPhyAddr()) + peer * AGGREGATE_MESSAGE_SIZE / sizeof(T));
        AscendC::GlobalTensor<T> src_tensor;
        src_tensor.SetGlobalBuffer(const_cast<__gm__ T*>(gva.GetPhyAddr()) + peer * AGGREGATE_MESSAGE_SIZE / sizeof(T));
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t op = 0; op < AGGREGATE_OP_COUNT; ++op) {
            dst_tensor.SetGlobalBuffer(
                const_cast<__gm__ T*>(gva.GetPhyAddr()) + peer * AGGREGATE_MESSAGE_SIZE / sizeof(T) +
                op * aggregate_elem_size);
            src_tensor.SetGlobalBuffer(
                const_cast<__gm__ T*>(gva.GetPhyAddr()) + peer * AGGREGATE_MESSAGE_SIZE / sizeof(T) +
                op * aggregate_elem_size);
            if (op + 1 == AGGREGATE_OP_COUNT) {
                aclshmemx_roce_qp_get_nbi(
                    dst_tensor, src_tensor, ub_local, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id,
                    submit_action);
            } else {
                aclshmemx_roce_qp_get_nbi(
                    dst_tensor, src_tensor, ub_local, aggregate_elem_size, static_cast<int>(peer), qp_idx, sync_id,
                    defer_action);
            }
        }
        aclshmemx_roce_qp_quiet(static_cast<uint32_t>(peer), qp_idx, ub_local, sync_id);
    }
    roce_qp_barrier_all(ub_local, sync_id);
#endif
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_put_nbi_raw_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    roce_qp_put_nbi_raw_impl<uint32_t>(reinterpret_cast<__gm__ uint32_t*>(gva));
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_get_nbi_raw_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    roce_qp_get_nbi_raw_impl<uint32_t>(reinterpret_cast<__gm__ uint32_t*>(gva));
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_put_nbi_tensor_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::GlobalTensor<uint32_t> gva_tensor;
    gva_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gva));
    roce_qp_put_nbi_tensor_impl<uint32_t>(gva_tensor);
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_get_nbi_tensor_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::GlobalTensor<uint32_t> gva_tensor;
    gva_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gva));
    roce_qp_get_nbi_tensor_impl<uint32_t>(gva_tensor);
}

void test_rdma_roce_qp_put_nbi_raw_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_put_nbi_raw_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}

void test_rdma_roce_qp_get_nbi_raw_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_get_nbi_raw_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}

void test_rdma_roce_qp_put_nbi_tensor_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_put_nbi_tensor_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}

void test_rdma_roce_qp_get_nbi_tensor_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_get_nbi_tensor_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_put_nbi_raw_aggregate_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    roce_qp_put_nbi_raw_aggregate_impl<uint32_t>(reinterpret_cast<__gm__ uint32_t*>(gva));
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_get_nbi_raw_aggregate_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    roce_qp_get_nbi_raw_aggregate_impl<uint32_t>(reinterpret_cast<__gm__ uint32_t*>(gva));
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_put_nbi_tensor_aggregate_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::GlobalTensor<uint32_t> gva_tensor;
    gva_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gva));
    roce_qp_put_nbi_tensor_aggregate_impl<uint32_t>(gva_tensor);
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_roce_qp_get_nbi_tensor_aggregate_kernel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::GlobalTensor<uint32_t> gva_tensor;
    gva_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gva));
    roce_qp_get_nbi_tensor_aggregate_impl<uint32_t>(gva_tensor);
}

void test_rdma_roce_qp_put_nbi_raw_aggregate_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_put_nbi_raw_aggregate_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}

void test_rdma_roce_qp_get_nbi_raw_aggregate_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_get_nbi_raw_aggregate_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}

void test_rdma_roce_qp_put_nbi_tensor_aggregate_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_put_nbi_tensor_aggregate_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}

void test_rdma_roce_qp_get_nbi_tensor_aggregate_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    test_rdma_roce_qp_get_nbi_tensor_aggregate_kernel<<<block_dim, nullptr, stream>>>(gva, config);
}
