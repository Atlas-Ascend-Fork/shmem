/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _UDMAPERF_KERNEL_UDMA_PERFTEST_
#define _UDMAPERF_KERNEL_UDMA_PERFTEST_

#include <cstring>
#include "kernel_operator.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"
#include "perftest_common_types.h"

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_put_once(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elements, int peer, uint32_t qp_idx)
{
    if constexpr (USE_QP) {
        aclshmemx_udma_qp_put_nbi<T>(dst, src, buf, elements, peer, qp_idx, 0);
    } else {
        aclshmemx_udma_put_nbi<T>(dst, src, buf, elements, peer);
    }
}

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_get_once(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elements, int peer, uint32_t qp_idx)
{
    if constexpr (USE_QP) {
        aclshmemx_udma_qp_get_nbi<T>(dst, src, buf, elements, peer, qp_idx, 0);
    } else {
        aclshmemx_udma_get_nbi<T>(dst, src, buf, elements, peer);
    }
}

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_put_aggregate(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elements, int peer, uint32_t qp_idx, int operation_count)
{
    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);
    if constexpr (USE_QP) {
        for (int i = 0; i < operation_count - 1; ++i) {
            aclshmemx_udma_qp_put_nbi<T>(dst, src, buf, elements, peer, qp_idx, 0, defer_action);
        }
        aclshmemx_udma_qp_put_nbi<T>(dst, src, buf, elements, peer, qp_idx, 0, submit_action);
    } else {
        for (int i = 0; i < operation_count - 1; ++i) {
            aclshmemx_udma_put_nbi<T>(dst, src, buf, elements, peer, 0, defer_action);
        }
        aclshmemx_udma_put_nbi<T>(dst, src, buf, elements, peer, 0, submit_action);
    }
}

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_get_aggregate(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elements, int peer, uint32_t qp_idx, int operation_count)
{
    aclshmemx_submit_state_t state{};
    aclshmemx_defer_t defer_action(state);
    aclshmemx_submit_t submit_action(state);
    if constexpr (USE_QP) {
        for (int i = 0; i < operation_count - 1; ++i) {
            aclshmemx_udma_qp_get_nbi<T>(dst, src, buf, elements, peer, qp_idx, 0, defer_action);
        }
        aclshmemx_udma_qp_get_nbi<T>(dst, src, buf, elements, peer, qp_idx, 0, submit_action);
    } else {
        for (int i = 0; i < operation_count - 1; ++i) {
            aclshmemx_udma_get_nbi<T>(dst, src, buf, elements, peer, 0, defer_action);
        }
        aclshmemx_udma_get_nbi<T>(dst, src, buf, elements, peer, 0, submit_action);
    }
}

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_put_group(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elements, int peer, uint32_t qp_idx, int operation_count,
    bool aggregate)
{
    if (!aggregate || operation_count <= 1) {
        for (int i = 0; i < operation_count; ++i) {
            udma_perf_put_once<T, USE_QP>(dst, src, buf, elements, peer, qp_idx);
        }
        return;
    }
    udma_perf_put_aggregate<T, USE_QP>(dst, src, buf, elements, peer, qp_idx, operation_count);
}

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_get_group(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elements, int peer, uint32_t qp_idx, int operation_count,
    bool aggregate)
{
    if (!aggregate || operation_count <= 1) {
        for (int i = 0; i < operation_count; ++i) {
            udma_perf_get_once<T, USE_QP>(dst, src, buf, elements, peer, qp_idx);
        }
        return;
    }
    udma_perf_get_aggregate<T, USE_QP>(dst, src, buf, elements, peer, qp_idx, operation_count);
}

template <bool USE_QP>
__aicore__ inline void udma_perf_quiet(int peer, uint32_t qp_idx)
{
    if constexpr (USE_QP) {
        aclshmemx_udma_qp_quiet(peer, qp_idx);
    } else {
        aclshmemx_udma_quiet(peer);
    }
}

// 时延模式接口形态槽位，与 host 侧 CSV PlainLat/QpLat/DeferLat/SubmitLat 列对应
constexpr int UDMA_PERF_IFACE_PLAIN = 0;
constexpr int UDMA_PERF_IFACE_QP = 1;
constexpr int UDMA_PERF_IFACE_DEFER = 2;
constexpr int UDMA_PERF_IFACE_SUBMIT = 3;
constexpr int UDMA_PERF_IFACE_COUNT = 4;

// 按接口形态提交 count 个操作（warmup 用）：plain/QP/submit 逐次独立提交，defer 一次聚合提交
template <typename T>
__aicore__ inline void udma_perf_lat_submit_batch(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, int iface, uint32_t count)
{
    if (iface == UDMA_PERF_IFACE_DEFER) {
        if (count > 1) {
            aclshmemx_submit_state_t submit_state{};
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (uint32_t i = 0; i < count - 1; ++i) {
                aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe, 0, defer_action);
            }
            aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe, 0, submit_action);
        } else {
            aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe);
        }
        return;
    }
    if (iface == UDMA_PERF_IFACE_SUBMIT) {
        for (uint32_t op = 0; op < count; ++op) {
            aclshmemx_submit_state_t submit_state{};
            aclshmemx_submit_t submit_action(submit_state);
            aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe, 0, submit_action);
        }
        return;
    }
    if (iface == UDMA_PERF_IFACE_QP) {
        for (uint32_t op = 0; op < count; ++op) {
            aclshmemx_udma_qp_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe, 0, 0);
        }
        return;
    }
    for (uint32_t op = 0; op < count; ++op) {
        aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe);
    }
}

// 按接口形态选择 quiet：QP 形态走 QP0 指定 quiet；其余（含聚合 defer/submit，均使用普通重载）走普通 quiet，
// 避免 relay 构建下 qp_quiet 不支持导致 kernel_abort
__aicore__ inline void udma_perf_lat_quiet(int peer_pe, int iface)
{
    if (iface == UDMA_PERF_IFACE_QP) {
        aclshmemx_udma_qp_quiet(peer_pe, 0);
    } else {
        aclshmemx_udma_quiet(peer_pe);
    }
}

// defer 接口时延：窗口内 N 次 defer 连续整体计时后除以 N 摊薄到单次调用，最后 1 次 submit 补齐并提交
// 聚合组（不计时），窗口间 quiet 不计时；返回摊薄到单次 defer 的 cycles
template <typename T>
__aicore__ inline int64_t udma_perf_lat_time_defer(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, int loop_test, int window)
{
    int64_t total_cycles = 0;
    int64_t total_ops = 0;
    const int full_groups = loop_test / window;
    const int remainder = loop_test - full_groups * window;
    for (int g = 0; g < full_groups + (remainder > 0 ? 1 : 0); ++g) {
        const int group = g < full_groups ? window : remainder;
        if (group <= 0) {
            break;
        }
        aclshmemx_submit_state_t submit_state{};
        aclshmemx_defer_t defer_action(submit_state);
        aclshmemx_submit_t submit_action(submit_state);
        AscendC::PipeBarrier<PIPE_ALL>();
        const int64_t start = AscendC::GetSystemCycle();
        for (int op = 0; op < group; ++op) {
            aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe, 0, defer_action);
        }
        total_cycles += AscendC::GetSystemCycle() - start;
        total_ops += group;
        aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe, 0, submit_action);
        aclshmemx_udma_quiet(peer_pe);
    }
    return total_ops > 0 ? total_cycles / total_ops : -1;
}

// submit 接口时延：连续调用 loop_test 次单操作 submit（逐次独立 submit state）后结束计时，再除以
// loop_test；quiet 不计时且只在整段计时结束后执行
template <typename T>
__aicore__ inline int64_t udma_perf_lat_time_submit(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, int loop_test)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t start = AscendC::GetSystemCycle();
    for (int op = 0; op < loop_test; ++op) {
        aclshmemx_submit_state_t submit_state{};
        aclshmemx_submit_t submit_action(submit_state);
        aclshmemx_udma_put_nbi<T>(dst, src, ub_ptr, elements, peer_pe, 0, submit_action);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t total_cycles = AscendC::GetSystemCycle() - start;
    aclshmemx_udma_quiet(peer_pe);
    return loop_test > 0 ? total_cycles / loop_test : -1;
}

// lat 模式主逻辑：遍历全部接口形态（plain/QP/defer/submit），各自 warmup + 计时，
// 单次调用摊薄 cycles 写入 timing_out[iface]；-1 表示窗口不满足聚合条件
template <typename T>
__aicore__ inline void udma_perf_lat_run_ifaces(
    __gm__ T* dst_gm, __gm__ T* src_gm, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, int loop_test,
    int batch_size, int warmup, int ub_size_kb, int32_t frame_id, __gm__ int64_t* timing_out)
{
    // UB staging 内可同时暂存的 WQE 数（每个 64B）；defer 窗口 N + 1 条 submit 需落在限额内
    const int ub_wqe_limit = ub_size_kb > 0 ? ub_size_kb * 1024 / 64 : 1;
    int agg_window = batch_size > 1 ? batch_size : 2;
    if (agg_window > ub_wqe_limit - 1) {
        agg_window = ub_wqe_limit - 1;
    }
    const int defer_warmup = warmup > ub_wqe_limit ? ub_wqe_limit : warmup;
    static const int kIfaceOrder[UDMA_PERF_IFACE_COUNT] = {
        UDMA_PERF_IFACE_PLAIN, UDMA_PERF_IFACE_QP, UDMA_PERF_IFACE_DEFER, UDMA_PERF_IFACE_SUBMIT};
    for (int order = 0; order < UDMA_PERF_IFACE_COUNT; ++order) {
        const int iface = kIfaceOrder[order];
#if defined(ACLSHMEM_RELAY_SUPPORT)
        // relay 构建不支持 QP 专有接口，跳过 QP 槽并输出 N/A（-1）
        if (iface == UDMA_PERF_IFACE_QP) {
            if (timing_out != nullptr) {
                timing_out[iface] = -1;
            }
            continue;
        }
#endif
        const uint32_t warmup_count = static_cast<uint32_t>(iface == UDMA_PERF_IFACE_DEFER ? defer_warmup : warmup);
        udma_perf_lat_submit_batch<T>(dst_gm, src_gm, ub_ptr, elements, peer_pe, iface, warmup_count);
        udma_perf_lat_quiet(peer_pe, iface);

        int64_t per_op_cycles = -1;
        const int prof_frame = frame_id * UDMA_PERF_IFACE_COUNT + iface;
        if (iface == UDMA_PERF_IFACE_DEFER) {
            SHMEMI_PROF_START(prof_frame);
            per_op_cycles = agg_window >= 2 ? udma_perf_lat_time_defer<T>(
                                                  dst_gm, src_gm, ub_ptr, elements, peer_pe, loop_test, agg_window) :
                                              -1;
            SHMEMI_PROF_END(prof_frame);
        } else if (iface == UDMA_PERF_IFACE_SUBMIT) {
            SHMEMI_PROF_START(prof_frame);
            per_op_cycles = udma_perf_lat_time_submit<T>(dst_gm, src_gm, ub_ptr, elements, peer_pe, loop_test);
            SHMEMI_PROF_END(prof_frame);
        } else {
            SHMEMI_PROF_START(prof_frame);
            AscendC::PipeBarrier<PIPE_ALL>();
            const int64_t pass_start = AscendC::GetSystemCycle();
            udma_perf_lat_submit_batch<T>(
                dst_gm, src_gm, ub_ptr, elements, peer_pe, iface, static_cast<uint32_t>(loop_test));
            AscendC::PipeBarrier<PIPE_ALL>();
            const int64_t total_cycles = AscendC::GetSystemCycle() - pass_start;
            udma_perf_lat_quiet(peer_pe, iface);
            SHMEMI_PROF_END(prof_frame);
            per_op_cycles = loop_test > 0 ? total_cycles / loop_test : 0;
        }
        if (timing_out != nullptr) {
            timing_out[iface] = per_op_cycles;
        }
    }
    if (timing_out != nullptr) {
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(int64_t) * UDMA_PERF_IFACE_COUNT);
    }
}

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_test_put_impl(
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::udma_mode_t test_mode, int ub_size_kb,
    int64_t prof_pe_val, int loop_count, int metric, int batch, GM_ADDR timing_out_gva)
{
    const int64_t pe = aclshmem_my_pe();
    const int peer_pe = (pe + 1) % aclshmem_n_pes();
    const uint32_t qp_idx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool is_unilateral = (test_mode == perftest::TEST_MODE_UDMA_PUT);
    if (is_unilateral && pe != prof_pe_val) {
        aclshmemx_sync_vec_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_kb * 1024);
    AscendC::LocalTensor<uint8_t> ub_local = buf.GetWithOffset<uint8_t>(ub_size_kb * 1024, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ub_local.GetPhyAddr());
    const uint64_t qp_offset = static_cast<uint64_t>(qp_idx) * static_cast<uint64_t>(elements);
    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva) + qp_offset;
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva) + qp_offset;
    const int warmup = perftest::PERFTEST_WARMUP_ITERS;
    const int loop_test = loop_count;
    const int quiet_batch = (batch <= 0 || batch > loop_test) ? loop_test : batch;
    const bool aggregate = batch > 1;
    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        udma_perf_lat_run_ifaces<T>(
            dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, loop_test, batch, warmup, ub_size_kb,
            frame_id, reinterpret_cast<__gm__ int64_t*>(timing_out_gva));
        aclshmemx_sync_vec_all();
        return;
    }

    for (int i = 0; i < warmup; ++i) {
        udma_perf_put_once<T, USE_QP>(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_idx);
    }
    udma_perf_quiet<USE_QP>(peer_pe, qp_idx);
    SHMEMI_PROF_START(frame_id);
    const int full_groups = loop_test / quiet_batch;
    const int remainder = loop_test - full_groups * quiet_batch;
    for (int g = 0; g < full_groups; ++g) {
        udma_perf_put_group<T, USE_QP>(
            dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_idx, quiet_batch, aggregate);
        udma_perf_quiet<USE_QP>(peer_pe, qp_idx);
    }
    if (remainder > 0) {
        udma_perf_put_group<T, USE_QP>(
            dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_idx, remainder, aggregate);
        udma_perf_quiet<USE_QP>(peer_pe, qp_idx);
    }
    SHMEMI_PROF_END(frame_id);
    aclshmemx_sync_vec_all();
}

template <typename T, bool USE_QP>
__aicore__ inline void udma_perf_test_get_impl(
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::udma_mode_t test_mode, int ub_size_kb,
    int64_t prof_pe_val, int loop_count, int batch)
{
    const int64_t pe = aclshmem_my_pe();
    const int peer_pe = (pe + 1) % aclshmem_n_pes();
    const uint32_t qp_idx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool is_unilateral = (test_mode == perftest::TEST_MODE_UDMA_GET);
    if (is_unilateral && pe != prof_pe_val) {
        aclshmemx_sync_vec_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_kb * 1024);
    AscendC::LocalTensor<uint8_t> ub_local = buf.GetWithOffset<uint8_t>(ub_size_kb * 1024, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ub_local.GetPhyAddr());
    const uint64_t qp_offset = static_cast<uint64_t>(qp_idx) * static_cast<uint64_t>(elements);
    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva) + qp_offset;
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva) + qp_offset;
    const int warmup = perftest::PERFTEST_WARMUP_ITERS;
    const int loop_test = loop_count;
    const int quiet_batch = (batch <= 0 || batch > loop_test) ? loop_test : batch;
    const bool aggregate = batch > 1;
    AscendC::PipeBarrier<PIPE_ALL>();

    for (int i = 0; i < warmup; ++i) {
        udma_perf_get_once<T, USE_QP>(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_idx);
    }
    udma_perf_quiet<USE_QP>(peer_pe, qp_idx);
    SHMEMI_PROF_START(frame_id);
    const int full_groups = loop_test / quiet_batch;
    const int remainder = loop_test - full_groups * quiet_batch;
    for (int g = 0; g < full_groups; ++g) {
        udma_perf_get_group<T, USE_QP>(
            dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_idx, quiet_batch, aggregate);
        udma_perf_quiet<USE_QP>(peer_pe, qp_idx);
    }
    if (remainder > 0) {
        udma_perf_get_group<T, USE_QP>(
            dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_idx, remainder, aggregate);
        udma_perf_quiet<USE_QP>(peer_pe, qp_idx);
    }
    SHMEMI_PROF_END(frame_id);
    aclshmemx_sync_vec_all();
}

// put_signal has its own immediate API and intentionally does not use QP/action aggregation.
template <typename T>
__aicore__ inline void udma_perf_test_put_signal_impl(
    GM_ADDR dst_gva, GM_ADDR src_gva, GM_ADDR sig_gva, int elements, int32_t frame_id, int64_t prof_pe_val,
    int loop_count, uint64_t signal_base, int batch)
{
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();

    if (pe != prof_pe_val) {
        aclshmemx_sync_vec_all();
        return;
    }

    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva);
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva);
    __gm__ uint64_t* sig_addr = reinterpret_cast<__gm__ uint64_t*>(sig_gva) + pe;

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (batch <= 0 || batch > loop_test) ? loop_test : batch;
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < warmup; ++i) {
        aclshmemx_udma_put_signal_nbi(
            dst_gm, src_gm, static_cast<uint32_t>(elements), sig_addr, signal_base + static_cast<uint64_t>(i), peer_pe);
    }
    aclshmemx_udma_quiet(peer_pe);
    int full_groups = loop_test / batch_size;
    int remainder = loop_test - full_groups * batch_size;
    uint64_t sig_value = signal_base + static_cast<uint64_t>(warmup);
    SHMEMI_PROF_START(frame_id);
    for (int g = 0; g < full_groups; ++g) {
        for (int j = 0; j < batch_size; ++j) {
            aclshmemx_udma_put_signal_nbi(
                dst_gm, src_gm, static_cast<uint32_t>(elements), sig_addr, sig_value++, peer_pe);
        }
        aclshmemx_udma_quiet(peer_pe);
    }
    for (int j = 0; j < remainder; ++j) {
        aclshmemx_udma_put_signal_nbi(dst_gm, src_gm, static_cast<uint32_t>(elements), sig_addr, sig_value++, peer_pe);
    }
    if (remainder > 0) {
        aclshmemx_udma_quiet(peer_pe);
    }
    SHMEMI_PROF_END(frame_id);
    aclshmemx_sync_vec_all();
}

#define DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(type_name, cpp_type)                                                         \
    extern "C" __global__ __vector__ void udma_perf_test_##type_name##_put(                                           \
        GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::udma_mode_t test_mode,            \
        int ub_size_kb, int64_t prof_pe_val, int loop_count, int metric, int batch, int qp_count, GM_ADDR timing_gva) \
    {                                                                                                                 \
        if (qp_count > 1) {                                                                                           \
            udma_perf_test_put_impl<cpp_type, true>(                                                                  \
                dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count, metric, batch,  \
                timing_gva);                                                                                          \
        } else {                                                                                                      \
            udma_perf_test_put_impl<cpp_type, false>(                                                                 \
                dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count, metric, batch,  \
                timing_gva);                                                                                          \
        }                                                                                                             \
    }                                                                                                                 \
    extern "C" __global__ __vector__ void udma_perf_test_##type_name##_get(                                           \
        GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::udma_mode_t test_mode,            \
        int ub_size_kb, int64_t prof_pe_val, int loop_count, int batch, int qp_count, GM_ADDR timing_gva)             \
    {                                                                                                                 \
        (void)timing_gva;                                                                                             \
        if (qp_count > 1) {                                                                                           \
            udma_perf_test_get_impl<cpp_type, true>(                                                                  \
                dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count, batch);         \
        } else {                                                                                                      \
            udma_perf_test_get_impl<cpp_type, false>(                                                                 \
                dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count, batch);         \
        }                                                                                                             \
    }                                                                                                                 \
    extern "C" __global__ __vector__ void udma_perf_test_##type_name##_put_signal(                                    \
        GM_ADDR dst_gva, GM_ADDR src_gva, GM_ADDR sig_gva, int elements, int32_t frame_id, int64_t prof_pe_val,       \
        int loop_count, uint64_t signal_base, int batch, GM_ADDR timing_gva)                                          \
    {                                                                                                                 \
        (void)timing_gva;                                                                                             \
        udma_perf_test_put_signal_impl<cpp_type>(                                                                     \
            dst_gva, src_gva, sig_gva, elements, frame_id, prof_pe_val, loop_count, signal_base, batch);              \
    }

DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(float, float)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int8, int8_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int16, int16_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int64, int64_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint8, uint8_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint16, uint16_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint32, uint32_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint64, uint64_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(char, char)

#define DISPATCH_UDMA_PERF_PUT(type_name, cpp_type)                                                                 \
    udma_perf_test_##type_name##_put<<<block_dim, nullptr, stream>>>(                                               \
        dst_gva, src_gva, elements, frame_id, t_mode, ub_size_kb, prof_pe_val, loop_count, metric, batch, qp_count, \
        timing_gva)

#define DISPATCH_UDMA_PERF_GET(type_name, cpp_type)                                                         \
    udma_perf_test_##type_name##_get<<<block_dim, nullptr, stream>>>(                                       \
        dst_gva, src_gva, elements, frame_id, t_mode, ub_size_kb, prof_pe_val, loop_count, batch, qp_count, \
        timing_gva)

#define DISPATCH_UDMA_PERF_PUT_SIGNAL(type_name, cpp_type)                   \
    udma_perf_test_##type_name##_put_signal<<<block_dim, nullptr, stream>>>( \
        dst_gva, src_gva, sig_gva, elements, frame_id, prof_pe_val, loop_count, signal_base, batch, timing_gva)

#define DISPATCH_UDMA_PERF_FOR_ALL_TYPES(MACRO) \
    switch (d_type) {                           \
        case perftest::DATA_TYPE_FLOAT:         \
            MACRO(float, float);                \
            break;                              \
        case perftest::DATA_TYPE_INT8:          \
            MACRO(int8, int8_t);                \
            break;                              \
        case perftest::DATA_TYPE_INT16:         \
            MACRO(int16, int16_t);              \
            break;                              \
        case perftest::DATA_TYPE_INT32:         \
            MACRO(int32, int32_t);              \
            break;                              \
        case perftest::DATA_TYPE_INT64:         \
            MACRO(int64, int64_t);              \
            break;                              \
        case perftest::DATA_TYPE_UINT8:         \
            MACRO(uint8, uint8_t);              \
            break;                              \
        case perftest::DATA_TYPE_UINT16:        \
            MACRO(uint16, uint16_t);            \
            break;                              \
        case perftest::DATA_TYPE_UINT32:        \
            MACRO(uint32, uint32_t);            \
            break;                              \
        case perftest::DATA_TYPE_UINT64:        \
            MACRO(uint64, uint64_t);            \
            break;                              \
        case perftest::DATA_TYPE_CHAR:          \
            MACRO(char, char);                  \
            break;                              \
        default:                                \
            MACRO(float, float);                \
            break;                              \
    }

extern "C" void launch_udma_perf_kernel(
    uint32_t block_dim, void* stream, uint8_t* dst_gva, uint8_t* src_gva, uint8_t* sig_gva, int elements,
    int32_t frame_id, int test_mode, int data_type, int ub_size_kb, int64_t prof_pe_val, int loop_count,
    uint64_t signal_base, int metric, int batch, int qp_count, uint8_t* timing_gva)
{
    perftest::udma_mode_t t_mode = static_cast<perftest::udma_mode_t>(test_mode);
    perftest::perf_data_type_t d_type = static_cast<perftest::perf_data_type_t>(data_type);

    switch (t_mode) {
        case perftest::TEST_MODE_UDMA_PUT:
        case perftest::TEST_MODE_UDMA_BI_PUT:
            DISPATCH_UDMA_PERF_FOR_ALL_TYPES(DISPATCH_UDMA_PERF_PUT);
            break;
        case perftest::TEST_MODE_UDMA_GET:
        case perftest::TEST_MODE_UDMA_BI_GET:
            DISPATCH_UDMA_PERF_FOR_ALL_TYPES(DISPATCH_UDMA_PERF_GET);
            break;
        case perftest::TEST_MODE_UDMA_PUT_SIGNAL:
            DISPATCH_UDMA_PERF_FOR_ALL_TYPES(DISPATCH_UDMA_PERF_PUT_SIGNAL);
            break;
        default:
            break;
    }
}

#endif // _UDMAPERF_KERNEL_UDMA_PERFTEST_
