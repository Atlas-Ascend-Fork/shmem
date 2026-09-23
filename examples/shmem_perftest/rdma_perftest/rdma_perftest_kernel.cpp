/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RDMAPERF_KERNEL_RDMA_PERFTEST_
#define _RDMAPERF_KERNEL_RDMA_PERFTEST_

#include "kernel_operator.h"
#include "shmem.h"
#include "device/shmem_def.h"
#include "perftest_common_types.h"

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
constexpr uint64_t RDMA_AGGREGATE_THRESHOLD = 64 * 1024;
#else
constexpr uint64_t RDMA_AGGREGATE_THRESHOLD = 32 * 1024;
#endif

template <typename T>
__aicore__ inline void rdma_perf_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_num, int qp_index,
    uint32_t op_idx, uint32_t sync_id)
{
    constexpr bool use_qp_specific = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                                     ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    const uint32_t block_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool multi_core = AscendC::GetBlockNum() > 1;
    const uint32_t selected_qp =
        qp_index >= 0 ? static_cast<uint32_t>(qp_index) : (multi_core ? block_id : op_idx % qp_num);
    if constexpr (use_qp_specific) {
        aclshmemx_roce_qp_put_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id);
    } else {
        aclshmemx_roce_put_nbi(dst, src, buf, elem_size, pe, sync_id);
    }
}

template <typename T>
__aicore__ inline void rdma_perf_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_num, int qp_index,
    uint32_t op_idx, uint32_t sync_id)
{
    constexpr bool use_qp_specific = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                                     ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    const uint32_t block_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool multi_core = AscendC::GetBlockNum() > 1;
    const uint32_t selected_qp =
        qp_index >= 0 ? static_cast<uint32_t>(qp_index) : (multi_core ? block_id : op_idx % qp_num);
    if constexpr (use_qp_specific) {
        aclshmemx_roce_qp_get_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id);
    } else {
        aclshmemx_roce_get_nbi(dst, src, buf, elem_size, pe, sync_id);
    }
}

#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
template <typename T>
__aicore__ inline void rdma_perf_put_aggregate_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t selected_qp, uint32_t op_count,
    uint32_t sync_id)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);
    for (uint32_t op = 0; op < op_count; ++op) {
        const bool is_last = op + 1 == op_count;
        if constexpr (ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::IN_DIE) {
            if (is_last)
                aclshmemx_roce_put_nbi(dst, src, buf, elem_size, pe, sync_id, submit_action);
            else
                aclshmemx_roce_put_nbi(dst, src, buf, elem_size, pe, sync_id, defer_action);
        } else if (is_last) {
            aclshmemx_roce_qp_put_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, submit_action);
        } else {
            aclshmemx_roce_qp_put_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, defer_action);
        }
    }
}

template <typename T>
__aicore__ inline void rdma_perf_get_aggregate_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t selected_qp, uint32_t op_count,
    uint32_t sync_id)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);
    for (uint32_t op = 0; op < op_count; ++op) {
        const bool is_last = op + 1 == op_count;
        if constexpr (ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::IN_DIE) {
            if (is_last)
                aclshmemx_roce_get_nbi(dst, src, buf, elem_size, pe, sync_id, submit_action);
            else
                aclshmemx_roce_get_nbi(dst, src, buf, elem_size, pe, sync_id, defer_action);
        } else if (is_last) {
            aclshmemx_roce_qp_get_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, submit_action);
        } else {
            aclshmemx_roce_qp_get_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, defer_action);
        }
    }
}
#endif

__aicore__ inline bool rdma_perf_should_aggregate(uint32_t qp_num, uint64_t msg_size, int metric)
{
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    const bool qp_supported =
        qp_num >= 1 && qp_num <= (ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ? 2 : 1);
    return metric == static_cast<int>(perftest::PERF_METRIC_BW) && qp_supported && msg_size < RDMA_AGGREGATE_THRESHOLD;
#else
    (void)qp_num;
    (void)msg_size;
    (void)metric;
    return false;
#endif
}

template <typename T>
__aicore__ inline void rdma_perf_quiet(
    uint32_t pe, __ubuf__ T* buf, uint32_t qp_num, uint32_t op_count, int qp_index, uint32_t qp_start_idx,
    uint32_t sync_id)
{
    constexpr bool use_qp_specific = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                                     ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    const uint32_t block_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool multi_core = AscendC::GetBlockNum() > 1;
    if constexpr (use_qp_specific) {
        if (qp_index >= 0) {
            aclshmemx_roce_qp_quiet(pe, static_cast<uint32_t>(qp_index), buf, sync_id);
            return;
        }
        if (multi_core) {
            aclshmemx_roce_qp_quiet(pe, block_id, buf, sync_id);
            return;
        }
        if (qp_num == 1) {
            aclshmemx_roce_qp_quiet(pe, 0, buf, sync_id);
            return;
        }

        uint32_t quiet_qp_count = op_count < qp_num ? op_count : qp_num;
        for (uint32_t qp_idx = 0; qp_idx < quiet_qp_count; ++qp_idx) {
            aclshmemx_roce_qp_quiet(pe, (qp_start_idx + qp_idx) % qp_num, buf, sync_id);
        }
    } else {
        aclshmemx_roce_quiet(pe, buf, sync_id);
    }
}

// lat 模式接口形态：一次时延测试自动遍历当前后端支持的全部形态
constexpr int RDMA_PERF_IFACE_PLAIN = 0; // 普通非聚合接口（逐次立即提交，所有后端）
constexpr int RDMA_PERF_IFACE_QP = 1;    // QP 指定接口（XSCALE/HNS_1825，逐次立即提交）
constexpr int RDMA_PERF_IFACE_DEFER = 2; // 聚合 defer 接口（攒批不提交；IN_DIE 普通 defer，XSCALE QP defer）
constexpr int RDMA_PERF_IFACE_SUBMIT = 3; // 聚合 submit 接口（提交攒批；IN_DIE 普通 submit，XSCALE QP submit）
constexpr int RDMA_PERF_IFACE_COUNT = 4;

constexpr int RDMA_PERF_LAT_IFACE_ORDER[RDMA_PERF_IFACE_COUNT] = {
    RDMA_PERF_IFACE_PLAIN,
    RDMA_PERF_IFACE_QP,
    RDMA_PERF_IFACE_DEFER,
    RDMA_PERF_IFACE_SUBMIT,
};
// N 次 defer 后追加一条 submit WQE；N+1 必须小于 SQ 深度 1024。
constexpr int RDMA_PERF_MAX_AGG_WINDOW = 1022;

__aicore__ inline bool rdma_perf_lat_iface_supported(int iface, uint32_t qp_num)
{
    constexpr bool has_qp = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                            ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    if (iface == RDMA_PERF_IFACE_QP) {
        return has_qp;
    }
    if (iface == RDMA_PERF_IFACE_DEFER || iface == RDMA_PERF_IFACE_SUBMIT) {
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        return ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ? qp_num <= 2 : qp_num == 1;
#else
        return false;
#endif
    }
    return true;
}

// 按接口形态提交 count 个操作。submit warmup 逐次独立提交，defer warmup 才使用一次聚合提交。
template <typename T, bool IS_GET>
__aicore__ inline void rdma_perf_lat_submit_batch(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, uint32_t selected_qp,
    uint32_t sync_id, int iface, uint32_t count)
{
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    if (iface == RDMA_PERF_IFACE_SUBMIT) {
        for (uint32_t op = 0; op < count; ++op) {
            aclshmemx_submit_state_t submit_state{};
            aclshmemx_submit_t submit_action(submit_state);
            if constexpr (ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::IN_DIE) {
                if constexpr (IS_GET) {
                    aclshmemx_roce_get_nbi(dst, src, ub_ptr, elements, peer_pe, sync_id, submit_action);
                } else {
                    aclshmemx_roce_put_nbi(dst, src, ub_ptr, elements, peer_pe, sync_id, submit_action);
                }
            } else {
                if constexpr (IS_GET) {
                    aclshmemx_roce_qp_get_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, sync_id, submit_action);
                } else {
                    aclshmemx_roce_qp_put_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, sync_id, submit_action);
                }
            }
        }
        return;
    }
    if (iface == RDMA_PERF_IFACE_DEFER && count > 1) {
        if constexpr (IS_GET) {
            rdma_perf_get_aggregate_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, count, sync_id);
        } else {
            rdma_perf_put_aggregate_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, count, sync_id);
        }
        return;
    }
#endif
    if (iface == RDMA_PERF_IFACE_PLAIN) {
        if constexpr (IS_GET) {
            for (uint32_t op = 0; op < count; ++op) {
                aclshmemx_roce_get_nbi(dst, src, ub_ptr, elements, peer_pe, sync_id);
            }
        } else {
            for (uint32_t op = 0; op < count; ++op) {
                aclshmemx_roce_put_nbi(dst, src, ub_ptr, elements, peer_pe, sync_id);
            }
        }
        return;
    }
    // QP 接口（或聚合余量 1 次时退化为单发，与带宽路径处理一致）
    if constexpr (
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
        ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825) {
        if constexpr (IS_GET) {
            for (uint32_t op = 0; op < count; ++op) {
                aclshmemx_roce_qp_get_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, sync_id);
            }
        } else {
            for (uint32_t op = 0; op < count; ++op) {
                aclshmemx_roce_qp_put_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, sync_id);
            }
        }
    }
}

// 按接口形态选择 quiet 目标：PLAIN 固定走 QP0（普通接口内部固定 QP0），QP/DEFER/SUBMIT 走所选 QP
template <typename T>
__aicore__ inline void rdma_perf_lat_quiet(
    uint32_t pe, __ubuf__ T* buf, uint32_t qp_num, int qp_index, uint32_t aggregate_qp, uint32_t sync_id, int iface)
{
    constexpr bool use_qp_specific = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                                     ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    if constexpr (use_qp_specific) {
        const uint32_t quiet_qp = iface == RDMA_PERF_IFACE_PLAIN ? 0u : aggregate_qp;
        (void)qp_num;
        (void)qp_index;
        aclshmemx_roce_qp_quiet(pe, quiet_qp, buf, sync_id);
    } else {
        (void)qp_num;
        (void)qp_index;
        (void)aggregate_qp;
        aclshmemx_roce_quiet(pe, buf, sync_id);
    }
}

#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
// 单次聚合调用分派：IN_DIE 普通 defer/submit 接口，XSCALE QP defer/submit 接口
template <typename T, bool IS_GET, typename ACTION>
__aicore__ inline void rdma_perf_lat_agg_op(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, uint32_t selected_qp,
    uint32_t sync_id, ACTION action)
{
    if constexpr (ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::IN_DIE) {
        if constexpr (IS_GET) {
            aclshmemx_roce_get_nbi(dst, src, ub_ptr, elements, peer_pe, sync_id, action);
        } else {
            aclshmemx_roce_put_nbi(dst, src, ub_ptr, elements, peer_pe, sync_id, action);
        }
    } else {
        if constexpr (IS_GET) {
            aclshmemx_roce_qp_get_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, sync_id, action);
        } else {
            aclshmemx_roce_qp_put_nbi(dst, src, ub_ptr, elements, peer_pe, selected_qp, sync_id, action);
        }
    }
}

// defer 接口时延：整窗摊薄计时，开始计时后连续调用 N 次 defer 再结束计时，除以 N 摊薄到单次调用，
// 最后 1 次 submit 补齐并提交聚合组（不计时），窗口间 quiet 不计时；返回摊薄到单次 defer 的 cycles
template <typename T, bool IS_GET>
__aicore__ inline int64_t rdma_perf_lat_time_defer(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, uint32_t qp_num,
    int qp_index, uint32_t aggregate_qp, uint32_t sync_id, int loop_test, int window)
{
    int64_t total_cycles = 0;
    int64_t total_ops = 0;
    const int full_groups = loop_test / window;
    const int remainder = loop_test - full_groups * window;
    for (int g = 0; g < full_groups + (remainder > 0 ? 1 : 0); ++g) {
        const int group = g < full_groups ? window : remainder;
        aclshmemx_submit_state_t submit_state{};
        aclshmemx_defer_t defer_action(submit_state);
        aclshmemx_submit_t submit_action(submit_state);
        AscendC::PipeBarrier<PIPE_ALL>();
        const int64_t start = AscendC::GetSystemCycle();
        for (int op = 0; op < group; ++op) { // 窗口内 N 次 defer 全部计时
            rdma_perf_lat_agg_op<T, IS_GET>(dst, src, ub_ptr, elements, peer_pe, aggregate_qp, sync_id, defer_action);
        }
        total_cycles += AscendC::GetSystemCycle() - start;
        total_ops += group;
        // 组内最后一个操作固定为 submit，补齐并提交本聚合组（不计时）
        rdma_perf_lat_agg_op<T, IS_GET>(dst, src, ub_ptr, elements, peer_pe, aggregate_qp, sync_id, submit_action);
        rdma_perf_lat_quiet<T>(peer_pe, ub_ptr, qp_num, qp_index, aggregate_qp, sync_id, RDMA_PERF_IFACE_DEFER);
    }
    return total_ops > 0 ? total_cycles / total_ops : -1;
}

// submit 接口时延：与普通接口一致，连续调用 loop_test 次 submit 后结束计时，再除以 loop_test。
// submit 每次只提交本次操作，quiet 不计时且只在整段计时结束后执行。
template <typename T, bool IS_GET>
__aicore__ inline int64_t rdma_perf_lat_time_submit(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, uint32_t qp_num,
    int qp_index, uint32_t aggregate_qp, uint32_t sync_id, int loop_test)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_submit_t submit_action(submit_state);
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t start = AscendC::GetSystemCycle();
    for (int op = 0; op < loop_test; ++op) {
        rdma_perf_lat_agg_op<T, IS_GET>(dst, src, ub_ptr, elements, peer_pe, aggregate_qp, sync_id, submit_action);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t total_cycles = AscendC::GetSystemCycle() - start;
    rdma_perf_lat_quiet<T>(peer_pe, ub_ptr, qp_num, qp_index, aggregate_qp, sync_id, RDMA_PERF_IFACE_SUBMIT);
    return loop_test > 0 ? total_cycles / loop_test : -1;
}
#endif

// lat 模式主逻辑：遍历全部受支持接口形态，各自 warmup + 计时，结果写入 timing_out[slot]
// timing 槽位布局：slot = pe * RDMA_PERF_IFACE_COUNT + iface（BW 模式仍使用前 2 槽 slot=pe）
template <typename T, bool IS_GET>
__aicore__ inline void rdma_perf_lat_run_ifaces(
    __gm__ T* dst_gm, __gm__ T* src_gm, __ubuf__ T* ub_ptr, uint32_t elements, uint32_t peer_pe, uint32_t qp_num,
    int qp_index, uint32_t aggregate_qp, int loop_test, int batch_size, int warmup, uint32_t sync_id, bool is_bidir,
    int64_t pe, __gm__ int64_t* timing_out)
{
    const int agg_window = batch_size > RDMA_PERF_MAX_AGG_WINDOW ? RDMA_PERF_MAX_AGG_WINDOW : batch_size;
    for (int order = 0; order < RDMA_PERF_IFACE_COUNT; ++order) {
        const int iface = RDMA_PERF_LAT_IFACE_ORDER[order];
        const int slot = static_cast<int>(pe) * RDMA_PERF_IFACE_COUNT + iface;
        if (!rdma_perf_lat_iface_supported(iface, qp_num)) {
            if (timing_out != nullptr) {
                timing_out[slot] = -1; // -1 表示该后端不支持此接口形态
            }
            continue;
        }
        // warmup：submit/plain/QP 逐次独立提交；defer 一次聚合提交 warmup 个
        rdma_perf_lat_submit_batch<T, IS_GET>(
            dst_gm, src_gm, ub_ptr, elements, peer_pe, aggregate_qp, sync_id, iface, static_cast<uint32_t>(warmup));
        rdma_perf_lat_quiet<T>(peer_pe, ub_ptr, qp_num, qp_index, aggregate_qp, sync_id, iface);

        int64_t per_op_cycles = -1;
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        if (iface == RDMA_PERF_IFACE_DEFER) {
            per_op_cycles = agg_window >= 2 ? rdma_perf_lat_time_defer<T, IS_GET>(
                                                  dst_gm, src_gm, ub_ptr, elements, peer_pe, qp_num, qp_index,
                                                  aggregate_qp, sync_id, loop_test, agg_window) :
                                              -1;
        } else if (iface == RDMA_PERF_IFACE_SUBMIT) {
            per_op_cycles = rdma_perf_lat_time_submit<T, IS_GET>(
                dst_gm, src_gm, ub_ptr, elements, peer_pe, qp_num, qp_index, aggregate_qp, sync_id, loop_test);
        } else
#endif
        {
            // 普通接口时延：全部 loop_test 次提交计时，quiet 不计时，摊薄到单次调用
            AscendC::PipeBarrier<PIPE_ALL>();
            const int64_t pass_start = AscendC::GetSystemCycle();
            rdma_perf_lat_submit_batch<T, IS_GET>(
                dst_gm, src_gm, ub_ptr, elements, peer_pe, aggregate_qp, sync_id, iface,
                static_cast<uint32_t>(loop_test));
            AscendC::PipeBarrier<PIPE_ALL>();
            const int64_t total_cycles = AscendC::GetSystemCycle() - pass_start;
            rdma_perf_lat_quiet<T>(peer_pe, ub_ptr, qp_num, qp_index, aggregate_qp, sync_id, iface);
            per_op_cycles = loop_test > 0 ? total_cycles / loop_test : 0;
        }
        if (timing_out != nullptr) {
            timing_out[slot] = per_op_cycles; // 单次调用摊薄 cycles
        }
    }
    if (timing_out != nullptr) {
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * RDMA_PERF_IFACE_COUNT * 2);
        if (is_bidir) {
            aclshmemx_roce_barrier_all();
            // 本 PE 的 4 个时延槽连续（slot = pe*4 + iface），一次性同步到对端
            __gm__ T* slot_base =
                reinterpret_cast<__gm__ T*>(&timing_out[static_cast<size_t>(pe) * RDMA_PERF_IFACE_COUNT]);
            rdma_perf_put_nbi(
                slot_base, slot_base, ub_ptr, sizeof(int64_t) * RDMA_PERF_IFACE_COUNT / sizeof(T), peer_pe, qp_num,
                qp_index, 0, sync_id);
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, 1, qp_index, 0, sync_id);
            aclshmemx_roce_barrier_all();
        }
    }
}

template <typename T>
__aicore__ inline void rdma_perf_test_put_impl(
    uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, int ub_size_b,
    int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index, GM_ADDR timing_out_gva)
{
    util_set_ffts_config(fftsAddr);
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();
    __gm__ int64_t* timing_out = reinterpret_cast<__gm__ int64_t*>(timing_out_gva);

    bool is_bidir = (test_mode == perftest::TEST_MODE_RDMA_BI_PUT);
    const bool use_multi_core =
        (metric == static_cast<int>(perftest::PERF_METRIC_BW) && qp_num > 1 && qp_index < 0 &&
         AscendC::GetBlockNum() > 1);
    const uint32_t block_id = use_multi_core ? static_cast<uint32_t>(AscendC::GetBlockIdx()) : 0;
    const bool all_blocks_active = is_bidir || pe == 0;

    if (use_multi_core && !is_bidir && pe != 0 && block_id != 0) {
        return;
    }
    if (use_multi_core) {
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        // Ensure both PEs are synchronized before any RDMA operations begin
        aclshmemx_roce_barrier_all();
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
    }

    if (!is_bidir && pe != 0) {
        if (timing_out != nullptr) {
            // 清空本 PE 相关槽位（BW 布局 slot=pe，lat 布局 slot=pe*4+iface），PE0 槽保留
            for (int slot = 1; slot < 2 * RDMA_PERF_IFACE_COUNT; ++slot) {
                timing_out[slot] = 0;
            }
        }
        aclshmemx_roce_barrier_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_b);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_b, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ubLocal.GetPhyAddr());

    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva);
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva);
    if (use_multi_core) {
        const uint64_t qp_region_offset = static_cast<uint64_t>(elements) * block_id;
        dst_gm += qp_region_offset;
        src_gm += qp_region_offset;
    }

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (loop_test <= 0 || batch <= 0 || batch > loop_test) ? loop_test : batch;
    uint32_t qp_op_seq = 0;

    const uint64_t msg_size = static_cast<uint64_t>(elements) * sizeof(T);
    const bool use_aggregate = rdma_perf_should_aggregate(qp_num, msg_size, metric);
    const uint32_t aggregate_qp = qp_index >= 0 ? static_cast<uint32_t>(qp_index) : block_id;

    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        // 时延模式：自动遍历当前后端支持的接口形态（普通/QP/聚合 defer+submit），分别计时
        rdma_perf_lat_run_ifaces<T, false>(
            dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, aggregate_qp, loop_test,
            batch_size, warmup, sync_id, is_bidir, pe, timing_out);
    } else {
        // Bandwidth: submit NBIs in groups of `batch_size`, quiet at each group boundary.
        const uint32_t warmup_qp_start = qp_op_seq;
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        if (use_aggregate && warmup > 1) {
            rdma_perf_put_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(warmup), sync_id);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                rdma_perf_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, warmup, qp_index, warmup_qp_start % qp_num, sync_id);
        int full_groups = loop_test / batch_size;
        int remainder = loop_test - full_groups * batch_size;
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t waiting_time_start = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                waiting_time_start = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            waiting_time_start = AscendC::GetSystemCycle();
        }
        for (int g = 0; g < full_groups; ++g) {
            const uint32_t group_qp_start = qp_op_seq;
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
            if (use_aggregate && batch_size > 1) {
                rdma_perf_put_aggregate_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                    static_cast<uint32_t>(batch_size), sync_id);
            } else
#endif
            {
                for (int j = 0; j < batch_size; ++j) {
                    rdma_perf_put_nbi(
                        dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                        sync_id);
                }
            }
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, batch_size, qp_index, group_qp_start % qp_num, sync_id);
        }
        const uint32_t remainder_qp_start = qp_op_seq;
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        if (use_aggregate && remainder > 1) {
            rdma_perf_put_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(remainder), sync_id);
        } else
#endif
        {
            for (int j = 0; j < remainder; ++j) {
                rdma_perf_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        if (remainder > 0) {
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, remainder, qp_index, remainder_qp_start % qp_num, sync_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t waiting_time_end = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                waiting_time_end = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            waiting_time_end = AscendC::GetSystemCycle();
        }

        // Output: PE0 -> timing_out[0], PE1 -> timing_out[1]
        if (timing_out != nullptr && (!use_multi_core || block_id == 0)) {
            int64_t waiting_total_time = waiting_time_end - waiting_time_start;
            if (pe == 0) {
                timing_out[0] = waiting_total_time;
            } else {
                timing_out[1] = waiting_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                const uint32_t timing_qp_start = qp_op_seq;
                rdma_perf_put_nbi(
                    slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, qp_num, qp_index, qp_op_seq++, sync_id);
                rdma_perf_quiet(peer_pe, ub_ptr, qp_num, 1, qp_index, timing_qp_start % qp_num, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        aclshmemx_roce_barrier_all();
    }
}

template <typename T>
__aicore__ inline void rdma_perf_test_get_impl(
    uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, int ub_size_b,
    int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index, GM_ADDR timing_out_gva)
{
    util_set_ffts_config(fftsAddr);
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();
    __gm__ int64_t* timing_out = reinterpret_cast<__gm__ int64_t*>(timing_out_gva);

    bool is_bidir = (test_mode == perftest::TEST_MODE_RDMA_BI_GET);
    const bool use_multi_core =
        (metric == static_cast<int>(perftest::PERF_METRIC_BW) && qp_num > 1 && qp_index < 0 &&
         AscendC::GetBlockNum() > 1);
    const uint32_t block_id = use_multi_core ? static_cast<uint32_t>(AscendC::GetBlockIdx()) : 0;
    const bool all_blocks_active = is_bidir || pe == 0;

    if (use_multi_core && !is_bidir && pe != 0 && block_id != 0) {
        return;
    }
    if (use_multi_core) {
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        // Ensure both PEs are synchronized before any RDMA operations begin
        aclshmemx_roce_barrier_all();
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
    }

    if (!is_bidir && pe != 0) {
        if (timing_out != nullptr) {
            // 清空本 PE 相关槽位（BW 布局 slot=pe，lat 布局 slot=pe*4+iface），PE0 槽保留
            for (int slot = 1; slot < 2 * RDMA_PERF_IFACE_COUNT; ++slot) {
                timing_out[slot] = 0;
            }
        }
        aclshmemx_roce_barrier_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_b);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_b, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ubLocal.GetPhyAddr());

    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva);
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva);
    if (use_multi_core) {
        const uint64_t qp_region_offset = static_cast<uint64_t>(elements) * block_id;
        dst_gm += qp_region_offset;
        src_gm += qp_region_offset;
    }

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (loop_test <= 0 || batch <= 0 || batch > loop_test) ? loop_test : batch;
    uint32_t qp_op_seq = 0;

    const uint64_t msg_size = static_cast<uint64_t>(elements) * sizeof(T);
    const bool use_aggregate = rdma_perf_should_aggregate(qp_num, msg_size, metric);
    const uint32_t aggregate_qp = qp_index >= 0 ? static_cast<uint32_t>(qp_index) : block_id;

    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        // 时延模式：自动遍历当前后端支持的接口形态（普通/QP/聚合 defer+submit），分别计时
        rdma_perf_lat_run_ifaces<T, true>(
            dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, aggregate_qp, loop_test,
            batch_size, warmup, sync_id, is_bidir, pe, timing_out);
    } else {
        // Bandwidth: submit NBIs in groups of `batch_size`, quiet at each group boundary.
        const uint32_t warmup_qp_start = qp_op_seq;
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        if (use_aggregate && warmup > 1) {
            rdma_perf_get_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(warmup), sync_id);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                rdma_perf_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, warmup, qp_index, warmup_qp_start % qp_num, sync_id);
        int full_groups = loop_test / batch_size;
        int remainder = loop_test - full_groups * batch_size;
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t get_time_start = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                get_time_start = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            get_time_start = AscendC::GetSystemCycle();
        }
        for (int g = 0; g < full_groups; ++g) {
            const uint32_t group_qp_start = qp_op_seq;
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
            if (use_aggregate && batch_size > 1) {
                rdma_perf_get_aggregate_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                    static_cast<uint32_t>(batch_size), sync_id);
            } else
#endif
            {
                for (int j = 0; j < batch_size; ++j) {
                    rdma_perf_get_nbi(
                        dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                        sync_id);
                }
            }
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, batch_size, qp_index, group_qp_start % qp_num, sync_id);
        }
        const uint32_t remainder_qp_start = qp_op_seq;
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        if (use_aggregate && remainder > 1) {
            rdma_perf_get_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(remainder), sync_id);
        } else
#endif
        {
            for (int j = 0; j < remainder; ++j) {
                rdma_perf_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        if (remainder > 0) {
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, remainder, qp_index, remainder_qp_start % qp_num, sync_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t get_time_end = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                get_time_end = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            get_time_end = AscendC::GetSystemCycle();
        }

        // dcci removed — relying on host-side 64B-aligned allocation instead
        // Output: PE0 -> timing_out[0], PE1 -> timing_out[1]
        if (timing_out != nullptr && (!use_multi_core || block_id == 0)) {
            int64_t get_total_time = get_time_end - get_time_start;
            if (pe == 0) {
                timing_out[0] = get_total_time;
            } else {
                timing_out[1] = get_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                const uint32_t timing_qp_start = qp_op_seq;
                rdma_perf_put_nbi(
                    slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, qp_num, qp_index, qp_op_seq++, sync_id);
                rdma_perf_quiet(peer_pe, ub_ptr, qp_num, 1, qp_index, timing_qp_start % qp_num, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        aclshmemx_roce_barrier_all();
    }
}

#define DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(type_name, cpp_type)                                                       \
    extern "C" __global__ __vector__ void rdma_perf_test_##type_name##_put(                                         \
        uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode,         \
        int ub_size_b, int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index,      \
        GM_ADDR timing_out_gva)                                                                                     \
    {                                                                                                               \
        rdma_perf_test_put_impl<cpp_type>(                                                                          \
            fftsAddr, dst_gva, src_gva, elements, test_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, \
            qp_index, timing_out_gva);                                                                              \
    }                                                                                                               \
    extern "C" __global__ __vector__ void rdma_perf_test_##type_name##_get(                                         \
        uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode,         \
        int ub_size_b, int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index,      \
        GM_ADDR timing_out_gva)                                                                                     \
    {                                                                                                               \
        rdma_perf_test_get_impl<cpp_type>(                                                                          \
            fftsAddr, dst_gva, src_gva, elements, test_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, \
            qp_index, timing_out_gva);                                                                              \
    }

DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(float, float)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int8, int8_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int16, int16_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int64, int64_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint8, uint8_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint16, uint16_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint32, uint32_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint64, uint64_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(char, char)

#define DISPATCH_RDMA_PERF_PUT(type_name, cpp_type)                                                                    \
    rdma_perf_test_##type_name##_put<<<block_dim, nullptr, stream>>>(                                                  \
        fftsAddr, dst_gva, src_gva, elements, t_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, qp_index, \
        timing_out_gva)

#define DISPATCH_RDMA_PERF_GET(type_name, cpp_type)                                                                    \
    rdma_perf_test_##type_name##_get<<<block_dim, nullptr, stream>>>(                                                  \
        fftsAddr, dst_gva, src_gva, elements, t_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, qp_index, \
        timing_out_gva)

#define DISPATCH_RDMA_PERF_FOR_ALL_TYPES(MACRO) \
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

extern "C" void launch_rdma_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t fftsAddr, uint8_t* dst_gva, uint8_t* src_gva, int elements,
    int test_mode, int data_type, int ub_size_b, int loop_count, int metric, int batch, int sync_id, int qp_num,
    int qp_index, uint8_t* timing_out_gva)
{
    perftest::rdma_mode_t t_mode = static_cast<perftest::rdma_mode_t>(test_mode);
    perftest::perf_data_type_t d_type = static_cast<perftest::perf_data_type_t>(data_type);

    switch (t_mode) {
        case perftest::TEST_MODE_RDMA_PUT:
        case perftest::TEST_MODE_RDMA_BI_PUT:
            DISPATCH_RDMA_PERF_FOR_ALL_TYPES(DISPATCH_RDMA_PERF_PUT);
            break;
        case perftest::TEST_MODE_RDMA_GET:
        case perftest::TEST_MODE_RDMA_BI_GET:
            DISPATCH_RDMA_PERF_FOR_ALL_TYPES(DISPATCH_RDMA_PERF_GET);
            break;
        default:
            break;
    }
}

#endif // _RDMAPERF_KERNEL_RDMA_PERFTEST_
