/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * SIMT UDMA QP-specific Performance Test.
 *
 * Measures put/get bandwidth of simt::aclshmemx_udma_qp_{put,get}_nbi_{warp|block}
 * and simt::aclshmemx_udma_qp_quiet_{warp|block} across warp/block scopes.
 *
 * Architecture:
 * - 2 PE model: in unidirectional mode (put/get) only Active PE (PE0) issues
 *   transfers and Passive PE (PE1) is the remote endpoint; in bidirectional
 *   mode (bi_put/bi_get) both PEs issue transfers simultaneously. Prof data
 *   is collected on SHMEM_CYCLE_PROF_PE (PE0).
 * - Each warp/block drives one QP. QP count is set via aclshmemx_set_qp_num
 *   before init. Each QP transfers a contiguous slice of the symmetric buffer.
 * - The transfer VF loops internally (iters put/get calls per QP, no per-call
 *   quiet), then a separate quiet VF drains all QPs. The prof window covers
 *   both the issue loop and the quiet drain, giving end-to-end bandwidth.
 * - Source and destination use separate symmetric buffers (src and dst):
 *   put reads from the local src buffer and writes to the remote dst buffer;
 *   get reads from the remote src buffer and writes to the local dst buffer.
 *   This avoids the read-write race that occurs in a same-address model when
 *   both PEs transfer simultaneously (bidirectional mode).
 *
 * Scope and operation selection (--scope / -t|--test-type):
 * - Execution scope is a source-level property of the SIMT UDMA QP interfaces
 *   (warp / block variants are different calls with different launch
 *   shapes), so each scope/op combination is instantiated at compile time as a
 *   separate VF template specialization. The VF to launch is picked through a
 *   runtime dispatch ladder (asc_vf_call requires a concrete VF function), and
 *   `all` expands to every combination sequentially in one process.
 *
 * Runtime metric selection (--metric):
 *   bw : Bandwidth = (slice * qp_count * iters) / prof_time; the prof window
 *        covers the issue loop, the quiet drain and SyncAll.
 *   lat: net per-call latency = (issue-loop window - isomorphic baseline
 *        window) / iters; quiet is drained outside the window.
 *
 * Command-line flags follow the shmem_perftest family conventions
 * (--pes/--pe-id/--gnpus/--fnpu/--fpe/--ipport, -e/--exponent, ...).
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "utils.h"
#include "simt_runtime.h"
#include "utils/prof/shmemi_prof.h"

constexpr int32_t ACTIVE_PE = 0;
constexpr int32_t PASSIVE_PE = 1;
constexpr uint32_t BLOCK_VF_THREADS = 128;
// Max in-flight WQEs per QP before the SQ exhausts credit; loop_count/warmup must stay
// below this so the issue loop never triggers a mid-loop poll_cq. Derived from the 32768 BB
// SQ depth, the 10 BB credit guard, and the 2 BB each SIMT WQE occupies.
constexpr uint32_t SQ_CREDIT_LIMIT = (32768U - 10U - 2U) / 2U;

// A single SIMT UDMA request transfers at most 256 MiB (the simt_udma_qp_contract upper bound,
// 256 * 1024 * 1024 = 2^28 B). The host narrows the per-QP slice to the uint32_t elem_size the
// interface takes, so a larger slice would either be rejected by the device-side range check or
// (for exponent >= 32) silently truncate to 0. Hence the transfer-size exponent is capped at 28.
constexpr int32_t MAX_TRANSFER_EXPONENT = 28;

// Metric selection (runtime, passed to the kernel).
constexpr int32_t METRIC_BW = 0;  // end-to-end bandwidth: window covers issue loop + quiet + SyncAll
constexpr int32_t METRIC_LAT = 1; // net per-call latency: issue-only window minus isomorphic baseline window

// Execution scope and operation type of one test configuration.
enum class Scope : int32_t { WARP = 0, BLOCK = 1 };
enum class Op : int32_t { PUT = 0, GET = 1, BI_PUT = 2, BI_GET = 3 };

/*
 * Compile-time operation properties used by device code. They are exposed as
 * static constants rather than functions because the device compiler rejects
 * plain (host) function calls from __aicore__/__simt_vf__ code.
 */
template <Op O>
struct OpTraits {
    static constexpr bool kIsPut = (O == Op::PUT || O == Op::BI_PUT);
    static constexpr bool kIsBidirectional = (O == Op::BI_PUT || O == Op::BI_GET);
};

// Host-side runtime mirrors of OpTraits, used while iterating the selected ops.
constexpr bool IsPut(Op op) { return op == Op::PUT || op == Op::BI_PUT; }

constexpr bool IsBidirectional(Op op) { return op == Op::BI_PUT || op == Op::BI_GET; }

constexpr const char* ScopeName(Scope scope)
{
    switch (scope) {
        case Scope::WARP:
            return "warp";
        default:
            return "block";
    }
}

constexpr const char* OpName(Op op)
{
    switch (op) {
        case Op::PUT:
            return "put";
        case Op::GET:
            return "get";
        case Op::BI_PUT:
            return "bi_put";
        default:
            return "bi_get";
    }
}

/*
 * Per-scope VF traits: default __launch_bounds__ value for each scope.
 *
 * __launch_bounds__(N) is a compile-time constant that determines the
 * per-thread register quota: 1~256 -> 127 regs, 257~512 -> 64,
 * 513~1024 -> 32, 1025~2048 -> 16.  Lower tiers give more registers per
 * thread, reducing spilling and improving latency.
 *
 * For BLOCK the default N equals the worst-case thread count
 * (always <= 128, so the 127-reg tier is guaranteed).  For WARP the
 * worst-case is ACLSHMEM_MAX_QP_NUM * 32 = 1792, which would fall in the
 * 16-reg tier and cause spilling.  Instead, WARP uses a tier-dispatch
 * mechanism (see RunConfigWarp) that picks the smallest N whose tier
 * still fits qp_count * 32, maximizing the register budget for the
 * actual QP count.
 */
template <Scope S>
struct ScopeTraits;

template <>
struct ScopeTraits<Scope::WARP> {
    static constexpr uint32_t kThreadsMax = ACLSHMEM_MAX_QP_NUM * 32U; // worst-case; overridden per qp_count
};

template <>
struct ScopeTraits<Scope::BLOCK> {
    static constexpr uint32_t kThreadsMax = BLOCK_VF_THREADS; // 128, 1~256 tier (127 regs)
};

template <Scope S>
__aicore__ inline cce::dim3 VfDim(uint32_t qp_count)
{
    if constexpr (S == Scope::WARP) {
        return cce::dim3(qp_count * 32U);
    } else {
        return cce::dim3(BLOCK_VF_THREADS);
    }
}

aclshmemx_uniqueid_t default_flag_uid;
static aclshmem_prof_pe_t* out_profs = nullptr;

struct VfSync {
    __aicore__ static inline void start()
    {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }
    __aicore__ static inline void end()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    }
};

/*
 * Transfer VF: each warp/block drives one QP, loops iters times
 * issuing put/get without per-call quiet. The VF launch dimension is
 * scope-specific; each driver computes its own qp_idx internally.
 *
 * kThreadsMax overrides __launch_bounds__ for WARP tier dispatch; for
 * BLOCK it defaults to ScopeTraits<S>::kThreadsMax.
 */
template <Scope S, Op O, uint32_t kThreadsMax = ScopeTraits<S>::kThreadsMax>
__simt_vf__ __launch_bounds__(kThreadsMax) inline void transfer_vf(
    __gm__ uint8_t* sym_src, __gm__ uint8_t* sym_dst, uint32_t slice, int32_t peer, uint32_t iters)
{
    uint32_t qp_idx = 0;
    if constexpr (S == Scope::BLOCK) {
        qp_idx = blockIdx.x;
    } else {
        const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        qp_idx = tid / warpSize;
    }

    __gm__ uint8_t* src_ptr = sym_src + static_cast<uint64_t>(qp_idx) * slice;
    __gm__ uint8_t* dst_ptr = sym_dst + static_cast<uint64_t>(qp_idx) * slice;

    for (uint32_t i = 0; i < iters; ++i) {
        if constexpr (OpTraits<O>::kIsPut) {
            if constexpr (S == Scope::WARP) {
                simt::aclshmemx_udma_qp_put_nbi_warp<uint8_t>(dst_ptr, src_ptr, slice, peer, qp_idx);
            } else {
                simt::aclshmemx_udma_qp_put_nbi_block<uint8_t>(dst_ptr, src_ptr, slice, peer, qp_idx);
            }
        } else {
            if constexpr (S == Scope::WARP) {
                simt::aclshmemx_udma_qp_get_nbi_warp<uint8_t>(dst_ptr, src_ptr, slice, peer, qp_idx);
            } else {
                simt::aclshmemx_udma_qp_get_nbi_block<uint8_t>(dst_ptr, src_ptr, slice, peer, qp_idx);
            }
        }
    }
}

/*
 * Quiet VF: each driver drains its own QP. Called after transfer_vf to
 * complete all pending WQEs before the prof window closes.
 */
template <Scope S, uint32_t kThreadsMax = ScopeTraits<S>::kThreadsMax>
__simt_vf__ __launch_bounds__(kThreadsMax) inline void quiet_vf(int32_t peer)
{
    uint32_t qp_idx = 0;
    if constexpr (S == Scope::BLOCK) {
        qp_idx = blockIdx.x;
    } else {
        const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        qp_idx = tid / warpSize;
    }

    if constexpr (S == Scope::WARP) {
        simt::aclshmemx_udma_qp_quiet_warp(peer, qp_idx);
    } else {
        simt::aclshmemx_udma_qp_quiet_block(peer, qp_idx);
    }
}

/*
 * Baseline VF: isomorphic empty loop with a volatile sink (no interface call).
 * Used by the lat metric as the subtraction term, removing the loop-skeleton
 * cost from the real window so the remainder is the pure per-call interface cost.
 * The volatile store cannot be optimized away, so the loop is not eliminated.
 */
template <Scope S, uint32_t kThreadsMax = ScopeTraits<S>::kThreadsMax>
__simt_vf__ __launch_bounds__(kThreadsMax) inline void baseline_vf(
    __gm__ uint8_t* sym_addr, uint32_t slice, int32_t peer, uint32_t iters)
{
    (void)sym_addr;
    (void)slice;
    (void)peer;
    volatile uint32_t sink = 0;
    for (uint32_t i = 0; i < iters; ++i) {
        sink = i;
    }
    (void)sink;
}

/*
 * Runs one scope/op configuration on the current block.
 *
 * In unidirectional mode only Active PE issues transfers; in bidirectional
 * mode both PEs issue transfers simultaneously.
 *
 * bw metric (frame_id): window covers the issue loop, the quiet drain and the
 *   trailing SyncAll, giving end-to-end bandwidth.
 * lat metric (frame_id, frame_id + 1): phase 1 covers the issue loop only
 *   (quiet stays outside the window); phase 2 measures an isomorphic empty
 *   loop. The host subtracts phase 2 from phase 1 and divides by iters to get
 *   the net per-call latency.
 *
 * kThreadsMax selects the __launch_bounds__ tier; for BLOCK it defaults
 * to ScopeTraits<S>::kThreadsMax, and for WARP it is selected per qp_count
 * by RunConfigWarp.
 */
template <Scope S, Op O, uint32_t kThreadsMax = ScopeTraits<S>::kThreadsMax>
__aicore__ inline void RunConfig(
    int32_t metric, int32_t frame_id, int32_t do_prof, __gm__ uint8_t* sym_src, __gm__ uint8_t* sym_dst, uint32_t slice,
    int32_t peer, uint32_t qp_count, uint32_t iters)
{
    if constexpr (!OpTraits<O>::kIsBidirectional) {
        if (aclshmem_my_pe() != ACTIVE_PE) {
            return;
        }
    }

    AscendC::SyncAll<true>();

    if (metric == METRIC_LAT) {
        // Phase 1 (frame_id): issue loop only; quiet is drained outside the window.
        VfSync::start();
        if (do_prof) {
            SHMEMI_PROF_START(frame_id);
        }
        asc_vf_call<transfer_vf<S, O, kThreadsMax>>(VfDim<S>(qp_count), sym_src, sym_dst, slice, peer, iters);
        VfSync::end();
        if (do_prof) {
            SHMEMI_PROF_END(frame_id);
        }

        VfSync::start();
        asc_vf_call<quiet_vf<S, kThreadsMax>>(VfDim<S>(qp_count), peer);
        VfSync::end();

        // Phase 2 (frame_id + 1): isomorphic empty-loop baseline.
        if (do_prof) {
            VfSync::start();
            SHMEMI_PROF_START(frame_id + 1);
            asc_vf_call<baseline_vf<S, kThreadsMax>>(VfDim<S>(qp_count), sym_src, slice, peer, iters);
            VfSync::end();
            SHMEMI_PROF_END(frame_id + 1);
        }

        AscendC::SyncAll<true>();
        return;
    }

    VfSync::start();
    if (do_prof) {
        SHMEMI_PROF_START(frame_id);
    }

    asc_vf_call<transfer_vf<S, O, kThreadsMax>>(VfDim<S>(qp_count), sym_src, sym_dst, slice, peer, iters);
    asc_vf_call<quiet_vf<S, kThreadsMax>>(VfDim<S>(qp_count), peer);

    AscendC::SyncAll<true>();

    if (do_prof) {
        SHMEMI_PROF_END(frame_id);
    }
    VfSync::end();
}

/*
 * WARP tier dispatcher: selects the RunConfig specialization whose
 * __launch_bounds__ tier best fits the actual qp_count, maximizing the
 * per-thread register budget.  Empirically, using the worst-case bound
 * (1792) when qp_count is small doubles measured latency due to register
 * spilling; selecting the tightest tier avoids this.
 */
template <Op O>
__aicore__ inline void RunConfigWarp(
    int32_t metric, int32_t frame_id, int32_t do_prof, __gm__ uint8_t* sym_src, __gm__ uint8_t* sym_dst, uint32_t slice,
    int32_t peer, uint32_t qp_count, uint32_t iters)
{
    const uint32_t threads = qp_count * 32U;
    if (threads <= 256U) {
        RunConfig<Scope::WARP, O, 256U>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
    } else if (threads <= 512U) {
        RunConfig<Scope::WARP, O, 512U>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
    } else if (threads <= 1024U) {
        RunConfig<Scope::WARP, O, 1024U>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
    } else {
        RunConfig<Scope::WARP, O, ACLSHMEM_MAX_QP_NUM * 32U>(
            metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
    }
}

/*
 * Dispatch the requested operation for one execution scope. The scope/op pair
 * arrives as runtime values, so the ladder below names every VF template
 * specialization explicitly; only the requested configuration is launched.
 *
 * For WARP, the dispatch goes through RunConfigWarp which selects the
 * __launch_bounds__ tier matching the actual qp_count, instead of using the
 * worst-case bound.
 */
template <Scope S>
__aicore__ inline void DispatchOp(
    int32_t op, int32_t metric, int32_t frame_id, int32_t do_prof, __gm__ uint8_t* sym_src, __gm__ uint8_t* sym_dst,
    uint32_t slice, int32_t peer, uint32_t qp_count, uint32_t iters)
{
    if constexpr (S == Scope::WARP) {
        if (op == static_cast<int32_t>(Op::PUT)) {
            RunConfigWarp<Op::PUT>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        } else if (op == static_cast<int32_t>(Op::GET)) {
            RunConfigWarp<Op::GET>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        } else if (op == static_cast<int32_t>(Op::BI_PUT)) {
            RunConfigWarp<Op::BI_PUT>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        } else {
            RunConfigWarp<Op::BI_GET>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        }
    } else {
        if (op == static_cast<int32_t>(Op::PUT)) {
            RunConfig<S, Op::PUT>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        } else if (op == static_cast<int32_t>(Op::GET)) {
            RunConfig<S, Op::GET>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        } else if (op == static_cast<int32_t>(Op::BI_PUT)) {
            RunConfig<S, Op::BI_PUT>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        } else {
            RunConfig<S, Op::BI_GET>(metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
        }
    }
}

__global__ __vector__ void run_perf_kernel(
    __gm__ uint8_t* sym_src, __gm__ uint8_t* sym_dst, uint32_t slice, int32_t peer, uint32_t qp_count, uint32_t iters,
    int32_t frame_id, int32_t do_prof, int32_t metric, int32_t scope, int32_t op)
{
    if (scope == static_cast<int32_t>(Scope::WARP)) {
        DispatchOp<Scope::WARP>(op, metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
    } else {
        DispatchOp<Scope::BLOCK>(op, metric, frame_id, do_prof, sym_src, sym_dst, slice, peer, qp_count, iters);
    }
}

// Fills a host staging buffer with this PE's verification pattern.
static void InitData(void* data, size_t len, int pe)
{
    std::fill_n(static_cast<int8_t*>(data), len, static_cast<int8_t>(pe + 10));
}

static bool CheckData(const void* data, size_t len, int expected_pe)
{
    const int8_t* p = static_cast<const int8_t*>(data);
    const int8_t expected = static_cast<int8_t>(expected_pe + 10);
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != expected) {
            std::cerr << "[ERROR] data[" << i << "]=" << static_cast<int>(p[i])
                      << " expected=" << static_cast<int>(expected) << std::endl;
            return false;
        }
    }
    return true;
}

struct Options {
    int pe{0};
    int pes{2};
    int gnpus{2};
    int first_npu{0};
    int first_pe{0}; // accepted for CLI parity with the other shmem_perftest samples; unused
    uint32_t qp_count{4};
    int min_exponent{3};
    int max_exponent{17};
    uint32_t loop_count{1000};
    uint32_t warmup{100};
    uint64_t heap_mb{1024};
    int32_t metric{METRIC_BW};
    std::string ipport{"tcp://127.0.0.1:8999"};
    std::vector<Scope> scopes{Scope::WARP, Scope::BLOCK};
    std::vector<Op> ops{Op::PUT, Op::GET, Op::BI_PUT, Op::BI_GET};
};

static bool ParseScopeList(const std::string& text, std::vector<Scope>& scopes)
{
    if (text == "all") {
        scopes = {Scope::WARP, Scope::BLOCK};
    } else if (text == "warp") {
        scopes = {Scope::WARP};
    } else if (text == "block") {
        scopes = {Scope::BLOCK};
    } else {
        return false;
    }
    return true;
}

static bool ParseOpList(const std::string& text, std::vector<Op>& ops)
{
    if (text == "all") {
        ops = {Op::PUT, Op::GET, Op::BI_PUT, Op::BI_GET};
    } else if (text == "put") {
        ops = {Op::PUT};
    } else if (text == "get") {
        ops = {Op::GET};
    } else if (text == "bi_put") {
        ops = {Op::BI_PUT};
    } else if (text == "bi_get") {
        ops = {Op::BI_GET};
    } else {
        return false;
    }
    return true;
}

static bool ParseOptions(int argc, char* argv[], Options& o)
{
    int i = 1;
    while (i < argc) {
        std::string a = argv[i];
        int j = i + 1;
        auto val = [&]() { return (j < argc) ? argv[j++] : ""; };
        try {
            if (a == "--pe-id") {
                o.pe = std::stoi(val());
            } else if (a == "--pes") {
                o.pes = std::stoi(val());
            } else if (a == "--gnpus") {
                o.gnpus = std::stoi(val());
            } else if (a == "--fnpu") {
                o.first_npu = std::stoi(val());
            } else if (a == "--fpe") {
                o.first_pe = std::stoi(val());
            } else if (a == "--scope") {
                if (!ParseScopeList(val(), o.scopes)) {
                    return false;
                }
            } else if (a == "-t" || a == "--test-type") {
                if (!ParseOpList(val(), o.ops)) {
                    return false;
                }
            } else if (a == "--qp") {
                long v = std::stol(val());
                if (v <= 0 || v > ACLSHMEM_MAX_QP_NUM) {
                    return false;
                }
                o.qp_count = static_cast<uint32_t>(v);
            } else if (a == "-e" || a == "--exponent") {
                o.min_exponent = o.max_exponent = std::stoi(val());
            } else if (a == "--exponent-range") {
                o.min_exponent = std::stoi(val());
                o.max_exponent = std::stoi(val());
            } else if (a == "--loop-count") {
                o.loop_count = static_cast<uint32_t>(std::stoul(val(), nullptr, 0));
            } else if (a == "--warmup") {
                o.warmup = static_cast<uint32_t>(std::stoul(val(), nullptr, 0));
            } else if (a == "--heap-mb") {
                o.heap_mb = std::stoull(val(), nullptr, 0);
            } else if (a == "--metric") {
                const std::string m = val();
                if (m == "bw") {
                    o.metric = METRIC_BW;
                } else if (m == "lat") {
                    o.metric = METRIC_LAT;
                } else {
                    return false;
                }
            } else if (a == "--ipport") {
                o.ipport = val();
            } else {
                return false;
            }
        } catch (const std::exception&) {
            return false;
        }
        i = j;
    }
    return o.pe >= 0 && o.pes >= 2 && o.pe < o.pes && o.gnpus >= 1 && o.first_npu >= 0 && o.first_pe >= 0 &&
           o.qp_count >= 1 && o.qp_count <= ACLSHMEM_MAX_QP_NUM && o.min_exponent >= 1 &&
           o.min_exponent <= o.max_exponent && o.max_exponent <= MAX_TRANSFER_EXPONENT && o.loop_count >= 1 &&
           o.warmup >= 0;
}

// Per-block latency sample of the lat metric, in microseconds per call.
struct LatSample {
    double gross_per_call{0.0};    // phase 1 (issue loop) window / loop_count
    double baseline_per_call{0.0}; // phase 2 (empty loop) window / loop_count
    double net_per_call{0.0};      // gross - baseline
    int32_t block_id{-1};          // originating block index
};

// Cycles per microsecond, matching utils.h collect_prof_data_to_csv_v2.
static double CyclesPerUs()
{
    const char* soc_name = aclrtGetSocName();
    return (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) ? 1000.0 : 50.0;
}

/*
 * lat metric: collect the per-block net latency from the two prof frames
 * (frame_real = issue loop, frame_base = isomorphic empty loop). A block is
 * included only when both frames have records. Returns false when the local PE
 * has no prof data (not the SHMEM_CYCLE_PROF_PE collector).
 */
static bool CollectLatSamples(
    const aclshmem_prof_pe_t* profs, int32_t frame_real, int32_t frame_base, uint32_t loop_count,
    std::vector<LatSample>& samples)
{
    if (profs == nullptr || frame_real >= ACLSHMEM_CYCLE_PROF_FRAME_CNT ||
        frame_base >= ACLSHMEM_CYCLE_PROF_FRAME_CNT || loop_count == 0) {
        return false;
    }
    const double cycles_per_us = CyclesPerUs();
    for (int32_t block_id = 0; block_id < ACLSHMEM_CYCLE_PROF_MAX_BLOCK; ++block_id) {
        const aclshmem_prof_block_t* prof = &profs->block_prof[block_id];
        if (prof->ccount[frame_real] == 0 || prof->ccount[frame_base] == 0) {
            continue;
        }
        LatSample sample;
        sample.block_id = block_id;
        sample.gross_per_call =
            static_cast<double>(prof->cycles[frame_real]) / prof->ccount[frame_real] / cycles_per_us / loop_count;
        sample.baseline_per_call =
            static_cast<double>(prof->cycles[frame_base]) / prof->ccount[frame_base] / cycles_per_us / loop_count;
        sample.net_per_call = sample.gross_per_call - sample.baseline_per_call;
        samples.push_back(sample);
    }
    return !samples.empty();
}

/*
 * The prof collector PE writes one CSV per scope/op configuration.
 */
static void WriteConfigCsv(
    const Options& options, Scope scope, Op op, const std::vector<std::vector<std::string>>& csv_data)
{
    if (options.pe != ACTIVE_PE) {
        return;
    }
    const std::string metric_name = (options.metric == METRIC_LAT) ? "lat" : "bw";
    std::string csv_filename = "output/simt_udma_" + metric_name + "_" + std::string(OpName(op)) + "_" +
                               std::string(ScopeName(scope)) + "_qp" + int_to_string(options.qp_count) + "_" +
                               int_to_string(options.min_exponent) + "-" + int_to_string(options.max_exponent) + "_l" +
                               int_to_string(options.loop_count) + ".csv";
    write_csv(csv_filename, csv_data);
    std::cout << "[CSV] " << csv_filename << std::endl;
}

/*
 * Runs one scope/op configuration over the whole exponent range: warmup,
 * timed round, data verification and prof collection. Returns 0 on success, or 1 when the
 * prof frame budget is exhausted and the caller must stop the remaining configurations.
 * Local failures are accumulated in ever_failed and never return early, so the PE keeps
 * hitting every aclshmem_barrier_all() on the same collective path as its peer.
 */
static int32_t RunOneConfig(
    const Options& options, RuntimeResources& resources, uint8_t* sym_src, uint8_t* sym_dst, bool buffers_ok,
    Scope scope, Op op, int32_t& frame_id, bool& ever_failed)
{
    // In unidirectional mode: put - Passive verifies Active's value;
    //                           get - Active verifies Passive's value.
    // In bidirectional mode: both PEs issue transfers, so each PE verifies
    //                         the peer's value.
    int32_t expected_pe = (options.pe + 1) % options.pes;
    bool should_verify = true;
    if (!IsBidirectional(op)) {
        expected_pe = IsPut(op) ? ACTIVE_PE : PASSIVE_PE;
        should_verify = (options.pe == (IsPut(op) ? PASSIVE_PE : ACTIVE_PE));
    }

    std::cout << "[INFO] scope=" << ScopeName(scope) << " op=" << OpName(op) << " pe=" << options.pe
              << " qp_count=" << options.qp_count << std::endl;

    const uint32_t grid = (scope == Scope::BLOCK) ? options.qp_count : 1;
    const int32_t scope_arg = static_cast<int32_t>(scope);
    const int32_t op_arg = static_cast<int32_t>(op);

    const int expected_blocks =
        (scope == Scope::BLOCK) ? std::min(static_cast<int>(options.qp_count), ACLSHMEM_CYCLE_PROF_MAX_BLOCK) : 1;

    std::vector<std::string> csv_header = {
        "DataSize/B", "Npus", "QPs", "UBsize/KB", "Bandwidth/GB/s(1000)", "Bandwidth/GiB/s(1024)", "CoreMaxTime/us"};
    for (int b = 0; b < expected_blocks; ++b) {
        csv_header.push_back("SingleCoreTime_B" + int_to_string(b) + "/us");
    }
    std::vector<std::vector<std::string>> csv_data = {csv_header};

    for (int exp = options.min_exponent; exp <= options.max_exponent; ++exp) {
        const uint64_t slice = 1ULL << exp;
        const uint64_t total = slice * options.qp_count;
        std::cout << "pe: " << options.pe << " size(per QP): " << slice << "B, total: " << total
                  << "B frame_id: " << frame_id << std::endl;

        // The symmetric heap lives in device memory; host-side data must be
        // staged in host memory first and transferred with an explicit copy.
        std::vector<int8_t> host_buf(total);
        InitData(host_buf.data(), total, options.pe);
        if (buffers_ok && aclrtMemcpy(sym_src, total, host_buf.data(), total, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            ERROR_LOG("H2D memcpy (sym_src) failed");
            ever_failed = true;
        }

        aclshmem_barrier_all();

        if (buffers_ok) {
            run_perf_kernel<<<grid, 0, resources.stream>>>(
                sym_src, sym_dst, static_cast<uint32_t>(slice), (options.pe + 1) % options.pes, options.qp_count,
                options.warmup, frame_id, 0, options.metric, scope_arg, op_arg);
            if (aclrtSynchronizeStream(resources.stream) != 0) {
                ERROR_LOG("warmup kernel sync failed, errMsg: %s", aclGetRecentErrMsg());
                ever_failed = true;
            }
        }
        aclshmem_barrier_all();

        InitData(host_buf.data(), total, options.pe);
        if (buffers_ok && aclrtMemcpy(sym_src, total, host_buf.data(), total, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            ERROR_LOG("H2D memcpy (sym_src, timed) failed");
            ever_failed = true;
        }
        // Reset the destination so a dropped QP in the timed run is not masked by the
        // warmup result that still resides in sym_dst.
        std::fill_n(host_buf.data(), total, static_cast<int8_t>(0));
        if (buffers_ok && aclrtMemcpy(sym_dst, total, host_buf.data(), total, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            ERROR_LOG("H2D memcpy (sym_dst) failed");
            ever_failed = true;
        }
        aclshmem_barrier_all();

        if (buffers_ok) {
            run_perf_kernel<<<grid, 0, resources.stream>>>(
                sym_src, sym_dst, static_cast<uint32_t>(slice), (options.pe + 1) % options.pes, options.qp_count,
                options.loop_count, frame_id, 1, options.metric, scope_arg, op_arg);
            if (aclrtSynchronizeStream(resources.stream) != 0) {
                ERROR_LOG("timed kernel sync failed, errMsg: %s", aclGetRecentErrMsg());
                ever_failed = true;
            }
        }
        aclshmem_barrier_all();

        if (should_verify && buffers_ok) {
            if (aclrtMemcpy(host_buf.data(), total, sym_dst, total, ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                ERROR_LOG("D2H memcpy failed");
                ever_failed = true;
            } else if (!CheckData(host_buf.data(), total, expected_pe)) {
                ERROR_LOG("Data verification FAILED on PE %d", options.pe);
                ever_failed = true;
            } else {
                std::cout << "[Verification] SUCCESS" << std::endl;
            }
        }

        aclshmemx_get_prof(&out_profs, false);
        if (options.metric == METRIC_LAT) {
            std::vector<LatSample> samples;
            if (CollectLatSamples(out_profs, frame_id, frame_id + 1, options.loop_count, samples)) {
                const LatSample* worst = &samples[0];
                for (const LatSample& s : samples) {
                    if (s.net_per_call > worst->net_per_call) {
                        worst = &s;
                    }
                }
                std::vector<std::string> row = {
                    uint64_to_string(slice),
                    int_to_string(options.gnpus),
                    int_to_string(options.qp_count),
                    "0",
                    "0.00",
                    "0.00",
                    double_to_string(worst->net_per_call)};
                for (int b = 0; b < expected_blocks; ++b) {
                    const LatSample* match = nullptr;
                    for (const LatSample& s : samples) {
                        if (s.block_id == b) {
                            match = &s;
                            break;
                        }
                    }
                    row.push_back(match ? double_to_string(match->net_per_call) : "N/A");
                }
                csv_data.push_back(row);
                std::cout << "[lat] pe=" << options.pe << " slice=" << slice << "B net=" << worst->net_per_call
                          << " us/call (gross=" << worst->gross_per_call << ", baseline=" << worst->baseline_per_call
                          << ")" << std::endl;
            }
        } else {
            if (out_profs != nullptr) {
                const double cycles_per_us = CyclesPerUs();
                double max_core_time = 0.0;
                std::vector<std::string> core_time_strs;
                core_time_strs.reserve(expected_blocks);
                for (int b = 0; b < expected_blocks; ++b) {
                    const aclshmem_prof_block_t* prof = &out_profs->block_prof[b];
                    if (prof->ccount[frame_id] == 0) {
                        core_time_strs.push_back("N/A");
                        continue;
                    }
                    const double window_us =
                        static_cast<double>(prof->cycles[frame_id]) / prof->ccount[frame_id] / cycles_per_us;
                    const double per_iter_us =
                        (options.loop_count > 0) ? window_us / static_cast<double>(options.loop_count) : window_us;
                    if (per_iter_us > max_core_time) {
                        max_core_time = per_iter_us;
                    }
                    core_time_strs.push_back(double_to_string(per_iter_us));
                }
                if (max_core_time > 0.0) {
                    double bandwidth_gb = 0.0;
                    double bandwidth_gib = 0.0;
                    const double bytes_per_sec =
                        static_cast<double>(slice) * static_cast<double>(options.qp_count) / max_core_time * 1000000.0;
                    bandwidth_gb = bytes_per_sec / 1000.0 / 1000.0 / 1000.0;
                    bandwidth_gib = bytes_per_sec / 1024.0 / 1024.0 / 1024.0;
                    std::vector<std::string> row = {uint64_to_string(slice),         int_to_string(options.gnpus),
                                                    int_to_string(options.qp_count), "0",
                                                    double_to_string(bandwidth_gb),  double_to_string(bandwidth_gib),
                                                    double_to_string(max_core_time)};
                    for (const std::string& s : core_time_strs) {
                        row.push_back(s);
                    }
                    csv_data.push_back(row);
                }
            }
        }

        frame_id += (options.metric == METRIC_LAT) ? 2 : 1;
        if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
            std::cerr << "warning: frame_id reached limit " << ACLSHMEM_CYCLE_PROF_FRAME_CNT
                      << "; stopping the remaining configurations" << std::endl;
            WriteConfigCsv(options, scope, op, csv_data);
            return 1;
        }
    }

    WriteConfigCsv(options, scope, op, csv_data);
    return 0;
}

static int32_t ExecuteTestLoop(const Options& options, RuntimeResources& resources)
{
    const uint64_t max_per_qp = 1ULL << options.max_exponent;
    if (max_per_qp > UINT64_MAX / options.qp_count) {
        ERROR_LOG(
            "max_per_qp=%llu * qp_count=%u overflows uint64_t", static_cast<unsigned long long>(max_per_qp),
            options.qp_count);
        return -1;
    }
    const uint64_t max_total = max_per_qp * options.qp_count;
    const uint64_t heap_bytes = options.heap_mb * 1024ULL * 1024ULL;
    if (max_total > UINT64_MAX / 2U || max_total * 2U > heap_bytes) {
        ERROR_LOG(
            "Required buffers (%lluB x2) exceeds heap (%lluB)", static_cast<unsigned long long>(max_total),
            static_cast<unsigned long long>(heap_bytes));
        return -1;
    }

    // A buffer that could not be allocated only disables the transfers and the checks; the
    // loop and its barriers stay unconditional, so no PE leaves its peer waiting.
    uint8_t* sym_src = static_cast<uint8_t*>(aclshmem_malloc(max_total));
    if (sym_src == nullptr) {
        ERROR_LOG("aclshmem_malloc(sym_src) failed");
    }
    uint8_t* sym_dst = static_cast<uint8_t*>(aclshmem_malloc(max_total));
    if (sym_dst == nullptr) {
        ERROR_LOG("aclshmem_malloc(sym_dst) failed");
    }
    const bool buffers_ok = (sym_src != nullptr && sym_dst != nullptr);

    bool ever_failed = !buffers_ok;
    bool stop_all = false;
    int32_t frame_id = 0;

    for (Scope scope : options.scopes) {
        for (Op op : options.ops) {
            const int32_t rc =
                RunOneConfig(options, resources, sym_src, sym_dst, buffers_ok, scope, op, frame_id, ever_failed);
            if (rc != 0) {
                // rc > 0: prof frame budget exhausted. Both PEs reach this at the same
                // configuration (frame_id advances identically), so stopping here is a
                // paired collective decision. A local failure only sets ever_failed via
                // the reference and must NOT break: the failing PE keeps marching through
                // every remaining aclshmem_barrier_all() on the peer's collective path.
                stop_all = true;
                break;
            }
        }
        if (stop_all) {
            break;
        }
    }

    aclshmemx_get_prof(nullptr, true);
    // No aclshmem_free here: it is a collective that skips its barrier for a null pointer, so
    // freeing per buffer would desynchronize a PE whose allocation failed. aclshmem_finalize
    // in the caller destroys the heap for every PE.

    if (ever_failed) {
        ERROR_LOG("Some tests failed on PE %d", options.pe);
    }
    return ever_failed ? -1 : 0;
}

int main(int argc, char* argv[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("SHMEM_CYCLE_PROF_PE", "0", 1);

    Options options;
    if (!ParseOptions(argc, argv, options)) {
        printf(
            "Usage: %s --pe-id <ID> --pes <N> [--gnpus <N>] [--fnpu <ID>] [--fpe <ID>]"
            " [--scope warp|block|all] [-t|--test-type put|get|bi_put|bi_get|all] [--qp <N>]"
            " [-e|--exponent <E>] [--exponent-range <min> <max>] [--loop-count <N>] [--warmup <N>]"
            " [--metric bw|lat] [--heap-mb <N>] [--ipport <tcp://IP:PORT>]\n",
            argv[0]);
        return 1;
    }

    if (options.loop_count >= SQ_CREDIT_LIMIT || options.warmup >= SQ_CREDIT_LIMIT) {
        ERROR_LOG(
            "loop_count/warmup must < %u (SQ credit limit); got loop=%u warmup=%u", SQ_CREDIT_LIMIT, options.loop_count,
            options.warmup);
        return 1;
    }

    INFO_LOG(
        "pe=%d pes=%d qp_count=%u exp=%d-%d loop=%u warmup=%u metric=%s scopes=%u ops=%u", options.pe, options.pes,
        options.qp_count, options.min_exponent, options.max_exponent, options.loop_count, options.warmup,
        (options.metric == METRIC_LAT) ? "lat" : "bw", static_cast<unsigned>(options.scopes.size()),
        static_cast<unsigned>(options.ops.size()));

    RuntimeResources resources;
    resources.device_id = options.pe % options.gnpus + options.first_npu;

    int status = aclInit(nullptr);
    if (status != 0) {
        ERROR_LOG("aclInit failed, status=%d", status);
        return status;
    }
    resources.acl_initialized = true;

    status = aclrtSetDevice(resources.device_id);
    if (status != 0) {
        ERROR_LOG("aclrtSetDevice failed, status=%d", status);
        return CleanupResources(resources, status);
    }
    resources.device_set = true;

    status = aclrtCreateStream(&resources.stream);
    if (status != 0) {
        ERROR_LOG("aclrtCreateStream failed, status=%d", status);
        return CleanupResources(resources, status);
    }
    resources.stream_created = true;

    const uint64_t heap_bytes = options.heap_mb * 1024ULL * 1024ULL;
    aclshmemx_init_attr_t attr{};
    status = test_set_attr(options.pe, options.pes, heap_bytes, options.ipport.c_str(), default_flag_uid, &attr);
    if (status != 0) {
        ERROR_LOG("test_set_attr failed, status=%d", status);
        return CleanupResources(resources, status);
    }
    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status = aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, options.qp_count);
    if (status != 0) {
        ERROR_LOG("aclshmemx_set_qp_num(%u) failed, status=%d", options.qp_count, status);
        return CleanupResources(resources, status);
    }
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    if (status != 0) {
        ERROR_LOG("aclshmemx_init_attr failed, status=%d", status);
        return CleanupResources(resources, status);
    }
    resources.shmem_initialized = true;

    int32_t ret = ExecuteTestLoop(options, resources);

    status = CleanupResources(resources, 0);
    if (ret != 0 || status != 0) {
        std::cerr << "[FAILED] simt_udma_perftest failed in pe " << options.pe << std::endl;
        return (ret != 0) ? ret : status;
    }

    std::cout << "[SUCCESS] simt_udma_perftest done in pe " << options.pe << std::endl;
    return 0;
}
