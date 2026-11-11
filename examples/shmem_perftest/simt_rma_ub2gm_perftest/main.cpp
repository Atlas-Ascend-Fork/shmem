/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "utils.h"
#include "utils/prof/shmemi_prof.h"

#include "argparser.h"

/*
 * SIMT RMA ub2gm Device-to-Device Performance Test
 *
 * This test measures the performance of ub2gm RMA interfaces:
 * - PUT: UB → Remote GM
 * - GET: Remote GM → UB
 *
 * Key differences from gm2gm test:
 * - Uses __ubuf__ as intermediate buffer
 * - UB data is prepared outside the measurement loop
 * - Only measures pure ub2gm interface performance
 */

// Element type of the transferred payload: the only place the tested interface's
// data width is decided. DATA_SIZE below is derived from it, so the two can never
// drift apart.
using T = int32_t;

// Compile-time configuration (simplified). The measured operation (put/get) is
// not among them: it is selected at run time via -t/--test-type (default: get).
constexpr int32_t DATA_SIZE = sizeof(T) * 8; // RMA interface data width, in bits
constexpr int32_t THREAD_COUNT = 1024;
constexpr int32_t UB_BUFFER_SIZE = 16384; // UB buffer size in elements (covers up to 2^16 B)
constexpr int32_t WARMUP_LOOPS = 128;

// Only the 32-bit ub2gm interfaces (aclshmemx_int32_{put,get}_nbi_warp) are wired
// up in the transfer functions below; widening T alone would not switch them.
static_assert(DATA_SIZE == 32, "This sample is fixed to the 32-bit ub2gm interfaces");

// The exponent sweep is capped so that one transfer always fits in the UB buffer.
// argparser.h enforces the bound; keep the two in sync when UB_BUFFER_SIZE changes.
static_assert(
    (1 << BYTES_IN_EXP_UPPER) <= UB_BUFFER_SIZE * static_cast<int32_t>(sizeof(T)),
    "BYTES_IN_EXP_UPPER exceeds the UB buffer capacity; raise UB_BUFFER_SIZE");

constexpr int32_t ACTIVE_PE = 0;
constexpr int32_t PASSIVE_PE = 1;

aclshmemx_uniqueid_t default_flag_uid;
static aclshmem_prof_pe_t* out_profs;

// Location and values of the first differing byte found by check_data().
struct DataMismatch {
    bool ok = true;
    int32_t block = -1;
    size_t offset_in_block = 0;
    size_t absolute_offset = 0;
    int8_t expected = 0;
    int8_t actual = 0;
    uint64_t mismatch_count = 0; // total differing bytes across all blocks
};

// Data preparation and validation.
//
// Both the fill and the check work at byte granularity rather than on T: the fill
// value (my_pe + 10) is written to every byte, so each T element reads back as
// 0x0A0A0A0A / 0x0B0B0B0B whatever the width of T is. Checking bytes also keeps the
// validated region exact when a transfer size is not a multiple of sizeof(T).
class PerfDataPrepValidator {
private:
    int32_t my_pe;
    OpType op_type;
    int8_t expected_value;

public:
    PerfDataPrepValidator(int32_t pe, OpType type) : my_pe(pe), op_type(type)
    {
        if (op_type == OpType::Put) {
            expected_value = ACTIVE_PE + 10;
        } else {
            expected_value = PASSIVE_PE + 10;
        }
    }

    void init_data(void* data, size_t len)
    {
        int8_t fill_val = my_pe + 10;
        std::fill_n(static_cast<int8_t*>(data), len, fill_val);
    }

    // Scans every block so mismatch_count reflects the full extent of the corruption.
    DataMismatch check_data(void* data, size_t effective_bytes_per_block, int32_t block_count, size_t block_stride)
    {
        DataMismatch result;
        if ((op_type == OpType::Put && my_pe == ACTIVE_PE) || (op_type == OpType::Get && my_pe == PASSIVE_PE)) {
            return result;
        }

        const int8_t* base = static_cast<const int8_t*>(data);
        for (int32_t b = 0; b < block_count; ++b) {
            const int8_t* block_start = base + b * block_stride;
            for (size_t i = 0; i < effective_bytes_per_block; ++i) {
                if (block_start[i] != expected_value) {
                    if (result.ok) {
                        result.ok = false;
                        result.block = b;
                        result.offset_in_block = i;
                        result.absolute_offset = static_cast<size_t>(b) * block_stride + i;
                        result.expected = expected_value;
                        result.actual = block_start[i];
                    }
                    ++result.mismatch_count;
                }
            }
        }
        return result;
    }
};

// Invoke parameters (same as gm2gm version)
struct InvokeParameters {
    constexpr static int32_t per_core_bytes = 1024 * 1024;
    int32_t per_invocation_bytes;
    int32_t logical_segment_bytes;
    int32_t warmup;
    int32_t loops;

    InvokeParameters(int32_t invoke_exp, int32_t warmup, int32_t loops) : warmup(warmup), loops(loops)
    {
        per_invocation_bytes = 1 << invoke_exp;
        int64_t total_requested = (warmup + loops) * static_cast<int64_t>(per_invocation_bytes);
        logical_segment_bytes = total_requested;
        if (total_requested > per_core_bytes) {
            logical_segment_bytes = per_core_bytes;
        }
        if (total_requested < per_invocation_bytes) {
            logical_segment_bytes = per_invocation_bytes;
        }
    }
};

// Scalar-Vector synchronization for SIMT
struct VSSync {
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

// SIMT transfer functions
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void transfer_vf_put(
    __gm__ T* dst, __ubuf__ T* src, size_t count, int32_t pe)
{
    simt::aclshmemx_int32_put_nbi_block(dst, src, count, pe);
}

__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void transfer_vf_get(
    __ubuf__ T* dst, __gm__ T* src, size_t count, int32_t pe)
{
    simt::aclshmemx_int32_get_nbi_block(dst, src, count, pe);
}

// Helper: copy GM to UB (used in __global__). 'bytes' is a byte count; DataCopyPad
// handles sizes that are not a multiple of the 32B UB alignment, which matters for
// the smallest transfer sizes in the sweep.
template <typename Elem>
ACLSHMEM_DEVICE void copy_gm2ub(__ubuf__ Elem* dstUb, __gm__ Elem* srcGva, uint32_t bytes)
{
    AscendC::LocalTensor<Elem> ubTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dstUb);
    ubTensor.address_.dataLen = ALIGN_UP(bytes, UB_ALIGN_SIZE);

    AscendC::GlobalTensor<Elem> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Elem*>(srcGva));

    AscendC::DataCopyExtParams dataCopyParams(1, bytes, 0, 0, 0);
    AscendC::DataCopyPadExtParams<Elem> padParams;
    AscendC::DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

// Helper: copy UB to GM (used in __global__). 'bytes' is a byte count.
template <typename Elem>
ACLSHMEM_DEVICE void copy_ub2gm(__gm__ Elem* dstGva, __ubuf__ Elem* srcUb, uint32_t bytes)
{
    AscendC::LocalTensor<Elem> ubTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(srcUb);
    ubTensor.address_.dataLen = ALIGN_UP(bytes, UB_ALIGN_SIZE);

    AscendC::GlobalTensor<Elem> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Elem*>(dstGva));

    AscendC::DataCopyExtParams dataCopyParams(1, bytes, 0, 0, 0);
    AscendC::DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

// Main test kernel. The op is a template parameter so each specialization is
// resolved at compile time (no per-iteration branch in the timed loop); the
// host selects the instantiation to launch based on -t/--test-type.
template <OpType OP>
__global__ __vector__ void run_demo_mem_ub2gm(__gm__ void* sym_addr, InvokeParameters invp, int32_t frame_id)
{
    int32_t my_pe = aclshmem_my_pe();
    if (my_pe != ACTIVE_PE)
        return;

    int32_t next_pe = (my_pe + 1) % aclshmem_n_pes();
    int32_t my_block_id = AscendC::GetBlockIdx();
    int32_t total_block_num = AscendC::GetBlockNum();

    __ubuf__ T ub_buf[UB_BUFFER_SIZE];

    const int32_t ring_size = total_block_num * invp.logical_segment_bytes;
    const int32_t steps_per_ring_scan = ring_size / invp.per_invocation_bytes;
    size_t count = invp.per_invocation_bytes / sizeof(T);

    if (count > UB_BUFFER_SIZE) {
        // Error: transfer size exceeds UB buffer size
        return;
    }

    // Preparation phase: initialize UB data (not timed)
    if constexpr (OP == OpType::Put) {
        __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(sym_addr);
        copy_gm2ub(ub_buf, src_gm, invp.per_invocation_bytes);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Warmup phase
    for (int32_t i = 0; i < invp.warmup; i++) {
        int32_t window_start = (i % steps_per_ring_scan) * invp.per_invocation_bytes;
        int32_t offset = (window_start + my_block_id * invp.logical_segment_bytes) % ring_size;
        __gm__ T* target_addr = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ int8_t*>(sym_addr) + offset);

        VSSync::start();
        if constexpr (OP == OpType::Put) {
            asc_vf_call<transfer_vf_put>(dim3(THREAD_COUNT), target_addr, ub_buf, count, next_pe);
        } else {
            asc_vf_call<transfer_vf_get>(dim3(THREAD_COUNT), ub_buf, target_addr, count, next_pe);
        }
        AscendC::SyncAll<true>();
        VSSync::end();
    }

    SHMEMI_PROF_START(frame_id);
    AscendC::PipeBarrier<PIPE_ALL>();

    for (int32_t i = 0; i < invp.loops; i++) {
        int32_t iter = invp.warmup + i;
        int32_t window_start = (iter % steps_per_ring_scan) * invp.per_invocation_bytes;
        int32_t offset = (window_start + my_block_id * invp.logical_segment_bytes) % ring_size;
        __gm__ T* target_addr = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ int8_t*>(sym_addr) + offset);

        VSSync::start();

        if constexpr (OP == OpType::Put) {
            asc_vf_call<transfer_vf_put>(dim3(THREAD_COUNT), target_addr, ub_buf, count, next_pe);
        } else {
            asc_vf_call<transfer_vf_get>(dim3(THREAD_COUNT), ub_buf, target_addr, count, next_pe);
        }

        AscendC::SyncAll<true>();
        VSSync::end();
    }

    SHMEMI_PROF_END(frame_id);
    AscendC::PipeBarrier<PIPE_ALL>();

    if constexpr (OP == OpType::Get) {
        __gm__ T* seg_base = reinterpret_cast<__gm__ T*>(
            reinterpret_cast<__gm__ int8_t*>(sym_addr) + my_block_id * invp.logical_segment_bytes);
        copy_ub2gm(seg_base, ub_buf, invp.per_invocation_bytes);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Host-side test execution
int32_t execute_test_loop(const Config& config, aclrtStream stream)
{
    int npes = config.npes;
    int mype = config.mype;
    std::vector<std::vector<std::string>> csv_data = {
        {"DataSize/B", "Npus", "Blocks", "UBsize/elements", "Bandwidth/GB/s (1000)", "Bandwidth/GiB/s (1024)",
         "CoreMaxTime/us", "SingleCoreTime/us"},
    };

    constexpr size_t per_core_bytes = InvokeParameters::per_core_bytes;
    size_t max_bytes = per_core_bytes * config.block_size_max;

    void* host_mem = nullptr;
    void* device_mem = nullptr;
    OpType op_type = config.req_op_type.value_or(OpType::Get);
    PerfDataPrepValidator pdpv(mype, op_type);

    ACL_CHECK_WITH_RET(aclrtMallocHost(&host_mem, max_bytes), ERROR_LOG("Failed to allocate host memory"), return -1);
    device_mem = aclshmem_malloc(max_bytes);
    ACL_CHECK_WITH_RET(device_mem == nullptr, ERROR_LOG("Failed to allocate device memory"), {
        aclrtFreeHost(host_mem);
        return -1;
    });

    bool ever_failed = false;
    // Each (core count, transfer size) pair gets its own profiling frame. The frame
    // slot must not be reused: SHMEMI_PROF_START/END accumulate into
    // cycles[frame_id] / ccount[frame_id] rather than overwriting them, so a second
    // measurement landing on the same frame would be averaged together with the
    // first. Hence a monotonically increasing id across both loops.
    int32_t frame_id = 0;
    for (int32_t block : config.block_sizes) {
        size_t used_bytes = per_core_bytes * block;
        for (int32_t i = 0; i <= config.bytes_in_exp_max - config.bytes_in_exp_min; i++) {
            int32_t exp = config.bytes_in_exp_min + i;

            pdpv.init_data(host_mem, used_bytes);
            InvokeParameters invp(exp, WARMUP_LOOPS, config.loop_count);

            // Check if transfer size exceeds UB buffer
            size_t count = invp.per_invocation_bytes / sizeof(T);
            if (count > UB_BUFFER_SIZE) {
                printf(
                    "[WARN] Transfer size %d exceeds UB buffer size %zu, skipping\n", invp.per_invocation_bytes,
                    UB_BUFFER_SIZE * sizeof(T));
                continue;
            }

            if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
                printf(
                    "[ERROR] Ran out of profiling frames (limit %d); remaining combinations are skipped.\n",
                    ACLSHMEM_CYCLE_PROF_FRAME_CNT);
                ever_failed = true;
                break;
            }

            ACL_CHECK(aclrtMemcpy(device_mem, used_bytes, host_mem, used_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

            aclshmem_barrier_all();
            if (op_type == OpType::Put) {
                run_demo_mem_ub2gm<OpType::Put><<<block, 0, stream>>>(device_mem, invp, frame_id);
            } else {
                run_demo_mem_ub2gm<OpType::Get><<<block, 0, stream>>>(device_mem, invp, frame_id);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            aclshmem_barrier_all();

            ACL_CHECK(aclrtMemcpy(host_mem, used_bytes, device_mem, used_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

            size_t check_bytes = (op_type == OpType::Put) ? invp.logical_segment_bytes : invp.per_invocation_bytes;
            DataMismatch mm = pdpv.check_data(host_mem, check_bytes, block, invp.logical_segment_bytes);
            ever_failed = ever_failed || (!mm.ok);
            if (!mm.ok) {
                ERROR_LOG(
                    "Data validation FAILED on PE %d (%s), op=%s, data_size=%d bits, threads=%d\n"
                    "          test point: blocks=%d, transfer=2^%d=%d B/invocation, "
                    "logical_segment=%d B, used_bytes=%zu, checked=%zu B/block\n"
                    "          ub_buffer=%d elements, loops=%d (+%d warmup), frame_id=%d\n"
                    "          first bad byte: block=%d, offset_in_block=%zu, absolute_offset=%zu, "
                    "expected=%d, actual=%d\n"
                    "          total mismatched bytes: %llu of %zu checked",
                    mype, (mype == ACTIVE_PE ? "ACTIVE" : "PASSIVE"), to_string(op_type).c_str(), DATA_SIZE,
                    THREAD_COUNT, block, exp, invp.per_invocation_bytes, invp.logical_segment_bytes, used_bytes,
                    check_bytes, UB_BUFFER_SIZE, config.loop_count, WARMUP_LOOPS, frame_id, mm.block,
                    mm.offset_in_block, mm.absolute_offset, static_cast<int>(mm.expected), static_cast<int>(mm.actual),
                    static_cast<unsigned long long>(mm.mismatch_count), check_bytes * block);
            }

            aclshmemx_get_prof(&out_profs, false);
            collect_prof_data_to_csv_v2(
                out_profs, frame_id, static_cast<uint64_t>(invp.per_invocation_bytes), block, config.npes,
                UB_BUFFER_SIZE, config.loop_count, true, csv_data);

            frame_id++;
        }
        if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
            break;
        }
    }

    aclshmem_free(device_mem);
    aclrtFreeHost(host_mem);
    aclshmemx_get_prof(nullptr, true);

    if (config.mype == ACTIVE_PE) {
        const std::vector<int>& blocks = config.block_sizes;
        bool is_contiguous_range =
            !blocks.empty() && blocks.size() == static_cast<size_t>(config.block_size_max - config.block_size_min + 1);
        for (size_t i = 0; is_contiguous_range && i < blocks.size(); ++i) {
            if (blocks[i] != config.block_size_min + static_cast<int>(i)) {
                is_contiguous_range = false;
            }
        }

        std::string block_segment;
        if (is_contiguous_range) {
            block_segment = std::to_string(config.block_size_min) + "-" + std::to_string(config.block_size_max);
        } else {
            for (size_t i = 0; i < blocks.size(); ++i) {
                if (i > 0)
                    block_segment += "_";
                block_segment += std::to_string(blocks[i]);
            }
        }

        std::string csv_filename = "output/ub2gm_" + std::to_string(DATA_SIZE) + "_" + block_segment + "_" +
                                   to_string(op_type) + "_simt_" + std::to_string(config.bytes_in_exp_min) + "-" +
                                   std::to_string(config.bytes_in_exp_max) + "_l" + std::to_string(config.loop_count) +
                                   "_t" + std::to_string(THREAD_COUNT) + "_ub" + std::to_string(UB_BUFFER_SIZE) +
                                   ".csv";
        write_csv(csv_filename, csv_data);
    }

    if (ever_failed) {
        ERROR_LOG("Some tests failed on PE %d; see the per-test-point details above.", mype);
    }
    return ever_failed ? -1 : 0;
}

int32_t run_config_test(const Config& config)
{
    printf("[INFO] Starting test for PE %d of %d\n", config.mype, config.npes);
    int32_t device_id = (config.mype % config.gnpus + config.first_npu);

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));
    ACL_CHECK(aclrtCreateStream(&stream));

    printf("[INFO] Initializing ACLSHMEM...\n");
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(config.mype, config.npes, local_mem_size, config.ipport.c_str(), default_flag_uid, &attributes);

    ACL_CHECK_WITH_RET(
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), ERROR_LOG("aclshmemx_init failed"), {
            aclrtDestroyStream(stream);
            aclrtResetDevice(device_id);
            aclFinalize();
            return -1;
        });

    int32_t ret = execute_test_loop(config, stream);

    printf("[INFO] Cleaning up resources...\n");
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();

    return ret;
}

int32_t main(int argc, char* argv[])
{
    setenv("SHMEM_CYCLE_PROF_PE", "0", 1);

    // parse_args handles --help and reports the offending argument itself.
    auto result = parse_args(argc, argv);
    if (!result) {
        return 1;
    }
    Config config = *result;

    int32_t ret = run_config_test(config);
    return ret;
}
