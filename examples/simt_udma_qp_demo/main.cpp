/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * SIMT UDMA QP-specific multi-QP demo.
 *
 * Each PE communicates with the next PE in a ring (pes processes form a ring):
 *   Put: local origin -> next PE's res_put (each QP handles one data slice)
 *   Get: next PE's origin -> local res_get (each QP handles one data slice)
 *
 * Two execution modes, one VF per mode, selected at runtime via -mode:
 *   warp:   VF dim3(qp_count * 32) - warp i drives QP i (warp-collective call;
 *           threads within each warp cooperatively submit the WQE; all threads
 *           in the VF must call the API).
 *   block:  kernel grid <<<qp_count>>> - block i drives QP i via blockIdx.x
 *           (VF dim3(128) per block; only warp 0 assembles and submits the WQE,
 *           the remaining warps are inactive; all 128 threads must call the
 *           API so the quiet barrier is reached).
 *
 * Host must configure the same QP count on every PE via aclshmemx_set_qp_num
 * (ACLSHMEM_DATA_OP_UDMA, qp_count) before aclshmemx_init_attr.
 */

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "../utils/utils.h"
#include "simt_runtime.h"

aclshmemx_uniqueid_t default_flag_uid;

constexpr uint64_t MAX_ELEMS = 256ULL * 1024ULL * 1024ULL;
constexpr uint32_t BLOCK_VF_THREADS = 128;

enum class DemoMode : int32_t {
    WARP = 0,
    BLOCK = 1,
};

/*
 * warp mode: warp i drives QP i.
 * Every thread in the VF must call the API so that quiet is reached. The warp
 * quiet variant relies on warp lockstep (no block-level barrier); all threads
 * in the VF must still call it for collective correctness. The VF is launched
 * with exactly qp_count * 32 threads, so every warp has a QP to drive.
 */
__simt_vf__ __launch_bounds__(ACLSHMEM_MAX_QP_NUM * 32) inline void demo_warp_vf(
    __gm__ uint8_t* origin, __gm__ uint8_t* res_put, __gm__ uint8_t* res_get, uint64_t elements, int32_t peer,
    uint32_t qp_count)
{
    const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const uint32_t warp_id = tid / warpSize;
    const uint64_t base = elements / qp_count;
    const uint64_t remainder = elements % qp_count;
    const uint64_t slice_len = base + (warp_id < remainder ? 1 : 0);
    const uint64_t slice_off = static_cast<uint64_t>(warp_id) * base + (warp_id < remainder ? warp_id : remainder);

    simt::aclshmemx_udma_qp_put_nbi_warp<uint8_t>(
        res_put + slice_off, origin + slice_off, static_cast<uint32_t>(slice_len), peer, warp_id);
    simt::aclshmemx_udma_qp_quiet_warp(peer, warp_id);

    simt::aclshmemx_udma_qp_get_nbi_warp<uint8_t>(
        res_get + slice_off, origin + slice_off, static_cast<uint32_t>(slice_len), peer, warp_id);
    simt::aclshmemx_udma_qp_quiet_warp(peer, warp_id);
}

/*
 * block mode: each kernel block drives the QP given by blockIdx.x. blockIdx is
 * SIMT thread/block state, so it is read inside the VF instead of being passed
 * from the kernel entry.
 * The quiet API is block-collective (uses asc_syncthreads internally), so all
 * threads in the VF must reach the quiet API's internal barrier. Only warp 0
 * assembles and submits the WQE; the remaining warps are inactive but must
 * still call the API to reach the quiet barrier.
 */
__simt_vf__ __launch_bounds__(BLOCK_VF_THREADS) inline void demo_block_vf(
    __gm__ uint8_t* origin, __gm__ uint8_t* res_put, __gm__ uint8_t* res_get, uint64_t elements, int32_t peer,
    uint32_t qp_count)
{
    const uint32_t qp_idx = blockIdx.x;
    const uint64_t base = elements / qp_count;
    const uint64_t remainder = elements % qp_count;
    const uint64_t slice_len = base + (qp_idx < remainder ? 1 : 0);
    const uint64_t slice_off = static_cast<uint64_t>(qp_idx) * base + (qp_idx < remainder ? qp_idx : remainder);

    simt::aclshmemx_udma_qp_put_nbi_block<uint8_t>(
        res_put + slice_off, origin + slice_off, static_cast<uint32_t>(slice_len), peer, qp_idx);
    simt::aclshmemx_udma_qp_quiet_block(peer, qp_idx);

    simt::aclshmemx_udma_qp_get_nbi_block<uint8_t>(
        res_get + slice_off, origin + slice_off, static_cast<uint32_t>(slice_len), peer, qp_idx);
    simt::aclshmemx_udma_qp_quiet_block(peer, qp_idx);
}

/*
 * Kernel entry: dispatches to the selected VF.
 * Grid dimension depends on mode:
 *   warp: grid = 1 (one kernel block; VF warps handle QPs)
 *   block: grid = qp_count (each kernel block drives QP blockIdx.x, read inside the VF)
 */
__global__ __vector__ void demo_kernel(
    __gm__ uint8_t* origin, __gm__ uint8_t* res_put, __gm__ uint8_t* res_get, uint64_t elements, int32_t peer,
    uint32_t qp_count, int32_t mode)
{
    if (mode == static_cast<int32_t>(DemoMode::WARP)) {
        asc_vf_call<demo_warp_vf>(dim3(qp_count * 32), origin, res_put, res_get, elements, peer, qp_count);
    } else {
        asc_vf_call<demo_block_vf>(dim3(BLOCK_VF_THREADS), origin, res_put, res_get, elements, peer, qp_count);
    }
}

uint8_t Pattern(int pe, uint64_t index)
{
    return static_cast<uint8_t>((static_cast<uint64_t>(pe + 1) * 37 + index % 251) & 0xFF);
}

int RunDemo(
    void* stream, uint8_t* origin, uint8_t* res_put, uint8_t* res_get, uint64_t elements, int peer, uint32_t qp_count,
    DemoMode mode)
{
    const uint32_t grid = (mode == DemoMode::BLOCK) ? qp_count : 1;
    demo_kernel<<<grid, 0, stream>>>(origin, res_put, res_get, elements, peer, qp_count, static_cast<int32_t>(mode));
    return aclrtSynchronizeStream(stream);
}

struct Options {
    int pe{0};
    int pes{2};
    int first_npu{0};
    uint32_t qp_count{2};
    uint64_t elements{1024 * 1024};
    uint64_t heap_mb{1024};
    std::string mode{"all"};
    std::string ipport{"tcp://127.0.0.1:8999"};
};

static bool ParseOptions(int argc, char* argv[], Options& options)
{
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const char* value = argv[i + 1];
        try {
            if (key == "-pe") {
                options.pe = std::stoi(value);
            } else if (key == "-pes") {
                options.pes = std::stoi(value);
            } else if (key == "-mode") {
                options.mode = value;
            } else if (key == "-ipport") {
                options.ipport = value;
            } else if (key == "-qp_count") {
                long v = std::stol(value);
                if (v <= 0 || v > ACLSHMEM_MAX_QP_NUM) {
                    return false;
                }
                options.qp_count = static_cast<uint32_t>(v);
            } else if (key == "-elems") {
                unsigned long long v = std::stoull(value);
                if (v == 0 || v > MAX_ELEMS) {
                    return false;
                }
                options.elements = v;
            } else if (key == "-heap_mb") {
                unsigned long long v = std::stoull(value);
                if (v == 0 || v > 4096) {
                    return false;
                }
                options.heap_mb = v;
            } else if (key == "-first_npu") {
                long v = std::stol(value);
                if (v < 0 || v > std::numeric_limits<int>::max()) {
                    return false;
                }
                options.first_npu = static_cast<int>(v);
            } else {
                return false;
            }
        } catch (const std::exception&) {
            return false;
        }
    }
    if (options.pes < 2 || options.pe < 0 || options.pe >= options.pes) {
        return false;
    }
    if (options.mode != "warp" && options.mode != "block" && options.mode != "all") {
        return false;
    }
    if (options.elements < 2 * options.qp_count) {
        return false;
    }
    return true;
}

static bool Validate(const std::vector<uint8_t>& actual, int expected_pe, const std::string& tag, int pe)
{
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const uint8_t expected = Pattern(expected_pe, i);
        if (actual[i] != expected) {
            printf(
                "[FAIL] %s pe=%d index=%llu actual=%u expected=%u\n", tag.c_str(), pe,
                static_cast<unsigned long long>(i), static_cast<uint32_t>(actual[i]), static_cast<uint32_t>(expected));
            return false;
        }
    }
    printf("[PASS] %s pe=%d elements=%llu\n", tag.c_str(), pe, static_cast<unsigned long long>(actual.size()));
    return true;
}

static bool RunOneMode(
    DemoMode mode, const Options& options, RuntimeResources& resources, uint8_t* origin, uint8_t* res_put,
    uint8_t* res_get, bool buffers_ok)
{
    const std::string mode_name = (mode == DemoMode::WARP) ? "warp" : "block";
    const uint64_t data_bytes = options.elements;
    const int next_pe = (options.pe + 1) % options.pes;
    const int prev_pe = (options.pe == 0) ? (options.pes - 1) : (options.pe - 1);

    // Every PE runs the same collective sequence. A PE whose buffers are unusable skips only
    // the kernel and the validation, never a barrier, so the peer cannot block here.
    bool passed = buffers_ok;

    std::vector<uint8_t> host(data_bytes);
    if (passed) {
        for (uint64_t i = 0; i < options.elements; ++i) {
            host[i] = Pattern(options.pe, i);
        }
        if (aclrtMemcpy(origin, data_bytes, host.data(), data_bytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0 ||
            aclrtMemset(res_put, data_bytes, 0, data_bytes) != 0 ||
            aclrtMemset(res_get, data_bytes, 0, data_bytes) != 0) {
            ERROR_LOG("%s prepare buffers failed", mode_name.c_str());
            passed = false;
        }
    }

    aclshmem_barrier_all();
    if (passed) {
        const int ret =
            RunDemo(resources.stream, origin, res_put, res_get, options.elements, next_pe, options.qp_count, mode);
        if (ret != 0) {
            ERROR_LOG("%s run kernel failed, ret=%d, errMsg: %s", mode_name.c_str(), ret, aclGetRecentErrMsg());
            passed = false;
        }
    }
    aclshmem_barrier_all();

    if (passed) {
        if (aclrtMemcpy(host.data(), data_bytes, res_put, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0 ||
            !Validate(host, prev_pe, mode_name + "-put", options.pe)) {
            passed = false;
        }
        if (aclrtMemcpy(host.data(), data_bytes, res_get, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0 ||
            !Validate(host, next_pe, mode_name + "-get", options.pe)) {
            passed = false;
        }
    }

    return passed;
}

int main(int argc, char* argv[])
{
    Options options;
    if (!ParseOptions(argc, argv, options)) {
        printf(
            "Usage: %s -pe ID -pes N [-mode warp|block|all] [-qp_count N] [-elems N]"
            " [-heap_mb N] [-first_npu ID] [-ipport tcp://IP:PORT]\n",
            argv[0]);
        return 1;
    }

    if (options.elements > std::numeric_limits<size_t>::max() ||
        options.heap_mb > std::numeric_limits<uint64_t>::max() / (1024ULL * 1024ULL)) {
        ERROR_LOG("Size argument overflows this process");
        return 1;
    }
    const uint64_t heap_bytes = options.heap_mb * 1024ULL * 1024ULL;
    const uint64_t required_bytes = options.elements * 3;
    if (required_bytes > heap_bytes) {
        ERROR_LOG("-heap_mb must cover three symmetric buffers (origin + res_put + res_get)");
        return 1;
    }

    if (options.first_npu > std::numeric_limits<int>::max() - options.pe) {
        ERROR_LOG("Resolved logical device ID exceeds the supported range");
        return 1;
    }
    RuntimeResources resources;
    resources.device_id = options.first_npu + options.pe;

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

    INFO_LOG(
        "pe=%d pes=%d first_npu=%d qp_count=%u elements=%llu mode=%s", options.pe, options.pes, options.first_npu,
        options.qp_count, static_cast<unsigned long long>(options.elements), options.mode.c_str());

    // Allocate once and share the three symmetric buffers between the modes: this keeps the
    // heap requirement at 3x elements and gives every PE the same allocation sequence. The
    // heap is released by aclshmem_finalize() in CleanupResources(), so a PE that fails to
    // allocate does not have to run a different number of frees than its peer.
    uint8_t* origin = static_cast<uint8_t*>(aclshmemx_malloc(options.elements));
    uint8_t* res_put = static_cast<uint8_t*>(aclshmemx_malloc(options.elements));
    uint8_t* res_get = static_cast<uint8_t*>(aclshmemx_malloc(options.elements));
    const bool buffers_ok = (origin != nullptr && res_put != nullptr && res_get != nullptr);
    if (!buffers_ok) {
        ERROR_LOG("aclshmemx_malloc failed");
    }

    bool passed = true;
    if (options.mode == "all" || options.mode == "warp") {
        passed = RunOneMode(DemoMode::WARP, options, resources, origin, res_put, res_get, buffers_ok) && passed;
    }
    if (options.mode == "all" || options.mode == "block") {
        passed = RunOneMode(DemoMode::BLOCK, options, resources, origin, res_put, res_get, buffers_ok) && passed;
    }

    status = CleanupResources(resources, 0);
    const int ret = (status == 0 && passed) ? 0 : 1;
    printf(
        "[INFO] mode=%s qp_count=%u pe=%d result=%s\n", options.mode.c_str(), options.qp_count, options.pe,
        ret == 0 ? "PASS" : "FAIL");
    return ret;
}
