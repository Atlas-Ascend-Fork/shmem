/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <sys/stat.h>

#include "utils.h"
#include "param.h"
#include "shmem.h"
#include "acl/acl.h"
#include "kernel_operator.h"

#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#define CHECK_RET(func)                                                                 \
    do {                                                                                \
        int ret = func;                                                                 \
        if (ret != 0) {                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: " << ret << std::endl; \
            return ret;                                                                 \
        }                                                                               \
    } while (0)

enum class CMOEXAMPLE : uint32_t {
    NO_PREFETCH,
    HOST_PREFETCH,
    DEVICE_PREFETCH,
    DEVICE_BLOCK_PREFETCH,
};

typedef struct {
    float avg_cmo_send_us;
    float avg_cmo_flag_us;
    float avg_copy_us;
    float total_band;
} res_t;

typedef struct {
    uint32_t loop_times;
    uint32_t copy_size_per_loop;
    uint32_t copypad_size;

    std::vector<float> no_prefetch_us;
    std::vector<float> host_prefetch_us;
    std::vector<float> device_block_prefetch_us;
    std::vector<float> device_block_prefetch_cmo_us;
    std::vector<float> device_block_prefetch_cmo_flag_us;
    std::vector<float> no_prefetch_bands;
    std::vector<float> host_prefetch_bands;
    std::vector<float> device_block_prefetch_bands;
} res_csv_t;

int g_npus = 8;
const char* ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char* data_type = "int";
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;
const uint32_t max_block_nums = 20;
const uint32_t aivs_per_block = 2;
const uint32_t sdma_qp_num = max_block_nums * aivs_per_block;
const uint32_t gm_align_size = 512;
const uint32_t l2_cache_size = 192 * 1024 * 1024;

template <typename T>
__global__ __aicore__ void copy_perftest(
    GM_ADDR trash_gm, GM_ADDR copy_gm, GM_ADDR dump_gm, GM_ADDR res_gm, uint32_t copypad_size, uint32_t copypad_times,
    uint32_t is_block_prefetch, uint32_t use_explicit_qp)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, dump_gm, ALL_DUMPSIZE);
#endif
    AscendC::TPipe pipe;
    uint32_t block_id = AscendC::GetBlockIdx();
    uint32_t n_blocks = AscendC::GetBlockNum();

    uint32_t copy_setp_size = copypad_size > gm_align_size ? copypad_size : gm_align_size;
    uint32_t copy_block_step_size = copy_setp_size * copypad_times;
    uint32_t copy_step_element = copy_setp_size / sizeof(T);
    uint32_t copy_block_step_element = copy_block_step_size / sizeof(T);
    uint32_t cmo_size = copy_block_step_size;

    uint64_t copy_ub;
    AscendC::LocalTensor<T> ub_copy_tensor;
    ub_copy_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_copy_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    AscendC::GlobalTensor<T> gm_tensor;
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(copy_gm));
    gm_tensor = gm_tensor[copy_block_step_element * block_id];

    // Define temporary UB buffer
    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64; // 64B for temporary buffer
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));

    uint64_t start_cycle, end_cycle;
    uint64_t start_cmo_cycle, send_cmo_cycle, end_cmo_cycle;
    uint64_t start_copy_cycle, end_copy_cycle;

    start_cycle = AscendC::GetSystemCycle();
    start_cmo_cycle = start_cycle;
    send_cmo_cycle = start_cmo_cycle;
    end_cmo_cycle = start_cmo_cycle;

    // 不带QP的接口固定使用QP 0，只能由0号AIV单核调用；显式QP接口每个AIV使用自己的QP。
    if (use_explicit_qp != 0 || block_id == 0) {
        __gm__ uint8_t* cmo_target_gm =
            reinterpret_cast<__gm__ uint8_t*>((is_block_prefetch != 0 ? copy_gm : trash_gm) + cmo_size * block_id);
        if (use_explicit_qp != 0) {
            aclshmemx_cmo_qp_nbi(
                cmo_target_gm, cmo_size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, block_id, EVENT_ID0);
            send_cmo_cycle = AscendC::GetSystemCycle();
            aclshmemx_sdma_qp_quiet(tmp_buff, ub_size, block_id, EVENT_ID0);
        } else {
            aclshmemx_cmo_nbi(
                cmo_target_gm, cmo_size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
            send_cmo_cycle = AscendC::GetSystemCycle();
            aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
        }
        end_cmo_cycle = AscendC::GetSystemCycle();
    }

    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopyExtParams data_copy_params(1, copypad_size, 0, 0, 0);
    AscendC::DataCopyPadExtParams<T> pad_params;

    start_copy_cycle = AscendC::GetSystemCycle();

    for (size_t i = 0; i < copypad_times; i++) {
        AscendC::DataCopyPad(ub_copy_tensor, gm_tensor[i * copy_step_element], data_copy_params, pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }

    end_copy_cycle = AscendC::GetSystemCycle();
    end_cycle = end_copy_cycle;

    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    constexpr int64_t cycle2us = 50;
#else
    constexpr int64_t cycle2us = 1000;
#endif
    float cmo_send_us = ((float)(int32_t)(send_cmo_cycle - start_cmo_cycle)) / cycle2us;
    float cmo_flag_us = ((float)(int32_t)(end_cmo_cycle - send_cmo_cycle)) / cycle2us;
    float copy_us = ((float)(int32_t)(end_copy_cycle - start_copy_cycle)) / cycle2us / copypad_times;
    float copy_band = (float)(int32_t)(copypad_size) / copy_us / 1000.0f;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::PRINTF(
        "[AIV=%d] GM->UB, copypad_size:%d Bytes, copypad_times:%d, cmo_send_us: %f us, cmo_flag_us: %f us, copy_us: %f "
        "us, band: %f GBps\n",
        block_id, copypad_size, copypad_times, cmo_send_us, cmo_flag_us, copy_us, copy_band);
#endif

    AscendC::LocalTensor<float> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(tmp_buff);
    ub_tensor.address_.dataLen = ub_size;

    aclshmemi_set_value(
        (__gm__ uint8_t*)(res_gm + (block_id * 10 + 0) * sizeof(float)), cmo_send_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value(
        (__gm__ uint8_t*)(res_gm + (block_id * 10 + 1) * sizeof(float)), cmo_flag_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value((__gm__ uint8_t*)(res_gm + (block_id * 10 + 2) * sizeof(float)), copy_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value(
        (__gm__ uint8_t*)(res_gm + (block_id * 10 + 3) * sizeof(float)), copy_band, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <class T>
void copy_perftest_kernel(
    uint32_t block_dim, void* stream, uint8_t* trash_gm_ptr, uint8_t* cache_gm_ptr, uint8_t* dump_gm_ptr,
    uint8_t* res_gm_ptr, uint32_t move_size_each_time, uint32_t copypad_times, uint32_t is_block_prefetch,
    uint32_t use_explicit_qp)
{
    copy_perftest<T><<<block_dim, nullptr, stream>>>(
        trash_gm_ptr, cache_gm_ptr, dump_gm_ptr, res_gm_ptr, move_size_each_time, copypad_times, is_block_prefetch,
        use_explicit_qp);
}

__global__ __aicore__ void cmo_pretech(GM_ADDR src, uint32_t size)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    // Define temporary UB buffer
    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64; // 64B for temporary buffer
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));

    aclshmemx_cmo_nbi(
        reinterpret_cast<__gm__ uint8_t*>(src), size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
    aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
}

void cmo_pretech_kernel(uint8_t* src, uint32_t size, void* stream) { cmo_pretech<<<1, nullptr, stream>>>(src, size); }

__global__ __aicore__ void cmo_nbi_latency_perftest(GM_ADDR src, GM_ADDR res, uint32_t cmo_size)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64;
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));
    constexpr uint32_t copy_ub_offset = 2048;
    constexpr uint32_t copy_size = 512;
    __ubuf__ uint8_t* copy_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(copy_ub_offset));
    AscendC::LocalTensor<uint8_t> ub_copy_tensor;
    ub_copy_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_copy_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_buff);
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src));

    const uint64_t submit_start = AscendC::GetSystemCycle();
    aclshmemx_cmo_nbi(
        reinterpret_cast<__gm__ uint8_t*>(src), cmo_size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size,
        EVENT_ID0);
    const uint64_t submit_end = AscendC::GetSystemCycle();

    aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
    const uint64_t quiet_end = AscendC::GetSystemCycle();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams data_copy_params(1, copy_size, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> pad_params;
    for (uint32_t offset = 0; offset < cmo_size; offset += copy_size) {
        AscendC::DataCopyPad(ub_copy_tensor, gm_tensor[offset], data_copy_params, pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    constexpr float cycle2us = 50.0f;
#else
    constexpr float cycle2us = 1000.0f;
#endif
    const float submit_us = (float)(int32_t)(submit_end - submit_start) / cycle2us;
    const float execute_us = (float)(int32_t)(quiet_end - submit_start) / cycle2us;

    AscendC::LocalTensor<float> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(tmp_buff);
    ub_tensor.address_.dataLen = ub_size;
    aclshmemi_set_value(reinterpret_cast<__gm__ uint8_t*>(res), submit_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value(reinterpret_cast<__gm__ uint8_t*>(res + sizeof(float)), execute_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
}

void cmo_nbi_latency_perftest_kernel(uint8_t* src, uint8_t* res, uint32_t cmo_size, void* stream)
{
    cmo_nbi_latency_perftest<<<1, nullptr, stream>>>(src, res, cmo_size);
}

int cmo_nbi_latency_test(aclrtStream stream, uint32_t cmo_size, float& submit_us, float& execute_us)
{
    void* src_ptr = nullptr;
    void* res_ptr = nullptr;
    void* res_host = nullptr;
    const size_t src_size = static_cast<size_t>(cmo_size);
    const size_t res_size = sizeof(float) * 2;
    CHECK_RET(aclrtMalloc(&src_ptr, src_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMalloc(&res_ptr, res_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMallocHost(&res_host, res_size));

    cmo_nbi_latency_perftest_kernel(
        reinterpret_cast<uint8_t*>(src_ptr), reinterpret_cast<uint8_t*>(res_ptr), cmo_size, stream);
    CHECK_RET(aclrtSynchronizeStream(stream));
    CHECK_RET(aclrtMemcpy(res_host, res_size, res_ptr, res_size, ACL_MEMCPY_DEVICE_TO_HOST));

    const float* result = reinterpret_cast<const float*>(res_host);
    submit_us = result[0];
    execute_us = result[1];

    CHECK_RET(aclrtFree(src_ptr));
    CHECK_RET(aclrtFree(res_ptr));
    CHECK_RET(aclrtFreeHost(res_host));
    return 0;
}

// Measure the explicit-QP CMO path. GetBlockIdx() is the global AIV index.
__global__ __aicore__ void cmo_qp_latency_perftest(GM_ADDR src, GM_ADDR res, uint32_t cmo_size, uint32_t aiv_num)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    AscendC::TPipe pipe;
    const uint32_t block_id = AscendC::GetBlockIdx();
    if (block_id >= aiv_num) {
        return;
    }

    uint32_t copy_setp_size = cmo_size > gm_align_size ? cmo_size : gm_align_size;
    uint32_t copy_block_step_size = copy_setp_size;
    uint32_t copy_step_element = copy_setp_size / sizeof(uint8_t);
    uint32_t copy_block_step_element = copy_block_step_size / sizeof(uint8_t);
    uint32_t qp_idx = block_id;

    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64;
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));
    __gm__ uint8_t* aiv_src = reinterpret_cast<__gm__ uint8_t*>(src) + copy_block_step_size * block_id;
    constexpr uint32_t copy_ub_offset = 2048;
    constexpr uint32_t copy_size = 512;
    __ubuf__ uint8_t* copy_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(copy_ub_offset));
    AscendC::LocalTensor<uint8_t> ub_copy_tensor;
    ub_copy_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_copy_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_buff);
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src));
    gm_tensor = gm_tensor[copy_block_step_element * block_id];

    const uint64_t submit_start = AscendC::GetSystemCycle();
    aclshmemx_cmo_qp_nbi(aiv_src, cmo_size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, qp_idx, EVENT_ID0);
    const uint64_t submit_end = AscendC::GetSystemCycle();

    aclshmemx_sdma_qp_quiet(tmp_buff, ub_size, qp_idx, EVENT_ID0);
    const uint64_t quiet_end = AscendC::GetSystemCycle();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams data_copy_params(1, copy_size, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> pad_params;
    for (uint32_t offset = 0; offset < cmo_size; offset += copy_size) {
        AscendC::DataCopyPad(ub_copy_tensor, gm_tensor[offset], data_copy_params, pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    constexpr float cycle2us = 50.0f;
#else
    constexpr float cycle2us = 1000.0f;
#endif
    const float submit_us = (float)(int32_t)(submit_end - submit_start) / cycle2us;
    const float execute_us = (float)(int32_t)(quiet_end - submit_start) / cycle2us;

    AscendC::LocalTensor<float> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(tmp_buff);
    ub_tensor.address_.dataLen = ub_size;
    aclshmemi_set_value(
        reinterpret_cast<__gm__ uint8_t*>(res + block_id * 2 * sizeof(float)), submit_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value(
        reinterpret_cast<__gm__ uint8_t*>(res + (block_id * 2 + 1) * sizeof(float)), execute_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
}

void cmo_qp_latency_perftest_kernel(
    uint32_t block_dim, uint8_t* src, uint8_t* res, uint32_t cmo_size, uint32_t aiv_num, void* stream)
{
    cmo_qp_latency_perftest<<<block_dim, nullptr, stream>>>(src, res, cmo_size, aiv_num);
}

int cmo_qp_latency_test(
    aclrtStream stream, uint32_t cmo_size, uint32_t aiv_num, std::vector<float>& submit_us,
    std::vector<float>& execute_us)
{
    void* src_ptr = nullptr;
    void* res_ptr = nullptr;
    void* res_host = nullptr;
    const size_t src_size = static_cast<size_t>(cmo_size) * aiv_num;
    const size_t res_size = sizeof(float) * aiv_num * 2;
    CHECK_RET(aclrtMalloc(&src_ptr, src_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMalloc(&res_ptr, res_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMallocHost(&res_host, res_size));

    cmo_qp_latency_perftest_kernel(
        (aiv_num + aivs_per_block - 1) / aivs_per_block, reinterpret_cast<uint8_t*>(src_ptr),
        reinterpret_cast<uint8_t*>(res_ptr), cmo_size, aiv_num, stream);
    CHECK_RET(aclrtSynchronizeStream(stream));
    CHECK_RET(aclrtMemcpy(res_host, res_size, res_ptr, res_size, ACL_MEMCPY_DEVICE_TO_HOST));

    const float* result = reinterpret_cast<const float*>(res_host);
    submit_us.resize(aiv_num);
    execute_us.resize(aiv_num);
    for (uint32_t i = 0; i < aiv_num; ++i) {
        submit_us[i] = result[i * 2];
        execute_us[i] = result[i * 2 + 1];
    }

    CHECK_RET(aclrtFree(src_ptr));
    CHECK_RET(aclrtFree(res_ptr));
    CHECK_RET(aclrtFreeHost(res_host));
    return 0;
}

template <class T>
int copy_test(
    aclrtStream stream, uint32_t n_blocks, uint32_t copypad_size, uint32_t copypad_times, CMOEXAMPLE prefetch_type,
    res_t& res, uint32_t use_explicit_qp = 1)
{
    uint32_t copy_setp_size = copypad_size > gm_align_size ? copypad_size : gm_align_size;
    uint32_t copy_block_size = copy_setp_size * copypad_times;
    size_t cache_gm_size = n_blocks * aivs_per_block * copy_block_size;

    void* cache_gm_ptr;
    CHECK_RET(aclrtMalloc((void**)&(cache_gm_ptr), cache_gm_size, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t trash_gm_size = cache_gm_size > (size_t)l2_cache_size ? cache_gm_size : (size_t)l2_cache_size;
    void* trash_gm_ptr;
    CHECK_RET(aclrtMalloc((void**)&(trash_gm_ptr), trash_gm_size, ACL_MEM_MALLOC_HUGE_FIRST));

    void* device_dump;
#if defined(ENABLE_ASCENDC_DUMP)
    CHECK_RET(aclrtMalloc(reinterpret_cast<void**>(&device_dump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
#endif
    void* res_ptr;
    size_t res_size = sizeof(float) * 2048;
    CHECK_RET(aclrtMalloc((void**)&(res_ptr), res_size, ACL_MEM_MALLOC_HUGE_FIRST));
    void* res_host;
    CHECK_RET(aclrtMallocHost((void**)(&res_host), res_size));

    uint32_t is_device_block_prefetch = 0;
    if (prefetch_type == CMOEXAMPLE::NO_PREFETCH) {
        is_device_block_prefetch = 0;
    } else if (prefetch_type == CMOEXAMPLE::DEVICE_PREFETCH) {
        cmo_pretech_kernel(reinterpret_cast<uint8_t*>(cache_gm_ptr), cache_gm_size, stream);
        CHECK_RET(aclrtSynchronizeStream(stream));
        aclshmem_barrier_all();
    } else if (prefetch_type == CMOEXAMPLE::DEVICE_BLOCK_PREFETCH) {
        is_device_block_prefetch = 1;
    } else if (prefetch_type == CMOEXAMPLE::HOST_PREFETCH) {
        CHECK_RET(aclrtCmoAsync(cache_gm_ptr, cache_gm_size, ACL_RT_CMO_TYPE_PREFETCH, stream));
        CHECK_RET(aclrtSynchronizeStream(stream));
    }

    copy_perftest_kernel<T>(
        n_blocks, stream, reinterpret_cast<uint8_t*>(trash_gm_ptr), reinterpret_cast<uint8_t*>(cache_gm_ptr),
        reinterpret_cast<uint8_t*>(device_dump), reinterpret_cast<uint8_t*>(res_ptr), copypad_size, copypad_times,
        is_device_block_prefetch, use_explicit_qp);
    CHECK_RET(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();

    aclrtMemcpy(res_host, res_size, res_ptr, res_size, ACL_MEMCPY_DEVICE_TO_HOST);
    float* float_res = reinterpret_cast<float*>(res_host);
    // 不带QP的接口仅由0号AIV下发，其时延只统计0号AIV；拷贝指标仍统计全部AIV。
    const uint32_t cmo_aivs = (use_explicit_qp != 0) ? n_blocks * aivs_per_block : 1;
    float total_cmo_send_us = 0;
    float total_cmo_flag_us = 0;
    float total_copy_us = 0;
    float total_band = 0;
    for (int32_t i = 0; i < n_blocks * aivs_per_block; i++) {
        total_copy_us += float_res[i * 10 + 2];
        total_band += float_res[i * 10 + 3];
    }
    for (int32_t i = 0; i < (int32_t)cmo_aivs; i++) {
        total_cmo_send_us += float_res[i * 10 + 0];
        total_cmo_flag_us += float_res[i * 10 + 1];
    }
    res.avg_cmo_send_us = total_cmo_send_us / cmo_aivs;
    res.avg_cmo_flag_us = total_cmo_flag_us / cmo_aivs;
    res.avg_copy_us = total_copy_us / (n_blocks * aivs_per_block);
    res.total_band = total_band;
#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(device_dump, ALL_DUMPSIZE, stream, "cmo");
#endif
    CHECK_RET(aclrtFree(trash_gm_ptr));
    CHECK_RET(aclrtFree(cache_gm_ptr));
#if defined(ENABLE_ASCENDC_DUMP)
    CHECK_RET(aclrtFree(device_dump));
#endif
    CHECK_RET(aclrtFree(res_ptr));
    CHECK_RET(aclrtFreeHost(res_host));

    return 0;
}

static std::string format_size(int size)
{
    if (size < 1024) {
        return std::to_string(size) + "B";
    } else if (size < 1024 * 1024) {
        size_t kb = size / 1024;
        std::ostringstream oss;
        oss << kb << "KB";
        return oss.str();
    } else {
        double mb = static_cast<double>(size) / (1024.0 * 1024.0);
        size_t mb_int = static_cast<size_t>(mb);
        std::ostringstream oss;
        oss << mb_int << "MB";
        return oss.str();
    }
}

static std::vector<uint32_t> get_cmo_size_vector()
{
    std::vector<uint32_t> cmo_size_vector; // 512B-4MB and 96MB
    for (int exponent = 9; exponent <= 22; ++exponent) {
        cmo_size_vector.push_back(1U << exponent);
    }
    cmo_size_vector.push_back(96U * 1024U * 1024U);
    return cmo_size_vector;
}

static std::string core_column(const char* metric, uint32_t core_idx)
{
    return std::string(metric) + "_core_" + int_to_string(core_idx) + "/us";
}

template <class T>
int test_copy_perf(int my_pe, int n_pes)
{
    uint32_t copy_size_per_loop = 64 * 1024 * 1024; // less than L2 cache size
    uint32_t loop_times = 100;

    // ACLStream init
    aclrtStream stream = nullptr;
    CHECK_RET(aclrtCreateStream(&stream));

    // ===== 不带QP的CMO接口使用样例 =====
    // aclshmemx_cmo_nbi/aclshmemx_sdma_quiet固定使用QP 0，属于单核接口，仅由0号AIV执行
    // （见cmo_pretech内核），这里对一块独立内存做L2预取并等待完成，仅演示基本用法。
    {
        constexpr uint32_t cmo_demo_size = 1 * 1024 * 1024; // 1MB
        void* cmo_demo_ptr = nullptr;
        CHECK_RET(aclrtMalloc(&cmo_demo_ptr, cmo_demo_size, ACL_MEM_MALLOC_HUGE_FIRST));
        cmo_pretech_kernel(reinterpret_cast<uint8_t*>(cmo_demo_ptr), cmo_demo_size, stream);
        CHECK_RET(aclrtSynchronizeStream(stream));
        std::cout << "PE " << my_pe << " cmo_nbi (without QP) demo finished." << std::endl;
        CHECK_RET(aclrtFree(cmo_demo_ptr));
    }

    std::vector<int> copypad_size_vector; // 8B-256KB
    for (int exponent = 3; exponent <= 17; ++exponent) {
        int value = 1 << exponent;
        copypad_size_vector.push_back(value);
    }
    copypad_size_vector.push_back(192 * 1024);
    copypad_size_vector.push_back(256 * 1024);

    auto percentile = [](const std::vector<float>& v, float p) -> float {
        if (v.empty())
            return 0.0f;
        std::vector<float> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = (size_t)(p / 100.0 * sorted.size());
        return sorted[idx];
    };

    std::vector<std::vector<std::string>> csv_data = {
        {"loop_times", "copy_size_per_loop", "blocks", "copypad_size", "no_prefetch_time/us", "no_prefetch_band/Gbps",
         "host_prefetch_time/us", "host_prefetch_band/Gbps", "device_block_prefetch_time/us",
         "device_block_prefetch_band/Gbps"},
    };

    for (uint32_t n_blocks = 20; n_blocks <= max_block_nums; n_blocks++) {
        uint32_t aiv_num = n_blocks * aivs_per_block;
        for (uint32_t copypad_size : copypad_size_vector) {
            uint32_t copypad_setp_size = copypad_size > gm_align_size ? copypad_size : gm_align_size;
            uint32_t copypad_times = copy_size_per_loop / aiv_num / copypad_setp_size;
            uint32_t copy_block_size = copypad_setp_size * copypad_times;
            size_t cache_gm_size = aiv_num * copy_block_size;

            res_csv_t res_csv = {};
            std::vector<std::string> sub_data = {
                int_to_string(loop_times), format_size(copy_size_per_loop), int_to_string(n_blocks),
                format_size(copypad_size)};

            res_t res = {};
            // cmo warmup
            copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::NO_PREFETCH, res);

            // no prefetch
            for (uint32_t loop_i = 0; loop_i < loop_times; loop_i++) {
                copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::NO_PREFETCH, res);
                res_csv.no_prefetch_bands.push_back(res.total_band);
                res_csv.no_prefetch_us.push_back(res.avg_copy_us);
            }
            sub_data.push_back(float_to_string(percentile(res_csv.no_prefetch_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.no_prefetch_bands, 50.0f)));

            // host prefetch
            for (uint32_t loop_i = 0; loop_i < loop_times; loop_i++) {
                copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::HOST_PREFETCH, res);
                res_csv.host_prefetch_bands.push_back(res.total_band);
                res_csv.host_prefetch_us.push_back(res.avg_copy_us);
            }
            sub_data.push_back(float_to_string(percentile(res_csv.host_prefetch_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.host_prefetch_bands, 50.0f)));

            // device block prefetch
            for (uint32_t loop_i = 0; loop_i < loop_times; loop_i++) {
                copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::DEVICE_BLOCK_PREFETCH, res);
                res_csv.device_block_prefetch_bands.push_back(res.total_band);
                res_csv.device_block_prefetch_us.push_back(res.avg_copy_us);
            }
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_bands, 50.0f)));

            csv_data.push_back(sub_data);
        }
    }
    make_dir("output");
    write_csv("output/" + int_to_string(my_pe) + "_band.csv", csv_data);

    std::vector<std::vector<std::string>> csv_data_nbi = {
        {"loop_times", "blocks", "cmo_size", "cmo_submit_time_p05/us", "cmo_submit_time_p50/us",
         "cmo_submit_time_p95/us", "cmo_execute_time_p05/us", "cmo_execute_time_p50/us", "cmo_execute_time_p95/us"},
    };
    const std::vector<uint32_t> cmo_perf_size_vector = get_cmo_size_vector();
    const uint32_t nbi_blocks = 1;

    std::vector<std::vector<std::string>> csv_data_qp = {
        {"loop_times", "aiv_num", "cmo_size", "cmo_qp_submit_time_avg/us", "cmo_qp_submit_time_max/us",
         "cmo_qp_execute_time_avg/us", "cmo_qp_execute_time_max/us", "cmo_qp_submit_time_p05/us",
         "cmo_qp_submit_time_p50/us", "cmo_qp_submit_time_p95/us", "cmo_qp_execute_time_p05/us",
         "cmo_qp_execute_time_p50/us", "cmo_qp_execute_time_p95/us"},
    };
    for (uint32_t core_idx = 0; core_idx < sdma_qp_num; ++core_idx) {
        csv_data_qp[0].push_back(core_column("cmo_qp_submit_time_p05", core_idx));
        csv_data_qp[0].push_back(core_column("cmo_qp_submit_time_p50", core_idx));
        csv_data_qp[0].push_back(core_column("cmo_qp_submit_time_p95", core_idx));
        csv_data_qp[0].push_back(core_column("cmo_qp_execute_time_p05", core_idx));
        csv_data_qp[0].push_back(core_column("cmo_qp_execute_time_p50", core_idx));
        csv_data_qp[0].push_back(core_column("cmo_qp_execute_time_p95", core_idx));
    }

    const std::vector<uint32_t> qp_aiv_nums = {1U, 2U, 4U, 8U, 16U, 32U, sdma_qp_num};
    // 单核和多核测试使用同一组预取大小，保证横向对比口径一致。

    for (uint32_t cmo_size : cmo_perf_size_vector) {
        uint32_t copypad_size = 512;
        uint32_t copypad_times = cmo_size / copypad_size;
        std::vector<float> nbi_submit_times;
        std::vector<float> nbi_execute_times;
        res_t res = {};

        CHECK_RET(
            copy_test<T>(stream, nbi_blocks, copypad_size, copypad_times, CMOEXAMPLE::DEVICE_BLOCK_PREFETCH, res, 0));
        for (uint32_t loop_i = 0; loop_i < loop_times; ++loop_i) {
            CHECK_RET(copy_test<T>(
                stream, nbi_blocks, copypad_size, copypad_times, CMOEXAMPLE::DEVICE_BLOCK_PREFETCH, res, 0));
            nbi_submit_times.push_back(res.avg_cmo_send_us);
            nbi_execute_times.push_back(res.avg_cmo_flag_us);
        }

        csv_data_nbi.push_back({
            int_to_string(loop_times),
            int_to_string(nbi_blocks),
            format_size(cmo_size),
            float_to_string(percentile(nbi_submit_times, 5.0f)),
            float_to_string(percentile(nbi_submit_times, 50.0f)),
            float_to_string(percentile(nbi_submit_times, 95.0f)),
            float_to_string(percentile(nbi_execute_times, 5.0f)),
            float_to_string(percentile(nbi_execute_times, 50.0f)),
            float_to_string(percentile(nbi_execute_times, 95.0f)),
        });

        for (uint32_t aiv_num : qp_aiv_nums) {
            std::vector<std::vector<float>> qp_submit_samples(aiv_num);
            std::vector<std::vector<float>> qp_execute_samples(aiv_num);

            std::vector<float> qp_submit_once;
            std::vector<float> qp_execute_once;
            CHECK_RET(cmo_qp_latency_test(stream, cmo_size, aiv_num, qp_submit_once, qp_execute_once));
            for (uint32_t loop_i = 0; loop_i < loop_times; ++loop_i) {
                CHECK_RET(cmo_qp_latency_test(stream, cmo_size, aiv_num, qp_submit_once, qp_execute_once));
                for (uint32_t core_idx = 0; core_idx < aiv_num; ++core_idx) {
                    qp_submit_samples[core_idx].push_back(qp_submit_once[core_idx]);
                    qp_execute_samples[core_idx].push_back(qp_execute_once[core_idx]);
                }
            }

            std::vector<float> qp_submit_medians;
            std::vector<float> qp_execute_medians;
            qp_submit_medians.reserve(aiv_num);
            qp_execute_medians.reserve(aiv_num);
            for (uint32_t core_idx = 0; core_idx < sdma_qp_num; ++core_idx) {
                if (core_idx < aiv_num) {
                    float submit_med = percentile(qp_submit_samples[core_idx], 50.0f);
                    float execute_med = percentile(qp_execute_samples[core_idx], 50.0f);
                    qp_submit_medians.push_back(submit_med);
                    qp_execute_medians.push_back(execute_med);
                }
            }
            float submit_avg = std::accumulate(qp_submit_medians.begin(), qp_submit_medians.end(), 0.0f) /
                               static_cast<float>(qp_submit_medians.size());
            float execute_avg = std::accumulate(qp_execute_medians.begin(), qp_execute_medians.end(), 0.0f) /
                                static_cast<float>(qp_execute_medians.size());
            float submit_max = *std::max_element(qp_submit_medians.begin(), qp_submit_medians.end());
            float execute_max = *std::max_element(qp_execute_medians.begin(), qp_execute_medians.end());
            std::vector<std::string> sub_data = {
                int_to_string(loop_times),
                int_to_string(aiv_num),
                format_size(cmo_size),
                float_to_string(submit_avg),
                float_to_string(submit_max),
                float_to_string(execute_avg),
                float_to_string(execute_max),
                float_to_string(percentile(qp_submit_medians, 5.0f)),
                float_to_string(percentile(qp_submit_medians, 50.0f)),
                float_to_string(percentile(qp_submit_medians, 95.0f)),
                float_to_string(percentile(qp_execute_medians, 5.0f)),
                float_to_string(percentile(qp_execute_medians, 50.0f)),
                float_to_string(percentile(qp_execute_medians, 95.0f))};
            sub_data.reserve(csv_data_qp[0].size());
            for (uint32_t core_idx = 0; core_idx < sdma_qp_num; ++core_idx) {
                if (core_idx < aiv_num) {
                    sub_data.push_back(float_to_string(percentile(qp_submit_samples[core_idx], 5.0f)));
                    sub_data.push_back(float_to_string(percentile(qp_submit_samples[core_idx], 50.0f)));
                    sub_data.push_back(float_to_string(percentile(qp_submit_samples[core_idx], 95.0f)));
                    sub_data.push_back(float_to_string(percentile(qp_execute_samples[core_idx], 5.0f)));
                    sub_data.push_back(float_to_string(percentile(qp_execute_samples[core_idx], 50.0f)));
                    sub_data.push_back(float_to_string(percentile(qp_execute_samples[core_idx], 95.0f)));
                } else {
                    sub_data.push_back("N/A");
                    sub_data.push_back("N/A");
                    sub_data.push_back("N/A");
                    sub_data.push_back("N/A");
                    sub_data.push_back("N/A");
                    sub_data.push_back("N/A");
                }
            }
            csv_data_qp.push_back(sub_data);
        }
    }

    write_csv("output/" + int_to_string(my_pe) + "_cmo_nbi.csv", csv_data_nbi);
    write_csv("output/" + int_to_string(my_pe) + "_cmo_qp.csv", csv_data_qp);
    std::cout << "PE " << my_pe << " Finished !" << std::endl;

    CHECK_RET(aclrtDestroyStream(stream));
    return 0;
}

int32_t test_set_attr(
    int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char* ip_port, aclshmemx_init_attr_t* attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void*>(&default_flag_uid);
    aclshmemx_uniqueid_t* uid_args = (aclshmemx_uniqueid_t*)(attributes->comm_args);
    uid_args->my_pe = my_pe;
    uid_args->n_pes = n_pes;
    return ACLSHMEM_SUCCESS;
}

int main(int argc, char* argv[])
{
    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int my_pe = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = atoi(argv[INDEX4]);
    f_pe = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];

    // Acl && Shmem init
    int32_t device_id = my_pe % g_npus + f_npu;
    CHECK_RET(aclInit(nullptr));
    CHECK_RET(aclrtSetDevice(device_id));

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    CHECK_RET(test_set_attr(my_pe, n_pes, local_mem_size, ipport, &attributes));

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
    CHECK_RET(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, sdma_qp_num));
    CHECK_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    if (std::string(data_type) == "int") {
        CHECK_RET(test_copy_perf<int>(my_pe, n_pes));
    } else if (std::string(data_type) == "uint8") {
        CHECK_RET(test_copy_perf<uint8_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "int64") {
        CHECK_RET(test_copy_perf<int64_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "fp16") {
        CHECK_RET(test_copy_perf<half>(my_pe, n_pes));
    } else if (std::string(data_type) == "fp32") {
        CHECK_RET(test_copy_perf<float>(my_pe, n_pes));
    } else {
        printf("ERROR: Unsupported type\n");
        return -1;
    }

    CHECK_RET(aclshmem_finalize());
    CHECK_RET(aclrtResetDevice(device_id));
    CHECK_RET(aclFinalize());

    std::cout << "[SUCCESS] demo run success in pe " << my_pe << std::endl;
    return 0;
}
