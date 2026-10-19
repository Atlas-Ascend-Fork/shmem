/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "../utils/utils.h"

const char* ipport = "tcp://127.0.0.1:8998";
aclshmemx_uniqueid_t default_flag_uid;

// 定义传输大小(传输COPY_SIZE个int32_t的数据)
constexpr int32_t COPY_SIZE = 4096;

// CAS 演示参数：32 个线程并发执行，槽位初值/替换值与计数器自增轮数
constexpr uint32_t AMO_THREADS = 32;
constexpr uint32_t AMO_ITERS = 1000;
constexpr uint32_t AMO_SLOT_INIT = 0xA5A5A5A5u;
constexpr uint32_t AMO_SLOT_NEW = 0x5A5A5A5Au;

// CAS 演示对称堆布局（uint32_t 字为单位）：槽位、结果、计数器各占独立 64B cacheline，
// 避免同一行上多个原子目标互相干扰
constexpr uint32_t AMO_SLOT_OFF = 0;
constexpr uint32_t AMO_RESULT_OFF = AMO_SLOT_OFF + AMO_THREADS;
constexpr uint32_t AMO_COUNTER_OFF = (AMO_RESULT_OFF + 2 * AMO_THREADS + 15U) & ~15U;
constexpr uint32_t AMO_HEAP_WORDS = AMO_COUNTER_OFF + 16;

// 每个函数都做两件事：
//   get: 把上一个 PE 的 origin 拉取到自身 res_prev
//   put: 把自身 origin 推送到上一个 PE 的 res_next
// put 的目标之所以是 prev_pe 而非 next_pe：对 prev_pe 而言当前 PE 正是它的“下一个 PE”，
// 这样每个 PE 的 res_next 最终存放的就是它下一个 PE 的 origin 数据
__simt_callee__ inline void test_put_get_mem(
    __gm__ int32_t* origin, __gm__ int32_t* res_prev, __gm__ int32_t* res_next, int32_t prev_pe)
{
    simt::aclshmemx_getmem_warp((__gm__ void*)res_prev, (__gm__ void*)origin, COPY_SIZE * sizeof(int32_t), prev_pe);
    simt::aclshmemx_putmem_warp((__gm__ void*)res_next, (__gm__ void*)origin, COPY_SIZE * sizeof(int32_t), prev_pe);
}

__simt_callee__ inline void test_put_get_type(
    __gm__ int32_t* origin, __gm__ int32_t* res_prev, __gm__ int32_t* res_next, int32_t prev_pe)
{
    simt::aclshmemx_int16_get_warp(
        (__gm__ int16_t*)res_prev, (__gm__ int16_t*)origin, COPY_SIZE * sizeof(int32_t) / sizeof(int16_t), prev_pe);
    simt::aclshmemx_int16_put_warp(
        (__gm__ int16_t*)res_next, (__gm__ int16_t*)origin, COPY_SIZE * sizeof(int32_t) / sizeof(int16_t), prev_pe);
}

__simt_callee__ inline void test_put_get_bits(
    __gm__ int32_t* origin, __gm__ int32_t* res_prev, __gm__ int32_t* res_next, int32_t prev_pe)
{
    simt::aclshmemx_get128_warp((__gm__ void*)res_prev, (__gm__ void*)origin, COPY_SIZE * 32 / 128, prev_pe);
    simt::aclshmemx_put128_warp((__gm__ void*)res_next, (__gm__ void*)origin, COPY_SIZE * 32 / 128, prev_pe);
}

__simt_vf__ __launch_bounds__(1024) inline void demo_call_simt(
    __gm__ int32_t* origin, __gm__ int32_t* res_prev, __gm__ int32_t* res_next, __gm__ uint64_t* dbg)
{
    int32_t mype = simt::aclshmem_my_pe();
    int32_t npes = simt::aclshmem_n_pes();

    int32_t prev_pe = (mype - 1 + npes) % npes;

    test_put_get_bits(origin, res_prev, res_next, prev_pe);
}

__global__ __vector__ void demo_call(
    __gm__ int32_t* origin, __gm__ int32_t* res_prev, __gm__ int32_t* res_next, __gm__ uint64_t* dbg)
{
    asc_vf_call<demo_call_simt>(dim3(32), origin, res_prev, res_next, dbg);
}

void run_demo_mem(void* stream, int32_t* origin, int32_t* res_prev, int32_t* res_next, uint64_t* dbg)
{
    demo_call<<<1, 0, stream>>>(origin, res_prev, res_next, dbg);
}

// CAS 演示内核：每个线程对自身槽位先后执行命中/未命中两次 CAS（校验返回值与改写语义），
// 再通过 CAS 重试循环（读旧值 → CAS 自增 → 失败重试）并发自增计数器：
// selfCounter（本 PE）覆盖同卡多线程竞争，globalCounter（PE0）覆盖跨 PE 竞争
__simt_vf__ __launch_bounds__(1024) inline void demo_atomic_cas_simt(
    __gm__ uint32_t* slots, __gm__ uint32_t* results, __gm__ uint32_t* counters)
{
    const int32_t mype = simt::aclshmem_my_pe();
    const uint32_t tid = threadIdx.x;

    const uint32_t oldMatch = simt::aclshmem_uint32_atomic_compare_swap(slots + tid, AMO_SLOT_INIT, AMO_SLOT_NEW, mype);
    const uint32_t oldMiss = simt::aclshmem_uint32_atomic_compare_swap(slots + tid, AMO_SLOT_INIT, ~AMO_SLOT_NEW, mype);
    results[2 * tid] = oldMatch;
    results[2 * tid + 1] = oldMiss;

    __gm__ uint32_t* selfCounter = counters + 1;
    __gm__ uint32_t* globalCounter = counters;
    for (uint32_t i = 0; i < AMO_ITERS; ++i) {
        uint32_t old = *reinterpret_cast<__gm__ uint32_t*>(simt::aclshmem_ptr(selfCounter, mype));
        uint32_t assumed;
        do {
            assumed = old;
            old = simt::aclshmem_uint32_atomic_compare_swap(selfCounter, assumed, assumed + 1, mype);
        } while (old != assumed);
    }
    for (uint32_t i = 0; i < AMO_ITERS; ++i) {
        uint32_t old = *reinterpret_cast<__gm__ uint32_t*>(simt::aclshmem_ptr(globalCounter, 0));
        uint32_t assumed;
        do {
            assumed = old;
            old = simt::aclshmem_uint32_atomic_compare_swap(globalCounter, assumed, assumed + 1, 0);
        } while (old != assumed);
    }
}

__global__ __vector__ void demo_atomic_cas_call(
    __gm__ uint32_t* slots, __gm__ uint32_t* results, __gm__ uint32_t* counters)
{
    asc_vf_call<demo_atomic_cas_simt>(dim3(AMO_THREADS), slots, results, counters);
}

/**
 * @brief 打印 origin, res_prev 和 res_next 数组的内容
 * @param my_pe 当前节点的 ID，用于提示
 * @param print_all 是否打印全部元素。若为 false 且长度超过 20，则只打印首尾
 */
void print_buffers(
    int my_pe, int32_t* origin, int32_t* res_prev, int32_t* res_next, int32_t size, bool print_all = true)
{
    auto print_array = [&](const char* name, int32_t* arr) {
        printf("[PE %d] %s: [", my_pe, name);
        if (size <= 20 || print_all) {
            for (int i = 0; i < size; ++i) {
                printf("%d%s", arr[i], (i == size - 1 ? "" : ", "));
            }
        } else {
            // 长度较长时只打印首尾
            for (int i = 0; i < 10; ++i)
                printf("%d, ", arr[i]);
            printf("... , ");
            for (int i = size - 5; i < size; ++i) {
                printf("%d%s", arr[i], (i == size - 1 ? "" : ", "));
            }
        }
        printf("]\n");
    };

    printf("\n[PE %d] ======= Data Report (Size: %d) =======\n", my_pe, size);
    print_array("origin ", origin);
    print_array("res_prev", res_prev);
    print_array("res_next", res_next);
    printf("[PE %d] ======================================\n\n", my_pe);
}

int test_aclshmem_rma_mem(int my_pe, int n_pes)
{
    aclrtStream stream = nullptr;

    ACL_CHECK_WITH_RET(aclInit(nullptr), ERROR_LOG("aclInit failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtSetDevice(my_pe), ERROR_LOG("aclrtSetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtCreateStream(&stream), ERROR_LOG("aclrtCreateStream failed"), return -1);

    // 1. 准备 Host 端的初始化和校验缓冲区
    size_t data_bytes = COPY_SIZE * sizeof(int32_t);
    int32_t *origin_host, *res_prev_host, *res_next_host;
    ACL_CHECK_WITH_RET(
        aclrtMallocHost(reinterpret_cast<void**>(&origin_host), data_bytes), ERROR_LOG("malloc origin_host failed"),
        return -1);
    ACL_CHECK_WITH_RET(
        aclrtMallocHost(reinterpret_cast<void**>(&res_prev_host), data_bytes), ERROR_LOG("malloc res_prev_host failed"),
        return -1);
    ACL_CHECK_WITH_RET(
        aclrtMallocHost(reinterpret_cast<void**>(&res_next_host), data_bytes), ERROR_LOG("malloc res_next_host failed"),
        return -1);

    // 初始化 origin: Range[my_pe, my_pe + COPY_SIZE)
    for (int i = 0; i < COPY_SIZE; ++i) {
        origin_host[i] = my_pe + i;
        res_prev_host[i] = -1;
        res_next_host[i] = -1;
    }

    // 2. 准备 Debug 空间
    uint64_t* debug_host;
    constexpr int32_t debug_size = 32;
    ACL_CHECK_WITH_RET(
        aclrtMallocHost(reinterpret_cast<void**>(&debug_host), sizeof(uint64_t) * debug_size),
        ERROR_LOG("malloc debug_host failed"), return -1);
    std::memset(debug_host, 0, sizeof(uint64_t) * debug_size);

    uint64_t* debug_device = nullptr;
    ACL_CHECK_WITH_RET(
        aclrtMalloc((void**)&debug_device, sizeof(uint64_t) * debug_size, ACL_MEM_MALLOC_HUGE_FIRST),
        ERROR_LOG("malloc debug_device failed"), return -1);
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(
            debug_device, sizeof(uint64_t) * debug_size, debug_host, sizeof(uint64_t) * debug_size,
            ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy debug failed"), return -1);

    // 3. 初始化 ACLSHMEM 并申请对称堆空间
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(my_pe, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    ACL_CHECK_WITH_RET(
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), ERROR_LOG("aclshmemx_init failed"), return -1);

    int32_t* origin_device = (int32_t*)aclshmemx_malloc(data_bytes);
    int32_t* res_prev_device = (int32_t*)aclshmemx_malloc(data_bytes);
    int32_t* res_next_device = (int32_t*)aclshmemx_malloc(data_bytes);

    // 将初始化数据拷贝到 Device 对称堆
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(origin_device, data_bytes, origin_host, data_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy origin to device failed"), return -1);
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_prev_device, data_bytes, res_prev_host, data_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy res_prev to device failed"), return -1);
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_next_device, data_bytes, res_next_host, data_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy res_next to device failed"), return -1);

    // 4. 执行同步与计算
    aclshmem_barrier_all();
    run_demo_mem(stream, origin_device, res_prev_device, res_next_device, debug_device);
    ACL_CHECK_WITH_RET(aclrtSynchronizeStream(stream), ERROR_LOG("stream sync failed"), return -1);
    aclshmem_barrier_all();

    // 5. 拷贝回 Host 进行校验
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_prev_host, data_bytes, res_prev_device, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("memcpy res_prev back failed"), return -1);
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_next_host, data_bytes, res_next_device, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("memcpy res_next back failed"), return -1);

    sleep(my_pe + 1); // prevent multi-process interference(printing)
    print_buffers(my_pe, origin_host, res_prev_host, res_next_host, COPY_SIZE, false);

    // 6. 校验逻辑
    bool success = true;
    int32_t prev_pe = (my_pe - 1 + n_pes) % n_pes;
    int32_t next_pe = (my_pe + 1) % n_pes;

    for (int i = 0; i < COPY_SIZE; ++i) {
        if (res_prev_host[i] != (prev_pe + i)) {
            printf("[ERROR] PE %d: res_prev[%d] expected %d, got %d\n", my_pe, i, prev_pe + i, res_prev_host[i]);
            success = false;
            break;
        }
        if (res_next_host[i] != (next_pe + i)) {
            printf("[ERROR] PE %d: res_next[%d] expected %d, got %d\n", my_pe, i, next_pe + i, res_next_host[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[SUCCESS] PE %d: Verification passed for RMA transfers.\n", my_pe);
    } else {
        printf("[FAILURE] PE %d: Verification failed for RMA transfers.\n", my_pe);
    }

    // ===== SIMT 原子 CAS 演示 =====
    uint32_t* amo_device = (uint32_t*)aclshmemx_malloc(AMO_HEAP_WORDS * sizeof(uint32_t));

    uint32_t amoInit[AMO_HEAP_WORDS] = {};
    for (uint32_t t = 0; t < AMO_THREADS; ++t) {
        amoInit[AMO_SLOT_OFF + t] = AMO_SLOT_INIT;
    }
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(
            amo_device, AMO_HEAP_WORDS * sizeof(uint32_t), amoInit, AMO_HEAP_WORDS * sizeof(uint32_t),
            ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy amo init to device failed"), return -1);

    aclshmem_barrier_all();
    demo_atomic_cas_call<<<1, 0, stream>>>(
        amo_device + AMO_SLOT_OFF, amo_device + AMO_RESULT_OFF, amo_device + AMO_COUNTER_OFF);
    ACL_CHECK_WITH_RET(aclrtSynchronizeStream(stream), ERROR_LOG("stream sync failed"), return -1);
    // 全局计数器在 PE0 堆上被所有 PE 写，读回前需确保全部 PE 的 kernel 结束
    aclshmem_barrier_all();

    uint32_t amoOut[AMO_HEAP_WORDS] = {};
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(
            amoOut, AMO_HEAP_WORDS * sizeof(uint32_t), amo_device, AMO_HEAP_WORDS * sizeof(uint32_t),
            ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("memcpy amo result back failed"), return -1);

    const uint64_t amoTotal = static_cast<uint64_t>(AMO_THREADS) * AMO_ITERS;
    bool amoOk = true;
    for (uint32_t t = 0; t < AMO_THREADS; ++t) {
        if (amoOut[AMO_SLOT_OFF + t] != AMO_SLOT_NEW) {
            printf(
                "[ERROR] PE %d: amo slot %u expected 0x%x, got 0x%x\n", my_pe, t, AMO_SLOT_NEW,
                amoOut[AMO_SLOT_OFF + t]);
            amoOk = false;
        }
        if (amoOut[AMO_RESULT_OFF + 2 * t] != AMO_SLOT_INIT) {
            printf(
                "[ERROR] PE %d: amo result match %u expected 0x%x, got 0x%x\n", my_pe, t, AMO_SLOT_INIT,
                amoOut[AMO_RESULT_OFF + 2 * t]);
            amoOk = false;
        }
        if (amoOut[AMO_RESULT_OFF + 2 * t + 1] != AMO_SLOT_NEW) {
            printf(
                "[ERROR] PE %d: amo result miss %u expected 0x%x, got 0x%x\n", my_pe, t, AMO_SLOT_NEW,
                amoOut[AMO_RESULT_OFF + 2 * t + 1]);
            amoOk = false;
        }
    }
    if (amoOut[AMO_COUNTER_OFF + 1] != amoTotal) {
        printf(
            "[ERROR] PE %d: amo self counter expected %llu, got %u\n", my_pe, static_cast<unsigned long long>(amoTotal),
            amoOut[AMO_COUNTER_OFF + 1]);
        amoOk = false;
    }
    if (my_pe == 0 && amoOut[AMO_COUNTER_OFF] != static_cast<uint64_t>(n_pes) * amoTotal) {
        printf(
            "[ERROR] PE %d: amo global counter expected %llu, got %u\n", my_pe,
            static_cast<unsigned long long>(static_cast<uint64_t>(n_pes) * amoTotal), amoOut[AMO_COUNTER_OFF]);
        amoOk = false;
    }

    if (amoOk) {
        printf("[SUCCESS] PE %d: Verification passed for atomic compare swap.\n", my_pe);
    } else {
        printf("[FAILURE] PE %d: Verification failed for atomic compare swap.\n", my_pe);
    }

    aclshmemx_free(amo_device);

    aclshmemx_free(origin_device);
    aclshmemx_free(res_prev_device);
    aclshmemx_free(res_next_device);
    aclshmem_finalize();

    aclrtFreeHost(origin_host);
    aclrtFreeHost(res_prev_host);
    aclrtFreeHost(res_next_host);
    aclrtFreeHost(debug_host);
    aclrtFree(debug_device);

    aclrtDestroyStream(stream);
    aclrtResetDevice(my_pe);
    aclFinalize();
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        ERROR_LOG("Usage: %s <n_pes> <my_pe>", argv[0]);
        return -1;
    }
    int n_pes = 0;
    int my_pe = 0;
    try {
        n_pes = std::stoi(argv[1]);
        my_pe = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        ERROR_LOG("Invalid n_pes='%s' or my_pe='%s': %s", argv[1], argv[2], e.what());
        return -1;
    }
    if (n_pes <= 0 || my_pe < 0 || my_pe >= n_pes) {
        ERROR_LOG("Invalid n_pes=%d or my_pe=%d, expected n_pes > 0 and 0 <= my_pe < n_pes", n_pes, my_pe);
        return -1;
    }

    test_aclshmem_rma_mem(my_pe, n_pes);
    INFO_LOG("demo run end in pe %d.", my_pe);
    return 0;
}
