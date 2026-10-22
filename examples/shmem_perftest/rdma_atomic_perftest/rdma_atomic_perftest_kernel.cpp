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
#include "atomic_perftest_common.h"

using namespace perftest;

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)

template <typename T>
__aicore__ inline void store_atomic_result(__gm__ T* address, T value)
{
    AscendC::GlobalTensor<T> result;
    result.SetGlobalBuffer(address);
    result.SetValue(0, value);
}

template <typename T>
__aicore__ inline T issue_rdma_atomic(rdma_atomic_op_t op, __gm__ T* target, int pe, T operand, T compare)
{
    switch (op) {
        case RDMA_ATOMIC_INC:
            aclshmemx_roce_atomic_inc(target, pe);
            break;
        case RDMA_ATOMIC_FETCH_INC:
            return aclshmemx_roce_atomic_fetch_inc(target, pe);
        case RDMA_ATOMIC_SET:
            aclshmemx_roce_atomic_set(target, operand, pe);
            break;
        case RDMA_ATOMIC_ADD:
            aclshmemx_roce_atomic_add(target, operand, pe);
            break;
        case RDMA_ATOMIC_FETCH_ADD:
            return aclshmemx_roce_atomic_fetch_add(target, operand, pe);
        case RDMA_ATOMIC_AND:
            aclshmemx_roce_atomic_and(target, operand, pe);
            break;
        case RDMA_ATOMIC_FETCH_AND:
            return aclshmemx_roce_atomic_fetch_and(target, operand, pe);
        case RDMA_ATOMIC_OR:
            aclshmemx_roce_atomic_or(target, operand, pe);
            break;
        case RDMA_ATOMIC_FETCH_OR:
            return aclshmemx_roce_atomic_fetch_or(target, operand, pe);
        case RDMA_ATOMIC_XOR:
            aclshmemx_roce_atomic_xor(target, operand, pe);
            break;
        case RDMA_ATOMIC_FETCH_XOR:
            return aclshmemx_roce_atomic_fetch_xor(target, operand, pe);
        case RDMA_ATOMIC_SWAP:
            return aclshmemx_roce_atomic_swap(target, operand, pe);
        case RDMA_ATOMIC_COMPARE_SWAP:
            return aclshmemx_roce_atomic_compare_swap(target, compare, operand, pe);
        default:
            break;
    }
    return T(0);
}

template <typename T>
__aicore__ inline void run_rdma_atomic(
    GM_ADDR target_gva, GM_ADDR result_gva, uint64_t batch_size, rdma_atomic_op_t op, int iterations, uint32_t sync_id)
{
    const int pe = aclshmem_my_pe();
    constexpr int target_pe = 1;
    auto target = reinterpret_cast<__gm__ T*>(target_gva);
    auto results = reinterpret_cast<__gm__ rdma_atomic_result_t*>(result_gva);
    auto ub = reinterpret_cast<__ubuf__ char*>(aclshmemi_get_state()->rdma_config.aclshmem_ub);
    const bool fetching = atomic_has_fetch(op);
    const T operand = atomic_operand<T>(op);

    aclshmemx_roce_barrier_all(ub, sync_id);
    if (pe == 0) {
        T last_fetch = 0;
        const int64_t start = AscendC::GetSystemCycle();
        for (int iter = 0; iter < iterations; ++iter) {
            for (uint64_t idx = 0; idx < batch_size; ++idx) {
                last_fetch = issue_rdma_atomic(op, target + idx, target_pe, operand, T(0));
            }
            // Each batch completes before the next batch starts. Fetch calls complete before returning.
            if (!fetching) {
                aclshmemx_roce_quiet(target_pe, ub, sync_id);
            }
        }
        const int64_t end = AscendC::GetSystemCycle();
        store_atomic_result(&results[0].elapsed_cycles, end - start);
        store_atomic_result(&results[0].total_ops, batch_size * iterations);
        store_atomic_result(&results[0].last_fetch_bits, atomic_value_bits(last_fetch));
        dcci_cachelines(result_gva, sizeof(rdma_atomic_result_t) - 1);
    }
    aclshmemx_roce_barrier_all(ub, sync_id);

    // Remote completion precedes the target scan; verification traffic is outside the timed loop.
    if (pe == target_pe) {
        AscendC::GlobalTensor<T> target_data;
        target_data.SetGlobalBuffer(target);
        dcci_cachelines(target_gva, batch_size * sizeof(T) - 1);
        const uint64_t expected = atomic_expected_bits<T>(op, iterations);
        uint64_t mismatches = 0;
        uint64_t first_index = UINT64_MAX;
        uint64_t first_actual = 0;
        for (uint64_t idx = 0; idx < batch_size; ++idx) {
            const uint64_t actual = atomic_value_bits(target_data.GetValue(idx));
            if (actual != expected) {
                if (mismatches == 0) {
                    first_index = idx;
                    first_actual = actual;
                }
                ++mismatches;
            }
        }
        store_atomic_result(&results[1].total_ops, batch_size * iterations);
        store_atomic_result(&results[1].mismatch_count, mismatches);
        store_atomic_result(&results[1].first_mismatch_index, first_index);
        store_atomic_result(&results[1].first_actual_bits, first_actual);
        store_atomic_result(&results[1].expected_bits, expected);
        auto slot = reinterpret_cast<__gm__ char*>(results + 1);
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(slot), sizeof(rdma_atomic_result_t) - 1);
        aclshmemx_roce_put_nbi(slot, slot, ub, sizeof(rdma_atomic_result_t), 0, sync_id);
        aclshmemx_roce_quiet(0, ub, sync_id);
    }
    aclshmemx_roce_barrier_all(ub, sync_id);
}

#define DEFINE_ATOMIC_PERF_KERNEL(name, type)                                                                          \
    extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void rdma_atomic_perf_##name(                       \
        uint64_t ffts_addr, GM_ADDR target, GM_ADDR result, uint64_t batch_size, int op, int iterations, int sync_id)  \
    {                                                                                                                  \
        util_set_ffts_config(ffts_addr);                                                                               \
        if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                                            \
            run_rdma_atomic<type>(target, result, batch_size, static_cast<rdma_atomic_op_t>(op), iterations, sync_id); \
        }                                                                                                              \
    }

DEFINE_ATOMIC_PERF_KERNEL(uint32, uint32_t)
DEFINE_ATOMIC_PERF_KERNEL(uint64, uint64_t)
DEFINE_ATOMIC_PERF_KERNEL(int32, int32_t)
DEFINE_ATOMIC_PERF_KERNEL(int64, int64_t)

extern "C" void launch_rdma_atomic_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t ffts_addr, uint8_t* target_gva, uint8_t* result_gva, uint64_t batch_size,
    int atomic_op, int data_type, int iteration_count, int sync_id)
{
    switch (data_type) {
#define LAUNCH_ATOMIC_PERF(type_enum, name)                                                      \
    case type_enum:                                                                              \
        rdma_atomic_perf_##name<<<block_dim, nullptr, stream>>>(                                 \
            ffts_addr, target_gva, result_gva, batch_size, atomic_op, iteration_count, sync_id); \
        break;
        LAUNCH_ATOMIC_PERF(DATA_TYPE_UINT32, uint32)
        LAUNCH_ATOMIC_PERF(DATA_TYPE_UINT64, uint64)
        LAUNCH_ATOMIC_PERF(DATA_TYPE_INT32, int32)
        LAUNCH_ATOMIC_PERF(DATA_TYPE_INT64, int64)
#undef LAUNCH_ATOMIC_PERF
        default:
            break;
    }
}

#endif // ACLSHMEMI_RDMA_K_BACKEND_XSCALE || ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
