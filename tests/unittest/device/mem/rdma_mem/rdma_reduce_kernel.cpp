/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>

#include "kernel_operator.h"
#include "shmem.h"
#include "unittest/utils/rdma_reduce_kernel.h"

#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)

namespace {
constexpr int TARGET_PE = 1;

template <typename T, typename ReduceType, aclshmemx_reduce_op_t OP>
ACLSHMEM_DEVICE void put_reduce(
    __gm__ T* dst, __gm__ T* src, const AscendC::LocalTensor<T>& buf, uint32_t elem_size, RdmaReduceApi api,
    uint32_t qp_idx, uint32_t sync_id)
{
    auto* workspace = reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr());
    AscendC::GlobalTensor<T> dst_tensor;
    AscendC::GlobalTensor<T> src_tensor;
    dst_tensor.SetGlobalBuffer(dst, elem_size);
    src_tensor.SetGlobalBuffer(src, elem_size);
    switch (api) {
        case RdmaReduceApi::POINTER_DEFAULT:
            if constexpr (std::is_same_v<ReduceType, aclshmemx_half_sat_t>) {
                aclshmemx_roce_put_reduce_nbi<OP, aclshmemx_half_sat_t>(dst, src, workspace, elem_size, TARGET_PE);
            } else {
                aclshmemx_roce_put_reduce_nbi<OP>(dst, src, workspace, elem_size, TARGET_PE);
            }
            break;
        case RdmaReduceApi::POINTER_EVENT:
            aclshmemx_roce_put_reduce_nbi<OP, ReduceType>(dst, src, workspace, elem_size, TARGET_PE, sync_id);
            break;
        case RdmaReduceApi::POINTER_QP:
            aclshmemx_roce_qp_put_reduce_nbi<OP, ReduceType>(
                dst, src, workspace, elem_size, TARGET_PE, qp_idx, sync_id);
            break;
        case RdmaReduceApi::TENSOR_DEFAULT:
            if constexpr (std::is_same_v<ReduceType, aclshmemx_half_sat_t>) {
                aclshmemx_roce_put_reduce_nbi<OP, aclshmemx_half_sat_t>(
                    dst_tensor, src_tensor, buf, elem_size, TARGET_PE);
            } else {
                aclshmemx_roce_put_reduce_nbi<OP>(dst_tensor, src_tensor, buf, elem_size, TARGET_PE);
            }
            break;
        case RdmaReduceApi::TENSOR_EVENT:
            aclshmemx_roce_put_reduce_nbi<OP, ReduceType>(dst_tensor, src_tensor, buf, elem_size, TARGET_PE, sync_id);
            break;
        case RdmaReduceApi::TENSOR_QP:
            aclshmemx_roce_qp_put_reduce_nbi<OP, ReduceType>(
                dst_tensor, src_tensor, buf, elem_size, TARGET_PE, qp_idx, sync_id);
            break;
    }
}

template <typename T, typename ReduceType>
ACLSHMEM_DEVICE void put_reduce(
    __gm__ T* dst, __gm__ T* src, const AscendC::LocalTensor<T>& buf, uint32_t elem_size, aclshmemx_reduce_op_t op,
    RdmaReduceApi api, uint32_t qp_idx, uint32_t sync_id)
{
    switch (op) {
        case ACLSHMEMX_REDUCE_ADD:
            put_reduce<T, ReduceType, ACLSHMEMX_REDUCE_ADD>(dst, src, buf, elem_size, api, qp_idx, sync_id);
            break;
        case ACLSHMEMX_REDUCE_MAX:
            put_reduce<T, ReduceType, ACLSHMEMX_REDUCE_MAX>(dst, src, buf, elem_size, api, qp_idx, sync_id);
            break;
        case ACLSHMEMX_REDUCE_MIN:
            put_reduce<T, ReduceType, ACLSHMEMX_REDUCE_MIN>(dst, src, buf, elem_size, api, qp_idx, sync_id);
            break;
    }
}

template <typename T, typename ReduceType>
ACLSHMEM_DEVICE void run_put_reduce(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, aclshmemx_reduce_op_t op, RdmaReduceApi api, uint32_t qp_idx,
    uint32_t iterations)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> workspace;
    pipe.InitBuffer(workspace, UB_ALIGN_SIZE_64 * 2);
    auto buf = workspace.Get<T>();
    const uint32_t sync_id = aclshmemi_get_state()->rdma_config.sync_id;
    for (uint32_t i = 0; i < iterations; ++i) {
        put_reduce<T, ReduceType>(dst, src, buf, elem_size, op, api, qp_idx, sync_id);
    }
    // The same UB is reused immediately; quiet waits for the selected QP's source reads and remote updates.
    aclshmemx_roce_qp_quiet(TARGET_PE, qp_idx, buf, sync_id);
}
} // namespace

#define DEFINE_RDMA_REDUCE_KERNEL(NAME, HOST_TYPE, DEVICE_TYPE, REDUCE_TYPE)                                        \
    extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_put_reduce_##NAME##_kernel(                                    \
        __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t elem_size, aclshmemx_reduce_op_t op, RdmaReduceApi api,  \
        uint32_t qp_idx, uint32_t iterations, uint64_t config)                                                      \
    {                                                                                                               \
        util_set_ffts_config(config);                                                                               \
        run_put_reduce<DEVICE_TYPE, REDUCE_TYPE>(                                                                   \
            reinterpret_cast<__gm__ DEVICE_TYPE*>(dst), reinterpret_cast<__gm__ DEVICE_TYPE*>(src), elem_size, op,  \
            api, qp_idx, iterations);                                                                               \
    }                                                                                                               \
    void test_rdma_put_reduce_##NAME##_do(                                                                          \
        uint32_t block_dim, void* stream, uint8_t* dst, uint8_t* src, uint32_t elem_size, aclshmemx_reduce_op_t op, \
        RdmaReduceApi api, uint32_t qp_idx, uint32_t iterations, uint64_t config)                                   \
    {                                                                                                               \
        test_rdma_put_reduce_##NAME##_kernel<<<block_dim, nullptr, stream>>>(                                       \
            dst, src, elem_size, op, api, qp_idx, iterations, config);                                              \
    }
ACLSHMEM_RDMA_REDUCE_TEST_TYPES(DEFINE_RDMA_REDUCE_KERNEL);
#undef DEFINE_RDMA_REDUCE_KERNEL

extern "C" ACLSHMEM_GLOBAL_VECTOR void test_rdma_put_reduce_queue_kernel(
    __gm__ int32_t* dst, __gm__ int32_t* src, uint32_t count, bool complete, __gm__ uint32_t* queue_state,
    uint64_t config)
{
    util_set_ffts_config(config);
    constexpr uint32_t qp_idx = 1;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> workspace;
    pipe.InitBuffer(workspace, UB_ALIGN_SIZE_64 * 2);
    auto buf = workspace.Get<int32_t>();
    const uint32_t sync_id = aclshmemi_get_state()->rdma_config.sync_id;
    for (uint32_t i = 0; i < count; ++i) {
        aclshmemx_roce_qp_put_reduce_nbi(
            dst + i, src + i, reinterpret_cast<__ubuf__ int32_t*>(buf.GetPhyAddr()), 1, TARGET_PE, qp_idx, sync_id);
    }
    if (complete) {
        aclshmemx_roce_qp_quiet(TARGET_PE, qp_idx, buf, sync_id);
        // Ordinary WRITE must replace the destination after a reduction on the same QP and UB workspace.
        aclshmemx_roce_qp_put_nbi(
            dst + count, src + count, reinterpret_cast<__ubuf__ int32_t*>(buf.GetPhyAddr()), 1, TARGET_PE, qp_idx,
            sync_id);
        aclshmemx_roce_qp_quiet(TARGET_PE, qp_idx, buf, sync_id);
    }
    auto* info = aclshmemi_qp_info_fetch();
    const uint32_t index = TARGET_PE * info->qp_num + qp_idx;
    auto* sq = reinterpret_cast<__gm__ aclshmemi_rdma_sq_ctx*>(info->sq_ptr) + index;
    auto* cq = reinterpret_cast<__gm__ aclshmemi_rdma_cq_ctx*>(info->scq_ptr) + index;
    AscendC::GlobalTensor<uint32_t> state;
    state.SetGlobalBuffer(queue_state);
    state.SetValue(REDUCE_SQ_DEPTH, sq->depth);
    state.SetValue(REDUCE_CQ_DEPTH, cq->depth);
    state.SetValue(REDUCE_SQ_HEAD, aclshmemi_hns_1825_read_u32_gm(sq->head_addr));
    state.SetValue(REDUCE_SQ_TAIL, aclshmemi_hns_1825_read_u32_gm(sq->tail_addr));
    state.SetValue(REDUCE_CQ_TAIL, aclshmemi_hns_1825_read_u32_gm(cq->tail_addr));
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(queue_state), REDUCE_QUEUE_FIELDS * sizeof(uint32_t));
}

void test_rdma_put_reduce_queue_do(
    uint32_t block_dim, void* stream, int32_t* dst, int32_t* src, uint32_t count, bool complete, uint32_t* queue_state,
    uint64_t config)
{
    test_rdma_put_reduce_queue_kernel<<<block_dim, nullptr, stream>>>(dst, src, count, complete, queue_state, config);
}

#endif // ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
