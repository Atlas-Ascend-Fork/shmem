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

#include "gm2gm/engine/shmemi_device_udma.h"
#include "shmem.h"
constexpr uint64_t MESSAGE_SIZE = 64;
constexpr uint32_t UDMA_ACTION_BATCH_COUNT = 2;
constexpr uint32_t UDMA_ACTION_UB_SIZE = MESSAGE_SIZE * UDMA_ACTION_BATCH_COUNT;

ACLSHMEM_DEVICE GM_ADDR get_udma_action_slot(GM_ADDR gva, int64_t rank, uint32_t batch_idx)
{
    return gva + (rank * UDMA_ACTION_BATCH_COUNT + batch_idx) * MESSAGE_SIZE;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAGetTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR dest_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmemx_udma_get_nbi<uint8_t, PIPE_S>(
            dest_addr, dest_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer);
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_get(uint32_t block_dim, void* stream, uint8_t* gva) { UDMAGetTest<<<block_dim, nullptr, stream>>>(gva); }

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAGetActionPointerTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR dest_addr = get_udma_action_slot(gva, peer, i);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dest_addr, dest_addr, ub_ptr, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dest_addr, dest_addr, ub_ptr, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_get_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAGetActionPointerTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAGetActionTensorTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR dest_addr = get_udma_action_slot(gva, peer, i);
            AscendC::GlobalTensor<uint8_t> dst_tensor;
            AscendC::GlobalTensor<uint8_t> src_tensor;
            dst_tensor.SetGlobalBuffer((__gm__ uint8_t*)dest_addr, MESSAGE_SIZE);
            src_tensor.SetGlobalBuffer((__gm__ uint8_t*)dest_addr, MESSAGE_SIZE);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_get_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAGetActionTensorTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        src_addr = gva + rank * MESSAGE_SIZE;
        aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(
            src_addr, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer);
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_put(uint32_t block_dim, void* stream, uint8_t* gva) { UDMAPutTest<<<block_dim, nullptr, stream>>>(gva); }

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAOpConfigPutGetTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        GM_ADDR src_addr = gva + rank * MESSAGE_SIZE;
        aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
            src_addr, src_addr, ub_ptr, MESSAGE_SIZE, peer, 0);
        aclshmemx_udma_quiet(peer);

        GM_ADDR dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
            dest_addr, dest_addr, ub_ptr, MESSAGE_SIZE, peer, 0);
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_op_config_put_get(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAOpConfigPutGetTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAHighLevelLocalRmaTest(
    GM_ADDR symmetric, uint64_t config)
{
    util_set_ffts_config(config);
    int32_t my_pe = aclshmem_my_pe();
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint8_t original_transport = device_state->topo_list[my_pe];

    // Force the UDMA topology bit to verify that the high-level self-copy guard still selects MTE.
    device_state->topo_list[my_pe] = original_transport | ACLSHMEM_TRANSPORT_UDMA;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(&device_state->topo_list[my_pe]));

    GM_ADDR blocking_put_src = symmetric;
    GM_ADDR blocking_put_dst = symmetric + MESSAGE_SIZE;
    GM_ADDR blocking_get_src = symmetric + MESSAGE_SIZE * 2;
    GM_ADDR blocking_get_dst = symmetric + MESSAGE_SIZE * 3;
    GM_ADDR nbi_put_src = symmetric + MESSAGE_SIZE * 4;
    GM_ADDR nbi_put_dst = symmetric + MESSAGE_SIZE * 5;
    GM_ADDR nbi_get_src = symmetric + MESSAGE_SIZE * 6;
    GM_ADDR nbi_get_dst = symmetric + MESSAGE_SIZE * 7;

    aclshmem_putmem(blocking_put_dst, blocking_put_src, MESSAGE_SIZE, my_pe);
    aclshmem_getmem(blocking_get_dst, blocking_get_src, MESSAGE_SIZE, my_pe);
    aclshmem_putmem_nbi(nbi_put_dst, nbi_put_src, MESSAGE_SIZE, my_pe);
    aclshmemx_mte_quiet();
    aclshmem_getmem_nbi(nbi_get_dst, nbi_get_src, MESSAGE_SIZE, my_pe);
    aclshmemx_mte_quiet();

    device_state->topo_list[my_pe] = original_transport;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(&device_state->topo_list[my_pe]));
}

void test_udma_highlevel_local_rma(uint32_t block_dim, void* stream, uint8_t* symmetric, uint64_t config)
{
    UDMAHighLevelLocalRmaTest<<<block_dim, nullptr, stream>>>(symmetric, config);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutActionPointerTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR src_addr = get_udma_action_slot(gva, rank, i);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    src_addr, src_addr, ub_ptr, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    src_addr, src_addr, ub_ptr, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_put_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAPutActionPointerTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutActionTensorTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR src_addr = get_udma_action_slot(gva, rank, i);
            AscendC::GlobalTensor<uint8_t> dst_tensor;
            AscendC::GlobalTensor<uint8_t> src_tensor;
            dst_tensor.SetGlobalBuffer((__gm__ uint8_t*)src_addr, MESSAGE_SIZE);
            src_tensor.SetGlobalBuffer((__gm__ uint8_t*)src_addr, MESSAGE_SIZE);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_put_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAPutActionTensorTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAOpConfigActionPointerTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t put_state{};
        aclshmemx_defer_t put_defer(put_state);
        aclshmemx_submit_t put_submit(put_state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR src_addr = get_udma_action_slot(gva, rank, i);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
                    src_addr, src_addr, ub_ptr, MESSAGE_SIZE, peer, 0, put_submit);
            } else {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3, ACLSHMEMX_UDMA_OP_CONFIG_NO_CQE>(
                    src_addr, src_addr, ub_ptr, MESSAGE_SIZE, peer, 0, put_defer);
            }
        }
        aclshmemx_udma_quiet(peer);

        aclshmemx_submit_state_t get_state{};
        aclshmemx_defer_t get_defer(get_state);
        aclshmemx_submit_t get_submit(get_state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR dest_addr = get_udma_action_slot(gva, peer, i);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
                    dest_addr, dest_addr, ub_ptr, MESSAGE_SIZE, peer, 0, get_submit);
            } else {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3, ACLSHMEMX_UDMA_OP_CONFIG_NO_CQE>(
                    dest_addr, dest_addr, ub_ptr, MESSAGE_SIZE, peer, 0, get_defer);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_op_config_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAOpConfigActionPointerTest<<<block_dim, nullptr, stream>>>(gva);
}

#ifdef ACLSHMEM_RELAY_SUPPORT
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMARelayActionCompileTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int my_pe = static_cast<int>(aclshmem_my_pe());
    int npes = static_cast<int>(aclshmem_n_pes());
    if (npes < 3) {
        return;
    }
    int pe = (my_pe + 1) % npes;
    int relay_pe = (my_pe + 2) % npes;
    GM_ADDR addr = get_udma_action_slot(gva, my_pe, 0);
    AscendC::GlobalTensor<uint8_t> dst_tensor;
    AscendC::GlobalTensor<uint8_t> src_tensor;
    dst_tensor.SetGlobalBuffer((__gm__ uint8_t*)addr, MESSAGE_SIZE);
    src_tensor.SetGlobalBuffer((__gm__ uint8_t*)addr, MESSAGE_SIZE);

    aclshmemx_submit_state_t put_ptr_state{};
    aclshmemx_defer_t put_ptr_defer(put_ptr_state);
    aclshmemx_submit_t put_ptr_submit(put_ptr_state);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, put_ptr_defer);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, put_ptr_submit);

    aclshmemx_submit_state_t put_tensor_state{};
    aclshmemx_defer_t put_tensor_defer(put_tensor_state);
    aclshmemx_submit_t put_tensor_submit(put_tensor_state);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, put_tensor_defer);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, put_tensor_submit);

    aclshmemx_submit_state_t get_ptr_state{};
    aclshmemx_defer_t get_ptr_defer(get_ptr_state);
    aclshmemx_submit_t get_ptr_submit(get_ptr_state);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, get_ptr_defer);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, get_ptr_submit);

    aclshmemx_submit_state_t get_tensor_state{};
    aclshmemx_defer_t get_tensor_defer(get_tensor_state);
    aclshmemx_submit_t get_tensor_submit(get_tensor_state);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, get_tensor_defer);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, get_tensor_submit);
}
#endif

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutSignalTest(GM_ADDR gva, GM_ADDR sig_addr)
{
    int64_t my_pe = aclshmem_my_pe();
    int64_t npes = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < npes; peer++) {
        if (peer == my_pe) {
            continue;
        }
        src_addr = gva + my_pe * MESSAGE_SIZE;
        uint64_t signal = 1000;
        auto dst_sig_addr = sig_addr + sizeof(uint64_t) * my_pe;
        aclshmemx_udma_put_signal_nbi(src_addr, src_addr, MESSAGE_SIZE, (__gm__ uint64_t*)dst_sig_addr, signal, peer);
        aclshmemx_udma_quiet(peer);
#ifndef ACLSHMEM_RELAY_SUPPORT
        aclshmemx_udma_qp_put_signal_nbi(
            src_addr, src_addr, MESSAGE_SIZE, (__gm__ uint64_t*)dst_sig_addr, signal, peer, 0);
        aclshmemx_udma_qp_quiet(peer, 0);
#endif
    }
}

void test_udma_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr)
{
    UDMAPutSignalTest<<<block_dim, nullptr, stream>>>(gva, sig_addr);
}

constexpr uint32_t UDMA_WRAP_TEST_WQE_SIZE = 64;
constexpr uint32_t UDMA_WRAP_TEST_HEADER_BYTES = 1024;
constexpr uint32_t UDMA_WRAP_TEST_RING_GUARD_BYTES = 256;
constexpr uint32_t UDMA_WRAP_TEST_QP_OFFSET = 256;
constexpr uint32_t UDMA_WRAP_TEST_REMOTE_MEM_OFFSET = 512;
constexpr uint32_t UDMA_WRAP_TEST_EID_OFFSET = 768;
// aclshmemi_udma_queue_state_t backing the test wq_ctx (state_addr points here). Layout constants
// below must not overlap the header region (0..UDMA_WRAP_TEST_HEADER_BYTES).
constexpr uint32_t UDMA_WRAP_TEST_STATE_OFFSET = 896;
constexpr uint32_t UDMA_WRAP_TEST_NOTIFY_HEAD_QWORD = sizeof(aclshmemi_sqe_ctx_t) / sizeof(uint64_t);
constexpr uint64_t UDMA_WRAP_TEST_GUARD_VALUE = 0xC0FFEE1234567890ULL;
constexpr uint64_t UDMA_WRAP_TEST_PIPE_S_SIGNAL = 0x1122334455667788ULL;
constexpr uint64_t UDMA_WRAP_TEST_PIPE_MTE3_SIGNAL = 0x8877665544332211ULL;
constexpr uint64_t UDMA_WRAP_TEST_FAA_VALUE = 0x0102030405060708ULL;
constexpr uint64_t UDMA_WRAP_TEST_CAS_SWAP = 0x1112131415161718ULL;
constexpr uint64_t UDMA_WRAP_TEST_CAS_COND = 0x2122232425262728ULL;

enum class UdmaWrapTestResult : uint32_t {
    PIPE_S_LAYOUT_OK = 0,
    PIPE_MTE3_LAYOUT_OK = 1,
    PIPE_S_GUARD_VALUE = 2,
    PIPE_MTE3_GUARD_VALUE = 3,
    FAA_LAYOUT_OK = 4,
    CAS_LAYOUT_OK = 5,
    FAA_GUARD_VALUE = 6,
    CAS_GUARD_VALUE = 7,
    COUNT = 8,
};

__aicore__ inline uint64_t udma_wrap_test_ring_bytes()
{
    return static_cast<uint64_t>(shm::UDMA_SQ_BASKBLK_CNT) * UDMA_WRAP_TEST_WQE_SIZE;
}

__aicore__ inline __gm__ uint64_t* udma_wrap_test_result(GM_ADDR workspace)
{
    return reinterpret_cast<__gm__ uint64_t*>(workspace);
}

__aicore__ inline __gm__ aclshmemi_udma_wq_ctx_t* udma_wrap_test_qp(GM_ADDR workspace)
{
    return reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(workspace + UDMA_WRAP_TEST_QP_OFFSET);
}

__aicore__ inline __gm__ aclshmemi_udma_queue_state_t* udma_wrap_test_state(GM_ADDR workspace)
{
    return reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(workspace + UDMA_WRAP_TEST_STATE_OFFSET);
}

__aicore__ inline __gm__ aclshmemi_ubmem_info_t* udma_wrap_test_remote_mem(GM_ADDR workspace)
{
    return reinterpret_cast<__gm__ aclshmemi_ubmem_info_t*>(workspace + UDMA_WRAP_TEST_REMOTE_MEM_OFFSET);
}

__aicore__ inline __gm__ uint64_t* udma_wrap_test_eid(GM_ADDR workspace)
{
    return reinterpret_cast<__gm__ uint64_t*>(workspace + UDMA_WRAP_TEST_EID_OFFSET);
}

__aicore__ inline __gm__ uint8_t* udma_wrap_test_pipe_s_ring(GM_ADDR workspace)
{
    return reinterpret_cast<__gm__ uint8_t*>(workspace + UDMA_WRAP_TEST_HEADER_BYTES);
}

__aicore__ inline __gm__ uint8_t* udma_wrap_test_pipe_mte3_ring(GM_ADDR workspace)
{
    return udma_wrap_test_pipe_s_ring(workspace) + udma_wrap_test_ring_bytes() + UDMA_WRAP_TEST_RING_GUARD_BYTES;
}

__aicore__ inline __gm__ uint8_t* udma_wrap_test_local_addr(GM_ADDR workspace)
{
    return udma_wrap_test_pipe_mte3_ring(workspace) + udma_wrap_test_ring_bytes() + UDMA_WRAP_TEST_RING_GUARD_BYTES;
}

__aicore__ inline void udma_wrap_test_clear_wqebbs(__gm__ uint8_t* ring)
{
    __gm__ uint64_t* tail =
        reinterpret_cast<__gm__ uint64_t*>(ring + (shm::UDMA_SQ_BASKBLK_CNT - 1) * UDMA_WRAP_TEST_WQE_SIZE);
    __gm__ uint64_t* head = reinterpret_cast<__gm__ uint64_t*>(ring);
    for (uint32_t i = 0; i < UDMA_WRAP_TEST_WQE_SIZE / sizeof(uint64_t); ++i) {
        tail[i] = 0;
        head[i] = 0;
    }

    __gm__ uint64_t* guard = reinterpret_cast<__gm__ uint64_t*>(ring + udma_wrap_test_ring_bytes());
    for (uint32_t i = 0; i < UDMA_WRAP_TEST_RING_GUARD_BYTES / sizeof(uint64_t); ++i) {
        guard[i] = UDMA_WRAP_TEST_GUARD_VALUE;
    }
}

__aicore__ inline void udma_wrap_test_prepare_fixture(GM_ADDR workspace, __gm__ uint8_t* ring)
{
    __gm__ uint64_t* eid = udma_wrap_test_eid(workspace);
    eid[0] = 0x0123456789ABCDEFULL;
    eid[1] = 0xFEDCBA9876543210ULL;

    __gm__ aclshmemi_ubmem_info_t* remote_mem = udma_wrap_test_remote_mem(workspace);
    remote_mem->token_value_valid = true;
    remote_mem->rmt_jetty_type = ACLSHMEM_UDMA_REMOTE_JETTY_TYPE;
    remote_mem->target_hint = 0x12;
    remote_mem->tpn = 0x3456;
    remote_mem->tid = 0x789A;
    remote_mem->rmt_token_value = 0xBCDEF;
    remote_mem->len = 0;
    remote_mem->addr = 0;
    remote_mem->eid_addr = reinterpret_cast<uint64_t>(eid);

    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx = udma_wrap_test_qp(workspace);
    qp_ctx->wqn = 0;
    qp_ctx->buf_addr = reinterpret_cast<uint64_t>(ring);
    qp_ctx->wqe_size = UDMA_WRAP_TEST_WQE_SIZE;
    qp_ctx->depth = shm::UDMA_SQ_BASKBLK_CNT;
    // head / tail / cqe_cnt live in the shared queue state the data plane reads via state_addr.
    __gm__ aclshmemi_udma_queue_state_t* state = udma_wrap_test_state(workspace);
    qp_ctx->state_addr = reinterpret_cast<uint64_t>(state);
    state->sq_head = shm::UDMA_SQ_BASKBLK_CNT - 1;
    state->sq_tail = shm::UDMA_SQ_BASKBLK_CNT - 1;
    state->cqe_cnt = 0;
    qp_ctx->db_mode = aclshmemi_udma_db_mode_t::SW_DB;
    qp_ctx->db_addr = 0;
    qp_ctx->sl = 0;
    qp_ctx->amo_addr = reinterpret_cast<uint64_t>(udma_wrap_test_local_addr(workspace) + 2 * MESSAGE_SIZE);
    udma_wrap_test_clear_wqebbs(ring);
}

__aicore__ inline void udma_wrap_test_flush_cleared_wqebbs(__gm__ uint8_t* ring)
{
    // The MTE3 case overwrites WQEBBs cleared by scalar stores, then verifies them with scalar loads.
    dcci_cachelines(ring + (shm::UDMA_SQ_BASKBLK_CNT - 1) * UDMA_WRAP_TEST_WQE_SIZE, UDMA_WRAP_TEST_WQE_SIZE);
    dcci_cachelines(ring, UDMA_WRAP_TEST_WQE_SIZE);
}

__aicore__ inline bool udma_wrap_test_verify_layout(__gm__ uint8_t* ring, uint64_t expected_signal, uint64_t local_addr)
{
    __gm__ uint64_t* first_wqebb =
        reinterpret_cast<__gm__ uint64_t*>(ring + (shm::UDMA_SQ_BASKBLK_CNT - 1) * UDMA_WRAP_TEST_WQE_SIZE);
    __gm__ uint64_t* second_wqebb = reinterpret_cast<__gm__ uint64_t*>(ring);
    __gm__ aclshmemi_sge_ctx_t* sge = reinterpret_cast<__gm__ aclshmemi_sge_ctx_t*>(ring + 2 * sizeof(uint64_t));
    __gm__ uint64_t* guard = reinterpret_cast<__gm__ uint64_t*>(ring + udma_wrap_test_ring_bytes());

    bool notify_head_ok =
        first_wqebb[UDMA_WRAP_TEST_NOTIFY_HEAD_QWORD] != 0 && first_wqebb[UDMA_WRAP_TEST_NOTIFY_HEAD_QWORD + 1] != 0;
    bool notify_tail_ok = second_wqebb[0] == expected_signal && second_wqebb[1] == 0;
    bool sge_ok = sge->len == MESSAGE_SIZE && sge->va == local_addr;
    bool guard_ok = true;
    for (uint32_t i = 0; i < UDMA_WRAP_TEST_RING_GUARD_BYTES / sizeof(uint64_t); ++i) {
        guard_ok = guard_ok && guard[i] == UDMA_WRAP_TEST_GUARD_VALUE;
    }
    return notify_head_ok && notify_tail_ok && sge_ok && guard_ok;
}

__aicore__ inline bool udma_wrap_test_verify_faa_layout(__gm__ uint8_t* ring, uint64_t amo_addr)
{
    __gm__ uint8_t* first_wqebb = ring + (shm::UDMA_SQ_BASKBLK_CNT - 1) * UDMA_WRAP_TEST_WQE_SIZE;
    __gm__ aclshmemi_sge_ctx_t* sge =
        reinterpret_cast<__gm__ aclshmemi_sge_ctx_t*>(first_wqebb + sizeof(aclshmemi_sqe_ctx_t));
    __gm__ uint64_t* second_wqebb = reinterpret_cast<__gm__ uint64_t*>(ring);
    __gm__ uint64_t* guard = reinterpret_cast<__gm__ uint64_t*>(ring + udma_wrap_test_ring_bytes());

    bool sge_ok = sge->len == sizeof(uint64_t) && sge->va == amo_addr;
    bool payload_ok = second_wqebb[0] == UDMA_WRAP_TEST_FAA_VALUE && second_wqebb[1] == 0;
    bool guard_ok = true;
    for (uint32_t i = 0; i < UDMA_WRAP_TEST_RING_GUARD_BYTES / sizeof(uint64_t); ++i) {
        guard_ok = guard_ok && guard[i] == UDMA_WRAP_TEST_GUARD_VALUE;
    }
    return sge_ok && payload_ok && guard_ok;
}

__aicore__ inline bool udma_wrap_test_verify_cas_layout(__gm__ uint8_t* ring, uint64_t amo_addr)
{
    __gm__ uint8_t* first_wqebb = ring + (shm::UDMA_SQ_BASKBLK_CNT - 1) * UDMA_WRAP_TEST_WQE_SIZE;
    __gm__ aclshmemi_sge_ctx_t* sge =
        reinterpret_cast<__gm__ aclshmemi_sge_ctx_t*>(first_wqebb + sizeof(aclshmemi_sqe_ctx_t));
    __gm__ uint64_t* second_wqebb = reinterpret_cast<__gm__ uint64_t*>(ring);
    __gm__ uint64_t* guard = reinterpret_cast<__gm__ uint64_t*>(ring + udma_wrap_test_ring_bytes());

    bool sge_ok = sge->len == sizeof(uint64_t) && sge->va == amo_addr;
    bool payload_ok = second_wqebb[0] == UDMA_WRAP_TEST_CAS_SWAP && second_wqebb[1] == UDMA_WRAP_TEST_CAS_COND;
    bool guard_ok = true;
    for (uint32_t i = 0; i < UDMA_WRAP_TEST_RING_GUARD_BYTES / sizeof(uint64_t); ++i) {
        guard_ok = guard_ok && guard[i] == UDMA_WRAP_TEST_GUARD_VALUE;
    }
    return sge_ok && payload_ok && guard_ok;
}

__aicore__ inline void udma_wrap_test_run_pipe_s(
    GM_ADDR workspace, __gm__ uint8_t* local_addr, __gm__ uint64_t* signal_addr, __gm__ uint64_t* result)
{
    __gm__ uint8_t* pipe_s_ring = udma_wrap_test_pipe_s_ring(workspace);
    aclshmemi_udma_params_t<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> pipe_s_params{
        .sig_addr = signal_addr, .signal = UDMA_WRAP_TEST_PIPE_S_SIGNAL};
    udma_wrap_test_prepare_fixture(workspace, pipe_s_ring);
    aclshmemi_udma_fill_post_send_wqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>(
        udma_wrap_test_qp(workspace), udma_wrap_test_remote_mem(workspace), local_addr, local_addr,
        shm::UDMA_SQ_BASKBLK_CNT - 1, UDMA_WRAP_TEST_WQE_SIZE, MESSAGE_SIZE, pipe_s_params);
    result[static_cast<uint32_t>(UdmaWrapTestResult::PIPE_S_LAYOUT_OK)] =
        udma_wrap_test_verify_layout(pipe_s_ring, UDMA_WRAP_TEST_PIPE_S_SIGNAL, reinterpret_cast<uint64_t>(local_addr));
    result[static_cast<uint32_t>(UdmaWrapTestResult::PIPE_S_GUARD_VALUE)] =
        *reinterpret_cast<__gm__ uint64_t*>(pipe_s_ring + udma_wrap_test_ring_bytes());
}

__aicore__ inline void udma_wrap_test_run_faa(GM_ADDR workspace, __gm__ uint8_t* local_addr, __gm__ uint64_t* result)
{
    __gm__ uint8_t* pipe_s_ring = udma_wrap_test_pipe_s_ring(workspace);
    aclshmemi_udma_params_t<uint64_t, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA> faa_params{
        .value = UDMA_WRAP_TEST_FAA_VALUE};
    udma_wrap_test_prepare_fixture(workspace, pipe_s_ring);
    aclshmemi_udma_fill_post_send_wqe<uint64_t, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>(
        udma_wrap_test_qp(workspace), udma_wrap_test_remote_mem(workspace), local_addr, local_addr,
        shm::UDMA_SQ_BASKBLK_CNT - 1, UDMA_WRAP_TEST_WQE_SIZE, sizeof(uint64_t), faa_params);
    result[static_cast<uint32_t>(UdmaWrapTestResult::FAA_LAYOUT_OK)] =
        udma_wrap_test_verify_faa_layout(pipe_s_ring, udma_wrap_test_qp(workspace)->amo_addr);
    result[static_cast<uint32_t>(UdmaWrapTestResult::FAA_GUARD_VALUE)] =
        *reinterpret_cast<__gm__ uint64_t*>(pipe_s_ring + udma_wrap_test_ring_bytes());
}

__aicore__ inline void udma_wrap_test_run_cas(GM_ADDR workspace, __gm__ uint8_t* local_addr, __gm__ uint64_t* result)
{
    __gm__ uint8_t* pipe_s_ring = udma_wrap_test_pipe_s_ring(workspace);
    aclshmemi_udma_params_t<uint64_t, aclshmemi_udma_opcode_t::UDMA_OP_CAS> cas_params{
        .value = UDMA_WRAP_TEST_CAS_SWAP, .cond = UDMA_WRAP_TEST_CAS_COND};
    udma_wrap_test_prepare_fixture(workspace, pipe_s_ring);
    aclshmemi_udma_fill_post_send_wqe<uint64_t, aclshmemi_udma_opcode_t::UDMA_OP_CAS>(
        udma_wrap_test_qp(workspace), udma_wrap_test_remote_mem(workspace), local_addr, local_addr,
        shm::UDMA_SQ_BASKBLK_CNT - 1, UDMA_WRAP_TEST_WQE_SIZE, sizeof(uint64_t), cas_params);
    result[static_cast<uint32_t>(UdmaWrapTestResult::CAS_LAYOUT_OK)] =
        udma_wrap_test_verify_cas_layout(pipe_s_ring, udma_wrap_test_qp(workspace)->amo_addr);
    result[static_cast<uint32_t>(UdmaWrapTestResult::CAS_GUARD_VALUE)] =
        *reinterpret_cast<__gm__ uint64_t*>(pipe_s_ring + udma_wrap_test_ring_bytes());
}

__aicore__ inline void udma_wrap_test_run_pipe_mte3(
    GM_ADDR workspace, __gm__ uint8_t* local_addr, __gm__ uint64_t* signal_addr, __ubuf__ uint8_t* wqe_scratch,
    __gm__ uint64_t* result)
{
    __gm__ uint8_t* pipe_mte3_ring = udma_wrap_test_pipe_mte3_ring(workspace);
    aclshmemi_udma_params_t<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> pipe_mte3_params{
        .sig_addr = signal_addr, .signal = UDMA_WRAP_TEST_PIPE_MTE3_SIGNAL};
    udma_wrap_test_prepare_fixture(workspace, pipe_mte3_ring);
    udma_wrap_test_flush_cleared_wqebbs(pipe_mte3_ring);
    __ubuf__ aclshmemi_sqe_ctx_t* sqe_ub = reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(wqe_scratch);
    aclshmemi_udma_fill_sqe_ctx<
        uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY, __ubuf__ aclshmemi_sqe_ctx_t*>(
        sqe_ub, local_addr, udma_wrap_test_remote_mem(workspace), shm::UDMA_SQ_BASKBLK_CNT - 1, pipe_mte3_params);
    __ubuf__ aclshmemi_sge_ctx_t* sge_ub = reinterpret_cast<__ubuf__ aclshmemi_sge_ctx_t*>(
        wqe_scratch + sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_notify_ctx_t));
    aclshmemi_fill_sge_header_ctx(sge_ub, MESSAGE_SIZE, reinterpret_cast<uint64_t>(local_addr));
    aclshmemi_udma_copy_wqe_from_ub_to_sq<get_wqe_bb_cnt<aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>()>(
        udma_wrap_test_qp(workspace), shm::UDMA_SQ_BASKBLK_CNT - 1, wqe_scratch, UDMA_WRAP_TEST_WQE_SIZE, 0);
    result[static_cast<uint32_t>(UdmaWrapTestResult::PIPE_MTE3_LAYOUT_OK)] = udma_wrap_test_verify_layout(
        pipe_mte3_ring, UDMA_WRAP_TEST_PIPE_MTE3_SIGNAL, reinterpret_cast<uint64_t>(local_addr));
    result[static_cast<uint32_t>(UdmaWrapTestResult::PIPE_MTE3_GUARD_VALUE)] =
        *reinterpret_cast<__gm__ uint64_t*>(pipe_mte3_ring + udma_wrap_test_ring_bytes());
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutSignalSqWrapTest(GM_ADDR workspace)
{
    __gm__ uint64_t* result = udma_wrap_test_result(workspace);
    for (uint32_t i = 0; i < static_cast<uint32_t>(UdmaWrapTestResult::COUNT); ++i) {
        result[i] = 0;
    }

    __gm__ uint8_t* local_addr = udma_wrap_test_local_addr(workspace);
    __gm__ uint64_t* signal_addr = reinterpret_cast<__gm__ uint64_t*>(local_addr + MESSAGE_SIZE);
    udma_wrap_test_run_pipe_s(workspace, local_addr, signal_addr, result);
    udma_wrap_test_run_faa(workspace, local_addr, result);
    udma_wrap_test_run_cas(workspace, local_addr, result);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE, 0);
    __ubuf__ uint8_t* wqe_scratch = reinterpret_cast<__ubuf__ uint8_t*>(ubLocal.GetPhyAddr());
    __ubuf__ uint64_t* scratch64 = reinterpret_cast<__ubuf__ uint64_t*>(wqe_scratch);
    for (uint32_t i = 0; i < ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE / sizeof(uint64_t); ++i) {
        scratch64[i] = 0;
    }
    udma_wrap_test_run_pipe_mte3(workspace, local_addr, signal_addr, wqe_scratch, result);
}

void test_udma_put_signal_sq_wrap(uint32_t block_dim, void* stream, uint8_t* workspace)
{
    UDMAPutSignalSqWrapTest<<<block_dim, nullptr, stream>>>(workspace);
}

#ifndef ACLSHMEM_RELAY_SUPPORT
enum class UdmaQpTestOperation : int32_t {
    PUT = 0,
    GET = 1,
    PUT_SIGNAL = 2,
};

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAQpDataPathTest(
    GM_ADDR symmetric, GM_ADDR local_buffer, GM_ADDR signal_words, uint32_t slice_size, int32_t peer, int32_t operation)
{
    const uint32_t qp_idx = AscendC::GetBlockIdx();
    const uint64_t slice_offset = static_cast<uint64_t>(qp_idx) * slice_size;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> scratch;
    pipe.InitBuffer(scratch, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
    __ubuf__ uint8_t* wqe_scratch = reinterpret_cast<__ubuf__ uint8_t*>(scratch.Get<uint8_t>().GetPhyAddr());

    if (operation == static_cast<int32_t>(UdmaQpTestOperation::PUT)) {
        aclshmemx_udma_qp_put_nbi<uint8_t>(
            symmetric + slice_offset, local_buffer + slice_offset, wqe_scratch, slice_size, peer, qp_idx);
    } else if (operation == static_cast<int32_t>(UdmaQpTestOperation::GET)) {
        aclshmemx_udma_qp_get_nbi<uint8_t>(
            local_buffer + slice_offset, symmetric + slice_offset, wqe_scratch, slice_size, peer, qp_idx);
    } else {
        const uint64_t signal = (static_cast<uint64_t>(aclshmem_my_pe() + 1) << 32) | (qp_idx + 1);
        aclshmemx_udma_qp_put_signal_nbi<uint8_t>(
            symmetric + slice_offset, local_buffer + slice_offset, slice_size,
            reinterpret_cast<__gm__ uint64_t*>(signal_words) + qp_idx, signal, peer, qp_idx, wqe_scratch);
    }
    aclshmemx_udma_qp_quiet(peer, qp_idx);
}

void test_udma_qp_data_path(
    uint32_t block_dim, void* stream, uint8_t* symmetric, uint8_t* local_buffer, uint64_t* signal_words,
    uint32_t slice_size, int32_t peer, int32_t operation)
{
    UDMAQpDataPathTest<<<block_dim, nullptr, stream>>>(
        symmetric, local_buffer, reinterpret_cast<uint8_t*>(signal_words), slice_size, peer, operation);
}
#endif
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAHighLevelPutSignalTest(
    GM_ADDR gva, GM_ADDR sig_addr)
{
    int64_t my_pe = aclshmem_my_pe();
    int64_t npes = aclshmem_n_pes();
    auto src_addr = reinterpret_cast<__gm__ uint8_t*>(gva + my_pe * MESSAGE_SIZE);
    auto signal_addr = reinterpret_cast<__gm__ int32_t*>(sig_addr + my_pe * sizeof(int32_t));

    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (int64_t peer = 0; peer < npes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmem_uint8_put_signal_nbi(
                src_addr, src_addr, MESSAGE_SIZE, signal_addr, 1000, ACLSHMEM_SIGNAL_SET, peer);
        }
    }
}

void test_udma_highlevel_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr)
{
    UDMAHighLevelPutSignalTest<<<block_dim, nullptr, stream>>>(gva, sig_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAHighLevelPutSignalSyncTest(
    GM_ADDR gva, GM_ADDR sig_addr)
{
    int64_t my_pe = aclshmem_my_pe();
    int64_t npes = aclshmem_n_pes();
    auto src_addr = reinterpret_cast<__gm__ uint8_t*>(gva + my_pe * MESSAGE_SIZE);
    auto signal_addr = reinterpret_cast<__gm__ int32_t*>(sig_addr + my_pe * sizeof(int32_t));

    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (int64_t peer = 0; peer < npes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmem_uint8_put_signal(src_addr, src_addr, MESSAGE_SIZE, signal_addr, 2000, ACLSHMEM_SIGNAL_SET, peer);
        }
    }
}

void test_udma_highlevel_put_signal_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr)
{
    UDMAHighLevelPutSignalSyncTest<<<block_dim, nullptr, stream>>>(gva, sig_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAHighLevelPutSizeSignalTest(
    GM_ADDR gva, GM_ADDR sig_addr)
{
    int64_t my_pe = aclshmem_my_pe();
    int64_t npes = aclshmem_n_pes();
    auto src_addr = reinterpret_cast<__gm__ void*>(gva + my_pe * MESSAGE_SIZE);
    auto signal_addr = reinterpret_cast<__gm__ int32_t*>(sig_addr + my_pe * sizeof(int32_t));

    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (int64_t peer = 0; peer < npes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmem_put64_signal_nbi(
                src_addr, src_addr, MESSAGE_SIZE / sizeof(uint64_t), signal_addr, 3000, ACLSHMEM_SIGNAL_SET, peer);
        }
    }
}

void test_udma_highlevel_put_size_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr)
{
    UDMAHighLevelPutSizeSignalTest<<<block_dim, nullptr, stream>>>(gva, sig_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAHighLevelPutSizeSignalSyncTest(
    GM_ADDR gva, GM_ADDR sig_addr)
{
    int64_t my_pe = aclshmem_my_pe();
    int64_t npes = aclshmem_n_pes();
    auto src_addr = reinterpret_cast<__gm__ void*>(gva + my_pe * MESSAGE_SIZE);
    auto signal_addr = reinterpret_cast<__gm__ int32_t*>(sig_addr + my_pe * sizeof(int32_t));

    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (int64_t peer = 0; peer < npes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmem_put64_signal(
                src_addr, src_addr, MESSAGE_SIZE / sizeof(uint64_t), signal_addr, 4000, ACLSHMEM_SIGNAL_SET, peer);
        }
    }
}

void test_udma_highlevel_put_size_signal_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr)
{
    UDMAHighLevelPutSizeSignalSyncTest<<<block_dim, nullptr, stream>>>(gva, sig_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAHighLevelPutSignalSplitTest(
    GM_ADDR gva, GM_ADDR sig_addr, uint64_t elem_size, int32_t signal)
{
    int64_t my_pe = aclshmem_my_pe();
    int64_t npes = aclshmem_n_pes();
    auto src_addr = reinterpret_cast<__gm__ void*>(gva + my_pe * elem_size);
    auto signal_addr = reinterpret_cast<__gm__ int32_t*>(sig_addr + my_pe * sizeof(int32_t));

    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (int64_t peer = 0; peer < npes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmem_putmem_signal(src_addr, src_addr, elem_size, signal_addr, signal, ACLSHMEM_SIGNAL_SET, peer);
        }
    }
}

void test_udma_highlevel_put_signal_split(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr, uint64_t elem_size, int32_t signal)
{
    UDMAHighLevelPutSignalSplitTest<<<block_dim, nullptr, stream>>>(gva, sig_addr, elem_size, signal);
}
