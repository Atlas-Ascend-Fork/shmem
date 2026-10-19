/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_UDMA_HPP
#define SHMEM_DEVICE_UDMA_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "shmemi_device_common.h"
#include "shmemi_device_udma.h"
#include "utils/shmemi_kernel_debug.h"
#include "../host_device/shmemi_host_device_constant.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define ACLSHMEM_UDMA_SUPPORTED 1
#else
#define ACLSHMEM_UDMA_SUPPORTED 0
#endif

#if defined(ACLSHMEM_RELAY_SUPPORT)
#define ACLSHMEM_RELAY_SUPPORTED 1
#else
#define ACLSHMEM_RELAY_SUPPORTED 0
#endif

constexpr uint32_t MAX_RETRY_TIMES = 1000000;
constexpr uint32_t ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD = 10;
constexpr uint32_t ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE = sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_sge_ctx_t);
constexpr uint32_t ACLSHMEM_UDMA_REMOTE_JETTY_TYPE = 1;
constexpr uint32_t ACLSHMEM_UDMA_FLAG_ODR_MASK = 0x7U;
constexpr uint32_t ACLSHMEM_UDMA_FLAG_FENCE_SHIFT = 3U;
constexpr uint32_t ACLSHMEM_UDMA_FLAG_SE_SHIFT = 4U;
constexpr uint32_t ACLSHMEM_UDMA_FLAG_CQE_SHIFT = 5U;
constexpr uint32_t ACLSHMEM_UDMA_FLAG_UDF = 0x80U;
constexpr uint32_t ACLSHMEM_UDMA_CQE_ENTRY_IDX_MOD = 1U << 16;
static_assert(
    shm::UDMA_SQ_BASKBLK_CNT <= ACLSHMEM_UDMA_CQE_ENTRY_IDX_MOD,
    "UDMA CQE entry_idx is 16-bit and must cover the SQ BB count");

template <const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE constexpr void aclshmemi_udma_check_config()
{
    static_assert(CONFIG.cqe <= 1U, "UDMA op config cqe must be 0 or 1");
    static_assert(CONFIG.se == 0U, "UDMA op config se is not supported yet");
    static_assert(CONFIG.fence <= 1U, "UDMA op config fence must be 0 or 1");
    static_assert(CONFIG.odr <= ACLSHMEM_UDMA_FLAG_ODR_MASK, "UDMA op config odr must be in range 0..7");
}

template <const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE constexpr void aclshmemi_udma_require_cqe()
{
    aclshmemi_udma_check_config<CONFIG>();
    static_assert(CONFIG.cqe != 0U, "UDMA atomic operations require CONFIG.cqe != 0");
}

template <const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE constexpr void aclshmemi_udma_require_standalone_cqe()
{
    aclshmemi_udma_check_config<CONFIG>();
    static_assert(CONFIG.cqe != 0U, "UDMA immediate NBI operations require CONFIG.cqe != 0");
}

template <const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE constexpr void aclshmemi_udma_require_submit_cqe()
{
    aclshmemi_udma_check_config<CONFIG>();
    static_assert(CONFIG.cqe != 0U, "UDMA aggregate submit operations require CONFIG.cqe != 0");
}

template <const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE constexpr void aclshmemi_udma_require_defer_no_cqe()
{
    aclshmemi_udma_check_config<CONFIG>();
    static_assert(CONFIG.cqe == 0U, "UDMA aggregate defer operations require CONFIG.cqe == 0");
}

template <aclshmemi_udma_opcode_t OP_CODE, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE constexpr uint32_t aclshmemi_udma_build_flag()
{
    aclshmemi_udma_check_config<CONFIG>();
    uint32_t flag = CONFIG.odr | (CONFIG.fence << ACLSHMEM_UDMA_FLAG_FENCE_SHIFT) |
                    (CONFIG.se << ACLSHMEM_UDMA_FLAG_SE_SHIFT) | (CONFIG.cqe << ACLSHMEM_UDMA_FLAG_CQE_SHIFT);
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        flag |= ACLSHMEM_UDMA_FLAG_UDF;
    }
    return flag;
}

ACLSHMEM_DEVICE void aclshmemi_dump_sge(__gm__ uint8_t* wqe_addr, uint32_t sge_num);
ACLSHMEM_DEVICE void aclshmemi_dump_sge_ctx(__gm__ aclshmemi_sge_ctx_t* sge_ctx, uint32_t sge_num);
ACLSHMEM_DEVICE void aclshmemi_dump_sqe_ctx(__gm__ aclshmemi_sqe_ctx_t* sqe_ctx);
ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqe_addr, uint32_t atomic_len);
ACLSHMEM_DEVICE void aclshmemi_dump_sq_wqe(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size, uint32_t atomic_len);

ACLSHMEM_DEVICE __gm__ aclshmemi_aiv_udma_info_t* aclshmemi_udma_qp_info_fetch()
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info =
        (__gm__ aclshmemi_aiv_udma_info_t*)(aclshmemi_get_udma_info_address(0));
    return udma_info;
}

// Active queue/memory table for this build. Compile-time selected, inlined, and (since the union
// members share storage) resolves to the same address as the other member -- zero runtime cost.
ACLSHMEM_DEVICE __gm__ aclshmemi_udma_qp_table_t* aclshmemi_udma_active_table(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        return &udma_info->relay;
    } else {
        return &udma_info->direct;
    }
}

ACLSHMEM_DEVICE void aclshmemi_dump_cqe(__gm__ aclshmemi_jfc_cqe_ctx_t* cqe_addr)
{
    if (cqe_addr == nullptr) {
        AscendC::printf("CQE: nullptr pointer\n");
        return;
    }
    uint32_t s_r = cqe_addr->s_r;
    uint32_t is_jetty = cqe_addr->is_jetty;
    uint32_t owner = cqe_addr->owner;
    uint32_t inline_en = cqe_addr->inline_en;
    uint32_t opcode = cqe_addr->opcode;
    uint32_t fd = cqe_addr->fd;
    uint32_t substatus = cqe_addr->substatus;
    uint32_t status = cqe_addr->status;
    uint32_t entry_idx = cqe_addr->entry_idx;
    uint32_t local_num_l = cqe_addr->local_num_l;
    uint32_t local_num_h = cqe_addr->local_num_h;
    uint32_t rmt_idx = cqe_addr->rmt_idx;
    uint32_t tpn = cqe_addr->tpn;
    AscendC::printf(
        "CQE: DW0 - s_r: %d, is_jetty: %d, owner: %d, inline_en: %d, opcode: %d, fd: %d, substatus: %d, status: %d\n",
        s_r, is_jetty, owner, inline_en, opcode, fd, substatus, status);
    AscendC::printf("CQE: DW1 - entry_idx: %d, local_num_l: %d\n", entry_idx, local_num_l);
    AscendC::printf("CQE: DW2 - local_num_h: %d, rmt_idx: %d\n", local_num_h, rmt_idx);
    AscendC::printf("CQE: DW3 - tpn: %d\n", tpn);
    AscendC::printf("CQE: DW4 - byte_cnt: %d\n", cqe_addr->byte_cnt);
    AscendC::printf("CQE: DW5-DW6 - userData: 0x%x%x\n", cqe_addr->user_data_h, cqe_addr->user_data_l);
    AscendC::printf(
        "CQE: DW7-DW10 - rmt_eid: [0x%x, 0x%x, 0x%x, 0x%x]\n", cqe_addr->rmt_eid[0], cqe_addr->rmt_eid[1],
        cqe_addr->rmt_eid[2], cqe_addr->rmt_eid[3]);
    AscendC::printf("CQE: DW11-DW12 - data: 0x%x%x\n", cqe_addr->data_h, cqe_addr->data_l);
    AscendC::printf(
        "CQE: DW13-DW15 - inline_data: [0x%x, 0x%x, 0x%x]\n", cqe_addr->inline_data[0], cqe_addr->inline_data[1],
        cqe_addr->inline_data[2]);
}

// CQE entry_idx is a 16-bit next WQE BB counter. SQ buffer addressing still uses
// shm::UDMA_SQ_BASKBLK_CNT; this helper only restores the monotonic SQ tail.
ACLSHMEM_DEVICE uint32_t aclshmemi_udma_unwrap_wq_tail(uint32_t old_tail, uint32_t entry_idx)
{
    uint32_t new_tail =
        (old_tail & ~(ACLSHMEM_UDMA_CQE_ENTRY_IDX_MOD - 1U)) | (entry_idx & (ACLSHMEM_UDMA_CQE_ENTRY_IDX_MOD - 1U));
    if (new_tail < old_tail) {
        new_tail += ACLSHMEM_UDMA_CQE_ENTRY_IDX_MOD;
    }
    return new_tail;
}

ACLSHMEM_DEVICE uint32_t aclshmemi_udma_poll_cq(uint32_t slot, uint32_t qp_idx, uint32_t idx)
{
    if (idx == 0) {
        return 0;
    }
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    uint32_t qp_num = udma_info->qp_num;
    __gm__ aclshmemi_udma_cq_ctx_t* cq_ctx_entry =
        (__gm__ aclshmemi_udma_cq_ctx_t*)(tbl->scq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_cq_ctx_t));
    auto cq_base_addr = cq_ctx_entry->buf_addr;
    auto cqe_size = cq_ctx_entry->cqe_size;
    __gm__ aclshmemi_udma_queue_state_t* state = (__gm__ aclshmemi_udma_queue_state_t*)cq_ctx_entry->state_addr;
    uint32_t cur_tail = state->cq_tail;
    uint32_t last_entry_idx = 0;
    bool has_polled_cqe = false;
    while (cur_tail != idx) {
        __gm__ aclshmemi_jfc_cqe_ctx_t* cqe_addr =
            (__gm__ aclshmemi_jfc_cqe_ctx_t*)(cq_base_addr + cqe_size * (cur_tail & (shm::UDMA_CQ_DEPTH_DEFAULT - 1)));
        bool valid_owner = (cur_tail / shm::UDMA_CQ_DEPTH_DEFAULT) & 1;
        uint32_t times = 0;
        while ((valid_owner ^ cqe_addr->owner) == 0 && times < MAX_RETRY_TIMES) { // util cqe_addr->owner changed
            dcci_cachelines((__gm__ uint8_t*)cqe_addr, sizeof(aclshmemi_jfc_cqe_ctx_t));
            times++;
        }
        if (times >= MAX_RETRY_TIMES) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Poll cq timeout! cur_tail=%d idx=%d\n", cur_tail, idx);
            trap();
            return 0xFF;
        }
        // Check CQE status
        uint8_t status = cqe_addr->status & 0xFF;
        uint8_t sub_status = cqe_addr->substatus & 0xFF;
        constexpr uint8_t STATUS_SHIFT = 8;
        if (status != 0 || sub_status != 0) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_cqe, cqe_addr);
            trap();
            return (status << STATUS_SHIFT) | sub_status;
        }
        last_entry_idx = cqe_addr->entry_idx;
        has_polled_cqe = true;
        cur_tail++;
    }
    if (!has_polled_cqe) {
        return 0;
    }

    // Update CQ tail
    state->cq_tail = cur_tail;
    aclshmemi_udma_poll_cq_update_info(cur_tail, last_entry_idx, cq_ctx_entry, state);
    return 0;
}

ACLSHMEM_DEVICE void aclshmemi_udma_poll_cq_update_info(
    uint32_t cur_tail, uint32_t entry_idx, __gm__ aclshmemi_udma_cq_ctx_t* cq_ctx_entry,
    __gm__ aclshmemi_udma_queue_state_t* state)
{
    // Ring CQ Doorbell (reference URMA implementation)
    auto cq_db_addr = cq_ctx_entry->db_addr;
    // For JFC, we write the consumer index (cur_tail) directly
    __gm__ uint32_t* db_addr = (__gm__ uint32_t*)cq_ctx_entry->db_addr;
    st_dev((uint32_t)(cur_tail & 0xFFFFFF), db_addr, 0);
    // Update WQ tail: unwrap the 16-bit CQE entry_idx into the monotonic tail. The state is shared
    // per jetty group, so this stays coherent for slots landing on the same physical queue.
    state->sq_tail = aclshmemi_udma_unwrap_wq_tail(state->sq_tail, entry_idx);
}

ACLSHMEM_DEVICE void aclshmemi_dump_sqe_ctx(__gm__ aclshmemi_sqe_ctx_t* sqe_ctx)
{
    if (sqe_ctx == nullptr) {
        AscendC::printf("WQE: nullptr pointer\n");
        return;
    }
    auto sqe_bb_idx = sqe_ctx->sqe_bb_idx;
    auto flag = sqe_ctx->flag;
    auto rsv0 = sqe_ctx->rsv0;
    auto nf = sqe_ctx->nf;
    auto token_en = sqe_ctx->token_en;
    auto rmt_jetty_type = sqe_ctx->rmt_jetty_type;
    AscendC::printf(
        "WQE: sqe_bb_idx: %x flag: %x rsv0: %x nf: %x token_en: %x rmt_jetty_type: %x\n", sqe_bb_idx, flag, rsv0, nf,
        token_en, rmt_jetty_type);
    auto owner = sqe_ctx->owner;
    auto target_hint = sqe_ctx->target_hint;
    auto opcode = sqe_ctx->opcode;
    auto rsv1 = sqe_ctx->rsv1;
    auto inline_msg_len = sqe_ctx->inline_msg_len;
    auto tp_id = sqe_ctx->tp_id;
    AscendC::printf(
        "WQE: owner: %x target_hint: %x opcode: %x rsv1: %x inline_msg_len: %x tp_id: %x\n", owner, target_hint, opcode,
        rsv1, inline_msg_len, tp_id);
    auto sge_num = sqe_ctx->sge_num;
    auto rmt_jetty_or_seg_id = sqe_ctx->rmt_jetty_or_seg_id;
    auto rsv2 = sqe_ctx->rsv2;
    AscendC::printf("WQE: sge_num: %x rmt_jetty_or_seg_id: %x rsv2: %x\n", sge_num, rmt_jetty_or_seg_id, rsv2);
    auto rmt_eid_l = sqe_ctx->rmt_eid_l;
    auto rmt_eid_h = sqe_ctx->rmt_eid_h;
    AscendC::printf("WQE: rmt_eid: %x, %x\n", rmt_eid_l, rmt_eid_h);
    auto rmt_token_value = sqe_ctx->rmt_token_value;
    auto udf_type = sqe_ctx->udf_type;
    auto reduce_data_type = sqe_ctx->reduce_data_type;
    auto reduce_opcode = sqe_ctx->reduce_opcode;
    auto rmt_addr_l_or_token_id = sqe_ctx->rmt_addr_l_or_token_id;
    auto rmt_addr_h_or_token_value = sqe_ctx->rmt_addr_h_or_token_value;
    AscendC::printf(
        "WQE: rmt_token_value: %x udf_type: %x reduce_data_type: %x reduce_opcode: %x\n", rmt_token_value, udf_type,
        reduce_data_type, reduce_opcode);
    AscendC::printf(
        "WQE: rmt_addr_l_or_token_id: %x rmt_addr_h_or_token_value: %x\n", rmt_addr_l_or_token_id,
        rmt_addr_h_or_token_value);
}

ACLSHMEM_DEVICE void aclshmemi_dump_notify_ctx(__gm__ aclshmemi_notify_ctx_t* notify_ctx)
{
    auto notify_token_id = notify_ctx->notify_token_id;
    auto notify_token_value = notify_ctx->notify_token_value;
    auto notify_addr_l = notify_ctx->notify_addr_l;
    auto notify_addr_h = notify_ctx->notify_addr_h;
    auto notify_data_l = notify_ctx->notify_data_l;
    auto notify_data_h = notify_ctx->notify_data_h;
    AscendC::printf(
        "WQE: notify_token_id: %x notify_token_value: %x notify_addr_l: %x notify_addr_h: %x notify_data_l: %x "
        "notify_data_h: %x \n",
        notify_token_id, notify_token_value, notify_addr_l, notify_addr_h, notify_data_l, notify_data_h);
}

ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqe_addr, uint32_t atomic_len)
{
    if (wqe_addr == nullptr) {
        AscendC::printf("WQE: nullptr pointer\n");
        return;
    }
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx = (__gm__ aclshmemi_sqe_ctx_t*)wqe_addr;
    aclshmemi_dump_sqe_ctx(sqe_ctx);
    auto opcode = sqe_ctx->opcode;
    auto sge_num = sqe_ctx->sge_num;
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY)) {
        __gm__ aclshmemi_notify_ctx_t* notify_ctx =
            (__gm__ aclshmemi_notify_ctx_t*)((__gm__ uint8_t*)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t));
        aclshmemi_dump_notify_ctx(notify_ctx);
        aclshmemi_dump_sge_ctx(
            (__gm__ aclshmemi_sge_ctx_t*)((__gm__ uint8_t*)notify_ctx + sizeof(*notify_ctx)), sge_num);
        return;
    }
    aclshmemi_dump_sge(wqe_addr, sge_num);
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA)) {
        __gm__ uint8_t* amo_data_addr =
            (__gm__ uint8_t*)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_sge_ctx_t);
        uint64_t add_lo = (atomic_len == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)amo_data_addr) :
                                                             *(__gm__ uint64_t*)amo_data_addr;
        AscendC::printf("SGE: add_data: 0x%llx \n", (unsigned long long)add_lo);
    } else if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_CAS)) {
        __gm__ uint8_t* amo_data_addr =
            (__gm__ uint8_t*)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_sge_ctx_t);
        uint64_t swap_lo = (atomic_len == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)amo_data_addr) :
                                                              *(__gm__ uint64_t*)amo_data_addr;
        uint64_t cond_lo = (atomic_len == sizeof(uint32_t)) ?
                               static_cast<uint64_t>(*(__gm__ uint32_t*)(amo_data_addr + atomic_len)) :
                               *(__gm__ uint64_t*)(amo_data_addr + atomic_len);
        AscendC::printf(
            "SGE: cond_data: 0x%llx, swap_data: 0x%llx\n", (unsigned long long)cond_lo, (unsigned long long)swap_lo);
    }
}

ACLSHMEM_DEVICE void aclshmemi_dump_sge_ctx(__gm__ aclshmemi_sge_ctx_t* sge_ctx, uint32_t sge_num)
{
    if (sge_ctx == nullptr) {
        AscendC::printf("SGE: nullptr pointer\n");
        return;
    }
    for (size_t i = 0; i < sge_num; i++) {
        auto sge_len = sge_ctx->len;
        auto sge_rmt_addr = sge_ctx->va;
        AscendC::printf("SGE: sge idx: %d, va: %p sge_len: %d\n", i, sge_rmt_addr, sge_len);
        sge_ctx++;
    }
}

ACLSHMEM_DEVICE void aclshmemi_dump_sge(__gm__ uint8_t* wqe_addr, uint32_t sge_num)
{
    if (wqe_addr == nullptr) {
        AscendC::printf("WQE: nullptr pointer\n");
        return;
    }
    __gm__ aclshmemi_sge_ctx_t* sge_ctx = (__gm__ aclshmemi_sge_ctx_t*)(wqe_addr + sizeof(aclshmemi_sqe_ctx_t));
    aclshmemi_dump_sge_ctx(sge_ctx, sge_num);
}

ACLSHMEM_DEVICE constexpr uint32_t aclshmemi_udma_get_wqe_bb_cnt_by_opcode(uint32_t opcode)
{
    // 暂时不考虑inline功能，inline场景需要考虑inline data的大小
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) ||
        opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_CAS) ||
        opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY)) {
        return 2;
    }
    return 1;
}

template <aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE constexpr uint32_t get_wqe_bb_cnt()
{
    return aclshmemi_udma_get_wqe_bb_cnt_by_opcode(static_cast<uint32_t>(OP_CODE));
}

ACLSHMEM_DEVICE constexpr uint32_t aclshmemi_udma_get_wqe_bb_cnt(uint32_t opcode)
{
    return aclshmemi_udma_get_wqe_bb_cnt_by_opcode(opcode);
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR>
ACLSHMEM_DEVICE void aclshmemi_fill_notify_data(
    SQE_PTR sqe_ctx, uint32_t tid, uint32_t token_value, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        // The notify ctx lives immediately after the SQE in the same address space.
        using BytePtr = typename AscendC::Std::conditional<
            AscendC::IsSameType<SQE_PTR, __ubuf__ aclshmemi_sqe_ctx_t*>::value, __ubuf__ uint8_t*,
            __gm__ uint8_t*>::type;
        using NotifyQwordPtr = typename AscendC::Std::conditional<
            AscendC::IsSameType<SQE_PTR, __ubuf__ aclshmemi_sqe_ctx_t*>::value, __ubuf__ uint64_t*,
            __gm__ uint64_t*>::type;
        NotifyQwordPtr notify64 = (NotifyQwordPtr)((BytePtr)sqe_ctx + sizeof(aclshmemi_sqe_ctx_t));
        notify64[0] = static_cast<uint64_t>(tid & 0xFFFFFU) | (static_cast<uint64_t>(token_value) << 32);
        notify64[1] = reinterpret_cast<uint64_t>(params.sig_addr);
        notify64[2] = params.signal;
        notify64[3] = 0;
    }
}

template <typename SGE_PTR>
ACLSHMEM_DEVICE void aclshmemi_fill_sge_header_ctx(SGE_PTR sge_ctx, uint64_t message_len, uint64_t va)
{
    using SgeQwordPtr = typename AscendC::Std::conditional<
        AscendC::IsSameType<SGE_PTR, __ubuf__ aclshmemi_sge_ctx_t*>::value, __ubuf__ uint64_t*, __gm__ uint64_t*>::type;
    SgeQwordPtr sge64 = (SgeQwordPtr)sge_ctx;
    sge64[0] = static_cast<uint64_t>(static_cast<uint32_t>(message_len));
    sge64[1] = va;
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_fill_sge_ctx(
    __gm__ aclshmemi_sge_ctx_t* sge_ctx, uint64_t message_len, __gm__ uint8_t* local_addr,
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    // default
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) { // fetch and add
        auto amo_addr = qp_ctx_entry->amo_addr;
        aclshmemi_fill_sge_header_ctx(sge_ctx, message_len, amo_addr);
        __gm__ T* addDataAddr = (__gm__ T*)((__gm__ uint8_t*)sge_ctx + sizeof(aclshmemi_sge_ctx_t));
        *addDataAddr = params.value;                                        // fill in add_data
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS) { // compare and swap
        auto amo_addr = qp_ctx_entry->amo_addr;
        aclshmemi_fill_sge_header_ctx(sge_ctx, message_len, amo_addr);
        __gm__ T* swap_data_addr = (__gm__ T*)((__gm__ uint8_t*)sge_ctx + sizeof(aclshmemi_sge_ctx_t));
        *swap_data_addr = params.value; // fill in swap_data
        __gm__ T* cmp_data_addr = (__gm__ T*)((__gm__ uint8_t*)swap_data_addr + sizeof(T));
        *cmp_data_addr = params.cond; // fill in cmp_data
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        auto amo_addr = qp_ctx_entry->amo_addr;
        *(__gm__ T*)amo_addr = params.value;
        dcci_cachelines((__gm__ uint8_t*)amo_addr, sizeof(T));
        aclshmemi_fill_sge_header_ctx(sge_ctx, message_len, amo_addr);
    } else {
        aclshmemi_fill_sge_header_ctx(sge_ctx, message_len, reinterpret_cast<uint64_t>(local_addr));
    }
}

template <aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_udma_get_sge_ctx(__gm__ uint8_t* wqe_addr)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        constexpr size_t offset = sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_notify_ctx_t);
        return (wqe_addr + offset);
    } else {
        return (wqe_addr + sizeof(aclshmemi_sqe_ctx_t));
    }
}

ACLSHMEM_DEVICE void assert_remote_jetty_type_valid(__gm__ aclshmemi_ubmem_info_t* remote_mem_info)
{
    uint32_t rmt_jetty_type = remote_mem_info->rmt_jetty_type;
    if (rmt_jetty_type != ACLSHMEM_UDMA_REMOTE_JETTY_TYPE) {
        AscendC::printf(
            "udma_post_send: remote jetty type %u is not %u\n", rmt_jetty_type, ACLSHMEM_UDMA_REMOTE_JETTY_TYPE);
        trap();
    }
}

// Credit is computed against the shared queue state: slots on one jetty consume one SQ. Callers
// resolve state from qp_ctx_entry->state_addr once and pass it in, keeping the GM load off the
// per-post hot path.
ACLSHMEM_DEVICE bool aclshmemi_udma_sq_has_credit(__gm__ aclshmemi_udma_queue_state_t* state, uint32_t wqe_bb_cnt)
{
    uint32_t used_after_post = state->sq_head + wqe_bb_cnt - state->sq_tail;
    return used_after_post + ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD < shm::UDMA_SQ_BASKBLK_CNT;
}

ACLSHMEM_DEVICE bool aclshmemi_udma_sq_has_capacity(__gm__ aclshmemi_udma_queue_state_t* state, uint32_t wqe_bb_cnt)
{
    return state->sq_head + wqe_bb_cnt - state->sq_tail < shm::UDMA_SQ_BASKBLK_CNT;
}

// Poll CQ if the send queue lacks credit after this post. head / tail / cqe_cnt live in the shared
// queue state, so the credit check and the poll target go through state, not wq_ctx fields.
ACLSHMEM_DEVICE void aclshmemi_udma_poll_cq_when_sq_overflow(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t slot, uint32_t qp_idx, uint32_t wqe_bb_cnt)
{
    if (!aclshmemi_udma_sq_has_credit(state, wqe_bb_cnt)) {
        (void)aclshmemi_udma_poll_cq(slot, qp_idx, state->cqe_cnt);
    }
}

ACLSHMEM_DEVICE void aclshmemi_udma_debug_ensure_sq_credit(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t slot, uint32_t qp_idx, uint32_t wqe_bb_cnt)
{
    if (aclshmemi_udma_sq_has_credit(state, wqe_bb_cnt)) {
        return;
    }

    aclshmemi_kernel_abort(
        "UDMA SQ lacks credit: slot=%u qp_idx=%u need=%u head=%u tail=%u cqe_cnt=%u\n", slot, qp_idx, wqe_bb_cnt,
        state->sq_head, state->sq_tail, state->cqe_cnt);
}

ACLSHMEM_DEVICE void aclshmemi_udma_debug_ensure_sq_capacity(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t slot, uint32_t qp_idx, uint32_t wqe_bb_cnt)
{
    if (aclshmemi_udma_sq_has_capacity(state, wqe_bb_cnt)) {
        return;
    }

    aclshmemi_kernel_abort(
        "UDMA SQ lacks capacity: slot=%u qp_idx=%u need=%u head=%u tail=%u cqe_cnt=%u\n", slot, qp_idx, wqe_bb_cnt,
        state->sq_head, state->sq_tail, state->cqe_cnt);
}

ACLSHMEM_DEVICE void aclshmemi_udma_debug_check_aggregate_batch_size(uint32_t pending_wqe_cnt)
{
    constexpr uint32_t depth = shm::UDMA_SQ_BASKBLK_CNT;
    if (pending_wqe_cnt >= depth) {
        aclshmemi_kernel_abort("UDMA aggregate batch too large: pending=%u depth=%u\n", pending_wqe_cnt, depth);
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE constexpr uint32_t aclshmemi_udma_get_reduce_attr()
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        uint32_t reduce_data_type = 0;
        if constexpr (AscendC::IsSameType<T, float>::value) {
            reduce_data_type = 0x7; // fp32
        }
        return (reduce_data_type << 8) | (0xaU << 12);
    } else {
        return 0;
    }
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR = __gm__ aclshmemi_sqe_ctx_t*,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_fill_sqe_base_ctx(
    SQE_PTR sqe_ctx, __gm__ uint8_t* remote_addr, __gm__ aclshmemi_ubmem_info_t* remote_mem_info, uint32_t cur_head)
{
    ACLSHMEM_DEBUG_FUNC(assert_remote_jetty_type_valid, remote_mem_info);
    using SqeQwordPtr = typename AscendC::Std::conditional<
        AscendC::IsSameType<SQE_PTR, __ubuf__ aclshmemi_sqe_ctx_t*>::value, __ubuf__ uint64_t*, __gm__ uint64_t*>::type;
    SqeQwordPtr sqe64 = (SqeQwordPtr)sqe_ctx;

    constexpr uint32_t flag = aclshmemi_udma_build_flag<OP_CODE, CONFIG>();
    constexpr uint32_t opcode = OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE ?
                                    static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE) :
                                    static_cast<uint32_t>(OP_CODE);
    uint32_t dw0 = (flag << 16) | (static_cast<uint32_t>(remote_mem_info->token_value_valid) << 28) |
                   (ACLSHMEM_UDMA_REMOTE_JETTY_TYPE << 29);
    uint32_t dw1 = (remote_mem_info->target_hint & 0xFFU) | (opcode << 8);
    uint32_t dw2 = (remote_mem_info->tpn & 0xFFFFFFU) | (1U << 24);
    uint32_t dw3 = remote_mem_info->tid & 0xFFFFFU;
    uint32_t dw4 = remote_mem_info->rmt_token_value;
    uint32_t dw5 = aclshmemi_udma_get_reduce_attr<T, OP_CODE>();
    __gm__ uint64_t* rmt_eid = (__gm__ uint64_t*)(remote_mem_info->eid_addr);
    sqe64[0] = static_cast<uint64_t>(dw0) | (static_cast<uint64_t>(dw1) << 32);
    sqe64[1] = static_cast<uint64_t>(dw2) | (static_cast<uint64_t>(dw3) << 32);
    sqe64[2] = rmt_eid[0];
    sqe64[3] = rmt_eid[1];
    sqe64[4] = static_cast<uint64_t>(dw4) | (static_cast<uint64_t>(dw5) << 32);
    sqe64[5] = reinterpret_cast<uint64_t>(remote_addr);
    (void)cur_head;
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR = __gm__ aclshmemi_sqe_ctx_t*,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_fill_sqe_ctx(
    SQE_PTR sqe_ctx, __gm__ uint8_t* remote_addr, __gm__ aclshmemi_ubmem_info_t* remote_mem_info, uint32_t cur_head,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    aclshmemi_udma_fill_sqe_base_ctx<T, OP_CODE, SQE_PTR, CONFIG>(sqe_ctx, remote_addr, remote_mem_info, cur_head);
    aclshmemi_fill_notify_data<T, OP_CODE>(sqe_ctx, remote_mem_info->tid, remote_mem_info->rmt_token_value, params);
}

ACLSHMEM_DEVICE __gm__ aclshmemi_udma_wq_ctx_t* aclshmemi_udma_get_qp_ctx(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t slot, uint32_t qp_idx)
{
    uint32_t qp_num = udma_info->qp_num;
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
        (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_wq_ctx_t));
    return qp_ctx_entry;
}

ACLSHMEM_DEVICE __gm__ aclshmemi_ubmem_info_t* aclshmemi_udma_get_mem_info(
    __gm__ aclshmemi_aiv_udma_info_t* udma_info, uint32_t slot, uint32_t qp_idx)
{
    uint32_t qp_num = udma_info->qp_num;
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    return (__gm__ aclshmemi_ubmem_info_t*)(tbl->mem_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_ubmem_info_t));
}

ACLSHMEM_DEVICE __gm__ aclshmemi_sqe_ctx_t* aclshmemi_udma_get_sqe_ctx(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size)
{
    __gm__ uint8_t* wqe_addr =
        (__gm__ uint8_t*)(qp_ctx_entry->buf_addr + wqe_size * (cur_head % shm::UDMA_SQ_BASKBLK_CNT));
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx = (__gm__ aclshmemi_sqe_ctx_t*)wqe_addr;
    return sqe_ctx;
}

ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_udma_get_sq_wqebb_addr(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size)
{
    return (__gm__ uint8_t*)(qp_ctx_entry->buf_addr + wqe_size * (cur_head % shm::UDMA_SQ_BASKBLK_CNT));
}

struct aclshmemi_udma_sq_span_t {
    uint32_t first_wqebb_cnt;
    uint32_t second_wqebb_cnt;
};

template <uint32_t WQE_BB_CNT>
ACLSHMEM_DEVICE aclshmemi_udma_sq_span_t aclshmemi_udma_get_sq_span(uint32_t cur_head)
{
    constexpr uint32_t depth = shm::UDMA_SQ_BASKBLK_CNT;
    uint32_t first_wqebb_cnt = depth - (cur_head % depth);
    if (first_wqebb_cnt > WQE_BB_CNT) {
        first_wqebb_cnt = WQE_BB_CNT;
    }
    return {first_wqebb_cnt, WQE_BB_CNT - first_wqebb_cnt};
}

ACLSHMEM_DEVICE bool aclshmemi_udma_sq_span_wraps(uint32_t cur_head, uint32_t wqe_bb_cnt)
{
    constexpr uint32_t depth = shm::UDMA_SQ_BASKBLK_CNT;
    return (cur_head % depth) + wqe_bb_cnt > depth;
}

template <uint32_t WQE_BB_CNT>
ACLSHMEM_DEVICE bool aclshmemi_udma_sq_span_wraps(uint32_t cur_head)
{
    return aclshmemi_udma_sq_span_wraps(cur_head, WQE_BB_CNT);
}

ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_udma_get_sq_wqe_offset_addr(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size, uint32_t wqe_offset)
{
    constexpr uint32_t depth = shm::UDMA_SQ_BASKBLK_CNT;
    uint32_t ring_idx = cur_head % depth;
    uint32_t tail_bytes = (depth - ring_idx) * wqe_size;
    if (wqe_offset < tail_bytes) {
        return (__gm__ uint8_t*)(qp_ctx_entry->buf_addr + static_cast<uint64_t>(ring_idx) * wqe_size + wqe_offset);
    }
    return (__gm__ uint8_t*)(qp_ctx_entry->buf_addr + wqe_offset - tail_bytes);
}

ACLSHMEM_DEVICE void aclshmemi_dump_sq_wqe(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size, uint32_t atomic_len)
{
    __gm__ uint8_t* wqe_addr = aclshmemi_udma_get_sq_wqebb_addr(qp_ctx_entry, cur_head, wqe_size);
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx = (__gm__ aclshmemi_sqe_ctx_t*)wqe_addr;
    uint32_t opcode = sqe_ctx->opcode;
    uint32_t wqe_bb_cnt = aclshmemi_udma_get_wqe_bb_cnt(opcode);
    bool wraps_wqe = aclshmemi_udma_sq_span_wraps(cur_head, wqe_bb_cnt);
    if (!wraps_wqe) {
        aclshmemi_dump_wqe(wqe_addr, atomic_len);
        return;
    }

    aclshmemi_dump_sqe_ctx(sqe_ctx);

    uint32_t sge_offset = sizeof(aclshmemi_sqe_ctx_t);
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY)) {
        __gm__ uint64_t* notify_head0 = (__gm__ uint64_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sizeof(aclshmemi_sqe_ctx_t));
        __gm__ uint64_t* notify_head1 = (__gm__ uint64_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sizeof(aclshmemi_sqe_ctx_t) + sizeof(uint64_t));
        __gm__ uint64_t* notify_tail0 = (__gm__ uint64_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sizeof(aclshmemi_sqe_ctx_t) + 2 * sizeof(uint64_t));
        auto notify_token_id = static_cast<uint32_t>(*notify_head0 & 0xFFFFFU);
        auto notify_token_value = static_cast<uint32_t>(*notify_head0 >> 32);
        auto notify_addr_l = static_cast<uint32_t>(*notify_head1);
        auto notify_addr_h = static_cast<uint32_t>(*notify_head1 >> 32);
        auto notify_data_l = static_cast<uint32_t>(*notify_tail0);
        auto notify_data_h = static_cast<uint32_t>(*notify_tail0 >> 32);
        AscendC::printf(
            "WQE: notify_token_id: %x notify_token_value: %x notify_addr_l: %x notify_addr_h: %x notify_data_l: %x "
            "notify_data_h: %x \n",
            notify_token_id, notify_token_value, notify_addr_l, notify_addr_h, notify_data_l, notify_data_h);
        sge_offset += sizeof(aclshmemi_notify_ctx_t);
    }

    __gm__ aclshmemi_sge_ctx_t* sge_ctx = (__gm__ aclshmemi_sge_ctx_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
        qp_ctx_entry, cur_head, wqe_size, sge_offset);
    aclshmemi_dump_sge_ctx(sge_ctx, sqe_ctx->sge_num);
    uint32_t atomic_offset = sge_offset + sizeof(aclshmemi_sge_ctx_t);
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA)) {
        __gm__ uint8_t* amo_data_addr =
            aclshmemi_udma_get_sq_wqe_offset_addr(qp_ctx_entry, cur_head, wqe_size, atomic_offset);
        uint64_t add_lo = (atomic_len == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)amo_data_addr) :
                                                             *(__gm__ uint64_t*)amo_data_addr;
        AscendC::printf("SGE: add_data: 0x%llx \n", (unsigned long long)add_lo);
    } else if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_CAS)) {
        __gm__ uint8_t* swap_data_addr =
            aclshmemi_udma_get_sq_wqe_offset_addr(qp_ctx_entry, cur_head, wqe_size, atomic_offset);
        __gm__ uint8_t* cond_data_addr =
            aclshmemi_udma_get_sq_wqe_offset_addr(qp_ctx_entry, cur_head, wqe_size, atomic_offset + atomic_len);
        uint64_t swap_lo = (atomic_len == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)swap_data_addr) :
                                                              *(__gm__ uint64_t*)swap_data_addr;
        uint64_t cond_lo = (atomic_len == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)cond_data_addr) :
                                                              *(__gm__ uint64_t*)cond_data_addr;
        AscendC::printf(
            "SGE: cond_data: 0x%llx, swap_data: 0x%llx\n", (unsigned long long)cond_lo, (unsigned long long)swap_lo);
    }
}

template <uint32_t WQE_BB_CNT>
ACLSHMEM_DEVICE void aclshmemi_udma_flush_sq_wqebbs(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size)
{
    __gm__ uint8_t* wqe_addr = aclshmemi_udma_get_sq_wqebb_addr(qp_ctx_entry, cur_head, wqe_size);
    if constexpr (WQE_BB_CNT == 1) {
        dcci_cachelines(wqe_addr, wqe_size);
    } else {
        aclshmemi_udma_sq_span_t span = aclshmemi_udma_get_sq_span<WQE_BB_CNT>(cur_head);
        dcci_cachelines(wqe_addr, span.first_wqebb_cnt * wqe_size);
        if (span.second_wqebb_cnt != 0) {
            dcci_cachelines((__gm__ uint8_t*)qp_ctx_entry->buf_addr, span.second_wqebb_cnt * wqe_size);
        }
    }
}

ACLSHMEM_DEVICE void assert_qp_params_valid(__gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry)
{
    auto depth = qp_ctx_entry->depth;
    bool valid = (shm::UDMA_SQ_BASKBLK_CNT == depth);
    if (!valid) {
        AscendC::printf("sq depth [%d] is not equal with baskblkcnt [%d].\n", depth, shm::UDMA_SQ_BASKBLK_CNT);
        trap();
    }
}

ACLSHMEM_DEVICE void assert_not_self_send(uint32_t pe)
{
    if (pe == static_cast<uint32_t>(aclshmem_my_pe())) {
        AscendC::printf("udma_post_send: pe(%d) == my_pe, self-send not allowed\n", pe);
        trap();
    }
}

// Unified UDMA slot index, same signature for both builds; only the body is compile-time gated:
//   * OFF (direct): slot == pe. relay_pe ignored, myPe/rankCount not read -- zero hot-path cost.
//   * ON (relay): slot == pe * rankCount + actualRelayPe (actualRelayPe == myPe when relay_pe == -1).
// Overflow (RED-03 / TOP-03): slot indexes the [rankCount*rankCount] tables and is bounded by
// rankCount^2 << 2^32, so the uint32_t product cannot wrap; derived byte offsets widen to 64-bit at
// the pointer arithmetic (sizeof promotes to size_t) and host table sizing uses uint64_t (SlotCount()).
ACLSHMEM_DEVICE uint32_t aclshmemi_udma_compute_slot(uint32_t pe, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        uint32_t my_pe = static_cast<uint32_t>(aclshmemi_get_my_pe());
        uint32_t rank_count = static_cast<uint32_t>(aclshmemi_get_total_pe());
        uint32_t actual_relay_pe = (relay_pe == static_cast<uint32_t>(-1)) ? my_pe : relay_pe;
        return pe * rank_count + actual_relay_pe;
    } else {
        (void)relay_pe; // direct path: ignore relay_pe, no extra computation
        return pe;
    }
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_fill_contiguous_wqe(
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx, __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry,
    __gm__ aclshmemi_ubmem_info_t* remote_mem_info, __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr,
    uint32_t cur_head, uint64_t message_len, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    aclshmemi_udma_fill_sqe_ctx<T, OP_CODE, __gm__ aclshmemi_sqe_ctx_t*, CONFIG>(
        sqe_ctx, remote_addr, remote_mem_info, cur_head, params);
    __gm__ aclshmemi_sge_ctx_t* sge_ctx =
        (__gm__ aclshmemi_sge_ctx_t*)(aclshmemi_udma_get_sge_ctx<OP_CODE>((__gm__ uint8_t*)sqe_ctx));
    aclshmemi_fill_sge_ctx<T, OP_CODE>(sge_ctx, message_len, local_addr, qp_ctx_entry, params);
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_fill_multi_bb_notify_data(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, __gm__ aclshmemi_ubmem_info_t* remote_mem_info, uint32_t cur_head,
    uint32_t wqe_size, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        __gm__ uint64_t* notify_head0 = (__gm__ uint64_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sizeof(aclshmemi_sqe_ctx_t));
        __gm__ uint64_t* notify_head1 = (__gm__ uint64_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sizeof(aclshmemi_sqe_ctx_t) + sizeof(uint64_t));
        __gm__ uint64_t* notify_tail0 = (__gm__ uint64_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sizeof(aclshmemi_sqe_ctx_t) + 2 * sizeof(uint64_t));
        __gm__ uint64_t* notify_tail1 = (__gm__ uint64_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sizeof(aclshmemi_sqe_ctx_t) + 3 * sizeof(uint64_t));
        *notify_head0 = static_cast<uint64_t>(remote_mem_info->tid & 0xFFFFFU) |
                        (static_cast<uint64_t>(remote_mem_info->rmt_token_value) << 32);
        *notify_head1 = reinterpret_cast<uint64_t>(params.sig_addr);
        *notify_tail0 = params.signal;
        *notify_tail1 = 0;
    } else {
        (void)remote_mem_info;
        (void)params;
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_fill_multi_bb_sge_ctx(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, __gm__ uint8_t* local_addr, uint32_t cur_head, uint32_t wqe_size,
    uint64_t message_len, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    uint32_t sge_offset = static_cast<uint32_t>(
        (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) ?
            sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_notify_ctx_t) :
            sizeof(aclshmemi_sqe_ctx_t));
    __gm__ aclshmemi_sge_ctx_t* sge_ctx = (__gm__ aclshmemi_sge_ctx_t*)aclshmemi_udma_get_sq_wqe_offset_addr(
        qp_ctx_entry, cur_head, wqe_size, sge_offset);
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) {
        (void)local_addr;
        aclshmemi_fill_sge_header_ctx(sge_ctx, message_len, qp_ctx_entry->amo_addr);
        __gm__ T* add_data_addr = (__gm__ T*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sge_offset + sizeof(aclshmemi_sge_ctx_t));
        *add_data_addr = params.value;
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS) {
        (void)local_addr;
        aclshmemi_fill_sge_header_ctx(sge_ctx, message_len, qp_ctx_entry->amo_addr);
        __gm__ T* swap_data_addr = (__gm__ T*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sge_offset + sizeof(aclshmemi_sge_ctx_t));
        __gm__ T* cmp_data_addr = (__gm__ T*)aclshmemi_udma_get_sq_wqe_offset_addr(
            qp_ctx_entry, cur_head, wqe_size, sge_offset + sizeof(aclshmemi_sge_ctx_t) + sizeof(T));
        *swap_data_addr = params.value;
        *cmp_data_addr = params.cond;
    } else {
        aclshmemi_fill_sge_ctx<T, OP_CODE>(sge_ctx, message_len, local_addr, qp_ctx_entry, params);
    }
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_fill_multi_bb_wrap_wqe(
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx, __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry,
    __gm__ aclshmemi_ubmem_info_t* remote_mem_info, __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr,
    uint32_t cur_head, uint32_t wqe_size, uint64_t message_len, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    aclshmemi_udma_fill_sqe_base_ctx<T, OP_CODE, __gm__ aclshmemi_sqe_ctx_t*, CONFIG>(
        sqe_ctx, remote_addr, remote_mem_info, cur_head);
    aclshmemi_fill_multi_bb_notify_data<T, OP_CODE>(qp_ctx_entry, remote_mem_info, cur_head, wqe_size, params);
    aclshmemi_fill_multi_bb_sge_ctx<T, OP_CODE>(qp_ctx_entry, local_addr, cur_head, wqe_size, message_len, params);
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_fill_post_send_wqe(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, __gm__ aclshmemi_ubmem_info_t* remote_mem_info,
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t cur_head, uint32_t wqe_size, uint64_t message_len,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    constexpr uint32_t WQE_BB_CNT = get_wqe_bb_cnt<OP_CODE>();
    __gm__ aclshmemi_sqe_ctx_t* sqe_ctx = aclshmemi_udma_get_sqe_ctx(qp_ctx_entry, cur_head, wqe_size);
    if constexpr (WQE_BB_CNT == 1) {
        aclshmemi_udma_fill_contiguous_wqe<T, OP_CODE, CONFIG>(
            sqe_ctx, qp_ctx_entry, remote_mem_info, remote_addr, local_addr, cur_head, message_len, params);
        dcci_cachelines((__gm__ uint8_t*)sqe_ctx, wqe_size);
    } else {
        if (!aclshmemi_udma_sq_span_wraps<WQE_BB_CNT>(cur_head)) {
            aclshmemi_udma_fill_contiguous_wqe<T, OP_CODE, CONFIG>(
                sqe_ctx, qp_ctx_entry, remote_mem_info, remote_addr, local_addr, cur_head, message_len, params);
            dcci_cachelines((__gm__ uint8_t*)sqe_ctx, wqe_size * WQE_BB_CNT);
        } else {
            aclshmemi_udma_fill_multi_bb_wrap_wqe<T, OP_CODE, CONFIG>(
                sqe_ctx, qp_ctx_entry, remote_mem_info, remote_addr, local_addr, cur_head, wqe_size, message_len,
                params);
            aclshmemi_udma_flush_sq_wqebbs<WQE_BB_CNT>(qp_ctx_entry, cur_head, wqe_size);
        }
    }
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, OP_CODE>& params, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_require_standalone_cqe<CONFIG>();
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    ACLSHMEM_DEBUG_FUNC(assert_not_self_send, pe);
    // Unified slot computation for both builds (call site is identical, no #if here). OFF returns
    // pe (direct path); ON returns pe*N+actualRelayPe. Direct/OFF callers pass relay_pe == -1,
    // which compute_slot ignores while returning pe.
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry = aclshmemi_udma_get_qp_ctx(udma_info, slot, qp_idx);
    // Slots sharing a jetty share one state, keeping the producer index unique across the group.
    // Resolve the state pointer once; every helper below takes it as an argument instead of
    // re-loading qp_ctx_entry->state_addr.
    __gm__ aclshmemi_udma_queue_state_t* state = (__gm__ aclshmemi_udma_queue_state_t*)qp_ctx_entry->state_addr;
    auto wqe_size = qp_ctx_entry->wqe_size;
    ACLSHMEM_DEBUG_FUNC(assert_qp_params_valid, qp_ctx_entry);
    constexpr uint32_t WQE_BB_CNT = get_wqe_bb_cnt<OP_CODE>();
    aclshmemi_udma_poll_cq_when_sq_overflow(state, slot, qp_idx, WQE_BB_CNT);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_ensure_sq_credit, state, slot, qp_idx, WQE_BB_CNT);
    uint32_t cur_head = state->sq_head;
    uint32_t cqe_cnt = state->cqe_cnt;
    __gm__ aclshmemi_ubmem_info_t* remote_mem_info = aclshmemi_udma_get_mem_info(udma_info, slot, qp_idx);
    // Write SQE to HBM
    uint32_t posted_head = cur_head;
    aclshmemi_udma_fill_post_send_wqe<T, OP_CODE, CONFIG>(
        qp_ctx_entry, remote_mem_info, remote_addr, local_addr, cur_head, wqe_size, message_len, params);
    cur_head += WQE_BB_CNT;
    aclshmemi_udma_post_send_update_info(cur_head, qp_ctx_entry, state);
    // Only advance the group-wide CQE counter when this op actually requests a CQE.
    if constexpr (CONFIG.cqe != 0) {
        cqe_cnt++;
    }
    state->cqe_cnt = cqe_cnt;
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        // wq_ctx is read-only static data now; only the mutable queue state needs flushing.
        dcci_cachelines((__gm__ uint8_t*)state, sizeof(aclshmemi_udma_queue_state_t));
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_sq_wqe, qp_ctx_entry, posted_head, wqe_size, (uint32_t)sizeof(T));
}

// Rings the SQ doorbell and records the new producer index in the shared queue state. Callers
// resolve state once; the doorbell address still comes from the wq_ctx, so both are passed in.
ACLSHMEM_DEVICE void aclshmemi_udma_post_send_update_info(
    uint32_t cur_head, __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, __gm__ aclshmemi_udma_queue_state_t* state)
{
    // Ring SQ Doorbell (reference udma_update_sq_db in UDMA)
    // Note: db address is 64-bit, but we only update 32-bit value
    __gm__ uint32_t* door_bell_addr = (__gm__ uint32_t*)qp_ctx_entry->db_addr;
    st_dev(cur_head, door_bell_addr, 0);
    state->sq_head = cur_head;
}

ACLSHMEM_DEVICE void aclshmemi_udma_copy_wqe_from_ub(
    __gm__ uint8_t* dst_gm, AscendC::LocalTensor<uint8_t>& ub_local, uint32_t copy_len, uint32_t sync_id)
{
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    gm_tensor.SetGlobalBuffer(dst_gm, copy_len);
    ub_local.address_.dataLen = copy_len;
    AscendC::DataCopyExtParams copyParams{1, copy_len, 0, 0, 0};

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(gm_tensor, ub_local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

template <uint32_t WQE_BB_CNT>
ACLSHMEM_DEVICE void aclshmemi_udma_copy_wqe_from_ub_to_sq(
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t cur_head, __ubuf__ uint8_t* ub_scratch, uint32_t wqe_size,
    uint32_t sync_id)
{
    AscendC::LocalTensor<uint8_t> ub_local;
    ub_local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    if constexpr (WQE_BB_CNT == 1) {
        ub_local.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_scratch);
        aclshmemi_udma_copy_wqe_from_ub(
            aclshmemi_udma_get_sq_wqebb_addr(qp_ctx_entry, cur_head, wqe_size), ub_local, wqe_size, sync_id);
    } else {
        aclshmemi_udma_sq_span_t span = aclshmemi_udma_get_sq_span<WQE_BB_CNT>(cur_head);
        ub_local.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_scratch);
        aclshmemi_udma_copy_wqe_from_ub(
            aclshmemi_udma_get_sq_wqebb_addr(qp_ctx_entry, cur_head, wqe_size), ub_local,
            span.first_wqebb_cnt * wqe_size, sync_id);
        if (span.second_wqebb_cnt != 0) {
            ub_local.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_scratch) + span.first_wqebb_cnt * wqe_size;
            aclshmemi_udma_copy_wqe_from_ub(
                (__gm__ uint8_t*)qp_ctx_entry->buf_addr, ub_local, span.second_wqebb_cnt * wqe_size, sync_id);
        }
    }
}

static_assert(
    (sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_sge_ctx_t)) % sizeof(uint64_t) == 0,
    "UDMA WQE copy requires u64 alignment");
static_assert(ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE == 64, "UDMA aggregate data mover WQE must be one 64B WQEBB");

ACLSHMEM_DEVICE void aclshmemi_udma_poll_before_aggregate_flush(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t slot, uint32_t qp_idx, uint32_t staged_bb_cnt)
{
    uint32_t used_after_flush = state->sq_head + staged_bb_cnt - state->sq_tail;
    if (used_after_flush + ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD >= shm::UDMA_SQ_BASKBLK_CNT) {
        (void)aclshmemi_udma_poll_cq(slot, qp_idx, state->cqe_cnt);
    }
}

ACLSHMEM_DEVICE void aclshmemi_udma_flush_aggregate_wqes(
    __ubuf__ uint8_t* ub_wqe_base, __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry, uint32_t pe, uint32_t qp_idx,
    uint32_t sync_id, uint32_t pending_wqe_cnt, uint32_t wqe_size)
{
    uint32_t remaining_wqe_cnt = pending_wqe_cnt;
    if (remaining_wqe_cnt == 0) {
        return;
    }

    constexpr uint32_t depth = shm::UDMA_SQ_BASKBLK_CNT;
    constexpr uint32_t max_copy_wqe_cnt = depth - ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD;
    __gm__ aclshmemi_udma_queue_state_t* state = (__gm__ aclshmemi_udma_queue_state_t*)qp_ctx_entry->state_addr;
    ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_aggregate_batch_size, pending_wqe_cnt);
    uint32_t cur_head = state->sq_head;
    uint32_t ub_slot = 0;
    while (remaining_wqe_cnt != 0) {
        uint32_t sq_slot = cur_head % depth;
        uint32_t copy_wqe_cnt = depth - sq_slot;
        if (copy_wqe_cnt > remaining_wqe_cnt) {
            copy_wqe_cnt = remaining_wqe_cnt;
        }
        if (copy_wqe_cnt > max_copy_wqe_cnt) {
            copy_wqe_cnt = max_copy_wqe_cnt;
        }
        aclshmemi_udma_poll_before_aggregate_flush(state, pe, qp_idx, copy_wqe_cnt);
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_ensure_sq_capacity, state, pe, qp_idx, copy_wqe_cnt);

        __gm__ uint8_t* sq_addr = (__gm__ uint8_t*)(qp_ctx_entry->buf_addr + sq_slot * wqe_size);
        __ubuf__ uint8_t* ub_wqe = ub_wqe_base + ub_slot * wqe_size;
        AscendC::LocalTensor<uint8_t> ub_local;
        ub_local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ub_local.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_wqe);
        aclshmemi_udma_copy_wqe_from_ub(sq_addr, ub_local, copy_wqe_cnt * wqe_size, sync_id);

        cur_head += copy_wqe_cnt;
        ub_slot += copy_wqe_cnt;
        remaining_wqe_cnt -= copy_wqe_cnt;
        aclshmemi_udma_post_send_update_info(cur_head, qp_ctx_entry, state);
    }
    // One CQE per flushed aggregate batch (submit ops always request a CQE, checked by
    // aclshmemi_udma_require_submit_cqe). The counter lives in the shared state so the
    // group-wide poll target stays monotonic.
    state->cqe_cnt++;
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_stage_send_wqe(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* ub_scratch, aclshmemx_submit_state_t& state, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    static_assert(
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE || OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_READ,
        "UDMA aggregate action only supports UDMA_OP_WRITE / UDMA_OP_READ");
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    ACLSHMEM_DEBUG_FUNC(assert_not_self_send, pe);
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    __gm__ aclshmemi_ubmem_info_t* remote_mem_info = aclshmemi_udma_get_mem_info(udma_info, slot, qp_idx);

    uint32_t pending_count = state.pending_count;
    __ubuf__ uint8_t* ub_wqe = ub_scratch + pending_count * ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE;
    __ubuf__ aclshmemi_sqe_ctx_t* sqe_ub = (__ubuf__ aclshmemi_sqe_ctx_t*)ub_wqe;
    aclshmemi_udma_params_t<T, OP_CODE> params{};
    aclshmemi_udma_fill_sqe_ctx<T, OP_CODE, __ubuf__ aclshmemi_sqe_ctx_t*, CONFIG>(
        sqe_ub, remote_addr, remote_mem_info, pending_count, params);
    __ubuf__ aclshmemi_sge_ctx_t* sge_ub = (__ubuf__ aclshmemi_sge_ctx_t*)(ub_wqe + sizeof(aclshmemi_sqe_ctx_t));
    aclshmemi_fill_sge_header_ctx(sge_ub, message_len, reinterpret_cast<uint64_t>(local_addr));
    state.pending_count = pending_count + 1;
    (void)qp_idx;
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_submit_send_wqes(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* ub_scratch, uint32_t sync_id, aclshmemx_submit_state_t& state,
    uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_require_submit_cqe<CONFIG>();
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    aclshmemi_udma_stage_send_wqe<T, OP_CODE, CONFIG>(
        remote_addr, local_addr, pe, qp_idx, message_len, ub_scratch, state, relay_pe);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry = aclshmemi_udma_get_qp_ctx(udma_info, slot, qp_idx);
    aclshmemi_udma_flush_aggregate_wqes(
        ub_scratch, qp_ctx_entry, slot, qp_idx, sync_id, state.pending_count, ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE);
    state.pending_count = 0;
}

// ---- MTE3-staged WQE construction (PIPE_MTE3 path) -------------------------------
// The default PIPE_S path above writes the SQE/SGE directly to HBM via scalar stores
// + dcci_cachelines. The MTE3 path stages the full WQE block in caller-provided UB
// scratch and lands it on the SQ ring with a single DataCopyPad. Useful for hot
// loops where the per-iteration scalar->HBM bursts dominate.
//
// MTE3 path only supports UDMA_OP_WRITE / UDMA_OP_WRITE_WITH_NOTIFY / UDMA_OP_READ
// (the data-mover opcodes). FAA / CAS / WRITE_WITH_REDUCE remain on PIPE_S because
// their SGE side fills GM-resident AMO data which is incompatible with UB staging.
template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send_mte3(
    __gm__ uint8_t* remote_addr, __gm__ uint8_t* local_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* ub_scratch, uint32_t sync_id, const aclshmemi_udma_params_t<T, OP_CODE>& params = {},
    uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_require_standalone_cqe<CONFIG>();
    static_assert(
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE ||
            OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY ||
            OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_READ,
        "PIPE_MTE3 WQE path only supports UDMA_OP_WRITE / UDMA_OP_WRITE_WITH_NOTIFY / UDMA_OP_READ");

    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    // Unified slot computation for both builds (call site is identical, no #if here). OFF returns
    // pe (direct path); ON returns pe*N+actualRelayPe. Direct/OFF callers pass relay_pe == -1,
    // which compute_slot ignores while returning pe.
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry = aclshmemi_udma_get_qp_ctx(udma_info, slot, qp_idx);
    // Slots sharing a jetty share one state, keeping the producer index unique across the group.
    // Resolve the state pointer once; every helper below takes it as an argument instead of
    // re-loading qp_ctx_entry->state_addr.
    __gm__ aclshmemi_udma_queue_state_t* state = (__gm__ aclshmemi_udma_queue_state_t*)qp_ctx_entry->state_addr;
    auto wqe_size = qp_ctx_entry->wqe_size;
    ACLSHMEM_DEBUG_FUNC(assert_qp_params_valid, qp_ctx_entry);
    constexpr uint32_t WQE_BB_CNT = get_wqe_bb_cnt<OP_CODE>();
    aclshmemi_udma_poll_cq_when_sq_overflow(state, slot, qp_idx, WQE_BB_CNT);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_ensure_sq_credit, state, slot, qp_idx, WQE_BB_CNT);
    uint32_t cur_head = state->sq_head;
    uint32_t cqe_cnt = state->cqe_cnt;

    __gm__ aclshmemi_ubmem_info_t* remote_mem_info = aclshmemi_udma_get_mem_info(udma_info, slot, qp_idx);

    // Stage WQE (SQE + optional notify + SGE) in caller's UB scratch. Reuse the
    // address-space-templated fill helper so SQE field assignments are not duplicated.
    __ubuf__ aclshmemi_sqe_ctx_t* sqeUb = (__ubuf__ aclshmemi_sqe_ctx_t*)ub_scratch;
    aclshmemi_udma_fill_sqe_ctx<T, OP_CODE, __ubuf__ aclshmemi_sqe_ctx_t*, CONFIG>(
        sqeUb, remote_addr, remote_mem_info, cur_head, params);

    constexpr size_t SGE_OFF = (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) ?
                                   sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_notify_ctx_t) :
                                   sizeof(aclshmemi_sqe_ctx_t);
    __ubuf__ aclshmemi_sge_ctx_t* sge_ub = (__ubuf__ aclshmemi_sge_ctx_t*)((__ubuf__ uint8_t*)ub_scratch + SGE_OFF);
    // OP_CODE is restricted to non-AMO/non-REDUCE here, so the SGE only needs
    // len + va. The full FAA/CAS/REDUCE fan-out is intentionally unused in this path.
    aclshmemi_fill_sge_header_ctx(sge_ub, message_len, reinterpret_cast<uint64_t>(local_addr));

    // DMA UB -> SQ ring entry. The helper keeps the one-BB path as one copy and
    // splits only when a multi-BB WQE reaches the ring end.
    uint32_t posted_head = cur_head;
    aclshmemi_udma_copy_wqe_from_ub_to_sq<WQE_BB_CNT>(qp_ctx_entry, cur_head, ub_scratch, wqe_size, sync_id);

    cur_head += WQE_BB_CNT;
    aclshmemi_udma_post_send_update_info(cur_head, qp_ctx_entry, state);
    // Only advance the group-wide CQE counter when this op actually requests a CQE.
    if constexpr (CONFIG.cqe != 0) {
        cqe_cnt++;
    }
    state->cqe_cnt = cqe_cnt;
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        // wq_ctx is read-only static data now; only the mutable queue state needs flushing.
        dcci_cachelines((__gm__ uint8_t*)state, sizeof(aclshmemi_udma_queue_state_t));
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_sq_wqe, qp_ctx_entry, posted_head, wqe_size, (uint32_t)sizeof(T));
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_write_mte3(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    __ubuf__ uint8_t* ub_scratch, uint32_t sync_id, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, ub_scratch, sync_id, {}, relay_pe);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify_mte3(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>& params,
    __ubuf__ uint8_t* ub_scratch, uint32_t sync_id)
{
    aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, ub_scratch, sync_id, params);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_write(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, {}, relay_pe);
}

template <
    typename T, aclshmemi_udma_opcode_t OP_CODE,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t pe, uint32_t qp_idx, uint64_t message_len,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), pe, qp_idx,
        message_len, params);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_read(
    __gm__ T* dest_dma_addr, __gm__ T* src_dma_addr, uint32_t src_pe, uint32_t qp_idx, uint64_t message_len,
    uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
        reinterpret_cast<__gm__ uint8_t*>(src_dma_addr), reinterpret_cast<__gm__ uint8_t*>(dest_dma_addr), src_pe,
        qp_idx, message_len, {}, relay_pe);
}

// Debug-only: a null state means quiet was called on the self PE or an unconnected slot. Release
// builds trust the caller, as in post_send/poll_cq.
ACLSHMEM_DEVICE void aclshmemi_udma_debug_check_state_bound(
    __gm__ aclshmemi_udma_queue_state_t* state, uint32_t slot, uint32_t qp_idx)
{
    if (state == nullptr) {
        aclshmemi_kernel_abort(
            "UDMA queue state unbound: slot=%u qp_idx=%u (quiet on self PE or slot without a channel)\n", slot, qp_idx);
    }
}

// Waits for all submitted UDMA WQEs targeting `pe`. See the @note blocks on the declaration in
// include/device/gm2gm/engine/shmem_device_udma.h for the shared-jetty semantics: quiet(any pe)
// waits for the whole group, and intra-group post_send must be serialized by the caller.
ACLSHMEM_DEVICE void aclshmemx_udma_quiet(int pe)
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    uint32_t qp_num = udma_info->qp_num;
    uint32_t actual_pe = static_cast<uint32_t>(pe);
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        uint32_t rank_count = static_cast<uint32_t>(aclshmemi_get_total_pe());
        // Drain every (pe, relay_pe) slot. Slots that never produced a CQE have cqe_cnt==0
        // and poll_cq returns immediately. UDMA does not support self ops (pe == myPe), so the
        // (pe, pe) diagonal is never posted to and host leaves it unfilled -- skip it.
        for (uint32_t relay_pe = 0; relay_pe < rank_count; ++relay_pe) {
            if (relay_pe == actual_pe) {
                continue;
            }
            uint32_t slot = actual_pe * rank_count + relay_pe;
            __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
                (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (slot * qp_num + 0) * sizeof(aclshmemi_udma_wq_ctx_t));
            __gm__ aclshmemi_udma_queue_state_t* state = (__gm__ aclshmemi_udma_queue_state_t*)qp_ctx_entry->state_addr;
            ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_state_bound, state, slot, 0U);
            uint32_t cqe_cnt = state->cqe_cnt;
            if (cqe_cnt == 0) {
                continue;
            }
            aclshmemi_udma_poll_cq(slot, 0, cqe_cnt);
        }
    } else {
        // Original direct path: poll only the single slot == pe. Does not read rankCount.
        __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
            (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (actual_pe * qp_num + 0) * sizeof(aclshmemi_udma_wq_ctx_t));
        // Under sharing cqe_cnt is the group's total, so the wait widens to the whole group.
        __gm__ aclshmemi_udma_queue_state_t* state = (__gm__ aclshmemi_udma_queue_state_t*)qp_ctx_entry->state_addr;
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_state_bound, state, actual_pe, 0U);
        aclshmemi_udma_poll_cq(actual_pe, 0, state->cqe_cnt);
    }
}

ACLSHMEM_DEVICE void aclshmemi_udma_debug_check_qp_index(uint32_t qp_idx)
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    if (qp_idx >= udma_info->qp_num) {
        aclshmemi_kernel_abort("Invalid UDMA qp_idx=%u, qp_num=%u\n", qp_idx, udma_info->qp_num);
    }
}

ACLSHMEM_DEVICE void aclshmemx_udma_qp_quiet(int pe, uint32_t qp_idx)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        (void)pe;
        (void)qp_idx;
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
        __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
        const uint32_t actual_pe = static_cast<uint32_t>(pe);
        __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
            (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (actual_pe * udma_info->qp_num + qp_idx) *
                                                                sizeof(aclshmemi_udma_wq_ctx_t));
        __gm__ aclshmemi_udma_queue_state_t* state = (__gm__ aclshmemi_udma_queue_state_t*)qp_ctx_entry->state_addr;
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_state_bound, state, actual_pe, qp_idx);
        aclshmemi_udma_poll_cq(actual_pe, qp_idx, state->cqe_cnt);
    }
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_get_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_read<uint8_t, CONFIG>((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(src, pe);
            // For UDMA_OP_READ, the SQE's "remote_addr" slot carries the src (remote)
            // and "local_addr" carries dst (local), matching aclshmemi_udma_read().
            aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_get_nbi<T, CONFIG>(dst, src, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id)
{
    aclshmemx_udma_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, sync_id);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action get requires WQE_PIPE == PIPE_MTE3");
    aclshmemi_udma_require_defer_no_cqe<CONFIG>();
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        (void)sync_id;
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_stage_send_wqe<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(dst), static_cast<uint32_t>(pe),
            0, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action get requires WQE_PIPE == PIPE_MTE3");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_submit_send_wqes<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(dst), static_cast<uint32_t>(pe),
            0, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_defer_t action)
{
    aclshmemx_udma_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_submit_t action)
{
    aclshmemx_udma_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, sync_id, action);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemi_udma_put_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_write<uint8_t, CONFIG>((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            __ubuf__ uint8_t* ub_scratch = reinterpret_cast<__ubuf__ uint8_t*>(buf);
            auto ptr = aclshmem_ptr(dst, pe);
            aclshmemi_udma_write_mte3<T, CONFIG>(
                (__gm__ T*)ptr, src, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T), ub_scratch, sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_put_nbi<T, CONFIG>(dst, src, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        auto remote_src = aclshmem_ptr(src, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
                (__gm__ uint8_t*)remote_src, (__gm__ uint8_t*)dst, static_cast<uint32_t>(pe), qp_idx,
                elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_read<uint8_t, CONFIG>(
                (__gm__ uint8_t*)dst, (__gm__ uint8_t*)remote_src, static_cast<uint32_t>(pe), qp_idx,
                elem_size * sizeof(T));
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
{
    aclshmemx_udma_qp_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, qp_idx, sync_id);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA QP aggregate action get requires WQE_PIPE == PIPE_MTE3");
    aclshmemi_udma_require_defer_no_cqe<CONFIG>();
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        (void)sync_id;
        auto remote_src = aclshmem_ptr(src, pe);
        aclshmemi_udma_stage_send_wqe<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(remote_src), reinterpret_cast<__gm__ uint8_t*>(dst),
            static_cast<uint32_t>(pe), qp_idx, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf),
            action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA QP aggregate action get requires WQE_PIPE == PIPE_MTE3");
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        auto remote_src = aclshmem_ptr(src, pe);
        aclshmemi_udma_submit_send_wqes<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(remote_src), reinterpret_cast<__gm__ uint8_t*>(dst),
            static_cast<uint32_t>(pe), qp_idx, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id,
            action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id, aclshmemx_defer_t action)
{
    aclshmemx_udma_qp_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, qp_idx, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id, aclshmemx_submit_t action)
{
    aclshmemx_udma_qp_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, qp_idx, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        auto remote_dst = aclshmem_ptr(dst, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            aclshmemi_udma_write_mte3<T, CONFIG>(
                (__gm__ T*)remote_dst, src, static_cast<uint32_t>(pe), qp_idx, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_write<uint8_t, CONFIG>(
                (__gm__ uint8_t*)remote_dst, (__gm__ uint8_t*)src, static_cast<uint32_t>(pe), qp_idx,
                elem_size * sizeof(T));
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
{
    aclshmemx_udma_qp_put_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, qp_idx, sync_id);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA QP aggregate action put requires WQE_PIPE == PIPE_MTE3");
    aclshmemi_udma_require_defer_no_cqe<CONFIG>();
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        (void)sync_id;
        auto remote_dst = aclshmem_ptr(dst, pe);
        aclshmemi_udma_stage_send_wqe<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(remote_dst), reinterpret_cast<__gm__ uint8_t*>(src),
            static_cast<uint32_t>(pe), qp_idx, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf),
            action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA QP aggregate action put requires WQE_PIPE == PIPE_MTE3");
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        auto remote_dst = aclshmem_ptr(dst, pe);
        aclshmemi_udma_submit_send_wqes<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(remote_dst), reinterpret_cast<__gm__ uint8_t*>(src),
            static_cast<uint32_t>(pe), qp_idx, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id,
            action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id, aclshmemx_defer_t action)
{
    aclshmemx_udma_qp_put_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, qp_idx, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id, aclshmemx_submit_t action)
{
    aclshmemx_udma_qp_put_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, qp_idx, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action put requires WQE_PIPE == PIPE_MTE3");
    aclshmemi_udma_require_defer_no_cqe<CONFIG>();
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        (void)sync_id;
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_stage_send_wqe<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(src), static_cast<uint32_t>(pe),
            0, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action put requires WQE_PIPE == PIPE_MTE3");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_submit_send_wqes<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(src), static_cast<uint32_t>(pe),
            0, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, action.state);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto dst_phy_addr = (__gm__ T*)dst.GetPhyAddr();
        auto src_phy_addr = (__gm__ T*)src.GetPhyAddr();
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            __ubuf__ uint8_t* ub_scratch = reinterpret_cast<__ubuf__ uint8_t*>(buf.GetPhyAddr());
            auto ptr = aclshmem_ptr(dst_phy_addr, pe);
            aclshmemi_udma_write_mte3<T, CONFIG>(
                (__gm__ T*)ptr, src_phy_addr, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T), ub_scratch, sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_put_nbi<T, CONFIG>(dst_phy_addr, src_phy_addr, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_defer_t action)
{
    aclshmemx_udma_put_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_submit_t action)
{
    aclshmemx_udma_put_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, sync_id, action);
}

ACLSHMEM_DEVICE bool aclshmemi_udma_relay_params_valid(int pe, int relay_pe)
{
    int rank_count = aclshmemi_get_total_pe();
    int my_pe = aclshmemi_get_my_pe();
    if (pe < 0 || pe >= rank_count || relay_pe < 0 || relay_pe >= rank_count || pe == relay_pe || pe == my_pe ||
        relay_pe == my_pe) {
        ACLSHMEM_DEBUG_FUNC(
            aclshmemi_kernel_abort,
            "udma relay: invalid pe=%d relay_pe=%d (myPe=%d rankCount=%d); "
            "require 0<=actual,relay<rankCount, actual!=relay, neither equals myPe\n",
            pe, relay_pe, my_pe, rank_count);
        return false;
    }
    return true;
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        // sizeof(T) == 0 is always false but depends on T, so it only fires when this template is
        // actually instantiated (i.e. relay API is called) rather than at parse time.
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay put: pe=%d relay_pe=%d\n", pe, relay_pe);
        if (!aclshmemi_udma_relay_params_valid(pe, relay_pe)) {
            return;
        }
        auto ptr = aclshmem_ptr(dst, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            aclshmemi_udma_write_mte3<uint8_t, CONFIG>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, static_cast<uint32_t>(relay_pe));
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_write<uint8_t, CONFIG>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                static_cast<uint32_t>(relay_pe));
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action relay put requires WQE_PIPE == PIPE_MTE3");
    aclshmemi_udma_require_defer_no_cqe<CONFIG>();
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay put: pe=%d relay_pe=%d\n", pe, relay_pe);
        if (!aclshmemi_udma_relay_params_valid(pe, relay_pe)) {
            return;
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_stage_send_wqe<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(src), static_cast<uint32_t>(pe),
            0u, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), action.state,
            static_cast<uint32_t>(relay_pe));
        (void)sync_id;
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action relay put requires WQE_PIPE == PIPE_MTE3");
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay put: pe=%d relay_pe=%d\n", pe, relay_pe);
        if (!aclshmemi_udma_relay_params_valid(pe, relay_pe)) {
            return;
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_submit_send_wqes<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(src), static_cast<uint32_t>(pe),
            0u, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, action.state,
            static_cast<uint32_t>(relay_pe));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemx_udma_relay_put_nbi<T, WQE_PIPE, CONFIG>(
            (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
            elem_size, pe, relay_pe, sync_id);
    } else {
        (void)dst;
        (void)src;
        (void)buf;
        (void)elem_size;
        (void)pe;
        (void)relay_pe;
        (void)sync_id;
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_defer_t action)
{
    aclshmemx_udma_relay_put_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, relay_pe, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_submit_t action)
{
    aclshmemx_udma_relay_put_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, relay_pe, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        // sizeof(T) == 0 is always false but depends on T, so it only fires when this template is
        // actually instantiated (i.e. relay API is called) rather than at parse time.
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay get: pe=%d relay_pe=%d\n", pe, relay_pe);
        if (!aclshmemi_udma_relay_params_valid(pe, relay_pe)) {
            return;
        }
        auto ptr = aclshmem_ptr(src, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            // For UDMA_OP_READ the SQE's "remote_addr" slot carries src (remote) and
            // "local_addr" carries dst (local), matching aclshmemi_udma_read().
            aclshmemi_udma_post_send_mte3<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, {}, static_cast<uint32_t>(relay_pe));
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_read<uint8_t, CONFIG>(
                (__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                static_cast<uint32_t>(relay_pe));
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_defer_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action relay get requires WQE_PIPE == PIPE_MTE3");
    aclshmemi_udma_require_defer_no_cqe<CONFIG>();
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay get: pe=%d relay_pe=%d\n", pe, relay_pe);
        if (!aclshmemi_udma_relay_params_valid(pe, relay_pe)) {
            return;
        }
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_stage_send_wqe<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(dst), static_cast<uint32_t>(pe),
            0u, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), action.state,
            static_cast<uint32_t>(relay_pe));
        (void)sync_id;
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_submit_t action)
{
    static_assert(WQE_PIPE == PIPE_MTE3, "UDMA aggregate action relay get requires WQE_PIPE == PIPE_MTE3");
    if constexpr (!ACLSHMEM_RELAY_SUPPORTED) {
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay get: pe=%d relay_pe=%d\n", pe, relay_pe);
        if (!aclshmemi_udma_relay_params_valid(pe, relay_pe)) {
            return;
        }
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_submit_send_wqes<T, aclshmemi_udma_opcode_t::UDMA_OP_READ, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), reinterpret_cast<__gm__ uint8_t*>(dst), static_cast<uint32_t>(pe),
            0u, elem_size * sizeof(T), reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, action.state,
            static_cast<uint32_t>(relay_pe));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemx_udma_relay_get_nbi<T, WQE_PIPE, CONFIG>(
            (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
            elem_size, pe, relay_pe, sync_id);
    } else {
        (void)dst;
        (void)src;
        (void)buf;
        (void)elem_size;
        (void)pe;
        (void)relay_pe;
        (void)sync_id;
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_defer_t action)
{
    aclshmemx_udma_relay_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, relay_pe, sync_id, action);
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_submit_t action)
{
    aclshmemx_udma_relay_get_nbi<T, WQE_PIPE, CONFIG>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, relay_pe, sync_id, action);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        auto sig_addr_dst = aclshmem_ptr(sig_addr, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
            .sig_addr = (__gm__ uint64_t*)(sig_addr_dst), .signal = signal};
        aclshmemi_udma_write_notify<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY, CONFIG>(
            (__gm__ T*)ptr, src, pe, 0, elem_size * sizeof(T), signal_params);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

// Buf-taking overload. PIPE_MTE3 (default) stages the WRITE_WITH_NOTIFY WQE in the
// caller-provided UB scratch (size one full WQE block; wqe_size * 2 bytes, 128 B is
// safe for the current SQ basebk_shift). PIPE_S falls through to the no-buf path and
// ignores buf/sync_id, mirroring the put_nbi(buf) overload's S/MTE3 split.
template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    __ubuf__ uint8_t* buf, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(dst, pe);
            auto sig_addr_dst = aclshmem_ptr(sig_addr, pe);
            aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
                .sig_addr = (__gm__ uint64_t*)(sig_addr_dst), .signal = signal};
            aclshmemi_udma_write_notify_mte3<T, CONFIG>(
                (__gm__ T*)ptr, src, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T), signal_params, buf, sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemx_udma_put_signal_nbi<T, CONFIG>(dst, src, elem_size, sig_addr, signal, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    uint32_t qp_idx)
{
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
        auto remote_dst = aclshmem_ptr(dst, pe);
        auto remote_sig_addr = aclshmem_ptr(sig_addr, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
            .sig_addr = (__gm__ uint64_t*)remote_sig_addr, .signal = signal};
        aclshmemi_udma_write_notify<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY, CONFIG>(
            (__gm__ T*)remote_dst, src, static_cast<uint32_t>(pe), qp_idx, elem_size * sizeof(T), signal_params);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, pipe_t WQE_PIPE, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    uint32_t qp_idx, __ubuf__ uint8_t* buf, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        aclshmemi_kernel_abort("QP-specific UDMA APIs require ACLSHMEM_RELAY_SUPPORT=OFF\n");
    } else if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_udma_debug_check_qp_index, qp_idx);
            auto remote_dst = aclshmem_ptr(dst, pe);
            auto remote_sig_addr = aclshmem_ptr(sig_addr, pe);
            aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
                .sig_addr = (__gm__ uint64_t*)remote_sig_addr, .signal = signal};
            aclshmemi_udma_write_notify_mte3<T, CONFIG>(
                (__gm__ T*)remote_dst, src, static_cast<uint32_t>(pe), qp_idx, elem_size * sizeof(T), signal_params,
                buf, sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemx_udma_qp_put_signal_nbi<T, CONFIG>(dst, src, elem_size, sig_addr, signal, pe, qp_idx);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "QP-specific UDMA APIs require Ascend950 direct mode\n");
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE constexpr bool aclshmemi_udma_check_atomic_len()
{
    size_t atomic_len = sizeof(T);
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) {
        if (atomic_len != 4 && atomic_len != 8) {
            return false;
        }
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS) {
        if (atomic_len != 4 && atomic_len != 8 && atomic_len != 16) {
            return false;
        }
    }
    return true;
}

ACLSHMEM_DEVICE uint64_t aclshmemi_udma_get_amo_addr(uint32_t pe, uint32_t qp_idx)
{
    __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
    uint32_t qp_num = udma_info->qp_num;
    // Atomic ops only run on the direct path, so reuse the same slot as aclshmemi_udma_post_send's
    // default (relay = self). Gate the layout exactly like compute_slot: OFF is the original
    // single-dimension table where slot = pe; ON is the N*N table where the direct bucket
    // is (actual=pe, relay=self) = pe*N + myPe. Using the N*N formula on OFF would read amo_addr
    // from the wrong (out-of-bounds) slot and corrupt atomic fetch data.
    uint32_t slot = aclshmemi_udma_compute_slot(pe);
    __gm__ aclshmemi_udma_qp_table_t* tbl = aclshmemi_udma_active_table(udma_info);
    __gm__ aclshmemi_udma_wq_ctx_t* qp_ctx_entry =
        (__gm__ aclshmemi_udma_wq_ctx_t*)(tbl->sq_ptr + (slot * qp_num + qp_idx) * sizeof(aclshmemi_udma_wq_ctx_t));
    auto amo_addr = qp_ctx_entry->amo_addr;
    return amo_addr;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_udma_get_amo_addr_value(uint64_t amo_addr)
{
    dcci_cachelines((__gm__ uint8_t*)amo_addr, sizeof(T));
    __gm__ T* fetch_addr = reinterpret_cast<__gm__ T*>(amo_addr);
    T fetch_data = *fetch_addr;
    return fetch_data;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_udma_get_atomic_fetch_data(uint32_t pe, uint32_t qp_idx)
{
    auto amo_addr = aclshmemi_udma_get_amo_addr(pe, qp_idx);
    return aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_add(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic add\n", sizeof(T));
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> atomic_params{
            .value = value, .cond = 0};
        if constexpr (AscendC::IsSameType<T, float>::value) { // float使用write with reduce逻辑处理
            aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE, CONFIG>(
                reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);
        } else {
            aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA, CONFIG>(
                reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic fetch add\n", sizeof(T));
        }

        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> atomic_params{
            .value = value, .cond = 0};
        aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);

        aclshmemx_udma_quiet(pe);
        return aclshmemi_udma_get_atomic_fetch_data<T>(pe, 0);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
        return 0;
    }
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OP_CAS>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic compare swap\n", sizeof(T));
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> cas_params{
            .value = value, .cond = cond};
        aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_CAS, CONFIG>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), cas_params);
        aclshmemx_udma_quiet(pe);
        return aclshmemi_udma_get_atomic_fetch_data<T>(pe, 0);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
        return 0;
    }
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch(__gm__ T* dst, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    return aclshmemx_udma_atomic_fetch_add<T, CONFIG>(dst, 0, pe);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_set(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, value, pe) == old_value) {
            return;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_set timeout!\n");
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_swap(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_swap timeout!\n");
    return 0;
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_inc(__gm__ T* dst, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    return aclshmemx_udma_atomic_fetch_add<T, CONFIG>(dst, 1, pe);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_inc(__gm__ T* dst, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    aclshmemx_udma_atomic_add<T, CONFIG>(dst, 1, pe);
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value & value;
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_and timeout!\n");
    return 0;
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_and(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value & value;
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, new_value, pe) == old_value) {
            return;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_and timeout!\n");
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value | value;
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_or timeout!\n");
    return 0;
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_or(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value | value;
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, new_value, pe) == old_value) {
            return;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_or timeout!\n");
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value ^ value;
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_xor timeout!\n");
    return 0;
}

template <typename T, const aclshmemx_udma_op_config_t& CONFIG>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_xor(__gm__ T* dst, T value, int32_t pe)
{
    aclshmemi_udma_require_cqe<CONFIG>();
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi<T, CONFIG>((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value ^ value;
        if (aclshmemx_udma_atomic_compare_swap<T, CONFIG>(dst, old_value, new_value, pe) == old_value) {
            return;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_xor timeout!\n");
}

#endif
