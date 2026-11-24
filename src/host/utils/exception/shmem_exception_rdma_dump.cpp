/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/exception/shmem_exception_rdma_dump.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "dl_acl_api.h"
#include "dl_hccp_def.h"
#include "init/shmemi_init.h"
#include "host_device/shmemi_rdma_cqe_layout.h"
#include "shmemi_host_common.h"
#include "utils/exception/shmemi_device_rdma_exception_report_kernel.h"
#include "utils/exception/shmem_exception_rdma_dump_internal.h"

namespace {
struct aclshmemi_rdma_exception_source_t {
    uint64_t instance_id;
    uint32_t rank_id;
    uint32_t rank_count;
    uint64_t qp_info_address;
};

using shm::AiQpRMACQ;
using shm::AiQpRMAQueueInfo;
using shm::AiQpRMAWQ;

class DeviceEntryBuffer {
public:
    DeviceEntryBuffer() = default;
    DeviceEntryBuffer(const DeviceEntryBuffer&) = delete;
    DeviceEntryBuffer& operator=(const DeviceEntryBuffer&) = delete;

    ~DeviceEntryBuffer()
    {
        if (data == nullptr) {
            return;
        }
        const int ret = shm::DlAclApi::AclrtFree(data);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("[EXCEPTION][RDMA] result=partial reason=free_snapshot_failed ret=" << ret);
        }
    }

    void* data{nullptr};
};

bool ReadEntry(
    uint32_t entry_type, uint64_t src, size_t size, aclshmemi_rdma_exception_report_entry_t& entry,
    aclshmemi_rdma_exception_report_entry_t* device_entry, const char* name)
{
    const bool raw_entry = entry_type == ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_RAW ||
                           entry_type == ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQE_RAW;
    const bool valid_type = entry_type <= ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQE_RAW;
    if (!valid_type || src == 0 || size == 0 || (raw_entry && size > ACLSHMEMI_RDMA_EXCEPTION_REPORT_MAX_RAW_SIZE) ||
        device_entry == nullptr) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=invalid_" << name);
        return false;
    }
    int ret = shm::DlAclApi::AclrtMemset(
        device_entry, sizeof(aclshmemi_rdma_exception_report_entry_t), 0,
        sizeof(aclshmemi_rdma_exception_report_entry_t));
    if (ret == ACLSHMEM_SUCCESS) {
        ret = aclshmemi_rdma_exception_report_read_entry_on_stream(
            entry_type, src, size, device_entry, static_cast<aclrtStream>(g_state_host.default_stream));
    }
    if (ret == ACLSHMEM_SUCCESS) {
        ret = shm::DlAclApi::AclrtSynchronizeStream(g_state_host.default_stream);
    }
    if (ret == ACLSHMEM_SUCCESS) {
        ret = shm::DlAclApi::AclrtMemcpy(&entry, sizeof(entry), device_entry, sizeof(entry), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=read_" << name << "_failed ret=" << ret);
        return false;
    }
    if (entry.ret != ACLSHMEMI_RDMA_EXCEPTION_REPORT_SUCCESS || entry.entry_type != entry_type) {
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA] result=unavailable reason=invalid_" << name << "_snapshot entryRet=" << entry.ret
                                                                   << " entryType=" << entry.entry_type);
        return false;
    }
    return true;
}

bool ReadSnapshot(
    void* dst, size_t size, uint64_t src, aclshmemi_rdma_exception_report_entry_t* device_entry, const char* name,
    uint32_t entry_type = ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_RAW)
{
    if (dst == nullptr) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=null_" << name << "_destination");
        return false;
    }
    aclshmemi_rdma_exception_report_entry_t entry{};
    if (!ReadEntry(entry_type, src, size, entry, device_entry, name) || entry.raw.addr != src ||
        entry.raw.size != size) {
        return false;
    }
    std::copy_n(entry.raw.data, size, static_cast<uint8_t*>(dst));
    return true;
}

bool ReadWqSnapshot(uint64_t src, AiQpRMAWQ& dst, aclshmemi_rdma_exception_report_entry_t* device_entry)
{
    aclshmemi_rdma_exception_report_entry_t entry{};
    if (!ReadEntry(
            ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_WQ, src, sizeof(AiQpRMAWQ), entry, device_entry, "sq_context")) {
        return false;
    }
    dst.wqn = entry.wq.wqn;
    dst.bufAddr = entry.wq.buf_addr;
    dst.wqeSize = entry.wq.wqe_size;
    dst.depth = entry.wq.depth;
    dst.headAddr = entry.wq.head_addr;
    dst.tailAddr = entry.wq.tail_addr;
    dst.dbMode = static_cast<shm::DBMode>(entry.wq.db_mode);
    dst.dbAddr = entry.wq.db_addr;
    dst.sl = entry.wq.sl;
    dst.atomicAddr = entry.wq.amo_addr;
    dst.atomicLkey = entry.wq.amo_lkey;
    dst.dbSwVa = entry.wq.db_sw_addr;
    dst.mtuShift = entry.wq.mtu_shift;
    dst.dbCos = entry.wq.db_cos;
    return true;
}

bool ReadCqSnapshot(uint64_t src, AiQpRMACQ& dst, aclshmemi_rdma_exception_report_entry_t* device_entry)
{
    aclshmemi_rdma_exception_report_entry_t entry{};
    if (!ReadEntry(
            ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQ, src, sizeof(AiQpRMACQ), entry, device_entry, "scq_context")) {
        return false;
    }
    dst.cqn = entry.cq.cqn;
    dst.bufAddr = entry.cq.buf_addr;
    dst.cqeSize = entry.cq.cqe_size;
    dst.depth = entry.cq.depth;
    dst.headAddr = entry.cq.head_addr;
    dst.tailAddr = entry.cq.tail_addr;
    dst.dbMode = static_cast<shm::DBMode>(entry.cq.db_mode);
    dst.dbAddr = entry.cq.db_addr;
    dst.cqAttrFlags = entry.cq.cq_attr_flags;
    dst.dbSwVa = entry.cq.db_sw_addr;
    return true;
}

bool ValidQueueInfo(const AiQpRMAQueueInfo& info, const aclshmemi_rdma_exception_source_t& source)
{
    if (info.count == 0U || info.count > ACLSHMEM_MAX_QP_NUM || info.sq == nullptr || info.scq == nullptr) {
        return false;
    }
    return source.rank_count <= std::numeric_limits<uint32_t>::max() / info.count;
}

bool ValidCq(const AiQpRMACQ& cq)
{
    const uint32_t stride =
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        cq.cqeSize == 0 ? 64U : cq.cqeSize;
#else
        cq.cqeSize;
#endif
    return aclshmemi::exception::rdma::IsValidCqMetadata(cq.bufAddr, cq.tailAddr, cq.depth, stride);
}

uint32_t CqeStride(const AiQpRMACQ& cq)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    return cq.cqeSize == 0 ? 64U : cq.cqeSize;
#else
    return cq.cqeSize;
#endif
}

uint32_t CqRingDepth(const AiQpRMACQ& cq) { return cq.depth; }

std::string HexSummary(const std::vector<uint8_t>& raw)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < raw.size(); ++i) {
        if (i != 0) {
            out << ' ';
        }
        out << std::setw(2) << static_cast<unsigned>(raw[i]);
    }
    return out.str();
}

uint32_t ReadLe32(const std::vector<uint8_t>& raw, size_t offset)
{
    if (offset > raw.size() || raw.size() - offset < sizeof(uint32_t)) {
        return 0;
    }
    return static_cast<uint32_t>(raw[offset]) | (static_cast<uint32_t>(raw[offset + 1]) << 8U) |
           (static_cast<uint32_t>(raw[offset + 2]) << 16U) | (static_cast<uint32_t>(raw[offset + 3]) << 24U);
}

uint64_t ReadLe64(const std::vector<uint8_t>& raw, size_t offset)
{
    uint64_t value = 0;
    for (uint32_t i = 0; i < sizeof(uint64_t) && offset + i < raw.size(); ++i) {
        value |= static_cast<uint64_t>(raw[offset + i]) << (i * 8U);
    }
    return value;
}

uint32_t ReadBe32(const std::vector<uint8_t>& raw, size_t offset)
{
    if (offset > raw.size() || raw.size() - offset < sizeof(uint32_t)) {
        return 0;
    }
    return (static_cast<uint32_t>(raw[offset]) << 24U) | (static_cast<uint32_t>(raw[offset + 1]) << 16U) |
           (static_cast<uint32_t>(raw[offset + 2]) << 8U) | static_cast<uint32_t>(raw[offset + 3]);
}

uint64_t ReadBe64(const std::vector<uint8_t>& raw, size_t offset)
{
    uint64_t value = 0;
    for (uint32_t i = 0; i < sizeof(uint64_t) && offset + i < raw.size(); ++i) {
        value = (value << 8U) | raw[offset + i];
    }
    return value;
}

const char* OpcodeName(uint32_t opcode)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    switch (opcode) {
        case 0U:
            return "RDMA_READ";
        case 1U:
            return "RDMA_WRITE";
        case 2U:
            return "RDMA_WRITE_WITH_IMM";
        default:
            return "UNKNOWN";
    }
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    switch (opcode) {
        case 4U:
            return "RDMA_WRITE";
        case 8U:
            return "RDMA_READ";
        default:
            return "UNKNOWN";
    }
#else
    switch (opcode) {
        case 3U:
            return "RDMA_WRITE";
        case 4U:
            return "RDMA_WRITE_WITH_IMM";
        case 5U:
            return "RDMA_READ";
        default:
            return "UNKNOWN";
    }
#endif
}

std::string FormatCqeRaw(const std::vector<uint8_t>& raw, const aclshmemi_rdma_exception_report_cqe_t& cqe)
{
    std::ostringstream out;
    out << "decoded={" << aclshmemi::exception::rdma::FormatCqe(cqe)
        << " validBytes=" << std::min<size_t>(raw.size(), sizeof(aclshmemi_cqe_ctx));
    if ((cqe.valid_fields & ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID) != 0U) {
        out << " opcodeName=" << OpcodeName(cqe.opcode);
    }
    if (raw.size() > sizeof(aclshmemi_cqe_ctx)) {
        out << " reservedBytes=" << raw.size() - sizeof(aclshmemi_cqe_ctx);
    }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    out << " dw0=0x" << std::hex << ReadLe32(raw, 0U) << " dw1=0x" << ReadLe32(raw, 4U) << " dw2=0x"
        << ReadLe32(raw, 8U) << " dw3=0x" << ReadLe32(raw, 12U) << " dw4=0x" << ReadLe32(raw, 16U) << " dw5=0x"
        << ReadLe32(raw, 20U) << " dw6=0x" << ReadLe32(raw, 24U) << " dw7=0x" << ReadLe32(raw, 28U) << std::dec;
#endif
    out << "}";
    return out.str();
}

std::string BackendName()
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    return ACLSHMEMI_XSCALE_API_VERSION_VAR == 1 ? "XSCALE/V1" : "XSCALE/V2";
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    return "HNS_1825/V1";
#else
    return "IN_DIE/V1";
#endif
}

uint32_t CqeWqeIndex(const aclshmemi_rdma_exception_report_cqe_t& cqe, uint32_t sq_head, uint32_t depth)
{
    if (depth == 0U) {
        return 0U;
    }
    if ((cqe.valid_fields & ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID) != 0U) {
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        return (cqe.wqe_id >> 3U) % depth;
#else
        return cqe.wqe_id % depth;
#endif
    }
    return sq_head == 0U ? 0U : (sq_head - 1U) % depth;
}

std::string FormatWqeHeader(const std::vector<uint8_t>& raw)
{
    std::ostringstream out;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    if (raw.size() < 16U) {
        return "headerValid=0";
    }
    const uint32_t flags = ReadLe32(raw, 12U);
    out << "headerValid=1 opcode=" << static_cast<uint32_t>(raw[0]) << "(" << OpcodeName(raw[0]) << ")"
        << " withImm=" << (raw[1] & 1U) << " dsDataNum=" << ((raw[1] >> 3U) & 0x1fU) << " msgLen=" << ReadLe32(raw, 4U)
        << " wqeId=";
    if constexpr (ACLSHMEMI_XSCALE_API_VERSION_VAR == 1) {
        out << (static_cast<uint32_t>(raw[2]) | (static_cast<uint32_t>(raw[3]) << 8U));
    } else {
        out << ((flags >> 12U) & 0xfffffU);
    }
    out << " ce=" << ((flags >> 1U) & 1U) << " inline=" << ((flags >> 2U) & 1U);
    if (raw.size() >= 48U) {
        out << " remoteLen=" << (ReadLe32(raw, 16U) & 0x7fffffffU) << " remoteKey=0x" << std::hex << ReadLe32(raw, 20U)
            << " remoteAddr=0x" << ReadLe64(raw, 24U) << " localLen=" << (ReadLe32(raw, 32U) & 0x7fffffffU)
            << " localKey=0x" << ReadLe32(raw, 36U) << " localAddr=0x" << ReadLe64(raw, 40U) << std::dec;
    }
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    if (raw.size() < 16U) {
        return "headerValid=0";
    }
    const uint32_t task = ReadBe32(raw, 16U);
    out << "headerValid=1 owner=" << ((raw[0] >> 7U) & 1U) << " opcode=" << ((task >> 24U) & 0x1fU)
        << " signal=" << ((task >> 29U) & 1U) << " dataLen=" << ReadBe32(raw, 20U);
    if (raw.size() >= 64U) {
        out << " remoteAddr=0x" << std::hex << ReadBe64(raw, 32U) << " remoteKey=0x" << ReadBe32(raw, 40U)
            << " localAddr=0x" << ReadBe64(raw, 48U) << " localLen=" << ReadBe32(raw, 56U) << " localKey=0x"
            << (ReadBe32(raw, 60U) & 0x3fffffffU) << std::dec;
    }
#else
    if (raw.size() < 16U) {
        return "headerValid=0";
    }
    const uint32_t ctrl = ReadLe32(raw, 0U);
    out << "headerValid=1 opcode=" << (ctrl & 0x1fU) << "(" << OpcodeName(ctrl & 0x1fU)
        << ") owner=" << ((ctrl >> 7U) & 1U) << " signal=" << ((ctrl >> 8U) & 1U) << " msgLen=" << ReadLe32(raw, 4U);
    if (raw.size() >= 48U) {
        out << " rkey=0x" << std::hex << ReadLe32(raw, 20U) << " remoteAddr=0x" << ReadLe64(raw, 24U)
            << " sgeLen=" << ReadLe32(raw, 32U) << " lkey=0x" << ReadLe32(raw, 36U) << " localAddr=0x"
            << ReadLe64(raw, 40U) << std::dec;
    }
#endif
    return out.str();
}

struct WqeRemoteInfo {
    uint64_t address{0};
    uint64_t length{0};
    uint32_t key{0};
    bool valid{false};
};

WqeRemoteInfo DecodeWqeRemoteInfo(const std::vector<uint8_t>& raw)
{
    WqeRemoteInfo info;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    if (raw.size() >= 64U) {
        info.address = ReadBe64(raw, 32U);
        info.length = ReadBe32(raw, 20U);
        info.key = ReadBe32(raw, 40U);
        info.valid = true;
    }
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    if (raw.size() >= 48U) {
        info.address = ReadLe64(raw, 24U);
        info.length = ReadLe32(raw, 16U) & 0x7fffffffU;
        info.key = ReadLe32(raw, 20U);
        info.valid = true;
    }
#else
    if (raw.size() >= 48U) {
        info.address = ReadLe64(raw, 24U);
        info.length = ReadLe32(raw, 4U);
        info.key = ReadLe32(raw, 20U);
        info.valid = true;
    }
#endif
    return info;
}

std::string FormatWqeRaw(size_t offset, const std::vector<uint8_t>& full_raw)
{
    std::ostringstream out;
    out << "decoded={segmentOffset=" << offset;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    if (offset == 0U) {
        out << " " << FormatWqeHeader(full_raw);
    } else if (offset == 32U && full_raw.size() >= 48U) {
        out << " localLen=" << (ReadLe32(full_raw, 32U) & 0x7fffffffU) << " localKey=0x" << std::hex
            << ReadLe32(full_raw, 36U) << " localAddr=0x" << ReadLe64(full_raw, 40U) << std::dec;
    } else {
        out << " reserved=1";
    }
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    if (offset == 0U) {
        out << " " << FormatWqeHeader(full_raw);
    } else {
        out << " reserved=1";
    }
#else
    if (offset == 0U) {
        out << " " << FormatWqeHeader(full_raw);
    } else {
        out << " reserved=1";
    }
#endif
    out << "}";
    return out.str();
}

bool ValidSq(const AiQpRMAWQ& sq)
{
    return sq.bufAddr != 0U && sq.depth != 0U && sq.depth <= aclshmemi::exception::rdma::kMaxCqDepth &&
           sq.wqeSize != 0U && sq.wqeSize <= ACLSHMEMI_RDMA_EXCEPTION_REPORT_MAX_RAW_SIZE;
}

void DumpSqWqe(
    uint64_t instance, uint32_t rank, uint32_t peer, uint32_t qp, const AiQpRMAWQ& sq, uint32_t index,
    const shm::RdmaMemRegionInfo* mr, aclshmemi_rdma_exception_report_entry_t* device_entry)
{
    if (!ValidSq(sq)) {
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA SQ WQE DETAIL] engine=RDMA instance=" << instance << " rank=" << rank << " peer=" << peer
                                                                    << " qp=" << qp << " result=unavailable"
                                                                    << " reason=invalid_queue_metadata");
        return;
    }
    uint64_t address = 0;
    if (!aclshmemi::exception::rdma::CheckedEntryAddress(sq.bufAddr, index % sq.depth, sq.wqeSize, address)) {
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA SQ WQE DETAIL] engine=RDMA instance="
            << instance << " rank=" << rank << " peer=" << peer << " qp=" << qp << " index=" << index
            << " result=unavailable reason=wqe_address_overflow");
        return;
    }
    std::vector<uint8_t> raw(sq.wqeSize);
    if (!ReadSnapshot(raw.data(), raw.size(), address, device_entry, "sq_wqe")) {
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA SQ WQE DETAIL] engine=RDMA instance="
            << instance << " rank=" << rank << " peer=" << peer << " qp=" << qp << " index=" << index << " addr=0x"
            << std::hex << address << std::dec << " result=unavailable reason=read_failed");
        return;
    }
    std::string header = FormatWqeHeader(raw);
    const WqeRemoteInfo remote = DecodeWqeRemoteInfo(raw);
    bool addr_out_of_mr = false;
    bool key_mismatch = false;
    bool mr_check_valid = mr != nullptr && remote.valid && remote.address != 0U;
    if (mr_check_valid) {
        const auto result = aclshmemi::exception::rdma::CheckMrRange(
            mr->addr, mr->size, mr->rkey, remote.address, remote.length, remote.key);
        addr_out_of_mr = result == aclshmemi::exception::rdma::MrCheckResult::ADDR_OUT_OF_MR;
        key_mismatch = result == aclshmemi::exception::rdma::MrCheckResult::RKEY_MISMATCH;
    }
    SHM_LOG_ERROR(
        "[EXCEPTION][RDMA MR CHECK] engine=RDMA instance="
        << instance << " rank=" << rank << " peer=" << peer << " qp=" << qp << " index=" << index
        << " mrRead=" << (mr != nullptr) << " mrAddr=0x" << std::hex << (mr == nullptr ? 0U : mr->addr) << " mrSize=0x"
        << (mr == nullptr ? 0U : mr->size) << " mrRkey=0x" << (mr == nullptr ? 0U : mr->rkey) << " remoteAddr=0x"
        << remote.address << " remoteLen=0x" << remote.length << " remoteRkey=0x" << remote.key << std::dec
        << " result="
        << (mr == nullptr  ? "mr_unavailable" :
            !remote.valid  ? "wqe_remote_fields_unavailable" :
            addr_out_of_mr ? "addr_out_of_mr" :
            key_mismatch   ? "rkey_mismatch" :
                             "in_mr"));
    if (addr_out_of_mr) {
        header += " addr_out_of_mr";
    }
    if (key_mismatch) {
        header += " rkey_mismatch";
    }
    SHM_LOG_ERROR(
        "[EXCEPTION][RDMA SQ WQE HEADER] engine=RDMA instance="
        << instance << " backend=" << BackendName() << " rank=" << rank << " peer=" << peer << " qp=" << qp << " index="
        << index << " addr=0x" << std::hex << address << std::dec << " size=" << raw.size() << " " << header);
    for (size_t offset = 0; offset < raw.size(); offset += 32U) {
        const size_t line_size = std::min<size_t>(32U, raw.size() - offset);
        std::vector<uint8_t> line(raw.begin() + offset, raw.begin() + offset + line_size);
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA SQ WQE RAW] engine=RDMA instance="
            << instance << " rank=" << rank << " peer=" << peer << " qp=" << qp << " index=" << index
            << " rawOffset=" << offset << " data=" << HexSummary(line) << " " << FormatWqeRaw(offset, raw));
    }
}

bool ReadCqeSnapshot(
    uint64_t src, size_t size, std::vector<uint8_t>& raw, aclshmemi_rdma_exception_report_cqe_t& cqe,
    aclshmemi_rdma_exception_report_entry_t* device_entry)
{
    aclshmemi_rdma_exception_report_entry_t entry{};
    if (!ReadEntry(ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQE_RAW, src, size, entry, device_entry, "cqe") ||
        entry.raw.addr != src || entry.raw.size != size) {
        return false;
    }
    raw.assign(entry.raw.data, entry.raw.data + entry.raw.size);
    aclshmemi_rdma_exception_report_cqe_t host_cqe{};
    if (!aclshmemi::exception::rdma::DecodeRawCqe(raw.data(), raw.size(), host_cqe) ||
        !std::equal(
            reinterpret_cast<const uint8_t*>(&host_cqe), reinterpret_cast<const uint8_t*>(&host_cqe) + sizeof(host_cqe),
            reinterpret_cast<const uint8_t*>(&entry.cqe))) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=partial reason=cqe_snapshot_abi_mismatch");
        return false;
    }
    cqe = host_cqe;
    return true;
}

bool DumpCq(
    uint64_t instance, uint32_t rank, uint32_t peer, uint32_t qp, uint32_t wqn, uint32_t sq_head, bool sq_head_read,
    uint32_t cq_head, bool cq_head_read, const AiQpRMAWQ& sq, const AiQpRMACQ& cq, const shm::RdmaMemRegionInfo* mr,
    aclshmemi_rdma_exception_report_entry_t* device_entry)
{
    if (!ValidCq(cq)) {
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA] engine=RDMA instance=" << instance << " backend=" << BackendName() << " rank=" << rank
                                                      << " peer=" << peer << " qp=" << qp << " queue=SCQ"
                                                      << " result=unavailable reason=invalid_queue_metadata");
        return false;
    }
    uint32_t tail = 0;
    if (!ReadSnapshot(&tail, sizeof(tail), cq.tailAddr, device_entry, "cq_tail")) {
        return false;
    }
    SHM_LOG_ERROR(
        "[EXCEPTION][RDMA][QP STATE] engine=RDMA instance="
        << instance << " backend=" << BackendName() << " rank=" << rank << " peer=" << peer << " qp=" << qp
        << " wqn=" << wqn << " cqn=" << cq.cqn << " sqHead=" << sq_head << " sqHeadRead=" << sq_head_read
        << " cqHead=" << cq_head << " cqHeadRead=" << cq_head_read << " cqTail=" << tail
        << " result=" << (sq_head_read && cq_head_read ? "complete" : "partial"));
    const uint32_t stride = CqeStride(cq);
    const uint32_t ring_depth = CqRingDepth(cq);
    bool complete = true;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    const bool overrun = (cq.cqAttrFlags & (1U << 1U)) != 0U;
#else
    const bool overrun = false;
#endif
    // Overrun hardware retains only the latest CQE in slot zero.
    const uint32_t scan_depth = overrun ? 1U : ring_depth;
    // A poll can consume several CQEs before trapping without publishing its tail.
    // Scan the consecutive ready batch, bounded by one ring, using modular indices.
    aclshmemi::exception::rdma::VisitPendingCqes(tail, scan_depth, [&](uint32_t logical_index) {
        const uint32_t index = overrun ? 0U : aclshmemi::exception::rdma::CqeIndex(logical_index, ring_depth);
        uint64_t cqe_address = 0;
        if (!aclshmemi::exception::rdma::CheckedEntryAddress(cq.bufAddr, index, stride, cqe_address)) {
            SHM_LOG_ERROR(
                "[EXCEPTION][RDMA] engine=RDMA instance=" << instance
                                                          << " result=unavailable reason=cqe_address_overflow");
            complete = false;
            return false;
        }
        std::vector<uint8_t> raw;
        aclshmemi_rdma_exception_report_cqe_t cqe{};
        if (!ReadCqeSnapshot(
                cqe_address, std::min(stride, ACLSHMEMI_RDMA_EXCEPTION_REPORT_MAX_RAW_SIZE), raw, cqe, device_entry)) {
            complete = false;
            return false;
        }
        const bool ready = overrun ? aclshmemi::exception::rdma::IsOverrunCqeReady(cqe, ring_depth) :
                                     aclshmemi::exception::rdma::IsCqeReady(cqe, logical_index, ring_depth);
        if (!ready) {
            SHM_LOG_ERROR(
                "[EXCEPTION][RDMA CQE] engine=RDMA instance="
                << instance << " backend=" << BackendName() << " rank=" << rank << " peer=" << peer << " qp=" << qp
                << " queue=SCQ cq=" << cq.cqn << " tail=" << tail << " logicalIndex=" << logical_index
                << " index=" << index << " result=unavailable reason=cqe_not_ready");
            // A trap may occur before CQE publication. Inspect the current SQ WQE
            // so address and rkey validation is still available in that case.
            if (logical_index == tail) {
                DumpSqWqe(instance, rank, peer, qp, sq, CqeWqeIndex(cqe, sq_head, sq.depth), mr, device_entry);
            }
            complete = false;
            return false;
        }
        const std::string decoded = aclshmemi::exception::rdma::FormatCqe(cqe);
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA CQE] engine=RDMA instance="
            << instance << " backend=" << BackendName() << " rank=" << rank << " peer=" << peer << " qp=" << qp
            << " queue=SCQ cq=" << cq.cqn << " tail=" << tail << " logicalIndex=" << logical_index << " index=" << index
            << " addr=0x" << std::hex << cqe_address << std::dec << " " << decoded << " rawBytes=" << raw.size()
            << " result=complete");
        SHM_LOG_ERROR(
            "[EXCEPTION][RDMA CQE DETAIL] engine=RDMA instance="
            << instance << " rank=" << rank << " peer=" << peer << " qp=" << qp << " tail=" << tail << " index="
            << index << " cqeAddr=0x" << std::hex << cqe_address << std::dec << " " << FormatCqeRaw(raw, cqe));
        for (size_t offset = 0; offset < raw.size(); offset += 32U) {
            const size_t line_size = std::min<size_t>(32U, raw.size() - offset);
            std::vector<uint8_t> line(raw.begin() + offset, raw.begin() + offset + line_size);
            SHM_LOG_ERROR(
                "[EXCEPTION][RDMA CQE RAW] engine=RDMA instance="
                << instance << " rank=" << rank << " peer=" << peer << " qp=" << qp << " tail=" << tail
                << " index=" << index << " rawOffset=" << offset << " data=" << HexSummary(line));
        }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        const uint32_t wqe_index = CqeWqeIndex(cqe, sq_head, sq.depth);
#else
        // CQ and SQ indices diverge when deferred WQEs do not request CQEs.
        const uint32_t wqe_index = sq.depth == 0U ? 0U : (ReadLe32(raw, 0U) >> 16U) % sq.depth;
#endif
        DumpSqWqe(instance, rank, peer, qp, sq, wqe_index, mr, device_entry);
        return !aclshmemi::exception::rdma::IsErrorCqe(cqe);
    });
    return complete;
}
} // namespace

int aclshmemi_exception_report_dump_rdma(bool detail_enabled)
{
    if (!detail_enabled || init_manager == nullptr) {
        return ACLSHMEM_SUCCESS;
    }
    aclshmemi_rdma_exception_source_t source{};
    source.instance_id = g_instance_ctx == nullptr ? 0U : g_instance_ctx->id;
    source.rank_id = g_state.mype;
    source.rank_count = g_state.npes;
    source.qp_info_address = reinterpret_cast<uint64_t>(g_state.qp_info);
    const uint64_t instance_id = source.instance_id;
    DeviceEntryBuffer device_entry;
    int ret = shm::DlAclApi::AclrtMalloc(
        &device_entry.data, sizeof(aclshmemi_rdma_exception_report_entry_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=alloc_snapshot_failed ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    AiQpRMAQueueInfo info{};
    if (!ReadSnapshot(
            &info, sizeof(info), source.qp_info_address,
            static_cast<aclshmemi_rdma_exception_report_entry_t*>(device_entry.data), "qp_info")) {
        return ACLSHMEM_INNER_ERROR;
    }
    if (!ValidQueueInfo(info, source)) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=invalid_qp_metadata");
        return ACLSHMEM_INNER_ERROR;
    }
    auto* entry = static_cast<aclshmemi_rdma_exception_report_entry_t*>(device_entry.data);
    if (source.rank_count > std::numeric_limits<size_t>::max() / info.count) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=qp_count_overflow");
        return ACLSHMEM_INNER_ERROR;
    }
    const size_t entry_count = static_cast<size_t>(source.rank_count) * info.count;
    if (entry_count > std::numeric_limits<uint64_t>::max() / sizeof(AiQpRMAWQ) ||
        entry_count > std::numeric_limits<uint64_t>::max() / sizeof(AiQpRMACQ)) {
        SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=qp_context_size_overflow");
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_ERROR(
        "[EXCEPTION][RDMA INFO] engine=RDMA instance="
        << instance_id << " backend=" << BackendName() << " rank=" << source.rank_id
        << " rankCount=" << source.rank_count << " qpCount=" << info.count << " slotCount=" << entry_count
        << " qpInfo=0x" << std::hex << source.qp_info_address << " sqPtr=" << reinterpret_cast<uint64_t>(info.sq)
        << " rqPtr=" << reinterpret_cast<uint64_t>(info.rq) << " scqPtr=" << reinterpret_cast<uint64_t>(info.scq)
        << " rcqPtr=" << reinterpret_cast<uint64_t>(info.rcq) << " mrPtr=" << reinterpret_cast<uint64_t>(info.mr)
        << std::dec);
    bool complete = true;
    for (uint32_t rank = 0; rank < source.rank_count; ++rank) {
        if (rank == source.rank_id) {
            continue;
        }
        for (uint32_t qp = 0; qp < info.count; ++qp) {
            const size_t index = static_cast<size_t>(rank) * info.count + qp;
            const uint64_t scq_base = reinterpret_cast<uint64_t>(info.scq);
            const uint64_t sq_base = reinterpret_cast<uint64_t>(info.sq);
            uint64_t sq_addr = 0;
            uint64_t scq_addr = 0;
            if (!aclshmemi::exception::rdma::CheckedEntryAddress(sq_base, index, sizeof(AiQpRMAWQ), sq_addr) ||
                !aclshmemi::exception::rdma::CheckedEntryAddress(scq_base, index, sizeof(AiQpRMACQ), scq_addr)) {
                SHM_LOG_ERROR("[EXCEPTION][RDMA] result=unavailable reason=qp_context_address_overflow");
                complete = false;
                continue;
            }
            AiQpRMAWQ sq{};
            AiQpRMACQ scq{};
            shm::RdmaMemRegionInfo mr{};
            const bool mr_read =
                info.mr != nullptr &&
                ReadSnapshot(
                    &mr, sizeof(mr), reinterpret_cast<uint64_t>(info.mr) + static_cast<uint64_t>(rank) * sizeof(mr),
                    entry, "mr");
            if (ReadWqSnapshot(sq_addr, sq, entry) && ReadCqSnapshot(scq_addr, scq, entry)) {
                uint32_t sq_head = 0;
                const bool sq_valid =
                    sq.headAddr != 0U && sq.depth != 0U && sq.depth <= aclshmemi::exception::rdma::kMaxCqDepth;
                if (!sq_valid) {
                    SHM_LOG_ERROR(
                        "[EXCEPTION][RDMA] engine=RDMA instance=" << instance_id << " rank=" << source.rank_id
                                                                  << " peer=" << rank << " qp=" << qp
                                                                  << " queue=SQ result=unavailable"
                                                                     " reason=invalid_queue_metadata");
                }
                const bool sq_head_read =
                    sq_valid && ReadSnapshot(&sq_head, sizeof(sq_head), sq.headAddr, entry, "sq_head");
                uint32_t sq_tail = 0;
                const bool sq_tail_read =
                    sq_valid && ReadSnapshot(&sq_tail, sizeof(sq_tail), sq.tailAddr, entry, "sq_tail");
                uint32_t cq_head = 0;
                const bool cq_head_addr_available = scq.headAddr != 0U;
                const bool cq_head_read =
                    cq_head_addr_available && ReadSnapshot(&cq_head, sizeof(cq_head), scq.headAddr, entry, "cq_head");
                const char* cq_head_reason = !cq_head_addr_available ? "address_unavailable" :
                                             cq_head_read            ? "read_success" :
                                                                       "read_failed";
                if (!sq_head_read) {
                    complete = false;
                }
                if (!sq_tail_read) {
                    complete = false;
                }
                if (!cq_head_read && cq_head_addr_available) {
                    complete = false;
                }
                SHM_LOG_ERROR(
                    "[EXCEPTION][RDMA QP] engine=RDMA instance="
                    << instance_id << " backend=" << BackendName() << " rank=" << source.rank_id << " peer=" << rank
                    << " qp=" << qp << " wqn=" << sq.wqn << " cqn=" << scq.cqn << " sqHead=" << sq_head
                    << " sqTail=" << sq_tail << " cqHead=" << cq_head << " cqTailAddr=0x" << std::hex << scq.tailAddr
                    << " cqHeadAddr=" << scq.headAddr << " sqHeadRead=" << std::dec << sq_head_read << " sqTailRead="
                    << sq_tail_read << " cqHeadRead=" << cq_head_read << " cqHeadReason=" << cq_head_reason);
                SHM_LOG_ERROR(
                    "[EXCEPTION][RDMA QP DETAIL] engine=RDMA instance="
                    << instance_id << " rank=" << source.rank_id << " peer=" << rank << " qp=" << qp
                    << " sq={bufAddr=0x" << std::hex << sq.bufAddr << ",wqeSize=" << std::dec << sq.wqeSize
                    << ",depth=" << sq.depth << ",headAddr=0x" << std::hex << sq.headAddr << ",tailAddr=0x"
                    << sq.tailAddr << ",dbMode=" << std::dec << static_cast<int>(sq.dbMode) << ",dbAddr=0x" << std::hex
                    << sq.dbAddr << ",dbSwVa=0x" << sq.dbSwVa << ",sl=" << std::dec << sq.sl << ",mtuShift="
                    << static_cast<uint32_t>(sq.mtuShift) << ",dbCos=" << static_cast<uint32_t>(sq.dbCos)
                    << "} scq={bufAddr=0x" << std::hex << scq.bufAddr << ",cqeSize=" << std::dec << CqeStride(scq)
                    << ",depth=" << scq.depth << ",headAddr=0x" << std::hex << scq.headAddr << ",tailAddr=0x"
                    << scq.tailAddr << ",dbMode=" << std::dec << static_cast<int>(scq.dbMode) << ",dbAddr=0x"
                    << std::hex << scq.dbAddr << ",dbSwVa=0x" << scq.dbSwVa << std::dec << "}");
                complete = DumpCq(
                               instance_id, source.rank_id, rank, qp, sq.wqn, sq_head, sq_head_read, cq_head,
                               cq_head_read, sq, scq, mr_read ? &mr : nullptr, entry) &&
                           complete;
            } else {
                complete = false;
            }
        }
    }
    // A bounded diagnostic read is allowed to be partial; the snapshot is still consumed once.
    (void)complete;
    return ACLSHMEM_SUCCESS;
}
