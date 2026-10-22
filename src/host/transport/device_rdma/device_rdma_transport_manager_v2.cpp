/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_entity_def.h"
#include "shmemi_logger.h"
#include "shmemi_scope_guard.h"
#include "dl_acl_api.h"
#include "dl_hcomm_api.h"
#include "dl_hcomm_def.h"
#include "device_rdma_common.h"
#include "device_rdma_helper.h"
#include "init/shmemi_init.h"
#include "transport/topo/topo_reader.h"
#include "device_rdma_transport_manager_v2.h"

#include <chrono>
#include <cstring>
#include <algorithm>
#include <limits>
#include <string>
#include <thread>

namespace shm {
namespace transport {
namespace device {
namespace {

constexpr uint32_t ATOMIC_MAX_NUM = 128;
constexpr uint32_t RDMA_CHANNEL_MEM_HANDLE_NUM = 2;
// 非阻塞建链状态轮询参数(按通道个数缩放)
constexpr uint32_t CHANNEL_STATUS_POLL_INTERVAL_PER_CH_MS = 10;   // 单通道单次轮询间隔(ms)
constexpr uint32_t CHANNEL_STATUS_POLL_TIMEOUT_PER_CH_MS = 60000; // 单通道建链就绪等待超时(ms) = 1min

std::string BuildStableChannelName(uint32_t localRank, uint32_t remoteRank, uint32_t qpIdx)
{
    const uint32_t serverRank = std::min(localRank, remoteRank);
    const uint32_t clientRank = std::max(localRank, remoteRank);
    return "aclshmem-r" + std::to_string(serverRank) + "-r" + std::to_string(clientRank) + "-q" + std::to_string(qpIdx);
}

constexpr uint32_t HNS_1825_DB_VENDOR_FIELD_MASK = 0x7;
constexpr uint32_t HNS_1825_DB_COS_SHIFT = 24;
constexpr uint32_t HNS_1825_MTU_SHIFT_SHIFT = 50;
constexpr uint8_t HNS_1825_DEFAULT_MTU_SHIFT = 4;
constexpr uint8_t HNS_1825_DEFAULT_DB_COS = 0x7;
constexpr uint32_t RDMA_CONTEXT_ENTRY_SIZE_MIN = 32;
constexpr uint32_t RDMA_CONTEXT_ENTRY_SIZE_MAX = 4096;
constexpr uint64_t HNS_1825_DB_VENDOR_SPECIFIED_MASK =
    (static_cast<uint64_t>(HNS_1825_DB_VENDOR_FIELD_MASK) << HNS_1825_DB_COS_SHIFT) |
    (static_cast<uint64_t>(HNS_1825_DB_VENDOR_FIELD_MASK) << HNS_1825_MTU_SHIFT_SHIFT);

struct DecodedRoceSqContext {
    const char* abiName;
    uint32_t qpn;
    uint64_t sqVa;
    uint32_t wqeSize;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbHwVa;
    uint64_t dbSwVa;
    DBMode dbMode;
    uint8_t sl;
    uint8_t mtuShift;
    uint8_t dbCos;
};

struct DecodedRoceCqContext {
    const char* abiName;
    uint32_t cqn;
    uint64_t cqVa;
    uint32_t cqeSize;
    uint32_t cqDepth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbHwVa;
    uint64_t dbSwVa;
    DBMode dbMode;
};

bool IsDbModeValid(int8_t dbMode) noexcept
{
    return dbMode == static_cast<int8_t>(DBMode::HW_DB) || dbMode == static_cast<int8_t>(DBMode::SW_DB);
}

bool IsRdmaContextEntrySizeValid(uint32_t size) noexcept
{
    return size >= RDMA_CONTEXT_ENTRY_SIZE_MIN && size <= RDMA_CONTEXT_ENTRY_SIZE_MAX && (size & (size - 1)) == 0;
}

bool IsDbVendorSpecifiedValid(uint64_t dbVendorSpecified) noexcept
{
    if (dbVendorSpecified == 0) {
        return false;
    }

    // The transition ABI used the low byte for dbCos; only the 3-bit value is valid.
    if ((dbVendorSpecified & ~0xffULL) == 0) {
        return dbVendorSpecified <= HNS_1825_DB_VENDOR_FIELD_MASK;
    }

    // SplitDb producers may not initialize the bytes after offset 56. Accept
    // only the current ABI's known bit-field shape to avoid treating tail data
    // as DbVendorSpecified.
    return (dbVendorSpecified & ~HNS_1825_DB_VENDOR_SPECIFIED_MASK) == 0;
}

bool LooksLikeInitialSqContext(const SqContext& context) noexcept
{
    if (context.type != SQ_CONTEXT_TYPE_ROCE) {
        return false;
    }

    /*
     * There is no ABI version tag in SqContext. In the SplitDb layout, the
     * Initial view aliases wqeSize with the low byte of the real wqeSize and
     * aliases qpn with the low part of dbSwVa. Validate the Initial entry
     * size as an additional discriminator. The supported RDMA backends use
     * power-of-two entries in this range; this check documents the assumption
     * and avoids accepting the known 256-byte SplitDb false-positive case.
     */
    const auto initial = ExtractSqContextRoceInitial(context);
    return initial.qpn != 0 && IsRdmaContextEntrySizeValid(initial.wqeSize) && initial.depth != 0 &&
           IsDbModeValid(initial.dbMode) && initial.sl <= 7;
}

bool LooksLikeInitialCqContext(const CqContext& context) noexcept
{
    if (context.type != CQ_CONTEXT_TYPE_ROCE) {
        return false;
    }

    // See LooksLikeInitialSqContext: cqeSize provides the equivalent extra ABI discriminator for CQ.
    const auto initial = ExtractCqContextRoceInitial(context);
    return initial.cqn != 0 && IsRdmaContextEntrySizeValid(initial.cqeSize) && initial.cqDepth != 0 &&
           IsDbModeValid(initial.dbMode);
}

void DecodePackedDbVendorSpecified(uint64_t dbVendorSpecified, uint8_t& mtuShift, uint8_t& dbCos) noexcept
{
    if (dbVendorSpecified == 0) {
        mtuShift = HNS_1825_DEFAULT_MTU_SHIFT;
        dbCos = HNS_1825_DEFAULT_DB_COS;
        return;
    }

    // A short-lived transition ABI carried vendorPrivInfo as a low-byte value.
    if ((dbVendorSpecified & ~0xffULL) == 0) {
        mtuShift = HNS_1825_DEFAULT_MTU_SHIFT;
        dbCos = static_cast<uint8_t>(dbVendorSpecified & HNS_1825_DB_VENDOR_FIELD_MASK);
        return;
    }

    mtuShift = static_cast<uint8_t>((dbVendorSpecified >> HNS_1825_MTU_SHIFT_SHIFT) & HNS_1825_DB_VENDOR_FIELD_MASK);
    dbCos = static_cast<uint8_t>((dbVendorSpecified >> HNS_1825_DB_COS_SHIFT) & HNS_1825_DB_VENDOR_FIELD_MASK);
}

DecodedRoceSqContext DecodeRoceSqContext(const SqContext& context) noexcept
{
    if (LooksLikeInitialSqContext(context)) {
        const auto initial = ExtractSqContextRoceInitial(context);
        return {
            "initial",
            initial.qpn,
            initial.sqVa,
            initial.wqeSize,
            initial.depth,
            initial.headAddr,
            initial.tailAddr,
            initial.dbVa,
            0,
            (initial.dbMode == 0) ? DBMode::HW_DB : DBMode::SW_DB,
            initial.sl,
            HNS_1825_DEFAULT_MTU_SHIFT,
            HNS_1825_DEFAULT_DB_COS};
    }

    const auto splitDb = ExtractSqContextRoceSplitDb(context);
    uint8_t mtuShift = splitDb.mtuShift;
    uint8_t dbCos = HNS_1825_DEFAULT_DB_COS;
    const auto vendor = ExtractSqContextRoceVendorSpecified(context);
    const char* abiName = "split-db-mtushift";
    if (IsDbVendorSpecifiedValid(vendor.DbVendorSpecified)) {
        abiName = "db-vendor-specified";
        DecodePackedDbVendorSpecified(vendor.DbVendorSpecified, mtuShift, dbCos);
    }

    return {abiName,          splitDb.qpn,      splitDb.sqVa,   splitDb.wqeSize, splitDb.depth,
            splitDb.headAddr, splitDb.tailAddr, splitDb.dbHwVa, splitDb.dbSwVa,  DBMode::SW_DB,
            splitDb.sl,       mtuShift,         dbCos};
}

DecodedRoceCqContext DecodeRoceCqContext(const CqContext& context) noexcept
{
    if (LooksLikeInitialCqContext(context)) {
        const auto initial = ExtractCqContextRoceInitial(context);
        return {
            "initial",
            initial.cqn,
            initial.cqVa,
            initial.cqeSize,
            initial.cqDepth,
            initial.headAddr,
            initial.tailAddr,
            initial.dbVa,
            0,
            (initial.dbMode == 0) ? DBMode::HW_DB : DBMode::SW_DB};
    }

    const auto splitDb = ExtractCqContextRoceSplitDb(context);
    const auto vendor = ExtractCqContextRoceVendorSpecified(context);
    return {
        IsDbVendorSpecifiedValid(vendor.DbVendorSpecified) ? "db-vendor-specified" : "split-db",
        splitDb.cqn,
        splitDb.cqVa,
        splitDb.cqeSize,
        splitDb.cqDepth,
        splitDb.headAddr,
        splitDb.tailAddr,
        splitDb.dbHwVa,
        splitDb.dbSwVa,
        DBMode::SW_DB};
}
} // namespace

RdmaTransportManagerV2::~RdmaTransportManagerV2()
{
    ClearAllRegisterMRs();
    DestroyEndpoint();
}

Result RdmaTransportManagerV2::OpenDevice(const TransportOptions& options)
{
    int32_t userId = -1;
    int32_t logicId = -1;

    SHM_LOG_DEBUG("rank[" << rankId_ << "] begin to open device with " << options);
    auto ret = DlAclApi::AclrtGetDevice(&userId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && userId >= 0, "AclrtGetDevice() return=" << ret << ", output deviceId=" << userId,
        ACLSHMEM_INNER_ERROR);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(userId, &logicId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && logicId >= 0, "RtGetLogicDevIdByUserDevId() return=" << ret << ", output deviceId=" << logicId,
        ACLSHMEM_INNER_ERROR);

    int32_t phyId = -1;
    // HCCP/topo use global phyId; pass userId to deprecated API (MR !407), not logicId.
    ret = DlAclApi::AclrtGetPhyDevIdByLogicDevId(userId, &phyId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && phyId >= 0,
        "AclrtGetPhyDevIdByLogicDevId() return=" << ret << ", userId=" << userId << ", logicDeviceId=" << logicId
                                                 << ", output phyId=" << phyId,
        ACLSHMEM_INNER_ERROR);
    phyId_ = static_cast<uint32_t>(phyId);

    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    role_ = options.role;
    qpNum_ = options.rdmaQpConfig.qpNum;
    if (qpNum_ == 0 || qpNum_ > ACLSHMEM_MAX_QP_NUM) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] invalid rdma qp num: " << qpNum_);
        return ACLSHMEM_INVALID_PARAM;
    }
    if (options.type == IpV4) {
        deviceIp_.type = IpV4;
    } else if (options.type == IpV6) {
        deviceIp_.type = IpV6;
    }

    if (!TopoReader::ParseRdmaNetAddr(phyId_, deviceIp_)) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] ParseRdmaNetAddr failed for phyId " << phyId_);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = CreateEndpoint();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] CreateEndpoint failed: " << ret);
        return ret;
    }

    ret = InitActualListenPort();
    if (ret != ACLSHMEM_SUCCESS) {
        DestroyEndpoint();
        return ret;
    }

    nicInfo_ = GenerateDeviceNic(deviceIp_, devicePort_);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] nicInfo_=" << nicInfo_);

    if (!ReserveRdmaInfoSpace()) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] reserve rdma info space failed.");
        DestroyEndpoint();
        return ACLSHMEM_INNER_ERROR;
    }

    SHM_LOG_INFO("rank[" << rankId_ << "] open device with " << options << " success.");
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::InitActualListenPort()
{
    if (endpointHandle_ == nullptr) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] endpoint not created, cannot get listen port.");
        return ACLSHMEM_INNER_ERROR;
    }

    uint32_t listenPort = 0U;
    HcommResult hret = DlHcommApi::HcommEndpointGetListenPort(endpointHandle_, &listenPort);
    if (hret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] HcommEndpointGetListenPort failed: " << hret);
        return ACLSHMEM_INNER_ERROR;
    }
    if (listenPort == 0U || listenPort > std::numeric_limits<uint16_t>::max()) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] invalid actual rdma listen port: " << listenPort);
        return ACLSHMEM_INNER_ERROR;
    }

    devicePort_ = static_cast<uint16_t>(listenPort);
    SHM_LOG_INFO("rank[" << rankId_ << "] actual rdma listen port: " << devicePort_);
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::BuildEndpointDesc(EndpointDesc& desc)
{
    EndpointDescInit(&desc, 1);

    desc.protocol = COMM_PROTOCOL_ROCE;
    if (deviceIp_.type == IpV4) {
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = deviceIp_.ip.ipv4;
    } else if (deviceIp_.type == IpV6) {
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V6;
        desc.commAddr.addr6 = deviceIp_.ip.ipv6;
    } else {
        SHM_LOG_ERROR("rank[" << rankId_ << "] unsupported ip type: " << deviceIp_.type);
        return ACLSHMEM_INVALID_PARAM;
    }

    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    desc.loc.device.devPhyId = phyId_;
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::CreateEndpoint()
{
    if (endpointHandle_ != nullptr) {
        SHM_LOG_INFO("rank[" << rankId_ << "] endpoint already created.");
        return ACLSHMEM_SUCCESS;
    }

    EndpointDesc endpointDesc{};
    auto ret = BuildEndpointDesc(endpointDesc);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] build endpoint desc failed: " << ret);
        return ret;
    }

    SHM_LOG_DEBUG(
        "rank[" << rankId_ << "] HcommEndpointCreate, protocol=COMM_PROTOCOL_ROCE, locType=ENDPOINT_LOC_TYPE_HOST"
                << ", devPhyId=" << phyId_);
    HcommResult hret = DlHcommApi::HcommEndpointCreate(&endpointDesc, &endpointHandle_);
    if (hret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] HcommEndpointCreate failed: " << hret);
        return ACLSHMEM_INNER_ERROR;
    }

    SHM_LOG_INFO("rank[" << rankId_ << "] HcommEndpointCreate success, endpointHandle=" << endpointHandle_);
    return ACLSHMEM_SUCCESS;
}

void RdmaTransportManagerV2::DestroyEndpoint()
{
    if (atomicMemHandle_ != nullptr && endpointHandle_ != nullptr) {
        HcommResult hret = DlHcommApi::HcommMemUnreg(endpointHandle_, atomicMemHandle_);
        if (hret != 0) {
            SHM_LOG_WARN("rank[" << rankId_ << "] HcommMemUnreg for atomic memory failed: " << hret);
        }
        atomicMemHandle_ = nullptr;
        atomicLkey_ = 0;
    }

    if (endpointHandle_ != nullptr) {
        HcommResult hret = DlHcommApi::HcommEndpointDestroy(endpointHandle_);
        if (hret != 0) {
            SHM_LOG_WARN("rank[" << rankId_ << "] HcommEndpointDestroy failed: " << hret);
        }
        endpointHandle_ = nullptr;
    }

    if (atomicSharedMemory_ != nullptr) {
        DlAclApi::AclrtFree(atomicSharedMemory_);
        atomicSharedMemory_ = nullptr;
    }

    if (qpInfo_ != nullptr) {
        DlAclApi::AclrtFree(reinterpret_cast<void*>(qpInfo_));
        qpInfo_ = nullptr;
    }
}

Result RdmaTransportManagerV2::CloseDevice()
{
    DlHcommApi::HcommChannelDestroy(reinterpret_cast<const ChannelHandle*>(channelPtrs_.data()), channelPtrs_.size());
    channelPtrs_.clear();

    ClearAllRegisterMRs();
    DestroyEndpoint();
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::RegisterMemoryRegion(const TransportMemoryRegion& mr)
{
    if (!channelPtrs_.empty()) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] cannot register MR after RDMA channels are connected.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (endpointHandle_ == nullptr) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] endpoint not created, cannot register MR.");
        return ACLSHMEM_INNER_ERROR;
    }

    CommMem commMem{};
    if ((mr.flags & REG_MR_FLAG_HBM) || IsVirtualAddressNpu(mr.addr)) {
        commMem.type = CommMemType::DEVICE;
    } else {
        commMem.type = CommMemType::HOST;
    }
    commMem.addr = reinterpret_cast<void*>(static_cast<ptrdiff_t>(mr.addr));
    commMem.size = mr.size;

    HcommMemHandle memHandle = nullptr;
    SHM_LOG_DEBUG(
        "rank[" << rankId_ << "] HcommMemReg, addr=" << mr.addr << ", size=" << mr.size
                << ", type=" << static_cast<int32_t>(commMem.type));
    HcommResult hret = DlHcommApi::HcommMemReg(endpointHandle_, "HcclBuffer", &commMem, &memHandle);
    if (hret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] HcommMemReg failed: " << hret);
        return ACLSHMEM_INNER_ERROR;
    }

    HcommMemRegEntry entry{};
    entry.memHandle = memHandle;
    entry.addr = mr.addr;
    entry.size = mr.size;

    registeredMRs_.emplace(mr.addr, entry);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] HcommMemReg success, addr=" << mr.addr << ", memHandle=" << memHandle);
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::UnregisterMemoryRegion(uint64_t addr)
{
    if (!channelPtrs_.empty()) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] cannot unregister MR after RDMA channels are connected.");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto pos = registeredMRs_.find(addr);
    if (pos == registeredMRs_.end()) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] input address not registered!");
        return ACLSHMEM_INVALID_PARAM;
    }

    HcommResult hret = DlHcommApi::HcommMemUnreg(endpointHandle_, pos->second.memHandle);
    if (hret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] HcommMemUnreg failed: " << hret);
        return ACLSHMEM_INNER_ERROR;
    }

    registeredMRs_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::GetPrimaryMemoryHandle(HcommMemHandle& memHandle) const
{
    if (registeredMRs_.empty()) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] no registered user MR before RDMA channel create.");
        return ACLSHMEM_INVALID_PARAM;
    }

    memHandle = registeredMRs_.begin()->second.memHandle;
    if (memHandle == nullptr) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] primary user MR handle is null.");
        return ACLSHMEM_INNER_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

void RdmaTransportManagerV2::ClearAllRegisterMRs()
{
    for (auto it = registeredMRs_.begin(); it != registeredMRs_.end(); ++it) {
        HcommResult hret = DlHcommApi::HcommMemUnreg(endpointHandle_, it->second.memHandle);
        if (hret != 0) {
            SHM_LOG_WARN("rank[" << rankId_ << "] HcommMemUnreg addr=" << it->first << " failed: " << hret);
        }
    }
    registeredMRs_.clear();
}

Result RdmaTransportManagerV2::Prepare(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG("rank[" << rankId_ << "] RdmaTransportManagerV2 Prepare with : " << options);

    int ret;
    if ((ret = CheckPrepareOptions(options)) != 0) {
        return ret;
    }

    mf_sockaddr deviceNetwork;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(
                "rank[" << rankId_ << "] parse networks[" << it->first << "]=" << it->second.nic << " failed: " << ret);
            return ACLSHMEM_INVALID_PARAM;
        }

        rankInfo_.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    SHM_LOG_DEBUG("rank[" << rankId_ << "] rankInfo_.size=" << rankInfo_.size());

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::Connect()
{
    SHM_LOG_DEBUG("rank[" << rankId_ << "] RdmaTransportManagerV2 Connect");

    if (endpointHandle_ == nullptr) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] endpoint not created, please call Prepare first");
        return ACLSHMEM_INNER_ERROR;
    }

    auto consistencyRet = CheckQpNumConsistency();
    if (consistencyRet != ACLSHMEM_SUCCESS) {
        return consistencyRet;
    }

    const uint64_t peerCount = static_cast<uint64_t>(rankCount_ - 1);
    const uint64_t channelNum64 = peerCount * static_cast<uint64_t>(qpNum_);
    if (channelNum64 > std::numeric_limits<uint32_t>::max()) {
        SHM_LOG_ERROR(
            "rank[" << rankId_ << "] channel count overflow, peerCount=" << peerCount << ", qpNum=" << qpNum_);
        return ACLSHMEM_INVALID_PARAM;
    }
    uint32_t channelNum = static_cast<uint32_t>(channelNum64);

    if (!RegisterAtomicMemory()) {
        return ACLSHMEM_INNER_ERROR;
    }

    HcommMemHandle primaryMemHandle = nullptr;
    auto ret = GetPrimaryMemoryHandle(primaryMemHandle);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    std::vector<HcommChannelDesc> channelDescs(channelNum);
    auto desc_init_ret = ShmemHcommChannelDescInit(channelDescs.data(), channelNum);
    if (desc_init_ret != 0) {
        SHM_LOG_ERROR("HcommChannelDescInit failed, ret = " << desc_init_ret);
        return ACLSHMEM_INNER_ERROR;
    }
    std::vector<HcommMemHandle> channelMemHandles(static_cast<size_t>(channelNum) * RDMA_CHANNEL_MEM_HANDLE_NUM);
    channelNames_.clear();
    channelNames_.resize(channelNum);

    uint8_t roceTc = GetEnvUint8("HCCL_RDMA_TC", DEFAULT_RDMA_TC, 0, 255, true);
    uint8_t roceSl = GetEnvUint8("HCCL_RDMA_SL", DEFAULT_RDMA_SL, 0, 7);

    uint32_t chIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < rankCount_; ++remoteRank) {
        if (remoteRank == rankId_) {
            continue;
        }

        auto rankIt = rankInfo_.find(remoteRank);
        if (rankIt == rankInfo_.end()) {
            SHM_LOG_ERROR("rank[" << rankId_ << "] rank " << remoteRank << " not found in rankInfo_");
            return ACLSHMEM_INVALID_PARAM;
        }

        uint16_t remotePort = 0U;
        auto portRet = GetDeviceNicPort(rankIt->second.network, remotePort);
        if (portRet != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("rank[" << rankId_ << "] rank " << remoteRank << " has invalid rdma port.");
            return portRet;
        }
        const bool isServer = (rankId_ < remoteRank);
        for (uint32_t qpIdx = 0; qpIdx < qpNum_; ++qpIdx) {
            auto& desc = channelDescs[chIdx];
            desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
            if (rankIt->second.network.type == IpV4) {
                desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
                desc.remoteEndpoint.commAddr.addr = rankIt->second.network.ip.ipv4.sin_addr;
            } else {
                desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V6;
                desc.remoteEndpoint.commAddr.addr6 = rankIt->second.network.ip.ipv6.sin6_addr;
            }
            desc.notifyNum = 3;
            desc.exchangeAllMems = false;
            HcommMemHandle* memHandles = channelMemHandles.data() + chIdx * RDMA_CHANNEL_MEM_HANDLE_NUM;
            memHandles[0] = primaryMemHandle;
            memHandles[1] = atomicMemHandle_;
            desc.memHandles = memHandles;
            desc.memHandleNum = RDMA_CHANNEL_MEM_HANDLE_NUM;
            desc.roceAttr.queueNum = 1;
            desc.roceAttr.tc = roceTc;
            desc.roceAttr.sl = roceSl;
            desc.roceAttr.retryCnt = DEFAULT_ROCE_RETRY_CNT;
            desc.roceAttr.retryInterval = DEFAULT_ROCE_RETRY_INTERVAL;
            SetChannelDescCqAttrFlags(desc, cqAttrFlags_);
            desc.socket = nullptr;
            desc.role = isServer ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
            desc.port = isServer ? devicePort_ : remotePort;
            channelNames_[chIdx] = BuildStableChannelName(rankId_, remoteRank, qpIdx);
            desc.channelName = channelNames_[chIdx].c_str();
            SHM_LOG_INFO(
                "rank[" << rankId_ << "] rdma channel to rank[" << remoteRank << "] qp=" << qpIdx
                        << " role=" << (isServer ? "server" : "client") << ", port=" << desc.port
                        << ", name=" << channelNames_[chIdx]);
            ++chIdx;
        }
    }

    channelPtrs_.resize(channelNum);

    auto hcommRet = DlHcommApi::HcommChannelCreate(
        endpointHandle_, COMM_ENGINE_AIV, channelDescs.data(), channelNum,
        reinterpret_cast<ChannelHandle*>(channelPtrs_.data()));
    if (hcommRet != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] HcommChannelCreate failed: " << hcommRet);
        channelPtrs_.clear();
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("rank[" << rankId_ << "] HcommChannelCreate success, channelNum=" << channelNum);

    auto channelGuard = shm::utils::make_scope_guard(channelPtrs_.data(), [this, channelNum](ChannelHandle*) {
        DlHcommApi::HcommChannelDestroy(reinterpret_cast<const ChannelHandle*>(channelPtrs_.data()), channelNum);
        channelPtrs_.clear();
    });

    const uint32_t pollIntervalMs = channelNum * CHANNEL_STATUS_POLL_INTERVAL_PER_CH_MS;
    const uint32_t pollTimeoutMs = channelNum * CHANNEL_STATUS_POLL_TIMEOUT_PER_CH_MS;
    std::vector<int32_t> statusList(channelNum, HCOMM_CHANNEL_STATUS_CONNECTING);
    bool allReady = false;
    bool connectFailed = false;
    uint32_t elapsedMs = 0;
    while (elapsedMs <= pollTimeoutMs) {
        auto statusRet = DlHcommApi::HcommChannelGetStatus(channelPtrs_.data(), channelNum, statusList.data());
        if (statusRet != 0) {
            SHM_LOG_ERROR("rank[" << rankId_ << "] HcommChannelGetStatus failed: " << statusRet);
            return ACLSHMEM_INNER_ERROR;
        }

        allReady = true;
        connectFailed = false;
        for (uint32_t i = 0; i < channelNum; ++i) {
            const int32_t status = statusList[i];
            if (status == HCOMM_CHANNEL_STATUS_READY) {
                continue;
            }
            allReady = false;
            if (status == HCOMM_CHANNEL_STATUS_CONNECTING) {
                continue; // 建链中，继续等待
            }
            if (status == HCOMM_CHANNEL_STATUS_FAILED) {
                SHM_LOG_ERROR("rank[" << rankId_ << "] channel[" << i << "] connect failed, status=" << status);
                connectFailed = true;
            } else if (status == HCOMM_CHANNEL_STATUS_TIMEOUT) {
                SHM_LOG_ERROR("rank[" << rankId_ << "] channel[" << i << "] connect timeout, status=" << status);
                connectFailed = true;
            } else {
                SHM_LOG_ERROR("rank[" << rankId_ << "] channel[" << i << "] unknown status=" << status);
                connectFailed = true;
            }
        }
        if (allReady) {
            break;
        }
        if (connectFailed) {
            SHM_LOG_ERROR(
                "rank[" << rankId_ << "] channel connect failed, stop polling, channelNum=" << channelNum
                        << ", elapsedMs=" << elapsedMs);
            return ACLSHMEM_INNER_ERROR;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
        elapsedMs += pollIntervalMs;
    }

    if (!allReady) {
        SHM_LOG_ERROR(
            "rank[" << rankId_ << "] wait channel connect timeout, channelNum=" << channelNum << ", pollIntervalMs="
                    << pollIntervalMs << ", pollTimeoutMs=" << pollTimeoutMs << ", elapsedMs=" << elapsedMs);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("rank[" << rankId_ << "] all channels connected, elapsedMs=" << elapsedMs);

    auto fillRet = FillRdmaInfo();
    if (fillRet != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] FillRdmaInfo failed: " << fillRet);
        return fillRet;
    }

    channelGuard.release();
    SHM_LOG_INFO("rank[" << rankId_ << "] Connect success, created " << channelNum << " channels");
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::CheckQpNumConsistency() const
{
    if (rankCount_ <= 1) {
        return ACLSHMEM_SUCCESS;
    }
    if (g_boot_handle.allgather == nullptr) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] bootstrap allgather is unavailable for QP consistency check");
        return ACLSHMEM_INNER_ERROR;
    }

    std::vector<uint32_t> allQpNums(rankCount_, 0);
    auto ret = g_boot_handle.allgather(&qpNum_, allQpNums.data(), sizeof(qpNum_), &g_boot_handle);
    if (ret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] allgather rdma qp num failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    for (uint32_t rank = 0; rank < rankCount_; ++rank) {
        if (allQpNums[rank] != qpNum_) {
            SHM_LOG_ERROR(
                "rank[" << rankId_ << "] inconsistent rdma qp num: local=" << qpNum_ << ", rank[" << rank
                        << "]=" << allQpNums[rank]);
            return ACLSHMEM_INVALID_PARAM;
        }
    }
    return ACLSHMEM_SUCCESS;
}

int RdmaTransportManagerV2::CheckPrepareOptions(const shm::transport::HybmTransPrepareOptions& options)
{
    if (role_ != hybm_role_type::HYBM_ROLE_PEER) {
        SHM_LOG_INFO(
            "rank[" << rankId_ << "] transport role: " << static_cast<uint32_t>(role_) << " check options passed.");
        return ACLSHMEM_SUCCESS;
    }

    if (options.options.size() > rankCount_) {
        SHM_LOG_ERROR(
            "rank[" << rankId_ << "] options size():" << options.options.size()
                    << " larger than rank count: " << rankCount_);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.options.find(rankId_) == options.options.end()) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] options not contains self rankId: " << rankId_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rankCount_) {
            SHM_LOG_ERROR(
                "rank[" << rankId_ << "] input options of nics contains rankId:" << it->first
                        << ", rank count: " << rankCount_);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    return ACLSHMEM_SUCCESS;
}

void RdmaTransportManagerV2::CopyAiWQInfo(struct AiQpRMAWQ& dest, const SqContext& src) noexcept
{
    const auto roceSq = DecodeRoceSqContext(src);
    dest.wqn = roceSq.qpn;
    dest.bufAddr = roceSq.sqVa;
    dest.wqeSize = roceSq.wqeSize;
    dest.depth = roceSq.depth;
    dest.headAddr = roceSq.headAddr;
    dest.tailAddr = roceSq.tailAddr;
    dest.dbMode = roceSq.dbMode;
    dest.dbAddr = roceSq.dbHwVa;
    dest.dbSwVa = roceSq.dbSwVa;
    dest.sl = roceSq.sl;
    dest.mtuShift = roceSq.mtuShift;
    dest.dbCos = roceSq.dbCos;
    SHM_LOG_DEBUG(
        "rank[" << rankId_ << "] CopyAiWQInfo(" << roceSq.abiName << "), wqn=" << dest.wqn << ", bufAddr=0x" << std::hex
                << dest.bufAddr << std::dec << ", wqeSize=" << dest.wqeSize << ", depth=" << dest.depth
                << ", headAddr=0x" << std::hex << dest.headAddr << std::dec << ", tailAddr=0x" << std::hex
                << dest.tailAddr << std::dec << ", sl=" << dest.sl << ", dbAddr=0x" << std::hex << dest.dbAddr
                << ", dbSwVa=0x" << dest.dbSwVa << std::dec << ", mtuShift=" << static_cast<int>(dest.mtuShift)
                << ", dbCos=" << static_cast<int>(dest.dbCos) << ", dbMode=" << static_cast<int>(dest.dbMode));
}

void RdmaTransportManagerV2::CopyAiCQInfo(struct AiQpRMACQ& dest, const CqContext& src) noexcept
{
    const auto roceCq = DecodeRoceCqContext(src);
    dest.cqn = roceCq.cqn;
    dest.bufAddr = roceCq.cqVa;
    dest.cqeSize = roceCq.cqeSize;
    dest.depth = roceCq.cqDepth;
    dest.headAddr = roceCq.headAddr;
    dest.tailAddr = roceCq.tailAddr;
    dest.dbMode = roceCq.dbMode;
    dest.dbAddr = roceCq.dbHwVa;
    dest.dbSwVa = roceCq.dbSwVa;
    if constexpr (IsHcommSupportCqOverrun()) {
        dest.cqAttrFlags = (std::strcmp(roceCq.abiName, "initial") == 0) ? 0 : cqAttrFlags_;
    } else {
        dest.cqAttrFlags = 0;
    }
    SHM_LOG_DEBUG(
        "rank[" << rankId_ << "] CopyAiCQInfo(" << roceCq.abiName << "), cqn=" << dest.cqn << ", bufAddr=0x" << std::hex
                << dest.bufAddr << std::dec << ", cqeSize=" << dest.cqeSize << ", depth=" << dest.depth
                << ", headAddr=0x" << std::hex << dest.headAddr << std::dec << ", tailAddr=0x" << std::hex
                << dest.tailAddr << std::dec << ", dbAddr=0x" << std::hex << dest.dbAddr << ", dbSwVa=0x" << dest.dbSwVa
                << std::dec << ", dbMode=" << static_cast<int>(dest.dbMode) << ", cqAttrFlags=" << dest.cqAttrFlags);
}

void RdmaTransportManagerV2::FillQpPreSettingCopyInfo(AiQpRMAQueueInfo*& copyInfo)
{
    copyInfo->count = qpNum_;
    copyInfo->sq = (AiQpRMAWQ*)(void*)(copyInfo + 1);
    copyInfo->rq = (AiQpRMAWQ*)(void*)(copyInfo->sq + static_cast<uint64_t>(rankCount_) * qpNum_);
    copyInfo->scq = (AiQpRMACQ*)(void*)(copyInfo->rq + static_cast<uint64_t>(rankCount_) * qpNum_);
    copyInfo->rcq = (AiQpRMACQ*)(void*)(copyInfo->scq + static_cast<uint64_t>(rankCount_) * qpNum_);
    copyInfo->mr = (RdmaMemRegionInfo*)(void*)(copyInfo->rcq + static_cast<uint64_t>(rankCount_) * qpNum_);
}

void RdmaTransportManagerV2::FillQpPostSettingCopyInfo(AiQpRMAQueueInfo*& copyInfo)
{
    auto pointer = (ptrdiff_t)(void*)(qpInfo_);
    pointer += sizeof(AiQpRMAQueueInfo);
    copyInfo->sq = (AiQpRMAWQ*)(void*)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMAWQ) * static_cast<uint64_t>(rankCount_) * qpNum_);
    copyInfo->rq = (AiQpRMAWQ*)(void*)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMAWQ) * static_cast<uint64_t>(rankCount_) * qpNum_);
    copyInfo->scq = (AiQpRMACQ*)(void*)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMACQ) * static_cast<uint64_t>(rankCount_) * qpNum_);
    copyInfo->rcq = (AiQpRMACQ*)(void*)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMACQ) * static_cast<uint64_t>(rankCount_) * qpNum_);
    copyInfo->mr = (RdmaMemRegionInfo*)(void*)pointer;
}

Result RdmaTransportManagerV2::FillRdmaInfo()
{
    std::vector<uint8_t> qpInfoBuffer(qpInfoSize_);
    auto copyInfo = (AiQpRMAQueueInfo*)(void*)qpInfoBuffer.data();

    FillQpPreSettingCopyInfo(copyInfo);

    auto ret = GetRdmaInfoFromChannelEntity(copyInfo, channelPtrs_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] GetRdmaInfoFromChannelEntity failed: " << ret);
        return ret;
    }

    PrintHostInfo(*copyInfo);

    FillQpPostSettingCopyInfo(copyInfo);

    auto aclRet = DlAclApi::AclrtMemcpy(qpInfo_, qpInfoSize_, copyInfo, qpInfoSize_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] copy qp info to device failed: " << aclRet);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("rank[" << rankId_ << "] copy qp info success");
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManagerV2::GetRdmaInfoFromChannelEntity(
    AiQpRMAQueueInfo* copyInfo, const std::vector<ChannelHandle>& channelPtrs)
{
    ChannelEntity hostEntity{};
    bool localInfoRead = false;

    for (const auto& channelPtr : channelPtrs) {
        if (channelPtr == 0) {
            continue;
        }
        auto aclRet = DlAclApi::AclrtMemcpy(
            &hostEntity, sizeof(ChannelEntity), reinterpret_cast<void*>(channelPtr), sizeof(ChannelEntity),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != 0) {
            SHM_LOG_ERROR("rank[" << rankId_ << "] pre-read channel entity failed: " << aclRet);
            continue;
        }
        if (hostEntity.localBufferNum != RDMA_CHANNEL_MEM_HANDLE_NUM) {
            SHM_LOG_ERROR(
                "rank[" << rankId_ << "] invalid localBufferNum=" << hostEntity.localBufferNum << ", expected "
                        << RDMA_CHANNEL_MEM_HANDLE_NUM);
            return ACLSHMEM_INNER_ERROR;
        }
        if (hostEntity.localBufferAddr == nullptr) {
            SHM_LOG_ERROR("rank[" << rankId_ << "] localBufferAddr is null.");
            return ACLSHMEM_INNER_ERROR;
        }
        if (hostEntity.localBufferNum > 0 && hostEntity.localBufferAddr != nullptr) {
            RegedBufferEntity localBuffer{};
            aclRet = DlAclApi::AclrtMemcpy(
                &localBuffer, sizeof(RegedBufferEntity), hostEntity.localBufferAddr, sizeof(RegedBufferEntity),
                ACL_MEMCPY_DEVICE_TO_HOST);
            if (aclRet != 0) {
                SHM_LOG_ERROR("rank[" << rankId_ << "] copy local buffer from device failed: " << aclRet);
            } else {
                copyInfo->mr[rankId_].addr = localBuffer.bufferInfo.rma.addr;
                copyInfo->mr[rankId_].size = localBuffer.bufferInfo.rma.size;
                copyInfo->mr[rankId_].lkey = localBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
                copyInfo->mr[rankId_].rkey = localBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.rkey;
                localInfoRead = true;
            }
        }
        if (hostEntity.localBufferNum > 1) {
            RegedBufferEntity atomicLocalBuffer{};
            aclRet = DlAclApi::AclrtMemcpy(
                &atomicLocalBuffer, sizeof(RegedBufferEntity),
                reinterpret_cast<RegedBufferEntity*>(hostEntity.localBufferAddr) + 1, sizeof(RegedBufferEntity),
                ACL_MEMCPY_DEVICE_TO_HOST);
            if (aclRet != 0) {
                SHM_LOG_ERROR("rank[" << rankId_ << "] pre-read atomic local buffer failed: " << aclRet);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
                // Atomics return the old value to the registered AtomicBuffer.
                return ACLSHMEM_INNER_ERROR;
#endif
            } else {
                atomicLkey_ = atomicLocalBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
                SHM_LOG_DEBUG("rank[" << rankId_ << "] atomicLkey=" << atomicLkey_);
            }
        }
        break;
    }

    if (!localInfoRead) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] failed to read local buffer info from any channel entity");
        return ACLSHMEM_INNER_ERROR;
    }

    bool remoteInfoRead = false;
    for (auto it = rankInfo_.begin(); it != rankInfo_.end(); ++it) {
        auto& map = it->second.memoryMap;
        if (map.empty()) {
            continue;
        }
        copyInfo->mr[it->first].size = map.begin()->second.size;
        copyInfo->mr[it->first].addr = map.begin()->second.address;
        if (it->first == rankId_) {
            continue;
        }

        const uint32_t remoteRank = it->first;
        const uint32_t remoteOrdinal = remoteRank - (remoteRank > rankId_ ? 1U : 0U);
        for (uint32_t qpIdx = 0; qpIdx < qpNum_; ++qpIdx) {
            const uint64_t channelIdx64 = static_cast<uint64_t>(remoteOrdinal) * qpNum_ + qpIdx;
            if (channelIdx64 >= channelPtrs.size()) {
                SHM_LOG_ERROR(
                    "rank[" << rankId_ << "] channel index overflow for rank " << remoteRank << ", qp=" << qpIdx
                            << ", idx=" << channelIdx64);
                return ACLSHMEM_INNER_ERROR;
            }

            ChannelHandle channelPtr = channelPtrs[static_cast<size_t>(channelIdx64)];
            if (channelPtr == 0) {
                SHM_LOG_ERROR(
                    "rank[" << rankId_ << "] channel ptr is null for rank " << remoteRank << ", qp=" << qpIdx);
                return ACLSHMEM_INNER_ERROR;
            }

            hostEntity = {};
            auto aclRet = DlAclApi::AclrtMemcpy(
                &hostEntity, sizeof(ChannelEntity), reinterpret_cast<void*>(channelPtr), sizeof(ChannelEntity),
                ACL_MEMCPY_DEVICE_TO_HOST);
            if (aclRet != 0) {
                SHM_LOG_ERROR(
                    "rank[" << rankId_ << "] copy channel entity from device failed: " << aclRet
                            << ", rank=" << remoteRank << ", qp=" << qpIdx);
                return ACLSHMEM_INNER_ERROR;
            }
            if (hostEntity.sqNum != 1U || hostEntity.cqNum != 1U ||
                hostEntity.remoteBufferNum != RDMA_CHANNEL_MEM_HANDLE_NUM) {
                SHM_LOG_ERROR(
                    "rank[" << rankId_ << "] invalid channel entity for rank " << remoteRank << ", qp=" << qpIdx
                            << ", sqNum=" << hostEntity.sqNum << ", cqNum=" << hostEntity.cqNum
                            << ", remoteBufferNum=" << hostEntity.remoteBufferNum << ", expected sqNum=1, cqNum=1"
                            << ", remoteBufferNum=" << RDMA_CHANNEL_MEM_HANDLE_NUM);
                return ACLSHMEM_INNER_ERROR;
            }

            if (hostEntity.remoteBufferNum > 0 && hostEntity.remoteBufferAddr != nullptr) {
                RegedBufferEntity remoteBuffer{};
                aclRet = DlAclApi::AclrtMemcpy(
                    &remoteBuffer, sizeof(RegedBufferEntity), hostEntity.remoteBufferAddr, sizeof(RegedBufferEntity),
                    ACL_MEMCPY_DEVICE_TO_HOST);
                if (aclRet != 0) {
                    SHM_LOG_ERROR(
                        "rank[" << rankId_ << "] copy remote buffer from device failed: " << aclRet
                                << ", rank=" << remoteRank << ", qp=" << qpIdx);
                    return ACLSHMEM_INNER_ERROR;
                }
                copyInfo->mr[remoteRank].addr = remoteBuffer.bufferInfo.rma.addr;
                copyInfo->mr[remoteRank].size = remoteBuffer.bufferInfo.rma.size;
                copyInfo->mr[remoteRank].lkey = remoteBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
                copyInfo->mr[remoteRank].rkey = remoteBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.rkey;
                remoteInfoRead = true;
            } else {
                SHM_LOG_ERROR(
                    "rank[" << rankId_ << "] remoteBufferNum = 0 || remoteBufferAddr is null, rank=" << remoteRank
                            << ", qp=" << qpIdx);
                return ACLSHMEM_INNER_ERROR;
            }

            if (hostEntity.sqNum > 0 && hostEntity.sqContextAddr != nullptr) {
                std::vector<SqContext> sqContexts(hostEntity.sqNum);
                aclRet = DlAclApi::AclrtMemcpy(
                    sqContexts.data(), sizeof(SqContext) * hostEntity.sqNum, hostEntity.sqContextAddr,
                    sizeof(SqContext) * hostEntity.sqNum, ACL_MEMCPY_DEVICE_TO_HOST);
                if (aclRet != 0) {
                    SHM_LOG_ERROR(
                        "rank[" << rankId_ << "] copy sq context from device failed: " << aclRet
                                << ", rank=" << remoteRank << ", qp=" << qpIdx);
                    return ACLSHMEM_INNER_ERROR;
                }

                CopyAiWQInfo(copyInfo->sq[static_cast<uint64_t>(remoteRank) * qpNum_ + qpIdx], sqContexts[0]);
            } else {
                SHM_LOG_ERROR(
                    "rank[" << rankId_ << "] sqNum = 0 || sqContextAddr is null, rank=" << remoteRank
                            << ", qp=" << qpIdx);
                return ACLSHMEM_INNER_ERROR;
            }

            if (hostEntity.cqNum > 0 && hostEntity.cqContextAddr != nullptr) {
                std::vector<CqContext> cqContexts(hostEntity.cqNum);
                aclRet = DlAclApi::AclrtMemcpy(
                    cqContexts.data(), sizeof(CqContext) * hostEntity.cqNum, hostEntity.cqContextAddr,
                    sizeof(CqContext) * hostEntity.cqNum, ACL_MEMCPY_DEVICE_TO_HOST);
                if (aclRet != 0) {
                    SHM_LOG_ERROR(
                        "rank[" << rankId_ << "] copy cq context from device failed: " << aclRet
                                << ", rank=" << remoteRank << ", qp=" << qpIdx);
                    return ACLSHMEM_INNER_ERROR;
                }

                CopyAiCQInfo(copyInfo->scq[static_cast<uint64_t>(remoteRank) * qpNum_ + qpIdx], cqContexts[0]);
            } else {
                SHM_LOG_ERROR(
                    "rank[" << rankId_ << "] cqNum = 0 || cqContextAddr is null, rank=" << remoteRank
                            << ", qp=" << qpIdx);
                return ACLSHMEM_INNER_ERROR;
            }
        }

        const size_t atomicSizePerRank = ATOMIC_MAX_NUM * sizeof(uint64_t);
        const uint64_t atomicAddr =
            reinterpret_cast<uint64_t>(static_cast<char*>(atomicSharedMemory_) + it->first * atomicSizePerRank);
        for (uint32_t qpIdx = 0; qpIdx < qpNum_; ++qpIdx) {
            const uint64_t qpOffset = static_cast<uint64_t>(it->first) * qpNum_ + qpIdx;
            copyInfo->sq[qpOffset].atomicAddr = atomicAddr;
            copyInfo->sq[qpOffset].atomicLkey = atomicLkey_;
            copyInfo->rq[qpOffset].atomicAddr = atomicAddr;
            copyInfo->rq[qpOffset].atomicLkey = atomicLkey_;
        }
    }

    if (!remoteInfoRead && rankCount_ > 1) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] failed to read remote buffer info from any channel entity");
        return ACLSHMEM_INNER_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

void RdmaTransportManagerV2::PrintHostInfo(AiQpRMAQueueInfo& copyInfo)
{
    SHM_LOG_DEBUG("=======================rank [" << rankId_ << "] host info====================");
    auto tempMemInfo = ((RdmaMemRegionInfo*)copyInfo.mr)[rankId_];
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.size: " << tempMemInfo.size);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.addr: " << tempMemInfo.addr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.lkey: " << tempMemInfo.lkey);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.rkey: " << tempMemInfo.rkey);
}

bool RdmaTransportManagerV2::ReserveRdmaInfoSpace() noexcept
{
    // reserve qp info space
    if (qpInfo_ != nullptr) {
        return true;
    }

    void* ptr = nullptr;
    auto oneQpSize = 2U * (sizeof(AiQpRMAWQ) + sizeof(AiQpRMACQ)) + sizeof(RdmaMemRegionInfo);
    const uint64_t qpEntryCount = static_cast<uint64_t>(rankCount_) * qpNum_;
    qpInfoSize_ = sizeof(AiQpRMAQueueInfo) + oneQpSize * qpEntryCount;
    auto ret = DlAclApi::AclrtMalloc(&ptr, qpInfoSize_, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] allocate device size: " << qpInfoSize_ << ", failed: " << ret);
        return false;
    }

    qpInfo_ = (AiQpRMAQueueInfo*)ptr;

    // reserve atomic info space
    if (atomicSharedMemory_ != nullptr) {
        return true;
    }

    uint32_t atomicSize = ATOMIC_MAX_NUM * sizeof(uint64_t) * rankCount_; // 128 是最大的 atomic 并发数
    // HNS 1825 atomic MR 注册需要 4K 对齐。
    atomicSize = ALIGN_UP(atomicSize, 4096);
    ret = DlAclApi::AclrtMalloc(&atomicSharedMemory_, atomicSize, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] allocate device atomic size: " << atomicSize << ", failed: " << ret);
        return false;
    }
    atomicLkey_ = 0;
    return true;
}

bool RdmaTransportManagerV2::RegisterAtomicMemory() noexcept
{
    if (atomicMemHandle_ != nullptr) {
        return true;
    }

    if (atomicSharedMemory_ == nullptr || endpointHandle_ == nullptr) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] atomic memory or endpoint not ready for registration");
        return false;
    }

    uint32_t atomicSize = ATOMIC_MAX_NUM * sizeof(uint64_t) * rankCount_;
    // HNS 1825 atomic MR 注册需要 4K 对齐。
    atomicSize = ALIGN_UP(atomicSize, 4096);

    CommMem commMem{};
    commMem.type = CommMemType::DEVICE;
    commMem.addr = atomicSharedMemory_;
    commMem.size = atomicSize;

    HcommMemHandle memHandle = nullptr;
    SHM_LOG_DEBUG(
        "rank[" << rankId_ << "] HcommMemReg for atomic memory, addr=" << atomicSharedMemory_
                << ", size=" << atomicSize);
    HcommResult hret = DlHcommApi::HcommMemReg(endpointHandle_, "AtomicBuffer", &commMem, &memHandle);
    if (hret != 0) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] HcommMemReg for atomic memory failed: " << hret);
        return false;
    }

    atomicMemHandle_ = memHandle;
    SHM_LOG_DEBUG("rank[" << rankId_ << "] atomic memory registered, memHandle=" << memHandle);

    return true;
}

Result RdmaTransportManagerV2::UpdateRankOptions(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG("rank[" << rankId_ << "] RdmaTransportManagerV2 Prepare with : " << options);

    mf_sockaddr deviceNetwork;
    std::unordered_map<uint32_t, ConnectRankInfo> ranksInfo;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        auto ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("rank[" << rankId_ << "] update rank network(" << it->second.nic << ") invalid.");
            return ACLSHMEM_INVALID_PARAM;
        }
        SHM_LOG_INFO("rank[" << rankId_ << "] UpdateRankOptions update rank: " << it->first);
        ranksInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    SHM_LOG_DEBUG("rank[" << rankId_ << "] UpdateRankOptions ranksInfo.size=" << ranksInfo.size());

    rankInfo_ = ranksInfo;

    return ACLSHMEM_SUCCESS;
}

const std::string& RdmaTransportManagerV2::GetNic() const { return nicInfo_; }

const void* RdmaTransportManagerV2::GetQpInfo() const
{
    if (qpInfo_ == nullptr) {
        SHM_LOG_ERROR("rank[" << rankId_ << "] GetQpInfo():qpInfo_ is nullptr.");
        return nullptr;
    }
    return qpInfo_;
}

} // namespace device
} // namespace transport
} // namespace shm
