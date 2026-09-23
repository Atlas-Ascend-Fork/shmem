/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE

#include <cstring>
#include <iomanip>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"

#include "dl_comm_def.h"
#include "mem_entity_def.h"
#include "devmm_svm_gva.h"
#include "shmemi_logger.h"
#include "shmemi_net_util.h"
#include "hybm_ex_info_transfer.h"
#include "hybm_device_mem_segment.h"
#include "init/shmemi_user_buffer_heap.h"

namespace shm {
Result AclMemSegmentDevice::ValidateOptions() noexcept
{
    if (options_.size == 0 || options_.devId < 0 || (options_.size % DEVICE_LARGE_PAGE_SIZE) != 0) {
        return ACLSHMEM_INVALID_PARAM;
    }

    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::ReserveEachPeMemorySpace(
    size_t reserveAlignedSize, size_t totalReservedSize, uint64_t expectSt) noexcept
{
    // because 950 not support specify va
    void* base = nullptr;
    if (socType_ == AscendSocType::ASCEND_950) {
        auto flags = options_.userBufferHeapInput == nullptr ? 1U : 0U;
        auto ret = aclrtReserveMemAddress(&base, totalReservedSize, 0, nullptr, flags);
        if (ret != 0 || base == 0) {
            SHM_LOG_ERROR("prepare virtual memory size(" << totalReservedSize << ") failed. ret: " << ret);
            return ACLSHMEM_MALLOC_FAILED;
        }
        uint8_t* currentAddr = static_cast<uint8_t*>(base);
        for (uint32_t i = 0; i < options_.rankCnt; i++) {
            reservedVirtualAddresses_.emplace_back(reinterpret_cast<uint64_t>(currentAddr));
            SHM_LOG_INFO(
                "rankId: " << i << ", vaddr: " << (void*)currentAddr << " size: " << options_.size
                           << " align_size: " << reserveAlignedSize);
            currentAddr += reserveAlignedSize;
        }
        totalVirtualSize_ = totalReservedSize;
        return ACLSHMEM_SUCCESS;
    }
    uint8_t* curBase = (uint8_t*)expectSt;
    for (uint32_t i = 0; i < options_.rankCnt; i++) {
        auto ret = aclrtReserveMemAddress(&base, reserveAlignedSize, 0, curBase, 1);
        if (ret != 0 || base == 0) {
            SHM_LOG_ERROR("prepare virtual memory size(" << totalVirtualSize_ << ") failed. ret: " << ret);
            const size_t reservedCount = reservedVirtualAddresses_.size();
            size_t retainedCount = 0;
            for (size_t index = 0; index < reservedCount; ++index) {
                const auto reserved = reservedVirtualAddresses_[index];
                if (aclrtReleaseMemAddress(reinterpret_cast<void*>(reserved)) != ACL_SUCCESS) {
                    reservedVirtualAddresses_[retainedCount++] = reserved;
                }
            }
            reservedVirtualAddresses_.resize(retainedCount);
            totalVirtualSize_ = retainedCount * reserveAlignedSize;
            return ACLSHMEM_MALLOC_FAILED;
        }
        SHM_LOG_INFO(
            "success to reserve memory space for logic deviceid " << logicDeviceId_ << ", vaddr: " << (void*)base
                                                                  << " size: " << options_.size << ", rankId: " << i);
        totalVirtualSize_ += reserveAlignedSize;
        reservedVirtualAddresses_.emplace_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)));
        curBase = (uint8_t*)base + reserveAlignedSize;
    }
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::ReserveMemorySpace(void** address) noexcept
{
    if (globalVirtualAddress_ != nullptr) {
        SHM_LOG_ERROR("already prepare virtual memory.");
        return ACLSHMEM_INNER_ERROR;
    }
    if (options_.size > std::numeric_limits<size_t>::max() - (DEVMM_HEAP_SIZE - 1)) {
        return ACLSHMEM_INVALID_VALUE;
    }
    size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    if (options_.rankCnt != 0 && reserveAlignedSize > std::numeric_limits<size_t>::max() / options_.rankCnt) {
        return ACLSHMEM_INVALID_VALUE;
    }
    size_t totalReservedSize = options_.rankCnt * reserveAlignedSize;
    try {
        reservedVirtualAddresses_.reserve(options_.rankCnt);
    } catch (const std::bad_alloc&) {
        return ACLSHMEM_MALLOC_FAILED;
    }
    uint64_t expectSt = HYBM_GVM_START_ADDR;
    const bool isAscend910BOr910C = socType_ == AscendSocType::ASCEND_910B || socType_ == AscendSocType::ASCEND_910C;
    const bool useFixedBase = options_.userBufferHeapInput != nullptr && isAscend910BOr910C &&
                              totalReservedSize >= HYBM_MAX_SINGLE_RESERVE_SIZE;
    if (useFixedBase) {
        if (expectSt >= HYBM_DEVICE_META_ADDR || totalReservedSize > HYBM_DEVICE_META_ADDR - expectSt) {
            SHM_LOG_ERROR(
                "user-buffer virtual memory size(" << totalReservedSize
                                                   << ") exceeds the fixed virtual address range.");
            return ACLSHMEM_INVALID_VALUE;
        }
        SHM_LOG_INFO(
            "use fixed user-buffer base " << reinterpret_cast<void*>(expectSt) << " for virtual memory size "
                                          << totalReservedSize);
    } else if (FindAvaliableVirtualAddr(totalReservedSize, expectSt) != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("prepare virtual memory size(" << totalReservedSize << ") failed.");
        return ACLSHMEM_MALLOC_FAILED;
    }
    if (ReserveEachPeMemorySpace(reserveAlignedSize, totalReservedSize, expectSt) != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("ReserveEachPeMemorySpace virtual memory size(" << totalReservedSize << ") failed.");
        return ACLSHMEM_MALLOC_FAILED;
    }

    globalVirtualAddress_ = reinterpret_cast<uint8_t*>(reservedVirtualAddresses_[0]);
    allocatedSize_ = 0UL;
    sliceCount_ = 0;
    *address = reinterpret_cast<void*>(reservedVirtualAddresses_[0]);
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::FindAvaliableVirtualAddr(uint64_t size, uint64_t& baseVa) noexcept
{
    void* base;
    auto ret = aclrtReserveMemAddress(&base, size, 0, nullptr, 1);
    if (ret != 0 || base == 0) {
        SHM_LOG_ERROR("prepare virtual memory size to (" << size << ") failed. ret: " << ret);
        return ACLSHMEM_MALLOC_FAILED;
    }
    baseVa = (uint64_t)base;
    ret = aclrtReleaseMemAddress(base);
    if (ret != 0) {
        SHM_LOG_ERROR("aclrtReleaseMemAddress failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::SetMemAccess() noexcept
{
    if (options_.segType == HYBM_MST_DRAM) {
        aclrtMemLocation loc;
        loc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        loc.id = options_.rankId;
        aclrtMemAccessDesc des;
        des.location = loc;
        des.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
        ACLSHMEM_CHECK_RET(
            aclrtMemSetAccess(reinterpret_cast<void*>(localVirtualBase), options_.size, &des, 1),
            "aclrtMemSetAccess failed.", ACLSHMEM_SMEM_ERROR);
    }
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::UnReserveMemorySpace() noexcept
{
    SHM_LOG_INFO("un-reserve memory space.");
    if (options_.userBufferHeapInput != nullptr) {
        return ReleaseUserBufferHeapResources();
    }
    FreeMemory();
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::AllocLocalMemory(uint64_t size, std::shared_ptr<MemSlice>& slice) noexcept
{
    if (options_.userBufferHeapInput != nullptr) {
        if (size != options_.size) {
            return ACLSHMEM_INVALID_PARAM;
        }
        return BuildUserBufferHeap(slice);
    }
    if ((size % DEVICE_LARGE_PAGE_SIZE) != 0UL || size + allocatedSize_ > options_.size) {
        SHM_LOG_ERROR(
            "invalid allocate memory size : " << size << ", now used " << allocatedSize_ << " of " << options_.size);
        return ACLSHMEM_INVALID_PARAM;
    }
    // VA is pre-reserved by ReserveMemorySpace and released by UnReserveMemorySpace/FreeMemory.
    // AllocLocalMemory does not own the VA — failure paths must not call aclrtReleaseMemAddress on it.
    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = (options_.segType == HYBM_MST_DRAM) ? ACL_DDR_MEM_P2P_HUGE : ACL_HBM_MEM_HUGE;
    prop.location.id = (options_.segType == HYBM_MST_DRAM) ? -1 : deviceId_;
    prop.location.type =
        (options_.segType == HYBM_MST_DRAM) ? ACL_MEM_LOCATION_TYPE_HOST_NUMA : ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;
    auto ret = aclrtMallocPhysical(&local_handle_, size, &prop, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(
            options_.rankId << " aclrtMallocPhysical memory failed: " << ret << " segType: " << (int)options_.segType);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    SHM_LOG_DEBUG(
        options_.rankId << " aclrtMallocPhysical memory success size: " << size << " vaddr: "
                        << reinterpret_cast<void*>(localVirtualBase) << " segType: " << (int)options_.segType);
    ret = aclrtMapMem(reinterpret_cast<void*>(localVirtualBase), size, 0, local_handle_, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("aclrtMapMem failed: " << ret);
        (void)aclrtFreePhysical(local_handle_);
        local_handle_ = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    if (SetMemAccess() != ACLSHMEM_SUCCESS) {
        (void)aclrtUnmapMem(reinterpret_cast<void*>(localVirtualBase));
        (void)aclrtFreePhysical(local_handle_);
        local_handle_ = nullptr;
        return ACLSHMEM_SMEM_ERROR;
    }

    slice = std::make_shared<MemSlice>(
        sliceCount_++, MEM_TYPE_DEVICE_HBM, MEM_PT_TYPE_SVM, reinterpret_cast<uint64_t>(localVirtualBase), size);
    slices_.emplace(slice->index_, slice);
    SHM_LOG_DEBUG("allocate slice(idx:" << slice->index_ << ", size:" << slice->size_ << ").");
    return ACLSHMEM_SUCCESS;
}

uint64_t AclMemSegmentDevice::ExternalBytes() const noexcept
{
    uint64_t total = 0;
    if (options_.userBufferHeapInput == nullptr) {
        return total;
    }
    for (const auto& entry : options_.userBufferHeapInput->entries) {
        total += entry.size;
    }
    return total;
}

uint64_t AclMemSegmentDevice::SegmentOffset(size_t segmentIndex) const noexcept
{
    if (options_.userBufferHeapInput == nullptr) {
        return 0;
    }
    const auto& entries = options_.userBufferHeapInput->entries;
    if (segmentIndex < entries.size()) {
        return entries[segmentIndex].segment_offset;
    }
    return segmentIndex == entries.size() ? ExternalBytes() : 0;
}

uint64_t AclMemSegmentDevice::SegmentSize(size_t segmentIndex) const noexcept
{
    if (options_.userBufferHeapInput == nullptr) {
        return 0;
    }
    const auto& entries = options_.userBufferHeapInput->entries;
    if (segmentIndex < entries.size()) {
        return entries[segmentIndex].size;
    }
    if (segmentIndex == entries.size()) {
        return options_.size - ExternalBytes();
    }
    return 0;
}

Result AclMemSegmentDevice::BuildUserBufferHeap(std::shared_ptr<MemSlice>& slice) noexcept
try {
    if (options_.userBufferHeapInput == nullptr || reservedVirtualAddresses_.size() != options_.rankCnt ||
        options_.rankId >= reservedVirtualAddresses_.size() || local_handle_ != nullptr || !slices_.empty()) {
        return ACLSHMEM_INVALID_PARAM;
    }
    const auto& entries = options_.userBufferHeapInput->entries;
    const uint64_t externalBytes = ExternalBytes();
    if (entries.empty() || externalBytes > options_.size) {
        return ACLSHMEM_INVALID_PARAM;
    }
    const uint64_t ownedBytes = options_.size - externalBytes;
    const uint64_t localSlot = reservedVirtualAddresses_[options_.rankId];

    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId_;
    prop.reserve = 0;
    auto ret = aclrtMallocPhysical(&local_handle_, ownedBytes, &prop, 0);
    if (ret != ACL_SUCCESS) {
        local_handle_ = nullptr;
        return ACLSHMEM_MALLOC_FAILED;
    }
    auto rollbackAndReturn = [&](Result failure) -> Result {
        const auto cleanupRet = ReleaseUserBufferHeapResources();
        if (cleanupRet != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("Failed to roll back a partially built user buffer heap, ret=" << cleanupRet);
        }
        return failure;
    };

    uint64_t offset = 0;
    for (const auto& entry : entries) {
        const uint64_t target = localSlot + offset;
        mappedMem_.insert(target);
        ret = aclrtMapMem(reinterpret_cast<void*>(target), entry.size, 0, entry.mem_handle, 0);
        if (ret != ACL_SUCCESS) {
            mappedMem_.erase(target);
            return rollbackAndReturn(ACLSHMEM_DL_FUNC_FAILED);
        }
        offset += entry.size;
    }
    const uint64_t tailTarget = localSlot + externalBytes;
    mappedMem_.insert(tailTarget);
    ret = aclrtMapMem(reinterpret_cast<void*>(tailTarget), ownedBytes, 0, local_handle_, 0);
    if (ret != ACL_SUCCESS) {
        mappedMem_.erase(tailTarget);
        return rollbackAndReturn(ACLSHMEM_DL_FUNC_FAILED);
    }
    auto accessRet = SetMemAccess();
    if (accessRet != ACLSHMEM_SUCCESS) {
        return rollbackAndReturn(accessRet);
    }

    auto candidate =
        std::make_shared<MemSlice>(sliceCount_, MEM_TYPE_DEVICE_HBM, MEM_PT_TYPE_SVM, localSlot, options_.size);
    slices_.emplace(candidate->index_, candidate);
    ++sliceCount_;
    allocatedSize_ = options_.size;
    slice = std::move(candidate);
    return ACLSHMEM_SUCCESS;
} catch (const std::bad_alloc&) {
    const auto cleanupRet = ReleaseUserBufferHeapResources();
    if (cleanupRet != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Failed to roll back user buffer heap after allocation failure, ret=" << cleanupRet);
    }
    return ACLSHMEM_MALLOC_FAILED;
}

Result AclMemSegmentDevice::RegisterMemory(const void* addr, uint64_t size, std::shared_ptr<MemSlice>& slice) noexcept
{
    SHM_LOG_INFO("MemSegmentDevice NOT SUPPORT RegisterMemory");
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::ReleaseSliceMemory(const std::shared_ptr<MemSlice>& slice) noexcept
{
    SHM_VALIDATE_RETURN(slice != nullptr, "input slice is nullptr", ACLSHMEM_INVALID_PARAM);

    auto pos = slices_.find(slice->index_);
    if (pos == slices_.end()) {
        SHM_LOG_ERROR("input slice(idx:" << slice->index_ << ") not exist.");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (pos->second.slice != slice) {
        SHM_LOG_ERROR("input slice(magic:" << std::hex << slice->magic_ << ") not match.");
        return ACLSHMEM_INVALID_PARAM;
    }
    auto res = aclrtUnmapMem(reinterpret_cast<void*>(slice->vAddress_));
    SHM_LOG_INFO("unmap slice(idx:" << slice->index_ << "), size: " << slice->size_ << " return:" << res);

    slices_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::Export(std::string& exInfo) noexcept
{
    SHM_LOG_INFO("MemSegmentDevice not supported export device info.");
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::Export(const std::shared_ptr<MemSlice>& slice, std::string& exInfo) noexcept
{
    if (slice == nullptr) {
        SHM_LOG_ERROR("input slice is nullptr");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip export");
        return ACLSHMEM_SUCCESS;
    }
    auto pos = slices_.find(slice->index_);
    if (pos == slices_.end()) {
        SHM_LOG_ERROR("input slice(idx:" << slice->index_ << ") not exist.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (pos->second.slice != slice) {
        SHM_LOG_ERROR("input slice(magic:" << std::hex << slice->magic_ << ") not match.");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto exp = exportMap_.find(slice->index_);
    if (exp != exportMap_.end()) {
        exInfo = exp->second;
        return ACLSHMEM_SUCCESS;
    }

    HbmExportInfo info{};
    auto handle_type = ACL_MEM_SHARE_HANDLE_TYPE_FABRIC;
    auto ret = aclrtMemExportToShareableHandleV2(local_handle_, 1, handle_type, &info.shareHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("aclrtMemExportToShareableHandleV2 failed, ret: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
    info.mappingOffset = slice->vAddress_ - (uint64_t)(ptrdiff_t)localVirtualBase;
    info.sliceIndex = static_cast<uint32_t>(slice->index_);
    info.deviceId = options_.devId;
    info.logicDeviceId = logicDeviceId_;
    info.devicePhyId = devicePhyId_;
    info.pid = pid_;
    info.rankId = options_.rankId;
    info.size = slice->size_;
    info.entityId = entityId_;
    info.sdid = sdid_;
    info.serverId = serverId_;
    info.superPodId = superPodId_;
    info.pageTblType = MEM_PT_TYPE_SVM;
    info.memSegType = options_.segType;
    info.exchangeType = HYBM_INFO_EXG_IN_NODE;
    info.magic = (options_.segType == HYBM_MST_DRAM) ? DRAM_SLICE_EXPORT_INFO_MAGIC : HBM_SLICE_EXPORT_INFO_MAGIC;
    ret = LiteralExInfoTranslater<HbmExportInfo>{}.Serialize(info, exInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("export info failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    exportMap_[slice->index_] = exInfo;
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::ExportUserBufferHeap(std::vector<std::string>& infos) noexcept
try {
    if (options_.userBufferHeapInput == nullptr || local_handle_ == nullptr) {
        return ACLSHMEM_INVALID_PARAM;
    }
    const auto& entries = options_.userBufferHeapInput->entries;
    const size_t wireCount = entries.size() + 1;
    infos.clear();
    infos.reserve(wireCount);

    auto fillCommon = [&](HbmExportInfo& info, size_t index, uint64_t offset, uint64_t size) {
        info.mappingOffset = offset;
        info.sliceIndex = static_cast<uint32_t>(index);
        info.deviceId = options_.devId;
        info.logicDeviceId = logicDeviceId_;
        info.devicePhyId = devicePhyId_;
        info.pid = pid_;
        info.rankId = options_.rankId;
        info.size = size;
        info.entityId = entityId_;
        info.sdid = sdid_;
        info.serverId = serverId_;
        info.superPodId = superPodId_;
        info.pageTblType = MEM_PT_TYPE_SVM;
        info.memSegType = options_.segType;
        info.exchangeType = HYBM_INFO_EXG_IN_NODE;
        info.magic = HBM_SLICE_EXPORT_INFO_MAGIC;
        info.version = EXPORT_INFO_VERSION;
    };

    auto exportHandle = [&](aclrtDrvMemHandle handle, HbmExportInfo& info, const char* description) -> Result {
        auto handleType = ACL_MEM_SHARE_HANDLE_TYPE_FABRIC;
        auto ret = aclrtMemExportToShareableHandleV2(
            handle, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION, handleType, &info.shareHandle);
        if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
            handleType = ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
            SHM_LOG_INFO(
                "Fabric VMM is not supported; exporting " << description << " with the default V2 share handle");
            ret = aclrtMemExportToShareableHandleV2(handle, 0, handleType, &info.shareHandle);
        }
        if (ret != ACL_SUCCESS) {
            SHM_LOG_ERROR(
                "failed to export " << description << ", handleType=" << static_cast<uint32_t>(handleType)
                                    << ", ret=" << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        if (handleType == ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT && options_.trustedPids != nullptr &&
            options_.trustedPidCount != 0) {
            ret = aclrtMemSetPidToShareableHandleV2(
                &info.shareHandle, handleType, options_.trustedPids, options_.trustedPidCount);
            if (ret != ACL_SUCCESS) {
                SHM_LOG_ERROR("failed to set the DEFAULT handle PID whitelist for " << description << ", ret=" << ret);
                return ACLSHMEM_INNER_ERROR;
            }
        }
        info.shareHandleType = static_cast<uint32_t>(handleType);
        return ACLSHMEM_SUCCESS;
    };

    uint64_t offset = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        HbmExportInfo info{};
        fillCommon(info, i, offset, entries[i].size);
        if (entries[i].has_fabric_handle) {
            info.shareHandle = entries[i].fabric_handle;
            info.shareHandleType = ACL_MEM_SHARE_HANDLE_TYPE_FABRIC;
        } else {
            const std::string description = "caller-provided buffer " + std::to_string(i);
            auto ret = exportHandle(entries[i].mem_handle, info, description.c_str());
            if (ret != ACLSHMEM_SUCCESS) {
                return ret;
            }
        }
        std::string serialized;
        auto ret = LiteralExInfoTranslater<HbmExportInfo>{}.Serialize(info, serialized);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("failed to serialize caller-provided buffer " << i << ", ret=" << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        infos.emplace_back(std::move(serialized));
        offset += entries[i].size;
    }

    HbmExportInfo tailInfo{};
    fillCommon(tailInfo, entries.size(), offset, options_.size - offset);
    auto ret = exportHandle(local_handle_, tailInfo, "the SHMEM-owned mixed-heap tail");
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    std::string serialized;
    ret = LiteralExInfoTranslater<HbmExportInfo>{}.Serialize(tailInfo, serialized);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("failed to serialize the SHMEM-owned mixed-heap tail, ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    infos.emplace_back(std::move(serialized));
    return ACLSHMEM_SUCCESS;
} catch (const std::bad_alloc&) {
    return ACLSHMEM_MALLOC_FAILED;
}

Result AclMemSegmentDevice::GetExportSliceSize(size_t& size) noexcept
{
    size = sizeof(HbmExportInfo);
    return ACLSHMEM_SUCCESS;
}

// import可重入
Result AclMemSegmentDevice::Import(const std::vector<std::string>& allExInfo, void* addresses[]) noexcept
{
    if (options_.userBufferHeapInput != nullptr) {
        return ImportUserBufferHeap(allExInfo);
    }
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip import");
        return ACLSHMEM_SUCCESS;
    }
    std::map<uint16_t, HbmExportInfo> importMap;
    LiteralExInfoTranslater<HbmExportInfo> translator;
    std::vector<HbmExportInfo> desInfos(allExInfo.size());
    const uint64_t expectedMagic =
        options_.segType == HYBM_MST_DRAM ? DRAM_SLICE_EXPORT_INFO_MAGIC : HBM_SLICE_EXPORT_INFO_MAGIC;
    for (auto i = 0U; i < allExInfo.size(); i++) {
        auto ret = translator.Deserialize(allExInfo[i], desInfos[i]);
        if (ret != 0) {
            SHM_LOG_ERROR("deserialize imported info(" << i << ") failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
        if (desInfos[i].magic != expectedMagic || desInfos[i].version != EXPORT_INFO_VERSION) {
            SHM_LOG_ERROR(
                "import info(" << i << ") has incompatible magic or version, magic=" << std::hex << desInfos[i].magic
                               << ", version=" << desInfos[i].version);
            return ACLSHMEM_INVALID_PARAM;
        }
        importMap.emplace(desInfos[i].rankId, desInfos[i]);
    }
    importMap_ = std::move(importMap);
    std::copy(desInfos.begin(), desInfos.end(), std::back_inserter(imports_));
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::ImportUserBufferHeap(const std::vector<std::string>& allInfos) noexcept
try {
    const size_t wireCount = options_.userBufferHeapInput->entries.size() + 1;
    if (wireCount == 0 || allInfos.size() != static_cast<size_t>(options_.rankCnt) * wireCount) {
        return ACLSHMEM_INVALID_PARAM;
    }

    LiteralExInfoTranslater<HbmExportInfo> translator;
    std::vector<HbmExportInfo> validated(allInfos.size());
    std::map<uint16_t, HbmExportInfo> canonical;
    for (uint32_t rank = 0; rank < options_.rankCnt; ++rank) {
        HbmExportInfo route{};
        for (size_t segment = 0; segment < wireCount; ++segment) {
            const size_t flat = static_cast<size_t>(rank) * wireCount + segment;
            auto ret = translator.Deserialize(allInfos[flat], validated[flat]);
            if (ret != ACLSHMEM_SUCCESS) {
                return ACLSHMEM_INVALID_PARAM;
            }
            const auto& info = validated[flat];
            if (info.magic != HBM_SLICE_EXPORT_INFO_MAGIC || info.version != EXPORT_INFO_VERSION ||
                info.rankId != rank || info.sliceIndex != segment || info.mappingOffset != SegmentOffset(segment) ||
                info.size != SegmentSize(segment) || info.memSegType != options_.segType ||
                info.pageTblType != MEM_PT_TYPE_SVM ||
                (info.shareHandleType != ACL_MEM_SHARE_HANDLE_TYPE_FABRIC &&
                 info.shareHandleType != ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT)) {
                return ACLSHMEM_INVALID_PARAM;
            }
            if (segment == 0) {
                route = info;
            } else if (
                info.deviceId != route.deviceId || info.logicDeviceId != route.logicDeviceId ||
                info.devicePhyId != route.devicePhyId || info.sdid != route.sdid || info.serverId != route.serverId ||
                info.superPodId != route.superPodId || info.entityId != route.entityId ||
                info.exchangeType != route.exchangeType) {
                return ACLSHMEM_INVALID_PARAM;
            }
        }
        canonical.emplace(static_cast<uint16_t>(rank), route);
    }
    imports_ = std::move(validated);
    importMap_ = std::move(canonical);
    return ACLSHMEM_SUCCESS;
} catch (const std::bad_alloc&) {
    return ACLSHMEM_MALLOC_FAILED;
}

Result AclMemSegmentDevice::Mmap() noexcept
{
    if (options_.userBufferHeapInput != nullptr) {
        return MapUserBufferHeap();
    }
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip map");
        return ACLSHMEM_SUCCESS;
    }
    if (imports_.empty()) {
        return ACLSHMEM_SUCCESS;
    }
    uint32_t unreachableCnt = 0;
    bool anySuspicious = false;
    for (auto& im : imports_) {
        if (im.rankId == options_.rankId) {
            continue;
        }

        auto remoteAddress = reservedVirtualAddresses_[im.rankId];
        if (mappedMem_.find((uint64_t)remoteAddress) != mappedMem_.end()) {
            SHM_LOG_INFO("remote slice on rank(" << im.rankId << ") has maped: " << (void*)remoteAddress);
            continue;
        }

        if (!CanMapRemote(im)) {
            anySuspicious = LogRemoteUnreachable(im) || anySuspicious;
            ++unreachableCnt;
            continue;
        }

        SHM_LOG_DEBUG(
            "remote slice on rank(" << im.rankId << ") should map to: " << (void*)remoteAddress
                                    << ", size = " << im.size);
        aclrtDrvMemHandle imported_handle;
        auto handle_type = ACL_MEM_SHARE_HANDLE_TYPE_FABRIC;
        ACLSHMEM_CHECK_RET(
            aclrtMemImportFromShareableHandleV2(&im.shareHandle, handle_type, 0, &imported_handle),
            "aclrtMemImportFromShareableHandleV2 failed", ACLSHMEM_SMEM_ERROR);
        auto ret = aclrtMapMem(reinterpret_cast<void*>(remoteAddress), im.size, 0, imported_handle, 0);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("aclrtMapMem memory failed:" << ret);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        mappedMem_.insert((uint64_t)remoteAddress);
    }
    LogUnreachableSummary(unreachableCnt, anySuspicious);
    imports_.clear();
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::MapUserBufferHeap() noexcept
try {
    for (auto& info : imports_) {
        if (info.rankId == options_.rankId || !CanMapRemote(info)) {
            continue;
        }
        if (info.rankId >= reservedVirtualAddresses_.size()) {
            return ACLSHMEM_INVALID_PARAM;
        }
        const uint64_t target = reservedVirtualAddresses_[info.rankId] + info.mappingOffset;
        if (mappedMem_.find(target) != mappedMem_.end()) {
            continue;
        }
        mappedMem_.insert(target);
        retainedImportedHandles_.emplace(target, nullptr);
        aclrtDrvMemHandle importedHandle = nullptr;
        const auto handleType = static_cast<aclrtMemSharedHandleType>(info.shareHandleType);
        auto importRet = aclrtMemImportFromShareableHandleV2(&info.shareHandle, handleType, 0, &importedHandle);
        if (importRet != ACL_SUCCESS) {
            SHM_LOG_ERROR(
                "failed to import rank " << info.rankId << " segment " << info.sliceIndex
                                         << ", handleType=" << info.shareHandleType << ", ret=" << importRet);
            mappedMem_.erase(target);
            retainedImportedHandles_.erase(target);
            return ACLSHMEM_SMEM_ERROR;
        }
        retainedImportedHandles_[target] = importedHandle;
        auto mapRet = aclrtMapMem(reinterpret_cast<void*>(target), info.size, 0, importedHandle, 0);
        if (mapRet != ACL_SUCCESS) {
            mappedMem_.erase(target);
            (void)aclrtFreePhysical(importedHandle);
            retainedImportedHandles_.erase(target);
            SHM_LOG_ERROR(
                "failed to map rank " << info.rankId << " segment " << info.sliceIndex
                                      << ", handleType=" << info.shareHandleType << ", ret=" << mapRet);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        if (handleType == ACL_MEM_SHARE_HANDLE_TYPE_FABRIC) {
            const auto releaseRet = aclrtFreePhysical(importedHandle);
            if (releaseRet != ACL_SUCCESS) {
                SHM_LOG_ERROR(
                    "failed to release the imported Fabric reference for rank "
                    << info.rankId << " segment " << info.sliceIndex << ", ret=" << releaseRet);
                return ACLSHMEM_DL_FUNC_FAILED;
            }
            retainedImportedHandles_.erase(target);
        }
    }
    imports_.clear();
    return ACLSHMEM_SUCCESS;
} catch (const std::bad_alloc&) {
    return ACLSHMEM_MALLOC_FAILED;
}

Result AclMemSegmentDevice::Unmap() noexcept
{
    if (options_.userBufferHeapInput != nullptr) {
        return ACLSHMEM_SUCCESS;
    }
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip unmap");
        return ACLSHMEM_SUCCESS;
    }
    for (auto va : mappedMem_) {
        int32_t ret = aclrtUnmapMem(reinterpret_cast<void*>(va));
        if (ret != 0) {
            SHM_LOG_ERROR("aclrtUnmapMem memory failed:" << ret);
        }
    }
    mappedMem_.clear();
    return ACLSHMEM_SUCCESS;
}

Result AclMemSegmentDevice::RemoveImported(const std::vector<uint32_t>& ranks) noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip remove");
        return ACLSHMEM_SUCCESS;
    }
    for (auto& rank : ranks) {
        if (rank >= options_.rankCnt) {
            SHM_LOG_ERROR("input rank is invalid! rank:" << rank << " rankSize:" << options_.rankCnt);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    for (auto& rank : ranks) {
        size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
        uint64_t addr = reinterpret_cast<uint64_t>(globalVirtualAddress_) + reserveAlignedSize * rank;
        auto it = mappedMem_.lower_bound(addr);
        auto st = it;
        while (it != mappedMem_.end() && (*it) < addr + reserveAlignedSize) {
            int32_t ret = aclrtUnmapMem(reinterpret_cast<void*>(*it));
            if (ret != 0) {
                SHM_LOG_ERROR("aclrtUnmapMem failed " << ret);
            }
            it++;
        }

        if (st != it) {
            mappedMem_.erase(st, it);
        }
    }
    return 0;
}

std::shared_ptr<MemSlice> AclMemSegmentDevice::GetMemSlice(hybm_mem_slice_t slice) const noexcept
{
    auto index = MemSlice::GetIndexFrom(slice);
    auto pos = slices_.find(index);
    if (pos == slices_.end()) {
        return nullptr;
    }

    auto target = pos->second.slice;
    if (!target->ValidateId(slice)) {
        return nullptr;
    }

    return target;
}

bool AclMemSegmentDevice::MemoryInRange(const void* begin, uint64_t size) const noexcept
{
    if (options_.userBufferHeapInput != nullptr) {
        return ResolveUserBufferHeapRange(begin, size, nullptr, nullptr);
    }
    if (begin < globalVirtualAddress_) {
        return false;
    }

    if (static_cast<const uint8_t*>(begin) + size >= globalVirtualAddress_ + totalVirtualSize_) {
        return false;
    }

    return true;
}

bool AclMemSegmentDevice::ResolveUserBufferHeapRange(
    const void* address, uint64_t length, uint32_t* rankId, uint32_t* segmentIndex) const noexcept
{
    if (address == nullptr || length == 0 || globalVirtualAddress_ == nullptr ||
        options_.userBufferHeapInput == nullptr || options_.size == 0) {
        return false;
    }
    const uint64_t windowBegin = reinterpret_cast<uint64_t>(globalVirtualAddress_);
    const uint64_t addr = reinterpret_cast<uint64_t>(address);
    if (addr < windowBegin) {
        return false;
    }
    const uint64_t alignedHeapSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    const uint64_t offset = addr - windowBegin;
    const uint64_t rank = offset / alignedHeapSize;
    const uint64_t inSlot = offset % alignedHeapSize;
    if (rank >= options_.rankCnt || inSlot >= options_.size || length > options_.size - inSlot) {
        return false;
    }
    const size_t wireCount = options_.userBufferHeapInput->entries.size() + 1;
    for (size_t segment = 0; segment < wireCount; ++segment) {
        const uint64_t segmentBegin = SegmentOffset(segment);
        const uint64_t segmentSize = SegmentSize(segment);
        if (inSlot >= segmentBegin && inSlot < segmentBegin + segmentSize) {
            if (length > segmentSize - (inSlot - segmentBegin)) {
                return false;
            }
            if (rankId != nullptr) {
                *rankId = static_cast<uint32_t>(rank);
            }
            if (segmentIndex != nullptr) {
                *segmentIndex = static_cast<uint32_t>(segment);
            }
            return true;
        }
    }
    return false;
}

void AclMemSegmentDevice::UnmapUserBufferTarget(uint64_t target, Result& firstError) noexcept
{
    auto recordError = [&](Result status) {
        if (status != ACLSHMEM_SUCCESS && firstError == ACLSHMEM_SUCCESS) {
            firstError = status;
        }
    };

    auto position = mappedMem_.find(target);
    if (position != mappedMem_.end()) {
        if (aclrtUnmapMem(reinterpret_cast<void*>(target)) != ACL_SUCCESS) {
            recordError(ACLSHMEM_DL_FUNC_FAILED);
        } else {
            mappedMem_.erase(position);
        }
    }
    auto imported = retainedImportedHandles_.find(target);
    if (imported != retainedImportedHandles_.end()) {
        if (imported->second != nullptr && aclrtFreePhysical(imported->second) != ACL_SUCCESS) {
            recordError(ACLSHMEM_DL_FUNC_FAILED);
        } else {
            retainedImportedHandles_.erase(imported);
        }
    }
}

void AclMemSegmentDevice::UnmapUserBufferRank(uint32_t rank, size_t wireCount, Result& firstError) noexcept
{
    if (rank >= reservedVirtualAddresses_.size()) {
        return;
    }
    for (size_t segment = wireCount; segment > 0; --segment) {
        UnmapUserBufferTarget(reservedVirtualAddresses_[rank] + SegmentOffset(segment - 1), firstError);
    }
}

void AclMemSegmentDevice::ReleaseUserBufferLocalHandle(Result& firstError) noexcept
{
    if (local_handle_ != nullptr) {
        if (aclrtFreePhysical(local_handle_) != ACL_SUCCESS) {
            if (firstError == ACLSHMEM_SUCCESS) {
                firstError = ACLSHMEM_DL_FUNC_FAILED;
            }
        } else {
            local_handle_ = nullptr;
        }
    }
}

void AclMemSegmentDevice::ReleaseUserBufferAddresses(Result& firstError) noexcept
{
    if (!reservedVirtualAddresses_.empty()) {
        if (socType_ == AscendSocType::ASCEND_950) {
            if (reservedVirtualAddresses_[0] != 0 &&
                aclrtReleaseMemAddress(reinterpret_cast<void*>(reservedVirtualAddresses_[0])) != ACL_SUCCESS) {
                if (firstError == ACLSHMEM_SUCCESS) {
                    firstError = ACLSHMEM_DL_FUNC_FAILED;
                }
            } else {
                reservedVirtualAddresses_.clear();
            }
        } else {
            bool allReleased = true;
            for (auto& reserved : reservedVirtualAddresses_) {
                if (reserved != 0 && aclrtReleaseMemAddress(reinterpret_cast<void*>(reserved)) != ACL_SUCCESS) {
                    if (firstError == ACLSHMEM_SUCCESS) {
                        firstError = ACLSHMEM_DL_FUNC_FAILED;
                    }
                    allReleased = false;
                } else {
                    reserved = 0;
                }
            }
            if (allReleased) {
                reservedVirtualAddresses_.clear();
            }
        }
    }
}

Result AclMemSegmentDevice::ReleaseUserBufferHeapResources() noexcept
{
    if (options_.userBufferHeapInput == nullptr) {
        return ACLSHMEM_INVALID_PARAM;
    }
    Result firstError = ACLSHMEM_SUCCESS;
    const size_t wireCount = options_.userBufferHeapInput->entries.size() + 1;
    for (uint32_t rank = options_.rankCnt; rank > 0; --rank) {
        const uint32_t current = rank - 1;
        if (current != options_.rankId) {
            UnmapUserBufferRank(current, wireCount, firstError);
        }
    }
    UnmapUserBufferRank(options_.rankId, wireCount, firstError);

    slices_.clear();
    allocatedSize_ = 0;
    sliceCount_ = 0;
    imports_.clear();
    importMap_.clear();
    ReleaseUserBufferLocalHandle(firstError);
    ReleaseUserBufferAddresses(firstError);
    globalVirtualAddress_ = nullptr;
    totalVirtualSize_ = 0;
    return firstError;
}

void AclMemSegmentDevice::FreeMemory() noexcept
{
    while (!slices_.empty()) {
        auto slice = slices_.begin()->second.slice;
        ReleaseSliceMemory(slice);
    }

    if (local_handle_ != nullptr) {
        auto ret = aclrtFreePhysical(local_handle_);
        if (ret != 0) {
            SHM_LOG_ERROR("aclrtFreePhysical failed. ret: " << ret);
            return;
        }
    }

    allocatedSize_ = 0;
    sliceCount_ = 0;
    if (globalVirtualAddress_ != nullptr) {
        if (socType_ == AscendSocType::ASCEND_950 && !reservedVirtualAddresses_.empty()) {
            aclrtReleaseMemAddress(reinterpret_cast<void*>(reservedVirtualAddresses_[0]));
        } else {
            for (auto reserved : reservedVirtualAddresses_) {
                aclrtReleaseMemAddress(reinterpret_cast<void*>(reserved));
            }
        }
        reservedVirtualAddresses_.clear();
        totalVirtualSize_ = 0;
        globalVirtualAddress_ = nullptr;
    }
}

bool AclMemSegmentDevice::CanMapRemote(const HbmExportInfo& rmi) noexcept
{
    return IsSdmaAccessible(rmi.superPodId, rmi.serverId, rmi.logicDeviceId);
}

void AclMemSegmentDevice::GetDeviceInfo(uint32_t& sdId, uint32_t& serverId, uint32_t& superPodId) noexcept
{
    sdId = sdid_;
    serverId = serverId_;
    superPodId = superPodId_;
}

bool AclMemSegmentDevice::GetRankIdByAddr(const void* addr, uint64_t size, uint32_t& rankId) const noexcept
{
    if (options_.userBufferHeapInput != nullptr) {
        return ResolveUserBufferHeapRange(addr, size, &rankId, nullptr);
    }
    return true;
}

bool AclMemSegmentDevice::CheckSdmaReaches(uint32_t remoteRankId) const noexcept
{
    auto pos = importMap_.find(static_cast<uint16_t>(remoteRankId));
    if (pos == importMap_.end()) {
        return false;
    }

    return IsSdmaAccessible(pos->second.superPodId, pos->second.serverId, pos->second.logicDeviceId);
}
} // namespace shm

#endif
