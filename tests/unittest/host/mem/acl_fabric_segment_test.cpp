/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 */

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "dl_comm_def.h"
#include "dl_api.h"
#include "host/shmem_host_def.h"
#include "mem_entity_entry.h"
#include "mem_entity_def.h"
#include "mem/heap/hybm_device_mem_segment.h"
#include "unittest_main_test.h"

#if defined(HAS_ACLRT_MEM_FABRIC_HANDLE) && defined(USE_ACLRT_MEM_FABRIC_HANDLE)

namespace {

class AclFabricSegmentTest : public testing::Test {
protected:
    void SetUp() override
    {
        const auto initRet = aclInit(nullptr);
        if (initRet != ACL_SUCCESS) {
            GTEST_SKIP() << "aclInit unavailable: " << initRet;
        }
        aclInitialized_ = true;

        const auto setDeviceRet = aclrtSetDevice(test_first_npu);
        if (setDeviceRet != ACL_SUCCESS) {
            GTEST_SKIP() << "aclrtSetDevice unavailable: " << setDeviceRet;
        }
        deviceSet_ = true;

        const auto precheckRet = HalGvaPrecheck();
        if (precheckRet != ACLSHMEM_SUCCESS) {
            GTEST_SKIP() << "GVA driver precheck unavailable: " << precheckRet;
        }
        const auto loadRet = shm::DlApi::LoadLibrary();
        if (loadRet != ACLSHMEM_SUCCESS) {
            GTEST_SKIP() << "SHMEM dynamic API loading unavailable: " << loadRet;
        }
        const auto hybmInitRet = hybm_init(test_first_npu, 0);
        if (hybmInitRet != ACLSHMEM_SUCCESS) {
            GTEST_SKIP() << "SHMEM device-memory initialization unavailable: " << hybmInitRet;
        }
        hybmInitialized_ = true;
    }

    void TearDown() override
    {
        if (importedSegment_ != nullptr) {
            EXPECT_EQ(importedSegment_->Unmap(), ACLSHMEM_SUCCESS);
            EXPECT_EQ(importedSegment_->UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
            importedSegment_.reset();
        }
        if (segment_ != nullptr) {
            EXPECT_EQ(segment_->UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
            segment_.reset();
        }
        if (hybmInitialized_) {
            hybm_uninit();
        }
        if (deviceSet_) {
            EXPECT_EQ(aclrtResetDevice(test_first_npu), ACL_SUCCESS);
        }
        if (aclInitialized_) {
            EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
        }
    }

    shm::MemSegmentPtr segment_;
    shm::MemSegmentPtr importedSegment_;
    bool aclInitialized_{false};
    bool deviceSet_{false};
    bool hybmInitialized_{false};
};

TEST_F(AclFabricSegmentTest, AllocatesContiguousSlicesWithIndependentMappings)
{
    constexpr uint64_t sliceSize = shm::DEVICE_LARGE_PAGE_SIZE;
    shm::MemSegmentOptions options{};
    options.devId = test_first_npu;
    options.segType = shm::HYBM_MST_HBM;
    options.shared = false;
    options.size = 2 * sliceSize;
    options.rankId = 0;
    options.rankCnt = 1;

    segment_ = shm::MemSegment::Create(options, 0);
    ASSERT_NE(segment_, nullptr);

    void* base = nullptr;
    ASSERT_EQ(segment_->ReserveMemorySpace(&base), ACLSHMEM_SUCCESS);

    std::shared_ptr<shm::MemSlice> first;
    std::shared_ptr<shm::MemSlice> second;
    ASSERT_EQ(segment_->AllocLocalMemory(sliceSize, first), ACLSHMEM_SUCCESS);
    ASSERT_EQ(segment_->AllocLocalMemory(sliceSize, second), ACLSHMEM_SUCCESS);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->vAddress_, reinterpret_cast<uint64_t>(base));
    EXPECT_EQ(second->vAddress_, first->vAddress_ + sliceSize);

    ASSERT_EQ(aclrtMemset(reinterpret_cast<void*>(first->vAddress_), sliceSize, 0xA5, 1), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(reinterpret_cast<void*>(second->vAddress_), sliceSize, 0x5A, 1), ACL_SUCCESS);

    uint8_t values[2]{};
    ASSERT_EQ(
        aclrtMemcpy(values, sizeof(values), reinterpret_cast<void*>(first->vAddress_), 1, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(
            values + 1, sizeof(values) - 1, reinterpret_cast<void*>(second->vAddress_), 1, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(values[0], 0xA5);
    EXPECT_EQ(values[1], 0x5A);

    EXPECT_EQ(segment_->ReleaseSliceMemory(first), ACLSHMEM_SUCCESS);
    EXPECT_EQ(segment_->ReleaseSliceMemory(second), ACLSHMEM_SUCCESS);
}

TEST_F(AclFabricSegmentTest, MapsSharedSlicesAtTheirExportedOffsets)
{
    constexpr uint64_t sliceSize = shm::DEVICE_LARGE_PAGE_SIZE;
    shm::MemSegmentOptions options{};
    options.devId = test_first_npu;
    options.segType = shm::HYBM_MST_HBM;
    options.shared = true;
    options.size = 2 * sliceSize;
    options.rankId = 0;
    options.rankCnt = 2;

    segment_ = shm::MemSegment::Create(options, 0);
    ASSERT_NE(segment_, nullptr);

    void* localBase = nullptr;
    ASSERT_EQ(segment_->ReserveMemorySpace(&localBase), ACLSHMEM_SUCCESS);

    std::shared_ptr<shm::MemSlice> first;
    std::shared_ptr<shm::MemSlice> second;
    ASSERT_EQ(segment_->AllocLocalMemory(sliceSize, first), ACLSHMEM_SUCCESS);
    ASSERT_EQ(segment_->AllocLocalMemory(sliceSize, second), ACLSHMEM_SUCCESS);
    EXPECT_EQ(first->vAddress_, reinterpret_cast<uint64_t>(localBase));
    EXPECT_EQ(second->vAddress_, first->vAddress_ + sliceSize);
    ASSERT_EQ(aclrtMemset(reinterpret_cast<void*>(first->vAddress_), sliceSize, 0xA5, 1), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(reinterpret_cast<void*>(second->vAddress_), sliceSize, 0x5A, 1), ACL_SUCCESS);

    std::string firstInfo;
    std::string secondInfo;
    const auto firstExportRet = segment_->Export(first, firstInfo);
    if (firstExportRet != ACLSHMEM_SUCCESS) {
        GTEST_SKIP() << "ACL Fabric share-handle export unavailable: " << firstExportRet;
    }
    ASSERT_EQ(segment_->Export(second, secondInfo), ACLSHMEM_SUCCESS);

    options.rankId = 1;
    importedSegment_ = shm::MemSegment::Create(options, 0);
    ASSERT_NE(importedSegment_, nullptr);
    void* remoteBase = nullptr;
    ASSERT_EQ(importedSegment_->ReserveMemorySpace(&remoteBase), ACLSHMEM_SUCCESS);
    ASSERT_EQ(importedSegment_->Import({firstInfo, secondInfo}, nullptr), ACLSHMEM_SUCCESS);
    ASSERT_EQ(importedSegment_->Mmap(), ACLSHMEM_SUCCESS);

    uint8_t values[2]{};
    ASSERT_EQ(aclrtMemcpy(values, sizeof(values), remoteBase, 1, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(
            values + 1, sizeof(values) - 1, static_cast<uint8_t*>(remoteBase) + sliceSize, 1,
            ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(values[0], 0xA5);
    EXPECT_EQ(values[1], 0x5A);
}

} // namespace

#endif
