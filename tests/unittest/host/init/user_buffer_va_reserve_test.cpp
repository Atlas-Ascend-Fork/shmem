/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "hybm_device_mem_segment.h"
#include "hybm_mem_segment.h"
#include "hybm_user_buffer_heap.h"
#include "mem_entity_def.h"
#include "mem_entity_entry.h"
#include "unittest_main_test.h"

#if defined(HAS_ACLRT_MEM_FABRIC_HANDLE) && !defined(ACLSHMEM_SOC_950)

namespace shm {
namespace {

constexpr uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr uint32_t kRankCount = 8;
constexpr uint32_t kHbmHalfDivisor = 2;

class UserBufferVaReserveTest : public testing::Test {
protected:
    void SetUp() override
    {
        deviceId_ = test_first_npu;
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        aclInitialized_ = true;
        ASSERT_EQ(aclrtSetDevice(deviceId_), ACL_SUCCESS);
        deviceSet_ = true;

        size_t hbmFree = 0;
        size_t hbmTotal = 0;
        ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &hbmFree, &hbmTotal), ACL_SUCCESS);
        const size_t hbmSafeLimit = hbmTotal / kHbmHalfDivisor;
        if (hbmSafeLimit < HYBM_MAX_SINGLE_RESERVE_SIZE) {
            GTEST_SKIP() << "Skip large virtual-address reservation: total HBM=" << hbmTotal
                         << " bytes, 50% limit=" << hbmSafeLimit << " bytes, required=" << HYBM_MAX_SINGLE_RESERVE_SIZE
                         << " bytes.";
        }

        ASSERT_EQ(hybm_init(deviceId_, 0), ACLSHMEM_SUCCESS);
        hybmInitialized_ = true;
    }

    void TearDown() override
    {
        if (segment_ != nullptr) {
            EXPECT_EQ(segment_->UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
            segment_.reset();
        }
        if (hybmInitialized_) {
            hybm_uninit();
        }
        if (deviceSet_) {
            EXPECT_EQ(aclrtResetDevice(deviceId_), ACL_SUCCESS);
        }
        if (aclInitialized_) {
            EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
        }
    }

    void ReserveAndVerify(uint64_t heapSize, bool expectFixedBase)
    {
        UserBufferHeapInput input;
        MemSegmentOptions options{};
        options.devId = deviceId_;
        options.role = hybm_role_type::HYBM_ROLE_PEER;
        options.dataOpType = HYBM_DOP_TYPE_MTE;
        options.segType = HYBM_MST_HBM;
        options.infoExType = HYBM_INFO_EXG_IN_NODE;
        options.size = heapSize;
        options.rankId = 0;
        options.rankCnt = kRankCount;
        options.userBufferHeapInput = &input;

        segment_ = MemSegment::Create(options, 0);
        ASSERT_NE(segment_, nullptr);

        void* base = nullptr;
        const auto reserveStatus = segment_->ReserveMemorySpace(&base);
        EXPECT_EQ(reserveStatus, ACLSHMEM_SUCCESS);
        if (reserveStatus != ACLSHMEM_SUCCESS) {
            segment_.reset();
            return;
        }
        EXPECT_NE(base, nullptr);
        if (base == nullptr) {
            EXPECT_EQ(segment_->UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
            segment_.reset();
            return;
        }
        if (expectFixedBase) {
            EXPECT_EQ(reinterpret_cast<uint64_t>(base), HYBM_GVM_START_ADDR);
        }

        const uint64_t slotSize = ALIGN_UP(heapSize, DEVMM_HEAP_SIZE);
        for (uint32_t rank = 0; rank < kRankCount; ++rank) {
            uint32_t resolvedRank = kRankCount;
            const auto* slotBase = static_cast<const uint8_t*>(base) + rank * slotSize;
            EXPECT_TRUE(segment_->GetRankIdByAddr(slotBase, 1, resolvedRank));
            EXPECT_EQ(resolvedRank, rank);
        }

        EXPECT_EQ(segment_->UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
        segment_.reset();
    }

private:
    int32_t deviceId_{0};
    bool aclInitialized_{false};
    bool deviceSet_{false};
    bool hybmInitialized_{false};
    MemSegmentPtr segment_;
};

TEST_F(UserBufferVaReserveTest, ReserveBelow128GiBBoundary) { ReserveAndVerify(15 * kGiB, false); }

TEST_F(UserBufferVaReserveTest, ReserveAt128GiBBoundary) { ReserveAndVerify(16 * kGiB, true); }

TEST_F(UserBufferVaReserveTest, ReserveEightSlotsAbove128GiB)
{
    ReserveAndVerify(16 * kGiB + 6 * 1024ULL * 1024ULL, true);
}

TEST_F(UserBufferVaReserveTest, ReserveAfterRelease)
{
    for (uint32_t iteration = 0; iteration < 10; ++iteration) {
        ReserveAndVerify(16 * kGiB + 6 * 1024ULL * 1024ULL, true);
    }
}

} // namespace
} // namespace shm

#endif
