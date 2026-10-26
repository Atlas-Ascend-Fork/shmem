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
#include <vector>

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

constexpr uint32_t kHbmLimitNumerator = 3;
constexpr uint32_t kHbmLimitDenominator = 4;

class UserBufferVaReserveTest : public testing::Test {
protected:
    void SetUp() override
    {
        deviceId_ = test_first_npu;
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        aclInitialized_ = true;

        ASSERT_GT(test_gnpu_num, 0);
        rankCount_ = static_cast<uint32_t>(test_gnpu_num);
        for (int index = 0; index < test_gnpu_num; ++index) {
            const int32_t deviceId = test_first_npu + index;
            ASSERT_EQ(aclrtSetDevice(deviceId), ACL_SUCCESS);
            size_t hbmFree = 0;
            size_t hbmTotal = 0;
            const auto queryStatus = aclrtGetMemInfo(ACL_HBM_MEM, &hbmFree, &hbmTotal);
            const auto resetStatus = aclrtResetDevice(deviceId);
            ASSERT_EQ(queryStatus, ACL_SUCCESS);
            ASSERT_EQ(resetStatus, ACL_SUCCESS);
            const size_t hbmSafeLimit = hbmTotal / kHbmLimitDenominator * kHbmLimitNumerator +
                                        hbmTotal % kHbmLimitDenominator * kHbmLimitNumerator / kHbmLimitDenominator;
            hbmSafeLimits_.push_back(hbmSafeLimit);
        }
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
        const uint64_t slotSize = ALIGN_UP(heapSize, DEVMM_HEAP_SIZE);
        for (uint32_t index = 0; index < rankCount_; ++index) {
            if (slotSize > hbmSafeLimits_[index]) {
                GTEST_SKIP() << "Skip large virtual-address reservation: device=" << test_first_npu + index
                             << ", 75% HBM limit=" << hbmSafeLimits_[index]
                             << " bytes, required per rank slot=" << slotSize << " bytes, rank slots=" << rankCount_
                             << ", total VA=" << slotSize * rankCount_ << " bytes.";
            }
        }
        if (!hybmInitialized_) {
            // hybm_init selects the device; TearDown releases that device reference.
            ASSERT_EQ(hybm_init(deviceId_, 0), ACLSHMEM_SUCCESS);
            deviceSet_ = true;
            hybmInitialized_ = true;
        }
        UserBufferHeapInput input;
        MemSegmentOptions options{};
        options.devId = deviceId_;
        options.role = hybm_role_type::HYBM_ROLE_PEER;
        options.dataOpType = HYBM_DOP_TYPE_MTE;
        options.segType = HYBM_MST_HBM;
        options.infoExType = HYBM_INFO_EXG_IN_NODE;
        options.size = heapSize;
        options.rankId = 0;
        options.rankCnt = rankCount_;
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

        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            uint32_t resolvedRank = rankCount_;
            const auto* slotBase = static_cast<const uint8_t*>(base) + rank * slotSize;
            EXPECT_TRUE(segment_->GetRankIdByAddr(slotBase, 1, resolvedRank));
            EXPECT_EQ(resolvedRank, rank);
        }

        EXPECT_EQ(segment_->UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
        segment_.reset();
    }

    uint64_t BoundarySlotUnits() const { return HYBM_MAX_SINGLE_RESERVE_SIZE / DEVMM_HEAP_SIZE / rankCount_; }

    uint32_t rankCount_{0};

private:
    std::vector<size_t> hbmSafeLimits_;
    int32_t deviceId_{0};
    bool aclInitialized_{false};
    bool deviceSet_{false};
    bool hybmInitialized_{false};
    MemSegmentPtr segment_;
};

TEST_F(UserBufferVaReserveTest, ReserveBelow128GiBBoundary)
{
    const uint64_t slotUnits = (HYBM_MAX_SINGLE_RESERVE_SIZE / DEVMM_HEAP_SIZE - 1) / rankCount_;
    if (slotUnits == 0) {
        GTEST_SKIP() << "No positive aligned slot size produces total VA below 128 GiB for " << rankCount_ << " ranks.";
    }
    ASSERT_LT(slotUnits * DEVMM_HEAP_SIZE * rankCount_, HYBM_MAX_SINGLE_RESERVE_SIZE);
    ReserveAndVerify(slotUnits * DEVMM_HEAP_SIZE, false);
}

TEST_F(UserBufferVaReserveTest, ReserveAt128GiBBoundary)
{
    if (HYBM_MAX_SINGLE_RESERVE_SIZE % (DEVMM_HEAP_SIZE * rankCount_) != 0) {
        GTEST_SKIP() << "Equal aligned rank slots cannot produce exactly 128 GiB for " << rankCount_ << " ranks.";
    }
    ASSERT_EQ(BoundarySlotUnits() * DEVMM_HEAP_SIZE * rankCount_, HYBM_MAX_SINGLE_RESERVE_SIZE);
    ReserveAndVerify(BoundarySlotUnits() * DEVMM_HEAP_SIZE, true);
}

TEST_F(UserBufferVaReserveTest, ReserveAbove128GiBBoundary)
{
    const uint64_t slotSize = (BoundarySlotUnits() + 1) * DEVMM_HEAP_SIZE;
    ASSERT_GT(slotSize * rankCount_, HYBM_MAX_SINGLE_RESERVE_SIZE);
    ReserveAndVerify(slotSize, true);
}

TEST_F(UserBufferVaReserveTest, ReserveAfterRelease)
{
    for (uint32_t iteration = 0; iteration < 10; ++iteration) {
        ReserveAndVerify((BoundarySlotUnits() + 1) * DEVMM_HEAP_SIZE, true);
        if (IsSkipped() || HasFatalFailure() || HasFailure()) {
            return;
        }
    }
}

} // namespace
} // namespace shm

#endif
