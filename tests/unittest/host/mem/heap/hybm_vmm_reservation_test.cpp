/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "dl_acl_api.h"
#include "hybm_vmm_based_segment.h"
#include "mem_entity_def.h"

namespace shm {
namespace {
constexpr uint64_t gib = 1UL << 30;
constexpr uint64_t tib = 1UL << 40;

class VmmSocAccess : public HybmVmmBasedSegment {
public:
    static AscendSocType GetSoc() { return socType_; }
    static void SetSoc(AscendSocType type) { socType_ = type; }
};
} // namespace

// All runtime allocation entry points are replaced by host recorders. These tests
// never initialize a device, reserve real device VA or create communication channels.
class HybmVmmReservationTest : public testing::Test {
protected:
    void SetUp() override
    {
        savedReserve_ = DlAclApi::pAclrtReserveMemAddress;
        savedRelease_ = DlAclApi::pAclrtReleaseMemAddress;
        savedSoc_ = VmmSocAccess::GetSoc();
        DlAclApi::pAclrtReserveMemAddress = Reserve;
        DlAclApi::pAclrtReleaseMemAddress = Release;
        VmmSocAccess::SetSoc(AscendSocType::ASCEND_950);
        sizes_.clear();
        hints_.clear();
        releases_.clear();
        base_ = HYBM_DEVICE_VA_START;
        fail_ = false;
    }

    void TearDown() override
    {
        DlAclApi::pAclrtReserveMemAddress = savedReserve_;
        DlAclApi::pAclrtReleaseMemAddress = savedRelease_;
        VmmSocAccess::SetSoc(savedSoc_);
    }

    static MemSegmentOptions Options(uint64_t size, uint32_t ranks)
    {
        MemSegmentOptions options;
        options.size = size;
        options.rankCnt = ranks;
        return options;
    }

    static int Reserve(void** address, size_t size, size_t alignment, void* hint, uint64_t flags)
    {
        EXPECT_EQ(alignment, 0U);
        EXPECT_EQ(flags, 1U);
        sizes_.push_back(size);
        hints_.push_back(hint);
        *address = fail_ ? nullptr : (hint == nullptr ? reinterpret_cast<void*>(base_) : hint);
        return fail_ ? ACLSHMEM_MALLOC_FAILED : ACLSHMEM_SUCCESS;
    }

    static int Release(void* address)
    {
        releases_.push_back(address);
        return ACLSHMEM_SUCCESS;
    }

    static inline std::vector<size_t> sizes_;
    static inline std::vector<void*> hints_;
    static inline std::vector<void*> releases_;
    static inline uint64_t base_;
    static inline bool fail_;
    aclrtReserveMemAddressFunc savedReserve_{};
    aclrtReleaseMemAddressFunc savedRelease_{};
    AscendSocType savedSoc_{};
};

TEST_F(HybmVmmReservationTest, Ep256SixTiBUsesRuntimeBaseAndReleasesOnce)
{
    // Actual EP256/T16384 heap request: 23.765625 GiB rounds to 24 GiB per rank.
    HybmVmmBasedSegment segment(Options(25518145536UL, 256), 0);
    void* address = nullptr;
    ASSERT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_SUCCESS);
    ASSERT_EQ(sizes_, std::vector<size_t>{6 * tib});
    ASSERT_EQ(hints_, std::vector<void*>{nullptr});
    EXPECT_EQ(address, reinterpret_cast<void*>(base_));
    EXPECT_TRUE(segment.MemoryInRange(address, 6 * tib));
    EXPECT_TRUE(segment.MemoryInRange(reinterpret_cast<void*>(base_ + 255 * 24 * gib), 24 * gib));
    EXPECT_FALSE(segment.MemoryInRange(reinterpret_cast<void*>(base_ + 6 * tib - 1), 2));
    EXPECT_FALSE(segment.MemoryInRange(reinterpret_cast<void*>(base_ - 1), 1));
    EXPECT_FALSE(segment.MemoryInRange(address, UINT64_MAX));
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(releases_, std::vector<void*>{address});
    EXPECT_FALSE(segment.MemoryInRange(address, 1));
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(releases_.size(), 1U);
    ASSERT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_SUCCESS);
    EXPECT_TRUE(segment.MemoryInRange(address, 6 * tib));
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(releases_.size(), 2U);
}

TEST_F(HybmVmmReservationTest, Ep128AlignedHeapRemainsValid)
{
    HybmVmmBasedSegment segment(Options(14415822848UL, 128), 0);
    void* address = nullptr;
    ASSERT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_SUCCESS);
    EXPECT_EQ(sizes_, std::vector<size_t>{14 * gib * 128});
    EXPECT_TRUE(segment.MemoryInRange(address, 14 * gib * 128));
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
}

TEST_F(HybmVmmReservationTest, SmallHeapOnOlderSocUsesAutomaticReservation)
{
    VmmSocAccess::SetSoc(AscendSocType::ASCEND_910C);
    HybmVmmBasedSegment segment(Options(DEVICE_LARGE_PAGE_SIZE, 8), 0);
    void* address = nullptr;
    ASSERT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_SUCCESS);
    EXPECT_EQ(sizes_, std::vector<size_t>{8 * gib});
    EXPECT_EQ(hints_, std::vector<void*>{nullptr});
    EXPECT_TRUE(segment.MemoryInRange(address, 8 * gib));
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(releases_.size(), 1U);
}

TEST_F(HybmVmmReservationTest, OlderSocKeepsFixedAddressLimit)
{
    VmmSocAccess::SetSoc(AscendSocType::ASCEND_910C);
    HybmVmmBasedSegment segment(Options(24 * gib, 256), 0);
    void* address = nullptr;
    EXPECT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_MALLOC_FAILED);
    EXPECT_TRUE(sizes_.empty());
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_TRUE(releases_.empty());
}

TEST_F(HybmVmmReservationTest, OlderSocFixedReservationReleasesEachRank)
{
    VmmSocAccess::SetSoc(AscendSocType::ASCEND_910C);
    HybmVmmBasedSegment segment(Options(64 * gib, 2), 0);
    void* address = nullptr;
    ASSERT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_SUCCESS);
    ASSERT_EQ(hints_.size(), 2U);
    EXPECT_EQ(hints_[0], reinterpret_cast<void*>(HYBM_GVM_START_ADDR));
    EXPECT_EQ(hints_[1], reinterpret_cast<void*>(HYBM_GVM_START_ADDR + 64 * gib));
    EXPECT_TRUE(segment.MemoryInRange(address, 128 * gib));
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(releases_, hints_);
}

TEST_F(HybmVmmReservationTest, RuntimeFailureLeavesNoReservationAndCanRetry)
{
    HybmVmmBasedSegment segment(Options(24 * gib, 256), 0);
    void* address = nullptr;
    fail_ = true;
    EXPECT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_MALLOC_FAILED);
    EXPECT_FALSE(segment.MemoryInRange(nullptr, 0));
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_TRUE(releases_.empty());
    fail_ = false;
    ASSERT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_SUCCESS);
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(releases_.size(), 1U);
}

TEST_F(HybmVmmReservationTest, AlignmentOverflowRejectedBeforeRuntime)
{
    HybmVmmBasedSegment segment(Options(UINT64_MAX - DEVICE_LARGE_PAGE_SIZE + 1, 1), 0);
    void* address = nullptr;
    EXPECT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_INVALID_PARAM);
    EXPECT_TRUE(sizes_.empty());
}

TEST_F(HybmVmmReservationTest, AlignedTotalOverflowRejectedBeforeRuntime)
{
    // Raw size times rank count fits, but the 1 GiB aligned total does not.
    HybmVmmBasedSegment segment(Options((1UL << 63) - DEVICE_LARGE_PAGE_SIZE, 2), 0);
    void* address = nullptr;
    EXPECT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_INVALID_PARAM);
    EXPECT_TRUE(sizes_.empty());
}

TEST_F(HybmVmmReservationTest, ReturnedAddressOverflowReleasesReservation)
{
    base_ = UINT64_MAX - gib + 1;
    HybmVmmBasedSegment segment(Options(gib, 2), 0);
    void* address = nullptr;
    EXPECT_EQ(segment.ReserveMemorySpace(&address), ACLSHMEM_MALLOC_FAILED);
    EXPECT_EQ(releases_, std::vector<void*>{reinterpret_cast<void*>(base_)});
    EXPECT_EQ(segment.UnReserveMemorySpace(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(releases_.size(), 1U);
}
} // namespace shm
