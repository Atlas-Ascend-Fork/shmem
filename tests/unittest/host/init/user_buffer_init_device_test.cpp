/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "host/shmem_host_def.h"
#include "unittest_main_test.h"

#if defined(HAS_ACLRT_MEM_FABRIC_HANDLE) && !defined(ACLSHMEM_SOC_950)

namespace {

class UserBufferInitDeviceTest : public testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        aclInitialized_ = true;
        ASSERT_EQ(aclrtSetDevice(test_first_npu), ACL_SUCCESS);
        deviceSet_ = true;
        ASSERT_EQ(aclrtGetDevice(&userDeviceId_), ACL_SUCCESS);
    }

    void TearDown() override
    {
        if (mapped_) {
            EXPECT_EQ(aclrtUnmapMem(address_), ACL_SUCCESS);
            mapped_ = false;
        }
        if (address_ != nullptr) {
            EXPECT_EQ(aclrtReleaseMemAddress(address_), ACL_SUCCESS);
            address_ = nullptr;
        }
        if (handle_ != nullptr) {
            EXPECT_EQ(aclrtFreePhysical(handle_), ACL_SUCCESS);
            handle_ = nullptr;
        }
        if (deviceSet_) {
            EXPECT_EQ(aclrtResetDevice(userDeviceId_), ACL_SUCCESS);
        }
        if (aclInitialized_) {
            EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
        }
    }

    int32_t userDeviceId_{-1};
    aclrtDrvMemHandle handle_{nullptr};
    void* address_{nullptr};
    bool mapped_{false};
    bool aclInitialized_{false};
    bool deviceSet_{false};
};

TEST_F(UserBufferInitDeviceTest, SetAccessUsesDriverLogicDevice)
{
    int32_t logicDeviceId = -1;
    ASSERT_EQ(aclrtGetLogicDevIdByUserDevId(userDeviceId_, &logicDeviceId), ACL_SUCCESS);
    ASSERT_GE(logicDeviceId, 0);

    aclrtPhysicalMemProp property{};
    property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    property.memAttr = ACL_HBM_MEM_HUGE;
    property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = userDeviceId_;

    size_t granularity = 0;
    ASSERT_EQ(
        aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity), ACL_SUCCESS);
    ASSERT_GT(granularity, 0U);
    ASSERT_EQ(aclrtMallocPhysical(&handle_, granularity, &property, 0), ACL_SUCCESS);
    ASSERT_NE(handle_, nullptr);
    ASSERT_EQ(aclrtReserveMemAddress(&address_, granularity, 0, nullptr, 0), ACL_SUCCESS);
    ASSERT_NE(address_, nullptr);
    ASSERT_EQ(aclrtMapMem(address_, granularity, 0, handle_, 0), ACL_SUCCESS);
    mapped_ = true;

    aclrtMemAccessDesc access{};
    access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = logicDeviceId;
    access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    const auto setAccessStatus = aclrtMemSetAccess(address_, granularity, &access, 1);
    if (setAccessStatus == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        GTEST_SKIP() << "aclrtMemSetAccess is not supported by the current runtime and driver";
    }
    ASSERT_EQ(setAccessStatus, ACL_SUCCESS);
    EXPECT_EQ(aclrtMemset(address_, granularity, 0, granularity), ACL_SUCCESS);
}

} // namespace

#endif
