/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for details.
 */
#if defined(ACLSHMEM_UT_ASCEND950)
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#define private public
#include "transport/topo/rootinfo/aclshmemi_hal.h"
#undef private

namespace shm::topo {
namespace {
int returned_eid_count = 0;

class RootInfoHalTest : public testing::Test {
protected:
    aclshmemi_dcmi_api_t& api = aclshmemi_dcmi_api_t::instance();
    const bool initialized = api.is_initialized_;
    const getLogicIdFromPhyIdFunc logic_id = DlDcmiApi::pGetLogicIdFromPhyId;
    const dcmiv2GetUrmaDeviceCntFunc device_count = DlDcmiApi::pDcmiv2GetUrmaDeviceCnt;
    const dcmiv2GetEidListByUrmaDevIndexFunc eid_list = DlDcmiApi::pDcmiv2GetEidListByUrmaDevIndex;

    void SetUp() override
    {
        api.is_initialized_ = true;
        DlDcmiApi::pGetLogicIdFromPhyId = [](unsigned int, unsigned int* id) {
            *id = 0;
            return 0;
        };
        DlDcmiApi::pDcmiv2GetUrmaDeviceCnt = [](int, unsigned int* count) {
            *count = 1;
            return 0;
        };
        DlDcmiApi::pDcmiv2GetEidListByUrmaDevIndex = [](int, int, dcmi_urma_eid_info_t*, int* count) {
            EXPECT_EQ(*count, MAX_EID_PER_UE);
            *count = returned_eid_count;
            return 0;
        };
    }
    void TearDown() override
    {
        api.is_initialized_ = initialized;
        DlDcmiApi::pGetLogicIdFromPhyId = logic_id;
        DlDcmiApi::pDcmiv2GetUrmaDeviceCnt = device_count;
        DlDcmiApi::pDcmiv2GetEidListByUrmaDevIndex = eid_list;
    }
};

TEST_F(RootInfoHalTest, ValidatesDriverEidCountBeforePublishing)
{
    for (int count : {-1, 0, MAX_EID_PER_UE, MAX_EID_PER_UE + 1}) {
        SCOPED_TRACE(count);
        returned_eid_count = count;
        UEList list{};
        const bool valid = count >= 0 && count <= MAX_EID_PER_UE;
        EXPECT_EQ(api.get_ue_list(0, &list), valid ? 0 : -1);
        EXPECT_EQ(list.ueList[0].eidNum, valid ? static_cast<uint32_t>(count) : 0U);
    }
}
} // namespace
} // namespace shm::topo

#endif
