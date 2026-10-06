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
#include <cstdint>
#include "host/init/shmem_host_init.h"
#include "host/shmem_host_def.h"

class UdmaPeerQpMasksTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(aclshmemx_set_udma_peer_qp_masks(nullptr, 0), ACLSHMEM_SUCCESS); }
    void TearDown() override
    {
        EXPECT_EQ(aclshmemx_set_udma_peer_qp_masks(nullptr, 0), ACLSHMEM_SUCCESS);
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 1), ACLSHMEM_SUCCESS);
    }
};

TEST_F(UdmaPeerQpMasksTest, CopiesInputRejectsInvalidBitsAndProtectsQpCount)
{
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 32), ACLSHMEM_SUCCESS);
    uint64_t masks[] = {0, uint64_t{1} | (uint64_t{1} << 31)};
    ASSERT_EQ(aclshmemx_set_udma_peer_qp_masks(masks, 2), ACLSHMEM_SUCCESS);
    masks[1] = 1; // Caller storage is no longer owned by SHMEM.
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 16), ACLSHMEM_INVALID_VALUE);
    masks[1] = uint64_t{1} << 32;
    EXPECT_EQ(aclshmemx_set_udma_peer_qp_masks(masks, 2), ACLSHMEM_INVALID_VALUE);
    // Rejection must preserve the previous valid mask and QP count.
    uint32_t qps = 0;
    ASSERT_EQ(aclshmemx_get_qp_num(ACLSHMEM_DATA_OP_UDMA, &qps), ACLSHMEM_SUCCESS);
    EXPECT_EQ(qps, 32U);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 16), ACLSHMEM_INVALID_VALUE);
    ASSERT_EQ(aclshmemx_set_udma_peer_qp_masks(nullptr, 0), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 16), ACLSHMEM_SUCCESS);
}

TEST_F(UdmaPeerQpMasksTest, ValidatesPointerCountAndHighestSupportedBit)
{
    uint64_t masks[] = {0, 1};
    EXPECT_EQ(aclshmemx_set_udma_peer_qp_masks(nullptr, 1), ACLSHMEM_INVALID_VALUE);
    EXPECT_EQ(aclshmemx_set_udma_peer_qp_masks(masks, 0), ACLSHMEM_INVALID_VALUE);
    EXPECT_EQ(aclshmemx_set_udma_peer_qp_masks(masks, ACLSHMEM_MAX_PES + 1), ACLSHMEM_INVALID_VALUE);
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, ACLSHMEM_MAX_QP_NUM), ACLSHMEM_SUCCESS);
    masks[1] |= uint64_t{1} << (ACLSHMEM_MAX_QP_NUM - 1);
    EXPECT_EQ(aclshmemx_set_udma_peer_qp_masks(masks, 2), ACLSHMEM_SUCCESS);
}
