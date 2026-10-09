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

#include <string>

#include "utils/under_api/dl_comm_def.h"

TEST(DriverVersionTest, ParsesSingleAndMultiCharacterFields)
{
    const std::string version = "V100R001C23SPC005B219";

    EXPECT_EQ(GetValueFromVersion(version, "V"), 100);
    EXPECT_EQ(GetValueFromVersion(version, "R"), 1);
    EXPECT_EQ(GetValueFromVersion(version, "C"), 23);
    EXPECT_EQ(GetValueFromVersion(version, "SPC"), 5);
    EXPECT_EQ(GetValueFromVersion(version, "B"), 219);
}

TEST(DriverVersionTest, ComparesSpcBeforeBuild)
{
    const std::string required = "V100R001C23SPC005B219";

    EXPECT_FALSE(DriverVersionCheck(required, "V100R001C23SPC004B999"));
    EXPECT_TRUE(DriverVersionCheck(required, "V100R001C23SPC005B219"));
    EXPECT_TRUE(DriverVersionCheck(required, "V100R001C23SPC006B001"));
}
