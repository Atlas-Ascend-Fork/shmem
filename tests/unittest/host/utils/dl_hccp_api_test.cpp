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

#include "utils/under_api/dl_hccp_api.h"

TEST(HccpSocketLayoutTest, CannFallbackUsesLegacyLayoutFor920BetaReleases)
{
    for (const char* version : {"9.2.0-beta.1", "9.2.0-beta.2", "9.2.0.beta.1", "9.2.0.beta.2"}) {
        EXPECT_EQ(shm::DlHccpApi::SelectSocketConnectInfoStride("", version), sizeof(shm::HccpSocketConnectInfo))
            << version;
    }
}

TEST(HccpSocketLayoutTest, CannFallbackUsesNewLayoutFrom920OfficialRelease)
{
    for (const char* version : {"9.2.0", "9.2.1", "9.3.0", "9.10.0", "10.0.0"}) {
        EXPECT_EQ(shm::DlHccpApi::SelectSocketConnectInfoStride("", version), sizeof(shm::HccpSocketConnectInfoV2))
            << version;
    }
}

TEST(HccpSocketLayoutTest, FallsBackToLegacyLayoutForOlderOrMissingVersions)
{
    for (const char* version : {"8.2.T101", "8.3.T101", "8.5.0", "9.1.0", "9.2.T101", "9.2.T530", "", "invalid"}) {
        EXPECT_EQ(shm::DlHccpApi::SelectSocketConnectInfoStride("", version), sizeof(shm::HccpSocketConnectInfo))
            << version;
    }
}

TEST(HccpSocketLayoutTest, HcommVersionTakesPrecedenceOverCannVersion)
{
    for (const char* version : {"8.5.1", "9.0.1", "9.1.0"}) {
        EXPECT_EQ(shm::DlHccpApi::SelectSocketConnectInfoStride(version, "9.3.0"), sizeof(shm::HccpSocketConnectInfo))
            << version;
    }
    for (const char* version : {"9.3.0", "9.3.1", "9.10.0", "10.0.0"}) {
        EXPECT_EQ(
            shm::DlHccpApi::SelectSocketConnectInfoStride(version, "9.2.T530"), sizeof(shm::HccpSocketConnectInfoV2))
            << version;
        EXPECT_EQ(shm::DlHccpApi::SelectSocketConnectInfoStride(version, ""), sizeof(shm::HccpSocketConnectInfoV2))
            << version;
    }
}

TEST(HccpSocketLayoutTest, Hcomm92UsesCannReleaseType)
{
    for (const char* hcommVersion : {"9.2.0", "9.2.1", "9.2.99"}) {
        for (const char* cannVersion :
             {"9.2.0-beta.1", "9.2.0-beta.2", "9.2.0.beta.1", "9.2.0.beta.2", "9.2.1-beta.1", "9.2.T101", "9.2.T530"}) {
            EXPECT_EQ(
                shm::DlHccpApi::SelectSocketConnectInfoStride(hcommVersion, cannVersion),
                sizeof(shm::HccpSocketConnectInfo))
                << hcommVersion << ", " << cannVersion;
        }
        for (const char* cannVersion : {"9.2.0", "9.2.1", "9.3.0", "", "invalid"}) {
            EXPECT_EQ(
                shm::DlHccpApi::SelectSocketConnectInfoStride(hcommVersion, cannVersion),
                sizeof(shm::HccpSocketConnectInfoV2))
                << hcommVersion << ", " << cannVersion;
        }
    }
}

TEST(HccpSocketLayoutTest, InvalidHcommVersionFallsBackToCannVersion)
{
    for (const char* version :
         {"", "invalid", "9.3", "9.3.T101", "9.3.0garbage", "-1.3.0", "9.3.0.1", "999999999999999999999.3.0"}) {
        EXPECT_EQ(
            shm::DlHccpApi::SelectSocketConnectInfoStride(version, "9.2.T530"), sizeof(shm::HccpSocketConnectInfo))
            << version;
        EXPECT_EQ(shm::DlHccpApi::SelectSocketConnectInfoStride(version, "9.3.0"), sizeof(shm::HccpSocketConnectInfoV2))
            << version;
    }
}
