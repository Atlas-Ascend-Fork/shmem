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

#include "device_udma_transport_manager.h"
#include "host/shmem_host_def.h"

namespace shm {
namespace transport {
namespace device {

TEST(UdmaTransportManagerTest, RejectsInvalidQpCountBeforeOpeningDevice)
{
    UdmaTransportManager manager;
    TransportOptions options{};

    options.udmaQpConfig.qpNum = 0;
    EXPECT_EQ(manager.OpenDevice(options), ACLSHMEM_INVALID_VALUE);

    options.udmaQpConfig.qpNum = ACLSHMEM_MAX_QP_NUM + 1;
    EXPECT_EQ(manager.OpenDevice(options), ACLSHMEM_INVALID_VALUE);
}

#if defined(ACLSHMEM_RELAY_SUPPORT)
TEST(UdmaTransportManagerTest, RejectsMultipleQpsInRelayModeBeforeOpeningDevice)
{
    UdmaTransportManager manager;
    TransportOptions options{};
    options.udmaQpConfig.qpNum = 2;

    EXPECT_EQ(manager.OpenDevice(options), ACLSHMEM_NOT_SUPPORTED);
}
#endif

TEST(UdmaTransportManagerTest, RejectsInvalidPeerMasksBeforeOpeningDevice)
{
    UdmaTransportManager manager;
    TransportOptions options{};
    options.rankCount = 3;
    options.rankId = 0;
    options.udmaQpConfig.qpNum = 1;
    options.udmaQpConfig.peerQpMasks = {0, 1, 1};
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        EXPECT_EQ(manager.OpenDevice(options), ACLSHMEM_NOT_SUPPORTED);
        return;
    }
    options.udmaQpConfig.qpNum = 4;
    for (const auto& masks : std::vector<std::vector<uint64_t>>{{0, 1}, {1, 1, 1}, {0, 2, 1}, {0, 17, 1}, {0, 0, 1}}) {
        options.udmaQpConfig.peerQpMasks = masks;
        EXPECT_EQ(manager.OpenDevice(options), ACLSHMEM_INVALID_VALUE);
    }
    options.udmaQpConfig.peerQpMasks = {0, 1, 1};
    options.rankId = 3;
    EXPECT_EQ(manager.OpenDevice(options), ACLSHMEM_INVALID_VALUE);
    options.rankCount = 0;
    EXPECT_EQ(manager.OpenDevice(options), ACLSHMEM_INVALID_VALUE);
}

} // namespace device
} // namespace transport
} // namespace shm
