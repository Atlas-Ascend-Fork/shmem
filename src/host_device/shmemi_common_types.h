/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEMI_COMMON_TYPES
#define SHMEMI_COMMON_TYPES

#include <cstdint>
#include <cstddef>

// This state is appended to the device extra context, after the public
// aclshmem_device_host_state_t payload. It must not change that public ABI.
// The per-rank mask contains non-MTE backends whose asynchronous operations
// may require quiet completion. MTE is handled by PipeBarrier<PIPE_ALL>().
// Typical values: MTE only = 0x00; MTE + UDMA = 0x08; MTE + SDMA + RoCE = 0x06.
typedef struct {
    std::uint8_t quiet_transport_mask;
} aclshmemi_device_quiet_state_t;

// Private per-instance configuration, appended to the public state in extra context.
// Keep quiet at offset zero so existing completion-mask readers keep the same address.
typedef struct {
    aclshmemi_device_quiet_state_t quiet;
    std::uint32_t sync_timeout_us;
} aclshmemi_device_runtime_config_t;

static_assert(offsetof(aclshmemi_device_runtime_config_t, quiet) == 0, "Preserve the quiet trailer prefix");
static_assert(offsetof(aclshmemi_device_runtime_config_t, sync_timeout_us) == 4, "Align the private timeout field");

#endif
