/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHEMEI_DEVICE_MO_H
#define ACLSHEMEI_DEVICE_MO_H

#include "shmemi_device_common.h"

ACLSHMEM_DEVICE uint8_t aclshmemi_filter_unsupported_quiet_transports(uint8_t quiet_transport_mask);

ACLSHMEM_DEVICE uint8_t aclshmemi_get_quiet_transport_mask();

ACLSHMEM_DEVICE void aclshmemi_quiet();

#endif
