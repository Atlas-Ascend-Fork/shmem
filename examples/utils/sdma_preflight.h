/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_EXAMPLE_SDMA_PREFLIGHT_H
#define SHMEM_EXAMPLE_SDMA_PREFLIGHT_H

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include "shmem.h"

// Check the host-side SDMA prerequisites before launching a kernel that uses
// SDMA.  There is no public ACL query for an idle/healthy SDMA engine; this
// check covers the capabilities and QP configuration that can be validated
// before SHMEM initialization.
inline bool CheckSdmaAvailable(uint32_t qp_num, const char* example_name)
{
    int device_id = -1;
    if (aclrtGetDevice(&device_id) != ACL_SUCCESS) {
        std::cerr << "[" << example_name << "] failed to get current device before SDMA launch." << std::endl;
        return false;
    }

    int64_t vector_core_num = 0;
    aclError ret = aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector_core_num);
    if (ret != ACL_SUCCESS || vector_core_num <= 0) {
        ret = aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vector_core_num);
    }
    if (ret != ACL_SUCCESS || vector_core_num <= 0) {
        std::cerr << "[" << example_name << "] SDMA unavailable: failed to query vector-core count, aclError=" << ret
                  << std::endl;
        return false;
    }

    const uint32_t max_qp_num = std::min<uint32_t>(static_cast<uint32_t>(vector_core_num), ACLSHMEM_MAX_AIV_PER_NPU);
    if (qp_num == 0 || qp_num > max_qp_num) {
        std::cerr << "[" << example_name << "] SDMA unavailable: requested QP count=" << qp_num
                  << ", device capacity=" << max_qp_num << std::endl;
        return false;
    }

    const int32_t qp_ret = aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num);
    if (qp_ret != ACLSHMEM_SUCCESS) {
        std::cerr << "[" << example_name << "] SDMA QP initialization failed, ret=" << qp_ret << std::endl;
        return false;
    }
    return true;
}

#endif // SHMEM_EXAMPLE_SDMA_PREFLIGHT_H
