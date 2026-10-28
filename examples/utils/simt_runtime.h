/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SIMT_RUNTIME_H
#define SIMT_RUNTIME_H

#include "acl/acl.h"
#include "shmem.h"

struct RuntimeResources {
    int device_id{0};
    aclrtStream stream{nullptr};
    bool acl_initialized{false};
    bool device_set{false};
    bool stream_created{false};
    bool shmem_initialized{false};
};

static void RecordCleanupError(int cleanup_status, const char* operation, int& status)
{
    if (cleanup_status == 0) {
        return;
    }
    ERROR_LOG("%s failed during cleanup, status=%d", operation, cleanup_status);
    if (status == 0) {
        status = cleanup_status;
    }
}

static int CleanupResources(RuntimeResources& resources, int status)
{
    if (resources.shmem_initialized) {
        RecordCleanupError(aclshmem_finalize(), "aclshmem_finalize", status);
        resources.shmem_initialized = false;
    }
    if (resources.stream_created) {
        RecordCleanupError(aclrtDestroyStream(resources.stream), "aclrtDestroyStream", status);
        resources.stream = nullptr;
        resources.stream_created = false;
    }
    if (resources.device_set) {
        RecordCleanupError(aclrtResetDevice(resources.device_id), "aclrtResetDevice", status);
        resources.device_set = false;
    }
    if (resources.acl_initialized) {
        RecordCleanupError(aclFinalize(), "aclFinalize", status);
        resources.acl_initialized = false;
    }
    return status;
}

#endif // SIMT_RUNTIME_H
