/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_UT_USER_BUFFER_MAPPING_H
#define SHMEM_UT_USER_BUFFER_MAPPING_H

#include <cstdio>
#include <dlfcn.h>
#include "acl/acl.h"

#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
namespace user_buffer_test {

inline aclError SelectAccessDevice(const char* version, int32_t userDeviceId, int32_t logicDeviceId, int32_t* accessId)
{
    int major = 0;
    int minor = 0;
    int patch = 0;
    if (version == nullptr || accessId == nullptr || std::sscanf(version, "%d.%d.%d", &major, &minor, &patch) != 3 ||
        major < 0 || minor < 0 || patch < 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // Runtime versions before 9.2 forward MemSetAccess descriptors unchanged; 9.2+ converts user IDs itself.
    *accessId = (major < 9 || (major == 9 && minor < 2)) ? logicDeviceId : userDeviceId;
    return ACL_SUCCESS;
}

inline aclError GetAccessDevice(int32_t userDeviceId, int32_t* accessId)
{
    if (accessId == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    int32_t logicDeviceId = -1;
    auto status = aclrtGetLogicDevIdByUserDevId(userDeviceId, &logicDeviceId);
    if (status != ACL_SUCCESS) {
        return status;
    }
    if (logicDeviceId == userDeviceId) {
        *accessId = userDeviceId;
        return ACL_SUCCESS;
    }
    using QueryVersion = aclError (*)(char*, char*);
    auto query = reinterpret_cast<QueryVersion>(dlsym(RTLD_DEFAULT, "aclsysGetVersionStr"));
    char package[] = "runtime";
    char version[ACL_PKG_VERSION_MAX_SIZE]{};
    if (query != nullptr) {
        status = query(package, version);
        if (status != ACL_SUCCESS) {
            return status;
        }
        return SelectAccessDevice(version, userDeviceId, logicDeviceId, accessId);
    }
    // Older packages expose the enum-based query instead of the string-based query.
    using QueryLegacyVersion = aclError (*)(aclCANNPackageName, aclCANNPackageVersion*);
    auto legacyQuery = reinterpret_cast<QueryLegacyVersion>(dlsym(RTLD_DEFAULT, "aclsysGetCANNVersion"));
    if (legacyQuery == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    aclCANNPackageVersion legacyVersion{};
    status = legacyQuery(ACL_PKG_NAME_RUNTIME, &legacyVersion);
    if (status != ACL_SUCCESS) {
        return status;
    }
    return SelectAccessDevice(legacyVersion.version, userDeviceId, logicDeviceId, accessId);
}

} // namespace user_buffer_test
#endif
#endif
