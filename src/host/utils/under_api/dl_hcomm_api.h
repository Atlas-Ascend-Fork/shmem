/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_CORE_DL_HCOMM_API_H
#define MF_HYBM_CORE_DL_HCOMM_API_H

#include <cstdint>
#include <mutex>

#include "hcomm_entity_compat.h"
#include "host/shmem_host_def.h"

namespace shm {

using HcommEndpointCreateFunc = HcommResult (*)(const EndpointDesc*, EndpointHandle*);
using HcommEndpointDestroyFunc = HcommResult (*)(EndpointHandle);
using HcommEndpointGetListenPortFunc = HcommResult (*)(EndpointHandle, uint32_t*);
using HcommMemRegFunc = HcommResult (*)(EndpointHandle, const char*, CommMem*, HcommMemHandle*);
using HcommMemUnregFunc = HcommResult (*)(EndpointHandle, HcommMemHandle);
using HcommChannelCreateFunc = HcommResult (*)(EndpointHandle, CommEngine, HcommChannelDesc*, uint32_t, ChannelHandle*);
using HcommChannelDestroyFunc = HcommResult (*)(const ChannelHandle*, uint32_t);
using HcommChannelGetStatusFunc = HcommResult (*)(const ChannelHandle*, uint32_t, int32_t*);

// HCOM's channel config handle is an opaque pointer (`typedef void *HcommChannelConfig`) and the
// config type is an enum whose underlying type is int32_t. Both are mirrored here in project scope
// so the declarations stay independent of the CANN header spelling: these symbols are resolved with
// dlsym and older CANN packages do not ship the channel config declarations at all.
using HcommChannelConfig = void*;
using HcommChannelConfigType = int32_t;
inline constexpr HcommChannelConfigType HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0;

// Returned by the channel config wrappers when the symbol is not available (matches HCCL_E_PTR).
inline constexpr HcommResult HCOMM_RESULT_SYMBOL_UNAVAILABLE = 2;

using HcommChannelConfigCreateFunc = HcommResult (*)(HcommChannelConfig*);
using HcommChannelConfigDestroyFunc = HcommResult (*)(HcommChannelConfig);
using HcommChannelConfigSetIntFunc = HcommResult (*)(HcommChannelConfig, HcommChannelConfigType, uint32_t);
using HcommChannelCreateWithConfigFunc =
    HcommResult (*)(EndpointHandle, CommEngine, HcommChannelDesc*, uint32_t, HcommChannelConfig, ChannelHandle*);

class DlHcommApi {
public:
    static Result LoadLibrary();
    static void CleanupLibrary();

    static inline HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle)
    {
        return gHcommEndpointCreate(endpoint, endpointHandle);
    }

    static inline HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle)
    {
        return gHcommEndpointDestroy(endpointHandle);
    }

    static inline HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port)
    {
        return gHcommEndpointGetListenPort(endpointHandle, port);
    }

    static inline HcommResult HcommMemReg(
        EndpointHandle endpointHandle, const char* memTag, CommMem* mem, HcommMemHandle* memHandle)
    {
        return gHcommMemReg(endpointHandle, memTag, mem, memHandle);
    }

    static inline HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
    {
        return gHcommMemUnreg(endpointHandle, memHandle);
    }

    static inline HcommResult HcommChannelCreate(
        EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
        ChannelHandle* channels)
    {
        return gHcommChannelCreate(endpointHandle, engine, channelDescs, channelNum, channels);
    }

    static inline HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum)
    {
        return gHcommChannelDestroy(channels, channelNum);
    }

    static inline HcommResult HcommChannelGetStatus(
        const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
    {
        return gHcommChannelGetStatus(channelList, listNum, statusList);
    }

    // True only when all four channel-config symbols were resolved. Callers must check this before
    // using the wrappers below and fall back to HcommChannelCreate when it returns false.
    static bool HasChannelConfigApis();

    static inline HcommResult HcommChannelConfigCreate(HcommChannelConfig* config)
    {
        if (gHcommChannelConfigCreate == nullptr) {
            return HCOMM_RESULT_SYMBOL_UNAVAILABLE;
        }
        return gHcommChannelConfigCreate(config);
    }

    static inline HcommResult HcommChannelConfigDestroy(HcommChannelConfig config)
    {
        if (gHcommChannelConfigDestroy == nullptr) {
            return HCOMM_RESULT_SYMBOL_UNAVAILABLE;
        }
        return gHcommChannelConfigDestroy(config);
    }

    static inline HcommResult HcommChannelConfigSetInt(
        HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
    {
        if (gHcommChannelConfigSetInt == nullptr) {
            return HCOMM_RESULT_SYMBOL_UNAVAILABLE;
        }
        return gHcommChannelConfigSetInt(config, type, value);
    }

    static inline HcommResult HcommChannelCreateWithConfig(
        EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
        HcommChannelConfig config, ChannelHandle* channels)
    {
        if (gHcommChannelCreateWithConfig == nullptr) {
            return HCOMM_RESULT_SYMBOL_UNAVAILABLE;
        }
        return gHcommChannelCreateWithConfig(endpointHandle, engine, channelDescs, channelNum, config, channels);
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void* gHcommHandle;
    static const char* gHcommLibName;

    static HcommEndpointCreateFunc gHcommEndpointCreate;
    static HcommEndpointDestroyFunc gHcommEndpointDestroy;
    static HcommEndpointGetListenPortFunc gHcommEndpointGetListenPort;
    static HcommMemRegFunc gHcommMemReg;
    static HcommMemUnregFunc gHcommMemUnreg;
    static HcommChannelCreateFunc gHcommChannelCreate;
    static HcommChannelDestroyFunc gHcommChannelDestroy;
    static HcommChannelGetStatusFunc gHcommChannelGetStatus;

    // Soft-loaded symbols: absent on older libhcomm.so, so they stay nullptr instead of failing load.
    static HcommChannelConfigCreateFunc gHcommChannelConfigCreate;
    static HcommChannelConfigDestroyFunc gHcommChannelConfigDestroy;
    static HcommChannelConfigSetIntFunc gHcommChannelConfigSetInt;
    static HcommChannelCreateWithConfigFunc gHcommChannelCreateWithConfig;
};

} // namespace shm

#endif // MF_HYBM_CORE_DL_HCOMM_API_H
