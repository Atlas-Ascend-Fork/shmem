/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <atomic>
#include <cstring>
#include <vector>
#include <sstream>
#include "shmemi_functions.h"
#include "dl_hccp_api.h"
#include "dl_comm_def.h"
#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"

namespace shm {
bool DlHccpApi::gLoaded = false;
std::mutex DlHccpApi::gMutex;
void* DlHccpApi::raHandle;
void* DlHccpApi::tsdHandle;
int DlHccpApi::gConnectInfoStride = 0;

const char* DlHccpApi::gRaLibName = "libra.so";
const char* DlHccpApi::gTsdLibName = "libtsdclient.so";

raRdevGetHandleFunc DlHccpApi::gRaRdevGetHandle;
raInitFunc DlHccpApi::gRaInit;
raDeinitFunc DlHccpApi::gRaDeinit;
raSocketInitFunc DlHccpApi::gRaSocketInit;
raSocketDeinitFunc DlHccpApi::gRaSocketDeinit;
raRdevInitV2Func DlHccpApi::gRaRdevInitV2;
raSocketBatchConnectFunc DlHccpApi::gRaSocketBatchConnect;
raSocketBatchCloseFunc DlHccpApi::gRaSocketBatchClose;
raSocketListenStartFunc DlHccpApi::gRaSocketListenStart;
raSocketListenStopFunc DlHccpApi::gRaSocketListenStop;
raGetSocketsFunc DlHccpApi::gRaGetSockets;
raGetIfNumFunc DlHccpApi::gRaGetIfNum;
raGetIfAddrsFunc DlHccpApi::gRaGetIfAddrs;
raSocketWhiteListAddFunc DlHccpApi::gRaSocketWhiteListAdd;
raSocketWhiteListDelFunc DlHccpApi::gRaSocketWhiteListDel;
raQpCreateFunc DlHccpApi::gRaQpCreate;
raQpAiCreateFunc DlHccpApi::gRaQpAiCreate;
raQpDestroyFunc DlHccpApi::gRaQpDestroy;
raGetQpStatusFunc DlHccpApi::gRaGetQpStatus;
raQpConnectAsyncFunc DlHccpApi::gRaQpConnectAsync;
raRegisterMrFunc DlHccpApi::gRaRegisterMR;
raDeregisterMrFunc DlHccpApi::gRaDeregisterMR;
raMrRegFunc DlHccpApi::gRaMrReg;
raSetQpAttrQosFunc DlHccpApi::gRaSetQpAttrQos;
tsdOpenFunc DlHccpApi::gTsdOpen;

int DlHccpApi::SelectSocketConnectInfoStride(const std::string& hcommVersion, const std::string& cannVersion)
{
    unsigned int hcommMajor = 0;
    unsigned int hcommMinor = 0;
    unsigned int hcommPatch = 0;
    char firstSeparator = 0;
    char secondSeparator = 0;
    std::istringstream hcomm(hcommVersion);
    // hcomm 9.2 needs the CANN release type to distinguish the socket layouts.
    if (hcommVersion.find_first_not_of("0123456789.") == std::string::npos &&
        (hcomm >> hcommMajor >> firstSeparator >> hcommMinor >> secondSeparator >> hcommPatch) &&
        firstSeparator == '.' && secondSeparator == '.' && hcomm.eof()) {
        int stride = hcommMajor > 9 || (hcommMajor == 9 && hcommMinor >= 3) ? sizeof(HccpSocketConnectInfoV2) :
                                                                              sizeof(HccpSocketConnectInfo);
        if (hcommMajor == 9 && hcommMinor == 2) {
            const bool legacyRelease =
                cannVersion.find("beta") != std::string::npos || cannVersion.find('T') != std::string::npos;
            stride = legacyRelease ? sizeof(HccpSocketConnectInfo) : sizeof(HccpSocketConnectInfoV2);
        }
        SHM_LOG_INFO("hcomm version=" << hcommVersion << ", SocketConnectInfoT stride=" << stride);
        return stride;
    }

    SHM_LOG_WARN(
        "hcomm version unavailable or invalid [" << hcommVersion << "], falling back to CANN version judgement ["
                                                 << cannVersion << "]");
    // Preserve the CANN fallback parser: accepting T suffixes would misclassify 9.2.T530 as V2.
    const auto legacyFallback = [&cannVersion]() {
        SHM_LOG_WARN(
            "Using V1(224) socket layout from CANN version fallback ["
            << cannVersion << "]; this is only safe for CANN < 9.3. Verify if CANN >= 9.3.");
        return sizeof(HccpSocketConnectInfo);
    };
    // The two 9.2.0 beta releases still use the legacy connect/listen ABI.
    if (cannVersion == "9.2.0-beta.1" || cannVersion == "9.2.0-beta.2" || cannVersion == "9.2.0.beta.1" ||
        cannVersion == "9.2.0.beta.2") {
        return legacyFallback();
    }

    unsigned int major = 0;
    unsigned int minor = 0;
    unsigned int patch = 0;
    char firstDot = 0;
    char secondDot = 0;
    std::istringstream version(cannVersion);
    if (!(version >> major >> firstDot >> minor >> secondDot >> patch) || firstDot != '.' || secondDot != '.') {
        return legacyFallback();
    }
    return major > 9 || (major == 9 && minor >= 2) ? sizeof(HccpSocketConnectInfoV2) : legacyFallback();
}

Result DlHccpApi::LoadLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    raHandle = dlopen(gRaLibName, RTLD_NOW);
    if (raHandle == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gRaLibName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    tsdHandle = dlopen(gTsdLibName, RTLD_NOW);
    if (tsdHandle == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gTsdLibName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror());
        dlclose(raHandle);
        raHandle = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM_ALT(gRaSocketInit, raSocketInitFunc, raHandle, "ra_socket_init", "RaSocketInit");
    DL_LOAD_SYM_ALT(gRaInit, raInitFunc, raHandle, "ra_init", "RaInit");
    DL_LOAD_SYM_ALT(gRaDeinit, raDeinitFunc, raHandle, "ra_deinit", "RaDeinit");
    DL_LOAD_SYM_ALT(gRaSocketDeinit, raSocketDeinitFunc, raHandle, "ra_socket_deinit", "RaSocketDeinit");
    DL_LOAD_SYM_ALT(gRaRdevInitV2, raRdevInitV2Func, raHandle, "ra_rdev_init_v2", "RaRdevInitV2");
    DL_LOAD_SYM_ALT(gRaRdevGetHandle, raRdevGetHandleFunc, raHandle, "ra_rdev_get_handle", "RaRdevGetHandle");
    DL_LOAD_SYM_ALT(
        gRaSocketBatchConnect, raSocketBatchConnectFunc, raHandle, "ra_socket_batch_connect", "RaSocketBatchConnect");
    DL_LOAD_SYM_ALT(
        gRaSocketBatchClose, raSocketBatchCloseFunc, raHandle, "ra_socket_batch_close", "RaSocketBatchClose");
    DL_LOAD_SYM_ALT(
        gRaSocketListenStart, raSocketListenStartFunc, raHandle, "ra_socket_listen_start", "RaSocketListenStart");
    DL_LOAD_SYM_ALT(
        gRaSocketListenStop, raSocketListenStopFunc, raHandle, "ra_socket_listen_stop", "RaSocketListenStop");
    DL_LOAD_SYM_ALT(gRaGetSockets, raGetSocketsFunc, raHandle, "ra_get_sockets", "RaGetSockets");
    DL_LOAD_SYM_ALT(gRaGetIfNum, raGetIfNumFunc, raHandle, "ra_get_ifnum", "RaGetIfnum");
    DL_LOAD_SYM_ALT(gRaGetIfAddrs, raGetIfAddrsFunc, raHandle, "ra_get_ifaddrs", "RaGetIfaddrs");
    DL_LOAD_SYM_ALT(
        gRaSocketWhiteListAdd, raSocketWhiteListAddFunc, raHandle, "ra_socket_white_list_add", "RaSocketWhiteListAdd");
    DL_LOAD_SYM_ALT(
        gRaSocketWhiteListDel, raSocketWhiteListDelFunc, raHandle, "ra_socket_white_list_del", "RaSocketWhiteListDel");
    DL_LOAD_SYM_ALT(gRaQpCreate, raQpCreateFunc, raHandle, "ra_qp_create", "RaQpCreate");
    DL_LOAD_SYM_ALT(gRaQpAiCreate, raQpAiCreateFunc, raHandle, "ra_ai_qp_create", "RaAiQpCreate");
    DL_LOAD_SYM_ALT(gRaQpDestroy, raQpDestroyFunc, raHandle, "ra_qp_destroy", "RaQpDestroy");
    DL_LOAD_SYM_ALT(gRaGetQpStatus, raGetQpStatusFunc, raHandle, "ra_get_qp_status", "RaGetQpStatus");
    DL_LOAD_SYM_ALT(gRaQpConnectAsync, raQpConnectAsyncFunc, raHandle, "ra_qp_connect_async", "RaQpConnectAsync");
    DL_LOAD_SYM_ALT(gRaRegisterMR, raRegisterMrFunc, raHandle, "ra_register_mr", "RaRegisterMr");
    DL_LOAD_SYM_ALT(gRaDeregisterMR, raDeregisterMrFunc, raHandle, "ra_deregister_mr", "RaDeregisterMr");
    DL_LOAD_SYM_ALT(gRaMrReg, raMrRegFunc, raHandle, "ra_mr_reg", "RaMrReg");
    DL_LOAD_SYM_ALT(gRaSetQpAttrQos, raSetQpAttrQosFunc, raHandle, "ra_set_qp_attr_qos", "RaSetQpAttrQos");

    DL_LOAD_SYM(gTsdOpen, tsdOpenFunc, tsdHandle, "TsdOpen");

    const std::string hcommVersion = GetHcommVersion();
    const std::string cannVersion = GetCannVersion();
    gConnectInfoStride = SelectSocketConnectInfoStride(hcommVersion, cannVersion);
    Dl_info libInfo = {};
    const char* libPath =
        (dladdr(reinterpret_cast<void*>(gRaSocketBatchConnect), &libInfo) != 0 && libInfo.dli_fname != nullptr) ?
            libInfo.dli_fname :
            gRaLibName;
    SHM_LOG_INFO(
        "LoadLibrary for DlHccpApi success, libra="
        << libPath << ", hcomm version=" << hcommVersion << ", CANN version=" << cannVersion
        << ", SocketConnectInfoT stride=" << gConnectInfoStride
        << (gConnectInfoStride == sizeof(HccpSocketConnectInfoV2) ? " (V2, conn array auto-converted)" :
                                                                    " (V1, conn array used as-is)"));
    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

int DlHccpApi::RaSocketBatchConnect(HccpSocketConnectInfo conn[], uint32_t num)
{
    /* 首次调用时记录实际使用的布局, 便于环境排查 */
    static std::atomic<bool> layoutLogged(false);
    if (!layoutLogged.exchange(true)) {
        SHM_LOG_INFO(
            "RaSocketBatchConnect layout in use: stride="
            << gConnectInfoStride << " (host struct=" << sizeof(HccpSocketConnectInfo) << "), num=" << num
            << (gConnectInfoStride == sizeof(HccpSocketConnectInfo) ? ", direct pass-through" :
                                                                      ", convert host array to V2 before call"));
    }

    if (gConnectInfoStride == sizeof(HccpSocketConnectInfo)) {
        return gRaSocketBatchConnect(conn, num);
    }
    if (gConnectInfoStride == sizeof(HccpSocketConnectInfoV2)) {
        /* V2(232) 布局: 转换后调用, tagChkDis 置 false 保持 tag 校验行为与旧版一致 */
        std::vector<HccpSocketConnectInfoV2> connV2(num);
        for (uint32_t i = 0; i < num; ++i) {
            connV2[i].handle = conn[i].handle;
            connV2[i].remoteIp = conn[i].remoteIp;
            connV2[i].port = conn[i].port;
            std::memcpy(connV2[i].tag, conn[i].tag, sizeof(connV2[i].tag));
            connV2[i].tagChkDis = false;
            std::memset(connV2[i].resv, 0, sizeof(connV2[i].resv));
        }
        return gRaSocketBatchConnect(reinterpret_cast<HccpSocketConnectInfo*>(connV2.data()), num);
    }
    return ACLSHMEM_DL_FUNC_FAILED;
}

namespace {

int RaSocketListenCompat(raSocketListenStartFunc func, int stride, HccpSocketListenInfo conn[], uint32_t num)
{
    /* stride 为 connect 结构体步长(224/232), 与 listen 布局同代次: V1<->24B, V2<->32B */
    if (stride == sizeof(HccpSocketConnectInfo)) {
        return func(conn, num);
    }
    if (stride == sizeof(HccpSocketConnectInfoV2)) {
        std::vector<HccpSocketListenInfoV2> connV2(num);
        for (uint32_t i = 0; i < num; ++i) {
            connV2[i].handle = conn[i].handle;
            connV2[i].port = conn[i].port;
            connV2[i].phase = conn[i].phase;
            connV2[i].err = conn[i].err;
            connV2[i].tagChkDis = false; /* 保持与旧版一致的 tag 校验行为 */
            std::memset(connV2[i].resv, 0, sizeof(connV2[i].resv));
        }
        const int ret = func(reinterpret_cast<HccpSocketListenInfo*>(connV2.data()), num);
        for (uint32_t i = 0; i < num; ++i) {
            conn[i].handle = connV2[i].handle;
            conn[i].port = connV2[i].port;
            conn[i].phase = connV2[i].phase;
            conn[i].err = connV2[i].err;
        }
        return ret;
    }
    return ACLSHMEM_DL_FUNC_FAILED;
}
} /* namespace */

int DlHccpApi::RaSocketListenStart(HccpSocketListenInfo conn[], uint32_t num)
{
    return RaSocketListenCompat(gRaSocketListenStart, gConnectInfoStride, conn, num);
}

int DlHccpApi::RaSocketListenStop(HccpSocketListenInfo conn[], uint32_t num)
{
    return RaSocketListenCompat(gRaSocketListenStop, gConnectInfoStride, conn, num);
}

void DlHccpApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    gRaRdevGetHandle = nullptr;
    gRaInit = nullptr;
    gRaDeinit = nullptr;
    gRaSocketInit = nullptr;
    gRaSocketDeinit = nullptr;
    gRaRdevInitV2 = nullptr;
    gRaSocketBatchConnect = nullptr;
    gRaSocketBatchClose = nullptr;
    gRaSocketListenStart = nullptr;
    gRaSocketListenStop = nullptr;
    gRaGetSockets = nullptr;
    gRaGetIfNum = nullptr;
    gRaGetIfAddrs = nullptr;
    gRaSocketWhiteListAdd = nullptr;
    gRaSocketWhiteListDel = nullptr;
    gRaQpCreate = nullptr;
    gRaQpAiCreate = nullptr;
    gRaQpDestroy = nullptr;
    gRaGetQpStatus = nullptr;
    gRaQpConnectAsync = nullptr;
    gRaRegisterMR = nullptr;
    gRaDeregisterMR = nullptr;
    gRaMrReg = nullptr;
    gRaSetQpAttrQos = nullptr;
    gTsdOpen = nullptr;
    gConnectInfoStride = 0;

    if (raHandle != nullptr) {
        dlclose(raHandle);
        raHandle = nullptr;
    }

    if (tsdHandle != nullptr) {
        dlclose(tsdHandle);
        tsdHandle = nullptr;
    }
    gLoaded = false;
}
} // namespace shm
