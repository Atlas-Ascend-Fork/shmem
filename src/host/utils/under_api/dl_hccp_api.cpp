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
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>
#include "shmemi_functions.h"
#include "dl_hccp_api.h"
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

    /* 探测 libra.so 的 SocketConnectInfoT 布局(224/232); 探测失败时降级为 V1 直传(等价于兼容层引入前的行为),
     * 避免因个别环境(如容器 overlayfs)下镜像定位失败而连带关闭整个 RDMA 能力 */
    bool strideDetected = true;
    gConnectInfoStride = DetectConnectInfoStride(reinterpret_cast<void*>(gRaSocketBatchConnect));
    if (gConnectInfoStride != sizeof(HccpSocketConnectInfo) && gConnectInfoStride != sizeof(HccpSocketConnectInfoV2)) {
        SHM_LOG_WARN(
            "Cannot detect SocketConnectInfoT layout of ["
            << gRaLibName << "], got stride=" << gConnectInfoStride
            << ", fall back to V1 pass-through, stride=" << sizeof(HccpSocketConnectInfo));
        gConnectInfoStride = sizeof(HccpSocketConnectInfo);
        strideDetected = false;
    }
    Dl_info libInfo = {};
    const char* libPath =
        (dladdr(reinterpret_cast<void*>(gRaSocketBatchConnect), &libInfo) != 0 && libInfo.dli_fname != nullptr) ?
            libInfo.dli_fname :
            gRaLibName;
    SHM_LOG_INFO(
        "LoadLibrary for DlHccpApi success, libra=" << libPath << ", SocketConnectInfoT stride=" << gConnectInfoStride
                                                    << (gConnectInfoStride == sizeof(HccpSocketConnectInfoV2) ?
                                                            " (V2, conn array auto-converted)" :
                                                        strideDetected ? " (V1, conn array used as-is)" :
                                                                         " (V1, degraded, conn array used as-is)"));
    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

namespace {
constexpr const char kV2Marker[] = "tagChkDis";
constexpr size_t kV2MarkerLen = sizeof(kV2Marker) - 1;
/* 文件内容回退扫描的分块大小, 读取时保留 kV2MarkerLen - 1 字节跨块重叠, 防止标记串被块边界截断 */
constexpr size_t kScanChunkSize = 4096U;

/* 在 [begin, end) 中搜索 V2 标记串 */
bool RangeContainsV2Marker(const char* begin, const char* end)
{
    return std::search(begin, end, kV2Marker, kV2Marker + kV2MarkerLen) != end;
}

bool ParseMapsLine(
    const char* line, uintptr_t& start, uintptr_t& end, char* perms, size_t permsSize, unsigned int& devMajor,
    unsigned int& devMinor, unsigned long long& inode)
{
    std::istringstream stream(line);
    std::string permsStr;
    std::string offsetStr; /* offset 为 16 进制(可含字母), 仅作占位跳过, 不做数值解析 */
    stream >> std::hex >> start;
    if (stream.fail() || stream.peek() != '-') {
        return false;
    }
    stream.ignore(1);
    if (!(stream >> end >> permsStr >> offsetStr)) {
        return false;
    }
    /* dev 字段: "major:minor", 内核以 16 进制输出, 冒号留在流中单独跳过 */
    stream >> std::hex >> devMajor;
    if (stream.fail() || stream.peek() != ':') {
        return false;
    }
    stream.ignore(1);
    if (!(stream >> devMinor >> std::dec >> inode)) {
        return false;
    }
    if (permsStr.size() + 1 > permsSize) {
        return false;
    }
    std::memcpy(perms, permsStr.c_str(), permsStr.size() + 1);
    return true;
}
/* 方式一: 按 dev/ino 定位库文件在 /proc/self/maps 中的可读映射段, 扫描运行时镜像。
 * 返回 1(命中 V2 标记) / 0(完整扫描未命中) / -1(未能定位到本库的映射段) */
int ScanMappedImageForV2Marker(const struct stat& target)
{
    FILE* maps = fopen("/proc/self/maps", "r");
    if (maps == nullptr) {
        return -1;
    }
    bool rangeFound = false;
    bool markerFound = false;
    char line[1024] = {0};
    while (fgets(line, sizeof(line), maps) != nullptr && !markerFound) {
        uintptr_t start = 0;
        uintptr_t end = 0;
        char perms[8] = {0};
        unsigned int devMajor = 0;
        unsigned int devMinor = 0;
        unsigned long long inode = 0;
        if (!ParseMapsLine(line, start, end, perms, sizeof(perms), devMajor, devMinor, inode)) {
            continue;
        }
        if (perms[0] != 'r') {
            continue;
        }
        if (major(target.st_dev) != devMajor || minor(target.st_dev) != devMinor ||
            static_cast<unsigned long long>(target.st_ino) != inode) {
            continue;
        }
        rangeFound = true;
        markerFound = RangeContainsV2Marker(reinterpret_cast<const char*>(start), reinterpret_cast<const char*>(end));
    }
    fclose(maps);
    if (!rangeFound) {
        return -1;
    }
    return markerFound ? 1 : 0;
}

/* 方式二(回退): 直接扫描库文件内容。容器 overlayfs 等场景下 stat() 与 /proc/self/maps 的 dev/ino
 * 可能不一致导致方式一定位失败, 而 V2 标记串同样存在于文件镜像的 .rodata 中。
 * 返回值语义同上; 文件不可读或为空时返回 -1 */
int ScanFileForV2Marker(const char* path)
{
    FILE* file = fopen(path, "rb");
    if (file == nullptr) {
        return -1;
    }
    const int stride = DlHccpApi::DetectStrideFromFile(file);
    fclose(file);
    return stride == sizeof(HccpSocketConnectInfoV2) ? 1 : stride == sizeof(HccpSocketConnectInfo) ? 0 : -1;
}
} /* namespace */

int DlHccpApi::DetectStrideFromFile(std::FILE* file)
{
    if (file == nullptr) {
        return 0;
    }
    std::vector<char> buf(kScanChunkSize + kV2MarkerLen - 1);
    size_t carry = 0; /* 上一块末尾保留的重叠字节数 */
    for (;;) {
        const size_t nRead = fread(buf.data() + carry, 1, kScanChunkSize, file);
        const bool eof = feof(file) != 0;
        const size_t total = carry + nRead;
        if (ferror(file) != 0 || total == 0) {
            return 0; /* 读取出错或空文件, 无法判定 */
        }
        if (RangeContainsV2Marker(buf.data(), buf.data() + total)) {
            return sizeof(HccpSocketConnectInfoV2);
        }
        if (eof) {
            return sizeof(HccpSocketConnectInfo);
        }
        carry = std::min(total, kV2MarkerLen - 1U);
        std::memmove(buf.data(), buf.data() + total - carry, carry);
    }
}

int DlHccpApi::DetectStrideFromImage(const char* imagePath)
{
    /*
     * 在 libra.so 镜像中搜索 V2(232) 新增字段 "tagChkDis" 的日志格式串以判定布局
     * (该串位于 .rodata, V1(224) 版本无此字段), 只读镜像、不解析指令, 与宿主指令集无关:
     *   优先按 dev/ino 匹配 /proc/self/maps 定位本库可读映射段, 扫描运行时镜像;
     *   定位失败(如容器 overlayfs 下 stat 与 maps 的 dev/ino 不一致)时回退为直接扫描库文件内容;
     *   两者均无法判定 -> 返回 0, 由调用方降级为 V1。
     */
    if (imagePath == nullptr) {
        return 0;
    }
    struct stat target = {};
    if (stat(imagePath, &target) == 0) {
        const int mapped = ScanMappedImageForV2Marker(target);
        if (mapped >= 0) {
            return mapped == 1 ? sizeof(HccpSocketConnectInfoV2) : sizeof(HccpSocketConnectInfo);
        }
    }
    const int fromFile = ScanFileForV2Marker(imagePath);
    if (fromFile >= 0) {
        return fromFile == 1 ? sizeof(HccpSocketConnectInfoV2) : sizeof(HccpSocketConnectInfo);
    }
    return 0;
}

int DlHccpApi::DetectConnectInfoStride(void* funcAddr)
{
    if (funcAddr == nullptr) {
        return 0;
    }
    Dl_info info = {};
    if (dladdr(funcAddr, &info) == 0 || info.dli_fname == nullptr) {
        return 0;
    }
    return DetectStrideFromImage(info.dli_fname);
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
