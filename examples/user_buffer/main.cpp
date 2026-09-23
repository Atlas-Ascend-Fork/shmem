/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "user_buffer_kernel.h"
#include "utils.h"

namespace {

constexpr uint32_t kPayloadBytes = 64U * 1024U;
constexpr uint64_t kTailRequestBytes = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kUserBufferCount = 2;

enum class Engine {
    Mte,
    Udma,
};

struct SourceMapping {
    void* addr{nullptr};
    bool mapped{false};
};

bool CheckAcl(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "[ERROR] " << operation << " failed, aclError=" << status << std::endl;
    return false;
}

bool CheckShmem(int32_t status, const char* operation)
{
    if (status == ACLSHMEM_SUCCESS) {
        return true;
    }
    std::cerr << "[ERROR] " << operation << " failed, status=" << status << std::endl;
    return false;
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t* result)
{
    if (alignment == 0 || result == nullptr || value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
        return false;
    }
    *result = ((value + alignment - 1) / alignment) * alignment;
    return true;
}

bool CreateSourceMapping(aclrtDrvMemHandle handle, uint64_t size, int32_t logicDeviceId, SourceMapping* mapping)
{
    if (handle == nullptr || size == 0 || mapping == nullptr ||
        !CheckAcl(aclrtReserveMemAddress(&mapping->addr, size, 0, nullptr, 0), "aclrtReserveMemAddress") ||
        !CheckAcl(aclrtMapMem(mapping->addr, size, 0, handle, 0), "aclrtMapMem")) {
        return false;
    }
    mapping->mapped = true;

    aclrtMemAccessDesc access{};
    access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = logicDeviceId;
    access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    return CheckAcl(aclrtMemSetAccess(mapping->addr, size, &access, 1), "aclrtMemSetAccess");
}

bool DestroySourceMapping(SourceMapping* mapping)
{
    bool success = true;
    if (mapping->mapped) {
        success = CheckAcl(aclrtUnmapMem(mapping->addr), "aclrtUnmapMem") && success;
        mapping->mapped = false;
    }
    if (mapping->addr != nullptr) {
        success = CheckAcl(aclrtReleaseMemAddress(mapping->addr), "aclrtReleaseMemAddress") && success;
        mapping->addr = nullptr;
    }
    return success;
}

bool SetInitAttributes(
    int32_t myPe, int32_t nPes, uint64_t localMemSize, const char* ipPort, Engine engine,
    aclshmemx_init_attr_t* attributes)
{
    const size_t ipPortLength = std::strlen(ipPort);
    if (ipPortLength == 0 || ipPortLength >= sizeof(attributes->ip_port)) {
        return false;
    }
    attributes->my_pe = myPe;
    attributes->n_pes = nPes;
    attributes->local_mem_size = localMemSize;
    attributes->option_attr.data_op_engine_type = engine == Engine::Mte ? ACLSHMEM_DATA_OP_MTE : ACLSHMEM_DATA_OP_UDMA;
    std::copy_n(ipPort, ipPortLength, attributes->ip_port);
    attributes->ip_port[ipPortLength] = '\0';
    return true;
}

const char* EngineName(Engine engine) { return engine == Engine::Mte ? "mte" : "udma"; }

uint8_t BufferPattern(int32_t pe, size_t bufferIndex) { return static_cast<uint8_t>(pe + 1 + bufferIndex * 64U); }

bool VerifyBuffer(const std::vector<uint8_t>& data, uint8_t expected, int32_t myPe, size_t bufferIndex)
{
    const auto mismatch =
        std::find_if(data.begin(), data.end(), [expected](uint8_t value) { return value != expected; });
    if (mismatch == data.end()) {
        return true;
    }
    std::cerr << "[ERROR] PE " << myPe << " buffer " << bufferIndex << " mismatch at byte "
              << std::distance(data.begin(), mismatch) << ", expected=" << static_cast<uint32_t>(expected)
              << ", actual=" << static_cast<uint32_t>(*mismatch) << std::endl;
    return false;
}

int RunExample(int32_t myPe, int32_t nPes, int32_t deviceId, const char* ipPort, Engine engine)
{
    aclrtStream stream = nullptr;
    std::array<aclrtDrvMemHandle, kUserBufferCount> handles{};
    std::array<SourceMapping, kUserBufferCount> sourceMappings{};
    uint8_t* scratch = nullptr;
    bool aclInitialized = false;
    bool deviceSet = false;
    bool shmemInitialized = false;
    bool globalExitRequested = false;
    bool success = false;

    auto requestGlobalExit = [&globalExitRequested]() {
        globalExitRequested = true;
        aclshmem_global_exit(EXIT_FAILURE);
    };

    do {
        if (!CheckAcl(aclInit(nullptr), "aclInit")) {
            break;
        }
        aclInitialized = true;
        if (!CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
            break;
        }
        deviceSet = true;
        int32_t logicDeviceId = -1;
        if (!CheckAcl(aclrtGetLogicDevIdByUserDevId(deviceId, &logicDeviceId), "aclrtGetLogicDevIdByUserDevId")) {
            break;
        }
        if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
            break;
        }

        aclrtPhysicalMemProp property{};
        property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        property.memAttr = ACL_HBM_MEM_HUGE;
        property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        property.location.id = deviceId;

        size_t granularity = 0;
        if (!CheckAcl(
                aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity),
                "aclrtMemGetAllocationGranularity")) {
            break;
        }
        uint64_t bufferBytes = 0;
        uint64_t tailBytes = 0;
        if (!AlignUp(kPayloadBytes, granularity, &bufferBytes) ||
            !AlignUp(kTailRequestBytes, granularity, &tailBytes)) {
            break;
        }

        std::array<aclshmemx_buffer_optional_attr_t, kUserBufferCount> optionalAttributes{};
        std::array<aclshmemx_buffer_desc_t, kUserBufferCount> buffers{};
        bool buffersPrepared = true;
        for (size_t bufferIndex = 0; bufferIndex < kUserBufferCount; ++bufferIndex) {
            if (!CheckAcl(
                    aclrtMallocPhysical(&handles[bufferIndex], bufferBytes, &property, 0), "aclrtMallocPhysical") ||
                !CreateSourceMapping(handles[bufferIndex], bufferBytes, logicDeviceId, &sourceMappings[bufferIndex]) ||
                !CheckAcl(aclrtMemset(sourceMappings[bufferIndex].addr, bufferBytes, 0, bufferBytes), "aclrtMemset")) {
                buffersPrepared = false;
                break;
            }
            optionalAttributes[bufferIndex].mem_handle = handles[bufferIndex];
            buffers[bufferIndex].addr = sourceMappings[bufferIndex].addr;
            buffers[bufferIndex].size = bufferBytes;
            buffers[bufferIndex].optional_attr = &optionalAttributes[bufferIndex];
        }
        if (!buffersPrepared) {
            break;
        }

        aclshmemx_init_attr_t attributes{};
        if (!SetInitAttributes(myPe, nPes, tailBytes, ipPort, engine, &attributes) ||
            !CheckShmem(
                aclshmemx_init_attr_with_buffers(
                    ACLSHMEMX_INIT_WITH_DEFAULT, &attributes, buffers.data(), buffers.size()),
                "aclshmemx_init_attr_with_buffers")) {
            break;
        }
        shmemInitialized = true;

        std::array<uint8_t*, kUserBufferCount> userBuffers{};
        bool bufferLookupSucceeded = true;
        for (size_t bufferIndex = 0; bufferIndex < kUserBufferCount; ++bufferIndex) {
            userBuffers[bufferIndex] =
                static_cast<uint8_t*>(aclshmemx_get_buffer_ptr(sourceMappings[bufferIndex].addr));
            if (userBuffers[bufferIndex] == nullptr) {
                std::cerr << "[ERROR] aclshmemx_get_buffer_ptr failed for buffer " << bufferIndex << std::endl;
                bufferLookupSucceeded = false;
                break;
            }
        }
        if (!bufferLookupSucceeded) {
            requestGlobalExit();
            break;
        }
        constexpr size_t kScratchBytes = kPayloadBytes * kUserBufferCount;
        scratch = static_cast<uint8_t*>(aclshmem_malloc(kScratchBytes));
        if (scratch == nullptr) {
            std::cerr << "[ERROR] aclshmem_malloc failed" << std::endl;
            requestGlobalExit();
            break;
        }

        std::vector<uint8_t> input(kScratchBytes);
        for (size_t bufferIndex = 0; bufferIndex < kUserBufferCount; ++bufferIndex) {
            std::fill_n(input.begin() + bufferIndex * kPayloadBytes, kPayloadBytes, BufferPattern(myPe, bufferIndex));
        }
        if (!CheckAcl(
                aclrtMemcpy(scratch, kScratchBytes, input.data(), input.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy H2D")) {
            requestGlobalExit();
            break;
        }

        aclshmem_barrier_all();
        const int32_t peer = (myPe + 1) % nPes;
        for (size_t bufferIndex = 0; bufferIndex < kUserBufferCount; ++bufferIndex) {
            uint8_t* source = scratch + bufferIndex * kPayloadBytes;
            if (engine == Engine::Mte) {
                LaunchUserBufferMtePut(
                    stream, util_get_ffts_config(), userBuffers[bufferIndex], source, kPayloadBytes, peer);
            } else {
                LaunchUserBufferUdmaPut(stream, userBuffers[bufferIndex], source, kPayloadBytes, peer);
            }
        }
        if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
            requestGlobalExit();
            break;
        }
        aclshmem_barrier_all();

        const int32_t sourcePe = (myPe + nPes - 1) % nPes;
        success = true;
        for (size_t bufferIndex = 0; bufferIndex < kUserBufferCount; ++bufferIndex) {
            std::vector<uint8_t> output(kPayloadBytes);
            if (!CheckAcl(
                    aclrtMemcpy(
                        output.data(), output.size(), userBuffers[bufferIndex], kPayloadBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST),
                    "aclrtMemcpy D2H")) {
                requestGlobalExit();
                success = false;
                break;
            }
            success = VerifyBuffer(output, BufferPattern(sourcePe, bufferIndex), myPe, bufferIndex) && success;
        }
        if (globalExitRequested) {
            break;
        }
        if (success) {
            std::cout << "[SUCCESS] PE " << myPe << " verified " << kUserBufferCount << " caller-owned buffers with "
                      << EngineName(engine) << std::endl;
        }
    } while (false);

    // A failure after SHMEM initialization may occur on only one PE. Once global exit is requested, do not enter
    // aclshmem_free/finalize from this PE: another PE may still be in a different collective. The launcher observes
    // this process failure and terminates the remaining local PE processes.
    if (globalExitRequested) {
        return EXIT_FAILURE;
    }

    if (scratch != nullptr && shmemInitialized) {
        aclshmem_free(scratch);
    }
    bool finalizeSucceeded = true;
    if (shmemInitialized) {
        finalizeSucceeded = CheckShmem(aclshmem_finalize(), "aclshmem_finalize");
        success = success && finalizeSucceeded;
    }
    if (!shmemInitialized || finalizeSucceeded) {
        for (size_t bufferIndex = kUserBufferCount; bufferIndex > 0; --bufferIndex) {
            const size_t index = bufferIndex - 1;
            success = DestroySourceMapping(&sourceMappings[index]) && success;
            if (handles[index] != nullptr) {
                success = CheckAcl(aclrtFreePhysical(handles[index]), "aclrtFreePhysical") && success;
            }
        }
    }
    if (stream != nullptr) {
        success = CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream") && success;
    }
    if (deviceSet) {
        success = CheckAcl(aclrtResetDevice(deviceId), "aclrtResetDevice") && success;
    }
    if (aclInitialized) {
        success = CheckAcl(aclFinalize(), "aclFinalize") && success;
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <n_pes> <my_pe> <ip_port> <n_devices> <first_device> <mte|udma>"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const int32_t nPes = std::atoi(argv[1]);
    const int32_t myPe = std::atoi(argv[2]);
    const int32_t nDevices = std::atoi(argv[4]);
    const int32_t firstDevice = std::atoi(argv[5]);
    const std::string engineName = argv[6];
    if (nPes < 2 || myPe < 0 || myPe >= nPes || nDevices < nPes || firstDevice < 0 ||
        (engineName != "mte" && engineName != "udma")) {
        std::cerr << "[ERROR] invalid PE, device, or engine arguments" << std::endl;
        return EXIT_FAILURE;
    }
#ifndef ACLSHMEM_SOC_950
    if (engineName == "udma") {
        std::cerr << "[ERROR] UDMA requires an Ascend950 build" << std::endl;
        return EXIT_FAILURE;
    }
#endif
    const Engine engine = engineName == "mte" ? Engine::Mte : Engine::Udma;
    return RunExample(myPe, nPes, myPe % nDevices + firstDevice, argv[3], engine);
}
