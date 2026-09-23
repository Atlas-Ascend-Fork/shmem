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
#include <string>

#include <gtest/gtest.h>

#include "shmem.h"
#include "init/shmemi_init.h"
#include "init/shmemi_user_buffer_heap.h"
#include "mem/heap/hybm_device_mem_segment.h"
#include "unittest_main_test.h"

namespace {

int CopyStatuses(const void* sendbuf, void* recvbuf, int size, aclshmemi_bootstrap_handle_t* bootHandle)
{
    if (sendbuf == nullptr || recvbuf == nullptr || size != static_cast<int>(sizeof(int32_t)) ||
        bootHandle == nullptr || bootHandle->bootstrap_state == nullptr) {
        return ACLSHMEM_INVALID_PARAM;
    }
    const auto* statuses = static_cast<const std::array<int32_t, 4>*>(bootHandle->bootstrap_state);
    std::copy_n(statuses->data(), bootHandle->npes, static_cast<int32_t*>(recvbuf));
    return ACLSHMEM_SUCCESS;
}

int FailAllgather(const void*, void*, int, aclshmemi_bootstrap_handle_t*) { return ACLSHMEM_INNER_ERROR; }

bool CreateSourceMapping(aclrtDrvMemHandle handle, uint64_t size, int32_t logicDeviceId, void** sourceAddress)
{
    if (handle == nullptr || size == 0 || sourceAddress == nullptr) {
        return false;
    }
    if (aclrtReserveMemAddress(sourceAddress, size, 0, nullptr, 0) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtMapMem(*sourceAddress, size, 0, handle, 0) != ACL_SUCCESS) {
        (void)aclrtReleaseMemAddress(*sourceAddress);
        *sourceAddress = nullptr;
        return false;
    }
    aclrtMemAccessDesc access{};
    access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = logicDeviceId;
    access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    if (aclrtMemSetAccess(*sourceAddress, size, &access, 1) != ACL_SUCCESS) {
        (void)aclrtUnmapMem(*sourceAddress);
        (void)aclrtReleaseMemAddress(*sourceAddress);
        *sourceAddress = nullptr;
        return false;
    }
    return true;
}

void DestroySourceMapping(void* sourceAddress)
{
    if (sourceAddress == nullptr) {
        return;
    }
    EXPECT_EQ(aclrtUnmapMem(sourceAddress), ACL_SUCCESS);
    EXPECT_EQ(aclrtReleaseMemAddress(sourceAddress), ACL_SUCCESS);
}

TEST(UserBufferHeapTest, CollectiveStatusGateReturnsFirstPeError)
{
    std::array<int32_t, 4> statuses{ACLSHMEM_SUCCESS, ACLSHMEM_INVALID_PARAM, ACLSHMEM_INVALID_VALUE, ACLSHMEM_SUCCESS};
    aclshmemi_bootstrap_handle_t handle{};
    handle.bootstrap_state = &statuses;
    handle.npes = 4;
    handle.allgather = CopyStatuses;

    EXPECT_EQ(aclshmemi_collective_status_gate(ACLSHMEM_SUCCESS, handle.npes, &handle), ACLSHMEM_INVALID_PARAM);
}

TEST(UserBufferHeapTest, CollectiveStatusGateSucceedsWhenEveryPeSucceeds)
{
    std::array<int32_t, 4> statuses{};
    aclshmemi_bootstrap_handle_t handle{};
    handle.bootstrap_state = &statuses;
    handle.npes = 4;
    handle.allgather = CopyStatuses;

    EXPECT_EQ(aclshmemi_collective_status_gate(ACLSHMEM_SUCCESS, handle.npes, &handle), ACLSHMEM_SUCCESS);
}

TEST(UserBufferHeapTest, CollectiveStatusGateValidatesInputsAndPropagatesCollectiveFailure)
{
    aclshmemi_bootstrap_handle_t handle{};
    EXPECT_EQ(aclshmemi_collective_status_gate(ACLSHMEM_SUCCESS, 1, nullptr), ACLSHMEM_INVALID_PARAM);
    EXPECT_EQ(aclshmemi_collective_status_gate(ACLSHMEM_SUCCESS, 0, &handle), ACLSHMEM_INVALID_PARAM);

    handle.allgather = FailAllgather;
    EXPECT_EQ(aclshmemi_collective_status_gate(ACLSHMEM_SUCCESS, 1, &handle), ACLSHMEM_INNER_ERROR);
}

TEST(UserBufferHeapTest, InternalLimitsMatchRfc)
{
    EXPECT_EQ(shm::kMaxBuffers, 1024U);
    EXPECT_EQ(shm::kMaxBufferMetadataBytes, 64ULL * 1024ULL * 1024ULL);
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    EXPECT_EQ(
        aclshmemx_init_attr_with_buffers(ACLSHMEMX_INIT_WITH_DEFAULT, nullptr, nullptr, 0), ACLSHMEM_INVALID_PARAM);
#else
    EXPECT_EQ(
        aclshmemx_init_attr_with_buffers(ACLSHMEMX_INIT_WITH_DEFAULT, nullptr, nullptr, 0), ACLSHMEM_NOT_SUPPORTED);
#endif
}

TEST(UserBufferHeapTest, RejectsOutdatedHbmExportInfoVersion)
{
    shm::MemSegmentOptions options{};
    options.shared = true;
    options.segType = shm::HYBM_MST_HBM;
    shm::MemSegmentDevice segment(options, 0);

    shm::HbmExportInfo info{};
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    info.magic = shm::HBM_SLICE_EXPORT_INFO_MAGIC;
#endif
    info.version = shm::EXPORT_INFO_VERSION - 1;
    const std::string serialized(reinterpret_cast<const char*>(&info), sizeof(info));
    EXPECT_EQ(segment.Import({serialized}, nullptr), ACLSHMEM_INVALID_PARAM);
}

TEST(UserBufferHeapTest, PublicBufferStructAbiIsStable)
{
    aclshmemx_buffer_optional_attr_t optionalAttr{};
    aclshmemx_buffer_desc_t buffer{};
    EXPECT_EQ(sizeof(aclshmemx_drv_mem_handle_t), sizeof(void*));
    EXPECT_EQ(sizeof(aclshmemx_mem_fabric_handle_t), sizeof(void*));
    EXPECT_EQ(sizeof(optionalAttr), 64U);
    EXPECT_EQ(alignof(decltype(optionalAttr)), alignof(uint64_t));
    EXPECT_EQ(sizeof(buffer), 64U);
    EXPECT_EQ(alignof(decltype(buffer)), alignof(uint64_t));
    EXPECT_EQ(optionalAttr.mem_handle, nullptr);
    EXPECT_EQ(optionalAttr.fabric_handle, nullptr);
    EXPECT_EQ(buffer.addr, nullptr);
    EXPECT_EQ(buffer.size, 0U);
    EXPECT_EQ(buffer.optional_attr, nullptr);
    EXPECT_TRUE(std::all_of(
        std::begin(optionalAttr.reserved), std::end(optionalAttr.reserved), [](uint64_t value) { return value == 0; }));
    EXPECT_TRUE(
        std::all_of(std::begin(buffer.reserved), std::end(buffer.reserved), [](uint64_t value) { return value == 0; }));
}

#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
TEST(UserBufferHeapTest, EngineValidationMatchesSocCapabilities)
{
    EXPECT_FALSE(aclshmemi_user_buffer_heap_engine_supported(static_cast<data_op_engine_type_t>(0)));
    EXPECT_TRUE(aclshmemi_user_buffer_heap_engine_supported(ACLSHMEM_DATA_OP_MTE));
    EXPECT_FALSE(aclshmemi_user_buffer_heap_engine_supported(ACLSHMEM_DATA_OP_SDMA));
    EXPECT_FALSE(aclshmemi_user_buffer_heap_engine_supported(ACLSHMEM_DATA_OP_ROCE));
    EXPECT_FALSE(aclshmemi_user_buffer_heap_engine_supported(static_cast<data_op_engine_type_t>(0x10)));
    EXPECT_FALSE(aclshmemi_user_buffer_heap_engine_supported(
        static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA)));
#if defined(ACLSHMEM_SOC_950)
    EXPECT_TRUE(aclshmemi_user_buffer_heap_engine_supported(ACLSHMEM_DATA_OP_UDMA));
    EXPECT_TRUE(aclshmemi_user_buffer_heap_engine_supported(
        static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA)));
#else
    EXPECT_FALSE(aclshmemi_user_buffer_heap_engine_supported(ACLSHMEM_DATA_OP_UDMA));
    EXPECT_FALSE(aclshmemi_user_buffer_heap_engine_supported(
        static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA)));
#endif
}

TEST(UserBufferHeapTest, PublicAttrValidationPrecedesBootstrap)
{
    aclshmemx_buffer_optional_attr_t optionalAttr{};
    optionalAttr.mem_handle = reinterpret_cast<aclrtDrvMemHandle>(static_cast<uintptr_t>(1));
    aclshmemx_buffer_desc_t buffer{};
    buffer.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(ACLSHMEM_PAGE_SIZE));
    buffer.size = ACLSHMEM_PAGE_SIZE;
    buffer.optional_attr = &optionalAttr;

    aclshmemx_init_attr_t attributes{};
    attributes.my_pe = 0;
    attributes.n_pes = 1;
    attributes.local_mem_size = ACLSHMEM_PAGE_SIZE - 1;
    EXPECT_EQ(
        aclshmemx_init_attr_with_buffers(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes, &buffer, 1), ACLSHMEM_INVALID_PARAM);

    attributes.local_mem_size = ACLSHMEM_PAGE_SIZE;
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
    EXPECT_EQ(
        aclshmemx_init_attr_with_buffers(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes, &buffer, 1), ACLSHMEM_NOT_SUPPORTED);
    EXPECT_EQ(aclshmemx_init_status(), ACLSHMEM_STATUS_NOT_INITIALIZED);
}

void RunTwoPeMixedHeap(int rankId, int rankCount, uint64_t localMemSize, bool provideMemHandle)
{
    const int deviceId = rankId % test_gnpu_num + test_first_npu;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(deviceId), ACL_SUCCESS);
    int32_t logicDeviceId = -1;
    ASSERT_EQ(aclrtGetLogicDevIdByUserDevId(deviceId, &logicDeviceId), ACL_SUCCESS);

    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    size_t granularity = 0;
    ASSERT_EQ(aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity), ACL_SUCCESS);
    const uint64_t externalBytes = ((ACLSHMEM_PAGE_SIZE + granularity - 1) / granularity) * granularity;
    aclrtDrvMemHandle inputHandle = nullptr;
    ASSERT_EQ(aclrtMallocPhysical(&inputHandle, externalBytes, &prop, 0), ACL_SUCCESS);
    void* sourceAddress = nullptr;
    ASSERT_TRUE(CreateSourceMapping(inputHandle, externalBytes, logicDeviceId, &sourceAddress));

    aclshmemx_buffer_desc_t buffer{};
    aclshmemx_buffer_optional_attr_t optionalAttr{};
    buffer.addr = sourceAddress;
    buffer.size = externalBytes;
    if (provideMemHandle) {
        optionalAttr.mem_handle = inputHandle;
        buffer.optional_attr = &optionalAttr;
    }
    aclshmemx_init_attr_t attributes{};
    ASSERT_EQ(test_set_attr(rankId, rankCount, localMemSize, test_global_ipport, &attributes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclshmemx_set_conf_store_tls(false, nullptr, 0), ACLSHMEM_SUCCESS);
    const int initRet = aclshmemx_init_attr_with_buffers(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes, &buffer, 1);
    EXPECT_EQ(initRet, ACLSHMEM_SUCCESS);
    if (initRet != ACLSHMEM_SUCCESS) {
        DestroySourceMapping(sourceAddress);
        (void)aclrtFreePhysical(inputHandle);
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
        return;
    }

    auto* heapBase = static_cast<uint8_t*>(aclshmemx_get_buffer_ptr(sourceAddress));
    ASSERT_NE(heapBase, nullptr);
    EXPECT_EQ(aclshmemx_get_buffer_ptr(static_cast<uint8_t*>(sourceAddress) + sizeof(int)), heapBase + sizeof(int));
    EXPECT_EQ(aclshmemx_get_buffer_ptr(static_cast<uint8_t*>(sourceAddress) + externalBytes), nullptr);
    const int localExternalValue = 100 + rankId;
    ASSERT_EQ(
        aclrtMemcpy(
            heapBase, sizeof(localExternalValue), &localExternalValue, sizeof(localExternalValue),
            ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);

    auto* tailAllocation = static_cast<int*>(aclshmem_malloc(sizeof(int)));
    ASSERT_NE(tailAllocation, nullptr);
    const int localTailValue = 200 + rankId;
    ASSERT_EQ(
        aclrtMemcpy(
            tailAllocation, sizeof(localTailValue), &localTailValue, sizeof(localTailValue), ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    ASSERT_EQ(aclshmemi_control_barrier_all(), ACLSHMEM_SUCCESS);

    const int peer = (rankId + 1) % rankCount;
    auto* remoteExternal = aclshmem_ptr(heapBase, peer);
    auto* remoteTail = aclshmem_ptr(tailAllocation, peer);
    ASSERT_NE(remoteExternal, nullptr);
    ASSERT_NE(remoteTail, nullptr);
    int remoteExternalValue = 0;
    int remoteTailValue = 0;
    ASSERT_EQ(
        aclrtMemcpy(
            &remoteExternalValue, sizeof(remoteExternalValue), remoteExternal, sizeof(remoteExternalValue),
            ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(
            &remoteTailValue, sizeof(remoteTailValue), remoteTail, sizeof(remoteTailValue), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(remoteExternalValue, 100 + peer);
    EXPECT_EQ(remoteTailValue, 200 + peer);

    aclshmem_free(tailAllocation);
    if (!provideMemHandle) {
        // SHMEM retained its own reference from sourceAddress. Drop the caller's
        // mapping and original handle first; finalize releases the retained one.
        DestroySourceMapping(sourceAddress);
        EXPECT_EQ(aclrtFreePhysical(inputHandle), ACL_SUCCESS);
        inputHandle = nullptr;
    }
    EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_get_buffer_ptr(sourceAddress), nullptr);
    if (provideMemHandle) {
        DestroySourceMapping(sourceAddress);
        EXPECT_EQ(aclrtFreePhysical(inputHandle), ACL_SUCCESS);
    }
    EXPECT_EQ(aclrtResetDevice(deviceId), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}

void RunTwoPeDefaultHandleMixedHeap(int rankId, int rankCount, uint64_t localMemSize)
{
    RunTwoPeMixedHeap(rankId, rankCount, localMemSize, true);
}

void RunTwoPeVaOnlyMixedHeap(int rankId, int rankCount, uint64_t localMemSize)
{
    RunTwoPeMixedHeap(rankId, rankCount, localMemSize, false);
}

TEST(UserBufferHeapTest, TwoPeDefaultHandleMapsExternalAndTailAcrossCards)
{
    if (test_global_ranks != 2 || test_gnpu_num < 2) {
        GTEST_SKIP() << "该用例需要以 2 PE、至少 2 张 NPU 启动";
    }
    test_mutil_task(RunTwoPeDefaultHandleMixedHeap, ACLSHMEM_PAGE_SIZE, 2);
}

TEST(UserBufferHeapTest, TwoPeVaOnlyMapsExternalAndTailAcrossCards)
{
    if (test_global_ranks != 2 || test_gnpu_num < 2) {
        GTEST_SKIP() << "该用例需要以 2 PE、至少 2 张 NPU 启动";
    }
    test_mutil_task(RunTwoPeVaOnlyMixedHeap, ACLSHMEM_PAGE_SIZE, 2);
}

TEST(UserBufferHeapTest, SinglePeMixedHeapInitMallocAndFinalize)
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        GTEST_SKIP() << "本机没有可用的 NPU Runtime，跳过 mixed-heap 真机 smoke";
    }
    int32_t logicDeviceId = -1;
    ASSERT_EQ(aclrtGetLogicDevIdByUserDevId(0, &logicDeviceId), ACL_SUCCESS);

    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    size_t granularity = 0;
    ASSERT_EQ(aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity), ACL_SUCCESS);
    const uint64_t externalBytes = ((ACLSHMEM_PAGE_SIZE + granularity - 1) / granularity) * granularity;
    aclrtDrvMemHandle inputHandle = nullptr;
    if (aclrtMallocPhysical(&inputHandle, externalBytes, &prop, 0) != ACL_SUCCESS) {
        (void)aclrtResetDevice(0);
        (void)aclFinalize();
        GTEST_SKIP() << "本机不能创建 physical allocation，跳过 mixed-heap 真机 smoke";
    }
    void* sourceAddress = nullptr;
    if (!CreateSourceMapping(inputHandle, externalBytes, logicDeviceId, &sourceAddress)) {
        (void)aclrtFreePhysical(inputHandle);
        (void)aclrtResetDevice(0);
        (void)aclFinalize();
        GTEST_SKIP() << "本机不能建立 source VA mapping，跳过 user-buffer-heap 真机 smoke";
    }

    aclshmemx_buffer_desc_t buffer{};
    aclshmemx_buffer_optional_attr_t optionalAttr{};
    optionalAttr.mem_handle = inputHandle;
    buffer.addr = sourceAddress;
    buffer.size = externalBytes;
    buffer.optional_attr = &optionalAttr;
    aclshmemx_init_attr_t attributes{};
    ASSERT_EQ(test_set_attr(0, 1, ACLSHMEM_PAGE_SIZE, test_global_ipport, &attributes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(aclshmemx_set_conf_store_tls(false, nullptr, 0), ACLSHMEM_SUCCESS);

    const int initRet = aclshmemx_init_attr_with_buffers(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes, &buffer, 1);
    EXPECT_EQ(initRet, ACLSHMEM_SUCCESS);
    if (initRet == ACLSHMEM_SUCCESS) {
        auto* heapBase = static_cast<uint8_t*>(aclshmemx_get_buffer_ptr(sourceAddress));
        EXPECT_NE(heapBase, nullptr);
        EXPECT_EQ(
            aclshmemx_get_buffer_ptr(static_cast<uint8_t*>(sourceAddress) + externalBytes - 1),
            heapBase + externalBytes - 1);
        EXPECT_EQ(aclshmemx_get_buffer_ptr(static_cast<uint8_t*>(sourceAddress) + externalBytes), nullptr);
        void* allocation = aclshmem_malloc(ACLSHMEM_PAGE_SIZE);
        EXPECT_NE(allocation, nullptr);
        if (allocation != nullptr && heapBase != nullptr) {
            EXPECT_GE(static_cast<uint8_t*>(allocation), heapBase + externalBytes);
            aclshmem_free(allocation);
        }
        EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    }

    DestroySourceMapping(sourceAddress);
    EXPECT_EQ(aclrtFreePhysical(inputHandle), ACL_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}
#endif

} // namespace
