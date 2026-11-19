/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include "shmem.h"
#include "host_device/shmem_common_macros.h"
#include "unittest_main_test.h"

#if ACLSHMEM_DYNAMIC_META
namespace {
#define CHECK(call)                                    \
    do {                                               \
        int rc = (call);                               \
        if (rc) {                                      \
            std::printf("FAIL %s rc=%d\n", #call, rc); \
            std::exit(1);                              \
        }                                              \
    } while (0)
constexpr size_t bytes = 128 * 64;
constexpr uint64_t fixedMeta = 0x180000000000ULL - (1ULL << 30) - (32ULL << 20);
using Reader = void (*)(aclrtStream, void*, int, bool);
struct Module {
    void* handle;
    Reader read;
    uint64_t symbol = 0;
};
Module Load(int id)
{
    char path[4096];
    std::snprintf(path, sizeof(path), ACLSHMEM_DYNAMIC_TEST_MODULE_DIR "/libaclshmem_dynamic_reader%d.so", id);
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::puts(dlerror());
        std::exit(2);
    }
    auto read = reinterpret_cast<Reader>(dlsym(h, "meta_read"));
    if (!read)
        std::exit(2);
    return {h, read, 0};
}
uint64_t Verify(Module& module, int id, int rank, int npes, aclrtStream stream, void* output, bool active)
{
    uint64_t data[bytes / sizeof(uint64_t)];
    uint64_t root = 0;
    for (int reader = 0; reader < 2; ++reader) {
        CHECK(aclrtMemset(output, bytes, 0xff, bytes));
        module.read(stream, output, reader, active);
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(data, bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (int side = reader == 0 ? 1 : 0; side < 2; ++side) {
            for (int core = 0; core < (reader == 1 && side == 1 ? 16 : 8); ++core) {
                auto p = data + (side * 64 + core) * 8;
                if (root == 0)
                    root = p[0];
                if (module.symbol == 0)
                    module.symbol = p[3];
                bool valid = p[0] == root && p[1] == uint64_t(active ? rank : 0) &&
                             p[2] == uint64_t(active ? npes : 0) && p[3] == module.symbol && p[4] == uint64_t(id);
                valid = valid && (active ? root != 0 && root != fixedMeta : root == 0);
                if (!valid) {
                    std::printf(
                        "MISMATCH module=%d reader=%d side=%d core=%d root=%lx rank=%lu npes=%lu\n", id, reader, side,
                        core, p[0], p[1], p[2]);
                    std::exit(3);
                }
            }
        }
    }
    std::printf("PASS rank=%d module=%d active=%d root=%lx symbol=%lx\n", rank, id, active, root, module.symbol);
    return root;
}
bool Transfer(int fd, void* buffer, size_t size, bool send)
{
    auto p = static_cast<char*>(buffer);
    while (size) {
        ssize_t count = send ? write(fd, p, size) : read(fd, p, size);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return false;
        p += count;
        size -= count;
    }
    return true;
}
int Worker(int device, int rank, int npes, int fd)
{
    // Keep child diagnostics when an assertion or the timeout terminates it.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    alarm(120);
    // Supported integration: load every operator before SHMEM initialization.
    auto first = Load(1);
    auto second = Load(2);
    auto multi = Load(3);
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(device));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));
    void* output;
    CHECK(aclrtMalloc(&output, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    // Make the two ranks' allocator histories different; meta bases need not agree.
    void* padding = nullptr;
    if (rank != 0)
        CHECK(aclrtMalloc(&padding, 2ULL << 20, ACL_MEM_MALLOC_HUGE_FIRST));
    void* occupied = mmap(
        reinterpret_cast<void*>(fixedMeta), 32ULL << 20, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
        -1, 0);
    if (occupied == MAP_FAILED) {
        std::perror("reserve old meta address");
        return 4;
    }
    std::printf("HOST_VA_OCCUPIED address=%p bytes=%lu\n", occupied, 32UL << 20);
    for (int cycle = 0; cycle < 2; ++cycle) {
        std::printf("INIT rank=%d device=%d cycle=%d\n", rank, device, cycle);
        aclshmemx_uniqueid_t instanceUid = ACLSHMEM_UNIQUEID_INITIALIZER;
        CHECK(aclshmemx_get_uniqueid(&instanceUid));
        aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;
        if (rank == 0) {
            CHECK(aclshmemx_get_uniqueid(&uid));
            if (!Transfer(fd, &uid, sizeof(uid), true))
                return 5;
        } else if (!Transfer(fd, &uid, sizeof(uid), false))
            return 5;
        aclshmemx_init_attr_t attr{};
        CHECK(aclshmemx_set_attr_uniqueid_args(rank, npes, 64ULL << 20, &uid, &attr));
        attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
        std::printf("BOOTSTRAP rank=%d cycle=%d\n", rank, cycle);
        CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr));
        auto root = Verify(first, 1, rank, npes, stream, output, true);
        // Query and execute in another Context on the same device. No owner
        // Context switch is performed by SHMEM's publication/cleanup code.
        aclrtContext owner = nullptr;
        aclrtContext loaderContext = nullptr;
        CHECK(aclrtGetCurrentContext(&owner));
        CHECK(aclrtCreateContext(&loaderContext, device));
        CHECK(aclrtSetCurrentContext(loaderContext));
        aclrtStream contextStream;
        CHECK(aclrtCreateStream(&contextStream));
        aclrtContext restored = nullptr;
        CHECK(aclrtGetCurrentContext(&restored));
        if (restored != loaderContext)
            return 12;
        auto peek = reinterpret_cast<int (*)(uint64_t*, void**)>(dlsym(second.handle, "meta_host_root"));
        uint64_t peekRoot = 0;
        void* peekAddress = nullptr;
        if (!peek)
            return 12;
        CHECK(peek(&peekRoot, &peekAddress));
        if (peekRoot != root || peekAddress == nullptr)
            return 12;
        std::printf("HOST other_context root=%lx slot=%p expected=%lx\n", peekRoot, peekAddress, root);
        if (Verify(second, 2, rank, npes, contextStream, output, true) != root)
            return 6;
        CHECK(aclrtSetCurrentContext(owner));
        uint64_t ownerRoot = 0;
        void* ownerAddress = nullptr;
        CHECK(peek(&ownerRoot, &ownerAddress));
        if (ownerRoot != peekRoot || ownerAddress != peekAddress)
            return 12;
        std::printf("SAME_DEVICE_CONTEXT_PASS rank=%d slot=%p root=%lx\n", rank, ownerAddress, ownerRoot);
#if defined(TEST_SIMT)
        auto simtRead =
            reinterpret_cast<void (*)(aclrtStream, void*, uint64_t)>(dlsym(second.handle, "meta_simt_read"));
        if (!simtRead)
            return 10;
        simtRead(stream, output, 0);
        CHECK(aclrtSynchronizeStream(stream));
        uint64_t simtOutput[32 * 4];
        CHECK(aclrtMemcpy(simtOutput, sizeof(simtOutput), output, sizeof(simtOutput), ACL_MEMCPY_DEVICE_TO_HOST));
        for (int tid = 0; tid < 32; ++tid) {
            if (simtOutput[tid * 4] != root || simtOutput[tid * 4 + 1] != uint64_t(rank) ||
                simtOutput[tid * 4 + 2] != uint64_t(npes) || simtOutput[tid * 4 + 3] != root + (64ULL << 10))
                return 10;
        }
        std::printf("SIMT_PASS rank=%d threads=32\n", rank);
#endif
        if (first.symbol == second.symbol)
            return 6;
        // A second, single-PE instance has different rank state from world on
        // rank 1. Its context is two entity slots beyond the default context.
        aclshmemx_init_attr_t instanceAttr{};
        CHECK(aclshmemx_set_attr_uniqueid_args(0, 1, 64ULL << 20, &instanceUid, &instanceAttr));
        instanceAttr.instance_id = 1;
        instanceAttr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
        CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &instanceAttr));
        auto instanceRead =
            reinterpret_cast<void (*)(aclrtStream, void*, uint64_t)>(dlsym(multi.handle, "meta_instance_read"));
        if (!instanceRead)
            return 11;
        for (uint64_t instance = 0; instance < 2; ++instance) {
            instanceRead(stream, output, instance);
            CHECK(aclrtSynchronizeStream(stream));
            uint64_t state[5];
            CHECK(aclrtMemcpy(state, sizeof(state), output, sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST));
            if (state[0] != root || state[1] != uint64_t(instance ? 0 : rank) ||
                state[2] != uint64_t(instance ? 1 : npes) ||
                state[3] != root + (64ULL << 10) + (instance << 1) * (64ULL << 10) || state[4] != (instance << 1)) {
                std::printf(
                    "INSTANCE_MISMATCH rank=%d instance=%lu root=%lx mype=%lu npes=%lu state=%lx\n", rank, instance,
                    state[0], state[1], state[2], state[3]);
                return 11;
            }
#if defined(TEST_SIMT)
            auto simtInstanceRead =
                reinterpret_cast<void (*)(aclrtStream, void*, uint64_t)>(dlsym(multi.handle, "meta_simt_read"));
            if (!simtInstanceRead)
                return 11;
            simtInstanceRead(stream, output, instance);
            CHECK(aclrtSynchronizeStream(stream));
            CHECK(aclrtMemcpy(simtOutput, sizeof(simtOutput), output, sizeof(simtOutput), ACL_MEMCPY_DEVICE_TO_HOST));
            for (int tid = 0; tid < 32; ++tid) {
                for (int field = 0; field < 4; ++field) {
                    if (simtOutput[tid * 4 + field] != state[field])
                        return 11;
                }
            }
#endif
        }
        CHECK(aclshmemx_finalize(1));
        CHECK(aclshmemx_instance_ctx_set(0));
        Verify(first, 1, rank, npes, stream, output, true);
        std::printf("MULTI_INSTANCE_PASS rank=%d\n", rank);
        void* symmetric = aclshmem_malloc(64);
        if (!symmetric)
            return 8;
        CHECK(aclrtMemset(symmetric, 64, 0, 64));
        aclshmem_barrier_all();
        auto communicate = reinterpret_cast<void (*)(aclrtStream, void*)>(dlsym(second.handle, "meta_communicate"));
        if (!communicate)
            return 8;
        communicate(stream, symmetric);
        CHECK(aclrtSynchronizeStream(stream));
        int32_t value = 0;
        CHECK(aclrtMemcpy(&value, sizeof(value), symmetric, sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST));
        if (value != 100 + (rank + npes - 1) % npes)
            return 9;
        aclshmem_free(symmetric);
        // Finalize in the initialization Context: existing SHMEM streams still
        // belong to it. Then verify cleared entries from the other Context.
        CHECK(aclrtSetCurrentContext(owner));
        CHECK(aclshmem_finalize());
        CHECK(aclrtSetCurrentContext(loaderContext));
        Verify(first, 1, rank, npes, contextStream, output, false);
        Verify(second, 2, rank, npes, contextStream, output, false);
        CHECK(aclrtDestroyStream(contextStream));
        CHECK(aclrtSetCurrentContext(owner));
        CHECK(aclrtDestroyContext(loaderContext));
        std::printf("CYCLE_PASS rank=%d cycle=%d communication=%d\n", rank, cycle, value);
    }
    munmap(occupied, 32ULL << 20);
    CHECK(aclrtFree(output));
    CHECK(aclrtDestroyStream(stream));
    if (padding != nullptr)
        CHECK(aclrtFree(padding));
    CHECK(dlclose(multi.handle));
    CHECK(dlclose(second.handle));
    CHECK(dlclose(first.handle));
    CHECK(aclrtResetDevice(device));
    CHECK(aclFinalize());
    if (fd >= 0)
        close(fd);
    alarm(0);
    std::puts("META_SYMBOLS_PASS");
    return 0;
}
int RunTwoRanks(int firstDevice)
{
    int fds[2];
    if (pipe(fds) != 0)
        return 2;
    pid_t children[2];
    for (int rank = 0; rank < 2; ++rank) {
        children[rank] = fork();
        if (children[rank] < 0) {
            close(fds[0]);
            close(fds[1]);
            for (int previous = 0; previous < rank; ++previous) {
                kill(children[previous], SIGKILL);
                waitpid(children[previous], nullptr, 0);
            }
            return 2;
        }
        if (children[rank] == 0) {
            close(fds[rank == 0 ? 0 : 1]);
            _exit(Worker(firstDevice + rank, rank, 2, fds[rank == 0 ? 1 : 0]));
        }
    }
    close(fds[0]);
    close(fds[1]);
    int result = 0;
    for (auto child : children) {
        int status;
        if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status))
            result = 1;
    }
    return result;
}

#undef CHECK
} // namespace
#endif

TEST(DynamicDeviceAddress, ModulesCommunicationAndReinitialization)
{
#if ACLSHMEM_DYNAMIC_META
    if (test_gnpu_num < 2 || test_global_ranks != 2 || test_first_rank != 0) {
        GTEST_SKIP() << "Requires a single-host two-PE run with two available devices";
    }
    ASSERT_EQ(RunTwoRanks(test_first_npu), 0);
#else
    GTEST_SKIP() << "Build with -enable_device_symbol to test dynamic device addresses";
#endif
}
