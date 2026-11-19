/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "host_device/shmemi_host_meta.h"
#if ACLSHMEM_DYNAMIC_META
#include <algorithm>
#include <array>
#include <mutex>
#include <vector>
#include "dl_acl_api.h"
#include "shmemi_logger.h"

namespace {
struct Registry {
    std::mutex mutex;
    std::vector<const void*> symbols;
    // Bounded diagnostic history only; never query or dereference retired tokens.
    std::array<const void*, 64> retiredSymbols{};
    size_t retiredNext = 0;
    uint64_t root = 0;
    int32_t device = -1;
    bool publicationFailed = false;
};
Registry& State()
{
    // Outlives mixed-code DSO destructors. Registry calls never acquire the
    // SHMEM instance mutex; init/finalize acquire that mutex before this one.
    static Registry* state = new Registry;
    return *state;
}

int BindSymbol(const void* symbol, uint64_t root)
{
    void* address = nullptr;
    int ret = shm::DlAclApi::AclrtGetSymbolAddress(symbol, &address);
    if (ret != ACL_SUCCESS || address == nullptr) {
        SHM_LOG_ERROR(
            "META_BIND_SYMBOL aclrtGetSymbolAddress failed or returned null: token=" << symbol << ", slot=" << address
                                                                                     << ", ret=" << ret);
        return ret != ACL_SUCCESS ? ret : ACL_ERROR_INVALID_PARAM;
    }
    ret = aclrtMemcpy(address, sizeof(root), &root, sizeof(root), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        SHM_LOG_ERROR(
            "META_BIND_SYMBOL aclrtMemcpy failed: token=" << symbol << ", slot=" << address << ", root=0x" << std::hex
                                                          << root << std::dec << ", ret=" << ret);
    } else {
        SHM_LOG_DEBUG(
            "META_BIND_SYMBOL token=" << symbol << ", slot=" << address << ", root=0x" << std::hex << root << std::dec);
    }
    return ret;
}

int CheckDevice(const Registry& state, int32_t* current)
{
    int ret = aclrtGetDevice(current);
    if (ret != ACL_SUCCESS) {
        SHM_LOG_ERROR("META_DEVICE_QUERY aclrtGetDevice failed: expected=" << state.device << ", ret=" << ret);
        return ret;
    }
    if (*current < 0 || (state.device >= 0 && *current != state.device)) {
        SHM_LOG_ERROR("META_DEVICE_MISMATCH current=" << *current << ", expected=" << state.device);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

int BindAll(const Registry& state, uint64_t root, bool bestEffort)
{
    int failure = ACL_SUCCESS;
    for (auto symbol : state.symbols) {
        int ret = BindSymbol(symbol, root);
        if (ret != ACL_SUCCESS) {
            failure = ret;
            if (!bestEffort)
                return ret;
        }
    }
    return failure;
}
} // namespace

int aclshmemi_meta_register(const void* symbol)
{
    if (symbol == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (std::find(state.symbols.begin(), state.symbols.end(), symbol) != state.symbols.end())
        return ACL_SUCCESS;
    if (state.publicationFailed)
        return ACL_ERROR_INVALID_PARAM;
    if (std::find(state.retiredSymbols.begin(), state.retiredSymbols.end(), symbol) != state.retiredSymbols.end()) {
        SHM_LOG_WARN(
            "META_MODULE_RELOAD_SUSPECT token="
            << symbol
            << " was previously unregistered; possible compiler-managed module reload. Runtime may retain an old"
               " symbol mapping. This is not proof of a stale slot. Compare META_BIND_SYMBOL with Runtime plog;"
               " for repeated dlclose/dlopen, consider keeping the module resident or linking -Wl,-z,nodelete.");
    }
    if (state.root != 0) {
        int32_t device = -1;
        int ret = CheckDevice(state, &device);
        if (ret == ACL_SUCCESS)
            ret = BindSymbol(symbol, state.root);
        if (ret != ACL_SUCCESS)
            return ret;
    }
    state.symbols.push_back(symbol);
    return ACL_SUCCESS;
}

void aclshmemi_meta_unregister(const void* symbol)
{
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto found = std::find(state.symbols.begin(), state.symbols.end(), symbol);
    if (found == state.symbols.end())
        return;
    state.retiredSymbols[state.retiredNext] = symbol;
    state.retiredNext = (state.retiredNext + 1) % state.retiredSymbols.size();
    state.symbols.erase(found);
    // Called from DSO destructors, including process exit. Record history only:
    // the logging singleton may already have been destroyed at that point.
}

int aclshmemi_meta_publish(uint64_t root)
{
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    int32_t device = -1;
    int ret = CheckDevice(state, &device);
    if (ret != ACL_SUCCESS)
        return ret;
    // Retain the device on failure so retries cannot clear another device's slots.
    // Do not cache slots across arena cycles.
    state.device = device;
    SHM_LOG_DEBUG(
        "META_PUBLISH device=" << device << ", previous_root=0x" << std::hex << state.root << ", root=0x" << root
                               << std::dec << ", symbols=" << state.symbols.size());
    ret = BindAll(state, root, root == 0);
    if (ret != ACL_SUCCESS) {
        SHM_LOG_ERROR(
            "META_PUBLISH failed: device=" << device << ", root=0x" << std::hex << root << std::dec << ", ret=" << ret);
        int rollback = BindAll(state, state.root, true);
        state.publicationFailed = rollback != ACL_SUCCESS;
        if (rollback != ACL_SUCCESS)
            SHM_LOG_ERROR("meta rollback failed: " << rollback);
        return ret;
    }
    state.root = root;
    state.publicationFailed = false;
    if (root == 0)
        state.device = -1;
    return ACL_SUCCESS;
}
#endif
