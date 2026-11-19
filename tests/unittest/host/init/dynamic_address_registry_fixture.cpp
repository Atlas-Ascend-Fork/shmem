/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include "host_device/shmemi_host_meta.h"
#include "dl_acl_api.h"

namespace {
// Test checks must execute in Release too, including calls with side effects.
void Check(bool condition, int line)
{
    if (!condition) {
        std::cerr << "Registry fixture check failed at line " << line << "\n";
        std::exit(EXIT_FAILURE);
    }
}
struct Module {
    uint64_t slots[2] = {};
    bool loaded = true;
};
int contexts[3]; // Contexts 0 and 1 share device 0; Context 2 uses device 1.
thread_local aclrtContext current = nullptr;
int calls = 0;
int symbolQueries = 0;
int failCopies = 0;
int failQueries = 0;
bool available = true;
constexpr int injected = 1234;
Module first, second, late, extra;
const void* failSymbol = nullptr;
bool nullAddress = false;
int reloadWarnings = 0;
bool queryErrorLogged = false;
bool copyErrorLogged = false;
bool slotErrorLogged = false;
} // namespace
void CaptureMetaLog(const char* level, const std::string& message)
{
    if (message.find("META_MODULE_RELOAD_SUSPECT") != std::string::npos) {
        Check(std::strcmp(level, "WARN") == 0, __LINE__);
        ++reloadWarnings;
    }
    if (std::strcmp(level, "ERROR") == 0) {
        queryErrorLogged |= message.find("aclrtGetSymbolAddress failed") != std::string::npos;
        copyErrorLogged |= message.find("aclrtMemcpy failed") != std::string::npos;
        slotErrorLogged |= message.find("returned null") != std::string::npos;
    }
    std::cerr << level << " " << message << "\n";
}
int aclrtGetDevice(int32_t* device)
{
    ++calls;
    *device = current == &contexts[2] ? 1 : 0;
    return current ? ACL_SUCCESS : ACL_ERROR_RT_CONTEXT_NULL;
}
int aclrtMemcpy(void* dest, size_t capacity, const void* source, size_t count, int kind)
{
    ++calls;
    Check(current && capacity >= count && count == 8 && kind == ACL_MEMCPY_HOST_TO_DEVICE, __LINE__);
    if (failCopies > 0) {
        --failCopies;
        return injected;
    }
    std::memcpy(dest, source, count);
    return ACL_SUCCESS;
}
int shm::DlAclApi::AclrtGetSymbolAddress(const void* symbol, void** address)
{
    if (!available)
        return injected; // Unloaded API wrapper rejects the call.
    ++calls;
    ++symbolQueries;
    Check(current, __LINE__);
    auto module = static_cast<const Module*>(symbol);
    Check(module->loaded, __LINE__); // Never query a retired/unloaded module.
    if (failQueries > 0 && (!failSymbol || failSymbol == symbol)) {
        --failQueries;
        return injected;
    }
    *address = nullAddress ? nullptr : const_cast<uint64_t*>(&module->slots[current == &contexts[2] ? 1 : 0]);
    return ACL_SUCCESS;
}
int main()
{
    // Early registration, including duplicates, must make no ACL calls.
    Check(aclshmemi_meta_register(&first) == 0, __LINE__);
    Check(aclshmemi_meta_register(&first) == 0, __LINE__);
    Check(aclshmemi_meta_register(&second) == 0, __LINE__);
    Check(calls == 0, __LINE__);
    Check(aclshmemi_meta_register(nullptr) != 0, __LINE__);
    Check(aclshmemi_meta_publish(100) != 0, __LINE__);

    current = &contexts[0];
    available = false;
    Check(aclshmemi_meta_publish(100) != 0, __LINE__);
    available = true;
    Check(aclshmemi_meta_publish(100) == 0, __LINE__);
    Check(first.slots[0] == 100 && second.slots[0] == 100, __LINE__);
    Check(first.slots[1] == 0, __LINE__);
    Check(symbolQueries == 2, __LINE__); // One lookup per registered symbol.
    Check(reloadWarnings == 0, __LINE__);

    Check(aclshmemi_meta_register(&late) == 0 && late.slots[0] == 100, __LINE__);

    // A different Context on the same device sees the same symbol slot.
    current = &contexts[1];
    Check(late.slots[1] == 0, __LINE__);
    Check(aclshmemi_meta_publish(100) == 0, __LINE__);
    Check(aclshmemi_meta_register(&extra) == 0, __LINE__);
    Check(current == &contexts[1] && extra.slots[0] == 100 && extra.slots[1] == 0, __LINE__);
    aclshmemi_meta_unregister(&extra);
    failQueries = 1;
    Check(aclshmemi_meta_register(&extra) == injected, __LINE__);
    Check(current == &contexts[1], __LINE__);
    // Reject another device before looking up or modifying any symbol there.
    current = &contexts[2];
    int beforeWrongDevice = symbolQueries;
    Check(aclshmemi_meta_publish(0) == ACL_ERROR_INVALID_PARAM, __LINE__);
    Check(aclshmemi_meta_register(&extra) == ACL_ERROR_INVALID_PARAM, __LINE__);
    Check(symbolQueries == beforeWrongDevice && extra.slots[1] == 0, __LINE__);
    current = nullptr;
    Check(aclshmemi_meta_register(&extra) == ACL_ERROR_RT_CONTEXT_NULL, __LINE__);
    Check(current == nullptr, __LINE__);
    current = &contexts[0];
    nullAddress = true;
    Check(aclshmemi_meta_register(&extra) == ACL_ERROR_INVALID_PARAM, __LINE__);
    nullAddress = false;

    // Fail after earlier slots changed, then fail rollback and retry publication.
    failSymbol = &late;
    failQueries = 2;
    Check(aclshmemi_meta_publish(200) == injected, __LINE__);
    Check(first.slots[0] == 100 && second.slots[0] == 100, __LINE__);
    Check(aclshmemi_meta_register(&extra) != 0, __LINE__); // rollback was incomplete
    failSymbol = nullptr;
    Check(aclshmemi_meta_publish(200) == 0, __LINE__);
    Check(first.slots[0] == 200 && second.slots[0] == 200 && late.slots[0] == 200, __LINE__);
    failCopies = 1;
    Check(aclshmemi_meta_publish(300) == injected, __LINE__);
    Check(first.slots[0] == 200 && second.slots[0] == 200 && late.slots[0] == 200, __LINE__);
    Check(aclshmemi_meta_register(&extra) == 0 && extra.slots[0] == 200, __LINE__);
    aclshmemi_meta_unregister(&extra);

    // Retirement failure retains the owning device until cleanup succeeds.
    failQueries = 2;
    Check(aclshmemi_meta_publish(0) != 0, __LINE__);
    current = &contexts[2];
    Check(aclshmemi_meta_publish(400) != 0, __LINE__);
    current = &contexts[1]; // Retry from a different Context on the same device.
    Check(aclshmemi_meta_publish(0) == 0, __LINE__);
    Check(first.slots[0] == 0 && second.slots[0] == 0 && late.slots[0] == 0, __LINE__);

    // Reinitialization republishes every registered module.
    int warningsBeforeReinit = reloadWarnings;
    current = &contexts[2];
    Check(aclshmemi_meta_publish(500) == 0, __LINE__);
    Check(first.slots[1] == 500 && second.slots[1] == 500 && late.slots[1] == 500, __LINE__);
    Check(first.slots[0] == 0, __LINE__);
    Check(reloadWarnings == warningsBeforeReinit, __LINE__); // Ordinary finalize/init is not a module reload.

    // Unregistered modules are excluded from later publication.
    aclshmemi_meta_unregister(&late);
    late.loaded = false;
    Check(aclshmemi_meta_publish(600) == 0, __LINE__);
    Check(aclshmemi_meta_publish(0) == 0, __LINE__);
    aclshmemi_meta_unregister(&first);
    aclshmemi_meta_unregister(&second);
    int warningsBeforeReload = reloadWarnings;
    int before = calls;
    Check(aclshmemi_meta_register(&second) == 0, __LINE__);
    Check(reloadWarnings == warningsBeforeReload + 1, __LINE__);
    Check(calls == before, __LINE__); // Diagnostic history does not introduce ACL calls before init.
    Check(aclshmemi_meta_register(&second) == 0, __LINE__);
    Check(reloadWarnings == warningsBeforeReload + 1, __LINE__); // An active duplicate is not a reload.
    aclshmemi_meta_unregister(&second);
    Check(queryErrorLogged && copyErrorLogged && slotErrorLogged, __LINE__);
    std::cout << "META_REGISTRY_LIFECYCLE_PASS\n";
}
