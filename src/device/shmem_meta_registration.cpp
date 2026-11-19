/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdio>
#include <cstdlib>
#include "host_device/shmemi_host_meta.h"
extern "C" __attribute__((visibility("hidden"))) const void* aclshmemi_meta_module_symbol();
// Referenced by the mixed-code archive member to retain this constructor-only
// object during ordinary static-library extraction.
extern "C" __attribute__((visibility("hidden"))) void aclshmemi_meta_registration_link() {}
namespace {
// Linked after the mixed-code object: the compiler's device-symbol constructor
// must finish before a late-loaded module publishes the currently active root.
struct ModuleRegistration {
    ModuleRegistration()
    {
        int ret = aclshmemi_meta_register(aclshmemi_meta_module_symbol());
        if (ret != 0) {
            std::fprintf(stderr, "SHMEM meta module registration failed: %d\n", ret);
            std::abort();
        }
    }
    ~ModuleRegistration() { aclshmemi_meta_unregister(aclshmemi_meta_module_symbol()); }
};
ModuleRegistration registration;
} // namespace
