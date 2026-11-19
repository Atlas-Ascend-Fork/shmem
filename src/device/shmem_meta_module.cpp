/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

// One definition per linked operator binary. Hidden host visibility prevents
// ELF symbol interposition between independent operator shared libraries.
extern "C" {
__attribute__((visibility("hidden"))) __gm__ uint64_t g_meta_root = 0;
}
// Referencing the variable in this translation unit makes the compiler emit
// its Host/Device variable registration. This kernel is never launched.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ __attribute__((visibility("hidden"))) void
aclshmemi_meta_anchor(__gm__ uint64_t* output)
{
    *output = g_meta_root;
}
extern "C" __attribute__((visibility("hidden"))) const void* aclshmemi_meta_module_symbol()
{
    return static_cast<const void*>(&g_meta_root);
}
// Internal archive extraction entry, local to each operator module.
extern "C" __attribute__((visibility("hidden"))) void aclshmemx_device_library_anchor() {}
#if defined(__NPU_HOST__)
// Pull the registrar after this object's compiler-generated registration.
extern "C" __attribute__((visibility("hidden"))) void aclshmemi_meta_registration_link();
__attribute__((used)) static auto const aclshmemi_meta_registration_reference = &aclshmemi_meta_registration_link;
#endif
