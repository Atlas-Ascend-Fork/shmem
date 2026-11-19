/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_HOST_META_H
#define ACLSHMEMI_HOST_META_H
#include <stdint.h>
#include "host_device/shmem_common_macros.h"
#if ACLSHMEM_DYNAMIC_META
#ifdef __cplusplus
extern "C" {
#endif
// Private module registration; not part of the public SHMEM ABI.
// Compiler-managed mixed-code modules supply their compiler-registered Host variable token.
int aclshmemi_meta_register(const void* symbol);
void aclshmemi_meta_unregister(const void* symbol);
int aclshmemi_meta_publish(uint64_t root);

#ifdef __cplusplus
}
#endif

#endif
#endif
