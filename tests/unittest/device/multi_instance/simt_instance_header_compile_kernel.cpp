/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#define MULTI_INSTANCE
#include "device_simt/shmem_device_simt_instance.h"

// No umbrella header: the public header must supply its own dependencies.
__simt_vf__ __launch_bounds__(32) inline void simt_instance_header_compile(__gm__ uint64_t* result)
{
    simt::aclshmemx_instance_ctx_set(0);
    if (threadIdx.x == 0) {
        *result = simt::aclshmemx_instance_ctx_get();
    }
}

__global__ __vector__ void simt_instance_header_compile_kernel(__gm__ uint64_t* result)
{
    asc_vf_call<simt_instance_header_compile>(dim3(32), result);
}
