/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#include "device_simt/gm2gm/shmem_device_simt_amo.h"

// No umbrella header or instance header may hide an AMO dependency.
__simt_vf__ __launch_bounds__(1) inline void simt_amo_header_compile(__gm__ int32_t* target, __gm__ int32_t* previous)
{
    *previous = simt::aclshmem_int32_atomic_compare_swap(target, 1, 2, 0);
}

__global__ __vector__ void simt_amo_header_compile_kernel(__gm__ int32_t* target, __gm__ int32_t* previous)
{
    asc_vf_call<simt_amo_header_compile>(dim3(1), target, previous);
}
