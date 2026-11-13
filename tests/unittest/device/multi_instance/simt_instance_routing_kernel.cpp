/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#define MULTI_INSTANCE
#include "shmem.h"

// One AIV/VF per rank; each receive span has one ring predecessor.
// Source [0,4096), receive [4096,8192), guard [8192,12288), in int32 elements.
// SIMT direct GM load/store; no UB staging or UDMA queue.
// Completion: VF return -> stream sync -> host control barrier -> D2H.
// No device collective and no AIV-to-AIV shared storage in this routing test.
constexpr uint32_t SLOT = 4096;

__simt_vf__ __launch_bounds__(128) inline void simt_copy(
    __gm__ int32_t* heap, uint32_t count, int peer, uint64_t instance_id, __gm__ int32_t* identity)
{
    simt::aclshmemx_instance_ctx_set(instance_id);
    if (threadIdx.x == 0) {
        identity[0] = simt::aclshmem_my_pe();
        identity[1] = simt::aclshmem_n_pes();
    }
    simt::aclshmemx_int32_put_nbi_block(heap + SLOT, heap, count, peer);
}

__global__ __vector__ void instance_simt_kernel(
    __gm__ int32_t* heap, uint32_t count, int peer, uint64_t instance_id, __gm__ int32_t* identity)
{
    // Deliberately do not set an AIV context: the VF must use its explicit ID.
    asc_vf_call<simt_copy>(dim3(128), heap, count, peer, instance_id, identity);
}

extern "C" void simt_instance_routing_transfer(
    void* stream, void* heap, uint32_t count, int peer, uint64_t id, void* identity)
{
    instance_simt_kernel<<<1, nullptr, stream>>>((int32_t*)heap, count, peer, id, (int32_t*)identity);
}
