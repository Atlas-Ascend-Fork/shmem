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
#include "kernel_operator.h"
#include "shmem.h"
#include "shmemi_device_meta.h"

__aicore__ inline void Report(__gm__ uint64_t* output, bool active)
{
#if defined(MULTI_INSTANCE)
    aclshmemx_instance_ctx_set(0);
#endif
    uint32_t side = 0;
    if ASCEND_IS_AIV {
        side = 1;
    }
    uint32_t core = AscendC::GetBlockIdx();
    auto slot = output + (side * 64 + core) * 8;
    slot[0] = aclshmemi_get_meta_base();
    slot[1] = active ? aclshmem_my_pe() : 0;
    slot[2] = active ? aclshmem_n_pes() : 0;
    slot[3] = reinterpret_cast<uint64_t>(&g_meta_root);
    slot[4] = MODULE_ID;
    AscendC::GlobalTensor<uint64_t> report;
    report.SetGlobalBuffer(slot);
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        report);
}
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void MetaReadAiv(__gm__ uint64_t* output, bool active)
{
    Report(output, active);
}
extern "C" [[bisheng::core_ratio(1, 2)]] __global__ __aicore__ void MetaReadMix(__gm__ uint64_t* output, bool active)
{
    Report(output, active);
}

extern "C" void meta_read(aclrtStream stream, void* output, int reader, bool active)
{
    if (reader == 0)
        MetaReadAiv<<<8, nullptr, stream>>>((__gm__ uint64_t*)output, active);
    else
        MetaReadMix<<<8, nullptr, stream>>>((__gm__ uint64_t*)output, active);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void MetaCommunicate(__gm__ int32_t* output)
{
#if defined(MULTI_INSTANCE)
    aclshmemx_instance_ctx_set(0);
#endif
    int rank = aclshmem_my_pe();
    aclshmem_int32_p(output, 100 + rank, (rank + 1) % aclshmem_n_pes());
    aclshmemx_barrier_all_vec();
}
extern "C" void meta_communicate(aclrtStream stream, void* output)
{
    MetaCommunicate<<<1, nullptr, stream>>>((__gm__ int32_t*)output);
}

#if defined(USE_SIMT)
__simt_vf__ __launch_bounds__(32) inline void MetaSimt(__gm__ uint64_t* output, uint64_t instance)
{
#if defined(MULTI_INSTANCE)
    simt::aclshmemx_instance_ctx_set(instance);
#endif
    auto tid = static_cast<uint32_t>(threadIdx.x);
    output[tid * 4] = simt::aclshmemi_get_meta_base();
    output[tid * 4 + 1] = simt::aclshmemi_get_my_pe();
    output[tid * 4 + 2] = simt::aclshmemi_get_total_pe();
    output[tid * 4 + 3] = reinterpret_cast<uint64_t>(simt::aclshmemi_get_state());
}
extern "C" __global__ __vector__ void MetaSimtKernel(__gm__ uint64_t* output, uint64_t instance)
{
    asc_vf_call<MetaSimt>(dim3(32), output, instance);
}
extern "C" void meta_simt_read(aclrtStream stream, void* output, uint64_t instance)
{
    MetaSimtKernel<<<1, nullptr, stream>>>((__gm__ uint64_t*)output, instance);
}
#endif

#if defined(MULTI_INSTANCE)
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void MetaInstance(
    __gm__ uint64_t* output, uint64_t instance)
{
    aclshmemx_instance_ctx_set(instance);
    output[0] = aclshmemi_get_meta_base();
    output[1] = aclshmem_my_pe();
    output[2] = aclshmem_n_pes();
    output[3] = reinterpret_cast<uint64_t>(aclshmemi_get_state());
    output[4] = aclshmemi_get_current_entity_id();
    AscendC::GlobalTensor<uint64_t> report;
    report.SetGlobalBuffer(output);
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        report);
}
extern "C" void meta_instance_read(aclrtStream stream, void* output, uint64_t instance)
{
    MetaInstance<<<1, nullptr, stream>>>((__gm__ uint64_t*)output, instance);
}
#endif

extern "C" int meta_host_root(uint64_t* root, void** address)
{
    int ret = aclrtGetSymbolAddress(static_cast<const void*>(&g_meta_root), address);
    if (ret == 0)
        ret = aclrtMemcpy(root, sizeof(*root), *address, sizeof(*root), ACL_MEMCPY_DEVICE_TO_HOST);
    return ret;
}
