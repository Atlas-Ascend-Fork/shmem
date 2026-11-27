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
#include "shmem.h"
#include "udma_put_value_reduce_kernel.h"

constexpr uint64_t INIT_DUMP_SIZE = 200 * 1024 * 1024;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void udma_put_value_reduce_kernel(
    GM_ADDR gva, GM_ADDR dump)
{
#if ASCENDC_DUMP == 1
    AscendC::InitDump(false, dump, INIT_DUMP_SIZE);
#endif

    const int32_t my_pe = aclshmem_my_pe();
    const int32_t pe_size = aclshmem_n_pes();
    const uint64_t region_offset = static_cast<uint64_t>(my_pe) * REGIONS_PER_PE * SLOT_BYTES;
    __gm__ int32_t* src = reinterpret_cast<__gm__ int32_t*>(gva + region_offset);
    __gm__ int32_t* sum_dst = src + ELEMS_PER_SLOT;
    __gm__ int32_t* max_dst = src + 2 * ELEMS_PER_SLOT;
    __gm__ int32_t* min_dst = src + 3 * ELEMS_PER_SLOT;
    __gm__ uint64_t* value_dst = reinterpret_cast<__gm__ uint64_t*>(src + 4 * ELEMS_PER_SLOT);
    const uint64_t value = VALUE_MAGIC | static_cast<uint64_t>(my_pe);

    for (int32_t peer = 0; peer < pe_size; peer++) {
        if (peer == my_pe) {
            continue;
        }
        // Inline write: the value travels in the WRITE WQE, so no local source buffer is needed.
        aclshmemx_udma_put_value_nbi<uint64_t>(value_dst, value, peer);
        aclshmemx_udma_quiet(peer);

        // Vector reduce: the remote UDMA engine combines src with the destination elements.
        aclshmemx_udma_put_reduce_nbi<int32_t, ACLSHMEMX_REDUCE_ADD>(sum_dst, src, ELEMS_PER_SLOT, peer);
        aclshmemx_udma_quiet(peer);
        aclshmemx_udma_put_reduce_nbi<int32_t, ACLSHMEMX_REDUCE_MAX>(max_dst, src, ELEMS_PER_SLOT, peer);
        aclshmemx_udma_quiet(peer);
        aclshmemx_udma_put_reduce_nbi<int32_t, ACLSHMEMX_REDUCE_MIN>(min_dst, src, ELEMS_PER_SLOT, peer);
        aclshmemx_udma_quiet(peer);
    }
}

void launch_udma_put_value_reduce(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dump)
{
    udma_put_value_reduce_kernel<<<block_dim, nullptr, stream>>>(gva, dump);
}
