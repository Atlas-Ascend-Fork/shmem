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
#include "barrier_all_kernel.h"

namespace barrier_all_demo {

extern "C" [[bisheng::core_ratio(1, 2)]] __global__ __aicore__ void device_barrier_all_demo(
    uint64_t fftsAddr, GM_ADDR buffer)
{
    util_set_ffts_config(fftsAddr);
    int pe = aclshmem_my_pe();
    int n_pes = aclshmem_n_pes();
    AscendC::PipeBarrier<PIPE_ALL>();

    // Only AIV sub-block 0 submits the puts; other AIV sub-blocks and AIC
    // still participate in the collective barrier below.
    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() == 0) {
            auto data = reinterpret_cast<__gm__ uint8_t*>(buffer);
            // The high-level API selects MTE for same-machine PEs and RDMA for cross-machine PEs.
            for (int peer = 0; peer < n_pes; ++peer) {
                if (peer == pe) {
                    continue;
                }
                aclshmem_putmem_nbi(data + pe * kSlotBytes, data + pe * kSlotBytes, kSlotBytes, peer);
            }
        }
    }

    aclshmem_barrier_all();
}

void launch_barrier_all(void* stream, uint64_t fftsAddr, uint32_t* buffer)
{
    device_barrier_all_demo<<<1, nullptr, stream>>>(fftsAddr, reinterpret_cast<uint8_t*>(buffer));
}
} // namespace barrier_all_demo
