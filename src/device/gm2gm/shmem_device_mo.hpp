/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_MO_HPP
#define SHMEM_DEVICE_MO_HPP

#ifndef ACLSHMEM_UDMA_SUPPORTED
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define ACLSHMEM_UDMA_SUPPORTED 1
#else
#define ACLSHMEM_UDMA_SUPPORTED 0
#endif
#endif

#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_sdma.h"
#if ACLSHMEM_UDMA_SUPPORTED
#include "device/gm2gm/engine/shmem_device_udma.h"
#endif
#include "host/shmem_host_def.h"
#include "host_device/shmem_common_types.h"
#include "../../host_device/shmemi_common_types.h"
#include "shmemi_device_common.hpp"
#include "shmemi_device_mo.h"

// Filter the rank-wide completion mask to optional backends this device build
// can quiet. RoCE quiet is available on all supported device builds.
ACLSHMEM_DEVICE uint8_t aclshmemi_filter_unsupported_quiet_transports(uint8_t quiet_transport_mask)
{
#if !ACLSHMEM_TRANSPORT_SDMA_SUPPORTED
    quiet_transport_mask &= static_cast<uint8_t>(~ACLSHMEM_TRANSPORT_SDMA);
#endif
#if !ACLSHMEM_UDMA_SUPPORTED
    quiet_transport_mask &= static_cast<uint8_t>(~ACLSHMEM_TRANSPORT_UDMA);
#endif
    return quiet_transport_mask;
}

ACLSHMEM_DEVICE uint8_t aclshmemi_get_quiet_transport_mask()
{
    __gm__ uint8_t* quiet_state_addr =
        reinterpret_cast<__gm__ uint8_t*>(aclshmemi_get_state()) + sizeof(aclshmem_device_host_state_t);
    return reinterpret_cast<__gm__ aclshmemi_device_quiet_state_t*>(quiet_state_addr)->quiet_transport_mask;
}

ACLSHMEM_DEVICE void aclshmemi_quiet_all_reachable_transports(
    __gm__ aclshmem_device_host_state_t* state, uint8_t quiet_transport_mask, bool include_udma = true)
{
    uint64_t sdma_ub = state->sdma_config.aclshmem_ub;
    uint32_t sdma_ub_size = state->sdma_config.ub_size;
    uint32_t sdma_sync_id = state->sdma_config.sync_id;
    uint64_t rdma_ub = state->rdma_config.aclshmem_ub;
    uint32_t rdma_sync_id = state->rdma_config.sync_id;
    // High-level SDMA RMA uses QP 0, so one quiet drains its requests to all SDMA-reachable PEs.
    if (quiet_transport_mask & ACLSHMEM_TRANSPORT_SDMA) {
        aclshmemx_sdma_quiet(reinterpret_cast<__ubuf__ char*>(sdma_ub), sdma_ub_size, sdma_sync_id);
    }

    for (int pe = 0; pe < state->npes; ++pe) {
        if (pe == state->mype) {
            continue;
        }

        // Per-peer RoCE/UDMA completion is selected solely by the peer topology.
        uint8_t topo = state->topo_list[pe];
        if (topo & ACLSHMEM_TRANSPORT_ROCE) {
            aclshmemx_roce_quiet(pe, reinterpret_cast<__ubuf__ char*>(rdma_ub), rdma_sync_id);
        }

#if ACLSHMEM_UDMA_SUPPORTED
        if (include_udma && (topo & ACLSHMEM_TRANSPORT_UDMA)) {
            aclshmemx_udma_quiet(pe);
        }
#endif
    }
}

ACLSHMEM_DEVICE void aclshmemi_quiet()
{
    // Host-precomputed rank-wide non-MTE completion mask. A zero value means
    // this rank needs no completion beyond MTE, so skip the per-PE backend loop.
    // A nonzero SDMA bit drains the current core queue once; per-peer RoCE/UDMA
    // completion is selected separately from topo_list[pe].
    uint8_t quiet_transport_mask = aclshmemi_filter_unsupported_quiet_transports(aclshmemi_get_quiet_transport_mask());
    AscendC::PipeBarrier<PIPE_ALL>();
    if (quiet_transport_mask != 0) {
        __gm__ aclshmem_device_host_state_t* state = aclshmemi_get_state();
        aclshmemi_quiet_all_reachable_transports(state, quiet_transport_mask);
    }
    dcci_entire_cache();
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE void aclshmem_quiet() { aclshmemi_quiet(); }

ACLSHMEM_DEVICE void aclshmem_fence() { aclshmemi_quiet(); }

#ifdef __cplusplus
}
#endif

#endif
