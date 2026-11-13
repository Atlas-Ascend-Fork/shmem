/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SIMT_INSTANCE_H
#define SHMEM_DEVICE_SIMT_INSTANCE_H

#include "device_simt/shmem_simt_common_types.h"
#include "device_simt/gm2gm/engine/shmem_device_simt_mte.h"

namespace simt {
/**
 * @brief Select the instance used by SIMT MTE operations in this block.
 * Requires MULTI_INSTANCE. Call inside every VF before any SHMEM operation,
 * including operations on instance 0. All block threads must participate in
 * the same order with the same initialized instance ID. The call synchronizes
 * the block before and after selection; do not call it from a divergent branch
 * or from just one warp. Valid HBM instance IDs are 0..255.
 *
 * Selection is shared by the block, independent of Host/SIMD selectors. No state
 * is inherited across VF invocations. Currently only SIMT MTE multi-instance
 * operation is supported; SIMT UDMA multi-instance operation is not yet supported.
 * Keep the instance and memory
 * alive until communication completes. Without MULTI_INSTANCE, existing SIMT
 * interfaces continue to use instance 0 without a set call.
 */
__simt_callee__ inline void aclshmemx_instance_ctx_set(uint64_t instance_id);

/**
 * @brief Return the selected device state address, matching the AIV getter.
 * Requires MULTI_INSTANCE and a completed set in this VF. Returns an address,
 * not an instance ID. The caller must not modify the returned state.
 */
__simt_callee__ inline uint64_t aclshmemx_instance_ctx_get();
} // namespace simt
#endif
