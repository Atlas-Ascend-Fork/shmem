/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can use, redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_SIMT_GM2GM_AMO_H
#define SHMEM_DEVICE_SIMT_GM2GM_AMO_H

#include "device_simt/gm2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/shmem_simt_common_types.h"

namespace simt {

/**
 * @brief Atomic compare-and-swap types and names
 *
 * |NAME   | TYPE     |
 * |-------|----------|
 * |int32  | int32_t  |
 * |int64  | int64_t  |
 * |uint32 | uint32_t |
 * |uint64 | uint64_t |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_SIMT(FUNC) \
    FUNC(int32, int32_t);                        \
    FUNC(int64, int64_t);                        \
    FUNC(uint32, uint32_t);                      \
    FUNC(uint64, uint64_t)

/**
 * @brief  Automatically generates aclshmem atomic compare swap functions for different data types.
 *         The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark __simt_callee__ inline TYPE aclshmem_NAME_atomic_compare_swap(\_\_gm\_\_ TYPE *dest, TYPE cond,
 * TYPE value, int32_t pe)
 *
 * @par Function Description
 *    Synchronous interface. Atomically compares the value at dest with cond. If they are equal, the value
 *    at dest is set to value. Returns the old value at dest. Each SIMT thread issues its own atomic
 *    operation on the translated symmetric address, which may reside in the device symmetric heap or in
 *    the HOST_SIDE symmetric heap (D2H).
 *
 * @par Parameters
 * - **dest**   - [in] Pointer to the target data in symmetric memory (device or HOST_SIDE heap).
 * - **cond**   - [in] Value to compare against.
 * - **value**  - [in] Value to set if comparison succeeds.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Return
 *    The old value at dest before the operation.
 *
 * @note Atomicity across SIMT threads of one PE and across PEs on the same symmetric address is provided
 *       by the underlying asc_atomic_cas instruction on the translated address.
 */
#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME_SIMT(NAME, TYPE)         \
    __simt_callee__ inline TYPE aclshmem_##NAME##_atomic_compare_swap( \
        __gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe)

ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_SIMT(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME_SIMT);

#undef ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME_SIMT
#undef ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_SIMT

} // namespace simt

#include "gm2gm/shmem_device_simt_amo.hpp"
#endif // SHMEM_DEVICE_SIMT_GM2GM_AMO_H
