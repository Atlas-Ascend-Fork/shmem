/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PERFTEST_ERROR_CHECK_H
#define PERFTEST_ERROR_CHECK_H

#include <iostream>

/**
 * @brief Evaluate @p call and jump to @p label on failure, recording the error code in @p ret_var.
 * @param call Expression whose return value is compared against 0 (acl/aclshmem style error codes).
 * @param ret_var Variable receiving the error code on failure.
 * @param label Cleanup label to jump to on failure.
 */
#define PERFTEST_CHECK_GOTO(call, ret_var, label)                   \
    do {                                                            \
        auto _ret = (call);                                         \
        if (_ret != 0) {                                            \
            std::cerr << #call << " failed: " << _ret << std::endl; \
            ret_var = _ret;                                         \
            goto label;                                             \
        }                                                           \
    } while (0)

/**
 * @brief Evaluate @p call in a cleanup path, recording the error code only if @p ret_var is still 0.
 * @param call Expression whose return value is compared against 0 (acl/aclshmem style error codes).
 * @param ret_var Variable receiving the first error code.
 */
#define PERFTEST_CHECK_CLEANUP(call, ret_var)                       \
    do {                                                            \
        auto _ret = (call);                                         \
        if (_ret != 0) {                                            \
            std::cerr << #call << " failed: " << _ret << std::endl; \
            if (ret_var == 0) {                                     \
                ret_var = _ret;                                     \
            }                                                       \
        }                                                           \
    } while (0)

#endif /* PERFTEST_ERROR_CHECK_H */
