/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_UDMA_QP_KERNELS_H
#define SIMT_UDMA_QP_KERNELS_H

#include <cstddef>
#include <cstdint>

void test_simt_udma_qp_put_warp(
    void* stream, uint8_t* dst, uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx);
void test_simt_udma_qp_get_warp(
    void* stream, uint8_t* dst, uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx);
void test_simt_udma_qp_put_block(
    void* stream, uint8_t* dst, uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx);
void test_simt_udma_qp_get_block(
    void* stream, uint8_t* dst, uint8_t* src, size_t elem_size, int32_t pe, uint32_t qp_idx);

void test_simt_udma_qp_mqp_put_warp(
    void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe);
void test_simt_udma_qp_mqp_get_warp(
    void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe);
void test_simt_udma_qp_mqp_put_block(
    void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe);
void test_simt_udma_qp_mqp_get_block(
    void* stream, uint8_t* dst, uint8_t* src, uint32_t slice, uint32_t qp_count, int32_t pe);

#endif // SIMT_UDMA_QP_KERNELS_H
