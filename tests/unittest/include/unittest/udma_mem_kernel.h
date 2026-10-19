/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UDMA_MEM_KERNEL_H
#define UDMA_MEM_KERNEL_H

#include <cstdint>

// Contiguous acceptance matrix:
// three families x pointer/GlobalTensor x sync/NBI. The host launches this
// matrix once for SIGNAL_SET and once for SIGNAL_ADD.
constexpr uint32_t UDMA_SIGNAL_SLOT_COUNT = 12;
constexpr uint32_t UDMA_SIGNAL_SLOT_BYTES = 512;
constexpr uint32_t UDMA_SIGNAL_PREFIX_BYTES = 32;
constexpr int32_t UDMA_SIGNAL_INITIAL = 17;
constexpr uint8_t UDMA_SIGNAL_GUARD = 0xA5;
constexpr uint32_t UDMA_SIGNAL_COPY_BYTES = 8;

// Non-contiguous Tensor signal coverage: four slots exercise sync/NBI and
// SIGNAL_SET/SIGNAL_ADD while leaving row gaps and guard bytes visible.
constexpr uint32_t UDMA_SIGNAL_NONCONTIG_SLOT_COUNT = 4;
constexpr uint32_t UDMA_SIGNAL_NONCONTIG_SLOT_BYTES = 512;
constexpr uint32_t UDMA_SIGNAL_NONCONTIG_PREFIX_BYTES = 32;
constexpr uint32_t UDMA_SIGNAL_NONCONTIG_REPEAT = 4;
constexpr uint32_t UDMA_SIGNAL_NONCONTIG_LENGTH = 64;
constexpr uint32_t UDMA_SIGNAL_NONCONTIG_LD = 128;

constexpr uint32_t UDMA_SIGNAL_SPLIT_PUTMEM = 0;
constexpr uint32_t UDMA_SIGNAL_SPLIT_SIZE = 1;
constexpr uint32_t UDMA_SIGNAL_SPLIT_TYPENAME = 2;
constexpr uint32_t UDMA_SIGNAL_SPLIT_FAMILY_COUNT = 3;

#define UDMA_SIGNAL_ACCEPTANCE_SLOT_IS_NBI(SLOT) ((SLOT) % 2 == 1)
void test_udma_put(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_get(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_op_config_put_get(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_put_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_put_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_get_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_get_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_op_config_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_put_signal_sq_wrap(uint32_t block_dim, void* stream, uint8_t* workspace);
void test_udma_highlevel_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_signal_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_size_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_size_signal_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_signal_overloads(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr, bool use_add);
void test_udma_highlevel_put_signal_noncontiguous_tensor(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_signal_split(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr, uint64_t elem_size, int32_t signal,
    uint32_t family, bool nbi, bool use_tensor);
void test_udma_highlevel_local_rma(uint32_t block_dim, void* stream, uint8_t* symmetric, uint64_t config);

#endif // UDMA_MEM_KERNEL_H
