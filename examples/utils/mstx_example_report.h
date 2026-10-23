/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_UTILS_MSTX_EXAMPLE_REPORT_H
#define EXAMPLES_UTILS_MSTX_EXAMPLE_REPORT_H

#ifdef __MSTX_DFX_REPORT__

#include <cstdint>

#include "sanitizer_report.h"

#define EXAMPLE_MSTX_FUSE_SCOPE_START() Sanitizer::SanitizerFuseScopeStart()

#define EXAMPLE_MSTX_FUSE_SCOPE_END() Sanitizer::SanitizerFuseScopeEnd()

#define EXAMPLE_MSTX_CROSS_NPU_BARRIER_REPORT(device_num_, core_num_, aiv_only_, pipe_barrier_all_) \
    do {                                                                                            \
        Sanitizer::MstxCrossNpuBarrier mstx_barrier__{};                                            \
        mstx_barrier__.usedDeviceNum = (device_num_);                                               \
        mstx_barrier__.usedCoreNum = (core_num_);                                                   \
        mstx_barrier__.isAIVOnly = (aiv_only_);                                                     \
        mstx_barrier__.pipeBarrierAll = (pipe_barrier_all_);                                        \
        Sanitizer::SanitizerReport(mstx_barrier__);                                                 \
    } while (0)

#define EXAMPLE_MSTX_CROSS_CORE_SET_FLAG_REPORT(event_id_, peer_core_id_, pipe_barrier_all_) \
    do {                                                                                     \
        Sanitizer::MstxCrossCoreSetFlag mstx_set__{};                                        \
        mstx_set__.eventId = (event_id_);                                                    \
        mstx_set__.peerCoreId = (peer_core_id_);                                             \
        mstx_set__.pipeBarrierAll = (pipe_barrier_all_);                                     \
        Sanitizer::SanitizerReport(mstx_set__);                                              \
    } while (0)

#define EXAMPLE_MSTX_CROSS_CORE_WAIT_FLAG_REPORT(event_id_, peer_core_id_, pipe_barrier_all_) \
    do {                                                                                      \
        Sanitizer::MstxCrossCoreWaitFlag mstx_wait__{};                                       \
        mstx_wait__.eventId = (event_id_);                                                    \
        mstx_wait__.peerCoreId = (peer_core_id_);                                             \
        mstx_wait__.pipeBarrierAll = (pipe_barrier_all_);                                     \
        Sanitizer::SanitizerReport(mstx_wait__);                                              \
    } while (0)

#define EXAMPLE_MSTX_SIGNAL_SET_REPORT(addr_, val_) \
    do {                                            \
        Sanitizer::MstxSignalSet mstx_set__{};      \
        mstx_set__.addr = (uint64_t)(addr_);        \
        mstx_set__.value = (val_);                  \
        Sanitizer::SanitizerReport(mstx_set__);     \
    } while (0)

#define EXAMPLE_MSTX_SIGNAL_WAIT_REPORT(addr_, cmp_, val_) \
    do {                                                   \
        Sanitizer::MstxSignalWait mstx_wait__{};           \
        mstx_wait__.addr = (uint64_t)(addr_);              \
        mstx_wait__.cmpValue = (val_);                     \
        mstx_wait__.cmpOp = (Sanitizer::CompareOp)(cmp_);  \
        Sanitizer::SanitizerReport(mstx_wait__);           \
    } while (0)

#else

#define EXAMPLE_MSTX_FUSE_SCOPE_START()
#define EXAMPLE_MSTX_FUSE_SCOPE_END()
#define EXAMPLE_MSTX_CROSS_NPU_BARRIER_REPORT(device_num_, core_num_, aiv_only_, pipe_barrier_all_)
#define EXAMPLE_MSTX_CROSS_CORE_SET_FLAG_REPORT(event_id_, peer_core_id_, pipe_barrier_all_)
#define EXAMPLE_MSTX_CROSS_CORE_WAIT_FLAG_REPORT(event_id_, peer_core_id_, pipe_barrier_all_)
#define EXAMPLE_MSTX_SIGNAL_SET_REPORT(addr_, val_)
#define EXAMPLE_MSTX_SIGNAL_WAIT_REPORT(addr_, cmp_, val_)

#endif

#endif
