/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/exception/shmem_exception_report.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <sstream>

#include "acl/acl_rt.h"
#include "utils/exception/shmem_exception_udma_dump.h"
#include "utils/exception/shmem_exception_rdma_dump.h"
#include "dl_acl_api.h"
#include "host/utils/shmem_host_exception.h"
#include "init/shmemi_init.h"
#include "shmemi_host_common.h"

namespace {
enum aclshmemi_exception_report_registration_state_t : uint32_t {
    ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_UNREGISTERED = 0,
    ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_REGISTERED,
    ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_DEFERRED,
    ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_UNSUPPORTED,
};

struct aclshmemi_runtime_exception_snapshot_t {
    uint64_t seq{0};
    uint32_t task_id{0};
    uint32_t stream_id{0};
    uint32_t thread_id{0};
    uint32_t device_id{0};
    uint32_t error_code{0};
    uint64_t lost_snapshot_count{0};
};

struct aclshmemi_runtime_exception_atomic_t {
    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> seq{0};
    std::atomic<uint32_t> task_id{0};
    std::atomic<uint32_t> stream_id{0};
    std::atomic<uint32_t> thread_id{0};
    std::atomic<uint32_t> device_id{0};
    std::atomic<uint32_t> error_code{0};
    std::atomic<uint64_t> lost_snapshot_count{0};
};

struct aclshmemi_exception_report_state_t {
    std::atomic<bool> explicit_configured{false};
    // Raw engine mask requested by init; the selected mask may be recomputed after DEBUG is enabled.
    std::atomic<uint32_t> requested_engines{0};
    std::atomic<uint32_t> enabled_engines{0};
    std::atomic<uint64_t> udma_info_address{0};
    std::atomic<uint32_t> mode{ACLSHMEMI_EXCEPTION_REPORT_MODE_OFF};
    std::atomic<aclshmemx_exception_info_callback_t> user_callback{nullptr};
    std::atomic<uint64_t> reported_seq{0};
    std::atomic<uint64_t> reported_lost_snapshot_count{0};
    std::atomic<uint32_t> registration_state{ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_UNREGISTERED};
    std::atomic_flag snapshot_lock = ATOMIC_FLAG_INIT;
    aclshmemi_runtime_exception_atomic_t last_exception{};
};

struct aclshmemi_exception_engine_reporter_t {
    uint32_t engine;
    const char* name;
    int (*dump)(bool detail_enabled);
};

aclshmemi_exception_report_state_t g_exception_report_state;
std::mutex g_exception_report_mutex;

const aclshmemi_exception_engine_reporter_t ACLSHMEMI_EXCEPTION_ENGINE_REPORTERS[] = {
    {static_cast<uint32_t>(ACLSHMEM_DATA_OP_ROCE), "RDMA", aclshmemi_exception_report_dump_rdma},
    {static_cast<uint32_t>(ACLSHMEM_DATA_OP_UDMA), "UDMA", aclshmemi_exception_report_dump_udma},
};

uint32_t aclshmemi_exception_report_selected_transport_engine(data_op_engine_type_t requested_engines)
{
    if (g_exception_report_state.mode.load(std::memory_order_acquire) != ACLSHMEMI_EXCEPTION_REPORT_MODE_DEBUG) {
        return 0;
    }
    if (g_state.npes <= 1) {
        return 0;
    }

    const uint32_t engines = static_cast<uint32_t>(requested_engines);
    // UDMA is the primary exception reporter when requested.  RDMA/ROCE is
    // selected only when UDMA is absent, preserving the single-reporter
    // transport selection contract for composite masks.
    if ((engines & static_cast<uint32_t>(ACLSHMEM_DATA_OP_UDMA)) != 0U) {
        return static_cast<uint32_t>(ACLSHMEM_DATA_OP_UDMA);
    }
    if (engines != static_cast<uint32_t>(ACLSHMEM_DATA_OP_ROCE)) {
        return 0;
    }
    uint32_t selected_engines = 0;
    for (const auto& reporter : ACLSHMEMI_EXCEPTION_ENGINE_REPORTERS) {
        selected_engines |= engines & reporter.engine;
    }
    return selected_engines;
}

bool aclshmemi_exception_report_detail_enabled()
{
    return g_exception_report_state.mode.load(std::memory_order_acquire) == ACLSHMEMI_EXCEPTION_REPORT_MODE_DEBUG;
}

bool aclshmemi_exception_report_enabled()
{
    return g_exception_report_state.mode.load(std::memory_order_acquire) != ACLSHMEMI_EXCEPTION_REPORT_MODE_OFF;
}

bool aclshmemi_exception_report_unsupported()
{
    return g_exception_report_state.explicit_configured.load(std::memory_order_acquire) &&
           g_exception_report_state.registration_state.load(std::memory_order_acquire) ==
               ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_UNSUPPORTED;
}

void aclshmemi_exception_callback(aclrtExceptionInfo* exception_info)
{
    if (exception_info == nullptr) {
        return;
    }

    auto& state = g_exception_report_state;
    if (aclshmemi_exception_report_enabled()) {
        (void)aclshmemi_exception_report_record_snapshot(
            shm::DlAclApi::AclrtGetTaskIdFromExceptionInfo(exception_info),
            shm::DlAclApi::AclrtGetStreamIdFromExceptionInfo(exception_info),
            shm::DlAclApi::AclrtGetThreadIdFromExceptionInfo(exception_info),
            shm::DlAclApi::AclrtGetDeviceIdFromExceptionInfo(exception_info),
            shm::DlAclApi::AclrtGetErrorCodeFromExceptionInfo(exception_info));
    }

    aclshmemx_exception_info_callback_t user_callback = state.user_callback.load(std::memory_order_acquire);
    if (user_callback != nullptr) {
        user_callback(static_cast<void*>(exception_info));
    }
}

aclshmemi_runtime_exception_snapshot_t aclshmemi_exception_report_snapshot()
{
    auto& last = g_exception_report_state.last_exception;
    aclshmemi_runtime_exception_snapshot_t snapshot{};
    while (true) {
        const uint64_t version_begin = last.version.load(std::memory_order_acquire);
        if ((version_begin & 1U) != 0) {
            continue;
        }
        snapshot.seq = last.seq.load(std::memory_order_relaxed);
        snapshot.task_id = last.task_id.load(std::memory_order_relaxed);
        snapshot.stream_id = last.stream_id.load(std::memory_order_relaxed);
        snapshot.thread_id = last.thread_id.load(std::memory_order_relaxed);
        snapshot.device_id = last.device_id.load(std::memory_order_relaxed);
        snapshot.error_code = last.error_code.load(std::memory_order_relaxed);
        snapshot.lost_snapshot_count = last.lost_snapshot_count.load(std::memory_order_relaxed);
        const uint64_t version_end = last.version.load(std::memory_order_acquire);
        if (version_begin == version_end) {
            break;
        }
    }
    return snapshot;
}

int aclshmemi_exception_report_set_callback()
{
    if (!shm::DlAclApi::AclrtExceptionInfoApisAvailable()) {
        SHM_LOG_WARN("Runtime exception report symbols are not available, skip callback registration.");
        return ACLSHMEM_NOT_SUPPORTED;
    }
    auto ret = shm::DlAclApi::AclrtSetExceptionInfoCallback(aclshmemi_exception_callback);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_WARN("aclrtSetExceptionInfoCallback failed, ret = " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

void aclshmemi_exception_report_unset_callback()
{
    if (!shm::DlAclApi::AclrtExceptionInfoApisAvailable()) {
        SHM_LOG_WARN("Runtime exception report symbols are not available, skip callback deregistration.");
        return;
    }
    auto ret = shm::DlAclApi::AclrtSetExceptionInfoCallback(nullptr);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_WARN("aclrtSetExceptionInfoCallback(nullptr) failed, ret = " << ret);
    }
}

void aclshmemi_exception_report_log_runtime(const aclshmemi_runtime_exception_snapshot_t& info)
{
    SHM_LOG_ERROR(
        "[EXCEPTION][RUNTIME] rank=" << g_state.mype << "/" << g_state.npes << " device=" << info.device_id
                                     << " streamId=" << info.stream_id << " taskId=" << info.task_id
                                     << " threadId=" << info.thread_id << " errorCode=" << info.error_code
                                     << " lostSnapshotCount=" << info.lost_snapshot_count << " seq=" << info.seq);
}

void aclshmemi_exception_report_log_state()
{
    const uint64_t udma_info_address = g_exception_report_state.udma_info_address.load(std::memory_order_acquire);
    const uint32_t enabled_engines = g_exception_report_state.enabled_engines.load(std::memory_order_acquire);
    const uint32_t requested_engines = g_exception_report_state.requested_engines.load(std::memory_order_acquire);
    const bool rdma_enabled = (enabled_engines & static_cast<uint32_t>(ACLSHMEM_DATA_OP_ROCE)) != 0U;
    const bool udma_enabled = (enabled_engines & static_cast<uint32_t>(ACLSHMEM_DATA_OP_UDMA)) != 0U;
    std::ostringstream out;
    out << "[EXCEPTION][STATE]";
    out << " requestedEngines=0x" << std::hex << requested_engines << " enabledEngines=0x" << enabled_engines
        << std::dec;
    if (rdma_enabled) {
        out << " rdmaInfo=0x" << std::hex << g_state.qp_info;
        out << " rdmaConfig={ub=0x" << std::hex << g_state.rdma_config.aclshmem_ub << ", size=" << std::dec
            << g_state.rdma_config.ub_size << ", syncId=" << g_state.rdma_config.sync_id << "}";
    }
    if (udma_enabled) {
        out << " udmaInfo=0x" << std::hex << udma_info_address;
        out << " udmaConfig={ub=0x" << std::hex << g_state.udma_config.aclshmem_ub << ", size=" << std::dec
            << g_state.udma_config.ub_size << ", syncId=" << g_state.udma_config.sync_id << "}";
    }
    out << " defaultStream=" << std::hex << g_state_host.default_stream << " heapBase=" << g_state.heap_base
        << " hostHeapBase=" << g_state.host_heap_base << " heapSize=0x" << g_state.heap_size << std::dec
        << " mype=" << g_state.mype << " npes=" << g_state.npes;
    SHM_LOG_ERROR(out.str());
}

void aclshmemi_exception_report_log_transport_dump_missing(uint32_t enabled_engines)
{
    if ((enabled_engines & static_cast<uint32_t>(ACLSHMEM_DATA_OP_ROCE | ACLSHMEM_DATA_OP_UDMA)) != 0U) {
        return; // UDMA or RDMA reporter dumped the transport state; skip reasons are logged by the reporter itself.
    }
    const uint32_t requested_engines = g_exception_report_state.requested_engines.load(std::memory_order_acquire);
    const char* reason = nullptr;
    if (g_state.npes <= 1) {
        reason = "npes<=1, no transport engine reporter is selected for single PE";
    } else {
        reason = "data_op_engine_type selects no dump reporter, only UDMA/ROCE engines support exception dump; "
                 "add ACLSHMEM_DATA_OP_UDMA, or request ACLSHMEM_DATA_OP_ROCE alone, in "
                 "attributes.option_attr.data_op_engine_type to enable transport dump";
    }
    SHM_LOG_ERROR(
        "[EXCEPTION][REPORTER] transport dump is missing: reason="
        << reason << ", requestedEngines=0x" << std::hex << requested_engines << ", enabledEngines=0x"
        << enabled_engines << std::dec << ", npes=" << g_state.npes);
}

int aclshmemi_exception_report_dump_enabled_engines()
{
    int ret = ACLSHMEM_SUCCESS;
    const uint32_t engines = g_exception_report_state.enabled_engines.load(std::memory_order_acquire);
    const bool detail_enabled = aclshmemi_exception_report_detail_enabled();
    aclshmemi_exception_report_log_transport_dump_missing(engines);
    for (const auto& reporter : ACLSHMEMI_EXCEPTION_ENGINE_REPORTERS) {
        if ((engines & reporter.engine) == 0) {
            continue;
        }
        const int reporter_ret = reporter.dump(detail_enabled);
        if (reporter_ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("[EXCEPTION][REPORTER] transport=" << reporter.name << " dump failed, ret=" << reporter_ret);
            ret = reporter_ret;
        }
    }
    return ret;
}

int aclshmemi_exception_report_dump_snapshot(const aclshmemi_runtime_exception_snapshot_t& snapshot)
{
    aclshmemi_exception_report_log_runtime(snapshot);
    if (!aclshmemi_exception_report_detail_enabled()) {
        return ACLSHMEM_SUCCESS;
    }
    aclshmemi_exception_report_log_state();
    return aclshmemi_exception_report_dump_enabled_engines();
}
} // namespace

bool aclshmemi_exception_report_pending(void)
{
    if (!aclshmemi_exception_report_enabled()) {
        return false;
    }
    const auto snapshot = aclshmemi_exception_report_snapshot();
    const uint64_t reported_seq = g_exception_report_state.reported_seq.load(std::memory_order_acquire);
    const uint64_t reported_lost =
        g_exception_report_state.reported_lost_snapshot_count.load(std::memory_order_acquire);
    return snapshot.seq != reported_seq || snapshot.lost_snapshot_count != reported_lost;
}

void aclshmemi_exception_report_set_udma_info_address(uint64_t udma_info_address)
{
    g_exception_report_state.udma_info_address.store(udma_info_address, std::memory_order_release);
}

uint64_t aclshmemi_exception_report_udma_info_address(void)
{
    return g_exception_report_state.udma_info_address.load(std::memory_order_acquire);
}

bool aclshmemi_exception_report_record_snapshot(
    uint32_t task_id, uint32_t stream_id, uint32_t thread_id, uint32_t device_id, uint32_t error_code)
{
    auto& state = g_exception_report_state;
    if (!aclshmemi_exception_report_enabled()) {
        return false;
    }
    if (state.snapshot_lock.test_and_set(std::memory_order_acquire)) {
        state.last_exception.lost_snapshot_count.fetch_add(1U, std::memory_order_relaxed);
        return false;
    }
    const uint64_t version = state.last_exception.version.load(std::memory_order_relaxed) + 1U;
    state.last_exception.version.store(version, std::memory_order_release);
    state.last_exception.seq.store(
        state.last_exception.seq.load(std::memory_order_relaxed) + 1U, std::memory_order_relaxed);
    state.last_exception.task_id.store(task_id, std::memory_order_relaxed);
    state.last_exception.stream_id.store(stream_id, std::memory_order_relaxed);
    state.last_exception.thread_id.store(thread_id, std::memory_order_relaxed);
    state.last_exception.device_id.store(device_id, std::memory_order_relaxed);
    state.last_exception.error_code.store(error_code, std::memory_order_relaxed);
    state.last_exception.version.store(version + 1U, std::memory_order_release);
    state.snapshot_lock.clear(std::memory_order_release);
    return true;
}

void aclshmemi_exception_report_save_context(aclshmemi_exception_report_context_t& context)
{
    const auto snapshot = aclshmemi_exception_report_snapshot();
    context.explicit_configured = g_exception_report_state.explicit_configured.load(std::memory_order_acquire);
    context.requested_engines = g_exception_report_state.requested_engines.load(std::memory_order_acquire);
    context.enabled_engines = g_exception_report_state.enabled_engines.load(std::memory_order_acquire);
    context.mode = g_exception_report_state.mode.load(std::memory_order_acquire);
    context.user_callback = g_exception_report_state.user_callback.load(std::memory_order_acquire);
    context.registration_state = g_exception_report_state.registration_state.load(std::memory_order_acquire);
    context.reported_seq = g_exception_report_state.reported_seq.load(std::memory_order_acquire);
    context.reported_lost_snapshot_count =
        g_exception_report_state.reported_lost_snapshot_count.load(std::memory_order_acquire);
    context.udma_info_address = g_exception_report_state.udma_info_address.load(std::memory_order_acquire);
    context.seq = snapshot.seq;
    context.task_id = snapshot.task_id;
    context.stream_id = snapshot.stream_id;
    context.thread_id = snapshot.thread_id;
    context.device_id = snapshot.device_id;
    context.error_code = snapshot.error_code;
    context.lost_snapshot_count = snapshot.lost_snapshot_count;
}

void aclshmemi_exception_report_restore_context(const aclshmemi_exception_report_context_t& context)
{
    auto& state = g_exception_report_state;
    while (state.snapshot_lock.test_and_set(std::memory_order_acquire)) {
    }
    const uint64_t version = state.last_exception.version.load(std::memory_order_relaxed);
    const uint64_t write_version = (version & 1U) == 0U ? version + 1U : version;
    state.last_exception.version.store(write_version, std::memory_order_release);
    state.last_exception.seq.store(context.seq, std::memory_order_relaxed);
    state.last_exception.task_id.store(context.task_id, std::memory_order_relaxed);
    state.last_exception.stream_id.store(context.stream_id, std::memory_order_relaxed);
    state.last_exception.thread_id.store(context.thread_id, std::memory_order_relaxed);
    state.last_exception.device_id.store(context.device_id, std::memory_order_relaxed);
    state.last_exception.error_code.store(context.error_code, std::memory_order_relaxed);
    state.last_exception.lost_snapshot_count.store(context.lost_snapshot_count, std::memory_order_relaxed);
    state.last_exception.version.store(write_version + 1U, std::memory_order_release);
    state.snapshot_lock.clear(std::memory_order_release);

    state.explicit_configured.store(context.explicit_configured, std::memory_order_release);
    state.requested_engines.store(context.requested_engines, std::memory_order_release);
    state.enabled_engines.store(context.enabled_engines, std::memory_order_release);
    state.mode.store(context.mode, std::memory_order_release);
    state.user_callback.store(context.user_callback, std::memory_order_release);
    state.registration_state.store(context.registration_state, std::memory_order_release);
    state.reported_seq.store(context.reported_seq, std::memory_order_release);
    state.reported_lost_snapshot_count.store(context.reported_lost_snapshot_count, std::memory_order_release);
    state.udma_info_address.store(context.udma_info_address, std::memory_order_release);
}

int aclshmemi_exception_report_dump(void)
{
    if (!aclshmemi_exception_report_enabled()) {
        return ACLSHMEM_SUCCESS;
    }
    const auto snapshot = aclshmemi_exception_report_snapshot();
    return aclshmemi_exception_report_dump_snapshot(snapshot);
}

int aclshmemi_exception_report_apply_deferred_config(data_op_engine_type_t enabled_engines)
{
    g_exception_report_state.requested_engines.store(static_cast<uint32_t>(enabled_engines), std::memory_order_release);
    g_exception_report_state.enabled_engines.store(
        aclshmemi_exception_report_selected_transport_engine(enabled_engines), std::memory_order_release);
    if (g_exception_report_state.explicit_configured.load(std::memory_order_acquire)) {
        int ret = aclshmemi_exception_report_set_callback();
        if (ret == ACLSHMEM_NOT_SUPPORTED) {
            g_exception_report_state.registration_state.store(
                ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_UNSUPPORTED, std::memory_order_release);
            return ACLSHMEM_SUCCESS;
        }
        if (ret == ACLSHMEM_SUCCESS) {
            g_exception_report_state.registration_state.store(
                ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_REGISTERED, std::memory_order_release);
        } else {
            aclshmemi_exception_report_finalize();
        }
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

void aclshmemi_exception_report_finalize(void)
{
    std::lock_guard<std::mutex> lock(g_exception_report_mutex);
    const auto snapshot = aclshmemi_exception_report_snapshot();
    const bool callback_registered = g_exception_report_state.registration_state.load(std::memory_order_acquire) ==
                                     ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_REGISTERED;
    g_exception_report_state.enabled_engines.store(0, std::memory_order_release);
    g_exception_report_state.requested_engines.store(0, std::memory_order_release);
    g_exception_report_state.udma_info_address.store(0, std::memory_order_release);
    g_exception_report_state.user_callback.store(nullptr, std::memory_order_release);
    g_exception_report_state.mode.store(ACLSHMEMI_EXCEPTION_REPORT_MODE_OFF, std::memory_order_release);
    g_exception_report_state.explicit_configured.store(false, std::memory_order_release);
    g_exception_report_state.registration_state.store(
        ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_UNREGISTERED, std::memory_order_release);
    g_exception_report_state.last_exception.lost_snapshot_count.store(0, std::memory_order_release);
    g_exception_report_state.reported_seq.store(snapshot.seq, std::memory_order_release);
    g_exception_report_state.reported_lost_snapshot_count.store(0, std::memory_order_release);
    if (callback_registered) {
        aclshmemi_exception_report_unset_callback();
    }
}

int aclshmemx_enable_exception_report(
    aclshmemx_exception_info_callback_t callback, aclshmemx_exception_report_level_t level)
{
    std::lock_guard<std::mutex> lock(g_exception_report_mutex);
    auto& state = g_exception_report_state;
    aclshmemi_exception_report_mode_t mode = ACLSHMEMI_EXCEPTION_REPORT_MODE_OFF;
    if (level == ACLSHMEMX_EXCEPTION_REPORT_INFO) {
        mode = ACLSHMEMI_EXCEPTION_REPORT_MODE_INFO;
    } else if (level == ACLSHMEMX_EXCEPTION_REPORT_DEBUG) {
        mode = ACLSHMEMI_EXCEPTION_REPORT_MODE_DEBUG;
    } else {
        return ACLSHMEM_INVALID_PARAM;
    }

    const bool old_explicit_configured = state.explicit_configured.load(std::memory_order_acquire);
    const uint32_t old_requested_engines = state.requested_engines.load(std::memory_order_acquire);
    const uint32_t old_enabled_engines = state.enabled_engines.load(std::memory_order_acquire);
    const uint32_t old_mode = state.mode.load(std::memory_order_acquire);
    const auto old_callback = state.user_callback.load(std::memory_order_acquire);
    const uint32_t old_registration_state = state.registration_state.load(std::memory_order_acquire);

    state.explicit_configured.store(true, std::memory_order_release);
    state.mode.store(static_cast<uint32_t>(mode), std::memory_order_release);
    state.user_callback.store(callback, std::memory_order_release);
    state.registration_state.store(ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_DEFERRED, std::memory_order_release);
    state.enabled_engines.store(
        aclshmemi_exception_report_selected_transport_engine(
            static_cast<data_op_engine_type_t>(state.requested_engines.load(std::memory_order_acquire))),
        std::memory_order_release);

    int ret = aclshmemi_exception_report_set_callback();
    if (ret == ACLSHMEM_NOT_SUPPORTED && !g_state.is_aclshmem_created) {
        return ACLSHMEM_SUCCESS;
    }
    if (ret != ACLSHMEM_SUCCESS) {
        state.explicit_configured.store(old_explicit_configured, std::memory_order_release);
        state.requested_engines.store(old_requested_engines, std::memory_order_release);
        state.enabled_engines.store(old_enabled_engines, std::memory_order_release);
        state.mode.store(old_mode, std::memory_order_release);
        state.user_callback.store(old_callback, std::memory_order_release);
        state.registration_state.store(old_registration_state, std::memory_order_release);
        return ret;
    }
    state.registration_state.store(ACLSHMEMI_EXCEPTION_REPORT_REGISTRATION_REGISTERED, std::memory_order_release);
    return ret;
}

int aclshmemx_report_exception(void)
{
    std::lock_guard<std::mutex> lock(g_exception_report_mutex);
    if (aclshmemi_exception_report_unsupported()) {
        return ACLSHMEM_SUCCESS;
    }
    if (!aclshmemi_exception_report_pending()) {
        return ACLSHMEM_SUCCESS;
    }
    const auto snapshot = aclshmemi_exception_report_snapshot();
    int ret = aclshmemi_exception_report_dump_snapshot(snapshot);
    if (ret == ACLSHMEM_SUCCESS) {
        g_exception_report_state.reported_seq.store(snapshot.seq, std::memory_order_release);
        g_exception_report_state.reported_lost_snapshot_count.store(
            snapshot.lost_snapshot_count, std::memory_order_release);
    }
    return ret;
}
