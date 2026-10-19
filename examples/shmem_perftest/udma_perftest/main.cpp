/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <iomanip>
#include <getopt.h>
#include "utils.h"
#include "perftest_common_types.h"
#include "mte_perftest_common.h"
#include "perftest_error_check.h"

int g_npus = 8;
const char* ipport;
static const char* fuc_data_type;
static const char* fuc_test_type;
int f_pe = 0;
int f_npu = 0;
// UDMA shared jetty switch: 1 enables it through attr.runtime_config.udma_shared_jetty.
int g_shared_jetty = 0;
aclshmemx_uniqueid_t default_flag_uid;

static aclshmem_prof_pe_t* out_profs;

// 时延模式每 PE 的接口形态槽位（普通/QP/defer/submit），与 kernel 侧 UDMA_PERF_IFACE_* 对应
constexpr int LAT_IFACE_COUNT = 4;

extern "C" void launch_udma_perf_kernel(
    uint32_t block_dim, void* stream, uint8_t* dst_gva, uint8_t* src_gva, uint8_t* sig_gva, int elements,
    int32_t frame_id, int test_mode, int data_type, int ub_size_kib, int64_t prof_pe_val, int loop_count,
    uint64_t signal_base, int metric, int batch, int qp_count, uint8_t* timing_gva);

static perftest::udma_mode_t get_udma_mode(const char* test_type_str)
{
    if (strcmp(test_type_str, "put") == 0)
        return perftest::TEST_MODE_UDMA_PUT;
    if (strcmp(test_type_str, "bi_put") == 0)
        return perftest::TEST_MODE_UDMA_BI_PUT;
    if (strcmp(test_type_str, "get") == 0)
        return perftest::TEST_MODE_UDMA_GET;
    if (strcmp(test_type_str, "bi_get") == 0)
        return perftest::TEST_MODE_UDMA_BI_GET;
    if (strcmp(test_type_str, "put_signal") == 0)
        return perftest::TEST_MODE_UDMA_PUT_SIGNAL;
    return perftest::TEST_MODE_UDMA_PUT;
}

static perftest::perf_metric_t get_perf_metric(const char* metric_str)
{
    if (strcmp(metric_str, "lat") == 0)
        return perftest::PERF_METRIC_LAT;
    return perftest::PERF_METRIC_BW;
}

template <typename T>
int test_udma_perf_test_impl(
    int pe_id, int n_pes, uint64_t local_mem_size, int min_exponent, int max_exponent, int loop_count,
    perftest::udma_mode_t test_mode, perftest::perf_data_type_t data_type_enum, int prof_pe, int ub_size_kib,
    perftest::perf_metric_t metric, int batch, int qp_count, std::vector<std::vector<std::string>>& csv_data)
{
    static const char* kFunc = "test_udma_perf_test_impl";
    const uint32_t block_dim =
        (test_mode == perftest::TEST_MODE_UDMA_PUT_SIGNAL || metric == perftest::PERF_METRIC_LAT) ?
            1U :
            static_cast<uint32_t>(qp_count);
    int32_t device_id = (pe_id % g_npus + f_npu);
    pe_id += f_pe;
    int ret = 0;
    aclrtStream stream = nullptr;
    void* sig_ptr = nullptr;
    void* dst_ptr = nullptr;
    void* src_ptr = nullptr;
    void* timing_ptr = nullptr;
    aclshmemx_init_attr_t attributes;
    bool acl_initialized = false;
    bool device_set = false;
    bool stream_created = false;
    bool shmem_initialized = false;
    int frame_id = 0;
    int executed_points = 0;

    PERFTEST_CHECK_GOTO(aclInit(nullptr), ret, cleanup);
    acl_initialized = true;

    PERFTEST_CHECK_GOTO(aclrtSetDevice(device_id), ret, cleanup);
    device_set = true;

    PERFTEST_CHECK_GOTO(aclrtCreateStream(&stream), ret, cleanup);
    stream_created = true;

    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    // Shared jetty must be enabled before initialization and kept identical on every PE.
    if (g_shared_jetty != 0) {
        attributes.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED;
    }
    if (qp_count > 1) {
        PERFTEST_CHECK_GOTO(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, static_cast<uint32_t>(qp_count)), ret, cleanup);
    }
    PERFTEST_CHECK_GOTO(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), ret, cleanup);
    shmem_initialized = true;

    if (test_mode == perftest::TEST_MODE_UDMA_PUT_SIGNAL) {
        sig_ptr = aclshmem_malloc(n_pes * sizeof(uint64_t));
        if (sig_ptr == nullptr) {
            std::cerr << "[ERROR] [" << kFunc << "] aclshmem_malloc failed for signal words" << std::endl;
            ret = -1;
            goto cleanup;
        }
        std::vector<uint64_t> sig_init(n_pes, 0);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(
                sig_ptr, n_pes * sizeof(uint64_t), sig_init.data(), n_pes * sizeof(uint64_t),
                ACL_MEMCPY_HOST_TO_DEVICE),
            ret, cleanup);
    }
    if (metric == perftest::PERF_METRIC_LAT) {
        timing_ptr = aclshmem_malloc(sizeof(int64_t) * LAT_IFACE_COUNT);
        if (timing_ptr == nullptr) {
            std::cerr << "[ERROR] [" << kFunc << "] aclshmem_malloc failed for timing slots" << std::endl;
            ret = -1;
            goto cleanup;
        }
        std::vector<int64_t> timing_zero(LAT_IFACE_COUNT, 0);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(
                timing_ptr, sizeof(int64_t) * LAT_IFACE_COUNT, timing_zero.data(), sizeof(int64_t) * LAT_IFACE_COUNT,
                ACL_MEMCPY_HOST_TO_DEVICE),
            ret, cleanup);
    }
    for (int exponent = min_exponent; exponent <= max_exponent; exponent++) {
        const uint64_t datasize = 1ULL << exponent;
        const uint64_t total_datasize = datasize * static_cast<uint64_t>(qp_count);
        std::cout << "pe: " << pe_id << " size(per QP): " << datasize << "B, total: " << total_datasize
                  << "B frame_id: " << frame_id << std::endl;

        const uint64_t elements_per_qp = datasize / sizeof(T);
        if (elements_per_qp == 0 || elements_per_qp > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            std::cerr << "WARN: skip size " << datasize << "B per QP: element count " << elements_per_qp
                      << " is outside kernel int range" << std::endl;
            continue;
        }
        const int elements = static_cast<int>(elements_per_qp);
        const size_t total_elements = static_cast<size_t>(elements_per_qp) * static_cast<size_t>(qp_count);
        executed_points++;
        dst_ptr = aclshmem_malloc(total_datasize);
        src_ptr = aclshmem_malloc(total_datasize);
        if (dst_ptr == nullptr || src_ptr == nullptr) {
            std::cerr << "[ERROR] [" << kFunc << "] aclshmem_malloc failed, size=" << total_datasize << std::endl;
            ret = -1;
            goto cleanup;
        }

        std::vector<T> src_input(total_elements, 0);
        std::vector<T> dst_input(total_elements, 0);
        for (size_t i = 0; i < total_elements; i++) {
            src_input[i] = (T)(pe_id + 10);
            dst_input[i] = (T)(pe_id + 100);
        }
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(src_ptr, total_datasize, src_input.data(), total_datasize, ACL_MEMCPY_HOST_TO_DEVICE), ret,
            cleanup);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(dst_ptr, total_datasize, dst_input.data(), total_datasize, ACL_MEMCPY_HOST_TO_DEVICE), ret,
            cleanup);

        uint64_t signal_base = static_cast<uint64_t>(frame_id) * 1000000ULL + 1;
        launch_udma_perf_kernel(
            block_dim, stream, (uint8_t*)dst_ptr, (uint8_t*)src_ptr, (uint8_t*)sig_ptr, elements, frame_id,
            static_cast<int>(test_mode), static_cast<int>(data_type_enum), ub_size_kib, prof_pe, loop_count,
            signal_base, static_cast<int>(metric), batch, qp_count, (uint8_t*)timing_ptr);
        PERFTEST_CHECK_GOTO(aclrtSynchronizeStream(stream), ret, cleanup);

        bool verify_success = true;
        auto compare_values = [&](const T* values, T expected, size_t count, const char* value_label,
                                  const char* expected_label) -> bool {
            for (size_t i = 0; i < count; i++) {
                if (values[i] != expected) {
                    std::cout << "  [ERROR] Mismatch at index " << i << ": " << value_label << "=" << (double)values[i]
                              << ", " << expected_label << "=" << (double)expected << std::endl;
                    return false;
                }
            }
            return true;
        };

        std::vector<T> dst_host(total_elements, 0);
        std::vector<T> src_host(total_elements, 0);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(dst_host.data(), total_datasize, dst_ptr, total_datasize, ACL_MEMCPY_DEVICE_TO_HOST), ret,
            cleanup);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(src_host.data(), total_datasize, src_ptr, total_datasize, ACL_MEMCPY_DEVICE_TO_HOST), ret,
            cleanup);

        int peer_pe = (pe_id + 1) % n_pes;
        const bool is_put_signal = test_mode == perftest::TEST_MODE_UDMA_PUT_SIGNAL;
        if (test_mode == perftest::TEST_MODE_UDMA_PUT || is_put_signal) {
            const char* verification_desc =
                is_put_signal ? "put_signal: checking data + signal..." : "put: checking...";
            std::cout << "\n[Verification] " << verification_desc << std::endl;
            // Unilateral put is issued only by prof_pe and targets its next PE. All
            // other PEs retain their locally initialized destination buffer.
            const int target_pe = (prof_pe + 1) % n_pes;
            const T expected_val = static_cast<T>(pe_id == target_pe ? prof_pe + 10 : pe_id + 100);
            if (!compare_values(dst_host.data(), expected_val, total_elements, "dst", "expected")) {
                verify_success = false;
            }
            if (is_put_signal) {
                std::vector<uint64_t> sig_init(n_pes, 0);
                PERFTEST_CHECK_GOTO(
                    aclrtMemcpy(
                        sig_init.data(), n_pes * sizeof(uint64_t), sig_ptr, n_pes * sizeof(uint64_t),
                        ACL_MEMCPY_DEVICE_TO_HOST),
                    ret, cleanup);
                const uint64_t expected_sig =
                    pe_id == target_pe ?
                        signal_base + static_cast<uint64_t>(perftest::PERFTEST_WARMUP_ITERS + loop_count - 1) :
                        0;
                if (sig_init[prof_pe] != expected_sig) {
                    std::cout << "  [ERROR] signal slot[" << prof_pe << "]=" << sig_init[prof_pe]
                              << ", expected=" << expected_sig << std::endl;
                    verify_success = false;
                }
            }
        } else if (test_mode == perftest::TEST_MODE_UDMA_GET) {
            std::cout << "\n[Verification] get: checking..." << std::endl;
            if (pe_id == prof_pe) {
                T expected_val = static_cast<T>(peer_pe + 10);
                if (!compare_values(dst_host.data(), expected_val, total_elements, "dst", "expected")) {
                    verify_success = false;
                }
            }
        } else if (test_mode == perftest::TEST_MODE_UDMA_BI_PUT) {
            std::cout << "\n[Verification] bi_put: checking..." << std::endl;
            // Each PE writes to its next PE, so this PE receives data from its
            // previous PE in the ring.
            const int sender_pe = (pe_id - 1 + n_pes) % n_pes;
            T expected_val = static_cast<T>(sender_pe + 10);
            if (!compare_values(dst_host.data(), expected_val, total_elements, "dst", "expected")) {
                verify_success = false;
            }
        } else if (test_mode == perftest::TEST_MODE_UDMA_BI_GET) {
            std::cout << "\n[Verification] bi_get: checking..." << std::endl;
            T expected_val = static_cast<T>(peer_pe + 10);
            if (!compare_values(dst_host.data(), expected_val, total_elements, "dst", "expected")) {
                verify_success = false;
            }
        }

        if (verify_success) {
            std::cout << "[Verification] SUCCESS" << std::endl;
        } else {
            std::cout << "[Verification] FAILED" << std::endl;
            if (ret == 0) {
                ret = ACLSHMEM_INVALID_VALUE;
            }
        }

        if (metric == perftest::PERF_METRIC_LAT) {
            // 槽位与 kernel 侧 UDMA_PERF_IFACE_* 对应，值为单次调用摊薄 cycles，-1 表示 N/A
            std::vector<int64_t> timing_host(LAT_IFACE_COUNT, 0);
            PERFTEST_CHECK_GOTO(
                aclrtMemcpy(
                    timing_host.data(), sizeof(int64_t) * LAT_IFACE_COUNT, timing_ptr,
                    sizeof(int64_t) * LAT_IFACE_COUNT, ACL_MEMCPY_DEVICE_TO_HOST),
                ret, cleanup);
            constexpr int64_t cycle2us = 1000; // Ascend950 计数器 1GHz；UDMA 仅在 950 编译期使能
            static const char* kLatIfaceNames[LAT_IFACE_COUNT] = {"plain", "qp", "defer", "submit"};
            std::string lat_cell[LAT_IFACE_COUNT] = {"-", "-", "-", "-"};
            if (pe_id == prof_pe) {
                for (int iface = 0; iface < LAT_IFACE_COUNT; ++iface) {
                    const int64_t cycles = timing_host[iface];
                    if (cycles < 0) {
                        std::cout << "[LAT][" << kLatIfaceNames[iface] << "] PE" << pe_id << ": N/A (window too small)"
                                  << std::endl;
                        continue;
                    }
                    const double latency_us = static_cast<double>(cycles) / cycle2us;
                    std::cout << "[LAT][" << kLatIfaceNames[iface] << "] PE" << pe_id << ": " << latency_us << " us"
                              << std::endl;
                    lat_cell[iface] = double_to_string(latency_us);
                }
            } else {
                std::cout << "[LAT] PE" << pe_id << ": idle (unidirectional)" << std::endl;
            }
            csv_data.push_back(
                {uint64_to_string(datasize), int_to_string(g_npus), int_to_string(static_cast<int>(block_dim)),
                 double_to_string(ub_size_kib), lat_cell[0], lat_cell[1], lat_cell[2], lat_cell[3]});
        } else {
            aclshmemx_get_prof(&out_profs, false);
            collect_prof_data_to_csv_v2(
                out_profs, frame_id, datasize, qp_count, g_npus, ub_size_kib, loop_count,
                metric == perftest::PERF_METRIC_BW, csv_data);
        }

        aclshmem_free(dst_ptr);
        dst_ptr = nullptr;
        aclshmem_free(src_ptr);
        src_ptr = nullptr;
        PERFTEST_CHECK_GOTO(aclrtSynchronizeStream(stream), ret, cleanup);

        frame_id++;
        if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
            std::cerr << "warning: frame_id reached limit " << ACLSHMEM_CYCLE_PROF_FRAME_CNT << std::endl;
            break;
        }
    }
    aclshmemx_get_prof(nullptr, true);

cleanup:
    if (dst_ptr != nullptr) {
        aclshmem_free(dst_ptr);
    }
    if (src_ptr != nullptr) {
        aclshmem_free(src_ptr);
    }
    if (sig_ptr != nullptr) {
        aclshmem_free(sig_ptr);
    }
    if (timing_ptr != nullptr) {
        aclshmem_free(timing_ptr);
    }
    if (shmem_initialized) {
        PERFTEST_CHECK_CLEANUP(aclshmem_finalize(), ret);
    }
    if (stream_created) {
        PERFTEST_CHECK_CLEANUP(aclrtDestroyStream(stream), ret);
    }
    if (device_set) {
        PERFTEST_CHECK_CLEANUP(aclrtResetDevice(device_id), ret);
    }
    if (acl_initialized) {
        PERFTEST_CHECK_CLEANUP(aclFinalize(), ret);
    }
    if (ret == 0 && executed_points == 0) {
        std::cerr << "Error: no data size in the exponent range is valid for dtype size and kernel limits" << std::endl;
        ret = 1;
    }
    return ret;
}

int main(int argc, char* argv[])
{
    int n_pes = 2;
    int pe_id = 0;
    ipport = "tcp://127.0.0.1:8768";
    g_npus = 2;
    f_pe = 0;
    f_npu = 4;
    const char* test_type = "put";
    fuc_data_type = "float";
    int min_exponent = 3;
    int max_exponent = 17;
    int loop_count = 1000;
    int ub_size_kib = 16;
    const char* metric_str = "bw";
    int batch = 0; // BW: 0 => full async; 1 => quiet after each NBI; N > 1 => defer/submit per group
    int qp_count = 1;
    int shared_jetty = 0; // 0 disables shared jetty (default), 1 enables it

    static struct option long_options[] = {
        {"pes", required_argument, 0, 0},          {"pe-id", required_argument, 0, 0},
        {"ipport", required_argument, 0, 0},       {"gnpus", required_argument, 0, 0},
        {"fpe", required_argument, 0, 0},          {"fnpu", required_argument, 0, 0},
        {"test-type", required_argument, 0, 't'},  {"datatype", required_argument, 0, 'd'},
        {"block-size", required_argument, 0, 'b'}, {"block-range", required_argument, 0, 0},
        {"exponent", required_argument, 0, 'e'},   {"exponent-range", required_argument, 0, 0},
        {"loop-count", required_argument, 0, 0},   {"ub-size", required_argument, 0, 0},
        {"metric", required_argument, 0, 0},       {"batch", required_argument, 0, 0},
        {"shared-jetty", required_argument, 0, 0}, {0, 0, 0, 0}};

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "t:d:b:e:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 't':
                test_type = optarg;
                break;
            case 'd':
                fuc_data_type = optarg;
                break;
            case 'b':
                qp_count = std::atoi(optarg);
                break;
            case 'e':
                min_exponent = max_exponent = std::atoi(optarg);
                break;
            case 0:
                if (strcmp(long_options[option_index].name, "pes") == 0)
                    n_pes = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "pe-id") == 0)
                    pe_id = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "ipport") == 0)
                    ipport = optarg;
                else if (strcmp(long_options[option_index].name, "gnpus") == 0)
                    g_npus = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "fpe") == 0)
                    f_pe = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "fnpu") == 0)
                    f_npu = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "block-range") == 0) {
                    if (optind < argc) {
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "exponent-range") == 0) {
                    min_exponent = std::atoi(optarg);
                    if (optind < argc) {
                        max_exponent = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "loop-count") == 0)
                    loop_count = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "ub-size") == 0)
                    ub_size_kib = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "metric") == 0)
                    metric_str = optarg;
                else if (strcmp(long_options[option_index].name, "batch") == 0)
                    batch = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "shared-jetty") == 0)
                    shared_jetty = std::atoi(optarg);
                break;
            default:
                std::cerr << "Unknown argument" << std::endl;
                return 1;
        }
    }

    if (strcmp(metric_str, "bw") != 0 && strcmp(metric_str, "lat") != 0) {
        std::cerr << "Error: --metric must be 'bw' or 'lat' (got '" << metric_str << "')" << std::endl;
        return 1;
    }
    perftest::perf_metric_t metric = get_perf_metric(metric_str);
    if (metric == perftest::PERF_METRIC_LAT && strcmp(test_type, "put") != 0) {
        std::cerr << "Error: --metric lat is only supported with -t put (got '" << test_type << "')" << std::endl;
        return 1;
    }
    if (metric == perftest::PERF_METRIC_LAT && qp_count > 1) {
        std::cerr << "Error: --metric lat does not support multi-QP; use -b 1 (got qp_count=" << qp_count << ")"
                  << std::endl;
        return 1;
    }

    if (batch < 0) {
        std::cerr << "Error: --batch must be >= 0 (got " << batch << ")" << std::endl;
        return 1;
    }
    if (shared_jetty != 0 && shared_jetty != 1) {
        std::cerr << "Error: --shared-jetty must be 0 or 1 (got " << shared_jetty << ")" << std::endl;
        return 1;
    }
    g_shared_jetty = shared_jetty;
    if (qp_count < 1 || qp_count > static_cast<int>(ACLSHMEM_MAX_QP_NUM)) {
        std::cerr << "Error: --block-size must be in [1, " << ACLSHMEM_MAX_QP_NUM << "] (got " << qp_count << ")"
                  << std::endl;
        return 1;
    }
    if (qp_count > 1 && (strcmp(test_type, "put_signal") == 0 || strcmp(test_type, "all") == 0)) {
        std::cerr << "Error: UDMA multi-QP is supported for put/bi_put/get/bi_get only; put_signal remains single-QP"
                  << std::endl;
        return 1;
    }
#if defined(ACLSHMEM_RELAY_SUPPORT)
    if (qp_count > 1) {
        std::cerr << "Error: UDMA multi-QP is unavailable when ACLSHMEM_RELAY_SUPPORT is enabled" << std::endl;
        return 1;
    }
#endif
    if (loop_count < 1) {
        std::cerr << "Error: --loop-count must be >= 1 (got " << loop_count << ")" << std::endl;
        return 1;
    }
    if (metric == perftest::PERF_METRIC_BW && batch > 1 && strcmp(test_type, "put_signal") != 0) {
        constexpr uint64_t UDMA_SQ_RING_DEPTH = 32768;  // UDMA SQ ring capacity in 64B WQEBBs (8192 WQEs x 4 BB).
        constexpr uint64_t UDMA_WQE_STAGING_BYTES = 64; // One staged data-mover WQE.
        const uint64_t effective_batch = static_cast<uint64_t>(batch > loop_count ? loop_count : batch);
        const uint64_t ub_bytes = ub_size_kib > 0 ? static_cast<uint64_t>(ub_size_kib) * 1024ULL : 0ULL;
        if (effective_batch >= UDMA_SQ_RING_DEPTH || effective_batch * UDMA_WQE_STAGING_BYTES > ub_bytes) {
            std::cerr << "Error: --batch must fit UDMA aggregate limits: batch < " << UDMA_SQ_RING_DEPTH
                      << " and batch * " << UDMA_WQE_STAGING_BYTES << " <= --ub-size * 1024" << std::endl;
            return 1;
        }
    }
    if (metric == perftest::PERF_METRIC_LAT) {
        // 时延模式：batch 为聚合接口的窗口大小，默认 warmup 次数；窗口 N 次 defer + 1 次 submit 需落在 UB 内
        if (batch == 0) {
            batch = perftest::PERFTEST_WARMUP_ITERS;
        }
        if (batch > loop_count) {
            batch = loop_count;
        }
        const uint64_t ub_bytes = ub_size_kib > 0 ? static_cast<uint64_t>(ub_size_kib) * 1024ULL : 0ULL;
        if (static_cast<uint64_t>(batch + 1) * 64ULL > ub_bytes) {
            std::cerr << "Error: --batch " << batch << " window needs " << ((batch + 1) * 64)
                      << "B UB staging (defer window + submit), exceeds --ub-size " << ub_size_kib << " KiB"
                      << std::endl;
            return 1;
        }
    }
    if (min_exponent < 0 || min_exponent > max_exponent || max_exponent > 62) {
        std::cerr << "Error: exponent range must satisfy 0 <= min <= max <= 62 (got " << min_exponent << "-"
                  << max_exponent << ")" << std::endl;
        return 1;
    }
    std::cout << "[INFO] udma_perftest start, pe=" << (pe_id + f_pe) << ", t=" << test_type << ", d=" << fuc_data_type
              << ", exp=" << min_exponent << "-" << max_exponent << ", loop=" << loop_count << ", ub=" << ub_size_kib
              << "KiB"
              << ", metric=" << metric_str << ", batch=" << batch << ", qp_count=" << qp_count
              << ", shared_jetty=" << g_shared_jetty << std::endl;

    fuc_test_type = test_type;
    perftest::udma_mode_t test_mode = get_udma_mode(test_type);
    perftest::perf_data_type_t data_type_enum = get_data_type(fuc_data_type);
    const uint64_t max_per_qp_datasize = 1ULL << max_exponent;
    if (max_per_qp_datasize > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(qp_count)) {
        std::cerr << "Error: required multi-QP buffer size overflows uint64_t" << std::endl;
        return 1;
    }
    const uint64_t max_datasize = max_per_qp_datasize * static_cast<uint64_t>(qp_count);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    const uint64_t ONE_GB = 1024UL * 1024UL * 1024;
    const uint64_t MAX_GB = 40;
    if (max_datasize > std::numeric_limits<uint64_t>::max() / 2) {
        std::cerr << "Error: required src/dst buffer size overflows uint64_t" << std::endl;
        return 1;
    }
    if (max_datasize * 2 > local_mem_size) {
        uint64_t gb_needed = (max_datasize * 2 + ONE_GB - 1) / ONE_GB;
        if (gb_needed > MAX_GB) {
            std::cerr << "Error: required mem exceeds 40GB" << std::endl;
            return 1;
        }
        local_mem_size = gb_needed * ONE_GB;
    }

    const char* prof_pe_env = std::getenv("SHMEM_CYCLE_PROF_PE");
    int prof_pe = (prof_pe_env != nullptr) ? std::atoi(prof_pe_env) : 0;

    std::vector<std::vector<std::string>> csv_data;
    if (metric == perftest::PERF_METRIC_LAT) {
        csv_data = {
            {"DataSize/B", "Npus", "Blocks", "UBsize/KiB", "PlainLat/us", "QpLat/us", "DeferLat/us", "SubmitLat/us"}};
    } else {
        csv_data = {
            {"DataSize/B", "Npus", "Blocks", "UBsize/KiB", "Bandwidth/GB/s(1000)", "Bandwidth/GiB/s(1024)",
             "CoreMaxTime/us", "SingleCoreTime/us"}};
    }

    int status = 0;
#define UDMA_TEST_IMPL_OP(type)                                                                                   \
    status = test_udma_perf_test_impl<type>(                                                                      \
        pe_id, n_pes, local_mem_size, min_exponent, max_exponent, loop_count, test_mode, data_type_enum, prof_pe, \
        ub_size_kib, metric, batch, qp_count, csv_data)
    DISPATCH_BY_TYPE(fuc_data_type, UDMA_TEST_IMPL_OP);
#undef UDMA_TEST_IMPL_OP

    if (status != 0) {
        std::cerr << "[FAILED] udma_perftest failed in pe " << (pe_id + f_pe) << std::endl;
        return status;
    }
    if ((pe_id + f_pe) == prof_pe) {
        std::string output_suffix;
        if (qp_count > 1)
            output_suffix += "_qp" + int_to_string(qp_count);
        std::string csv_filename = "output/udma_" + std::string(metric_str) + "_" + std::string(fuc_test_type) + "_" +
                                   std::string(fuc_data_type) + output_suffix + "_" + int_to_string(prof_pe) + ".csv";
        write_csv(csv_filename, csv_data);
    }

    std::cout << "[SUCCESS] udma_perftest done in pe " << (pe_id + f_pe) << std::endl;
    return 0;
}
