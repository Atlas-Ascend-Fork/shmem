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
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <getopt.h>
#include "utils.h"
#include "perftest_common_types.h"
#include "mte_perftest_common.h"
#include "perftest_error_check.h"

constexpr int UB_ALIGN_SIZE = 64;

// 时延模式每 PE 的接口形态槽位（普通/QP/defer/submit），与 kernel 侧 RDMA_PERF_IFACE_* 对应；
// timing 总槽位 = 2 PE × 4 接口，布局 slot = pe * 4 + iface
constexpr int LAT_IFACE_COUNT = 4;
constexpr int LAT_TIMING_SLOT_COUNT = LAT_IFACE_COUNT * 2;

int g_npus = 8;
const char* ipport;
static const char* fuc_data_type;
static const char* fuc_test_type;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

extern "C" void launch_rdma_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t fftsAddr, uint8_t* dst_gva, uint8_t* src_gva, int elements,
    int test_mode, int data_type, int ub_size_b, int loop_count, int metric, int batch, int sync_id, int qp_num,
    int qp_index, uint8_t* timing_out_gva);

static perftest::rdma_mode_t get_rdma_mode(const char* test_type_str)
{
    if (strcmp(test_type_str, "put") == 0)
        return perftest::TEST_MODE_RDMA_PUT;
    if (strcmp(test_type_str, "bi_put") == 0)
        return perftest::TEST_MODE_RDMA_BI_PUT;
    if (strcmp(test_type_str, "get") == 0)
        return perftest::TEST_MODE_RDMA_GET;
    if (strcmp(test_type_str, "bi_get") == 0)
        return perftest::TEST_MODE_RDMA_BI_GET;
    return perftest::TEST_MODE_RDMA_INVALID;
}

static bool is_valid_data_type(const char* data_type_str)
{
    if (strcmp(data_type_str, "float") == 0)
        return true;
    if (strcmp(data_type_str, "int8") == 0)
        return true;
    if (strcmp(data_type_str, "int16") == 0)
        return true;
    if (strcmp(data_type_str, "int32") == 0)
        return true;
    if (strcmp(data_type_str, "int64") == 0)
        return true;
    if (strcmp(data_type_str, "uint8") == 0)
        return true;
    if (strcmp(data_type_str, "uint16") == 0)
        return true;
    if (strcmp(data_type_str, "uint32") == 0)
        return true;
    if (strcmp(data_type_str, "uint64") == 0)
        return true;
    if (strcmp(data_type_str, "char") == 0)
        return true;
    return false;
}

static perftest::perf_metric_t get_perf_metric(const char* metric_str)
{
    if (strcmp(metric_str, "lat") == 0)
        return perftest::PERF_METRIC_LAT;
    if (strcmp(metric_str, "bw") == 0)
        return perftest::PERF_METRIC_BW;
    return perftest::PERF_METRIC_INVALID;
}

static uint64_t get_rdma_mem_factor(perftest::perf_metric_t metric, int qp_num, int qp_index)
{
    if (metric != perftest::PERF_METRIC_BW || qp_num <= 1 || qp_index >= 0) {
        return 1;
    }
    return static_cast<uint64_t>(qp_num);
}

template <typename T>
int test_rdma_perf_test_impl(
    int pe_id, int n_pes, uint64_t local_mem_size, int min_exponent, int max_exponent, int loop_count,
    perftest::rdma_mode_t test_mode, perftest::perf_data_type_t data_type_enum, int ub_size_b,
    perftest::perf_metric_t metric, int batch, int sync_id, int qp_num, int qp_index,
    std::vector<std::vector<std::string>>& csv_data)
{
    static const char* kFunc = "test_rdma_perf_test_impl";
    const uint32_t block_dim =
        (metric == perftest::PERF_METRIC_BW && qp_num > 1 && qp_index < 0) ? static_cast<uint32_t>(qp_num) : 1U;
    int32_t device_id = (pe_id % g_npus + f_npu);
    aclshmemx_init_attr_t attributes;
    aclrtStream stream = nullptr;
    void* dst_ptr = nullptr;
    void* src_ptr = nullptr;
    void* timing_out_ptr = nullptr;
    void* combo_ptr = nullptr;
    bool acl_initialized = false;
    bool device_set = false;
    bool stream_created = false;
    bool shmem_initialized = false;
    int ret = 0;
    uint64_t fftsAddr = 0;

    PERFTEST_CHECK_GOTO(aclInit(nullptr), ret, cleanup);
    acl_initialized = true;

    PERFTEST_CHECK_GOTO(aclrtSetDevice(device_id), ret, cleanup);
    device_set = true;

    PERFTEST_CHECK_GOTO(aclrtCreateStream(&stream), ret, cleanup);
    stream_created = true;

    fftsAddr = util_get_ffts_config();

    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
#if defined(ACLSHMEM_RDMA_V2_SUPPORT)
    // The process-wide QP setting is only consumed by the RDMA v2 backend.
    // Legacy RDMA (used on A2/indie) rejects this API, even for the default
    // single-QP configuration, so do not call it in that build.
    PERFTEST_CHECK_GOTO(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, static_cast<uint32_t>(qp_num)), ret, cleanup);
#endif
    PERFTEST_CHECK_GOTO(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), ret, cleanup);
    shmem_initialized = true;

    for (int exponent = min_exponent; exponent <= max_exponent; exponent++) {
        size_t datasize = static_cast<size_t>(1) << exponent;
        std::cout << "pe: " << pe_id << " size: " << datasize << std::endl;

        const uint64_t mem_factor = get_rdma_mem_factor(metric, qp_num, qp_index);
        const size_t alloc_datasize = datasize * mem_factor;
        // 合并一次分配 dst + src + timing_out，各段 64B 对齐，避免同一 cache line
        size_t dst_off = 0;
        size_t src_off = ((dst_off + alloc_datasize) + 63) & ~(size_t)63;
        size_t tim_off = ((src_off + alloc_datasize) + 63) & ~(size_t)63;
        size_t total = tim_off + sizeof(int64_t) * LAT_TIMING_SLOT_COUNT;

        combo_ptr = aclshmem_malloc(total);
        if (combo_ptr == nullptr) {
            std::cerr << "[ERROR] [" << kFunc << "] aclshmem_malloc failed for combo, size=" << total << std::endl;
            ret = -1;
            goto cleanup;
        }
        dst_ptr = static_cast<uint8_t*>(combo_ptr) + dst_off;
        src_ptr = static_cast<uint8_t*>(combo_ptr) + src_off;
        timing_out_ptr = static_cast<uint8_t*>(combo_ptr) + tim_off;

        {
            std::vector<int64_t> zero(LAT_TIMING_SLOT_COUNT, 0);
            PERFTEST_CHECK_GOTO(
                aclrtMemcpy(
                    timing_out_ptr, sizeof(int64_t) * LAT_TIMING_SLOT_COUNT, zero.data(),
                    sizeof(int64_t) * LAT_TIMING_SLOT_COUNT, ACL_MEMCPY_HOST_TO_DEVICE),
                ret, cleanup);
        }

        int trans_size = datasize / sizeof(T);

        const size_t alloc_elements = static_cast<size_t>(trans_size) * static_cast<size_t>(mem_factor);
        std::vector<T> src_input(alloc_elements, 0);
        std::vector<T> dst_input(alloc_elements, 0);
        for (size_t i = 0; i < alloc_elements; i++) {
            src_input[i] = (T)(pe_id + 10);
            dst_input[i] = (T)(pe_id + 100);
        }
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(src_ptr, alloc_datasize, src_input.data(), alloc_datasize, ACL_MEMCPY_HOST_TO_DEVICE), ret,
            cleanup);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(dst_ptr, alloc_datasize, dst_input.data(), alloc_datasize, ACL_MEMCPY_HOST_TO_DEVICE), ret,
            cleanup);

        launch_rdma_perf_kernel(
            block_dim, stream, fftsAddr, (uint8_t*)dst_ptr, (uint8_t*)src_ptr, trans_size, static_cast<int>(test_mode),
            static_cast<int>(data_type_enum), ub_size_b, loop_count, static_cast<int>(metric), batch, sync_id, qp_num,
            qp_index, (uint8_t*)timing_out_ptr);
        PERFTEST_CHECK_GOTO(aclrtSynchronizeStream(stream), ret, cleanup);

        bool is_put_get_mode =
            (test_mode == perftest::TEST_MODE_RDMA_PUT || test_mode == perftest::TEST_MODE_RDMA_BI_PUT ||
             test_mode == perftest::TEST_MODE_RDMA_GET || test_mode == perftest::TEST_MODE_RDMA_BI_GET);
        if (is_put_get_mode) {
            // Read back timing output: [0]=PE0_time(forward), [1]=PE1_time(reverse)
            // After cross-PE exchange in kernel, both PEs have both values.
            std::vector<int64_t> timing_host(LAT_TIMING_SLOT_COUNT, 0);
            PERFTEST_CHECK_GOTO(
                aclrtMemcpy(
                    timing_host.data(), sizeof(int64_t) * LAT_TIMING_SLOT_COUNT, timing_out_ptr,
                    sizeof(int64_t) * LAT_TIMING_SLOT_COUNT, ACL_MEMCPY_DEVICE_TO_HOST),
                ret, cleanup);

            bool is_bidir =
                (test_mode == perftest::TEST_MODE_RDMA_BI_PUT || test_mode == perftest::TEST_MODE_RDMA_BI_GET);

            // CSV: compute BW from timing_out
            const char* soc_name = aclrtGetSocName();
            int64_t cycle2us = 50;
            if (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) {
                cycle2us = 1000;
            }

            // CSV: compute BW from timing_out
            double per_iter_us0 = (loop_count > 0) ? static_cast<double>(timing_host[0]) / cycle2us / loop_count : 0.0;
            double per_iter_us1 = (loop_count > 0) ? static_cast<double>(timing_host[1]) / cycle2us / loop_count : 0.0;
            const uint32_t active_blocks = block_dim;
            const double bytes_per_iter = static_cast<double>(datasize) * static_cast<double>(mem_factor);

            double bw_gb = 0.0, bw_gib = 0.0, per_iter_us = 0.0;
            std::string lat_cell[LAT_IFACE_COUNT] = {"-", "-", "-", "-"};
            if (metric == perftest::PERF_METRIC_LAT) {
                // 时延模式：kernel 自动遍历接口形态（普通/QP/defer/submit），槽位 slot = pe * 4 + iface，
                // 值为单次调用摊薄 cycles；-1 表示当前后端不支持该形态
                static const char* kLatIfaceNames[LAT_IFACE_COUNT] = {"plain", "qp", "defer", "submit"};
                for (int iface = 0; iface < LAT_IFACE_COUNT; ++iface) {
                    const int64_t cycles0 = timing_host[iface];
                    const int64_t cycles1 = timing_host[LAT_IFACE_COUNT + iface];
                    if (is_bidir) {
                        if (cycles0 < 0 || cycles1 < 0) {
                            std::cout << "[LAT][" << kLatIfaceNames[iface] << "] PE" << pe_id
                                      << ": N/A (unsupported by backend)" << std::endl;
                            continue;
                        }
                        const double us0 = static_cast<double>(cycles0) / cycle2us;
                        const double us1 = static_cast<double>(cycles1) / cycle2us;
                        const double avg = (us0 + us1) / 2;
                        std::cout << "[LAT][" << kLatIfaceNames[iface] << "] PE" << pe_id << " forward(PE0→PE1)=" << us0
                                  << " us reverse(PE1→PE0)=" << us1 << " us avg=" << avg << " us" << std::endl;
                        lat_cell[iface] = double_to_string(avg);
                    } else if (pe_id == 0) {
                        if (cycles0 < 0) {
                            std::cout << "[LAT][" << kLatIfaceNames[iface] << "] PE0: N/A (unsupported by backend)"
                                      << std::endl;
                            continue;
                        }
                        const double us0 = static_cast<double>(cycles0) / cycle2us;
                        std::cout << "[LAT][" << kLatIfaceNames[iface] << "] PE0: " << us0 << " us" << std::endl;
                        lat_cell[iface] = double_to_string(us0);
                    }
                }
                if (!is_bidir && pe_id != 0) {
                    std::cout << "[LAT] PE" << pe_id << ": idle (unidirectional)" << std::endl;
                }
            } else if (is_bidir) {
                double bps0 = (per_iter_us0 > 0) ? bytes_per_iter / per_iter_us0 * 1000000.0 : 0.0;
                double bps1 = (per_iter_us1 > 0) ? bytes_per_iter / per_iter_us1 * 1000000.0 : 0.0;
                double bw_gb_0 = bps0 / 1000.0 / 1000.0 / 1000.0;
                double bw_gb_1 = bps1 / 1000.0 / 1000.0 / 1000.0;
                bw_gb = bw_gb_0 + bw_gb_1;
                double bw_gib_0 = bps0 / 1024.0 / 1024.0 / 1024.0;
                double bw_gib_1 = bps1 / 1024.0 / 1024.0 / 1024.0;
                bw_gib = bw_gib_0 + bw_gib_1;
                std::cout << "[BW] PE" << pe_id << " forward(PE0→PE1)=" << bw_gb_0 << " GB/s"
                          << " reverse(PE1→PE0)=" << bw_gb_1 << " GB/s"
                          << " total=" << bw_gb << " GB/s" << std::endl;
                per_iter_us = per_iter_us0 + per_iter_us1;
            } else {
                per_iter_us = (pe_id == 0) ? per_iter_us0 : per_iter_us1;
                double bps = (per_iter_us > 0) ? bytes_per_iter / per_iter_us * 1000000.0 : 0.0;
                bw_gb = bps / 1000.0 / 1000.0 / 1000.0;
                bw_gib = bps / 1024.0 / 1024.0 / 1024.0;
                std::cout << "[BW] PE" << pe_id << ": " << bw_gb << " GB/s" << std::endl;
            }

            std::vector<std::string> row;
            if (metric == perftest::PERF_METRIC_LAT) {
                row = {
                    uint64_to_string(datasize),
                    int_to_string(g_npus),
                    uint64_to_string(active_blocks),
                    double_to_string(ub_size_b / 1024.0),
                    lat_cell[0],
                    lat_cell[1],
                    lat_cell[2],
                    lat_cell[3],
                };
            } else {
                row = {
                    uint64_to_string(datasize),           int_to_string(g_npus),   uint64_to_string(active_blocks),
                    double_to_string(ub_size_b / 1024.0), double_to_string(bw_gb), double_to_string(bw_gib),
                    double_to_string(per_iter_us),
                };
            }
            csv_data.push_back(row);
        }

        bool verify_success = true;
        auto compare_expected = [&](const T* data, T expected, size_t count, const char* label) -> bool {
            for (size_t i = 0; i < count; i++) {
                if (data[i] != expected) {
                    std::cout << "PE[" << pe_id << "]  [ERROR] Mismatch at index " << i << ": " << label << "="
                              << (double)data[i] << ", expected=" << (double)expected << std::endl;
                    return false;
                }
            }
            return true;
        };

        std::vector<T> dst_host(alloc_elements, 0);
        std::vector<T> src_host(alloc_elements, 0);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(dst_host.data(), alloc_datasize, dst_ptr, alloc_datasize, ACL_MEMCPY_DEVICE_TO_HOST), ret,
            cleanup);
        PERFTEST_CHECK_GOTO(
            aclrtMemcpy(src_host.data(), alloc_datasize, src_ptr, alloc_datasize, ACL_MEMCPY_DEVICE_TO_HOST), ret,
            cleanup);

        int peer_pe = (pe_id + 1) % n_pes;
        if (test_mode == perftest::TEST_MODE_RDMA_PUT) {
            std::cout << "\n[Verification] put: checking..." << std::endl;
            if (pe_id != 0) {
                T expected_val = static_cast<T>(10);
                if (!compare_expected(dst_host.data(), expected_val, alloc_elements, "dst")) {
                    verify_success = false;
                }
            }
        } else if (test_mode == perftest::TEST_MODE_RDMA_GET) {
            std::cout << "\n[Verification] get: checking..." << std::endl;
            if (pe_id == 0) {
                T expected_val = static_cast<T>(peer_pe + 10);
                if (!compare_expected(dst_host.data(), expected_val, alloc_elements, "dst")) {
                    verify_success = false;
                }
            }
        } else if (test_mode == perftest::TEST_MODE_RDMA_BI_PUT) {
            std::cout << "\n[Verification] bi_put: checking..." << std::endl;
            T expected_val = static_cast<T>(peer_pe + 10);
            if (!compare_expected(dst_host.data(), expected_val, alloc_elements, "dst")) {
                verify_success = false;
            }
        } else if (test_mode == perftest::TEST_MODE_RDMA_BI_GET) {
            std::cout << "\n[Verification] bi_get: checking..." << std::endl;
            T expected_val = static_cast<T>(peer_pe + 10);
            if (!compare_expected(dst_host.data(), expected_val, alloc_elements, "dst")) {
                verify_success = false;
            }
        }

        if (verify_success) {
            std::cout << "[Verification] SUCCESS" << std::endl;
        } else {
            std::cout << "[Verification] FAILED" << std::endl;
            // Set non-zero return code so caller can detect verification failure
            if (ret == 0) {
                ret = ACLSHMEM_INVALID_VALUE;
            }
        }

        aclshmem_free(combo_ptr);
        combo_ptr = nullptr;
        dst_ptr = nullptr;
        src_ptr = nullptr;
        timing_out_ptr = nullptr;
        PERFTEST_CHECK_GOTO(aclrtSynchronizeStream(stream), ret, cleanup);
    }

    // Summary: report any verification failures across all exponents
    if (ret != 0 && ret == ACLSHMEM_INVALID_VALUE) {
        std::cerr << "[ERROR] [" << kFunc << "] At least one exponent failed data verification in PE " << pe_id
                  << std::endl;
    }

cleanup:
    if (combo_ptr != nullptr) {
        aclshmem_free(combo_ptr);
        combo_ptr = nullptr;
        dst_ptr = nullptr;
        src_ptr = nullptr;
        timing_out_ptr = nullptr;
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
    return ret;
}

int main(int argc, char* argv[])
{
    int n_pes = 2;
    int pe_id = 0;
    ipport = "tcp://127.0.0.1:8768";
    g_npus = 2;
    f_npu = 4;
    const char* test_type = "put";
    fuc_data_type = "float";
    int input_block_size = 1;
    int min_exponent = 3;
    int max_exponent = 17;
    int loop_count = 1000;
    int ub_size_b = 128;
    const char* metric_str = "bw";
    int batch = 0;
    int sync_id = 0;
    int qp_num = 1;
    int qp_index = -1;
    int qp_specified = 0;

    static struct option long_options[] = {
        {"pes", required_argument, 0, 0},
        {"pe-id", required_argument, 0, 0},
        {"ipport", required_argument, 0, 0},
        {"gnpus", required_argument, 0, 0},
        {"fnpu", required_argument, 0, 0},
        {"test-type", required_argument, 0, 't'},
        {"datatype", required_argument, 0, 'd'},
        {"block-size", required_argument, 0, 'b'},
        {"block-range", required_argument, 0, 0},
        {"exponent", required_argument, 0, 'e'},
        {"exponent-range", required_argument, 0, 0},
        {"loop-count", required_argument, 0, 0},
        {"ub-size", required_argument, 0, 0},
        {"metric", required_argument, 0, 0},
        {"batch", required_argument, 0, 0},
        {"sync-id", required_argument, 0, 0},
        {"qp", required_argument, 0, 'q'},
        {"qp-count", required_argument, 0, 'q'},
        {"qp-index", required_argument, 0, 'i'},
        {"qp-specified", required_argument, 0, 0},
        {0, 0, 0, 0}};

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "t:d:b:e:q:i:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 't':
                test_type = optarg;
                break;
            case 'd':
                fuc_data_type = optarg;
                break;
            case 'b':
                input_block_size = std::atoi(optarg);
                break;
            case 'e':
                min_exponent = max_exponent = std::atoi(optarg);
                break;
            case 'q':
                qp_num = std::atoi(optarg);
                qp_specified = 1;
                break;
            case 'i':
                qp_index = std::atoi(optarg);
                qp_specified = 1;
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
                else if (strcmp(long_options[option_index].name, "fnpu") == 0)
                    f_npu = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "block-range") == 0) {
                    input_block_size = std::atoi(optarg);
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
                    ub_size_b = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "metric") == 0)
                    metric_str = optarg;
                else if (strcmp(long_options[option_index].name, "batch") == 0)
                    batch = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "sync-id") == 0)
                    sync_id = std::atoi(optarg);
                else if (strcmp(long_options[option_index].name, "qp") == 0) {
                    qp_num = std::atoi(optarg);
                    qp_specified = 1;
                } else if (strcmp(long_options[option_index].name, "qp-specified") == 0)
                    qp_specified = std::atoi(optarg) != 0 ? 1 : 0;
                break;
            default:
                std::cerr << "Unknown argument" << std::endl;
                return 1;
        }
    }

    if (n_pes <= 0) {
        std::cerr << "Error: --pes must be greater than 0 (got " << n_pes << ")" << std::endl;
        return 1;
    }
    if (g_npus <= 0) {
        std::cerr << "Error: --gnpus must be greater than 0 (got " << g_npus << ")" << std::endl;
        return 1;
    }
    if (pe_id < 0 || pe_id >= n_pes) {
        std::cerr << "Error: --pe-id must be in range [0, " << n_pes << ") (got " << pe_id << ")" << std::endl;
        return 1;
    }
    if (f_npu < 0) {
        std::cerr << "Error: --fnpu must be greater than or equal to 0 (got " << f_npu << ")" << std::endl;
        return 1;
    }

    if (input_block_size != 1) {
        std::cerr << "WARN: RDMA perftest forces block_size=1 (got " << input_block_size << ")" << std::endl;
    }

    perftest::perf_metric_t metric = get_perf_metric(metric_str);
    if (metric == perftest::PERF_METRIC_INVALID) {
        std::cerr << "Error: --metric must be 'bw' or 'lat' (got '" << metric_str << "')" << std::endl;
        return 1;
    }

    if (strcmp(test_type, "put_signal") == 0) {
        std::cerr << "Error: test type 'put_signal' is not supported by rdma_perftest "
                     "(aclshmemx_roce_put_signal_nbi API unavailable)"
                  << std::endl;
        return 1;
    }

    if (batch < 0) {
        std::cerr << "Error: --batch must be >= 0 (got " << batch << ")" << std::endl;
        return 1;
    }
    if (loop_count < 1) {
        std::cerr << "Error: --loop-count must be >= 1 (got " << loop_count << ")" << std::endl;
        return 1;
    }
    if (metric == perftest::PERF_METRIC_LAT) {
        if (batch == 0)
            batch = perftest::PERFTEST_WARMUP_ITERS;
    } else if (batch > loop_count) {
        batch = loop_count;
    }
    if (qp_num < 1 || qp_num > static_cast<int>(ACLSHMEM_MAX_QP_NUM)) {
        std::cerr << "Error: --qp must be in range [1, " << ACLSHMEM_MAX_QP_NUM << "] (got " << qp_num << ")"
                  << std::endl;
        return 1;
    }
#if !defined(ACLSHMEM_RDMA_V2_SUPPORT)
    if (qp_specified != 0) {
        std::cerr << "Error: explicit --qp/--qp-count requires an RDMA V2 build" << std::endl;
        return 1;
    }
#endif
    if (qp_index < -1 || qp_index >= qp_num) {
        std::cerr << "Error: --qp-index must be -1 (automatic) or in range [0, " << qp_num << ") (got " << qp_index
                  << ")" << std::endl;
        return 1;
    }
    if (metric == perftest::PERF_METRIC_LAT && qp_num > 1) {
        std::cerr << "Error: latency test does not support qp_num > 1; use --qp 1" << std::endl;
        return 1;
    }
    if (sync_id < 0) {
        std::cerr << "Error: --sync-id must be >= 0 (got " << sync_id << ")" << std::endl;
        return 1;
    }

    if (!is_valid_data_type(fuc_data_type)) {
        std::cerr << "Error: --datatype must be one of: float, int8, int16, int32, int64, uint8, uint16, uint32, "
                     "uint64, char (got '"
                  << fuc_data_type << "')" << std::endl;
        return 1;
    }

    ub_size_b = ((ub_size_b + UB_ALIGN_SIZE - 1) / UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

    std::cout << "[INFO] rdma_perftest start, pe=" << pe_id << ", t=" << test_type << ", d=" << fuc_data_type
              << ", exp=" << min_exponent << "-" << max_exponent << ", loop=" << loop_count << ", ub=" << ub_size_b
              << "B, metric=" << metric_str << ", batch=" << batch << ", sync_id=" << sync_id << ", qp_count=" << qp_num
              << ", qp_index=" << qp_index << ", qp_specified=" << qp_specified << std::endl;

    fuc_test_type = test_type;
    perftest::rdma_mode_t test_mode = get_rdma_mode(test_type);
    if (test_mode == perftest::TEST_MODE_RDMA_INVALID) {
        std::cerr << "Error: -t/--test-type must be one of: put, bi_put, get, bi_get (got '" << test_type << "')"
                  << std::endl;
        return 1;
    }
    perftest::perf_data_type_t data_type_enum = get_data_type(fuc_data_type);

    const int MAX_ALLOWED_EXPONENT = 62;
    if (min_exponent < 0 || max_exponent < min_exponent || max_exponent > MAX_ALLOWED_EXPONENT) {
        std::cerr << "Error: exponent range invalid, must be [0, " << MAX_ALLOWED_EXPONENT << "] and min<=max (got "
                  << min_exponent << "-" << max_exponent << ")" << std::endl;
        return 1;
    }

    const uint64_t max_single_datasize = 1ULL << max_exponent;
    const uint64_t mem_factor = get_rdma_mem_factor(metric, qp_num, qp_index);
    if (max_single_datasize > UINT64_MAX / mem_factor) {
        std::cerr << "Error: required multi-QP buffer size overflows uint64_t" << std::endl;
        return 1;
    }
    uint64_t max_datasize = max_single_datasize * mem_factor;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    const uint64_t ONE_GB = 1024UL * 1024UL * 1024;
    const uint64_t MAX_GB = 40;
    if (max_datasize > UINT64_MAX / 2) {
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

    // lat 模式输出各接口形态时延列；bw 模式输出带宽列
    std::vector<std::vector<std::string>> csv_data;
    if (metric == perftest::PERF_METRIC_LAT) {
        csv_data = {
            {"DataSize/B", "Npus", "Blocks", "UBsize/KiB", "PlainLat/us", "QpLat/us", "DeferLat/us", "SubmitLat/us"}};
    } else {
        csv_data = {
            {"DataSize/B", "Npus", "Blocks", "UBsize/KiB", "Bandwidth/GB/s", "Bandwidth/GiB/s", "CoreMaxTime/us"}};
    }

    int status = 0;
#define RDMA_TEST_IMPL_OP(type)                                                                                     \
    status = test_rdma_perf_test_impl<type>(                                                                        \
        pe_id, n_pes, local_mem_size, min_exponent, max_exponent, loop_count, test_mode, data_type_enum, ub_size_b, \
        metric, batch, sync_id, qp_num, qp_index, csv_data)
    DISPATCH_BY_TYPE(fuc_data_type, RDMA_TEST_IMPL_OP);
#undef RDMA_TEST_IMPL_OP

    if (status != 0) {
        std::cerr << "[FAILED] rdma_perftest failed in pe " << pe_id << std::endl;
        return status;
    }
    if (pe_id == 0) {
        std::string csv_filename = "output/rdma_" + std::string(metric_str) + "_" + std::string(fuc_test_type) + "_" +
                                   std::string(fuc_data_type) + "_0.csv";
        write_csv(csv_filename, csv_data);
    }

    std::cout << "[SUCCESS] rdma_perftest done in pe " << pe_id << std::endl;
    return 0;
}
