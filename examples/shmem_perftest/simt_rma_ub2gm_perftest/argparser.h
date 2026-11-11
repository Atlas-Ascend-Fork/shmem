/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_RMA_UB2GM_PERFTEST_ARGPARSER
#define SIMT_RMA_UB2GM_PERFTEST_ARGPARSER

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// For ACLSHMEM_CYCLE_PROF_MAX_BLOCK: the per-PE profiling buffer holds this many
// per-core slots, which bounds the usable core count (see validate_config).
#include "host_device/shmem_common_types.h"

enum class OpType { Get, Put };

inline OpType str2op(const std::string& s)
{
    if (s == "get") {
        return OpType::Get;
    } else if (s == "put") {
        return OpType::Put;
    } else {
        throw std::invalid_argument("Invalid operation type: " + s);
    }
}

inline std::string to_string(OpType op)
{
    switch (op) {
        case OpType::Get:
            return "get";
        case OpType::Put:
            return "put";
        default:
            throw std::invalid_argument("Invalid operation type");
    }
}

// 单次传输数据量的指数上界。per_invocation_bytes = 1 << exp 必须能放进 UB，
// 而 UB 缓冲区容量为 UB_BUFFER_SIZE(16384) 个 int32_t = 64KB = 2^16 字节。
// 与 main.cpp 中的 UB_BUFFER_SIZE 保持一致；修改该常量后需同步此处。
constexpr int BYTES_IN_EXP_LOWER = 3;
constexpr int BYTES_IN_EXP_UPPER = 16;

struct Config {
    int npes = 2;                                // -pes : number of PEs (must be 2 for this test)
    int mype = -1;                               // --pe-id : this process's PE rank (0=ACTIVE, 1=PASSIVE)
    int gnpus = 2;                               // -gnpus : number of NPUs on this node
    int first_pe = 0;                            // -fpe : first PE id. Kept only for CLI compatibility with the
                                                 //        other shmem_perftest samples; unused here (rank comes from
                                                 //        --pe-id, device from mype % gnpus + first_npu).
    int first_npu = 0;                           // -fnpu : first NPU id; device = mype % gnpus + first_npu
    std::string ipport = "tcp://127.0.0.1:8760"; // -ipport : bootstrap ip:port
    int block_size_min = 32;                     // -b/--block-size / --block-range : cores sweep lower bound
    int block_size_max = 32;                     // --block-range : cores sweep upper bound
    std::vector<int> block_sizes;                // resolved core counts to test; filled from --block-list, or
                                                 // expanded from [block_size_min, block_size_max] when no list given.
    int bytes_in_exp_min = BYTES_IN_EXP_LOWER;
    int bytes_in_exp_max = BYTES_IN_EXP_UPPER;
    int loop_count = 1000;

    std::optional<OpType> req_op_type; // -t/--test-type : put/get

    friend std::ostream& operator<<(std::ostream& os, const Config& c)
    {
        os << "================ Config ================\n"
           << std::left << std::setw(20) << "npes:" << c.npes << "\n"
           << std::setw(20) << "mype:" << c.mype << "\n"
           << std::setw(20) << "gnpus:" << c.gnpus << "\n"
           << std::setw(20) << "first_pe:" << c.first_pe << "\n"
           << std::setw(20) << "first_npu:" << c.first_npu << "\n"
           << std::setw(20) << "ipport:" << c.ipport << "\n"
           << std::setw(20) << "block_size_min:" << c.block_size_min << "\n"
           << std::setw(20) << "block_size_max:" << c.block_size_max << "\n"
           << std::setw(20) << "bytes-in-exp-min:" << c.bytes_in_exp_min << "\n"
           << std::setw(20) << "bytes-in-exp-max:" << c.bytes_in_exp_max << "\n"
           << std::setw(20) << "loop_count:" << c.loop_count << "\n"
           << "========================================";
        return os;
    }
};

// ---------- 使用说明 ----------
inline void print_usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " --pe-id <int> [options]\n"
              << "Options:\n"
              << "  -pes <int>               Number of PEs, must be 2. Default: 2\n"
              << "  --pe-id <int>            This process's PE rank (0=ACTIVE, 1=PASSIVE). Required\n"
              << "  -gnpus <int>             Number of NPUs on this node, must be 2. Default: 2\n"
              << "  -fpe <int>               First PE id. Kept for CLI compatibility; unused. Default: 0\n"
              << "  -fnpu <int>              First NPU id; device = pe_id % gnpus + fnpu. Default: 0\n"
              << "  -ipport <ip:port>        Bootstrap address. Default: tcp://127.0.0.1:8760\n"
              << "  -t, --test-type <put|get>   Operation to measure. Default: get\n"
              << "  -b, --block-size <int>   Kernel grid (cores). Default: 32\n"
              << "  --block-range <min> <max>   Cores sweep range. Default: 32 32\n"
              << "  --block-list <b1,b2,...> Explicit core counts to test, comma-separated\n"
              << "                           (e.g. 1,8,16). Overrides -b/--block-range.\n"
              << "  -e, --exponent <exp>     Single data-size exponent (bytes = 1 << exp).\n"
              << "  --exponent-range <min> <max>  Data-size exponent range. Default: " << BYTES_IN_EXP_LOWER << " "
              << BYTES_IN_EXP_UPPER << "\n"
              << "  --loop-count <int>       Default: 1000\n"
              << "  -h, --help               Show this message\n";
}

// ---------- 参数校验 ----------
// 各项约束均来源于 main.cpp 中对 Config 字段的实际使用方式：
//  - pes / gnpus : 本测试固定两卡 Active/Passive 模型，两者都必须为 2。
//  - pe-id: 作为 PE rank，且固定两卡模型只区分 ACTIVE_PE(0) / PASSIVE_PE(1)，
//           因此只能取 0 或 1。实际设备号由 pe_id % gnpus + fnpu 推导。
//  - fpe / fnpu : 须为非负。
//  - block_size_* : kernel 的 grid 大小（核数）扫描区间，须为正且 min <= max。
//           host 侧按 per_core_bytes * block_size_max 分配缓冲区，因此 --block-list
//           解析后必须把 min/max 校正到实际集合的上下界（见 parse_args）。上界还受
//           profiling 缓冲区限制：aclshmem_prof_pe_t::block_prof 只有
//           ACLSHMEM_CYCLE_PROF_MAX_BLOCK 项，而 SHMEMI_PROF_START/END 仅校验
//           GetBlockIdx() < ACLSHMEM_CYCLE_PROF_FRAME_CNT（两者不等），故核数超过
//           前者会在 device 侧越界写。
//  - bytes-in-exp-* : per_invocation_bytes = 1 << exp，指数需落在
//           [BYTES_IN_EXP_LOWER, BYTES_IN_EXP_UPPER] 且 min <= max。上界受 UB
//           容量限制（见 BYTES_IN_EXP_UPPER 注释），超出会在 kernel 侧被跳过。
//  - loop-count 在 [1, 10000)（loops 是求平均时的除数）。
inline bool validate_config(const Config& c)
{
    if (c.npes != 2) {
        std::cerr << "Error: -pes must be 2 (this test uses a fixed 2-card "
                  << "Active/Passive model), got " << c.npes << ".\n";
        return false;
    }
    if (c.gnpus != 2) {
        std::cerr << "Error: -gnpus must be 2 for this fixed 2-card model, got " << c.gnpus << ".\n";
        return false;
    }
    if (c.mype != 0 && c.mype != 1) {
        std::cerr << "Error: --pe-id must be 0 (ACTIVE_PE) or 1 (PASSIVE_PE), got " << c.mype << ".\n";
        return false;
    }
    // first_pe is not consumed by the test (see Config); still range-check it so a
    // bogus value passed for CLI compatibility fails loudly rather than silently.
    if (c.first_pe < 0) {
        std::cerr << "Error: -fpe must be >= 0, got " << c.first_pe << ".\n";
        return false;
    }
    if (c.first_npu < 0) {
        std::cerr << "Error: -fnpu must be >= 0, got " << c.first_npu << ".\n";
        return false;
    }
    if (c.block_size_min < 1) {
        std::cerr << "Error: block count must be >= 1, got " << c.block_size_min << ".\n";
        return false;
    }
    if (c.block_size_max < 1) {
        std::cerr << "Error: block count must be >= 1, got " << c.block_size_max << ".\n";
        return false;
    }
    if (c.block_size_max > ACLSHMEM_CYCLE_PROF_MAX_BLOCK) {
        std::cerr << "Error: block count must be <= " << ACLSHMEM_CYCLE_PROF_MAX_BLOCK
                  << " (profiling buffer capacity), got " << c.block_size_max << ".\n";
        return false;
    }
    if (c.block_size_min > c.block_size_max) {
        std::cerr << "Error: block range lower bound (" << c.block_size_min << ") must not exceed upper bound ("
                  << c.block_size_max << ").\n";
        return false;
    }
    if (c.bytes_in_exp_min < BYTES_IN_EXP_LOWER || c.bytes_in_exp_min > BYTES_IN_EXP_UPPER) {
        std::cerr << "Error: exponent lower bound must be in [" << BYTES_IN_EXP_LOWER << ", " << BYTES_IN_EXP_UPPER
                  << "], got " << c.bytes_in_exp_min << ".\n";
        return false;
    }
    if (c.bytes_in_exp_max < BYTES_IN_EXP_LOWER || c.bytes_in_exp_max > BYTES_IN_EXP_UPPER) {
        std::cerr << "Error: exponent upper bound must be in [" << BYTES_IN_EXP_LOWER << ", " << BYTES_IN_EXP_UPPER
                  << "], got " << c.bytes_in_exp_max << ".\n";
        return false;
    }
    if (c.bytes_in_exp_min > c.bytes_in_exp_max) {
        std::cerr << "Error: exponent lower bound (" << c.bytes_in_exp_min << ") must not exceed upper bound ("
                  << c.bytes_in_exp_max << ").\n";
        return false;
    }
    if (c.loop_count < 1 || c.loop_count >= 10000) {
        std::cerr << "Error: --loop-count must be in [1, 10000), got " << c.loop_count << ".\n";
        return false;
    }
    return true;
}

// ---------- 参数解析 ----------
inline std::optional<Config> parse_args(int argc, char* argv[])
{
    Config config;
    bool pe_id_provided = false;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return std::nullopt;
            }

            // 需要两个取值的参数
            if (arg == "--block-range" || arg == "--exponent-range") {
                if (i + 2 >= argc) {
                    std::cerr << "Argument " << arg << " requires two values.\n";
                    return std::nullopt;
                }
                int lo = std::stoi(argv[++i]);
                int hi = std::stoi(argv[++i]);
                if (arg == "--block-range") {
                    config.block_size_min = lo;
                    config.block_size_max = hi;
                } else {
                    config.bytes_in_exp_min = lo;
                    config.bytes_in_exp_max = hi;
                }
                continue;
            }

            // 需要一个取值的参数
            if (i + 1 >= argc) {
                std::cerr << "Argument " << arg << " missing value.\n";
                return std::nullopt;
            }
            std::string val = argv[++i];

            if (arg == "-pes") {
                config.npes = std::stoi(val);
            } else if (arg == "--pe-id") {
                config.mype = std::stoi(val);
                pe_id_provided = true;
            } else if (arg == "-gnpus") {
                config.gnpus = std::stoi(val);
            } else if (arg == "-fpe") {
                config.first_pe = std::stoi(val); // accepted for compatibility; unused
            } else if (arg == "-fnpu") {
                config.first_npu = std::stoi(val);
            } else if (arg == "-ipport") {
                config.ipport = val;
            } else if (arg == "-t" || arg == "--test-type") {
                config.req_op_type = str2op(val);
            } else if (arg == "-b" || arg == "--block-size") {
                config.block_size_min = std::stoi(val);
                config.block_size_max = config.block_size_min;
            } else if (arg == "--block-list") {
                config.block_sizes.clear();
                size_t start = 0;
                while (start < val.size()) {
                    size_t comma = val.find(',', start);
                    std::string tok = val.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                    if (!tok.empty()) {
                        config.block_sizes.push_back(std::stoi(tok));
                    }
                    if (comma == std::string::npos) {
                        break;
                    }
                    start = comma + 1;
                }
                if (config.block_sizes.empty()) {
                    std::cerr << "Error: --block-list must contain at least one value.\n";
                    return std::nullopt;
                }
            } else if (arg == "--loop-count") {
                config.loop_count = std::stoi(val);
            } else if (arg == "-e" || arg == "--exponent") {
                int exp = std::stoi(val);
                config.bytes_in_exp_min = exp;
                config.bytes_in_exp_max = exp;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return std::nullopt;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return std::nullopt;
    }

    if (!pe_id_provided) {
        std::cerr << "Error: --pe-id is required.\n";
        return std::nullopt;
    }

    // Resolve the set of core counts to test. --block-list (if given) wins over
    // -b/--block-range. block_size_min/max are kept consistent with the resolved
    // set because they drive the symmetric-memory allocation upper bound and the
    // CSV file name.
    if (!config.block_sizes.empty()) {
        auto minmax = std::minmax_element(config.block_sizes.begin(), config.block_sizes.end());
        config.block_size_min = *minmax.first;
        config.block_size_max = *minmax.second;
    }

    if (!validate_config(config)) {
        return std::nullopt;
    }

    // No explicit list: expand the validated [min, max] range into block_sizes.
    if (config.block_sizes.empty()) {
        for (int b = config.block_size_min; b <= config.block_size_max; ++b) {
            config.block_sizes.push_back(b);
        }
    }

    return config;
}

#endif // SIMT_RMA_UB2GM_PERFTEST_ARGPARSER
