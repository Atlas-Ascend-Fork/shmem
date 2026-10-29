/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <cstdlib>
#include <getopt.h>
#include "utils.h"
#include "atomic_perftest_common.h"

using namespace perftest;

static const char* const OP_NAMES[] = {"inc", "fetch_inc", "set", "add",       "fetch_add", "and",         "fetch_and",
                                       "or",  "fetch_or",  "xor", "fetch_xor", "swap",      "compare_swap"};
static_assert(sizeof(OP_NAMES) / sizeof(OP_NAMES[0]) == RDMA_ATOMIC_OP_INVALID, "One name per atomic operation");
constexpr uint64_t HEAP_RESERVE_BYTES = 1ULL << 30;

struct Options {
    int pes = 2;
    int pe = 0;
    int npus = 2;
    int first_npu = 0;
    int iterations = 1000;
    int warmup = PERFTEST_WARMUP_ITERS;
    int repetitions = 10;
    int sync_id = 0;
    uint64_t min_batch = 1;
    uint64_t max_batch = 4096;
    uint64_t step_factor = 2;
    std::string ipport = "tcp://127.0.0.1:8768";
    std::string test_name = "latency";
    std::string op_name = "inc";
    std::string type_name = "uint64";
    rdma_atomic_test_t test = RDMA_ATOMIC_TEST_LATENCY;
    rdma_atomic_op_t op = RDMA_ATOMIC_INC;
    perf_data_type_t data_type = DATA_TYPE_UINT64;
};

static void print_help()
{
    std::cout << "Usage: rdma_atomic_perftest [options]\n"
              << "  --test <batch|latency>   Default: latency\n"
              << "  --op <operation>        Default: inc; operations:";
    for (const char* name : OP_NAMES) {
        std::cout << " " << name;
    }
    std::cout << "\n  --data-type <uint32|uint64|int32|int64>  Default: uint64; batch requires uint64\n"
              << "  --iterations <n>        Latency operations per sample (default: 1000)\n"
              << "  --warmup-iterations <n> Warmup batches or latency operations (default: 100; 0 disables)\n"
              << "  --repetitions <n>       Measured repetitions (default: 10)\n"
              << "  --min-batch <n>         Minimum operations per batch (default: 1)\n"
              << "  --max-batch <n>         Maximum operations per batch (default: 4096)\n"
              << "  --step-factor <n>       Batch size multiplier, integer > 1 (default: 2)\n"
              << "  --pes <2>              Exactly two PEs (default: 2)\n"
              << "  --pe-id <0|1>           Process rank (default: 0)\n"
              << "  --ipport <tcp://ip:port> Bootstrap address (default: tcp://127.0.0.1:8768)\n"
              << "  --gnpus <n>             Local device count (default: 2)\n"
              << "  --fnpu <id>             First local device (default: 0)\n"
              << "  --sync-id <id>          RoCE synchronization event ID (default: 0)\n"
              << "  -h, --help              Print this help\n"
              << "Device = pe-id % gnpus + fnpu. Both PEs must use matching case parameters.\n"
              << "PE0 issues to PE1 using one AIV and the default single QP (QP0).\n"
              << "Batch mode measures one complete batch per sample; --iterations applies only to latency.\n";
}

template <typename T>
static bool parse_number(const char* text, uint64_t minimum, uint64_t maximum, T& result)
{
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno != 0 || *end != '\0' || value < minimum || value > maximum) {
        std::cerr << "Invalid numeric argument: " << text << std::endl;
        return false;
    }
    result = static_cast<T>(value);
    return true;
}

static int parse_options(int argc, char** argv, Options& opts)
{
    enum {
        PES = 256,
        PE,
        IPPORT,
        NPUS,
        FIRST_NPU,
        TEST,
        OP,
        TYPE,
        ITERATIONS,
        WARMUP,
        REPS,
        MIN_BATCH,
        MAX_BATCH,
        STEP,
        SYNC_ID
    };
    static const option long_options[] = {
        {"pes", required_argument, nullptr, PES},
        {"pe-id", required_argument, nullptr, PE},
        {"ipport", required_argument, nullptr, IPPORT},
        {"gnpus", required_argument, nullptr, NPUS},
        {"fnpu", required_argument, nullptr, FIRST_NPU},
        {"test", required_argument, nullptr, TEST},
        {"op", required_argument, nullptr, OP},
        {"data-type", required_argument, nullptr, TYPE},
        {"iterations", required_argument, nullptr, ITERATIONS},
        {"warmup-iterations", required_argument, nullptr, WARMUP},
        {"repetitions", required_argument, nullptr, REPS},
        {"min-batch", required_argument, nullptr, MIN_BATCH},
        {"max-batch", required_argument, nullptr, MAX_BATCH},
        {"step-factor", required_argument, nullptr, STEP},
        {"sync-id", required_argument, nullptr, SYNC_ID},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
        switch (opt) {
            case TEST:
                opts.test_name = optarg;
                break;
            case OP:
                opts.op_name = optarg;
                break;
            case TYPE:
                opts.type_name = optarg;
                break;
            case IPPORT:
                opts.ipport = optarg;
                break;
            case PES:
                if (!parse_number(optarg, 2, 2, opts.pes))
                    return -1;
                break;
            case PE:
                if (!parse_number(optarg, 0, 1, opts.pe))
                    return -1;
                break;
            case NPUS:
                if (!parse_number(optarg, 1, INT_MAX, opts.npus))
                    return -1;
                break;
            case FIRST_NPU:
                if (!parse_number(optarg, 0, INT_MAX - 1, opts.first_npu))
                    return -1;
                break;
            case ITERATIONS:
                if (!parse_number(optarg, 1, INT_MAX, opts.iterations))
                    return -1;
                break;
            case WARMUP:
                if (!parse_number(optarg, 0, INT_MAX, opts.warmup))
                    return -1;
                break;
            case REPS:
                if (!parse_number(optarg, 1, INT_MAX, opts.repetitions))
                    return -1;
                break;
            case MIN_BATCH:
                if (!parse_number(optarg, 1, SIZE_MAX / sizeof(uint64_t), opts.min_batch))
                    return -1;
                break;
            case MAX_BATCH:
                if (!parse_number(optarg, 1, SIZE_MAX / sizeof(uint64_t), opts.max_batch))
                    return -1;
                break;
            case STEP:
                if (!parse_number(optarg, 2, UINT64_MAX, opts.step_factor))
                    return -1;
                break;
            case SYNC_ID:
                if (!parse_number(optarg, 0, INT_MAX, opts.sync_id))
                    return -1;
                break;
            case 'h':
                print_help();
                return 1;
            default:
                return -1;
        }
    }
    opts.test = opts.test_name == "batch"   ? RDMA_ATOMIC_TEST_BATCH :
                opts.test_name == "latency" ? RDMA_ATOMIC_TEST_LATENCY :
                                              RDMA_ATOMIC_TEST_INVALID;
    opts.op = RDMA_ATOMIC_OP_INVALID;
    for (int i = 0; i < RDMA_ATOMIC_OP_INVALID; ++i) {
        if (opts.op_name == OP_NAMES[i])
            opts.op = static_cast<rdma_atomic_op_t>(i);
    }
    if (opts.type_name == "uint32")
        opts.data_type = DATA_TYPE_UINT32;
    else if (opts.type_name == "uint64")
        opts.data_type = DATA_TYPE_UINT64;
    else if (opts.type_name == "int32")
        opts.data_type = DATA_TYPE_INT32;
    else if (opts.type_name == "int64")
        opts.data_type = DATA_TYPE_INT64;
    else {
        std::cerr << "--data-type must be uint32, uint64, int32 or int64" << std::endl;
        return -1;
    }
    if (optind != argc || opts.ipport.empty() || opts.ipport.size() >= ACLSHMEM_MAX_IP_PORT_LEN ||
        opts.test == RDMA_ATOMIC_TEST_INVALID || opts.op == RDMA_ATOMIC_OP_INVALID) {
        std::cerr << "Invalid test, operation, bootstrap address or positional argument; see --help" << std::endl;
        return -1;
    }
    if (opts.test == RDMA_ATOMIC_TEST_BATCH &&
        (opts.data_type != DATA_TYPE_UINT64 || opts.min_batch > opts.max_batch ||
         opts.max_batch >
             (ACLSHMEM_MAX_LOCAL_SIZE - HEAP_RESERVE_BYTES - 2 * sizeof(rdma_atomic_result_t)) / sizeof(uint64_t) ||
         opts.max_batch > UINT64_MAX / static_cast<uint64_t>(std::max(1, opts.warmup)))) {
        std::cerr << "Batch requires uint64, ordered positive batch sizes, a heap no larger than "
                  << ACLSHMEM_MAX_LOCAL_SIZE << " bytes, and non-overflowing warmup op counts" << std::endl;
        return -1;
    }
    return 0;
}

#ifdef ACLSHMEMI_RDMA_K_BACKEND_XSCALE
static double get_cycles_per_us()
{
    const char* soc = aclrtGetSocName();
    return soc != nullptr && std::string(soc).find("Ascend950") != std::string::npos ? 1000.0 : 50.0;
}

static std::string format_number(double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

static bool save_csv(const std::string& path, const CsvDataTable& rows)
{
    if (!make_dir(get_dir(path)))
        return false;
    std::ofstream out(path);
    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i != 0)
                out << ',';
            out << row[i];
        }
        out << '\n';
    }
    out.close();
    if (!out)
        std::cerr << "Failed to write CSV: " << path << std::endl;
    return static_cast<bool>(out);
}

static void append_summary(
    const Options& opts, uint64_t batch_size, const char* metric, std::vector<double> samples, CsvDataTable& summary)
{
    std::vector<std::string> row = {
        "ROCE",
        "1",
        "2",
        "0",
        "1",
        opts.test_name,
        opts.op_name,
        opts.type_name,
        std::to_string(batch_size),
        metric,
        std::to_string(samples.size()),
        "",
        "",
        "",
        ""};
    if (!samples.empty()) {
        std::sort(samples.begin(), samples.end());
        const size_t n = samples.size();
        const double median = n % 2 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2;
        const size_t p95 = static_cast<size_t>(std::ceil(0.95 * n)) - 1;
        row[11] = format_number(samples.front());
        row[12] = format_number(median);
        row[13] = format_number(samples[p95]);
        row[14] = format_number(samples.back());
    }
    std::cout << opts.test_name << '/' << opts.op_name << '/' << opts.type_name << " batch=" << batch_size << ' '
              << metric << " samples=" << row[10] << " min=" << row[11] << " median=" << row[12] << " p95=" << row[13]
              << " max=" << row[14] << std::endl;
    summary.push_back(row);
}

template <typename T>
static bool verify_results(
    const Options& opts, uint64_t batch_size, int iterations, const rdma_atomic_result_t* results)
{
    const auto& target = results[1];
    const uint64_t total_ops = batch_size * iterations;
    bool verified = target.total_ops == total_ops && target.mismatch_count == 0 &&
                    target.first_mismatch_index == UINT64_MAX &&
                    target.expected_bits == atomic_expected_bits<T>(opts.op, iterations);
    if (!verified) {
        std::cerr << "PE" << opts.pe << " target verification failed: mismatches=" << target.mismatch_count
                  << " first_index=" << target.first_mismatch_index << " actual_bits=" << target.first_actual_bits
                  << " expected_bits=" << target.expected_bits << std::endl;
    }
    if (opts.pe == 0) {
        const auto& initiator = results[0];
        const uint64_t expected_fetch = atomic_expected_fetch<T>(opts.op, iterations);
        if (initiator.total_ops != total_ops || initiator.elapsed_cycles <= 0 ||
            initiator.last_fetch_bits != expected_fetch) {
            std::cerr << "PE0 initiator verification failed: ops=" << initiator.total_ops
                      << " cycles=" << initiator.elapsed_cycles << " last_fetch_bits=" << initiator.last_fetch_bits
                      << " expected_fetch_bits=" << expected_fetch << std::endl;
            verified = false;
        }
    }
    return verified;
}

template <typename T>
static int run_point(
    const Options& opts, uint64_t batch_size, aclrtStream stream, uint64_t ffts_addr, CsvDataTable& raw,
    CsvDataTable& summary, bool& failed)
{
    const uint64_t bytes = batch_size * sizeof(T);
    const bool batch = opts.test == RDMA_ATOMIC_TEST_BATCH;
    const int measured_iterations = batch ? 1 : opts.iterations;
    const size_t target_bytes = (bytes + 63) & ~size_t(63);
    const size_t alloc_bytes = target_bytes + 2 * sizeof(rdma_atomic_result_t);
    std::vector<uint8_t> initial(alloc_bytes, 0);
    const T seed = atomic_initial_value<T>(opts.op);
    for (uint64_t idx = 0; idx < batch_size; ++idx) {
        std::memcpy(initial.data() + idx * sizeof(T), &seed, sizeof(T));
    }
    auto allocation = static_cast<uint8_t*>(aclshmem_malloc(alloc_bytes));
    if (allocation == nullptr) {
        std::cerr << "aclshmem_malloc failed for " << alloc_bytes << " bytes" << std::endl;
        return 1;
    }
    auto result_gva = allocation + target_bytes;
    std::vector<double> rates, latencies;
    const double cycles_per_us = get_cycles_per_us();
    int status = 0;
    // Warmup uses the same device flow. Every measured repetition starts from fresh targets and result slots.
    for (int64_t repetition = 0; repetition <= opts.repetitions; ++repetition) {
        const int iterations = repetition == 0 ? opts.warmup : measured_iterations;
        if (iterations == 0)
            continue;
        status = aclrtMemcpy(allocation, alloc_bytes, initial.data(), alloc_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != 0)
            break;
        launch_rdma_atomic_perf_kernel(
            1, stream, ffts_addr, allocation, result_gva, batch_size, opts.op, opts.data_type, iterations,
            opts.sync_id);
        status = aclrtSynchronizeStream(stream);
        if (status != 0)
            break;
        if (repetition == 0)
            continue;
        rdma_atomic_result_t results[2]{};
        const size_t offset = opts.pe == 0 ? 0 : sizeof(rdma_atomic_result_t);
        const size_t count = opts.pe == 0 ? sizeof(results) : sizeof(rdma_atomic_result_t);
        status = aclrtMemcpy(
            reinterpret_cast<uint8_t*>(results) + offset, count, result_gva + offset, count, ACL_MEMCPY_DEVICE_TO_HOST);
        if (status != 0)
            break;
        const bool verified = verify_results<T>(opts, batch_size, measured_iterations, results);
        failed |= !verified;
        if (opts.pe != 0)
            continue;
        const double elapsed_us = results[0].elapsed_cycles / cycles_per_us;
        const double rate = elapsed_us > 0 ? results[0].total_ops / elapsed_us : 0;
        const double latency = elapsed_us / measured_iterations;
        raw.push_back(
            {"ROCE",
             "1",
             "2",
             "0",
             "1",
             opts.test_name,
             opts.op_name,
             opts.type_name,
             std::to_string(sizeof(T)),
             std::to_string(batch_size),
             std::to_string(results[0].total_ops),
             std::to_string(measured_iterations),
             std::to_string(opts.warmup),
             std::to_string(repetition),
             std::to_string(results[0].elapsed_cycles),
             format_number(elapsed_us),
             batch ? format_number(elapsed_us) : "",
             batch ? format_number(rate) : "",
             batch ? "" : format_number(latency),
             verified ? "true" : "false"});
        if (verified) {
            rates.push_back(rate);
            latencies.push_back(latency);
        }
    }
    if (status == 0) {
        aclshmemx_barrier_all_on_stream(stream);
        status = aclrtSynchronizeStream(stream);
    }
    aclshmem_free(allocation);
    if (status != 0) {
        std::cerr << "Device copy or stream synchronization failed: " << status << std::endl;
        return status;
    }
    if (opts.pe == 0) {
        if (batch) {
            append_summary(opts, batch_size, "BatchLatencyUs", latencies, summary);
            append_summary(opts, batch_size, "AtomicRateMops", rates, summary);
        } else {
            append_summary(opts, batch_size, "LatencyUs", latencies, summary);
        }
    }
    return 0;
}

template <typename T>
static int run_points(
    const Options& opts, aclrtStream stream, uint64_t ffts_addr, CsvDataTable& raw, CsvDataTable& summary, bool& failed)
{
    if (opts.test == RDMA_ATOMIC_TEST_LATENCY) {
        return run_point<T>(opts, 1, stream, ffts_addr, raw, summary, failed);
    }
    for (uint64_t batch_size = opts.min_batch; batch_size <= opts.max_batch;) {
        const int status = run_point<T>(opts, batch_size, stream, ffts_addr, raw, summary, failed);
        if (status != 0)
            return status;
        if (batch_size > opts.max_batch / opts.step_factor)
            break;
        batch_size *= opts.step_factor;
    }
    return 0;
}

static int run_perftest(const Options& opts)
{
    const int device = opts.pe % opts.npus + opts.first_npu;
    const uint64_t max_bytes =
        opts.test == RDMA_ATOMIC_TEST_BATCH ? opts.max_batch * sizeof(uint64_t) : sizeof(uint64_t);
    // HBM reservation requires the complete heap size to be aligned to SHMEM pages.
    const uint64_t heap_bytes = ALIGN_TO(
        HEAP_RESERVE_BYTES + ((max_bytes + 63) & ~uint64_t(63)) + 2 * sizeof(rdma_atomic_result_t), ACLSHMEM_PAGE_SIZE);
    aclrtStream stream = nullptr;
    aclshmemx_init_attr_t attributes{};
    aclshmemx_uniqueid_t uid{};
    bool acl_initialized = false, device_set = false, shmem_initialized = false, failed = false;
    int ret = 0;
    uint64_t ffts_addr = 0;
    CsvDataTable raw{
        {"Backend",
         "QPCount",
         "PEs",
         "InitiatorPE",
         "TargetPE",
         "Test",
         "Op",
         "DataType",
         "OperandBytes",
         "BatchSize",
         "TotalOps",
         "Iterations",
         "WarmupIterations",
         "Repetition",
         "ElapsedCycles",
         "ElapsedUs",
         "BatchLatencyUs",
         "AtomicRateMops",
         "LatencyUs",
         "Verified"}};
    CsvDataTable summary{
        {"Backend", "QPCount", "PEs", "InitiatorPE", "TargetPE", "Test", "Op", "DataType", "BatchSize", "Metric",
         "Samples", "Min", "Median", "P95", "Max"}};

#define CHECK_GOTO(call)                                           \
    do {                                                           \
        ret = (call);                                              \
        if (ret != 0) {                                            \
            std::cerr << #call << " failed: " << ret << std::endl; \
            goto cleanup;                                          \
        }                                                          \
    } while (0)

    CHECK_GOTO(aclInit(nullptr));
    acl_initialized = true;
    CHECK_GOTO(aclrtSetDevice(device));
    device_set = true;
    CHECK_GOTO(aclrtCreateStream(&stream));
    CHECK_GOTO(test_set_attr(opts.pe, opts.pes, heap_bytes, opts.ipport.c_str(), uid, &attributes));
    attributes.comm_args = &uid;
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    // The transport default is one QP. Reserve one WQE staging buffer for the only active AIV.
    CHECK_GOTO(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));
    shmem_initialized = true;
    ffts_addr = util_get_ffts_config();
    CHECK_GOTO(aclshmemx_set_rdma_config(0, ACLSHMEM_RDMA_MTE_STAGING_UB_SIZE, opts.sync_id));
    std::cout << "ROCE atomic PE" << opts.pe << " device=" << device
              << " QPCount=1 AIV=1 cycles_per_us=" << get_cycles_per_us() << std::endl;
    switch (opts.data_type) {
#define RUN_ATOMIC_TYPE(type_enum, type)                                             \
    case type_enum:                                                                  \
        CHECK_GOTO(run_points<type>(opts, stream, ffts_addr, raw, summary, failed)); \
        break;
        RUN_ATOMIC_TYPE(DATA_TYPE_UINT32, uint32_t)
        RUN_ATOMIC_TYPE(DATA_TYPE_UINT64, uint64_t)
        RUN_ATOMIC_TYPE(DATA_TYPE_INT32, int32_t)
        RUN_ATOMIC_TYPE(DATA_TYPE_INT64, int64_t)
#undef RUN_ATOMIC_TYPE
        default:
            ret = 1;
            break;
    }
    if (opts.pe == 0) {
        const std::string base = "output/rdma_atomic_" + opts.test_name + "_" + opts.op_name + "_" + opts.type_name;
        const bool raw_saved = save_csv(base + "_pe0.csv", raw);
        const bool summary_saved = save_csv(base + "_summary.csv", summary);
        failed |= !raw_saved || !summary_saved;
    }
    ret = failed ? 1 : ret;

cleanup:
#define CHECK_CLEANUP(call)                                           \
    do {                                                              \
        const int status = (call);                                    \
        if (status != 0) {                                            \
            std::cerr << #call << " failed: " << status << std::endl; \
            if (ret == 0)                                             \
                ret = status;                                         \
        }                                                             \
    } while (0)
    if (shmem_initialized)
        CHECK_CLEANUP(aclshmem_finalize());
    if (stream != nullptr)
        CHECK_CLEANUP(aclrtDestroyStream(stream));
    if (device_set)
        CHECK_CLEANUP(aclrtResetDevice(device));
    if (acl_initialized)
        CHECK_CLEANUP(aclFinalize());
#undef CHECK_GOTO
#undef CHECK_CLEANUP
    return ret;
}

#endif // ACLSHMEMI_RDMA_K_BACKEND_XSCALE

int main(int argc, char** argv)
{
    Options opts;
    const int status = parse_options(argc, argv, opts);
    if (status != 0)
        return status > 0 ? 0 : 1;
#ifdef ACLSHMEMI_RDMA_K_BACKEND_XSCALE
    return run_perftest(opts);
#else
    std::cerr << "Atomic perftest is not enabled for this RDMA backend; currently enable it with "
                 "-rdma_backend XSCALE"
              << std::endl;
    return 1;
#endif
}
