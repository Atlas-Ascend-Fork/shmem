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
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"
#include "udma_put_value_reduce_kernel.h"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

int g_npus = 8;
const char* ipport;
int f_pe = 0;
int f_npu = 0;
constexpr uint64_t DEBUG_DUMP_SIZE = 200 * 1024 * 1024;

aclshmemx_uniqueid_t default_flag_uid;

static int32_t src_value(int32_t pe, uint32_t elem) { return pe * 100 + static_cast<int32_t>(elem); }

// Seed the destination one below, equal to, or one above the source so both operands win for
// MAX/MIN depending on the element index.
static int32_t reduce_seed_value(int32_t pe, uint32_t elem)
{
    return src_value(pe, elem) + static_cast<int32_t>(elem % 3) - 1;
}

static void prepare_input(std::vector<int32_t>& host_data, int n_pes)
{
    const size_t slot_words = SLOT_BYTES / sizeof(int32_t);
    for (int32_t pe = 0; pe < n_pes; ++pe) {
        int32_t* src = host_data.data() + static_cast<size_t>(pe) * REGIONS_PER_PE * slot_words;
        int32_t* sum_dst = src + slot_words;
        int32_t* max_dst = src + 2 * slot_words;
        int32_t* min_dst = src + 3 * slot_words;
        for (uint32_t i = 0; i < ELEMS_PER_SLOT; ++i) {
            src[i] = src_value(pe, i);
            sum_dst[i] = 1;
            max_dst[i] = reduce_seed_value(pe, i);
            min_dst[i] = reduce_seed_value(pe, i);
        }
    }
}

static bool check_reduce_result(const std::vector<int32_t>& host_data, int n_pes, int pe_id)
{
    const size_t slot_words = SLOT_BYTES / sizeof(int32_t);
    for (int32_t pe = 0; pe < n_pes; ++pe) {
        if (pe == pe_id) {
            continue; // a PE does not send to itself, so its own slot keeps the host-initialized value
        }
        const int32_t* src = host_data.data() + static_cast<size_t>(pe) * REGIONS_PER_PE * slot_words;
        const int32_t* sum_dst = src + slot_words;
        const int32_t* max_dst = src + 2 * slot_words;
        const int32_t* min_dst = src + 3 * slot_words;
        // UDMA WRITE_WITH_REDUCE reduces element-wise: dst[i] OP src[i].
        for (uint32_t i = 0; i < ELEMS_PER_SLOT; ++i) {
            const int32_t golden = src_value(pe, i);
            const int32_t seed = reduce_seed_value(pe, i);
            const int32_t expected_sum = 1 + golden;
            const int32_t expected_max = std::max(golden, seed);
            const int32_t expected_min = std::min(golden, seed);
            if (sum_dst[i] != expected_sum || max_dst[i] != expected_max || min_dst[i] != expected_min) {
                std::cout << "pe" << pe_id << ": reduce mismatch from pe" << pe << ", elem=" << i
                          << ", sum=" << sum_dst[i] << "(expect " << expected_sum << ")"
                          << ", max=" << max_dst[i] << "(expect " << expected_max << ")"
                          << ", min=" << min_dst[i] << "(expect " << expected_min << ")" << std::endl;
                return false;
            }
        }

        uint64_t actual_value = 0;
        std::memcpy(&actual_value, src + 4 * slot_words, sizeof(actual_value));
        const uint64_t expected_value = VALUE_MAGIC | static_cast<uint64_t>(pe);
        if (actual_value != expected_value) {
            std::cout << "pe" << pe_id << ": inline value mismatch from pe" << pe << ": " << actual_value << "(expect "
                      << expected_value << ")" << std::endl;
            return false;
        }
    }
    return true;
}

int test_aclshmem_udma_put_value_reduce(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    uint8_t* ptr = nullptr;
    uint8_t* dump = nullptr;
    bool acl_inited = false;
    bool device_set = false;
    bool stream_created = false;
    bool shmem_inited = false;

    do {
        status = aclInit(nullptr);
        if (status != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": aclInit failed, ret=" << status << std::endl;
            break;
        }
        acl_inited = true;

        status = aclrtSetDevice(device_id);
        if (status != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": aclrtSetDevice(" << device_id << ") failed, ret=" << status << std::endl;
            break;
        }
        device_set = true;

        status = aclrtCreateStream(&stream);
        if (status != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": aclrtCreateStream failed, ret=" << status << std::endl;
            break;
        }
        stream_created = true;

        aclshmemx_init_attr_t attributes;
        status = test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
        if (status != 0) {
            std::cout << "pe" << pe_id << ": test_set_attr failed, ret=" << status << std::endl;
            break;
        }
        attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
        if (status != 0) {
            std::cout << "pe" << pe_id << ": aclshmemx_init_attr failed, ret=" << status << std::endl;
            break;
        }
        shmem_inited = true;

        const size_t slot_words = SLOT_BYTES / sizeof(int32_t);
        const size_t total_words = static_cast<size_t>(n_pes) * REGIONS_PER_PE * slot_words;
        const size_t total_bytes = total_words * sizeof(int32_t);
        ptr = static_cast<uint8_t*>(aclshmem_malloc(total_bytes));
        if (ptr == nullptr) {
            std::cout << "pe" << pe_id << ": failed to allocate symmetric memory, bytes=" << total_bytes << std::endl;
            status = -1;
            break;
        }

        std::vector<int32_t> host_data(total_words, 0);
        prepare_input(host_data, n_pes);
        status = aclrtMemcpy(ptr, total_bytes, host_data.data(), total_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": H2D copy of initialization data failed, ret=" << status << std::endl;
            break;
        }

        // Wait for every PE to finish the H2D initialization before any PE starts
        // writing to peers, otherwise a late H2D copy can overwrite received data.
        status = aclshmem_barrier(ACLSHMEM_TEAM_WORLD);
        if (status != 0) {
            std::cout << "pe" << pe_id << ": failed to synchronize H2D initialization, ret=" << status << std::endl;
            break;
        }

#if defined(ENABLE_ASCENDC_DUMP)
        status = aclrtMalloc(reinterpret_cast<void**>(&dump), DEBUG_DUMP_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS || dump == nullptr) {
            std::cout << "pe" << pe_id << ": aclrtMalloc for dump failed, ret=" << status << std::endl;
            status = (status != ACL_SUCCESS) ? status : -1;
            break;
        }
#endif

        launch_udma_put_value_reduce(1, stream, ptr, dump);
        status = aclrtSynchronizeStream(stream);
        if (status != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": kernel execution failed, ret=" << status << std::endl;
            break;
        }

        // Wait for every PE's kernel to finish before reading back the symmetric buffer.
        status = aclshmem_barrier(ACLSHMEM_TEAM_WORLD);
        if (status != 0) {
            std::cout << "pe" << pe_id << ": failed to synchronize peer writes, ret=" << status << std::endl;
            break;
        }

#if defined(ENABLE_ASCENDC_DUMP)
        Adx::AdumpPrintWorkSpace(dump, DEBUG_DUMP_SIZE, stream, "udma_put_value_reduce");
#endif

        status = aclrtMemcpy(host_data.data(), total_bytes, ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (status != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": D2H copy of results failed, ret=" << status << std::endl;
            break;
        }

        if (!check_reduce_result(host_data, n_pes, pe_id)) {
            status = -1;
            break;
        }
        std::cout << "check transport result success, pe=" << pe_id << std::endl;
    } while (false);

    if (shmem_inited) {
        if (ptr != nullptr) {
            aclshmem_free(ptr);
        }
        int ret = aclshmem_finalize();
        if (ret != 0) {
            std::cout << "pe" << pe_id << ": aclshmem_finalize failed, ret=" << ret << std::endl;
            if (status == 0) {
                status = ret;
            }
        }
    }
#if defined(ENABLE_ASCENDC_DUMP)
    if (dump != nullptr) {
        int ret = aclrtFree(dump);
        if (ret != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": aclrtFree(dump) failed, ret=" << ret << std::endl;
            if (status == 0) {
                status = ret;
            }
        }
    }
#endif
    if (stream_created) {
        int ret = aclrtDestroyStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": aclrtDestroyStream failed, ret=" << ret << std::endl;
            if (status == 0) {
                status = ret;
            }
        }
    }
    if (device_set) {
        int ret = aclrtResetDevice(device_id);
        if (ret != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": aclrtResetDevice failed, ret=" << ret << std::endl;
            if (status == 0) {
                status = ret;
            }
        }
    }
    if (acl_inited) {
        int ret = aclFinalize();
        if (ret != ACL_SUCCESS) {
            std::cout << "pe" << pe_id << ": aclFinalize failed, ret=" << ret << std::endl;
            if (status == 0) {
                status = ret;
            }
        }
    }
    return status;
}

int main(int argc, char* argv[])
{
    int arg_idx = 1;
    int status = 0;
    int n_pes = atoi(argv[arg_idx++]);
    int pe_id = atoi(argv[arg_idx++]);
    ipport = argv[arg_idx++];
    g_npus = atoi(argv[arg_idx++]);
    f_pe = atoi(argv[arg_idx++]);
    f_npu = atoi(argv[arg_idx++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_aclshmem_udma_put_value_reduce(pe_id, n_pes, local_mem_size);
    if (status != 0) {
        std::cout << "[FAILED] demo run failed in pe " << pe_id << ", status=" << status << std::endl;
    } else {
        std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    }
    return status;
}
