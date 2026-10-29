# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Used at host-UT configure time, or with cmake -D... -P for a standalone CPU test.
# Read the constant and complete function from production; never maintain a test copy.
function(extract_sync_dispatch source output)
    file(READ "${source}" source_text)
    string(REPLACE "\r\n" "\n" source_text "${source_text}")
    string(REGEX MATCHALL "constexpr int32_t SYNC_V3_SMALL_TEAM_SIZE = [^;]+" thresholds "${source_text}")
    list(LENGTH thresholds threshold_count)
    if(NOT threshold_count EQUAL 1)
        message(FATAL_ERROR "Expected one SYNC_V3_SMALL_TEAM_SIZE definition in ${source}")
    endif()

    set(signature "template <bool IS_AIV_ONLY = true>\nACLSHMEM_DEVICE void aclshmemi_sync(aclshmem_team_t team)\n{")
    string(FIND "${source_text}" "${signature}" begin)
    if(begin LESS 0)
        message(FATAL_ERROR "Cannot locate the production sync dispatcher in ${source}")
    endif()
    string(SUBSTRING "${source_text}" ${begin} -1 tail)
    # A top-level closing brace terminates the function; nested blocks are indented.
    string(FIND "${tail}" "\n}\n" end)
    if(end LESS 0)
        message(FATAL_ERROR "Cannot locate the end of the sync dispatcher in ${source}")
    endif()
    math(EXPR length "${end} + 3")
    string(SUBSTRING "${tail}" 0 ${length} dispatcher)
    string(FIND "${dispatcher}" "aclshmemi_sync_npu_v3<IS_AIV_ONLY>(team)" v3_call)
    string(FIND "${dispatcher}" "aclshmemi_sync_npu_v4<IS_AIV_ONLY>(team)" v4_call)
    if(v3_call LESS 0 OR v4_call LESS 0)
        message(FATAL_ERROR "Sync dispatcher extraction lost a path; update the CPU harness")
    endif()

    get_filename_component(output_dir "${output}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_dir}")
    file(WRITE "${output}" "// Generated from shmemi_device_cc.h for host UT only.\n${thresholds};\n\n${dispatcher}\n")
endfunction()

extract_sync_dispatch("${SYNC_DISPATCH_SOURCE}" "${SYNC_DISPATCH_OUTPUT}")
