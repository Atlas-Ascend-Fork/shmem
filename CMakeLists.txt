# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(CMAKE_LINK_DEPENDS_NO_SHARED 1)

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER bisheng CACHE FILEPATH "C compiler")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER bisheng CACHE FILEPATH "CXX compiler")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)

project(SHMEM LANGUAGES C CXX)

# 生成 compile_commands.json，供 clang-tidy 等工具使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

# 配置 ccache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "ccache found: ${CCACHE_FOUND}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    execute_process(COMMAND ${CCACHE_FOUND} --version OUTPUT_VARIABLE CCACHE_V)
    message(STATUS "ccache version info: ${CCACHE_V}")
else()
    message(STATUS "ccache not found. Compilation will proceed without it.")
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_SKIP_RPATH TRUE)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 设置安装路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install/shmem)

# 获取CANN相关环境变量
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)
if(USE_CXX11_ABI)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
else()
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# AscendC head files
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(PLUGIN_VERSION 0)

# 分界版本时间戳：xasc 与 xcce 编译模式切换
set(TARGET_TIMESTAMP 20260327)
# Ascend950 支持 --cce-enable-sanitizer 的 bisheng 起始时间戳
set(MSSANITIZER_CCE_FLAG_TIMESTAMP 20260701)

set(CURRENT_TIMESTAMP 0)
set(IS_NEW_VERSION FALSE)

# 通过 bisheng -v 获取编译器版本时间
execute_process(
    COMMAND bisheng -v
    OUTPUT_VARIABLE BISHENG_STDOUT
    ERROR_VARIABLE BISHENG_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

set(BISHENG_VERSION_OUTPUT "${BISHENG_STDOUT} ${BISHENG_STDERR}")

if(BISHENG_VERSION_OUTPUT)
    message(STATUS "bisheng -v output: ${BISHENG_VERSION_OUTPUT}")

    string(REGEX MATCH "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])" DATE_MATCH "${BISHENG_VERSION_OUTPUT}")
    if(DATE_MATCH)
        string(REGEX REPLACE "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])" "\\1\\2\\3" CURRENT_TIMESTAMP "${DATE_MATCH}")
        message(STATUS "Extracted date: ${DATE_MATCH}, timestamp: ${CURRENT_TIMESTAMP}")

        if(CURRENT_TIMESTAMP GREATER_EQUAL 20260327)
            set(IS_NEW_VERSION TRUE)
            message(STATUS "Detected NEW version (${CURRENT_TIMESTAMP} >= 20260327)")
        else()
            set(IS_NEW_VERSION FALSE)
            message(STATUS "Detected OLD version (${CURRENT_TIMESTAMP} < 20260327)")
        endif()
    else()
        message(WARNING "Failed to extract date from bisheng -v output, treat as OLD version")
        set(IS_NEW_VERSION FALSE)
    endif()
else()
    message(WARNING "Failed to run bisheng -v, treat as OLD version")
    set(IS_NEW_VERSION FALSE)
endif()

# mssanitizer 编译选项：依赖 bisheng 版本时间戳判断是否启用 --cce-enable-sanitizer
if(USE_MSSANITIZER)
    add_compile_definitions(ACLSHMEM_MSSANITIZER_BUILD)
    if(SOC_TYPE STREQUAL "Ascend950")
        if(CURRENT_TIMESTAMP GREATER_EQUAL ${MSSANITIZER_CCE_FLAG_TIMESTAMP})
            set(SANITIZER_FLAGS
                -g --cce-enable-sanitizer
            )
            message(STATUS
                "Ascend950 mssanitizer: current bisheng timestamp=${CURRENT_TIMESTAMP}, "
                "required timestamp=${MSSANITIZER_CCE_FLAG_TIMESTAMP}; "
                "enable -g --cce-enable-sanitizer (AscendC API memory checks available)")
        else()
            set(SANITIZER_FLAGS -g)
            message(WARNING
                "Ascend950 mssanitizer: current bisheng timestamp=${CURRENT_TIMESTAMP}, "
                "required timestamp=${MSSANITIZER_CCE_FLAG_TIMESTAMP}. "
                "Current CANN does not support compiling Ascend950 with --cce-enable-sanitizer; "
                "falling back to SANITIZER_FLAGS=-g only, AscendC API memory checks are unavailable. "
                "Upgrade CANN to enable full mssanitizer instrumentation.")
        endif()
    else()
        set(SANITIZER_FLAGS
            -g --cce-enable-sanitizer
        )
    endif()
else()
    set(SANITIZER_FLAGS)
endif()

message(STATUS "SOC_TYPE:${SOC_TYPE}")

if(SOC_TYPE STREQUAL "Ascend950")
    set(SHMEM_BACKEND_NAME "950")
else()
    set(SHMEM_BACKEND_NAME "910")
endif()
message(STATUS "SHMEM_BACKEND_NAME:${SHMEM_BACKEND_NAME}")

if(IS_NEW_VERSION)
    set(CCE_AICORE_ARCH "")
    set(CCE_AICORE_ARCH_VEC "")
    set(FFTS_COMPILE_FLAG "")

    if(SOC_TYPE STREQUAL "Ascend950")
        set(NPU_ARCH dav-3510)
    else()
        set(NPU_ARCH dav-2201)
        set(FFTS_COMPILE_FLAG --cce-auto-set-ffts=false)
        add_compile_definitions(FFTS_CONFIG=1)
    endif()
    set(CMAKE_CCE_COMPILE_OPTIONS
        -xasc
        -Xhost-start -ftrapv -Xhost-end
        --npu-arch=${NPU_ARCH}
        --cce-auto-infer-kernel-type=false
        ${FFTS_COMPILE_FLAG}
    )
else()
    if(SOC_TYPE STREQUAL "Ascend950")
        set(CCE_AICORE_ARCH --cce-aicore-arch=dav-c310)
        set(CCE_AICORE_ARCH_VEC --cce-aicore-arch=dav-c310-vec)
    else()
        set(CCE_AICORE_ARCH --cce-aicore-arch=dav-c220)
        set(CCE_AICORE_ARCH_VEC --cce-aicore-arch=dav-c220-vec)
        add_compile_definitions(FFTS_CONFIG=1)
    endif()
    set(NPU_ARCH "")
    set(CMAKE_CCE_COMPILE_OPTIONS
        -xcce
        -Xhost-start -ftrapv -Xhost-end
        "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
        "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
        "SHELL:-mllvm -cce-aicore-record-overflow=true"
        "SHELL:-mllvm -cce-aicore-addr-transform"
        "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    )
endif()
message(STATUS "CMAKE_CCE_COMPILE_OPTIONS: ${CMAKE_CCE_COMPILE_OPTIONS}")

# CANN 9.2 renamed the scalar GM cache-bypass APIs. Inspect declarations in
# the installed AscendC headers instead of inferring the API from a package
# version or compiler build date.
if(SOC_TYPE STREQUAL "Ascend950")
    set(SHMEM_MEGA_MOE_ASCENDC_ACTIVE_ARCH dav_c310)
else()
    set(SHMEM_MEGA_MOE_ASCENDC_ACTIVE_ARCH dav_c220)
endif()
set(SHMEM_MEGA_MOE_ASCENDC_SCALAR_HEADERS
    "${ASCEND_HOME_PATH}/asc/impl/basic_api/${SHMEM_MEGA_MOE_ASCENDC_ACTIVE_ARCH}/kernel_operator_scalar_intf_impl.h"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/${SHMEM_MEGA_MOE_ASCENDC_ACTIVE_ARCH}/kernel_operator_scalar_intf_impl.h"
    "${ASCEND_HOME_PATH}/asc/include/basic_api/kernel_operator_scalar_intf.h"
    "${ASCEND_HOME_PATH}/asc/impl/basic_api/kernel_operator_scalar_intf_impl.h"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api/kernel_operator_scalar_intf.h"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/kernel_operator_scalar_intf_impl.h")
file(GLOB_RECURSE SHMEM_MEGA_MOE_ASCENDC_SCALAR_HEADER_VARIANTS CONFIGURE_DEPENDS
    "${ASCEND_HOME_PATH}/asc/impl/basic_api/*/kernel_operator_scalar_intf_impl.h"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/*/kernel_operator_scalar_intf_impl.h")
list(APPEND SHMEM_MEGA_MOE_ASCENDC_SCALAR_HEADERS ${SHMEM_MEGA_MOE_ASCENDC_SCALAR_HEADER_VARIANTS})
list(REMOVE_DUPLICATES SHMEM_MEGA_MOE_ASCENDC_SCALAR_HEADERS)

set(SHMEM_MEGA_MOE_HAS_NEW_READ_API FALSE)
set(SHMEM_MEGA_MOE_HAS_NEW_WRITE_API FALSE)
set(SHMEM_MEGA_MOE_HAS_OLD_READ_API FALSE)
set(SHMEM_MEGA_MOE_HAS_OLD_WRITE_API FALSE)
foreach(SHMEM_MEGA_MOE_SCALAR_HEADER IN LISTS SHMEM_MEGA_MOE_ASCENDC_SCALAR_HEADERS)
    if(EXISTS "${SHMEM_MEGA_MOE_SCALAR_HEADER}")
        file(READ "${SHMEM_MEGA_MOE_SCALAR_HEADER}" SHMEM_MEGA_MOE_SCALAR_HEADER_CONTENT)
        string(REGEX REPLACE "//[^\\n\\r]*" "" SHMEM_MEGA_MOE_SCALAR_HEADER_CONTENT
            "${SHMEM_MEGA_MOE_SCALAR_HEADER_CONTENT}")
        string(REGEX MATCH "ReadGmBypassDCache[ \\t]*\\([^;{}]*__gm__"
            SHMEM_MEGA_MOE_NEW_READ_MATCH "${SHMEM_MEGA_MOE_SCALAR_HEADER_CONTENT}")
        string(REGEX MATCH "WriteGmBypassDCache[ \\t]*\\([^;{}]*__gm__"
            SHMEM_MEGA_MOE_NEW_WRITE_MATCH "${SHMEM_MEGA_MOE_SCALAR_HEADER_CONTENT}")
        string(REGEX MATCH "ReadGmByPassDCache[ \\t]*\\([^;{}]*__gm__"
            SHMEM_MEGA_MOE_OLD_READ_MATCH "${SHMEM_MEGA_MOE_SCALAR_HEADER_CONTENT}")
        string(REGEX MATCH "WriteGmByPassDCache[ \\t]*\\([^;{}]*__gm__"
            SHMEM_MEGA_MOE_OLD_WRITE_MATCH "${SHMEM_MEGA_MOE_SCALAR_HEADER_CONTENT}")
        if(SHMEM_MEGA_MOE_NEW_READ_MATCH)
            set(SHMEM_MEGA_MOE_HAS_NEW_READ_API TRUE)
        endif()
        if(SHMEM_MEGA_MOE_NEW_WRITE_MATCH)
            set(SHMEM_MEGA_MOE_HAS_NEW_WRITE_API TRUE)
        endif()
        if(SHMEM_MEGA_MOE_OLD_READ_MATCH)
            set(SHMEM_MEGA_MOE_HAS_OLD_READ_API TRUE)
        endif()
        if(SHMEM_MEGA_MOE_OLD_WRITE_MATCH)
            set(SHMEM_MEGA_MOE_HAS_OLD_WRITE_API TRUE)
        endif()
        if((SHMEM_MEGA_MOE_NEW_READ_MATCH AND SHMEM_MEGA_MOE_NEW_WRITE_MATCH) OR
           (SHMEM_MEGA_MOE_OLD_READ_MATCH AND SHMEM_MEGA_MOE_OLD_WRITE_MATCH))
            message(STATUS "Detected GM bypass API declarations in ${SHMEM_MEGA_MOE_SCALAR_HEADER}")
            break()
        endif()
    endif()
endforeach()

set(SHMEM_MEGA_MOE_DETECTED_BYPASS_DCACHE_MODE "AUTO")
if(SHMEM_MEGA_MOE_HAS_NEW_READ_API AND SHMEM_MEGA_MOE_HAS_NEW_WRITE_API)
    set(SHMEM_MEGA_MOE_DETECTED_BYPASS_DCACHE_MODE "NEW")
elseif(SHMEM_MEGA_MOE_HAS_OLD_READ_API AND SHMEM_MEGA_MOE_HAS_OLD_WRITE_API)
    set(SHMEM_MEGA_MOE_DETECTED_BYPASS_DCACHE_MODE "OLD")
else()
    message(WARNING "Could not detect a complete GM bypass API pair in CANN headers")
endif()

set(SHMEM_MEGA_MOE_BYPASS_DCACHE_MODE "AUTO" CACHE STRING
    "GM bypass API selection: AUTO, OLD, or NEW")
set_property(CACHE SHMEM_MEGA_MOE_BYPASS_DCACHE_MODE PROPERTY STRINGS AUTO OLD NEW)
if(SHMEM_MEGA_MOE_BYPASS_DCACHE_MODE STREQUAL "AUTO")
    if(SHMEM_MEGA_MOE_DETECTED_BYPASS_DCACHE_MODE STREQUAL "NEW")
        set(SHMEM_MEGA_MOE_USE_NEW_BYPASS_DCACHE TRUE)
        set(SHMEM_MEGA_MOE_SELECTED_BYPASS_DCACHE_MODE "NEW")
    elseif(SHMEM_MEGA_MOE_DETECTED_BYPASS_DCACHE_MODE STREQUAL "OLD")
        set(SHMEM_MEGA_MOE_USE_NEW_BYPASS_DCACHE FALSE)
        set(SHMEM_MEGA_MOE_SELECTED_BYPASS_DCACHE_MODE "OLD")
    else()
        set(SHMEM_MEGA_MOE_USE_NEW_BYPASS_DCACHE FALSE)
        set(SHMEM_MEGA_MOE_SELECTED_BYPASS_DCACHE_MODE "UNKNOWN")
    endif()
elseif(SHMEM_MEGA_MOE_BYPASS_DCACHE_MODE STREQUAL "NEW")
    set(SHMEM_MEGA_MOE_USE_NEW_BYPASS_DCACHE TRUE)
    set(SHMEM_MEGA_MOE_SELECTED_BYPASS_DCACHE_MODE "NEW")
elseif(SHMEM_MEGA_MOE_BYPASS_DCACHE_MODE STREQUAL "OLD")
    set(SHMEM_MEGA_MOE_USE_NEW_BYPASS_DCACHE FALSE)
    set(SHMEM_MEGA_MOE_SELECTED_BYPASS_DCACHE_MODE "OLD")
else()
    message(FATAL_ERROR "SHMEM_MEGA_MOE_BYPASS_DCACHE_MODE must be AUTO, OLD, or NEW")
endif()
message(STATUS "Detected GM bypass API mode: ${SHMEM_MEGA_MOE_DETECTED_BYPASS_DCACHE_MODE}")
message(STATUS "Selected GM bypass API mode: ${SHMEM_MEGA_MOE_SELECTED_BYPASS_DCACHE_MODE}")

option(USE_EXAMPLES "USE_EXAMPLES" OFF)
option(ENABLE_ASCENDC_DUMP "ENABLE_ASCENDC_DUMP" OFF)
message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")
message(STATUS "USE_EXAMPLES:${USE_EXAMPLES}")
message(STATUS "ENABLE_ASCENDC_DUMP:${ENABLE_ASCENDC_DUMP}")
option(ACLSHMEM_RDMA_SUPPORT "ACLSHMEM_RDMA_SUPPORT" OFF)
message(STATUS "ACLSHMEM_RDMA_SUPPORT:${ACLSHMEM_RDMA_SUPPORT}")

if(DEFINED ACLSHMEM_RDMA_BACKEND AND NOT ACLSHMEM_RDMA_SUPPORT)
    message(FATAL_ERROR "ACLSHMEM_RDMA_BACKEND requires ACLSHMEM_RDMA_SUPPORT to be ON. Please add -DACLSHMEM_RDMA_SUPPORT=ON to enable RDMA support.")
endif()

if(ACLSHMEM_RDMA_SUPPORT)
    if(SOC_TYPE STREQUAL "Ascend950")
        if(NOT DEFINED ACLSHMEM_RDMA_BACKEND)
            message(FATAL_ERROR "ACLSHMEM_RDMA_BACKEND is not set. Please specify -rdma_backend with value 'XSCALE' or 'HNS_1825' when SOC_TYPE is Ascend950.")
        endif()

        string(TOUPPER "${ACLSHMEM_RDMA_BACKEND}" RDMA_BACKEND_UPPER)
        if(RDMA_BACKEND_UPPER STREQUAL "XSCALE")
            add_compile_definitions(ACLSHMEMI_RDMA_K_BACKEND_XSCALE=1)
            message(STATUS "ACLSHMEM_RDMA_BACKEND: XSCALE")
        elseif(RDMA_BACKEND_UPPER STREQUAL "HNS_1825")
            add_compile_definitions(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1)
            message(STATUS "ACLSHMEM_RDMA_BACKEND: HNS_1825")
            # An explicit build setting takes precedence over local firmware detection.
            if(NOT DEFINED ACLSHMEMI_HNS_1825_PI_ON_CHIP)
                execute_process(COMMAND hinicadm5 info RESULT_VARIABLE result
                    OUTPUT_VARIABLE nic_info ERROR_VARIABLE error TIMEOUT 15)
                set(probe_errors "")
                set(firmware_modes "")
                set(devices "")
                if(result STREQUAL "0")
                    string(REPLACE "\r" "" nic_info "${nic_info}")
                    # Tool diagnostics may precede the card count without a newline.
                    string(REGEX MATCHALL "Card[ \t]+num[ \t]*:[^\n]*" count_rows "${nic_info}")
                    list(LENGTH count_rows count)
                    set(card_count 0)
                    if(count EQUAL 1 AND count_rows MATCHES "^Card[ \t]+num[ \t]*:[ \t]*([0-9]+)[ \t]*$")
                        set(card_count "${CMAKE_MATCH_1}")
                    endif()
                    # Enumerate card names, not PCIe functions or a presumed contiguous index range.
                    string(REGEX MATCHALL "(^|\n)[ \t]*\\|-+hinic[0-9]+\\(" card_rows "${nic_info}")
                    foreach(row IN LISTS card_rows)
                        string(REGEX MATCH "hinic[0-9]+" device "${row}")
                        if(device IN_LIST devices)
                            list(APPEND probe_errors "duplicate card ${device}")
                        else()
                            list(APPEND devices "${device}")
                        endif()
                    endforeach()
                    list(LENGTH devices count)
                    if(card_count EQUAL 0 OR NOT count EQUAL card_count)
                        list(APPEND probe_errors "hinicadm5 info returned an empty or incomplete card list")
                    endif()
                else()
                    list(APPEND probe_errors "hinicadm5 info failed (${result}): ${error}")
                endif()
                foreach(device IN LISTS devices)
                    execute_process(COMMAND hinicadm5 version -i "${device}" RESULT_VARIABLE result
                        OUTPUT_VARIABLE firmware ERROR_VARIABLE error TIMEOUT 15)
                    if(NOT result STREQUAL "0")
                        list(APPEND probe_errors "firmware query failed for ${device} (${result}): ${error}")
                        continue()
                    endif()
                    set(device_pi_on_chip TRUE)
                    foreach(component npu hiroce5)
                        if(component STREQUAL "npu")
                            set(component_label "npu[ \t]*:")
                        else()
                            set(component_label "hiroce5[ \t]+driver[ \t]*:")
                        endif()
                        string(REGEX MATCHALL "(^|\n)[ \t]*${component_label}[^\n]*" rows "${firmware}")
                        list(LENGTH rows count)
                        if(NOT count EQUAL 1)
                            list(APPEND probe_errors "cannot read ${component} version for ${device}")
                            set(device_pi_on_chip FALSE)
                            continue()
                        endif()
                        string(REGEX MATCH "${component_label}[ \t]*([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)([ \t]|$)"
                            version_match "${rows}")
                        if(NOT version_match)
                            list(APPEND probe_errors "cannot read ${component} version for ${device}")
                            set(device_pi_on_chip FALSE)
                            continue()
                        endif()
                        set(component_version "${CMAKE_MATCH_1}")
                        # Both NPU firmware and hiroce5 driver must be newer than 30.5.1.36.
                        if(NOT component_version VERSION_GREATER "30.5.1.36")
                            set(device_pi_on_chip FALSE)
                        endif()
                    endforeach()
                    if(device_pi_on_chip)
                        list(APPEND firmware_modes PI_ON_CHIP)
                    else()
                        list(APPEND firmware_modes LEGACY)
                    endif()
                endforeach()
                list(REMOVE_DUPLICATES firmware_modes)
                list(LENGTH firmware_modes mode_count)
                set(ACLSHMEMI_HNS_1825_PI_ON_CHIP 1)
                set(doorbell_mode PI_ON_CHIP)
                set(selection_reason "firmware detected")
                if(mode_count GREATER 1)
                    message(WARNING "HNS1825 firmware uses mixed doorbell modes; selecting PI_ON_CHIP. "
                        "Run the resulting kernels on cards with matching firmware.")
                    set(selection_reason "mixed firmware")
                endif()
                if(probe_errors)
                    string(JOIN "; " probe_errors ${probe_errors})
                    message(WARNING "HNS1825 firmware detection incomplete: ${probe_errors}. Defaulting to PI_ON_CHIP.")
                    set(selection_reason "fallback")
                elseif(firmware_modes STREQUAL "LEGACY")
                    set(ACLSHMEMI_HNS_1825_PI_ON_CHIP 0)
                    set(doorbell_mode LEGACY)
                endif()
            else()
                set(doorbell_mode PI_ON_CHIP)
                set(selection_reason "compile option")
                if(ACLSHMEMI_HNS_1825_PI_ON_CHIP STREQUAL "0")
                    set(doorbell_mode LEGACY)
                elseif(NOT ACLSHMEMI_HNS_1825_PI_ON_CHIP STREQUAL "1")
                    message(WARNING "ACLSHMEMI_HNS_1825_PI_ON_CHIP must be 0 or 1; defaulting to PI_ON_CHIP.")
                    set(ACLSHMEMI_HNS_1825_PI_ON_CHIP 1)
                    set(selection_reason "fallback")
                endif()
            endif()
            message(STATUS "HNS1825 doorbell mode: ${doorbell_mode} (${selection_reason})")
            add_compile_definitions(ACLSHMEMI_HNS_1825_PI_ON_CHIP=${ACLSHMEMI_HNS_1825_PI_ON_CHIP})
        else()
            message(FATAL_ERROR "Invalid ACLSHMEM_RDMA_BACKEND value '${ACLSHMEM_RDMA_BACKEND}'. Must be 'XSCALE' or 'HNS_1825'.")
        endif()
    elseif(DEFINED ACLSHMEM_RDMA_BACKEND)
        message(FATAL_ERROR "ACLSHMEM_RDMA_BACKEND can only be specified when SOC_TYPE is Ascend950.")
    endif()
endif()

option(ACLSHMEM_HCCS_SIO_LINK "ACLSHMEM_HCCS_SIO_LINK" OFF)

option(ACLSHMEM_SIMT_SUPPORT "ACLSHMEM_SIMT_SUPPORT" OFF)
message(STATUS "ACLSHMEM_SIMT_SUPPORT:${ACLSHMEM_SIMT_SUPPORT}")

if(SOC_TYPE STREQUAL "Ascend950")
    message(STATUS "HCOM UDMA compile support enabled; HCOM symbols are checked at runtime")
    add_compile_definitions(ACLSHMEM_SOC_950)
endif()

option(PYEXPAND_EXAMPLE "PYEXPAND_EXAMPLE" OFF)
message(STATUS "PYEXPAND_EXAMPLE:${PYEXPAND_EXAMPLE}")

message(STATUS "CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_MODE)
endif()

option(ACLSHMEM_RELAY_SUPPORT "ACLSHMEM_RELAY_SUPPORT" OFF)
message(STATUS "ACLSHMEM_RELAY_SUPPORT:${ACLSHMEM_RELAY_SUPPORT}")

option(ACLSHMEM_MULTI_LINK_SUPPORT "ACLSHMEM_MULTI_LINK_SUPPORT" OFF)
message(STATUS "ACLSHMEM_MULTI_LINK_SUPPORT:${ACLSHMEM_MULTI_LINK_SUPPORT}")

# relay 依赖 UDMA 传输，OFF 时无影响；ON 时校验前置依赖
if(ACLSHMEM_RELAY_SUPPORT AND NOT SOC_TYPE STREQUAL "Ascend950")
    message(FATAL_ERROR "ACLSHMEM_RELAY_SUPPORT requires SOC_TYPE=Ascend950.")
endif()

if(ACLSHMEM_RELAY_SUPPORT)
    add_compile_definitions(ACLSHMEM_RELAY_SUPPORT)
endif()

if(ACLSHMEM_MULTI_LINK_SUPPORT AND NOT SOC_TYPE STREQUAL "Ascend950")
    message(FATAL_ERROR "ACLSHMEM_MULTI_LINK_SUPPORT requires SOC_TYPE=Ascend950.")
endif()
if(ACLSHMEM_MULTI_LINK_SUPPORT AND ACLSHMEM_RELAY_SUPPORT)
    message(FATAL_ERROR "ACLSHMEM_MULTI_LINK_SUPPORT currently requires direct UDMA; disable relay support.")
endif()
if(ACLSHMEM_MULTI_LINK_SUPPORT)
    add_compile_definitions(ACLSHMEM_MULTI_LINK_SUPPORT)
endif()

if(ACLSHMEM_RDMA_SUPPORT AND SOC_TYPE STREQUAL "Ascend950")
    add_compile_definitions(ACLSHMEM_RDMA_V2_SUPPORT)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Werror
    # avoid ascendc interference
    -DL2_CACHE_HINT
    -DTILING_KEY_VAR
    -fstack-protector-strong
)

check_cxx_compiler_flag("-Wno-unused-value" HAS_WNO_UNUSED_VALUE)
if(HAS_WNO_UNUSED_VALUE)
    add_compile_options(-Wno-unused-value)
endif()

check_cxx_compiler_flag("-Wno-macro-redefined" HAS_WNO_MACRO_REDEFINED)
if(HAS_WNO_MACRO_REDEFINED)
    add_compile_options(-Wno-macro-redefined)
endif()

check_cxx_compiler_flag("-Wno-ignored-attributes" HAS_WNO_IGNORED_ATTRIBUTES)
if(HAS_WNO_IGNORED_ATTRIBUTES)
    add_compile_options(-Wno-ignored-attributes)
endif()

add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
    )

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/aclnn
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof

    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/hcomm

    # adapt to CANN eco package
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/interface
)

if(ACLSHMEM_RDMA_SUPPORT AND SOC_TYPE STREQUAL "Ascend950")
    include_directories(${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/hcomm)
endif()

link_directories(
    ${ASCEND_HOME_PATH}/lib64
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread)

# CANN releases expose the VMM share-handle APIs as one capability set, but the
# exact declarations are not available in every supported toolkit version.
# Compile and link a minimal source during configure so user-buffer-heap code is
# enabled only when the installed headers and ascendcl library agree.
include(CheckCXXSymbolExists)

# Create a test file to check for the symbol
set(CMAKE_REQUIRED_LIBRARIES ascendcl)
set(CMAKE_REQUIRED_INCLUDES ${ASCEND_HOME_PATH}/include)
set(CMAKE_REQUIRED_LINK_OPTIONS -L${ASCEND_HOME_PATH}/lib64)

# Try to compile a simple program that uses the symbol
file(WRITE ${CMAKE_BINARY_DIR}/check_acl_symbol.cpp "
#include <acl/acl.h>
int main() {
    aclrtMemFabricHandle handle;
    aclrtDrvMemHandle mem_handle = nullptr;
    aclrtMemRetainAllocationHandle(nullptr, &mem_handle);
    aclrtMemExportToShareableHandleV2(nullptr, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION,
        ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, &handle);
    aclrtMemImportFromShareableHandleV2(&handle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0, nullptr);
    return 0;
}")

# Check if we can link the program with the symbol
try_compile(HAS_ACLRT_MEM_FABRIC_HANDLE
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/check_acl_symbol.cpp
    LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}" "-DLINK_DIRECTORIES=${ASCEND_HOME_PATH}/lib64"
    OUTPUT_VARIABLE COMPILE_OUTPUT
)

# Clean up the test file
file(REMOVE ${CMAKE_BINARY_DIR}/check_acl_symbol.cpp)

option(ENABLE_CANN_BUILD "Enable CANN Build" OFF)

if(HAS_ACLRT_MEM_FABRIC_HANDLE)
    message(STATUS "aclrtMemExportToShareableHandleV2 symbol found, enabling user-buffer V2 handle support")
    add_definitions(-DHAS_ACLRT_MEM_FABRIC_HANDLE)
else()
    message(STATUS "aclrtMemExportToShareableHandleV2 symbol not found, disabling user-buffer V2 handle support")
endif()

# Preserve the original ordinary-heap selection: only a CANN build uses the
# ACL/Fabric memory-segment implementation for heaps without user buffers.
if(HAS_ACLRT_MEM_FABRIC_HANDLE AND ENABLE_CANN_BUILD)
    add_definitions(-DUSE_ACLRT_MEM_FABRIC_HANDLE)
    message(STATUS "CANN build enabled, using ACL/Fabric handles for ordinary heaps")
endif()

# Check whether the ACL headers and library provide aclrtMemMapSetLink.
check_cxx_symbol_exists(aclrtMemMapSetLink "acl/acl.h" HAS_ACLRT_MEM_MAP_SET_LINK)

if(USE_ACLRT_MEM_FABRIC_HANDLE AND HAS_ACLRT_MEM_MAP_SET_LINK)
    set(ACLSHMEM_ONE_MULTI_PATH ON)
    message(STATUS "Fabric Handle and aclrtMemMapSetLink symbols found, enabling ACLSHMEM_ONE_MULTI_PATH support")
else()
    set(ACLSHMEM_ONE_MULTI_PATH OFF)
    message(STATUS "Fabric Handle or aclrtMemMapSetLink symbols not found, disabling ACLSHMEM_ONE_MULTI_PATH support")
endif()

# Check if aclrtMemMapSelectedLink symbol exists in the ACL library
file(WRITE ${CMAKE_BINARY_DIR}/check_aclrt_mem_map_selected_link.cpp "
#include <acl/acl.h>
int main() {
    void *ptr = nullptr;
    aclrtMemMapSelectedLink(ptr, 0, ptr, 0);
    return 0;
}")

try_compile(HAS_ACLRT_MEM_MAP_SELECTED_LINK
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/check_aclrt_mem_map_selected_link.cpp
    LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}" "-DLINK_DIRECTORIES=${ASCEND_HOME_PATH}/lib64"
    OUTPUT_VARIABLE COMPILE_OUTPUT
)

file(REMOVE ${CMAKE_BINARY_DIR}/check_aclrt_mem_map_selected_link.cpp)

if(HAS_ACLRT_MEM_MAP_SELECTED_LINK)
    set(ACLSHMEM_HCCS_SIO_LINK ON)
    message(STATUS "aclrtMemMapSelectedLink symbol found, enabling ACLSHMEM_HCCS_SIO_LINK support")
else()
    set(ACLSHMEM_HCCS_SIO_LINK OFF)
    message(STATUS "aclrtMemMapSelectedLink symbol not found, disabling ACLSHMEM_HCCS_SIO_LINK support")
endif()

# Detect the installed CANN version so feature gates that depend on a specific
# CANN release can use a single source of truth.
#
# version.info lives under $ASCEND_HOME_PATH/opp/version.info on standard
# installs (the OPP subdir is always present in a full CANN installation);
# $ASCEND_HOME_PATH/version.info is checked as a fallback for slimmed-down
# installs. This matches the lookup order used by
# _get_cann_build_version() in scripts/build.sh so both code paths agree on
# the same source of truth.
#
# version.info carries a line like "Version=9.2.0" (with optional surrounding
# quotes and pre-release suffix; the regex below tolerates them). The numeric
# triple (major.minor.patch) is captured into CANN_VERSION; the optional suffix
# is captured into CANN_VERSION_SUFFIX for informational message display.
set(_CANN_VERSION_INFO_FILE "")
foreach(_candidate IN ITEMS
        "${ASCEND_HOME_PATH}/opp/version.info"
        "${ASCEND_HOME_PATH}/version.info")
    if(EXISTS "${_candidate}")
        set(_CANN_VERSION_INFO_FILE "${_candidate}")
        break()
    endif()
endforeach()

if(_CANN_VERSION_INFO_FILE)
    file(READ "${_CANN_VERSION_INFO_FILE}" _cann_version_info)
    string(REGEX MATCH "Version=\"?([0-9]+\\.[0-9]+\\.[0-9]+)(-[A-Za-z0-9.]+)?\"?"
           _cann_version_match "${_cann_version_info}")
    set(CANN_VERSION "${CMAKE_MATCH_1}")
    set(CANN_VERSION_SUFFIX "${CMAKE_MATCH_2}")
    unset(_cann_version_info)
    unset(_cann_version_match)
else()
    set(CANN_VERSION "")
    set(CANN_VERSION_SUFFIX "")
    message(WARNING
        "CANN version.info not found under $ASCEND_HOME_PATH (checked "
        "${ASCEND_HOME_PATH}/opp/version.info and ${ASCEND_HOME_PATH}/version.info). "
        "CANN version gating will treat the install as too old.")
endif()
unset(_CANN_VERSION_INFO_FILE)

# SIMT UDMA requires CANN 9.2.0 and later. The flag follows the project
# convention ACLSHMEM_<FEATURE>_SUPPORT (see ACLSHMEM_SIMT_SUPPORT,
# ACLSHMEM_RDMA_SUPPORT, ACLSHMEM_HCCS_SIO_LINK) and gates the simt_udma*
# test/example CMake targets and the corresponding test files in the
# host/device test binaries.
set(ACLSHMEM_SIMT_UDMA_SUPPORT OFF)
if(CANN_VERSION AND NOT CANN_VERSION VERSION_LESS "9.2.0")
    set(ACLSHMEM_SIMT_UDMA_SUPPORT ON)
    add_definitions(-DACLSHMEM_SIMT_UDMA_SUPPORT)
    message(STATUS
        "CANN ${CANN_VERSION}${CANN_VERSION_SUFFIX} >= 9.2.0, enabling SIMT UDMA support")
else()
    message(STATUS
        "CANN ${CANN_VERSION}${CANN_VERSION_SUFFIX} < 9.2.0, disabling SIMT UDMA support")
endif()

# 添加子目录
add_subdirectory(src)
add_subdirectory(tools)

if(USE_UNIT_TEST)
    add_subdirectory(tests/unittest)
endif()

if(USE_EXAMPLES OR PYEXPAND_EXAMPLE)
    add_subdirectory(examples)
endif()
