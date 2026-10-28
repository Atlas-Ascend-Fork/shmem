# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Compile host source files here with coverage feature ON
file(GLOB_RECURSE ACLSHMEM_HOST_FILES ${PROJECT_SOURCE_DIR}/src/host/*.cpp)
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "python_wrapper")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/bootstrap")
if(NOT SOC_TYPE STREQUAL "Ascend950")
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/device_udma")
endif()
if(NOT SOC_TYPE STREQUAL "Ascend950")
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/topo")
endif()
if(NOT (ACLSHMEM_RDMA_SUPPORT AND SOC_TYPE STREQUAL "Ascend950"))
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/device_rdma/device_rdma_transport_manager_v2")
endif()
if(USE_MSSANITIZER)
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/mstx/mstx_mem_register_empty.cpp")
    message(STATUS "USE_MSSANITIZER=ON")
else()
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/mstx/mstx_mem_register.cpp")
    message(STATUS "USE_MSSANITIZER=OFF")
endif()

file(GLOB_RECURSE TEST_HOST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp")
file(GLOB_RECURSE TOPO_TEST_HOST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/topo/*_test.cpp")

if(NOT SOC_TYPE STREQUAL "Ascend950")
    list(FILTER TEST_HOST_FILES EXCLUDE REGEX "init/user_buffer_heap_test[.]cpp$")
    list(FILTER TEST_HOST_FILES EXCLUDE REGEX "topo/")
    list(FILTER TEST_HOST_FILES EXCLUDE REGEX "transport/device_udma/")
else()
    list(FILTER TEST_HOST_FILES EXCLUDE REGEX "topo/")
endif()

# SIMT UDMA host tests require CANN >= 9.2.0.
# Exclude them from the host test object library when the CANN version does
# not qualify so the remaining host tests still build. Excluding the
# *_test.cpp also drops the TEST() registrations, so the cases are not
# collected by the test runner during batch execution.
# The QP-specific data plane is direct-mode only; relay builds reject qp_num != 1
# at OpenDevice, so these tests would fail to initialize under relay.
if(NOT ACLSHMEM_SIMT_UDMA_SUPPORT OR ACLSHMEM_RELAY_SUPPORT)
    list(FILTER TEST_HOST_FILES EXCLUDE REGEX "simt_udma/")
endif()

file(GLOB_RECURSE CONFIG_STORE_FILES ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/*.cpp)
file(GLOB_RECURSE UNDER_API_FILES ${PROJECT_SOURCE_DIR}/src/host/utils/under_api/*.cpp)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

include(CheckCXXSourceCompiles)
set(_ACLSHMEM_SAVED_CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_INCLUDES
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/hcomm
)
unset(ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME CACHE)
check_cxx_source_compiles("#include <hcomm/hcomm_res_defs.h>
int main()
{
    HcommChannelDesc desc{};
    desc.channelName = nullptr;
    return 0;
}"
    ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME
)
set(CMAKE_REQUIRED_INCLUDES "${_ACLSHMEM_SAVED_CMAKE_REQUIRED_INCLUDES}")
unset(_ACLSHMEM_SAVED_CMAKE_REQUIRED_INCLUDES)

if(SOC_TYPE STREQUAL "Ascend950")
    file(GLOB_RECURSE TOPO_UNITTEST_SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/host/transport/topo/*.cpp)
    list(FILTER TOPO_UNITTEST_SOURCE_FILES EXCLUDE REGEX "root_info_generate")
    list(FILTER TOPO_UNITTEST_SOURCE_FILES EXCLUDE REGEX "aclshmemi_hal")
    set(ACLSHMEM_TOPO_UNITTEST_TARGET aclshmem_topo_unittest)
    add_executable(
        ${ACLSHMEM_TOPO_UNITTEST_TARGET}
        ${TOPO_TEST_HOST_FILES}
        ${TOPO_UNITTEST_SOURCE_FILES}
        ${UNDER_API_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/topo/mock_hal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/topo/topo_test_main.cpp)
    target_compile_options(${ACLSHMEM_TOPO_UNITTEST_TARGET} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -g -fprofile-arcs -ftest-coverage)
    target_include_directories(${ACLSHMEM_TOPO_UNITTEST_TARGET} PRIVATE
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/src/
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/device
        ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/tests/unittest/
        ${PROJECT_SOURCE_DIR}/tests/unittest/include
        ${PROJECT_SOURCE_DIR}/tests/unittest/include/unittest/utils/
        ${PROJECT_SOURCE_DIR}/src/host/entity
        ${PROJECT_SOURCE_DIR}/src/host/mem/heap
        ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver
        ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver/userspace
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/common
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/security
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api/openssl
        ${PROJECT_SOURCE_DIR}/src/host/utils
        ${PROJECT_SOURCE_DIR}/src/host/utils/under_api
        ${PROJECT_SOURCE_DIR}/src/host/transport
        ${PROJECT_SOURCE_DIR}/src/host/transport/device_rdma
        ${PROJECT_SOURCE_DIR}/src/host/transport/device_sdma
        ${PROJECT_SOURCE_DIR}/src/host/transport/device_udma
        ${PROJECT_SOURCE_DIR}/src/host/transport/topo/rootinfo)
    target_link_directories(${ACLSHMEM_TOPO_UNITTEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib)
    target_link_libraries(${ACLSHMEM_TOPO_UNITTEST_TARGET} PRIVATE gtest gmock gcov shmem_utils)
endif()

add_library(aclshmem_unittest_host OBJECT ${ACLSHMEM_HOST_FILES} ${TEST_HOST_FILES} ${CONFIG_STORE_FILES} ${UNDER_API_FILES})
target_compile_definitions(aclshmem_unittest_host PRIVATE ACLSHMEM_UNIT_TEST=1)

if(SOC_TYPE STREQUAL "Ascend950")
    target_compile_definitions(aclshmem_unittest_host PRIVATE ACLSHMEM_UT_ASCEND950=1)
endif()
if(ACLSHMEM_RDMA_SUPPORT)
    target_compile_definitions(aclshmem_unittest_host PRIVATE ACLSHMEM_UT_RDMA_ENABLED=1)
endif()

if(ACLSHMEM_SIMT_SUPPORT)
    target_compile_definitions(aclshmem_unittest_host PRIVATE USE_SIMT)
endif()
if(ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME)
    target_compile_definitions(aclshmem_unittest_host PRIVATE ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME=1)
    message(STATUS "HCOMM channelName detected: UDMA multi-QP data-path tests enabled")
else()
    message(STATUS "HCOMM channelName not detected: UDMA multi-QP data-path tests will pass without execution")
endif()

target_compile_options(aclshmem_unittest_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_compile_options(aclshmem_unittest_host PRIVATE -g -fprofile-arcs -ftest-coverage)
target_include_directories(aclshmem_unittest_host PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
    ${PROJECT_SOURCE_DIR}/tests/unittest/include
    ${PROJECT_SOURCE_DIR}/tests/unittest/include/unittest/utils/
    ${PROJECT_SOURCE_DIR}/tests/unittest/
    ${PROJECT_SOURCE_DIR}/tests/unittest/team/team/
    ${PROJECT_SOURCE_DIR}/tests/unittest/team/team_allgather/
    ${PROJECT_SOURCE_DIR}/src/host/entity
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/common
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/security
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api/openssl
    ${PROJECT_SOURCE_DIR}/src/host/utils
    ${PROJECT_SOURCE_DIR}/src/host/utils/under_api
    ${PROJECT_SOURCE_DIR}/src/host/transport
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_rdma
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_sdma
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_udma
    ${PROJECT_SOURCE_DIR}/src/host/transport/topo/rootinfo
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver/userspace
)
if(SOC_TYPE STREQUAL "Ascend950")
    target_include_directories(aclshmem_unittest_host PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/json/single_include)
endif()

if(ACLSHMEM_SIMT_SUPPORT)
    target_include_directories(aclshmem_unittest_host PRIVATE
        ${PROJECT_SOURCE_DIR}/src/device_simt/
    )
endif()
