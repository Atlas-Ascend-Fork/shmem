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

add_subdirectory(device)
add_subdirectory(host)
add_subdirectory(include)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

add_executable(aclshmem_unittest $<TARGET_OBJECTS:aclshmem_unittest_host>)
# Host initialization dlopens this plugin; build the matching revision for UTs.
add_dependencies(aclshmem_unittest aclshmem_bootstrap_config_store)
target_link_directories(aclshmem_unittest PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
)
set_target_properties(aclshmem_unittest PROPERTIES
    LINK_FLAGS "-rdynamic"
)

if(IS_NEW_VERSION)
    set(UT_LIBRARIES
        ascendc_runtime
        mmpa
        ascend_dump
        c_sec
        error_manager
        unified_dlog
        profapi
    )
else()
    set(UT_LIBRARIES "")
endif()

target_link_libraries(aclshmem_unittest PRIVATE ${UT_LIBRARIES} aclshmem_unittest_device aclshmem_sync_timeout_test_device gtest gmock gcov aclshmem_unittest_include shmem_utils)

if(ACLSHMEM_DYNAMIC_META)
    # Include tests contribute mixed-code objects directly to this executable.
    # Resolve their module reference with the device linker as for any consumer.
    # Only this directory's final executable link needs the device linker.
    # Keep the existing g++ compiler for Host objects.
    find_program(ACLSHMEM_UT_DEVICE_LINKER bisheng HINTS "${ASCEND_HOME_PATH}/bin" REQUIRED)
    string(REPLACE "<CMAKE_CXX_COMPILER>" "${ACLSHMEM_UT_DEVICE_LINKER}"
        CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")
    set_target_properties(aclshmem_unittest PROPERTIES ACLSHMEM_META_HOST_PROVIDED TRUE)
    aclshmem_target_meta(aclshmem_unittest)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    target_compile_definitions(aclshmem_unittest_host PRIVATE
        ACLSHMEM_DYNAMIC_TEST_MODULE_DIR="$<TARGET_FILE_DIR:aclshmem_dynamic_reader1>"
        ACLSHMEM_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        ACLSHMEM_TEST_PYTHON="${Python3_EXECUTABLE}")
    if(SOC_TYPE STREQUAL "Ascend950")
        target_compile_definitions(aclshmem_unittest_host PRIVATE TEST_SIMT)
    endif()
    add_dependencies(aclshmem_unittest
        aclshmem_dynamic_reader1 aclshmem_dynamic_reader2 aclshmem_dynamic_reader3
        aclshmem_bootstrap_config_store)
    target_link_libraries(aclshmem_unittest PRIVATE dl)
endif()
