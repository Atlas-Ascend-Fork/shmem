# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Single binary covering 2 execution scopes (warp/block) x 4 operations
# (put/get/bi_put/bi_get). The combination is selected at runtime via --scope
# and -t|--test-type; each combination is instantiated at compile time as a
# separate VF template specialization and launched through a dispatch ladder.
aclshmem_add_fusion_example(simt_udma_perftest main.cpp)
target_include_directories(simt_udma_perftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
