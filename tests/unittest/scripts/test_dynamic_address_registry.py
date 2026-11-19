# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the production registry against a fault-injecting ACL fixture.

No NPU required. CXX may contain a compiler plus arguments (e.g. 'zig c++').
This checks lifecycle/error handling, not CANN device symbol support.
"""
import os
import pathlib
import shlex
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]


def main():
    with tempfile.TemporaryDirectory(prefix="shmem-meta-registry-") as directory:
        temp = pathlib.Path(directory)
        (temp / "acl").mkdir()
        (temp / "host_device").mkdir()
        (temp / "host_device/shmem_common_macros.h").write_text("#define ACLSHMEM_DYNAMIC_META 1\n")
        (temp / "acl/acl_rt.h").write_text(r'''
#pragma once
#include <cstddef>
#include <cstdint>
using aclrtContext = void*;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_ERROR_INVALID_PARAM = 100000;
constexpr int ACL_ERROR_RT_CONTEXT_NULL = 107002;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
int aclrtGetDevice(int32_t*);
int aclrtMemcpy(void*, size_t, const void*, size_t, int);
''')
        (temp / "dl_acl_api.h").write_text(r'''
#pragma once
#include "acl/acl_rt.h"
namespace shm {
struct DlAclApi {
    static int AclrtGetSymbolAddress(const void*, void**);
};
}
''')
        (temp / "shmemi_logger.h").write_text(r'''
#pragma once
#include <sstream>
#include <string>
void CaptureMetaLog(const char*, const std::string&);
#define META_TEST_LOG(level, x) do { std::ostringstream out; out << x; CaptureMetaLog(level, out.str()); } while (0)
#define SHM_LOG_ERROR(x) META_TEST_LOG("ERROR", x)
#define SHM_LOG_WARN(x) META_TEST_LOG("WARN", x)
#define SHM_LOG_DEBUG(x) META_TEST_LOG("DEBUG", x)
''')
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        output = temp / ("registry.exe" if os.name == "nt" else "registry")
        subprocess.run(compiler + ["-std=c++17", "-pthread", "-I" + str(temp),
            "-I" + str(ROOT / "src"),
            str(ROOT / "src/host/init/shmem_meta_registry.cpp"),
            str(ROOT / "tests/unittest/host/init/dynamic_address_registry_fixture.cpp"), "-o", str(output)], check=True)
        subprocess.run([str(output)], check=True)
        # FIXED must compile without any ACL declarations or new symbol refs.
        (temp / "host_device/shmem_common_macros.h").write_text("#define ACLSHMEM_DYNAMIC_META 0\n")
        (temp / "acl/acl_rt.h").write_text("#error FIXED must not include ACL from the registry\n")
        subprocess.run(compiler + ["-std=c++17", "-I" + str(temp), "-I" + str(ROOT / "src"),
            "-c", str(ROOT / "src/host/init/shmem_meta_registry.cpp"),
            "-o", str(temp / "fixed.o")], check=True)
        print("FIXED_REGISTRY_COMPILE_PASS", flush=True)


if __name__ == "__main__":
    main()
