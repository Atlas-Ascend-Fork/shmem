# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile actual arena init/retire and backend-finalize bodies with ACL fault injection."""
import os
import pathlib
import shlex
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]


def function(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    entity = (ROOT / "src/host/entity/mem_entity_entry.cpp").read_text(encoding="utf-8")
    backend = (ROOT / "src/host/init/backends/shmem_init_backend.cpp").read_text(encoding="utf-8")
    init = (ROOT / "src/host/init/shmem_init.cpp").read_text(encoding="utf-8")
    finalize = function(init, "static int32_t aclshmemi_finalize_impl(")
    init_attr = function(init, "static int32_t aclshmemi_init_attr_impl(")
    # Reachability guards: retained ownership is independent of backend lifetime.
    assert finalize.index("hybm_retry_pending_meta()") < finalize.index("if (init_manager == nullptr)")
    assert init_attr.index("hybm_retry_pending_meta()") < init_attr.index("aclshmemi_instance_ctx_create")
    arena_namespace = entity[entity.index("namespace {"):entity.index("} // namespace") + len("} // namespace")]
    bodies = "\n".join(function(entity, signature) for signature in (
        "int32_t hybm_retry_pending_meta(", "HYBM_API int32_t hybm_init(",
        "int32_t hybm_uninit_with_status(", "HYBM_API void hybm_uninit("))
    backend_body = function(backend, "int aclshmemi_init_backend::finalize_device_state()")
    hal = (ROOT / "src/host/utils/under_api/dl_hal_api.h").read_text(encoding="utf-8")
    get_fd = function(hal, "static inline int32_t GetFd(")
    fixture = r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <map>
#include <iostream>
#include <type_traits>
#define ACLSHMEM_DYNAMIC_META 1
#define HYBM_API
#define SHM_LOG_ERROR(...) ((void)0)
#define SHM_LOG_INFO(...) ((void)0)
#define SHM_LOG_WARN(...) ((void)0)
#define SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(call, ...) do { int rc = (call); if (rc) return rc; } while (0)
constexpr int ACLSHMEM_SUCCESS=0, ACL_SUCCESS=0, ACL_ERROR_INVALID_PARAM=1;
constexpr int ACLSHMEM_INNER_ERROR=2, ACLSHMEM_MALLOC_FAILED=3, ACL_MEM_MALLOC_HUGE_FIRST=0;
constexpr int HYBM_GVA_V4=4;
constexpr size_t HYBM_DEVICE_INFO_SIZE=32ULL<<20;
using aclrtContext=void*;
using drv_mem_handle_t=int;
enum AscendSocType {ASCEND_UNKNOWN, ASCEND_950, ASCEND_910};
int contextStorage, otherContextStorage;
aclrtContext current=&contextStorage;
int currentDevice=0;
uint64_t arenaStorage, published=0;
bool live=false;
int syncFailures=0, clearFailures=0, freeFailures=0, memsetFailures=0, publishFailures=0;
constexpr int injected=27;
int aclrtGetDevice(int32_t* out) { *out=currentDevice; return current ? 0 : 1; }
int aclrtSynchronizeDevice() { if(syncFailures){--syncFailures; return injected;} return 0; }
int aclrtMalloc(void** ptr, size_t, int) { assert(!live); live=true; *ptr=&arenaStorage; return 0; }
int aclrtFree(void* ptr) {
    assert(live && ptr==&arenaStorage);
    if(freeFailures){--freeFailures; return injected;}
    assert(published==0); live=false; return 0;
}
int aclrtMemset(void*, size_t, int, size_t) { if(memsetFailures){--memsetFailures;return injected;} return 0; }
int aclshmemi_meta_publish(uint64_t root) {
    if(root && publishFailures){--publishFailures;published=root;return injected;}
    if(!root && clearFailures){--clearFailures;return injected;}
    published=root; return 0;
}
int HalGvaPrecheck(){return 0;}
int hybm_load_library(){return 0;}
int gvaVersion=4, legacyFd=7, legacyCalls=0;
AscendSocType testSoc=ASCEND_950;
int HybmGetGvaVersion(){return gvaVersion;}
namespace shm {
struct DlApi {static int LoadLibrary(){return 0;} static void CleanupLibrary() {assert(!live);} static AscendSocType GetAscendSocType(){return testSoc;}};
struct DlAclApi {static int RtGetLogicDevIdByUserDevId(int id,int* out){*out=id;return 0;} static int AclrtSetDevice(int id){currentDevice=id;return 0;}};
struct DlHalApi {inline static bool gLoaded=true; inline static int* pHalFd=&legacyFd; GET_FD_BODY};
}
namespace drv {void DevmmInitialize(int,int fd){assert(fd>=0);++legacyCalls;}}
struct entity_member {};
struct Instance {int id=0;} instance;
Instance* g_instance_ctx=&instance;
class aclshmemi_init_backend {
public:
    int hybm_init_count=1;
    entity_member member;
    std::mutex entity_map_mutex_;
    std::map<int,entity_member*> entity_map_{{0,&member}};
    int finalize_device_state();
};
'''
    tests = r'''
int main() {
    // Non-950 GVA V3: missing, failed-load and cleared HAL state must fail
    // before allocation or legacy mapper setup. Use the production accessor.
    testSoc=ASCEND_910; gvaVersion=3;
    shm::DlHalApi::pHalFd=nullptr;
    assert(hybm_init(0,0)==ACLSHMEM_INNER_ERROR && !live && legacyCalls==0);
    shm::DlHalApi::gLoaded=false;
    shm::DlHalApi::pHalFd=&legacyFd;
    assert(hybm_init(0,0)==ACLSHMEM_INNER_ERROR && !live && legacyCalls==0);
    shm::DlHalApi::pHalFd=nullptr;
    assert(hybm_init(0,0)==ACLSHMEM_INNER_ERROR && !live && legacyCalls==0);
    shm::DlHalApi::gLoaded=true; shm::DlHalApi::pHalFd=&legacyFd;
    legacyFd=-1;
    assert(hybm_init(0,0)==ACLSHMEM_INNER_ERROR && !live && legacyCalls==0);
    legacyFd=0; // fd 0 is valid.
    assert(hybm_init(0,0)==0 && live && legacyCalls==1);
    assert(hybm_uninit_with_status()==0 && !live);
    testSoc=ASCEND_950; gvaVersion=4;
    // Active metadata survives retry and another instance's finalize.
    assert(hybm_init(0,0)==0 && live);
    assert(hybm_retry_pending_meta()==0 && live);
    assert(hybm_init(0,0)!=0 && live && !metaCleanupPending);
    aclshmemi_init_backend shared;
    shared.hybm_init_count=2;
    assert(shared.finalize_device_state()==0 && live);
    assert(shared.finalize_device_state()==0 && !live);
    for(int fault=0;fault<3;++fault) {
        assert(hybm_init(0,0)==0 && live);
        if(fault==0)syncFailures=1;
        if(fault==1)clearFailures=1;
        if(fault==2)freeFailures=1;
        { // The backend may be destroyed after reporting an error.
            aclshmemi_init_backend backend;
            assert(backend.finalize_device_state()==injected);
            assert(backend.hybm_init_count==0);
        }
        assert(live && initialized && metaCleanupPending);
        currentDevice=1;
        assert(hybm_retry_pending_meta()!=0 && live);
        currentDevice=0;
        current=&otherContextStorage; // Same device, different Context must be accepted.
        assert(hybm_retry_pending_meta()==0 && !live && !initialized);
        assert(hybm_init(0,0)==0 && hybm_uninit_with_status()==0);
    }
    // Failed init has no counted backend; even failed cleanup remains reachable.
    publishFailures=1; clearFailures=1;
    assert(hybm_init(0,0)==injected && live && metaCleanupPending);
    assert(hybm_retry_pending_meta()==0 && !live);
    memsetFailures=1; freeFailures=1;
    assert(hybm_init(0,0)==injected && live && metaCleanupPending);
    // The lower-level init path also retries before allocating a second arena.
    assert(hybm_init(0,0)==0 && live && !metaCleanupPending);
    assert(hybm_uninit_with_status()==0 && !live);
    assert(hybm_retry_pending_meta()==0 && hybm_uninit_with_status()==0);
    // Existing void entry remains callable and retains failed cleanup for retry.
    static_assert(std::is_same<decltype(&hybm_uninit), void (*)()>::value, "legacy signature");
    assert(hybm_init(0,0)==0);
    freeFailures=1;
    hybm_uninit();
    assert(live && metaCleanupPending);
    assert(hybm_retry_pending_meta()==0 && !live);
    std::cout << "META_OWNER_RETRY_PASS\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="shmem-meta-owner-") as directory:
        base = pathlib.Path(directory)
        source = base / "owner.cpp"
        source.write_text("\n".join((fixture.replace("GET_FD_BODY", get_fd), arena_namespace, bodies, backend_body, tests)), encoding="utf-8")
        output = base / ("owner.exe" if os.name == "nt" else "owner")
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(compiler + ["-std=c++17", str(source), "-o", str(output)], check=True)
        subprocess.run([str(output)], check=True)


if __name__ == "__main__":
    main()
