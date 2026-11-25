# SHMEM Public Headers and Libraries

> Scope: public interface header files and the shared libraries (`.so`) in the installation artifacts.

## 1. Library Overview

| Library | Type | Purpose | Dependency Version Information |
|--------|------|------|--------------|
| `libshmem.so` | Core runtime | Host-side main library: initialization, heap, Team, Host RMA/synchronization/collective communication, and so on | Depends on the CANN libraries under `ASCEND_HOME_PATH` (`runtime`/`ascendcl`/`tiling_api`/`platform`/`c_sec`, and so on); links `libshmem_utils.so`; additionally links `hcomm` when `SOC_TYPE=Ascend950` |
| `libshmem_utils.so` | Utility/support library | Logging, internal tools, and common support symbols, for the main library and bootstrap plugins to link against | No independent external version number; depends on the nlohmann/json header files at compilation time when UDMA is enabled |
| `aclshmem_bootstrap_config_store.so` | Bootstrap plugin (without the `lib` prefix) | Control plane for Default / UniqueID initialization (Config Store). Both `ACLSHMEMX_INIT_WITH_DEFAULT` and `ACLSHMEMX_INIT_WITH_UNIQUEID` load this library | Dynamically loaded by init at runtime; links `libshmem_utils.so`; no independent version number |
| `aclshmem_bootstrap_uid.so` | Bootstrap plugin | Library for UniqueID bootstrap | Links `libshmem_utils.so`; no independent version number |
| `aclshmem_bootstrap_mpi.so` | Bootstrap plugin (optional) | Bootstrap in MPI mode | Compiled only when CMake `find_package(MPI)` succeeds; links `MPI::MPI_CXX` and `libshmem_utils.so` |
| `_pyshmem*.so` | Python extension (optional) | pybind11 bindings for `import shmem` | Build: `find_package(pybind11)`; links `libshmem.so` and so on; run: `torch-npu` (`setup.py`) |

Notes:

- Device-side interfaces are mostly inline in header files; `libshmem.so` provides Host-side symbols.
- Applications typically include `shmem.h`, link with `-lshmem`, and ensure that the dynamic linker can resolve `libshmem_utils.so` and the bootstrap plugins actually used.

## 2. List of Public Interface Header Files

Unified entry point: `shmem.h` (includes the following header files conditionally based on Host / Device / SIMT).

### 2.1 Main Entry Point and Common Definitions

| Public Interface File | Corresponding .so Library | File Purpose | Dependency Version Information |
|------------------|----------------|----------|--------------|
| `shmem.h` | The Host part corresponds to `libshmem.so` (and utils/bootstrap); the Device/SIMT part has none | General public header file: aggregates Host / Device / SIMT public APIs by compilation macros | Device/SIMT branches depend on compilation macros (for example, `__CCE_AICORE__` and `USE_SIMT`); no independent external version number |
| `host_device/shmem_common_types.h` | None | Types, enums, and macros such as `ACLSHMEM_DEVICE` shared by Host/Device | Depends on the CANN header file `acl/acl_rt.h` |
| `host_device/shmem_common_macros.h` | None | Macros shared by Host/Device | None |
| `host/shmem_host_def.h` | `libshmem.so` | Host-side constants, version macros, `ACLSHMEM_HOST_API`, and common structures/enums | No external version constraints (only defines API macros of this library) |
| `device/shmem_def.h` | None | Device-side common definitions and structures such as non-contiguous copy | Depends on CANN; has conditional branches when `USE_SIMT` is used |

### 2.2 Host-Side Public Interfaces

| Public Interface File | Corresponding .so Library | File Purpose | Dependency Version Information |
|------------------|----------------|----------|--------------|
| `host/init/shmem_host_init.h` | `libshmem.so` + bootstrap plugins loaded at runtime | Initialization/finalization, UniqueId, init status query, `aclshmemx_init_attr`, and so on | Default/UniqueID load `aclshmem_bootstrap_config_store.so`; MPI mode loads `aclshmem_bootstrap_mpi.so` (requires system MPI) |
| `host/mem/shmem_host_heap.h` | `libshmem.so` | Symmetric heap allocation: `malloc`/`calloc`/`align`/`free` and extended memory type interfaces | None |
| `host/team/shmem_host_team.h` | `libshmem.so` | Team creation/splitting/destruction, PE translation, `my_pe`/`n_pes`, and so on | None |
| `host/data_plane/shmem_host_rma.h` | `libshmem.so` | Host-side RMA (put/get, and so on) and `aclshmem_ptr` | None |
| `host/data_plane/shmem_host_so.h` | `libshmem.so` | Host-side put+signal synchronous/asynchronous interfaces | None |
| `host/data_plane/shmem_host_p2p_sync.h` | `libshmem.so` | Host-side P2P synchronization, `handle_wait`, signal wait, and ffts configuration retrieval | None |
| `host/data_plane/shmem_host_cc.h` | `libshmem.so` | Host-side collective communication: barrier/sync and on_stream variants | None |
| `host/utils/shmem_log.h` | `libshmem.so` / `libshmem_utils.so` | External logger, log levels, and profiling printing/retrieval | None |

### 2.3 Device-Side Public Interfaces (GM↔GM / UB↔GM)

| Public Interface File | Corresponding .so Library | File Purpose | Dependency Version Information |
|------------------|----------------|----------|--------------|
| `device/gm2gm/shmem_device_rma.h` | None | Device GM↔GM standard RMA (multi-dtype put/get, and so on) | Depends on CANN; the underlying engine can be MTE/RDMA/SDMA/UDMA (determined by the topology and compilation options); the remote operand is used for symmetric address conversion; when RDMA is not enabled, the source of a local Put or the destination of a Get can use any valid local GM address; after RDMA is enabled, both ends must point to symmetric memory; callers do not need to determine the actual engine for a single operation |
| `device/gm2gm/shmem_device_so.h` | None | Device-side put+signal synchronous interface | Depends on CANN |
| `device/gm2gm/shmem_device_amo.h` | None | Device atomic operations (atomic_add, and so on) | Depends on CANN; the header file notes that the types supported by each engine vary with the SoC (for example, UDMA targets Ascend950) |
| `device/gm2gm/shmem_device_mo.h` | None | Device memory ordering: `quiet` / `fence` | Depends on CANN |
| `device/gm2gm/shmem_device_p2p_sync.h` | None | Device P2P synchronization: signal_op, wait_until, and so on | Depends on CANN |
| `device/gm2gm/shmem_device_cc.h` | None | Device collective communication: barrier (including vector-only and other variants) and ffts address setting | Depends on CANN; the header file restricts its use mainly to MIX Kernel |
| `device/gm2gm/engine/shmem_device_mte.h` | None | MTE engine: symmetric address translation and contiguous/non-contiguous asynchronous put/get | Depends on CANN |
| `device/gm2gm/engine/shmem_device_rdma.h` | None | RDMA engine: asynchronous RMA where both the local and remote operands must point to symmetric memory and the full transfer range must not exceed the respective memory allocations | Depends on CANN; the Host side must enable `ACLSHMEM_RDMA_SUPPORT` (Ascend950 additionally requires specifying an RDMA backend) |
| `device/gm2gm/engine/shmem_device_sdma.h` | None | SDMA engine: parameter settings and asynchronous put/get | Depends on CANN |
| `device/gm2gm/engine/shmem_device_udma.h` | None | UDMA engine: low-level asynchronous RMA | Targets only Ascend950 and requires CANN 9.1.0 or later; loads and checks the required HCOMM resource interfaces during initialization |
| `device/ub2gm/shmem_device_rma.h` | None | UB↔GM standard RMA (UB-side get/put, and so on) | Depends on CANN |
| `device/ub2gm/engine/shmem_device_mte.h` | None | MTE engine asynchronous copy for UB↔GM | Depends on CANN |
| `device/team/shmem_device_team.h` | None | Device-side Team/PE query and PE translation | Depends on CANN |

### 2.4 Device SIMT Public Interfaces (SIMT Support Must Be Enabled)

| Public Interface File | Corresponding .so Library | File Purpose | Dependency Version Information |
|------------------|----------------|----------|--------------|
| `device_simt/shmem_simt_common_types.h` | None | SIMT common state/type definitions | Requires the compilation definition `USE_SIMT` (`ACLSHMEM_SIMT_SUPPORT`) |
| `device_simt/gm2gm/shmem_device_simt_rma.h` | None | GM↔GM RMA in the SIMT scope | Requires `USE_SIMT`; depends on CANN |
| `device_simt/gm2gm/shmem_device_simt_amo.h` | None | GM↔GM atomic operations (compare swap) in the SIMT scope | Requires `USE_SIMT`; depends on CANN |
| `device_simt/gm2gm/engine/shmem_device_simt_mte.h` | None | SIMT MTE: asynchronous copy in the thread/block/warp scope | Requires `USE_SIMT`; depends on CANN |
| `device_simt/gm2gm/engine/shmem_device_simt_udma.h` | None | SIMT UDMA: QP-specific asynchronous put/get and quiet (warp / block scope) | Requires `USE_SIMT`; targets only Ascend950 direct mode and requires CANN 9.2.0 or later |
| `device_simt/team/shmem_device_simt_team.h` | None | SIMT-side Team-related interfaces | Requires `USE_SIMT`; depends on CANN |
