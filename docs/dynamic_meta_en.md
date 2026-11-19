# Dynamic metadata design and integration

## Features and build modes

SHMEM uses a fixed metadata address by default. When device symbol support is enabled,
the Runtime allocates a 32 MiB metadata region. Each device binary has a `g_meta_root`
variable that stores the address of this region. SHMEM instances in the same process
share the region and use the existing instance selection APIs to access their own state.

Build maintainers enable the feature, update linking, and rebuild operators. Kernel
authors continue to use the existing SHMEM APIs without adding parameters or registration calls.

| Item | Fixed-address mode (FIXED, default) | Dynamic-address mode (ROOT) |
| --- | --- | --- |
| Build option | No additional option | `-enable_device_symbol` |
| Metadata lookup | Fixed address | Read the device variable `g_meta_root` |
| Mixed-code operator linking | Existing configuration | Add the matching `libshmem_device.a` and device link options |
| Additional Runtime dependencies | None | `aclrtGetSymbolAddress` |

**Dynamic mode requires the Runtime APIs in CANN 9.1.0 (final release) or later.**
The 9.1.0-beta.1 through beta.3 releases do not meet this requirement. The public Runtime
tag `v9.1.0.pre` already contains the API; `v9.1.0` is the final-release baseline. Its
[public declaration](https://gitcode.com/cann/runtime/blob/v9.1.0/include/external/acl/acl_rt.h#L2403),
[export wrapper](https://gitcode.com/cann/runtime/blob/v9.1.0/src/acl/aclrt_impl/acl_rt_wrapper.h#L333), and
[implementation](https://gitcode.com/cann/runtime/blob/v9.1.0/src/acl/aclrt_impl/memory.cpp#L2113)
are available in that tag. `v9.1.0-beta.2` provides `aclrtBinaryGetGlobal` but not
`aclrtGetSymbolAddress`, so the former does not establish compatibility. This implementation
uses `aclrtGetSymbolAddress + aclrtMemcpy`; it does not require `aclrtMemcpyToSymbol` or `aclrtBinaryGetGlobal`.

This is the minimum Runtime API version, subject to the target SoC's CANN compatibility
and compiler requirements. Hardware validation used A5 / Ascend950PR / CANN 9.2.0-beta.2.
CANN 9.1.0 API support is confirmed from public source, but has not been validated on hardware
for this feature. Fixed-address mode adds no such version requirement.

```bash
bash scripts/build.sh -soc_type Ascend950 -enable_device_symbol
```

For a direct CMake build, set `-DACLSHMEM_ENABLE_DEVICE_SYMBOL=ON`.
With `build.sh`, omitting the option selects fixed-address mode, even when reusing a build directory.

An explicitly enabled build warns about rebuilding operators and deployment requirements;
it does not reject the build based on the build SDK's ACL declarations. At deployment,
SHMEM resolves `aclrtGetSymbolAddress` from the loaded ACL library. A missing API causes initialization to fail; there is
no automatic fallback to fixed addresses. The compiler must still support the target SoC,
GM globals, and device linking. The option cannot add missing compiler capabilities.

The source header `include/host_device/shmem_common_macros.h` defaults to `ACLSHMEM_DYNAMIC_META=0`.
CMake selects the mode for in-tree targets with a compiler definition and installs a copy
of the header whose default matches the library. External projects inherit this setting
from the installed headers. Projects using source headers or explicit macro definitions
must match the linked library's mode.
**Host, SIMD, and SIMT code must use the same mode. Fixed-address operators cannot use a dynamic-address library.**
When switching to dynamic mode, rebuild SHMEM and use the new installed SDK to rebuild every
operator package that uses SHMEM Device APIs, including C++ libraries, Torch extensions, and
their separate kernel libraries. Replacing only `libshmem.so` while retaining old operator
binaries is not supported. Remove old-mode build artifacts and relevant JIT caches before
rebuilding. The same requirement applies when switching back to fixed-address mode.

## Integrating mixed-code operators

For projects that launch kernels with bisheng `<<<>>>`, the compiler manages the device
image. In-tree examples are configured automatically. External projects can add the following
settings to their build files without compiling the registration sources themselves.

This example assumes that `operator_kernel` already has `-xasc`, the correct `NPU_ARCH`,
and the existing SHMEM include and library paths. `SHMEM_ROOT` points to the installed SDK,
and `shmem` is the matching library target. Include paths must cover the package's `include`,
`src`, and the required Device/SIMT directories.

```cmake
file(STRINGS "${SHMEM_ROOT}/include/host_device/shmem_common_macros.h" shmem_root_enabled
    REGEX "^#define ACLSHMEM_DYNAMIC_META 1$")
if(shmem_root_enabled)
    target_compile_options(operator_kernel PRIVATE -dc)
    target_link_options(operator_kernel PRIVATE
        --cce-fatobj-link --npu-arch=${NPU_ARCH})
    target_link_libraries(operator_kernel PRIVATE
        "${SHMEM_ROOT}/lib/libshmem_device.a"
        shmem)
endif()
```

Intermediate OBJECT/STATIC device targets also need `-dc`. Link the archive once per final
device binary. The archive must match the SDK, SoC, and device compilation ABI of the package.
For a wheel, use `backends/<backend>/libshmem_device.a` for the selected backend.
The conditional block adds no dependencies in fixed-address mode.

Ordinary linking does not require `--whole-archive` or `-u`. Only in the Host phase of a dynamic
mixed-code build, `include/host_device/shmem_common_macros.h` includes
`src/host_device/shmem_device_symbol.h`. This implementation header declares the hidden link
anchor `aclshmemx_device_library_anchor` and retains a reference to pull in the archive member.
The public header does not declare the anchor directly. The implementation header is installed
under `src/host_device` and excluded from generated API documentation.
That object references the Host registration object, so compiler registration of the device
variable precedes SHMEM registration. Users do not call this hidden build entry.
Registration, unregistration, and address publication remain internal implementations under `src/`.
With the compilation and linking settings above, registration is automatic. Integrators do
not write module initialization functions or manage the internal registration order.

**Load all related operator libraries before SHMEM initialization. This version does not
support loading new SHMEM operator libraries after initialization.** Early-loaded modules
only register their tokens. Publication errors are returned by `aclshmemx_init_attr`; callers
must check the result and must not launch SHMEM kernels after a failure. The existing late-load
publication path remains in the code but is not a supported integration contract: a symbol
query or copy failure prints diagnostics and calls `abort()` in the constructor, rather than
returning an ordinary `dlopen` error to the caller.

This implementation supports compiler-managed mixed-code modules; it does not provide
integration for custom binary loaders.

### Unloading and reloading modules

Dynamic mode does not support repeatedly unloading and reloading the same operator `.so`
within one process. Load the module once, keep it resident, and launch its kernels as needed.
SHMEM emits a `META_MODULE_RELOAD_SUSPECT` warning when a previously unregistered Host token
is registered again. This is a diagnostic hint, not reliable reload detection or recovery.

In the tested CANN 9.2.0-beta.2 environment, actually unloading an operator with `dlclose`
and loading it again with `dlopen` in the same process can encounter an old Host symbol
mapping. A mismatch between the slot returned to the Host and the slot used by the new kernel
would invalidate address publication. The logs do not independently prove such an address mismatch.
Repeated kernel launches, SHMEM finalize/init while modules remain resident, and restarting
the process are different cases.

For projects that unload and reload modules, the following option is recommended for the
affected operator modules. It is not mandatory for ordinary integration:

```cmake
target_link_options(operator_kernel PRIVATE -Wl,-z,nodelete)
```

This keeps the module resident until process exit; `dlclose` does not immediately release
its code and globals. Metadata still follows the SHMEM lifecycle: finalizing the last instance
clears the module entries and frees the region; reinitialization publishes the new address.

Runtime mainline contains a [fix that clears variable symbols when a Program is released](https://gitcode.com/cann/runtime/commit/b372e14919c5ba4c3ae412b2d213da934b04568c).
However, the reproduction logs did not show a corresponding binary unload during Host
`dlclose`, so this fix has not been confirmed to resolve mixed-code module reloads.
`nodelete` is a tested workaround in that environment, not support for actual unload/reload.

## Address publication and lifecycle

Each device binary has an 8 B `g_meta_root` pointing to the shared local metadata region.
A mixed-code module supplies its compiler-registered Host variable token; SHMEM queries
the device variable address with `aclrtGetSymbolAddress`, then writes the metadata address
with `aclrtMemcpy`.
The SIMD accessor reads `g_meta_root` directly; SIMT uses `asc_ldcg`.

Metadata belongs to the initialization device. Instances share the region, and selecting an
instance does not overwrite the root. Publication and cleanup check the current device ID;
they do not retain, compare, or switch the initialization Context.

1. Load all operator libraries, select the target device, and initialize SHMEM. Launch SHMEM kernels only after initialization succeeds.
2. Before finalize, stop submitting related kernels and complete communication quiet operations and all relevant stream/graph synchronization.
3. If synchronization, root clearing, or memory release fails, finalize reports failure. Keep the device environment valid, resolve the failure, and retry init/finalize to clean up retained resources. This also applies after failed initialization. Do not resume SHMEM kernels until initialization succeeds.

Complete module loading, registration, and SHMEM initialization before graph capture. Modules, metadata,
instance state, and application heaps must remain valid during capture/replay.
Rebuild graphs that span a finalize/reinitialize cycle.

## Diagnostic logs

Set these variables before starting the process to inspect registration and address publication:

```bash
export SHMEM_LOG_LEVEL=DEBUG
export SHMEM_LOG_TO_STDOUT=1
```

Log levels are uppercase; use `WARN` for warnings and errors. Without stdout logging,
logs are written to `~/shmem/log` by default; override this with `SHMEM_LOG_PATH`.
You can also call `aclshmemx_set_log_level(0)` after each successful initialization to enable DEBUG.

| Log marker | Level | Contents |
| --- | --- | --- |
| `META_MODULE_RELOAD_SUSPECT` | WARN | A previously unregistered Host token was registered again; check reloads and old mappings |
| `META_BIND_SYMBOL` | DEBUG / ERROR | Host token, queried device slot, root; API and return code on failure |
| `META_PUBLISH` | DEBUG / ERROR | Device ID, root, module counts, and failure code during publication or clearing |
| `META_DEVICE_QUERY` / `META_DEVICE_MISMATCH` | ERROR | Failed device query or a mismatch with the metadata device |

Reload diagnostics retain the last 64 unregistered tokens. Ordinary finalize/init does not
trigger the warning. Host address reuse by another module can produce a false positive;
changed tokens or evicted history can miss a reload. A warning alone does not prove a slot mismatch.
Device slot addresses in DEBUG logs come from Host queries. Compare them with Runtime plog
messages such as `already registered, skip duplicate registration` and Program loading records.
Error `507035` or a GM access error alone does not prove an old mapping was used.

## Shared heaps and fixed metadata

This feature changes only metadata allocation and lookup. Shared-heap reservations,
contiguous layouts, mappings, and cleanup retain the existing `dev` implementation,
including the ACL/Fabric, VMM, and legacy driver allocation policies.

Fixed metadata retains its existing address and reservation flags; fixed-address conflicts
remain possible. Dynamic metadata lets `aclrtMalloc` select an available address instead
of requiring a predefined address to be free.

## Validation and scope

The `DynamicDeviceAddress` gtest suite covers module registration before initialization,
SIMD/MIX/SIMT, symbol queries and kernel reads across Contexts on one device, cleared entries after finalize, multiple
instances, communication, Host occupation of the old metadata address, reinitialization,
Host fault recovery, and packaging checks.

```bash
bash scripts/build.sh -soc_type Ascend950 -uttests -enable_device_symbol
bash scripts/run.sh -ranks 2 -gnpus 2 -fnpu 2 -test_filter 'DynamicDeviceAddress.*'
```

The command selects devices 2 and 3; replace them with two consecutive idle devices.
Device tests require two PEs on one host. Dynamic mode expects four passing tests;
fixed-address mode expects four skipped tests. Host fixtures require Python 3, a C++ compiler,
and setuptools. Build examples with `-examples -enable_device_symbol`, then follow each README.

This port targets `dev` and adapts both SIMD and SIMT instance selection to dynamic metadata.
Both modes passed library and full UT compilation with CANN 9.2.0-beta.2, installed
archive linking through umbrella, direct Device, and SIMT headers, and pure Host compilation.
Host fixtures for registration failure recovery, metadata cleanup retries, and wheel packaging
also passed. Device execution remains pending because `npu-smi` cannot query the A5 devices.
The earlier master implementation was tested on A5 / Ascend950PR / CANN 9.2.0-beta.2
for module publication, two-PE MTE communication, reinitialization, and Host occupation
of the old metadata address. See [PR #953](https://gitcode.com/cann/shmem/pull/953) for those
historical results; they do not replace validation of this dev port. A2/A3 hardware,
real older SDKs, cross-compiler archives, and full multi-SoC wheels remain unvalidated.
