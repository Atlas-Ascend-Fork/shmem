# Compilation and Build
## SHMEM Build
### Downloading the SHMEM Source Code

```shell
git clone https://gitcode.com/cann/shmem.git
```

You can choose your desired branch.

### Build

Enter the root directory of SHMEM and build the project.

```shell
cd shmem
bash scripts/build.sh
```

For more information about the commands, see the *README* and the `scripts/build.sh` file in the SHMEM's `master` directory.

### SHMEM Build Description
The basic SHMEM build command is `bash build.sh`. The default build mode generates version information and creates an installation package. By default, the RDMA capability, test cases, tests, and Python APIs are not built. The following parameters can be added to implement different functions:
- `-use_cxx11_abi1`: enables `C++11 ABI`. This is the default setting.
- `-use_cxx11_abi0`: disables `C++11 ABI`.
- `-cann`: CANN open APIs can be used for build if the CANN version is 8.5 or later.
- `-uttests`: builds all UT cases in the `tests/unittest` directory.
- `-examples`: displays all cases in the `examples` directory.
- `-python_example`: provides the PyTorch access capability for some cases in the `examples` directory.
- `-enable_rdma`: builds and enables RDMA capabilities. By default, the RDMA backend type is configured for Ascend 910B/C. For Ascend 950, you need to use `-rdma_backend` to specify the backend type.
- `-rdma_backend`: specifies the RDMA backend type (not supported by Ascend 910B/C). The value can be `XSCALE` (Yunsilicon NICs are used). This option must be used together with `-enable_rdma`. Otherwise, an error is returned. The parameter sequence is not limited.
- `-enable_ascendc_dump`: enables the `AscendC_Dump` mode for debugging the operator kernel code.
- `-package`: builds the installation package for delivery, and generates the run package and Python wheel at the same time (including the behavior of `-python_extension`)
    * generates the run package `SHMEM\_{version}\_linux-{arch}.run` in the `{project_root}/package/{arch}/` directory.
    * generates the Python wheel `cann_shmem-*.whl` containing both the 910/950 backends in the `{project_root}/package/{arch}/` directory.
- `-python_extension`: generates the Python wheel `cann_shmem-*.whl` containing both the 910/950 backends in the `{project_root}/dist/` directory.
- `-gendoc`: generates documents.
- `-onlygendoc`: generates documents without building the source code.
- `-debug`: sets the build type to `Debug`.
- `-mssanitizer`: enables the msSanitizer memory detection tool for samples. The script can be executed only after msSanitizer is used to start the task. The AllGather sample running script provides the `tool` option to use the tool. For details about how to use the tool, see [Exception Detection Tool (msSanitizer, MindStudio Sanitizer)](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0039.html). Note: If this option is enabled, ensure that the msSanitizer tool is used to start the operator. If other methods are used to start the operator, there may be unknown errors.
- `-soc_type`: If the SoC is Ascend 950, add the `-soc_type Ascend950` parameter. You can run the `npu-smi info` command to view the parameter. For other SoCs, you do not need to add this parameter.
- `-enable_simt`: enables the SIMT programming mode.
- `-enable_relay`: builds and enables the UDMA relay (detour) capability. Relay is built on top of UDMA, which is available only on Ascend 950, so this option **must be used together with `-soc_type Ascend950`**. It is also allowed in multi-SOC wheel/package builds (`-python_extension` / `-package` / `-full`), where relay is applied only to the 950 backend. For details, see [Relay Parameters](#relay-parameters) below.
- `-multi_link`: builds and enables UDMA multi-link routing, which supports per-QP network-layer route policies. It requires direct UDMA on Ascend 950 and cannot be used with `-enable_relay`. For details, see [Multi-link Parameters](#multi-link-parameters) below.
- `-clean`: removes the existing `build` directory before building to prevent CMake cache and previous build artifacts from affecting the current configuration.
- `-full`: compiles the package, Python extension, UT tests, and examples. The SHMEM automatic compilation script automatically downloads the dependency libraries, compiles the project and UT test cases, and packages the libraries.

### RDMA Parameters

To use the RDMA function, enable the `-enable_rdma` parameter during build. Then, you need to configure the backend based on the server type and NIC type.

- Ascend 910B/C platform
  - Only the `-enable_rdma` parameter is required. No additional parameter is required. The backend uses the default configuration.
- Ascend 950 platform
  - Use the `-enable_rdma` parameter.
  - Explicitly specify `-soc_type Ascend950`.
  - Specify the `-rdma_backend` parameter. Otherwise, the build script will report an error.

**Parameter dependency rules**:
- The `-enable_rdma` parameter must be specified when the `-rdma_backend` parameter is used.
- The `-rdma_backend` parameter is valid only when `-soc_type Ascend950` is used. Otherwise, the build script will report an error.
- If `-rdma_backend` is specified but `-enable_rdma` is not specified, the build will fail, and an error message will be displayed.
- The sequence of the three parameters (`-rdma_backend`, `-enable_rdma`, `-soc_type xxx`) is not limited, but all dependent parameters must be specified.

**Valid command examples**:

```shell
# Enabling RDMA on the Ascend 950 platform and specifying XSCALE as the backend (parameter sequence example 1)
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE

# Enabling RDMA on the Ascend 950 platform and specifying XSCALE as the backend (parameter sequence example 2)
bash scripts/build.sh -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE

# Enabling RDMA on the Ascend 950 platform and specifying XSCALE as the backend (parameter sequence example 3)
bash scripts/build.sh -rdma_backend XSCALE -enable_rdma -soc_type Ascend950

# Enabling RDMA on Ascend 910B/C (backend not specified)
bash scripts/build.sh -enable_rdma
```

**Invalid command examples**:

```shell
# Error: -rdma_backend is specified but -enable_rdma is not enabled.
bash scripts/build.sh -soc_type Ascend950 -rdma_backend XSCALE
# Error message: "Error: -rdma_backend requires -enable_rdma to be specified."

# Error: -rdma_backend is used on a non-Ascend 950 platform.
bash scripts/build.sh -enable_rdma -rdma_backend XSCALE
# Error message: "Error: -rdma_backend can only be specified when SOC_TYPE is Ascend950."
```

**RDMA backend types**:

The `-rdma_backend` parameter supports the following backend types:
- `XSCALE`: Yunsilicon NIC (supported only by Ascend 950)

**⚠️ Important**:

1. **Automatically-generated compilation definitions**:
   - CMake automatically generates the compilation definitions below based on the value of `-rdma_backend`. You do not need to manually set them.
     - When `-rdma_backend` is set to `XSCALE`, `-DACLSHMEMI_RDMA_K_BACKEND_XSCALE=1` is automatically added.
     - When no backend is specified, `-DACLSHMEMI_RDMA_K_BACKEND_IN_DIE=1` is automatically added.
   - Manually defining these macros may cause compilation conflicts or function exceptions.

2. **Internal macros that cannot be manually defined**:
   - Do not manually add the compilation definition named `ACLSHMEMI_K_RDMA_BACKEND`.
   - This is an internal macro used by the device code and is set by the system in `src/device/gm2gm/engine/shmem_device_rdma.hpp` based on the automatically-generated definition.
   - Manually defining this macro may cause compilation errors or runtime function exceptions.

### Relay Parameters

The relay (detour) capability allows communication traffic to be forwarded through other PEs in addition to the direct path, improving bandwidth utilization for specific topologies (for example, the tailcut mode of the `tp_allreduce_udma` sample). Relay is built on top of the UDMA transport, which is available only on Ascend 950. Therefore, the platform dependency must be satisfied when `-enable_relay` is used.

When `-enable_relay` is enabled, CMake automatically adds the `ACLSHMEM_RELAY_SUPPORT` compilation macro. You do not need to define it manually.

**Parameter dependency rules**:

- For a single-SOC build, `-enable_relay` **must** be used together with `-soc_type Ascend950`. Otherwise, the build script reports an error.
- In multi-SOC wheel/package builds (`-python_extension` / `-package` / `-full`), `-enable_relay` is allowed. In this case, the script enables relay only for the 950 backend, not the 910 backend.
- The sequence of `-enable_relay` and `-soc_type Ascend950` is not limited.

**Valid command examples**:

```shell
# Build examples and enable relay on the Ascend 950 platform
bash scripts/build.sh -examples -soc_type Ascend950 -enable_relay

# The reversed parameter sequence is also valid
bash scripts/build.sh -enable_relay -soc_type Ascend950 -examples

# Multi-SOC wheel build; relay is applied only to the 950 backend
bash scripts/build.sh -python_extension -enable_relay
```

**Invalid command examples**:

```shell
# Error: -enable_relay is used on a non-Ascend 950 platform (-soc_type Ascend950 not specified).
bash scripts/build.sh -examples -enable_relay
# Error message: "Error: -enable_relay requires UDMA support, which is only enabled for Ascend950."
```

> Note: Sample features that depend on relay, such as tailcut, must be compiled with `-enable_relay` enabled. For details, see `examples/tp_allreduce_udma/README.md`.

### Multi-link Parameters

Multi-link selects network-layer routes for different UDMA QPs targeting the same remote PE. When enabled, an
application can use `aclshmemx_set_qp_net_layer` to configure an explicit network-layer policy for an individual QP;
QPs without an explicit policy use default route selection. The current implementation supports UDMA layer 0 and
layer 1 only. The available routes are determined by runtime rootinfo and topology: peers in the same layer-0 network
instance can use MESH or CLOS, while peers in different instances select only reachable CLOS routes.

When `-multi_link` is enabled, CMake automatically adds the `ACLSHMEM_MULTI_LINK_SUPPORT` compilation macro. You do
not need to define it manually.

**Dependencies and limitations**:

- `-multi_link` **must** be used with `-soc_type Ascend950`.
- Multi-link uses direct UDMA and **cannot** be used with `-enable_relay`.
- Multi-link does not depend on `-enable_rdma` or `-rdma_backend`; those options configure the RoCE/RDMA backend.
- A multi-link build still requires available UDMA channels and rootinfo/topology at runtime. A successful build does
  not guarantee that every PE or network layer can establish a route.
- Call `aclshmemx_set_qp_num` and `aclshmemx_set_qp_net_layer` before SHMEM initialization. All participating PEs
  must use the same QP count and the same explicit per-QP network-layer policy. These settings return to defaults
  after the last SHMEM instance is finalized.
- Explicit policies apply only to peers in the same layer-0 network instance. Communication across instances ignores
  explicit layer policies and selects reachable CLOS routes. Therefore, the same QP can resolve to different actual
  network layers for different peers.

**Valid command examples**:

```shell
# Build Ascend 950 multi-link UDMA capability and examples
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link

# RDMA backend capability can be built together only on Ascend 950 systems with an XSCALE NIC; multi-link itself does not depend on RDMA
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link -enable_rdma -rdma_backend XSCALE
```

**Invalid command examples**:

```shell
# Error: multi-link is supported only on Ascend 950
bash scripts/build.sh -examples -multi_link
# Error message: "ACLSHMEM_MULTI_LINK_SUPPORT requires SOC_TYPE=Ascend950."

# Error: multi-link requires direct UDMA and cannot be used with relay
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link -enable_relay
# Error message: "ACLSHMEM_MULTI_LINK_SUPPORT currently requires direct UDMA; disable relay support."
```

For runtime QP configuration and multi-link validation, see `examples/udma_qp_demo/README.md`. The sample accepts
`-qp_net_layers qp_idx:netLayer[,qp_idx:netLayer...]` to set QP policies and uses `aclshmemx_get_qp_net_layer` to
query the actual network layer of an established route.

## Key SHMEM Files
### `scripts` Directory
   - `install.sh`: installation script
   - `uninstall.sh`: uninstallation script
   - `build.sh`: build script
   - `release.sh`: automatic build and packaging script
   - `set_env.sh`: SHMEM environment variable settings
### run.sh Script Usage
UT case execution script
```sh
bash scripts/run.sh
```
Multiple parameters are provided to support custom case execution.
```sh
-ranks          # Total number of ranks
-frank          # First rank on the server
-ipport         # IP address and port
-fnpu           # First NPU started on each server
-gnpus          # The number of devices used by a single server
-test_filter    # gtest_filter

# Example
bash scripts/run.sh -ranks 4 -fnpu 2 -gnpus 4 -test_filter ScalarP # The test will be performed on devices 2 to 6.

```
### install.sh
Installation/Uninstallation script of the generated .run file
Installation directory
```
${INSTALL_PATH}
    |--shmem
        |--latest
        |--${version}
            |--shmem
                |--include (header file)
                |--lib      (SO library)
            |--scripts      (uninstallation script)
```
Note: `INSTALL_PATH` indicates the user-defined installation directory.
### uninstall.sh
Uninstallation script, which can be used to uninstall the SHMEM library installed in the corresponding path or uninstall the SHMEM library in the default path using the `--uninstall` option of the .run file

### release.sh
Script for generating a package, which is used after compilation. After the compilation products are packaged, other files in the `install` directory are deleted. You are advised to use the `build.sh` script to complete the packaging.


### build.sh
File name: `scripts/build.sh`
SHMEM compilation file, which does not need to be modified

### set_env.sh
​File name: `scripts/set_env.sh`
After SHMEM is installed, the process-level environment variable setting script `set_env.sh` is provided to automatically set environment variables. The environment variables automatically become invalid after the user process ends.
