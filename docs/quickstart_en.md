# Quick Start

## Overview
This system is designed for model and operator developers on the Ascend platform. It provides a portable and easy-to-use multi-device multi-card memory access mode, facilitates the development of inter-device data synchronization, and accelerates the development of communication or communication-computation fused operators.

## Software Architecture
The shared memory library APIs are classified into host-side APIs and device-side APIs.
- Host-side APIs provide initialization, memory management, team management, and synchronization functions.
- Device-side APIs provide memory access, synchronization, and team management functions.

## Directory Structure
For details, see [Organizational Structure of Code](code_organization_en.md).
```
├── 3rdparty // Dependent third-party libraries
├── docs     // Documents
├── examples // Usage examples
├── include  # Header files
├── scripts // Related scripts
├── src      // Source Code
├── tests    // Test cases
```

## Software and Hardware Versions

### Hardware and toolchain

SHMEM supports Atlas 800I A2/A3, Atlas 800T A2/A3, and Ascend 950PR/950DT products on AArch64 and x86 hosts. For a source build, prepare GCC and G++ 7.3.0 or later (use matching versions), CMake 3.19 or later, glibc 2.28 or later, and Python 3.9 or later. The build uses the CANN toolkit's `bisheng` compiler for AscendC kernels. The Python wheel has additional requirements listed under [Python installation](#python-installation).

### CANN and driver compatibility

Use the following combinations from the [Chinese quick start](quickstart.md). Check the transfer direction as well as the CANN version when selecting a configuration.

| Minimum HDK | CANN release | Device to device | Device to/from host | Device to/from remote host |
| --- | --- | --- | --- | --- |
| 25.5.6 | 9.1.0 and later | MTE, RDMA, SDMA (CMO only on Ascend 950), UDMA (Ascend 950 only) | MTE, SDMA (A3 only for SDMA) | MTE and SDMA on A3 |
| 25.5.1 | 9.0.0 | MTE, RDMA, SDMA (A2/A3 only for SDMA) | MTE, SDMA (A3 only for SDMA) | MTE and SDMA on A3 |
| 25.5.1 | 8.5.0 | MTE, RDMA | MTE on A3 | MTE on A3 |
| 25.2.3 | 8.3.RC1 | MTE, RDMA | — | — |

For SDMA, install the ops package matching the toolkit release and SoC. On Ascend 950, UDMA, RDMA, and SDMA also require matching toolkit and HDK packages. For remote-host transfers on A3 with CANN 8.5.0, see the [LingQu Computing Network 1.5.0 downloads](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841/software).

SHMEM calls HCCP APIs when using RDMA on Atlas A2 and A3 products. The supported CANN T releases are `8.2.T101`, `8.3.T101`, `8.5.T101`, `9.0.T101`, `9.1.T101`, and `9.2.T101`.

### Install CANN

Download the [driver and firmware](https://www.hiascend.com/hardware/firmware-drivers) and the [CANN packages](https://www.hiascend.com/developer/download/community/result?module=cann) for your SoC, CPU architecture, and OS. Follow the [CANN installation guide](https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netyum&OS=openEuler).

For releases before 9.1.0, install the toolkit. Community releases also require the matching SoC ops package. Commercial releases from 8.5.0 moved components such as HCCL into the ops package. A missing ops package can leave `libhccl.so` unavailable at runtime. For example, on CANN 9.0.0 the ops package is named `Ascend-cann-910b-ops_9.0.0_linux-x86_64.run` for 910B, with `A3` or `950` in place of `910b` for those SoCs.

From the directory containing the downloaded packages, set `cann_version` and `soc_name` to match their filenames. To select a custom installation directory, add `--install-path="${install_path}"` to both installation commands.

```bash
chmod +x "Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run"
./"Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run" --install
chmod +x "Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run"
./"Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run" --install
```

Load the toolkit environment in each shell. For a custom installation, replace `/usr/local/Ascend` with the installation directory.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Python and other build dependencies

After cloning the repository, run the following commands from its root. The second file adds the packages used by the examples. Either file can be installed first and both use the same NumPy 1.x constraint.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-examples.txt
```

| Dependency | When needed | Offline preparation |
| --- | --- | --- |
| PyTorch and TorchNPU | Python wheel and examples that work with NPU tensors | Download wheels matching CANN, Python, and the CPU architecture. |
| googletest v1.14.x | Building unit tests with `-uttests` | Place the source tree in `3rdparty/googletest`. |
| nlohmann/json v3.11.3 | Building for `-soc_type Ascend950` | Place the source tree in `3rdparty/json`. |

The build script fetches the two source dependencies from GitCode when needed. For offline builds, also stage the CANN installation packages and Python wheels, or use an internal package index. Install OpenMPI 4.0 or later for MPI-based examples. For RDMA with an HNS_1825 NIC, driver and firmware versions above 30.5.1.36 require a SHMEM release later than v1.7.0.

### Docker

Choose a [CANN container image](https://www.hiascend.com/developer/ascendhub) for your chip and OS, then follow the same source and dependency setup inside it. See the [container quick start](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884) for device mounts. Running unit tests and device examples inside a container requires the host NPU, driver, and CANN runtime.

## Quick Start

Load the CANN environment and run `npu-smi info` to check that your devices are visible. Choose a source build for C/C++ development, a `.run` package for C/C++ deployment, or a Python wheel for the Python API.

### Build from source

```bash
git clone https://gitcode.com/cann/shmem.git
cd shmem
```

Build from the repository root. Use `bash scripts/build.sh` on A2/A3, or `bash scripts/build.sh -soc_type Ascend950` on Ascend 950. Then load the installation and check the shared library:

```bash
source install/set_env.sh
test -f install/shmem/lib/libshmem.so
```

The default build contains the core library. See the [compilation and build guide](compilation_build_guide_en.md) to add xDMA capabilities, examples, or tests.

### Install a binary package

Build the package with `bash scripts/build.sh -package` on A2/A3 or `bash scripts/build.sh -soc_type Ascend950 -package` on Ascend 950. The `package/$(uname -m)/` directory contains the `SHMEM_{version}_linux-{arch}.run` installer and the Python wheel.

From a directory containing a single matching installer, check and install the package:

```bash
cd "package/$(uname -m)"
chmod +x SHMEM_*.run
./SHMEM_*.run --check
./SHMEM_*.run --install
```

The check reports chip count, HDK and CANN versions, supported transport engines, and package contents. Review any `WARN` entries. Use `--install --check` to check and install in one command, or `--install-path=/opt/ascend` to install under `/opt/ascend/shmem`. The custom path must be absolute. Run the installer as the owner of the CANN installation.

After installation, load `/usr/local/Ascend/shmem/latest/set_env.sh` for a default root install, `"${HOME}/Ascend/shmem/latest/set_env.sh"` for a default non-root install, or `/opt/ascend/shmem/latest/set_env.sh` for the custom path above. For example:

```bash
source /usr/local/Ascend/shmem/latest/set_env.sh
test -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so"
```

### Python installation

The `cann-shmem` wheel supports Python 3.10 through 3.12. Install compatible PyTorch and TorchNPU wheels first. Select the combination for your CANN release, Python version, and CPU architecture from the [TorchNPU compatibility matrix](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md).

Install SHMEM from the Ascend index:

```bash
python3 -m pip install cann-shmem \
  --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/
```

If other dependencies are unavailable there, pip also accepts `--extra-index-url https://pypi.org/simple/`. Pip considers candidates from both indexes and does not guarantee source priority. For production, install the matching TorchNPU wheel first and use just the Ascend index for SHMEM.

Alternatively, build and install a wheel from the repository root. The `--no-deps` option preserves the PyTorch and TorchNPU versions installed above.

```bash
bash scripts/build.sh -python_extension
source install/set_env.sh
python3 -m pip install --no-deps --force-reinstall dist/cann_shmem-*.whl
```

Check the selected backend and package contents:

```bash
shmem-config --version
shmem-config --backend
shmem-config --check
shmem-config --diagnose
```

Expect backend `910` on A2/A3 and `950` on Ascend 950. In the diagnostic JSON, `native_load.ok` and `backend_artifacts.complete` should be true, while `degraded` and `multi_so_conflict.detected` should be false. For a source-built wheel, run `bash tests/package_smoke/run_install_smoke.sh dist/cann_shmem-*.whl` for the full installation smoke test. See the [shmem-config command reference](zh/tools/shmem_config_guide.md) for the report fields.

### Run an example

From the repository root, build examples with `bash scripts/build.sh -examples` on A2/A3 or `bash scripts/build.sh -soc_type Ascend950 -examples` on Ascend 950. Run allgather on two devices:

```bash
cd examples/allgather
bash run.sh -pes 2 -type int32_t
```

The [allgather README](../examples/allgather/README_en.md) describes the data type, address, port, and local device and PE options. For a debug build, add `-debug` to the build command. See the [troubleshooting guide](en/debug/Troubleshooting_FAQs.md).

## Function self-test case

Building tests requires the toolkit, `bisheng`, and the build dependencies above. Running them requires accessible NPUs, the driver, and the CANN runtime.

From the repository root, build unit tests with `bash scripts/build.sh -uttests` on A2/A3 or `bash scripts/build.sh -soc_type Ascend950 -uttests` on Ascend 950. Run the suite or select a subset by name:

```bash
bash scripts/run.sh
# Run tests whose names contain Init on eight devices.
bash scripts/run.sh -ranks 8 -test_filter Init
```

After building with `-examples`, use `bash scripts/run_examples.sh` to run the example suite. To run an individual example, follow the README in its directory under [examples](../examples/). For a new GoogleTest case, place the source under [tests/unittest](../tests/unittest/) and follow the test conventions in the [Chinese quick start](quickstart.md).

## Python Test Cases

After [installing and checking the wheel](#python-installation), run the supplied initialization test on two devices from the repository root:

```bash
torchrun --nproc-per-node=2 examples/python_extension/test/init_test.py
```

The test prints `init_test.py running success!` on success. For custom tests, see the [Python API list](en/api/pythonAPI.md) and [additional test cases](../examples/python_extension/test/).

## Unique ID Initialization Method

Note: When the API for the unique ID is used for initialization, you can manually configure the environment variable `SHMEM_UID_SESSION_ID` or `SHMEM_UID_SOCK_IFNAME`. If both are configured, only `SHMEM_UID_SESSION_ID` will be read. If neither is configured, available network ports are automatically searched for by default. `SHMEM_UID_SESSION_ID` supports IPv4 addresses, IPv6 addresses, and host names.
SHMEM_UID_SESSION_ID configuration example:

```bash
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666%eth]:886
SHMEM_UID_SESSION_ID=localhost:8888
```

SHMEM_UID_SOCK_IFNAME configuration example:

```bash
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  # IPv4
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  # IPv6
SHMEM_UID_SOCK_IFNAME=eth0          # Automatically detect available protocols (IPv4 preferred)
```

If this parameter is not set, available network ports (IPv4/IPv6) are automatically searched for by default. The search priority is as follows: non-virtual network ports (excluding lo/docker/veth/br-/virbr/tun/tap) >> virtual network ports.

In Python, rank 0 generates the ID and broadcasts it to its peers before SHMEM initialization. The [working example](../examples/python_extension/test/unique_id_test.py) also sets the device and initializes the process group:

```bash
torchrun --nproc-per-node=2 examples/python_extension/test/unique_id_test.py
```

In C++, share the ID from `aclshmemx_get_uniqueid` with all PEs, then call `aclshmemx_set_attr_uniqueid_args` and `aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes)` on each PE. See the [host initialization API](../include/host/init/shmem_host_init.h) for the arguments and return values.

## Configuration and tuning

- TLS is enabled by default. Configure certificates before initialization. For a trusted test network, the C API accepts `aclshmemx_set_conf_store_tls(false, NULL, 0)` and the Python API accepts `shm.set_conf_store_tls(False, "")`. See [Security Statements](SECURITY_en.md) for the production configuration.
- Set the per-PE symmetric heap size in bytes during initialization. For example, pass `32ULL * 1024 * 1024 * 1024` as the size to `aclshmemx_set_attr_uniqueid_args` for a 32 GiB heap on a device with sufficient memory. Use the same size on every PE.
- Use device-side operations to reduce host/device interactions and group PEs by physical node to reduce cross-node transfers. For large transfers over RDMA, build with `-enable_rdma`. For Ascend 950, see the [build guide](compilation_build_guide_en.md) to select the RDMA backend for your NIC.
