# Quick Start

## 1 Introduction

This system is designed for model and operator developers on the Ascend platform. It provides a convenient and easy-to-use multi-node multi-card memory access method, making it easy for users to synchronize data across cards during development, and accelerating the development of communication operators and communication-computation fusion operators.

## 2 Software Architecture

The shared memory library APIs are mainly divided into host APIs and device APIs:

- Host-side APIs provide initialization, memory management, communicator management, and synchronization functions.
- Device-side APIs provide memory access, synchronization, and communicator management functions.

## 3 Directory Structure Description

For details, see [code_organization](code_organization_en.md)

```text
├── 3rdparty // Third-party library dependencies
├── docs     // Documents
├── examples // Usage samples
├── include  // Header files
├── scripts  // Related scripts
├── src      // Source code
└── tests    // Test cases
```

## 4 Environment Preparation

### 4.1 Hardware Requirements

- Atlas series: 800I A2/A3, 800T A2/A3
- Ascend 950PR&950DT products
- Architecture compatibility: aarch64, x86

### 4.2 Toolchain Dependencies

   - gcc >= 7.3.0 / g++ >= 7.3.0 (Note: The gcc and g++ versions must be consistent)
   - cmake ≥ 3.19
   - GLIBC ≥ 2.28
   - Python >= 3.9.0
   - bisheng (installed with the CANN toolkit, used to compile device-side AscendC kernels)

By default, the project uses `bisheng` provided by the CANN toolkit as the C/C++ compiler. You must complete the CANN toolkit installation before compilation, and then run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

If CANN is installed in a custom path, run:

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 4.3 CANN

#### 4.3.1 CANN Version Description

| Driver and Firmware | CANN Version | D2D | D2H/H2D | D2rH/rH2D | Other Dependencies |
| --- | --- | --- | --- | --- | --- |
| Ascend HDK 25.5.6 or later<br>[HDK resources](https://www.hiascend.com/hardware/firmware-drivers?ids=d803%2C89dda9ba9de741349efa03687a487678%2C25%2CAArch64%2Conline_Yum) | 9.1.0 or later<br>[CANN resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0) | MTE<br>RDMA<br>SDMA (CMO only on Ascend950)<br>UDMA (Ascend950 only) | MTE<br>SDMA (A3 only) | MTE (A3 only)<br>SDMA (A3 only) | To enable UDMA, RDMA, or SDMA on Ascend950, download the corresponding toolkit and Ascend HDK packages<br> To enable SDMA, additionally install the [ops package](https://www.hiascend.com/developer/download/community/result?module=cann) that matches the toolkit version and device |
| Ascend HDK 25.5.1 or later<br>[HDK resources](https://www.hiascend.com/hardware/firmware-drivers?ids=d803%2C89dda9ba9de741349efa03687a487678%2C25%2CAArch64%2Conline_Yum) | 9.0.0 or later<br>[CANN resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | MTE<br>RDMA<br>SDMA (A2/A3 only) | MTE<br>SDMA (A3 only) | MTE (A3 only)<br>SDMA (A3 only) | To enable SDMA, download the [ops package](https://www.hiascend.com/developer/download/community/result?module=cann) that matches the toolkit version and device type |
| Ascend HDK 25.5.1 or later<br>[HDK resources](https://www.hiascend.com/hardware/firmware-drivers?ids=d803%2C89dda9ba9de741349efa03687a487678%2C25%2CAArch64%2Conline_Yum) | 8.5.0 or later<br>[CANN resources](https://www.hiascend.com/developer/download/community/result?module=cann) | MTE<br>RDMA | MTE (A3 only) | MTE (A3 only) | To enable A3 D2rH/rH2D: LingQu Computing Network [version 1.5.0](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841/software)<br>Upgrade guide: [Installation Guide](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841) |
| Ascend HDK 25.2.3 or later<br>[HDK resources](https://www.hiascend.com/hardware/firmware-drivers?ids=d803%2C89dda9ba9de741349efa03687a487678%2C25%2CAArch64%2Conline_Yum) | 8.3.RC1 or later<br>[CANN resources](https://www.hiascend.com/developer/download/community/result?module=cann) | MTE<br>RDMA |  |  | |

When SHMEM uses RDMA on Atlas A2 and A3 products, it calls HCCP APIs. Therefore, the compatibility with CANN T versions is described separately here. The currently compatible CANN T versions are `8.2.T101`, `8.3.T101`, `8.5.T101`, `9.0.T101`, `9.1.T101`, and `9.2.T101`.

#### 4.3.2 Installing the CANN Packages

Refer to [Install CANN Quickly](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=local)

Commercial releases earlier than 9.1.0 require the `.run` package of the `toolkit`. Since CANN 8.5.0, operators such as `hccl` have been migrated to the ops package. These commercial releases also require the ops package for the corresponding SoC. Otherwise, the runtime reports an error indicating that library files such as `libhccl.so` are missing.

Community releases earlier than 9.1.0 require an additional `.run` installation package named `Ascend-cann-{soc_name}-ops-{cann_version}-{os_arch}.run`. Select the CANN version, and then select the versions matching the SoC, operating system, and architecture. (If the `{soc_name}-ops` operator package is unavailable, a message indicating that the `libhccl.so` library file is missing may be displayed.)

Resources: [CANN 9.1.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0) and [CANN 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0).

The ops package file names corresponding to each SoC are as follows (using CANN 9.0.0 and x86_64 as an example):

| SoC Version | Remarks | ops Package File Name |
|----------|------|-------------|
| Ascend 910B | 910b | `Ascend-cann-910b-ops_9.0.0_linux-x86_64.run` |
| A3 | - | `Ascend-cann-A3-ops_9.0.0_linux-x86_64.run` |
| Ascend 950PR&950DT products | 950 | `Ascend-cann-950-ops_9.0.0_linux-x86_64.run` |

Install the toolkit package

```sh
chmod +x Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run
# Default installation path
./Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run --install
# Custom installation path
./Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run --install --install-path=${install_path}
```

Install the ops package

```sh
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run
# Default installation path
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run --install
# Custom installation path
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run --install --install-path=${install_path}
```

Configure the CANN environment variables (default installation path):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Configure the CANN environment variables (custom installation path):

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 4.4 Summary of Build and Run Dependencies

| Dependency | Usage Scenario | How to Obtain | Offline Environment Preparation |
| --- | --- | --- | --- |
| CANN toolkit | Compilation and running of the core library, device kernels, examples, and unit tests (UTs) | Install the CANN toolkit `.run` package | Download the toolkit installation package that matches the CPU architecture in advance |
| CANN ops package | Runtime dependency for feature support scenarios such as CANN and SDMA | Install the ops `.run` package that matches the SoC and CANN version | Download the ops package that matches the SoC, CANN version, and CPU architecture in advance |
| bisheng | Compilation of device-side AscendC kernels | Provided with the CANN toolkit | Run `set_env.sh` after the toolkit is installed |
| Python dependencies | Build scripts, Python examples, and tests | `requirements.txt` and `requirements-examples.txt` | Prepare pip wheels in advance or use the internal PyPI source |
| TorchNPU | Installation and running of the Python wheel; constructing NPU tensors from SHMEM memory | Install `torch-npu` according to the official matching relationship | Download the wheel that matches CANN, Python, and the CPU architecture in advance |
| googletest v1.14.x | Unit test build with `-uttests` | Automatically pulled from GitCode by `scripts/build.sh` | Pre-place the package in `3rdparty/googletest` |
| nlohmann/json v3.11.3 | Build dependency on the Ascend950 platform | Automatically pulled from GitCode by `scripts/build.sh -soc_type Ascend950` | Pre-place the package in `3rdparty/json` |

### 4.5 Installing the `Pytorch` Framework and the `TorchNPU` Plugin

You must install this package to compile and run operators that use torch tensors as input and output.

Using `CANN 9.0.0` as an example, refer to the [Ascend for PyTorch 26.0.0-Ascend Community](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) document.

Open the page and select the corresponding version based on the system architecture, CANN version, and Python version. Then perform the installation using the installation command provided at the bottom of the page.

The following uses `Python 3.11`, `PyTorch 2.7.1`, `TorchNPU 26.0.0`, and `aarch64` as an example.

+ Install the `PyTorch` framework:

```sh
# Download the PyTorch framework
wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl
# Install
pip3 install torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_aarch64.whl
```

+ Install the `TorchNPU` plugin

```sh
# Download the TorchNPU plugin
wget https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.7.1/torch_npu-2.7.1.post4-cp311-cp311-manylinux_2_28_aarch64.whl
# Installation command
pip3 install torch_npu-2.7.1.post4-cp311-cp311-manylinux_2_28_aarch64.whl
```


### 4.6 Installing Python Dependencies

> **Prerequisites**: This section uses the `requirements.txt` and `requirements-examples.txt` files in the repository root directory. Complete the code cloning step in [5.1.1 Source Compilation](#511-method-1-source-compilation) first.

```bash
# Basic dependencies (hard dependencies at build time and run time)
python3 -m pip install -r requirements.txt

# Additional dependencies for running Python examples (install as needed; can run independently)
# requirements-examples.txt has the same numpy 1.x constraint as requirements.txt
# (numpy>=1.24.4,<=1.26.4), and pandas/matplotlib/ml_dtypes are capped at versions
# compatible with numpy 1.x. Running it alone or in any order does not upgrade numpy
# to 2.x (code depending on the numpy 1.x ABI behaves unpredictably under numpy 2.x).
python3 -m pip install -r requirements-examples.txt
```

### 4.7 Third-Party Source Code Dependencies

Based on the build options, `scripts/build.sh` automatically prepares some third-party source code dependencies:

- `googletest v1.14.x`: Automatically pulled and installed to `3rdparty/googletest` when you run `bash scripts/build.sh -uttests`. It is used for gtest/gmock unit tests.
- `nlohmann/json v3.11.3`: Automatically pulled to `3rdparty/json` when you run `bash scripts/build.sh -soc_type Ascend950`. It is a build dependency related to the Ascend950 platform (such as UDMA/RDMA paths).

If the build environment cannot access GitCode, prepare the dependency directories in an environment with Internet access in advance, and copy them to `3rdparty/` in the repository root directory:

```text
3rdparty/
├── googletest/
└── json/
```

### 4.8 Optional Dependencies

- MPI: OpenMPI 4.0+ (distributed communication scenarios)
- PyTorch: 1.12+ (running Python examples)
- HNS_1825 network adapter: The network adapter driver and firmware of version 30.5.1.36 or later do not support shmem v1.7.0 or earlier (RDMA scenario)

### 4.9 Docker Container Environment

Based on the chip type and operating system, you can pull a container with the CANN environment preconfigured from the [Ascend image repository](https://www.hiascend.com/developer/ascendhub) for project development. For how to use Docker, refer to the [Quick Start](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884) section in the Ascend CANN image repository

This repository currently does not maintain an independent Dockerfile. It is recommended that you use the official Ascend CANN container image as the base environment, and then pull the source code, prepare third-party dependencies, and compile and run the project as described in this document. Running UTs or examples in a container still requires the host machine to provide an available NPU, driver, device mounting, and the CANN runtime.

## 5 Getting Started

### 5.1 Installation Methods

This section applies to A2/A3 and Ascend950. Before installation, complete [Environment Preparation](#4-environment-preparation), load the CANN environment variables, and run `npu-smi info` to confirm that the devices are visible.

Select an installation method based on the usage scenario:

| Usage Scenario | Installation Method | Artifact or Installation Location |
| --- | --- | --- |
| Developing C/C++ operators from source code | Source compilation | The `install/` directory of the repository |
| Deploying C/C++ dynamic libraries | Binary run package | `/usr/local/Ascend/shmem`, `$HOME/Ascend/shmem`, or a custom path |
| Using the Python API | pip or a local Python wheel | `site-packages` of the current Python environment |

#### 5.1.1 Method 1: Source Compilation

```bash
# Clone the code repository
git clone https://gitcode.com/cann/shmem.git
cd shmem

# Compile the core library (xDMA capability, examples, and tests are not included by default)
# A2/A3 platform
bash scripts/build.sh
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950

# Configure environment variables
source install/set_env.sh

# Confirm that the core dynamic library has been generated
test -f install/shmem/lib/libshmem.so
```

The default build does not include xDMA capability. For the build parameters and parameter dependencies of communication capabilities such as RDMA, see [Compilation and Build](compilation_build_guide_en.md).

#### 5.1.2 Method 2: Binary Package Installation

How to obtain:

- A2/A3 platform: `bash scripts/build.sh -package`
- Ascend950 platform: `bash scripts/build.sh -soc_type Ascend950 -package`

Build artifacts are located in `package/$(uname -m)/`:

- `SHMEM_{version}_linux-{arch}.run`: C/C++ library installation package.
- `cann_shmem-*.whl`: A Python wheel that contains both the 910 and 950 backends.

Here, `{version}` indicates the software version, and `{arch}` indicates the CPU architecture.

```bash
chmod +x package/$(uname -m)/SHMEM_*.run

# Perform only the chip, version, topology, communication capability, and package artifact checks
package/$(uname -m)/SHMEM_*.run --check

# Choose one of the following three installation methods
package/$(uname -m)/SHMEM_*.run --install --check  # Check first and then install
package/$(uname -m)/SHMEM_*.run --install          # Install directly
package/$(uname -m)/SHMEM_*.run --install-path=/opt/ascend
```

Before the run package is installed, the installer checks the CANN environment variables and confirms that the current user is the owner of the CANN installation directory. `--install-path` must be an absolute path. In the example, the package is installed to `/opt/ascend/shmem`.

`--check` outputs the number of chips and devices, HDK/CANN versions, MTE/SDMA/UDMA/RDMA support status, and package artifact status. If a check item reports `WARN`, handle it according to the final summary. A successful exit of the current command does not mean that all optional communication capabilities meet the requirements. For details about the check items, see the [shmem-config Command Reference](en/tools/shmem_config_guide.md).

After the terminal outputs `SHMEM {version} install success, InstallPath: {install_dir}`, load the environment variables based on the executing user:

```bash
# Default path for the root user
source /usr/local/Ascend/shmem/latest/set_env.sh

# Default path for a non-root user
source "${HOME}/Ascend/shmem/latest/set_env.sh"

# Custom path /opt/ascend
source /opt/ascend/shmem/latest/set_env.sh

# Confirm the installation result
test -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so"
```

#### 5.1.3 Method 3: pip Installation

You can install from the Ascend PyPI source or build a local wheel in the repository root directory. Choose either of the two methods:

> **Prerequisites**: The `cann-shmem` Python wheel supports only Python 3.10 to 3.12. First install the `torch-npu` version that matches the current CANN version, Python version, and CPU architecture as described in
> [Installing the `Pytorch` Framework and the `TorchNPU` Plugin](#45-installing-the-pytorch-framework-and-the-torchnpu-plugin). The Ascend PyPI source currently
> does not necessarily contain `torch-npu`. If you specify only this private source in a clean environment, the following error may occur:
> `No matching distribution found for torch-npu`.

```bash
# Recommended: After the matched torch-npu is installed, install cann-shmem from the Ascend PyPI source
python3 -m pip install cann-shmem \
  --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/

# Convenient method: Allow pip to query the public PyPI when the private source lacks dependencies
python3 -m pip install cann-shmem \
  --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/ \
  --extra-index-url https://pypi.org/simple/

# Or build and install a local wheel
bash scripts/build.sh -python_extension
python3 -m pip install --no-deps --force-reinstall dist/cann_shmem-*.whl
```

The `--no-deps` option in the local wheel installation command skips dependency installation and reinstalls only SHMEM, preventing the matched PyTorch and TorchNPU from being replaced. Before running the command, you must prepare these two packages according to the preceding prerequisites.

`--extra-index-url` makes pip merge all candidate versions from the two sources. It does not ensure that a package with the same name is downloaded preferentially from `--index-url`. To avoid dependency confusion and the automatic selection of the latest TorchNPU version that does not match CANN, it is recommended that in a production environment you first install the matched `torch-npu` by following the official TorchNPU documentation, and then install `cann-shmem` using only the Ascend private source.

The Python wheel contains both the 910 backend for A2/A3 and the 950 backend for Ascend950, and the backend is automatically selected at run time. After installation, run:

```bash
shmem-config --version
shmem-config --backend
shmem-config --diagnose
```

The pass criteria are as follows:

- In the `--diagnose` output, `native_load.ok` is `true`, and `degraded` is `false`.
- The `--backend` output is `910` on A2/A3 and `950` on Ascend950.
- The `--diagnose` output is valid JSON, `backend_artifacts.complete` is `true`, and `multi_so_conflict.detected` is `false`.

After building the wheel from the source code, you can also run the complete smoke test script:

```bash
bash tests/package_smoke/run_install_smoke.sh dist/cann_shmem-*.whl
```

For details about the automatic check upon the first `import shmem`, `--check`, `--diagnose`, and other parameters, see the [shmem-config Command Reference](en/tools/shmem_config_guide.md).

#### 5.1.4 General Configuration (Optional)

SHMEM enables TLS communication encryption by default, and this applies to the preceding three installation methods. To disable it, you need to proactively call the API before initialization:

```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```

For details, see the security statement section.

### 5.2 Running Samples

Use `allgather` as an example to verify the core functions:

1. Compile the code in the shmem/ directory:

- A2/A3 platform:

```sh
bash scripts/build.sh -examples
```

- Ascend950 platform:

```sh
bash scripts/build.sh -soc_type Ascend950 -examples
```

2. Run the demo in the shmem/examples/allgather directory:

```sh
bash run.sh -pes 2 -type int32_t
```

Note: The example and other sample code are for reference only. Use them with caution in production environments.

The `run.sh` script provides parameters such as `-pes`, `-type`, `-ipport`, `-gnpus`, `-fnpu`, and `-fpe` to customize the number of cards, data type, IP port, number of NPUs, starting NPU ID, and starting PE ID of the executed test case. Example:

```sh
# Two cards, data type int32_t
bash run.sh -pes 2 -type int32_t
```

For more details about the parameters, see `examples/allgather/README_en.md`. For the `-ranks` and `-test_filter` parameters of the unit test script `scripts/run.sh`, see [8.1 Running Unit Tests](#81-running-unit-tests).

### 5.3 Using the debug Mode

Compile the code in the source shmem/ directory:

- A2/A3 platform:

   ```sh
   bash scripts/build.sh -examples -debug
   ```

- Ascend950 platform:

   ```sh
   bash scripts/build.sh -soc_type Ascend950 -examples -debug
   ```

Note: The `-examples` parameter here is optional. It only provides a [usage reference](en/debug/Troubleshooting_FAQs.md#shmem-faqs).

## 6 Python-Side Test Cases

[Python API List](en/api/pythonAPI.md)

1. Build or install the Python wheel by following "Method 3: pip Installation" in [Installation Methods](#51-installation-methods), and complete the backend and installation integrity checks.

2. Set whether to enable TLS authentication. TLS authentication is enabled by default. To disable TLS authentication, use the following API:

```python
import shmem as shm
shm.set_conf_store_tls(False, "")   # Disable TLS authentication
```

   ```python
   import shmem as shm
   tls_info = "xxx"
   shm.set_conf_store_tls(True, tls_info)      # Enable TLS authentication
   ```

3. Use torchrun to run the test demo

`test.py` is a test script that you write yourself. You can write it by referring to the existing test files (for example, `core/test_init_final.py`) in the `examples/python_extension/test/` directory of the repository.

```sh
torchrun --nproc-per-node=<n> test.py # <n> is the rank size to run. Replace it with the actual value
```
If the log prints "test.py running success!", the demo runs successfully

## 7 `unique id` Initialization Method

Note: When initializing using the unique id APIs, you can manually configure the environment variable SHMEM_UID_SESSION_ID or SHMEM_UID_SOCK_IFNAME. If both variables are configured, only SHMEM_UID_SESSION_ID is read. If neither variable is configured, available network interfaces are automatically searched by default. SHMEM_UID_SESSION_ID supports IPv4 addresses, IPv6 addresses, and host names.
SHMEM_UID_SESSION_ID configuration examples:

```bash
export SHMEM_UID_SESSION_ID=127.0.0.1:1234
export SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
export SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666%eth]:886
export SHMEM_UID_SESSION_ID=localhost:8888
```

SHMEM_UID_SOCK_IFNAME configuration examples:

```bash
export SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  # Use IPv4
export SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  # Use IPv6
export SHMEM_UID_SOCK_IFNAME=eth0           # Automatically detect the available protocol (prefer IPv4)
```

If neither variable is configured, available network interfaces are automatically searched by default (both IPv4 and IPv6 are supported). Search priority: non-virtual network interfaces (excluding lo/docker/veth/br-/virbr/tun/tap) >> virtual network interfaces.

- Python initialization example

```python
import shmem as ash

# xxx

uid = ash.aclshmem_get_unique_id()
ret = ash.aclshmem_init_using_unique_id(rank, world_size, mem_size, uid)

# xxx
```

After preparing the preceding startup code `init.py`, run `torchrun --nproc-per-node 8 init.py`. Change the process count 8 based on actual requirements. For more detailed examples, refer to the `unique_id_test.py` file.

- C++ initialization example

```cpp
aclshmemx_uniqueid_t uid;
aclshmemx_init_attr_t attributes;
int ret = aclshmemx_get_uniqueid(&uid);
ret = aclshmemx_set_attr_uniqueid_args(my_pe, n_pes, mem_size, &uid, &attributes);
ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);
```

## 8 Testing Framework

- **Unit tests:** Cover core APIs (initialization, memory operations, synchronization, and so on), located in `tests/unittest/`

- **Operator generalization tests:** Target communication-computation fusion operator samples, and support dynamic test data generation and precision verification

Tests and samples are divided into the compilation phase and the running phase. The compilation phase requires the CANN toolkit, `bisheng`, and corresponding third-party dependencies. In the running phase, `scripts/run.sh` and `scripts/run_examples.sh` require an available NPU, driver, CANN runtime, and correct rank/NPU parameters. A common container environment without NPUs can be used for dependency checks and compilation verification, but cannot fully verify the runtime behavior of UTs or examples.

### 8.1 Running Unit Tests

```bash
# Compile and run unit tests
# A2/A3 platform
bash scripts/build.sh -uttests
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -uttests
bash scripts/run.sh
```

The run.sh script provides parameters such as `-ranks` and `-test_filter` to customize the number of cards, `gtest_filter`, and other settings of the executed test cases. For example:

```bash
# Eight cards, run all *Init* test cases
bash scripts/run.sh -ranks 8 -test_filter Init
```

For details about the parameters, see the "run.sh Script Usage" section in the [Compilation and Build](compilation_build_guide_en.md) document.

### 8.2 Compiling and Running Samples

```bash
# A2/A3 platform
bash scripts/build.sh -examples
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples
bash scripts/run_examples.sh
```

### 8.3 Running a Sample Independently

You can view the README.md in the corresponding sample directory under the examples directory

### 8.4 Running Python Test Cases

First install the matched PyTorch and TorchNPU by following the prerequisite description in [pip Installation](#513-method-3-pip-installation).
The following commands use `--no-deps` to retain installed dependencies.

```bash
# Build a Python wheel that contains both the 910 and 950 backends
bash scripts/build.sh -python_extension
# Install the wheel package generated by the build script
python3 -m pip install --no-deps --force-reinstall dist/cann_shmem-*.whl
# Confirm that the runtime environment selects the correct backend
shmem-config --backend
shmem-config --diagnose
```

```bash
# Run the Python test on two cards
torchrun --nproc-per-node=2 examples/python_extension/test/init_test.py
```

### 8.5 Custom Tests

Based on the gtest framework in the `tests/` directory, follow these rules when adding new test cases:

- Test file naming: `{module}_test.cc`
- Test case naming: `{FunctionName}_{Scenario}_Test`

## 9 Configuration and Tuning (Optional, Advanced Usage)

**1. Disable TLS Encryption (Improving Communication Performance)**

TLS encryption is enabled by default. In a trusted intranet environment, you can disable it to improve the communication speed:

```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```

```python
import shmem as shm
shm.set_conf_store_tls(False, "")
```

**2. Adjust the Shared Memory Size**

Specify the shared memory pool size during initialization (16GB by default) to adapt to large-memory scenarios:

```c
aclshmemx_init_attr_t attr;
attr.local_mem_size = 32 * 1024 * 1024 * 1024; // 32GB
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

**3. Performance Tuning Recommendations**

- Prefer device-side APIs to reduce Host-Device interactions;
- When dividing communicators into groups, divide them by physical machine node to reduce cross-node communication;
- For large-batch data transfer, enable the RDMA protocol (add `-DSHMEM_RDMA=ON` during compilation).
