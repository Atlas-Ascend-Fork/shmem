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
- Supported Hardware Models
  - Atlas 800I A2/A3 series
  - Atlas 800T A2/A3 series
- Platform: AArch64/x86
- Software: driver firmware Ascend HDK 25.0.RC1.1, CANN 8.3.RC1, and later (For details about how to install the CANN toolkit and the matching firmware and driver, see the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netyum&OS=openEuler).
CMake 3.19 or later
GLIBC 2.28 or later

## Quick Start
 - Set CANN environment variable.<br>
    ```sh
    # Installation as the root user (default path)
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
 - Build the shared memory library.<br>
    Build the shared memory library and set its environment variables.
    ```sh
    cd shmem
    bash scripts/build.sh
    source install/set_env.sh
    ```
 - Use the .run file.<br>
    The software package name is `SHMEM_{version}_linux-{arch}.run`.<br>
    `{version}` indicates the software version, and `{arch}` indicates the CPU architecture.<br>
    Install the .run file (depending on the CANN environment).<br>

    ```sh
    chmod +x Software package name.run # Grant the execute permission on the software package.
    ./Software package name.run --check # Check the consistency and integrity of the software package installation file.
    ./Software package name.run --install # Install the software. You can use `--help` to query installation options.
    ```
    If `xxx install success!` is displayed, the installation is successful.

By default, TLS communication encryption is enabled for SHMEM. To disable it, call the following API:
```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```
For details, see "Security Statements".

Execute a sample operator to verify core functions. The following uses `allgather` as an example:
1. Build in the `shmem/` directory.

```sh
bash scripts/build.sh -examples
```

2. Run the demo in the `shmem/examples/allgather` directory.

```sh
bash run.sh -pes 2 -type int32_t
```
Note: The examples and other sample code are for reference only. Exercise caution when using them in the production environment.

The `run.sh` script provides parameters such as `-pes`, `-type`, `-ipport`, `-gnpus`, `-fnpu`, and `-fpe` to customize the number of devices, data type, IP address and port, number of NPU cards, first NPU card ID, and first PE ID for executing tests.

```sh
# Two devices and the int32_t data type
bash run.sh -pes 2 -type int32_t
```

For more detailed parameter descriptions, see `examples/allgather/README.md`. For the `-ranks` and `-test_filter` parameters of the unit test script `scripts/run.sh`, see [Function self-test case](#function-self-test-case).

## Function self-test case

 - Unit test of shared memory library APIs
Run the commands in the project directory.
```sh
bash scripts/build.sh -uttests
bash scripts/run.sh
```
The run.sh script provides parameters such as `-ranks` and `-test_filter` to specify the number of devices and gtest filters for executing tests.

Example:

```sh
# Running All *Init* Test Cases on 8 Devices
bash scripts/run.sh -ranks 8 -test_filter Init
```

## Python Test Cases     [Python API List](api/pythonAPI_en.md)

The `cann-shmem` Python wheel supports only Python 3.10 through 3.12 and depends on TorchNPU.
Install `torch-npu` matching the current CANN, Python, and CPU architecture before installing the
wheel. SHMEM documentation does not maintain TorchNPU installation commands or its version matrix.
See the
[TorchNPU compatibility matrix](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
for other supported combinations.

When installing `cann-shmem` from the Ascend private index, public PyPI can be configured as a
supplementary index for dependencies that are not mirrored by the private index:

```bash
python3 -m pip install cann-shmem \
  --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/ \
  --extra-index-url https://pypi.org/simple/
```

Pip combines candidates from both indexes and does not guarantee source priority. For production
use, explicitly install the compatible `torch-npu` version first, and then install `cann-shmem`
using only the Ascend private index.

1. In the repository root directory, build the Python extension and wheel package.

```sh
bash scripts/build.sh -python_extension
```

2. Load the environment variables into the current environment.

```sh
source install/set_env.sh
```

3. Install the wheel package generated in the `dist` directory.

```sh
python3 -m pip install --force-reinstall dist/cann_shmem-*.whl
```

4. Decide whether to enable TLS authentication. By default, TLS authentication is enabled. To disable TLS authentication, use the following API:

```python
import shmem as shm
shm.set_conf_store_tls(False, "")   # Disable TLS authentication
```

```python
import shmem as shm
tls_info = "xxx"
shm.set_conf_store_tls(True, tls_info)      # Enable TLS authentication
```

5. Run the test demo using Torchrun.

```sh
torchrun --nproc-per-node=k test.py // k indicates the rank size to be run.
```
If "test.py running success!" is printed in the log, the demo is running successfully.

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

- Python initialization example
```python
import shmem as ash

# xxx

uid = ash.aclshmem_get_unique_id()
ret = ash.aclshmem_init_using_unique_id(rank, world_size, mem_size, uid)

# xxx
```

After preparing the preceding startup code `init.py`, use `torchrun --nproc-per-node 8 init.py`. The number of processes can be changed as needed. For more examples, see the `unique_id_test.py` file.

- C++ initialization example
```cpp
aclshmemx_uniqueid_t uid;
aclshmemx_init_attr_t attributes;
int ret = aclshmemx_get_uniqueid(&uid);
ret = aclshmemx_set_attr_uniqueid_args(my_pe, n_pes, mem_size, &uid, &attributes);
status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);
```
