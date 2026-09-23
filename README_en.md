# <div align="center">SHMEM</div>
<div align="center">
English | <a href="./README.md">简体中文</a>
</div>
<h4><div align="center">Symmetric Memory-based Ascend Distributed Memory Communication Acceleration Library</div></h4>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-SHMEM-blue)](https://shmem-doc.pages.dev/)[![Release](https://img.shields.io/badge/Release-v1.6.0-brightgreen)](https://gitcode.com/cann/shmem/releases/v1.6.0)[![Platform](https://img.shields.io/badge/Platform-Ascend%20NPU-red)](https://www.hiascend.com/)[![SIG](https://img.shields.io/badge/SIG-shmem-lightgrey)](https://gitcode.com/cann/community/tree/master/CANN/sigs/shmem)

</div>

## What's New
🚀 [July 2026] [SHMEM v1.6.0](https://gitcode.com/cann/shmem/releases/v1.6.0) is now available.
   - Expanded Ascend 950 RDMA support with XSCALE and HNS 1825 backends, plus Device-side RMA, Atomic, and Sync/Barrier APIs.
   - Enhanced Ascend 950 UDMA/MTE and SIMT data-movement capabilities, adding relay RMA, MTE Atomic, and UB↔GM APIs.
   - Improved installation and diagnostics: multi-platform backends integrated into the Python wheel, `shmem-config` environment checks, and additional performance tests and examples.

> Before upgrading, read the [v1.6.0 release notes](https://gitcode.com/cann/shmem/releases/v1.6.0), especially the changes to nonblocking completion semantics, ROCE `atomic_add` behavior, and version consistency requirements.

📌 [April 2026] [SHMEM v1.3.0](https://gitcode.com/cann/shmem/releases/v1.3.0): added AI Core direct-driven capabilities and 40+ communication APIs, with expanded DFX tooling.

🔥 [December 2025] Initial launch of the SHMEM project.

## ⚡️ Quick Start

### Installation

SHMEM provides three installation methods. Select one as required:

**Method 1: pip installation**

The Python wheel supports only Python 3.10 through 3.12 and depends on TorchNPU (pip package name: `torch-npu`; Python import name: `torch_npu`).
Before installation, prepare a `torch-npu` version that matches the current CANN, Python, and system architecture according to the
[TorchNPU version compatibility](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md).

```bash
python3 -m pip install cann-shmem \
  --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/
shmem-config --version   # Query the installed version
shmem-config --diagnose  # Check native library loading and package integrity
```

> Note:
> - If you want pip to fall back to the public PyPI when dependencies are missing from the Ascend private index, add
>   `--extra-index-url https://pypi.org/simple/` when installing `cann-shmem`. Pip merges candidates from both indexes
>   and does not guarantee that a same-named package comes from `--index-url`. For production environments, install the
>   matching `torch-npu` version first as described in the official TorchNPU compatibility documentation.
> - The runtime environment requires glibc 2.28 or later. Otherwise, `libshmem.so` may fail to load due to missing
>   symbols. Run `ldd --version` to check the local glibc version.

**Method 2: Binary package installation**

```bash
# Obtain the software package SHMEM_{version}_linux-{arch}.run (see the release page or build it locally)
chmod +x package_name.run
./package_name.run --install
source /usr/local/Ascend/shmem/latest/set_env.sh
```

**Method 3: Build from source**

```bash
git clone https://gitcode.com/cann/shmem.git
cd shmem
bash scripts/build.sh             # A2/A3 platforms
# bash scripts/build.sh -soc_type Ascend950  # Ascend 950 platform
# Optional: limit build parallelism via the standard CMake environment variable
# CMAKE_BUILD_PARALLEL_LEVEL=16 bash scripts/build.sh
source install/set_env.sh
```

By default, the build reuses the CMake cache and compiled objects in the repository-root `build/` directory. When switching toolchains, clearing stale configuration, or performing a full rebuild, use `bash scripts/build.sh -clean`.

> For complete installation steps (including CANN environment preparation, dependency descriptions, Docker containers, build execution, and local verification), see the [Quick Start document](docs/quickstart_en.md).

After installing the Python wheel, you can use the [shmem-config command reference](docs/tools/shmem_config_guide.md) to query backends and installation paths, and run environment checks and diagnostics.

For abbreviations and terms used in the documentation and APIs, see the [Glossary](docs/glossary.md).

## 1. Project Introduction
SHMEM is a multi-server, multi-device memory communication library designed for the Ascend platform. By abstracting host-side and device-side APIs, SHMEM enables fast cross-device memory access and data synchronization. Its core benefits include:
- Support for AI Core direct-driven MTE and xDMA, enabling D2D, D2H, H2D, D2rH, and rH2D communication paths
- Simplified inter-device communication logic in distributed workloads, lowering operator development complexity
- Deep integration with the CANN ecosystem, accelerating deployment of computing-communication fusion operators
- For more information, see [SHMEM](https://shmem-doc.pages.dev/).

## 2. Core Functions
![Core Functions](docs/images/readme-features_en.png)

**1. Dual-side API system**

- Host side: initialization, memory heap management, communicator (team) creation, and global synchronization
- Device side: remote memory access (RMA), device-level synchronization, and team operations

The API design follows the Ascend operator development paradigm and supports host-device collaboration.

**2. High-performance communication optimization**

- Built-in MTE and xDMA engines for direct remote memory read/write, minimizing latency
- MPI interoperability, supporting collective communication primitives such as AllGather and AllReduce
- Optimized data transfer paths for Ascend hardware features to improve multi-device collaboration efficiency

**3. Secure communication mechanism**

- TLS encryption is enabled by default to protect cross-device data transfer and can be disabled for specific APIs:

   ```c
   int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
   ```

- Enterprise-grade security guidelines are provided, covering permission configurations and cipher suite selection.

**4. Communication path coverage**

The following figure shows the full-link communication paths supported by SHMEM (using Ascend 910A3 as an example), covering different transmission engines on the host and device sides:

<img src="docs/images/dma_en.png" width="800"/>

As shown in the figure, SHMEM supports diverse communication engines and paths:

- **MTE Engine**: chip-level memory transfer engine, supporting D2D, D2H, H2D, D2rH, and rH2D paths
- **xDMA Engine**: high-speed direct memory access engine, supporting efficient intra-host and inter-host data transfer

**5. Multi-language and extensibility support**

- Native C++ APIs and Python bindings for different development scenarios
- Modular backend design enabling dynamic switching between MTE and xDMA for easy extension

**6. Rich example scenarios**
Covering scenarios from basic communication to complex operator fusion:

- rdma_demo: RDMA communication demonstration
- [CATCCOS](https://gitcode.com/cann/catccos): computing-communication fusion operator library covering common operators such as Matmul and ReduceScatter

## 3. Code Structure

```text
shmem/                                  # Project root directory
├── docs/                               # Documentation and description
├── examples/                           # A collection of examples
├── include/                            # External header files
│   ├── shmem.h                         # All SHMEM external APIs
│   ├── device/                         # Device-side header files
│   │   ├── gm2gm/                      # Data plane APIs gm2gm driven by AI Core
│   │   │   └── engine/                 # Low-level APIs gm2gm directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                      # Data plane APIs ub2gm driven by AI Core
│   │       └── engine/                 # Low-level APIs ub2gm directly driven by AI Core
│   ├── host/                           # Host-side header files
│   │   ├── data_plane/                 # Host-side data plane APIs
│   │   ├── init/                       # Host-side initialization APIs
│   │   ├── mem/                        # Host-side memory management APIs
│   │   ├── team/                       # Host-side team management APIs
│   │   └── utils/                      # Tools and general auxiliary code
│   └── host_device/                    # Shared directory
├── scripts/                            # Sample scripts (build/run)
├── src/                                # Source code implementation
│   ├── device/                         # Device-side implementation
│   │   ├── gm2gm/                      # Data plane APIs gm2gm directly driven by AI Core
│   │   │   └── engine/                 # Low-level APIs gm2gm directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                      # Data plane APIs ub2gm driven by AI Core
│   │       └── mte/                    # Low-level APIs ub2gm directly driven by AI Core
│   ├── host/                           # Host-side implementation
│   │   ├── bootstrap/                  # bootstrap
│   │   ├── data_plane/                 # Host-side data plane API implementation
│   │   ├── entity/                     # Memory entity (Hybrid Memory) management
│   │   ├── init/                       # Initialization
│   │   ├── mem/                        # Memory management
│   │   ├── python_wrapper/             # Python encapsulation/bindings
│   │   ├── sync/                       # Synchronization primitives (barrier/p2p/order)
│   │   ├── team/                       # Team (communicator)
│   │   ├── transport/                  # Transport layer implementation (such as RDMA, SDMA, and UDMA)
│   │   └── utils/                      # Tools and general auxiliary code
│   ├── host_device/                    # Shared directory
│   └── python/                         # Python-related directory
├── tools/                              # Project auxiliary tools
│   ├── pre-commit/                     # pre-commit check tools (header file dependency check, spelling check, and other configurations)
│   └── rootinfo/                       # root_info_generate tool source code
└── tests/                              # Test case set (UTs/functional tests)
```

## 4. Typical Use Cases

**1. Computing-communication fusion operator development**: Develop custom operators that merge compute and communication (such as Matmul + AllReduce) by leveraging device-side direct memory access APIs. This reduces inter-device data copies and improves operator execution efficiency.

**2. Multi-server, multi-device data synchronization**: Use host-side team management APIs to quickly establish shared memory channels for a multi-server, multi-device cluster, enabling cross-node data synchronization for distributed training workloads.

**3. Low-latency inter-device communication**: Employ RDMA-optimized device-side APIs to transfer data among devices in milliseconds, meeting the real-time requirements of latency-sensitive AI inference scenarios.

**4. Python distributed training adaptation**: Integrate SHMEM into PyTorch distributed workflows via Python extension APIs, replacing traditional MPI communication to reduce training communication overhead.

## 5. FAQs

**Q1: What do I do if "CANN environment not found" is reported during a build?**

A: Ensure that the CANN toolkit has been installed and `source /usr/local/Ascend/ascend-toolkit/set_env.sh` has been executed. By default, the project is built with the `bisheng` compiler provided by the CANN toolkit. The CANN version must meet the requirements in [CANN Version Description](docs/quickstart_en.md#software-and-hardware-versions).

**Q2: What do I do if "inter-device communication timeout" is reported when running an example?**

A: Check whether the RDMA NIC is available, whether the network between nodes is connected, whether the firewall allows the initialization communication port (8666 by default), whether the lossless network configuration on the switch is correct, and whether the clocks of all nodes are synchronized. The RDMA link setup listening port is automatically assigned by the system and does not need to be configured manually. For RDMA port usage rules, see [Troubleshooting](docs/debug/Troubleshooting_FAQs_en.md).

**Q3: What do I do if "module not found" is reported when importing shmem in Python?**

A: Ensure that the wheel package has been installed, the `set_env.sh` file in the install directory has been sourced, and the environment variable `PYTHONPATH` contains the shmem path.

**Q4: Why is an encryption failure still reported after TLS is disabled?**

A: Call `aclshmemx_set_conf_store_tls` before `aclshmemx_init_attr`. The TLS configuration cannot be modified after initialization.

**Q5: What do I do if a Git failure is reported for dependencies such as googletest and nlohmann/json when `build.sh` is executed?**

A: Check whether the Git configuration can access GitCode. `googletest v1.14.x` is used for UT builds and `nlohmann/json v3.11.3` is used for Ascend 950 platform builds. They are automatically downloaded by `scripts/build.sh` by default. In an offline environment, prepare them in `3rdparty/googletest` and `3rdparty/json` in advance. For details, see [Third-party Source Code Dependencies](docs/quickstart_en.md).

**Q6: What do I do if the CANN package fails to be installed?**

A: Go to the [Ascend Community](https://www.hiascend.com/search/result?keyword=&tab=1) and search in the box at the upper right corner or ask a question online.

**Q7: There is no Dockerfile in the repository root directory. How do I prepare a container environment?**

A: This repository currently does not maintain a standalone Dockerfile. You are advised to select a CANN container image from the [Ascend image repository](https://www.hiascend.com/developer/ascendhub) based on the chip, system architecture, and CANN version, and then complete the source build by following the steps in [Docker Container Environment](docs/quickstart_en.md) and the Quick Start document.

**Q8: Can UTs and examples be run without an NPU environment?**

A: Dependency checks and build verification can be performed, but `scripts/run.sh` and `scripts/run_examples.sh` require available NPUs, drivers, and the CANN runtime. Running results must be verified in a hardware environment or in project CI.

> For more troubleshooting information, see [Troubleshooting](docs/debug/Troubleshooting_FAQs_en.md).

## 6. Contributions
### Contributors
- [Professor Lu Lu's team, South China University of Technology](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### Participation Guide
Subscribe to the [SHMEM SIG meetings](https://mailweb.cann.osinfra.cn/mailman3/lists/shmem.cann.osinfra.cn/) to join regular community meetings and discussions, and exchange ideas on solution design, API planning, and usage with community members.

**1. Submitting issues**

- Bug reports: specify the environment (hardware/software versions), reproduction steps, and error logs.
- Feature requests: describe the use case, expected outcome, and supported hardware/software versions.

**2. Submitting a PR**

- Branch naming convention: use `feature/xxx` for feature development and `bugfix/xxx` for bug fixes.
- Coding standards: follow the project coding guidelines. New code must include unit tests.
- PR description: explain the purpose of the changes, core logic, and test validation results.

**3. Code review**

- PRs must pass CI automated checks (build, unit tests, and code style checks).
- At least one maintainer's approval is required before merging.

For detailed steps, see [Contribution Guide](CONTRIBUTING_en.md).

## 7. Security Statement
- Communication security: TLS encryption is enabled by default, and custom cipher suites are supported.
- Public network dependencies: for the open-source repositories and tool addresses, see [Public Network Address List](SECURITY_en.md#public-network-address-statement).
- Security hardening guide: configure system permissions and firewalls by referring to [Security Hardening Suggestions](SECURITY_en.md#security-hardening).

## 8. Copyright and License
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This project is licensed under CANN Open Software License Agreement Version 2.0 and can be used only for development related to Ascend processors.

## 9. Precautions
1. This project supports only the Ascend platform and does not support other hardware architectures (such as x86 general-purpose servers and NVIDIA GPUs).
2. The example code is for learning and reference only. Test its functionality and performance thoroughly before using it in a production environment.
3. Upgrading the CANN version may cause API compatibility issues. You are advised to use the CANN version specified in the documentation.
4. After TLS encryption is disabled, ensure that the communication network is a trusted intranet to prevent data leakage.
