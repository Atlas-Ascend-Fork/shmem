# SHMEM Glossary

This document is intended for developers who use SHMEM. It compiles common abbreviations and terms that appear in the README, docs, include, and examples.

Notes:

- In this project, `SHMEM` mainly focuses on the "symmetric memory" communication model. Each PE allocates remotely accessible symmetric data objects or symmetric heap memory according to the same rules. When "shared memory" appears in the documentation, it usually refers to a memory region managed by SHMEM that can be accessed across PEs. It may also refer to the communication capability built on that region.

- For descriptions of Huawei/Ascend/CANN-related terms other than terms specific to the `SHMEM` project, refer to the official websites of [Ascend](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html), [CANN](https://www.hiascend.com/cann), and [Unified Bus UB](https://www.unifiedbus.com/zh).

## Programming Model and APIs

| Term | English Expansion | Description | Project Context |
| --- | --- | --- | --- |
| `SHMEM` | Symmetric Hierarchical Memory / Shared Memory | Symmetric memory/shared memory communication library | Name of this repository. It denotes a multi-node, multi-device memory communication library for the Ascend platform. |
| `OpenSHMEM` | OpenSHMEM Application Programming Interface | OpenSHMEM application programming interface specification | API specification that standardizes the SHMEM programming model. It provides concepts such as PE, symmetric data objects, RMA, AMO, Signal, and collective communication. |
| `Host API` | Host-side API | Host-side interface | Interface called on the CPU/Host side. It is responsible for initialization, memory management, communicator management, synchronization, and task dispatching. |
| `Device API` | Device-side API | Device-side interface | Interface called in the AICore/kernel context. It is responsible for remote memory access, synchronization, and communicator operations. |
| `PE` | Processing Element | Processing element | Number of a communication participant in SHMEM. It usually corresponds to a process, a rank, or a device participant. |
| `Team` | Team | Communicator/group | A communicator consisting of a group of PEs. It is used to limit the scope of synchronization and communication operations. `ACLSHMEM_TEAM_WORLD` is the default global communicator after initialization. |
| `Symmetric Memory` | Symmetric Memory | Symmetric memory | A memory region allocated by each PE according to the same rules and remotely accessible through SHMEM APIs. |
| `Symmetric Heap` | Symmetric Heap | Symmetric heap | A symmetric memory heap managed by `aclshmem_malloc` / `aclshmem_free`. |

## Communication Semantics and Synchronization

| Term | English Expansion | Description | Project Context |
| --- | --- | --- | --- |
| `RMA` | Remote Memory Access | Remote memory access | A PE actively accesses the symmetric memory of another PE. It includes one-sided access operations such as put/get. |
| `AMO` | Atomic Memory Operation | Atomic memory operation | It performs atomic semantic operations such as atomic add, fetch, swap, and compare-swap on remote or local symmetric memory. |
| `SO` | Signal Operation | Signal variable/signal operation | SHMEM signal variable or signal operation. It is used in synchronization scenarios such as put-with-signal and signal wait/op. |
| `P2P Sync` | Point-to-Point Synchronization | Point-to-point synchronization | Semantics of point-to-point wait, test, and synchronization APIs. |
| `CC` | Collective Communication | Collective communication | A communication operation that multiple PEs perform together, for example Barrier, Allgather, and Allreduce. |
| `MO` | Memory Ordering | Memory ordering | Synchronization mechanism. It ensures the execution order of memory reads/writes across processing elements and of blocking/non-blocking API operations. It also ensures that the operation completion status is reliably delivered. |
| `put` | Put operation | Remote write operation | The local PE actively writes data to the symmetric address of the remote PE. |
| `get` | Get operation | Remote read operation | The local PE actively reads data from the symmetric address of the remote PE. |
| `NBI` | Non-Blocking Immediate | Non-blocking immediate operation | Suffix of `*_nbi` APIs. It indicates a non-blocking data operation: the call returns immediately and does not wait for the operation to complete. It must be used together with memory ordering barriers such as `quiet()` or `fence()` to ensure reliable delivery of the operation completion status and the global order. |
| `Blocking` | Blocking | Blocking operation | Indicates a blocking data operation: the call blocks the current thread, and the operation completion status is guaranteed to be reliably delivered when the call returns. |
| `Barrier` | Barrier Synchronization | Barrier synchronization | Collective synchronization within a Team. Participating PEs continue to execute only after they reach the synchronization point. The API semantics ensure that related preceding operations are completed or visible. |
| `Sync` | Synchronization | Synchronization | Similar to Barrier, but `Sync` guarantees only the completion and visibility of preceding memory stores. It does not guarantee that all SHMEM remote updates are completed. |
| `Quiet` | Quiet | Completion wait | It waits for the completion of operations on symmetric data objects previously issued by the calling PE. |
| `Fence` | Fence | Ordering fence | In OpenSHMEM semantics, it is used to constrain the delivery order of Put, AMO, and store operations. The current implementation of this project is close to the behavior of `quiet`, and guarantees both ordering and completion. |

## Communication Engines, Protocols, and Data Paths

| Term | English Expansion | Description | Project Context |
| --- | --- | --- | --- |
| `MTE` | Memory Transfer Engine | Memory transfer engine | Memory transfer engine on the Ascend AICore side. It is used for data paths such as D2D, D2H, H2D, D2rH, and rH2D. |
| `DMA` | Direct Memory Access | Direct memory access | A memory access mechanism that does not move data byte by byte through general computing units. |
| `xDMA` | Direct Memory Access family | Collective term for DMA-class engines | A collective term for high-speed DMA-class transfer engines. |
| `SDMA` | System Direct Memory Access | System direct memory access | Data transfer engine. It serves as one of the underlying transfer backends of RMA/AMO. |
| `UDMA` | Unified Direct Memory Access | Unified direct memory access | Data transfer engine. It serves as one of the underlying transfer backends of RMA/AMO. |
| `RDMA` | Remote Direct Memory Access | Remote direct memory access | Data transfer engine. It allows direct access to remote memory through the network or device interconnect, and serves as one of the underlying transfer backends of RMA/AMO. |
| `HCCS` | Huawei Cache Coherent System | Huawei cache coherent system | High-speed interconnect bus between CPUs and NPUs inside an Ascend server. |
| `UB` protocol | Unified Bus | Unified Bus | New-generation Huawei interconnect protocol. It aims to unify all interconnects inside a data center and provide low-latency, high-bandwidth communication. |
| `UBMEM` / `UBmem` | UBMEM | Memory semantics | New-generation Huawei interconnect protocol. It provides communication with memory semantics. |
| `URMA` | Unified Remote Memory Access | Unified remote memory access | New-generation Huawei interconnect protocol. Based on the UB bus, it provides remote memory operations with semantics similar to local access. |
| `PCIe` | Peripheral Component Interconnect Express | High-speed peripheral interconnect bus | Connection standard between the CPU and other peripherals such as NPUs. It is also used for cross-group interconnect in some topologies. |
| `SIO` | **S**erial **I**/**O** Link | SIO link/path | High-speed interconnect path between two physical dies (bare chips) inside an Ascend A3 processor. |
| `RoCE` / `ROCE` | RDMA over Converged Ethernet | RDMA over converged Ethernet | Network protocol used by the RDMA engine. It is commonly used for cross-node communication. |
| `D2D` | Device to Device | Device to device | Data access or transfer between the memory of the local NPU device and that of the peer NPU device. |
| `D2H` | Device to Host | Device to host | Data path from the memory of the local NPU device to the memory of the local host. |
| `H2D` | Host to Device | Host to device | Data path from the memory of the local host to the memory of the local NPU device. |
| `D2rH` | Device to remote Host | Device to remote host | Data path from the memory of the local NPU device to the host memory where the remote PE resides. |
| `rH2D` | remote Host to Device | Remote host to device | Data path from the host memory where the remote PE resides to the memory of the local NPU device. |
| `gm2gm` / `GM2GM` | Global Memory to Global Memory | Global memory to global memory | Device-side AICore-driven GM-to-GM data plane interface/path. It is commonly used for cross-PE symmetric memory access. |
| `ub2gm` / `UB2GM` | Unified Buffer to Global Memory | Unified buffer to global memory | Data plane interface/path from the local UB of the AICore to the GM of the remote device. |
| `CMO` | Cache Maintenance Operation | Cache maintenance operation | Cache maintenance API exposed on the SDMA side. It is used to explicitly control cache prefetch/maintenance operations such as GM2L2. |

## Memory, Hardware, and Execution Units

| Term | English Expansion | Description | Project Context |
| --- | --- | --- | --- |
| `Host` | Host | Host side | CPU-side control plane. It is responsible for initialization, memory management, bootstrap, team/sync management, and host API dispatching. |
| `Device` | Device | Device side | NPU/AICore data plane. It executes RMA, synchronization, and communication kernels. |
| `NPU` | Neural Processing Unit | Neural processing unit | Ascend AI processor device. |
| `AICore` / `AI Core` | AICore | Ascend AICore | Core on the device side that executes custom operators and communication data plane code. |
| `AIV` | AIV / Vector Core | Ascend vector core | Official name of the Vector Core in the separated architecture. It performs vector computation and low-latency communication. |
| `AIC` | AIC / Cube Core | Ascend cube core | Official name of the Cube Core in the separated architecture. It is responsible for matrix computation. |
| `VEC` | Vector Core | Vector core | Synonymous with VECTOR_CORE. It refers to the vector computing unit. |
| `CUBE` | Cube Core | Cube core | Synonymous with Cube Unit. It is dedicated to high-computing-power matrix multiply-add operations. |
| `GM` | Global Memory | Global memory | Globally addressable memory on the device side. It is usually identified by `__gm__` in APIs. |
| `UB` | Unified Buffer | Unified buffer | On-chip temporary buffer of the AICore. It is usually identified by `__ubuf__` or `LocalTensor` in APIs. |
| `HBM` | High Bandwidth Memory | High bandwidth memory | High-bandwidth device memory on the NPU side. |
| `DRAM` | Dynamic Random Access Memory | Dynamic random access memory | Memory on the host side. |
| `L2` | Level-2 Cache | Level-2 cache | Cache level that appears in CMO, cache prefetch, and samples. |

## Platforms, Environment Variables, and Tools

| Term | English Expansion | Description | Project Context |
| --- | --- | --- | --- |
| `CANN` | Compute Architecture for Neural Networks | Ascend heterogeneous computing architecture software stack | SHMEM depends on CANN for compilation, running, and device APIs. |
| `ACL` / `AscendCL` | Ascend Computing Language | Ascend computing language/runtime API | Dependency of host-side runtime APIs such as initialization, memory, stream, and Graph. |
| `MPI` | Message Passing Interface | Message passing interface | Can serve as one of the backends of the bootstrap control channel. It is used for allgather/barrier between ranks. |
| `TLS` | Transport Layer Security | Transport layer security protocol | Communication encryption mechanism enabled by default in SHMEM. |
| `SSL` | Secure Sockets Layer | Secure sockets layer | Often related to TLS in dependencies and tests for secure connections. |
| `UID` | Unique ID | Unique identifier | Communication identifier for initialization/bootstrap and naming of related environment variables. |
| `IP` / `IPv4` / `IPv6` | Internet Protocol / Internet Protocol version 4 / Internet Protocol version 6 | Internet protocol/IPv4/IPv6 | Used in bootstrap addresses, network interface protocols, and network configurations. |
| `HCCL` | Huawei Collective Communication Library | Huawei collective communication library | The RDMA environment variables `HCCL_RDMA_TC` and `HCCL_RDMA_SL` reuse the HCCL configuration. |
| `TC` | Traffic Class | Traffic class | QoS parameter of the RDMA network. It corresponds to `HCCL_RDMA_TC`. |
| `SL` | Service Level | Service level | Service level parameter of the RDMA network. It corresponds to `HCCL_RDMA_SL`. |
| `PFC` | Priority Flow Control | Priority flow control | Related to SL/network configuration in RDMA scenarios. |
| `SOC_TYPE` | System-on-Chip Type | System-on-chip type | Parameter for selecting a hardware model in build scripts and tool documentation. |
| `ASCEND_HOME_PATH` | Ascend home path | Ascend installation path | Environment variable for the CANN/Ascend installation path. |
| `ASCEND_TOOLKIT_HOME` | Ascend Toolkit home | Ascend Toolkit installation path | Environment variable for the Toolkit installation path. |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend Runtime Visible Devices | Ascend runtime visible devices | Environment variable that controls the devices visible to the current process. |
| `LD_LIBRARY_PATH` | Library load path | Dynamic library search path | Configures the SHMEM/CANN dynamic library search path when you run samples and scripts. |
| `SHMEM_UID_SESSION_ID` | SHMEM Unique ID Session ID | SHMEM UID session identifier | Specifies the IP address and port on which PE 0 listens for bootstrap. |
| `SHMEM_UID_SOCK_IFNAME` | SHMEM Unique ID Socket Interface Name | SHMEM UID network interface name | Specifies the network interface name and network layer protocol used by PE 0 for bootstrap. |
| `SHMEM_INSTANCE_PORT_RANGE` | SHMEM Instance Port Range | Multi-instance port range | Specifies the available port range for the bootstrap of each instance in multi-instance scenarios. |
| `SHMEM_CYCLE_PROF_PE` | SHMEM Cycle Profiling PE | SHMEM cycle profiling PE | Specifies the PE for profiling collection. |
| `SHMEM_LOG_LEVEL` | SHMEM Log Level | SHMEM log level | Controls the SHMEM log output level. |
| `ENABLE_ASCENDC_DUMP` | Enable AscendC Dump | Enable AscendC Dump | Compilation-time switch used to enable the AscendC dump debugging logic. |

## Samples, Debugging, and Performance Terms

| Term | English Expansion | Description | Project Context |
| --- | --- | --- | --- |
| [`SIMT`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10064.html) | Single Instruction Multiple Threads | Single instruction multiple threads | Instruction/execution mode in the samples. |
| [`SIMD`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0015.html) | Single Instruction Multiple Data | Single instruction multiple data | Instruction/execution mode in the samples. |
| [`VF_TYPE`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10052.html) | Vector Function Type | Vector function type parameter | Mixed programming mode of SIMT and SIMD. The entire execution process uses the VF as the basic scheduling unit. |
| `DFX` | Design for X | Engineering capabilities such as diagnosability and maintainability | Collective term for capabilities such as log, sanitizer, profiling, and debug. |
| [`mssanitizer`](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/install_guide/mssanitizer_install_guide.md) | MindStudio Sanitizer | MindStudio memory detection tool | Memory detection tool. |
| [`MSTX`](https://gitcode.com/Ascend/mstx/blob/master/docs/zh/install_guide/mstx_install_guide.md) | MindStudio Tools Extension | MindStudio tools extension (performance instrumentation API) | Appears in `mstx` tool installation and mssanitizer-related procedures. It is a CANN performance analysis API that supports custom instrumentation to locate performance bottlenecks. |
| `PROF` / `Profiling` | Profiling | Performance collection/instrumentation | Performance collection related terms such as cycle profiling and perftest. |
| `DUMP` | Dump | Dump of data/runtime information | AscendC dump debugging capability. |
| `PERF` / `perftest` | Performance Test | Performance test | Performance test samples, scripts, and output metrics. |
| `throughput` | Throughput | Throughput | Output metric of performance tests. |
| `bandwidth` | Bandwidth | Bandwidth | Output metric of performance tests. |
| `latency` | Latency | Latency | Output metric of performance tests. |
| `KB` / `MB` / `GB` | Kilobyte / Megabyte / Gigabyte | Kilobyte/megabyte/gigabyte | Capacity units in documentation, sample parameters, and performance outputs. |
| `Low-level API` | Low-level API (engine-specific API) | API that directly exposes the communication semantics of underlying transfer engines (UDMA, RDMA, SDMA, MTE, and so on) | Engine-specific APIs in SHMEM that reside in the `engine/` directory (for example, `aclshmemx_udma_*` and `aclshmemx_roce_*`). Callers need to manage buffer configuration and synchronization by themselves. Concurrent operations are supported. In code, they are often referred to as lowlevel. |
| `High-level API` | High-level API | High-level abstraction API that hides underlying engine details and provides a unified programming model | Encapsulated APIs in SHMEM that reside in `gm2gm/` or `host/data_plane/` (for example, `aclshmem_put_*` and `aclshmem_amo_*`). They automatically manage buffers and synchronization internally. The default buffer does not support concurrency. UDMA/RDMA/SDMA high-level APIs need to pre-configure engine parameters through `aclshmemx_*_config`. In code, they are often referred to as highlevel. |
