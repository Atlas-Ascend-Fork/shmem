# Organizational Structure of Code

## SHMEM Organizational Structure

```text
├── 3rdparty // Dependent third-party libraries
├── docs     // Documents
├── examples // Usage samples
├── include  // Header files
├── scripts  // Related scripts
├── src      // Source code
└── tests    // Test cases
```

## include

```text
Header files in the include directory are organized in the following file hierarchy:
include/
├── shmem.h                                 // Summary of all external SHMEM APIs
├── device/                                 // Device-side header files
│   ├── shmem_def.h                         // Public standard APIs defined on the device side
│   ├── ub2gm/                              // Device-side AI Core-driven ub2gm data plane low-level APIs, extended APIs + x
│   │   ├── shmem_device_rma.h              // AI Core ub2gm remote memory access (RMA) APIs
│   │   └── engine/                         // AI Core directly-driven ub2gm low-level APIs, extended APIs + x
│   │       └── shmem_device_mte.h          // AI Core directly-driven MTE APIs
│   ├── gm2gm/                              // Device-side AI Core-driven gm2gm data plane high-level and low-level APIs, extended APIs + x
│   │   ├── shmem_device_rma.h              // AI Core high-level RMA APIs
│   │   ├── shmem_device_amo.h              // AI Core high-level atomic memory operation APIs
│   │   ├── shmem_device_so.h               // AI Core high-level signal operation APIs
│   │   ├── shmem_device_cc.h               // AI Core high-level collective communication APIs
│   │   ├── shmem_device_p2p_sync.h         // AI Core high-level P2P synchronization APIs
│   │   ├── shmem_device_mo.h               // AI Core high-level memory ordering APIs
│   │   └── engine/                         // AI Core directly-driven gm2gm low-level APIs, extended APIs + x
│   │       ├── shmem_device_rdma.h         // AI Core directly-driven RDMA APIs
│   │       ├── shmem_device_sdma.h         // AI Core directly-driven SDMA APIs
│   │       └── shmem_device_mte.h          // AI Core directly-driven MTE APIs
│   └── team/                               // Device-side communicator management APIs
│       └── shmem_device_team.h             // Device-side communicator management APIs
├── host_device/                            // Directory shared by the device and host
│       └── shmem_common_types.h            // Common data structures, and so on
└── host/                                   // Host-side header files
    ├── shmem_host_def.h                    // APIs defined on the host side
    ├── init/                               // Host-side initialization APIs
    │   └── shmem_host_init.h               // Host initialization APIs
    ├── team/                               // Host-side communicator management APIs
    │   └── shmem_host_team.h               // Host communicator management APIs
    ├── mem/                                // Host-side memory management APIs
    │   └── shmem_host_heap.h               // Host memory management APIs
    ├── data_plane/                         // Host-side CPU-driven data plane APIs
    │   ├── shmem_host_rma.h                // Host high-level RMA APIs
    │   ├── shmem_host_so.h                 // Host high-level signal operation APIs
    │   ├── shmem_host_cc.h                 // Host high-level collective communication APIs
    │   └── shmem_host_p2p_sync.h           // Host high-level P2P synchronization APIs
    └── utils/                              // Host-side DFX APIs
        └── shmem_log.h                     // Host DFX-log APIs
```

## src

```text
└── src
    ├── device             // Device-side API implementation
    └── host
        ├─bootstrap       // Bootstrap API implementation
        ├─hybm            // Hybrid memory
        ├─init            // Host-side initialization API implementation
        ├─mem             // Host-side memory management API implementation
        ├─python_wrapper  // Python API encapsulation
        ├─sync            // Synchronization API implementation
        ├─team            // Host-side communicator management API implementation
        └─transport       // Link setup-related content
```

## examples

```text
└── examples
    ├── allgather                              // AllGather communication operator sample
    ├── cmo                                    // CMO cache performance sample
    ├── combine                                // MoE combine sample
    ├── dispatch                               // MoE dispatch sample
    ├── hccs_sio_link                          // HCCS/SIO link test sample
    ├── init                                   // SHMEM initialization flow sample
    ├── kv_shuffle                             // KV cache shuffle sample
    ├── multi_instance                         // Multi-instance sample
    ├── notifywait                             // SDMA notify/wait synchronization sample
    ├── one_multi_path                         // Ascend950 one path/multi path single-core sliced transfer sample
    ├── python_extension                       // Python extension and torch invocation sample
    ├── rdma_aclgraph_demo                     // Sample of invoking RDMA allgather in aclgraph
    ├── rdma_demo                              // RDMA implementation sample
    ├── rdma_handlewait_test                   // RDMA handle wait test sample
    ├── rdma_perftest_demo                     // RDMA performance test sample
    ├── rma_d2h_demo                           // RMA sample of the device accessing the host memory
    ├── sdma                                   // SDMA implementation sample
    ├── sdma_d2h_demo                          // SDMA sample of the device accessing the host memory
    ├── simt_rma                               // SIMT contiguous memory access sample
    ├── simt_rma_scalar                        // SIMT scalar access sample
    ├── torch_binding                          // PyTorch binding sample
    ├── udma_atomic_add                        // UDMA atomic add sample
    ├── udma_demo                              // UDMA implementation sample
    └── utils                                  // Common utilities for samples
```

## tests

```text
└── tests
    ├── examples
    └── unittest
        ├── init  // Unit test of initialization APIs
        ├── mem   // Unit test of memory management APIs
        ├── sync  // Unit test of synchronization management APIs
        └── team  // Unit test of communicator management APIs
```

## docs

```text
└── docs
    ├── api                          // SHMEM APIs
    ├── debug                        // SHMEM debugging
    ├── deployment                   // SO file deployment and usage guide
    ├── doxygen
    ├── example                      // Usage samples
    ├── images                       // Images
    ├── SECURITY.md                  // Security statement
    ├── code_organization.md         // Project organizational structure (this file)
    ├── compilation_build_guide.md   // Introduction to related scripts
    ├── conf.py                      // conf file
    ├── index.rst                    // Directory tree
    ├── multi_instance.md            // Multi-instance support
    ├── principles/                  // SHMEM principles overview (init/finalize, Config Store, team, and so on)
    └── quickstart.md                // Quick start
```

## scripts

Stores related scripts.
[Functions and usage of the scripts](compilation_build_guide_en.md)
