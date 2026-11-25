# Multi-instance Support

[简体中文](../../zh/principles/multi_instance.md)

## Overview

The SHMEM multi-instance capability allows **multiple independent SHMEM instances to be created within a single process**. Each instance has its own communicator, memory space, and synchronization resources. The capability applies to the following scenarios:

- **Coexistence of multiple communicators**: Multiple task domains in a single process require independent SHMEM contexts.
- **Resource isolation**: SHMEM resources are isolated between different tasks/modules to avoid mutual interference.
- **Elastic scaling**: Creating or releasing one instance does not affect other instances.
- **Dynamic instance management**: Instances are created or released by `instance_id` at runtime.

The communication links support MTE and UDMA.

## Core Concepts

### Instance

An SHMEM instance contains a complete set of initialization states, typically including:

- Independent communicator (`Team World`).
- Independent shared memory heap (`Heap`).
- Independent synchronization resource (`Sync Pool/Counter`).
- Independent runtime snapshots of State, Bootstrap, and MemoryManager.

### Instance Identifier (Instance ID)

The current repository uses `instance_id` (`uint64_t`) to identify instances. The instance lifecycle is managed by the initialization attributes and the `finalize` API.

### Context Switching

The Host completes instance switching through the **global variable swap mechanism**: it first writes back the global state of the current instance, and then loads the global state of the target instance. Both the Host and Device use `aclshmemx_instance_ctx_set`, but each selects the context of its own execution domain. UDMA submission and matching completion wait on the Device side must use the same instance; complete the outstanding operations of the old instance before switching. The Host can switch instances after a quiet operation is enqueued and its instance is captured; the old instance, streams, and buffers must be preserved until completion, and the cross-stream dependencies and Host multi-thread serialization requirements must be satisfied.

## Architecture Design

```text
┌──────────────────────────────────────────────────────────┐
│                          Process                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │                    SHMEM Runtime                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│  │  │  Instance 0 │  │  Instance 1 │  │ Instance N │  │  │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌──────┐  │  │  │
│  │  │  │ Team  │  │  │  │ Team  │  │  │  │ Team │  │  │  │
│  │  │  │ Heap  │  │  │  │ Heap  │  │  │  │ Heap │  │  │  │
│  │  │  │ Sync  │  │  │  │ Sync  │  │  │  │ Sync │  │  │  │
│  │  │  │ State │  │  │  │ State │  │  │  │ State│  │  │  │
│  │  │  └───────┘  │  │  └───────┘  │  │  └──────┘  │  │  │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Design Point | Current Implementation | Description |
|---|---|---|
| Instance isolation | Independent `aclshmem_context` snapshot | Each instance stores independent `state`, `host_state`, `boot_handle`, `memory_manager`, and so on |
| Context management | Global variable swap | Switching through `aclshmemx_instance_ctx_set` |
| Instance identifier | `instance_id` (`uint64_t`) | Located in `aclshmemx_init_attr_t` and used by `aclshmemx_finalize(instance_id)` |
| Device-side route | `instance_id << 1` | The kernel accesses the `state` of the corresponding instance through the hard-coded address |

## Specifications and Restrictions (Code Implementation)

| Item | Specification |
|---|---|
| Type of `instance_id` | `uint64_t` |
| Default instance | `instance_id = 0` |
| Instance ID slots | 0 to 255 for HBM-only; 0 to 254 when a Host DRAM companion entity is required, still subject to the port range and resource restrictions |
| Upper limit of PEs | `ACLSHMEM_MAX_PES = 16384` |
| Upper limit of local memory | `ACLSHMEM_MAX_LOCAL_SIZE = 40GB` |
| Upper limit of instance shared memory | Currently 128GB. Virtual address conflicts occur below the layer when the size exceeds 128GB |
| Team behavior of non-0 instances | Only the WORLD slot is retained during initialization. Host team management APIs such as team split/translate/get_config return `ACLSHMEM_NOT_SUPPORTED`; SIMT supports WORLD queries and PE mapping within WORLD |
| Default multi-instance port allocation | Depends on the environment variable `SHMEM_INSTANCE_PORT_RANGE=start:end` (start and end are specified by users; the actual port used by each instance is `start_port + instance_id`) |
| Communication engine support | SIMD calls support MTE and UDMA; SIMT calls support MTE only |
| UDMA QP configuration | Set the process-level QP count with `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num)` before the first initialization. It cannot be modified while an active instance exists. Queue resources, metadata, and completion states are isolated by instance |
| Host UDMA quiet | `aclshmemx_quiet_on_stream(stream)`: the current instance is captured at enqueue. A non-zero instance waits for all QPs (including shared Jetty); instance 0 keeps the original QP0 completion scope, and non-zero QPs use `aclshmemx_udma_qp_quiet` |
| UDMA instance ID range | 0 to 255 for HBM-only initialization; at most 254 when a Host DRAM companion entity is created with `USE_ACLRT_MEM_FABRIC_HANDLE`. Initialization beyond the HBM upper limit returns `ACLSHMEM_INVALID_VALUE`; quiet accepts only instances that have been successfully initialized and are still alive, and rejects IDs beyond 255 before enqueue |

## Environment Variables and Running Requirements

### Kernel-side Usage Requirements

- SIMD calls (including UDMA): add `MULTI_INSTANCE` at compilation. Each participating core calls `aclshmemx_instance_ctx_set(id)` before operations. The ID must correspond to an instance that has been successfully initialized and is still alive. All builds check the ID range before calculating the context address, and trap immediately if the ID is out of range; error logs are output only in Debug mode. The caller must still ensure that an in-range ID has been initialized and is alive; the context consistency check is only a Debug diagnosis. Complete the outstanding operations of the old instance before switching instances. The UDMA batch submission and completion wait must use the same instance.
- A UDMA producer kernel must complete local staging before returning and submit pending batches through public submission APIs; the library publishes the internal queue state. The producer and quiet must be executed in order on the same stream, or an explicit cross-stream event dependency must be established. Do not submit or quiet the same queue on other streams at the same time.
- When non-zero QPs are explicitly used, use `aclshmemx_udma_qp_quiet(peer, qp)` to wait for completion.
- Keep the corresponding instance, stream, and memory valid until communication completes.
- SIMT calls: define `MULTI_INSTANCE` for multi-instance kernels. All threads in each VF consistently call `simt::aclshmemx_instance_ctx_set(id)` before calling the existing APIs; selection is independent of the Host/SIMD.

### Default Multi-instance Mode

- `SHMEM_INSTANCE_PORT_RANGE` must be set in the format of `start:end`.
- The `ip_port` port in the initialization attributes must be `0`, for example, `tcp://x.x.x.x:0`.

### Unique ID Mode

- The existing `SHMEM_UID_SESSION_ID` / `SHMEM_UID_SOCK_IFNAME` rules still apply.

## Precautions

- If the same `instance_id` is created repeatedly, a WARN log is printed and the call returns immediately (the instance is not created repeatedly).
- `aclshmemx_finalize(instance_id)` switches the context only when the target instance is not the current active instance, and then releases it. Passing a nonexistent `instance_id` returns an error and does not affect the current active instance.
- `aclshmem_finalize()` releases the current active instance and does not switch the context. In multi-instance scenarios, use it together with `aclshmemx_instance_ctx_set`, or directly use `aclshmemx_finalize`.
- The instance context-related API path is protected by a mutex lock.
- The Host current instance is a process-level state: in multi-thread calls, the upper layer must use the same lock to protect the complete sequence of `aclshmemx_instance_ctx_set` and quiet calls, and all threads that switch instances must follow this convention.
- After a Host UDMA quiet is enqueued, the instance is fixed. Later switching of `instance_id` does not change the submitted quiet. The instance, stream, and communication buffers must be preserved until the quiet completes. Synchronize the stream and coordinate with receiving PEs before reading data or releasing/destroying the instance.
- `aclshmemx_quiet_on_stream(NULL)` uses the default stream of the instance; `aclshmem_quiet()` also synchronously waits for the captured default stream in addition to enqueuing.

## Quick Start (Current Repository APIs)

### 1. Creating an Instance

```cpp
#include "shmem.h"
#include <cstdio>

aclshmemx_init_attr_t attr{};
attr.my_pe = pe_id;
attr.n_pes = pe_size;
attr.local_mem_size = 1024UL * 1024UL * 1024UL;
std::snprintf(attr.ip_port, sizeof(attr.ip_port), "%s", "tcp://x.x.x.x:0");       // Set the port to 0 in advance. A port will be allocated later.
attr.comm_args = nullptr;               // default mode
attr.instance_id = 2;

int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

### 2. Switching to the Context of the Target Instance

```cpp
aclshmemx_instance_ctx_set(2);

// Subsequent SHMEM APIs apply to instance 2
void *ptr = aclshmem_malloc(1024);
aclshmem_free(ptr);
```

### 3. Obtaining the Current Instance

```cpp
aclshmem_instance_ctx *ctx = aclshmemx_instance_ctx_get();
// Currently, it is mainly used to read the ID. For other capabilities, see the previous section.
uint64_t cur_id = (ctx == nullptr) ? 0 : ctx->id;
```

### 4. Releasing an Instance

In multi-instance scenarios, directly use `aclshmemx_finalize` to release the instance specified by ID (the context is switched automatically inside the API, so manual set is not required):

```cpp
aclshmemx_finalize(2);
```

You can also use the standard API `aclshmem_finalize()` to release the **current active instance** (that is, the instance set by the latest `aclshmemx_instance_ctx_set`; instance 0 if set has never been called). To release a specific instance with this API, call set first:

```cpp
aclshmemx_instance_ctx_set(2);
aclshmem_finalize();   // Release the current active instance 2
```

In single-instance scenarios, `aclshmem_finalize()` directly releases instance 0, which is fully consistent with the historical usage without parameters.

### 5. UDMA Multi-instance

This section applies to the SIMD UDMA path of Ascend950. Select `SOC_TYPE=Ascend950` during build and include the corresponding Host UDMA backend and Device UDMA implementation. SIMT UDMA multi-instance is not supported yet; the platform conditions in this section do not define the support restrictions of SIMT MTE.

Set the UDMA engine of the instance when the Host creates the instance:

```cpp
attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

Build the Device kernel with `MULTI_INSTANCE`. Each participating core sets the instance at entry, and completes put and quiet within the same instance:

```cpp
aclshmemx_instance_ctx_set(instance_id);
aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(dst, src, ub, bytes, peer, 0);
aclshmemx_udma_quiet(peer);
```

Alternatively, after the producer kernel, the Host can select the corresponding instance and call `aclshmemx_quiet_on_stream` to complete the UDMA operations submitted on that instance:

```cpp
aclshmemx_instance_ctx_set(2);
aclshmemx_quiet_on_stream(stream);   // NULL selects the default stream of the instance
aclrtSynchronizeStream(stream);
```

### 6. Complete MTE Example

[examples/multi_instance](../../../examples/multi_instance/README_en.md) (based on `aclshmemx_mte_put_nbi` / `aclshmemx_mte_get_nbi`)

## Comparison with the Single-Instance Mode

| Feature | Single-Instance Mode | Multi-instance Mode |
|---|---|---|
| Initialization identifier | By default, `instance_id=0` | Instances use different IDs, which may include 0 |
| Resource isolation | Global sharing | Isolation by instance |
| Context switching | Default instance 0; set is not required when SIMT does not define `MULTI_INSTANCE` | The Host/Device use their respective `aclshmemx_instance_ctx_set`; when SIMT defines `MULTI_INSTANCE`, all blocks of every VF must set first, including instance 0 |
| Lifecycle management | `aclshmem_finalize()` | `aclshmemx_finalize(instance_id)` |
| Communication links | MTE, UDMA, and RDMA/SDMA under the platform configuration | MTE and UDMA routed by instance added in this document |
| Advanced team operations | Supported | Supported only when `instance_id=0` |

## SIMT Multi-instance Selection

SIMT MTE follows the existing usage of MTE multi-instance: select the instance through `simt::aclshmemx_instance_ctx_set(instance_id)` first, and then call the existing SIMT communication APIs. No extra instance parameter needs to be passed to the communication APIs. The Host-side instance initialization, memory allocation, and release flows remain unchanged.

SIMT stores the device state address of the selected instance in the UB variable of the current block, and subsequent APIs read the communicator and symmetric memory information of the corresponding instance from it. `simt::aclshmemx_instance_ctx_get()` returns the device state address. SIMT instance selection is independent of the Host and SIMD. It must be set every time a VF is entered, and cannot inherit the selection of the previous VF.

Enable `ACLSHMEM_SIMT_SUPPORT=ON` during use, and define `MULTI_INSTANCE` for multi-instance kernels. All threads in the same block must call set with the same order and the same instance ID. Complete the unfinished communication of the old instance before switching.

```cpp
// Inside the VF, all block threads consistently select the instance.
simt::aclshmemx_instance_ctx_set(instance_id);
simt::aclshmemx_int32_put_nbi_block(dst, src, count, peer);
```

If `MULTI_INSTANCE` is not defined, the existing SIMT APIs use instance 0 by default and no set is required. Non-zero instances support only WORLD team queries. The instance and related memory must remain valid until communication completes.

Current warp RMA also uses the barrier of the entire block: all block threads must call with the same order and count. Arguments are consistent within each warp, and warps can differ. Each warp independently transfers the complete `[0, count)` range; a warp with fewer than 32 threads at the end uses its actual thread count as the stride to complete the same range. Independent warp-branch calls are not supported.

Currently, only SIMT MTE multi-instance is supported; SIMT UDMA multi-instance is not supported yet. The set operation itself does not complete communication.

## Related Code and Documents

- Initialization/finalization flow (including multi-instance ctx creation, port allocation, and finalize): [init_finalize.md](init_finalize.md)
- Host multi-instance switching: [src/host/init/shmem_init.cpp](../../../src/host/init/shmem_init.cpp)
- Device multi-instance switching: [src/device/shmemi_device_common.hpp](../../../src/device/shmemi_device_common.hpp)
- Team multi-instance restriction logic: [src/host/team/shmem_team.cpp](../../../src/host/team/shmem_team.cpp)
- MTE example: [examples/multi_instance](../../../examples/multi_instance/README_en.md)
- UDMA multi-instance test case: [multi_instance_routing_host_test.cpp](../../../tests/unittest/host/multi_instance/multi_instance_routing_host_test.cpp)
