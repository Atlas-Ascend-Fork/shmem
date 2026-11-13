# Multi-instance Support

[简体中文](multi_instance.md)

## Overview

The SHMEM multi-instance capability allows multiple independent SHMEM instances to be created within a single process. Each instance has its own communicator, memory space, and synchronization resources. The multi-instance capability is applicable to the following scenarios:

- **Coexistence of multiple communicators**: Multiple task domains in a single process require independent SHMEM contexts.
- **Resource isolation**: SHMEM resources for different tasks/modules are isolated to avoid mutual interference.
- **Elastic scaling**: Creation and release of different instances do not affect each other.
- **Dynamic instance management**: Instances are created or released by `instance_id` at runtime.

Supported communication links are MTE and UDMA: UDMA operations are routed per instance, and the Host can complete UDMA operations submitted on a selected instance with `aclshmemx_quiet_on_stream`.

## Core Concepts

### Instance
An SHMEM instance contains a complete set of initialization states, typically including:
- Independent communicator (`Team World`)
- Independent shared memory heap (`Heap`)
- Independent synchronization resource (`Sync Pool/Counter`)
- Independent runtime snapshots of State, Bootstrap, and MemoryManager

### Instance Identifier (Instance ID)
The current repository uses `instance_id` (`uint64_t`) to identify instances. The instance lifecycle is managed by the initialization attributes and `finalize` API.

### Context Switching
The Host uses **global variable swap** to save the current instance state and load the target instance state. Host and Device both use `aclshmemx_instance_ctx_set`, but select contexts in their respective execution domains. Device UDMA submission and matching completion must use the same instance; complete outstanding operations before switching. After Host quiet captures its instance at enqueue, Host selection may change. Preserve the old instance, streams and buffers until completion, establish cross-stream dependencies, and serialize Host selection and calls across threads.

## Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                     Single process                     │
│  ┌───────────────────────────────────────────────────┐  │
│ │              SHMEM runtime (global)                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │
│  │  │  Instance 0 │  │  Instance 1 │  │  Instance N│ │  │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌──────┐ │  │  │
│  │  │  │ Team  │  │  │  │ Team  │  │  │  │ Team │ │  │  │
│  │  │  │ Heap  │  │  │  │ Heap  │  │  │  │ Heap │ │  │  │
│  │  │  │ Sync  │  │  │  │ Sync  │  │  │  │ Sync │ │  │  │
│  │  │  │ State │  │  │  │ State │  │  │  │ State│ │  │  │
│  │  │  └───────┘  │  │  └───────┘  │  │  └──────┘ │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Design Point| Current Implementation| Description|
|---|---|---|
| Instance isolation| Independent `aclshmem_context` snapshot| Each instance stores independent `state`, `host_state`, `boot_handle` and `memory_manager`.|
| Context management| Global variable swap| Switching through `aclshmemx_instance_ctx_set`|
| Instance identifier| `instance_id` (`uint64_t`) | Located in `aclshmemx_init_attr_t` and used for `aclshmemx_finalize(instance_id)`|
| Device-side route| `instance_id << 1` | The kernel accesses the `state` of the corresponding instance through the hard-coded address.|

## Quick Start (Current Repository APIs)

### 1. Creating an Instance

```cpp
#include "shmem.h"
#include <cstdio>

aclshmemx_init_attr_t attr{};
attr.my_pe = pe_id;
attr.n_pes = pe_size;
attr.local_mem_size = 1024UL * 1024UL * 1024UL;
std::snprintf(attr.ip_port, sizeof(attr.ip_port), "%s", "tcp://x.x.x.x:0");       // Set the port to 0. A port will be allocated later.
attr.comm_args = nullptr;               // default mode
attr.instance_id = 2;

int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

### 2. Switching to the Context of the Target Instance

```cpp
aclshmemx_instance_ctx_set(2);

// Subsequent SHMEM APIs apply to instance 2.
void *ptr = aclshmem_malloc(1024);
aclshmem_free(ptr);
```

### 3. Obtaining the Current Instance

```cpp
aclshmem_instance_ctx *ctx = aclshmemx_instance_ctx_get();
// Currently, it is mainly used to read the ID. For details about other capabilities, see the previous section.
uint64_t cur_id = (ctx == nullptr) ? 0 : ctx->id;
```

### 4. Releasing the Instance

```cpp
aclshmemx_finalize(2);
```
You can also refer to the MTE-based example in [examples/multi_instance](../../examples/multi_instance/README.md) (using `aclshmemx_mte_put_nbi` / `aclshmemx_mte_get_nbi`).

### 5. UDMA Multi-instance

Set the UDMA engine when the Host creates an instance:

```cpp
attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

Build the Device kernel with `MULTI_INSTANCE`; every participating AIV sets the instance at entry and issues put and quiet within the same instance:

```cpp
aclshmemx_instance_ctx_set(instance_id);
aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(dst, src, ub, bytes, peer, 0);
aclshmemx_udma_quiet(peer);
```

Alternatively, after the producer kernel, the Host can select the instance and call `aclshmemx_quiet_on_stream` to complete the UDMA operations submitted on that instance:

```cpp
aclshmemx_instance_ctx_set(2);
aclshmemx_quiet_on_stream(stream);   // NULL selects the instance's default stream
aclrtSynchronizeStream(stream);
```

## Specifications and Restrictions (Code Implementation)

| Project| Specifications|
|---|---|
| Type of `instance_id`| `uint64_t` |
| Default instance| `instance_id = 0` |
| Upper limit of PE| `ACLSHMEM_MAX_PES = 16384` |
| Upper limit of local memory| `ACLSHMEM_MAX_LOCAL_SIZE = 40GB` |
| Team behavior of non-0 instances| Only the WORLD slot is initialized. Host team management APIs (`split`, `translate`, `get_config`) return `ACLSHMEM_NOT_SUPPORTED`; SIMT WORLD queries and PE mappings remain supported.|
| Maximum length of `ip_port`| `ACLSHMEM_MAX_IP_PORT_LEN = 64` (including the ending `\0`)|
| Default multi-instance port range| Depending on `SHMEM_INSTANCE_PORT_RANGE=start:end`|
| Default port allocation rule| The input port must be `0`. The actual port is = `start_port + instance_id`.|
| Default available instance ID range| `instance_id <= end_port - start_port` (including the boundary)|
| Communication engine support | SIMD calls support MTE and UDMA; SIMT calls support MTE only |
| UDMA QP configuration| Set the process-wide QP count with `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num)` before the first initialization. It cannot change while an instance is active. Queue resources, metadata and completion state are isolated per instance|
| Host UDMA quiet| `aclshmemx_quiet_on_stream(stream)` captures the current instance at enqueue; nonzero instances complete all QPs including shared Jetty queues; instance 0 retains QP0 completion and explicit nonzero QPs require `aclshmemx_udma_qp_quiet`|
| UDMA instance ID range| HBM-only initialization supports 0–255; configurations creating a Host DRAM companion with `USE_ACLRT_MEM_FABRIC_HANDLE` support at most 254. Port ranges and resources impose additional limits. Initialization above the HBM limit returns `ACLSHMEM_INVALID_VALUE`. Quiet requires a successfully initialized live instance and rejects IDs above 255 before enqueue|

## Environment Variables and Running Requirements

### Kernel-side Usage Requirements
- SIMD calls (including UDMA): build with `MULTI_INSTANCE` and select a successfully initialized live instance with `aclshmemx_instance_ctx_set(id)` in every participating core before communication. All builds check the ID range before calculating the context address and trap on out-of-range IDs; diagnostic printing is Debug-only. Callers must still ensure that an in-range instance is initialized and alive. Context consistency checks remain Debug diagnostics. Complete outstanding operations before switching instances; UDMA submission and matching completion must use the same instance.
- A UDMA producer kernel must finish local staging and submit pending batches through public submission APIs before returning. The library publishes internal queue state. Order the producer and quiet on the same stream or with an explicit cross-stream event dependency, and do not concurrently submit to or quiet the same queues from another stream.
- Explicit nonzero QPs use `aclshmemx_udma_qp_quiet(peer, qp)` for completion.
- Keep the instance, streams and memory alive until communication completes.
- SIMT calls: define `MULTI_INSTANCE`; all threads in each VF collectively call `simt::aclshmemx_instance_ctx_set(id)` before the existing APIs. Selection is independent of Host/SIMD.

### Default Multi-instance Mode
- `SHMEM_INSTANCE_PORT_RANGE` must be set in the format of `start:end`.
- The port in `ip_port` in the initialization attribute must be `0`, for example, `tcp://x.x.x.x:0`.

### Unique ID Mode
- Follow the existing `SHMEM_UID_SESSION_ID` and `SHMEM_UID_SOCK_IFNAME` rules.

## Precautions

- If instances with the same `instance_id` are being created, a WARN log is generated and returned immediately (the instance will not be created repeatedly).
- `aclshmemx_finalize(instance_id)` switches to the target instance and then releases the instance to avoid incorrect release.
- The instance context-related API path is protected by a mutex lock.
- Host instance selection is process-wide: when multiple Host threads are used, serialize the whole `aclshmemx_instance_ctx_set` plus quiet call sequence with one external lock, and all threads that change instances must follow this convention.
- A Host UDMA quiet keeps the instance captured at enqueue, so later context switches do not redirect it. Keep the instance, stream and communication buffers alive until quiet completes; then synchronize the stream and coordinate receiving PEs before consuming data or releasing/finalizing the instance.
- `aclshmemx_quiet_on_stream(NULL)` uses the instance's default stream; `aclshmem_quiet()` also waits for the captured default stream.

## Comparison with the Single-Instance Mode

| Feature| Single-Instance Mode| Multi-instance Mode|
|---|---|---|
| Initialization identifier| `instance_id=0`| `instance_id>0` |
| Resource isolation| Global sharing| Isolation by instance|
| Context switching| Default instance 0| Host/Device: `aclshmemx_instance_ctx_set`|
| Lifecycle management| `aclshmem_finalize()` | `aclshmemx_finalize(instance_id)` |
| Communication links| MTE, UDMA and RDMA/SDMA as supported by the platform configuration| MTE and UDMA with per-instance routing described here|
| Advanced team operations| Supported| Supported only when `instance_id=0`|

## Related Code and Documents

- Host initialization and multi-instance implementation: [src/host/init/shmem_init.cpp](../../src/host/init/shmem_init.cpp)
- Device multi-instance state routing: [src/device/shmemi_device_common.hpp](../../src/device/shmemi_device_common.hpp)
- Team multi-instance restriction logic: [src/host/team/shmem_team.cpp](../../src/host/team/shmem_team.cpp)
- MTE example: [examples/multi_instance](../../examples/multi_instance/README.md)
- UDMA multi-instance tests: [multi_instance_routing_host_test.cpp](../../tests/unittest/host/multi_instance/multi_instance_routing_host_test.cpp)


## SIMT instance selection

Enable `ACLSHMEM_SIMT_SUPPORT` and define `MULTI_INSTANCE`
for multi-instance kernels. Inside every VF, all block threads collectively call
`simt::aclshmemx_instance_ctx_set(instance_id)` with the same ID before using the
existing SIMT MTE APIs. `simt::aclshmemx_instance_ctx_get()` returns the selected device
state address, not the instance ID. No separate context class is provided.

Selection is block-local compiler-allocated UB state, independent of Host/SIMD
selection. No fixed UB address or extra VF state parameter is required. Initialize
the selection in every VF, including instance 0; do not inherit previous VF state.
Without `MULTI_INSTANCE`, legacy interfaces continue to use instance 0 without set.
The setter synchronizes the whole block; divergent or warp-only calls are invalid.
Currently only SIMT MTE multi-instance operation is supported; SIMT UDMA
multi-instance operation is not yet supported. Complete pending communication
before switching; set does not complete communication.
Keep instances and allocations alive until communication completes.

Warp RMA currently uses block barriers: all block threads must enter
calls in the same order. Arguments are uniform within each warp; partial last
warps are supported. Each warp transfers its complete `[0, count)` range; a partial
tail warp uses its actual thread count as the stride for that range. Independent
warp branches and unequal per-warp call counts are unsupported. Nonzero instances
support WORLD team queries only.
