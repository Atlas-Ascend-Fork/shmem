## Overview

This sample shows how a device kernel uses multiple QPs over direct UDMA to concurrently perform Put, Get, and Put-with-Signal transfers to the same remote PE.

It provides three functional validation modes:

- `put`: The current PE uses multiple QPs to write local ordinary device memory to the SHMEM symmetric memory of the next PE.
- `get`: The current PE uses multiple QPs to read the SHMEM symmetric memory of the next PE into local ordinary device memory.
- `put_signal`: Each QP uses one `WRITE_WITH_NOTIFY` WQE to write its data slice and an independent signal word.

## Version and Platform Support

- The UDMA multi-QP sample supports Ascend 950 only.
- CANN 9.1.0 provides the required HCOMM resource APIs, including `HcommEndpointCreate`, `HcommMemReg`, `HcommChannelCreate`, and `HcommChannelGetStatus`. Install the Ascend 950 toolkit and ops packages from [CANN 9.1.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0). Earlier CANN versions are not supported by this sample.
- Multi-QP operation also requires the current HCOMM `HcommChannelDesc` to support `channelName`.
- Initialization loads and verifies the required HCOMM symbols. Before building, run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (or the corresponding `set_env.sh` for a custom installation) and build with `-soc_type Ascend950`. Initialization fails when the HCOMM runtime library or symbols are incomplete.

## Sample Implementation

The sample demonstrates the basic flow for concurrent UDMA Put, Get, and Put-with-Signal operations from multiple QPs to the same remote PE.

### Test Flow

1. Initialize ACL and SHMEM, configure the data path as `ACLSHMEM_DATA_OP_UDMA`, and call the process-wide `aclshmemx_set_qp_num` before SHMEM initialization to set the number of QPs per remote PE.
2. Each PE maps to one process and one NPU, and communicates with the next PE in a ring. Put and PutSignal write to the next PE; Get reads from the next PE.
3. Each PE allocates SHMEM symmetric memory and ordinary device memory, then prepares data containing the PE ID and element position. PutSignal additionally allocates one exclusive symmetric signal word per QP.
4. The host launches as many AIVs as QPs. Each AIV uses one QP and handles a non-overlapping data slice.
5. After the kernel completes, results are copied back to the host and checked byte by byte. Put and PutSignal data must come from the previous PE, while Get data must come from the next PE. PutSignal also validates the independently encoded signal value of each QP.
6. Release SHMEM, device, and ACL resources after validation.

### Kernel Flow

1. Each AIV uses its task block ID as its QP index, and divides the total bytes into non-overlapping contiguous slices using quotient and remainder.
2. Put calls `aclshmemx_udma_qp_put_nbi`, Get calls `aclshmemx_udma_qp_get_nbi`, and PutSignal calls `aclshmemx_udma_qp_put_signal_nbi`. Each AIV submits only its own slice.
3. Each AIV calls `aclshmemx_udma_qp_quiet` after submission to wait for operations on its current remote PE and QP. PutSignal data and signal are in the same WQE; no implicit ordering is provided between different QPs.

## Build and Run

For environment setup, see [Quick Start](../../docs/quickstart_en.md). Run the following commands from the repository root after the environment is ready.

### Default Link Mode

Without `-multi_link`, multiple QPs use the default link policy. If both MESH and CLOS routes exist for a peer, every QP uses MESH. Only when MESH is unavailable do QPs rotate among available CLOS routes. This mode supports `-qp_count`, but does not allow per-QP link selection or `-qp_net_layers`.

```bash
# Default build: multi-link is disabled. All QPs use MESH when available; otherwise they rotate among CLOS routes.
bash scripts/build.sh -examples -soc_type Ascend950

# Run the default case: 2 PEs, 2 QPs per remote PE, Put mode.
bash examples/udma_qp_demo/run.sh
```

The operation, QP count, and element count can also be specified:

```bash
# Validate Put with 4 QPs.
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put -elems 1048576

# Validate Get with 8 QPs.
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 8 -op get -elems 1048576

# Validate PutSignal with 4 QPs.
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put_signal -elems 1048576

# Bind 4 PEs starting from logical NPU 4.
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 4 -op put -first_npu 4

# Use non-contiguous logical NPU IDs: PE0/1/2/3 bind to NPU1/3/2/4 respectively.
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 4 -op put -device_ids 1,3,2,4
```

### Multi-link Mode

When built with `-multi_link`, QPs rotate among available MESH and CLOS routes when `-qp_net_layers` is not supplied. Use `-qp_net_layers` to configure a route for selected QPs; an explicit configuration replaces that QP's default assignment. QPs without an explicit configuration retain default route selection.

```bash
# Enable multi-link support.
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link

# QP 0 uses netLayer 0 and QP 2 uses netLayer 1; QP 1 retains default route selection.
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 3 -qp_net_layers 0:0,2:1

# All PEs still use the same QP/layer policy; this command only changes the PE-to-device mapping.
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 3 -device_ids 1,3,2,4 \
    -qp_net_layers 0:0,2:1
```

On success, each PE prints output similar to:

```text
[PASS] op=put pe=0 elements=1048576
[PASS] op=put pe=1 elements=1048576
```

PutSignal also prints signal validation results:

```text
[PASS] op=put_signal pe=0 elements=1048576
[PASS] op=put_signal pe=0 signals=4
[PASS] op=put_signal pe=1 elements=1048576
[PASS] op=put_signal pe=1 signals=4
```

On data validation failure, the program prints the first failing position, actual value, and expected value. `run.sh` waits for all PE processes and returns a non-zero status when any process fails.

### Runtime Parameters

`run.sh` supports the following options:

| Option | Default | Description |
| --- | --- | --- |
| `-pes` | `2` | Total number of PEs. It must be at least 2; the script launches the same number of processes. |
| `-qp_count` | `2` | QPs created for each remote PE. Range: `[1, 56]`. The value must be identical on all PEs. |
| `-qp_net_layers` | Not configured | Available only when built with `-multi_link`. Configures a UDMA route for selected QPs in `qp_idx:netLayer` form, for example `0:0,2:1`. Each `qp_idx` must be less than `qp_count` and must not be repeated. Only layers 0 and 1 are currently supported. An explicit configuration replaces the QP's default assignment; unspecified QPs use the default rotation policy across available MESH and CLOS routes. All PEs must use the same configuration. A default build does not support this option: all QPs use MESH when available, otherwise they rotate among available CLOS routes. |
| `-op` | `put` | Operation mode: `put`, `get`, or `put_signal`. |
| `-elems` | `1048576` | Bytes transferred by each PE. The value must be at least the QP count; uneven partitioning is supported. |
| `-heap_mb` | `1024` | SHMEM symmetric heap size per PE in MiB. PutSignal must accommodate the data buffer and one `uint64_t` signal word per QP. |
| `-peer_stride` | `1` | Target-PE distance in the ring. The target, QP net-layer query, and log all use `(pe + peer_stride) % pes`. Range: `[1, pes)`. |
| `-first_npu` | `0` | First local NPU. Local process `i` uses `first_npu + i`. |
| `-device_ids` | Not configured | Comma-separated logical device IDs. The count must equal `pes` and IDs must not repeat. Item `i` binds PE `i`, for example `1,3,2,4`. It takes precedence over `first_npu`. |
| `-ipport` | Dynamic local port | Bootstrap address. If omitted, the script selects a local port randomly. |
