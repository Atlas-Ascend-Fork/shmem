[中文](README.md)

## Overview

This example uses multiple QPs to transfer data to or from the same peer PE through direct UDMA. It supports three verification modes:

- `put`: copy local ordinary device memory to the next PE's symmetric memory.
- `get`: read the next PE's symmetric memory into local ordinary device memory.
- `put_signal`: each QP uses one `WRITE_WITH_NOTIFY` WQE to write its data partition and its own signal word.

## Platform and version requirements

- This example supports Ascend950 only.
- CANN 9.1.0 provides the required HCOMM resource interfaces, including `HcommEndpointCreate`, `HcommMemReg`, `HcommChannelCreate`, and `HcommChannelGetStatus`. Install the Ascend950 toolkit and ops packages from the [CANN 9.1.0 resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0). Earlier versions are unsupported.
- Multiple QPs also require `channelName` in the runtime's `HcommChannelDesc`.
- Source `/usr/local/Ascend/ascend-toolkit/set_env.sh` before building, or the equivalent script for your installation. Build with `-soc_type Ascend950`. Initialization fails if required HCOMM libraries or symbols are unavailable.

## Implementation

### Host flow

1. Initialize ACL, select `ACLSHMEM_DATA_OP_UDMA`, and call the process-level `aclshmemx_set_qp_num` before SHMEM initialization. To enable shared jetty, set `attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED` before initializing. It is disabled by default and must be configured consistently on all PEs. Peers resolving to the same `(local eid_index, QP slot)` share an underlying jetty; with one QP, this reduces to the same local EID.
2. Each PE owns one process and one NPU. Put and PutSignal write to the next PE in a ring; Get reads from that PE.
3. Allocate symmetric memory and ordinary device memory. Input values encode the PE and element position. PutSignal also allocates a separate symmetric signal word per QP.
4. Launch one AIV per QP. Each AIV owns one QP and a separate data partition.
5. Copy results to the host for byte-by-byte verification. Put and PutSignal results come from the previous PE; Get results come from the next PE. PutSignal also verifies each QP's encoded signal value.
6. Release SHMEM, device, and ACL resources.

### Kernel flow

Each AIV uses its block index as the QP index and divides the byte count into disjoint partitions using quotient and remainder. The selected mode calls `aclshmemx_udma_qp_put_nbi`, `aclshmemx_udma_qp_get_nbi`, or `aclshmemx_udma_qp_put_signal_nbi`, followed by `aclshmemx_udma_qp_quiet` for the current peer and QP. PutSignal data and signal share one WQE. Different QPs have no implicit ordering.

## Build and run

Follow [Quick Start](../../docs/quickstart_en.md) to configure the environment, then run the following commands from the repository root.

### Default link mode

Without `-multi_link`, all QPs use MESH when the peer has a MESH route. Otherwise, QPs rotate over available CLOS links. `-qp_count` selects the number of QPs; `-qp_net_layers` is unsupported in this build.

```bash
bash scripts/build.sh -examples -soc_type Ascend950

# Default: two PEs, two QPs per peer, put mode.
bash examples/udma_qp_demo/run.sh

bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put -elems 1048576
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 8 -op get -elems 1048576
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put_signal -elems 1048576

# Enable shared jetty through the initialization runtime_config.
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -shared_jetty 1 -op put -elems 1048576

# Four consecutive logical devices, starting from NPU 4.
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 4 -op put -first_npu 4

# Map PE0/1/2/3 to logical NPU1/3/2/4.
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 4 -op put -device_ids 1,3,2,4
```

### Multi-link mode

Build with `-multi_link` to rotate QPs over available MESH and CLOS links by default. `-qp_net_layers` overrides the route policy for specified QPs; unspecified QPs retain the default rotation policy.

```bash
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link

# QP0 uses netLayer 0, QP2 uses netLayer 1, and QP1 retains its default policy.
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 3 -qp_net_layers 0:0,2:1

# The QP policy is identical on all PEs; device mapping is independent.
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 3 -device_ids 1,3,2,4 \
    -qp_net_layers 0:0,2:1
```

### Verification output

Successful runs print a result on each PE:

```text
[PASS] op=put pe=0 elements=1048576
[PASS] op=put pe=1 elements=1048576
```

PutSignal additionally reports signal verification:

```text
[PASS] op=put_signal pe=0 elements=1048576
[PASS] op=put_signal pe=0 signals=4
[PASS] op=put_signal pe=1 elements=1048576
[PASS] op=put_signal pe=1 signals=4
```

On a mismatch, the program reports the first failing position, actual value, and expected value. `run.sh` waits for all PE processes and returns a nonzero status if any process fails.

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-pes` | `2` | Number of PEs and processes; at least 2. |
| `-qp_count` | `2` | QPs per peer, in `[1, 56]`; identical on all PEs. |
| `-shared_jetty` | `0` | `0` disables sharing; `1` sets `attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED` before initialization. All PEs must agree. |
| `-qp_net_layers` | Unspecified | Available only with `-multi_link`. Comma-separated `qp_idx:netLayer` overrides, such as `0:0,2:1`. QP indices must be unique and below `qp_count`; supported layers are 0/1. Unspecified QPs retain default MESH/CLOS rotation. All PEs must use the same policy. |
| `-op` | `put` | `put`, `get`, or `put_signal`. |
| `-elems` | `1048576` | Bytes per PE; at least the QP count. Uneven partitioning is supported. |
| `-heap_mb` | `1024` | Symmetric heap size per PE in MiB. PutSignal needs room for both data and one `uint64_t` signal per QP. |
| `-peer_stride` | `1` | Peer distance in the ring, in `[1, pes)`. Transfers, route queries, and logs use `(pe + peer_stride) % pes`. |
| `-first_npu` | `0` | First logical device; local process i uses `first_npu + i`. |
| `-device_ids` | Unspecified | Comma-separated logical devices, one unique ID per PE. Entry i binds PE i. Overrides `-first_npu`. |
| `-ipport` | Dynamic local port | Bootstrap address; the script chooses a local port when omitted. |
