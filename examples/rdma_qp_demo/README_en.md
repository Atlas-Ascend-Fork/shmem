# RDMA Multi-QP Demo

## Overview

This sample is based on the SHMEM project. It describes how device kernels use multiple QPs to concurrently perform regular and aggregated Put/Get data transfers to the same peer PE in the XSCALE and 1825 (HNS_1825) RDMA scenarios.

The sample covers regular and aggregated Put/Get on both backends:

- `put`: The current PE uses multiple QPs to write the data in the local SHMEM symmetric memory to the symmetric memory of the next PE.
- `get`: The current PE uses multiple QPs to read data from the SHMEM symmetric memory of the next PE to the local symmetric memory.
- `aggregate_put`: Each QP aggregates multiple Put operations into a batch and then submits the batch.
- `aggregate_get`: Each QP aggregates multiple Get operations into a batch and then submits the batch.

`-op all` runs the four modes on both XSCALE and HNS_1825.

## Requirements

For the RDMA NIC, driver, Ascend950 CANN version, and `IBV_EXTEND_DRIVERS` configuration requirements of this sample, see [rdma_demo requirements](../rdma_demo/README_en.md#environment-requirements).

- This sample supports the XSCALE and HNS_1825 RDMA backends on the Ascend950 platform.
- All PEs must be configured with the same QP count, in the range of `[1, 56]`.
- Each QP processes a fixed 512 KiB of data. The aggregate mode divides the slice into eight chunks of equal size and uses a 1088B UB workspace.

## Sample Implementation

This sample shows multiple QPs concurrently performing regular and aggregated RDMA Put/Get between the same pair of peer PEs.

### Test Case Implementation

(1) Initialize ACL and SHMEM, set the data path to `ACLSHMEM_DATA_OP_ROCE`, and set the QP count used for each peer PE through the process-level API `aclshmemx_set_qp_num` before SHMEM initialization.

(2) Each PE corresponds to one process and one NPU, and communicates with the next PE in the ring topology. Put mode writes data to the next PE, and Get mode reads data from the next PE.

(3) Each PE allocates a segment of SHMEM symmetric memory. The first half is used as the local source buffer, and the second half is used as the local destination buffer. Test data containing PE IDs and element positions is prepared.

(4) The Host starts the same number of AIVs as the QP count. Each AIV uses one QP and processes non-overlapping data slices; each QP processes a fixed 512 KiB.

(5) After the kernel is executed, the destination buffer is copied back to the Host for byte-by-byte verification. The Put result should come from the previous PE in the ring topology, and the Get result should come from the next PE.

(6) In `all` mode, both backends sequentially execute regular and aggregated Put/Get, where aggregate operations are submitted through the submit/defer APIs. Resources are released after all verifications are complete.

### Kernel Implementation

(1) Each AIV uses its own task block ID as the QP ID and locates the 512 KiB data slice it is responsible for based on the ID:

```text
block 0 -> QP 0
block 1 -> QP 1
...
```

(2) Regular Put mode calls `aclshmemx_roce_qp_put_nbi`, and regular Get mode calls `aclshmemx_roce_qp_get_nbi`. Both backends use the overloads without aggregate action parameters. Each AIV submits its own complete data slice once.

(3) In aggregate mode, each AIV splits its data slice into eight chunks and creates an independent `aclshmemx_submit_state_t` and UB workspace. The first seven operations join the batch using `aclshmemx_defer_t`, and the last operation submits the entire batch using `aclshmemx_submit_t`.

(4) Operations in each aggregate batch use the same peer PE, QP, UB base address, and internal synchronization event ID. Different QPs can concurrently access the same PE, but each QP must be used by a single AIV with an independent `aclshmemx_submit_state_t` and UB workspace. Only one active batch is allowed for the same `(PE, QP)` at a time. The total number of operations in a batch (including the final submit) must not exceed `ACLSHMEM_ROCE_QP_AGGREGATE_MAX_OPS`. If the limit is exceeded, the operations need to be split into multiple batches.

(5) At the beginning of the kernel, block 0 calls `aclshmemx_roce_sync_all()` to complete cross-PE synchronization, and then releases all local AIVs through `SyncAll`. After submitting operations, each AIV calls `aclshmemx_roce_qp_quiet()` to wait for its own QP to complete, and then joins through `SyncAll`. Block 0 then calls `aclshmemx_roce_sync_all()` to perform cross-PE synchronization. The final `SyncAll` ensures that all AIVs exit after the cross-PE synchronization is complete.

## Build and Execution

Enable RDMA XSCALE or HNS_1825 in the repository root directory to build the examples, and then run the demo. Both backends build and run the regular and aggregated Put/Get paths:

```bash
# XSCALE
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE -examples
# HNS_1825
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend HNS_1825 -examples
bash examples/rdma_qp_demo/run.sh
```

Specify the first NPU and the QP count. Both backends run the four modes by default:

```bash
bash examples/rdma_qp_demo/run.sh -fnpu 0 -qp 4
```

To debug a single mode, additionally specify the mode, for example, `-op put`.

In multi-server running, the `-pes`, `-qp`, and `-ipport` values of all servers must be the same, and `-ipport` points to the server where PE 0 resides. The script sets the same `SHMEM_UID_SESSION_ID` accordingly. Assume that two servers each use two NPUs and the IP address of server A is `192.168.1.10`. Run the following commands on the two servers respectively:

```bash
# Server A: start PE 0~1, using local NPUs 0~1
bash examples/rdma_qp_demo/run.sh -pes 4 -fpe 0 -gnpus 2 -fnpu 0 -qp 4 \
    -ipport tcp://192.168.1.10:8899

# Server B: start PE 2~3, using local NPUs 0~1
bash examples/rdma_qp_demo/run.sh -pes 4 -fpe 2 -gnpus 2 -fnpu 0 -qp 4 \
    -ipport tcp://192.168.1.10:8899
```

The default case uses two QPs, and the total data volume is 1 MiB. When the execution succeeds, each PE outputs:

```text
[PASS] op=put pe=0 elements=1048576
[PASS] op=get pe=0 elements=1048576
[PASS] op=aggregate_put pe=0 elements=1048576
[PASS] op=aggregate_get pe=0 elements=1048576
```

If data verification fails, the program prints the first error position, the actual value, and the expected value. `run.sh` waits for all PE processes on the current server and returns a non-zero value if any process fails.

### msprof Profiling

The profiling result of each PE is saved to `output/<timestamp>/pe_<PE_ID>` respectively. `-n N` specifies the number of kernel launches for each selected operation, and defaults to 1. Initialization and buffer preparation are performed only once, and the last result of each operation is used for verification. This parameter is also supported in direct running mode.

```bash
bash examples/rdma_qp_demo/run.sh -qp 4 -op put -tool msprof -n 100
```

`-tool none` runs the demo directly. In cross-server profiling, all servers must use the same `-n` and `-op`. msprof starts the program through a temporary Shell wrapper script and preserves the program path and argument boundaries.

Use Python 3 to extract the operator call count, total time, and average time of all local PEs. The script depends only on the Python standard library, so no third-party package needs to be installed. The time units follow those in the msprof CSV.

```bash
python3 examples/utils/extract_totaltime.py \
    examples/rdma_qp_demo/output/<timestamp>
```

The result is saved to `totaltime.csv` in the profiling directory by default, and the script prints the full path of the generated file when it finishes. Use `-o FILE` to specify another path. The output columns are `PE`, `Op Name`, `N` (call count), `Total Time`, and `Avg Time`.

The rows of each card also contain `Min Time` and `Max Time` of a single kernel. Each operator adds an `PE=ALL` summary row: `Mean PE Avg Time` is the average of the average time of all cards, `Max PE Avg Time` is the maximum of the average time of all cards, and `Slowest PE` is the ID of the corresponding slowest card.

### Runtime Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `-pes` | `2` | Global PE count, at least 2 |
| `-fpe` | `0` | First global PE started on the current server |
| `-gnpus` | `2` | Number of NPU/PE processes started on the current server |
| `-fnpu` | `0` | First logical NPU ID used on the current server |
| `-qp` | `2` | QP count for each peer PE, range 1 to 56 |
| `-op` | `all` | `put`, `get`, `aggregate_put`, `aggregate_get`, or `all` |
| `-ipport` | Dynamic local port | SHMEM initialization address |
| `-tool` | `none` | `msprof` enables profiling, `none` runs directly |
| `-n` | `1` | Kernel launches per selected operation, must be a positive integer |
