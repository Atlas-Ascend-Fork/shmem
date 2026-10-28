## Overview

This sample demonstrates, in the direct UDMA scenario, how a device kernel uses the SIMT programming model to concurrently perform Put/Get data transfers to the same remote PE through multiple QPs.

The sample covers both SIMT execution scopes (warp/block). Under each scope, every warp/block drives one QP, validating the data correctness of `simt::aclshmemx_udma_qp_{put,get}_nbi_{warp|block}` and `simt::aclshmemx_udma_qp_quiet_{warp|block}` under a multi-QP configuration. Two functional validation modes are provided:

- `put`: The current PE uses multiple QPs to write data from local SHMEM symmetric memory to the SHMEM symmetric memory of the next PE.
- `get`: The current PE uses multiple QPs to read data from the SHMEM symmetric memory of the next PE into local SHMEM symmetric memory.

## Version and Platform Support

- This sample supports Ascend950 (direct mode) only. Other platforms are not supported.
- CANN 9.2.0 and later provide the HCOMM resource APIs required by UDMA. Download and install the Ascend950 toolkit and ops packages from [CANN 9.2.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.2.0). CANN versions earlier than 9.2.0 are not supported by this sample.
- Multi-QP operation also requires the current HCOMM `HcommChannelDesc` to support `channelName`.
- Before building, run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (or the corresponding `set_env.sh` for a custom installation path), and build with `-enable_simt -soc_type Ascend950`. SIMT support must be enabled at compile time via `-enable_simt`.

## Sample Implementation

The sample demonstrates the basic flow for concurrent SIMT UDMA Put/Get operations from multiple QPs to the same remote PE, covering both warp and block scopes.

### Test Flow

1. Initialize ACL and SHMEM, configure the data path as `ACLSHMEM_DATA_OP_UDMA`, and call the process-wide `aclshmemx_set_qp_num` before SHMEM initialization to set the number of QPs per remote PE.

2. Each PE corresponds to one process and one NPU, and communicates with the next PE in a ring topology. Put mode writes data to the next PE; Get mode reads data from the next PE.

3. Each PE allocates three segments of SHMEM symmetric memory: `origin` (local data source), `res_put` (Put result area), and `res_get` (Get result area), and prepares test data containing the PE ID and element position. The buffers are allocated once and shared by both the warp and block modes.

4. The host launches the SIMT kernel via `<<<...>>>`, and the kernel dispatches SIMT VFs via `asc_vf_call` to perform data transfers. Under each scope, every warp/block drives one QP and processes a non-overlapping data slice.

5. After the kernel completes, results are copied back to the host and validated byte by byte. The Put result (`res_put`) should match the pattern from the previous PE; the Get result (`res_get`) should match the pattern from the next PE.

6. After validation, `aclshmem_finalize` releases the SHMEM heap together with the device and ACL resources. A verification failure only records the failure state; both PEs keep the same mode plan and perform cleanup at a single point before returning non-zero, so that one PE never exits early and leaves the other blocked in a collective operation.

### Kernel Flow

1. QP driving methods for the two scopes:

| Scope | kernel grid | VF launch form | QP driving method |
|-------|------------|----------------|-------------------|
| warp | `<<<1>>>` | `asc_vf_call<demo_warp_vf>(dim3(qp_count * 32), ...)` | warp i drives QP i (warp collective call; threads within the warp cooperate to submit WQEs; the quiet interface relies on warp lockstep with no block-level synchronization; all threads in the VF must participate) |
| block | `<<<qp_count>>>` | `asc_vf_call<demo_block_vf>(dim3(128), ...)` per block | block i drives QP i (reads `blockIdx.x` as the QP index inside the VF; only warp 0 assembles and submits WQEs; other warps do not participate in submission; the quiet interface includes block-level synchronization; all 128 threads must participate) |

2. Each driver (warp/block) divides non-overlapping contiguous slices based on its `qp_idx` using quotient and remainder.

3. Put mode calls `simt::aclshmemx_udma_qp_put_nbi_{warp|block}`; Get mode calls `simt::aclshmemx_udma_qp_get_nbi_{warp|block}`. Each driver submits only its own data slice.

4. After submission, call `simt::aclshmemx_udma_qp_quiet_{warp|block}` to wait for operations on the current remote PE and current QP to complete. No implicit ordering is provided between different QPs.

5. The warp/block variants are collective calls; every thread of the VF must invoke them. The block quiet variant internally uses `asc_syncthreads` to ensure all threads observe the completed state before returning; the warp quiet variant relies on warp lockstep with no block-level synchronization. The put/get warp/block variants are also collective calls; all threads of the VF must participate.

## Build and Run

For environment setup, see [Quick Start](../../docs/quickstart_en.md). After completing the environment configuration, run the following from the repository root:

```bash
# Build the SIMT UDMA sample (SIMT support must be enabled)
bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
```

Build artifact: `build/bin/simt_udma_qp_demo`.

### Run Examples

```bash
# Default: 2 PEs, 2 QPs, both modes (warp/block) run sequentially, 1 MB data
bash examples/simt_udma_qp_demo/run.sh

# Specify parameters: 4 PEs, 4 QPs, warp mode only, 1 MB data
bash examples/simt_udma_qp_demo/run.sh -pes 4 -qp_count 4 -mode warp -elems 1048576

# 8 QPs, block mode
bash examples/simt_udma_qp_demo/run.sh -pes 2 -qp_count 8 -mode block -elems 1048576

# Specify symmetric heap size (must accommodate three symmetric buffers: origin + res_put + res_get)
bash examples/simt_udma_qp_demo/run.sh -pes 2 -qp_count 4 -mode warp -heap_mb 2048
```

On success, each PE outputs:

```text
[PASS] warp-put pe=0 elements=1048576
[PASS] warp-get pe=0 elements=1048576
[PASS] block-put pe=0 elements=1048576
[PASS] block-get pe=0 elements=1048576
[INFO] mode=all qp_count=2 pe=0 result=PASS
```

On data validation failure, the program prints the first failing position, actual value, and expected value. `run.sh` waits for all PE processes and returns a non-zero status when any process fails.

### Runtime Parameters

`run.sh` supports the following parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `-pes` | `2` | Total number of PEs; must be at least 2. The script launches the same number of processes. |
| `-qp_count` | `2` | Number of QPs created for each remote PE. Range: `[1, 56]`. Must be identical on all PEs. |
| `-mode` | `all` | Execution mode: `warp`, `block`, or `all` (both modes run sequentially). |
| `-elems` | `1048576` | Number of bytes transferred by each PE. Must be at least 2 times the QP count (the UDMA engine requires each QP to transfer at least 2 bytes per request). Uneven partitioning is supported. |
| `-heap_mb` | `1024` | SHMEM symmetric heap size per PE in MiB. Must accommodate three symmetric buffers (origin + res_put + res_get). |
| `-first_npu` | `0` | PE i is bound to logical NPU `first_npu + i`. |
| `-ipport` | `tcp://127.0.0.1:8999` | Bootstrap initialization address. |

## Usage Constraints (Interface Contract Summary)

1. The QP count is set by the host via `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num)` before initialization and must be identical on all PEs.
2. At any given time, one `(pe, qp_idx)` may be driven by a single driver (warp/block: one thread group). Use distinct `qp_idx` values for multi-thread parallelism.
3. A `_nbi` return only indicates that the WQE has been submitted. The matching quiet must be called before reading or writing the relevant memory.
4. The warp/block variants are collective calls; every thread of the VF must invoke them. The block quiet variant internally uses `asc_syncthreads` to ensure all threads observe the completed state before returning; the warp quiet variant relies on warp lockstep with no block-level synchronization. The put/get warp/block variants are also collective calls; all threads of the VF must participate.
5. A single request transfers 2 B to 256 MB, and each QP must transfer at least 2 bytes per request.
6. This sample does not support Multi-link mode or PutSignal operations (these are capabilities of the SIMD version `udma_qp_demo`).
