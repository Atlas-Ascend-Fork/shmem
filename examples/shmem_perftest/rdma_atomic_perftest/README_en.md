# rdma_atomic_perftest

## Overview

`rdma_atomic_perftest` tests the performance of the SHMEM RDMA atomic operation APIs (`aclshmemx_roce_atomic_*`) and supports two test modes: whole-batch completion time (`batch`) and single-completion latency (`latency`).

The test uses two PEs, and each PE corresponds to one process and one NPU. PE0 initiates operations, and PE1 provides the symmetric HBM target memory. The device side uses one AIV core and the default single RDMA QP, and all atomic operations are completed through QP0.

## Test Modes

| Mode | Data Type | Execution Content of Each Formal Sample | Metric |
| --- | --- | --- | --- |
| `batch` | `uint64` | A single thread performs one atomic operation on each of N consecutive elements in sequence and waits for the whole batch to complete | Whole-batch completion time (us/batch), atomic operation rate (Mops/s) |
| `latency` | `uint32`, `uint64`, `int32`, `int64` | Repeatedly performs atomic operations on one scalar, and each next operation starts after the previous one completes | Average completion latency (us/op) |

The batch size indicates the number of atomic calls in a batch. By default, the sizes `1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096` are tested in sequence, 13 sizes in total. Each size is sampled 10 times independently, and **each formal sample executes only one batch**. For example, when the batch size is 4, a sample contains one operation on each of `target[0]` to `target[3]`, and timing stops immediately after the operations complete.

By default, each formal latency sample performs 1000 operations and is sampled 10 times independently. The average completion latency is obtained by dividing the total elapsed time by the number of operations. `--iterations` controls only the number of operations of latency and does not affect the formal sampling of batch.

### Supported Atomic Operations

All the following 13 operations support both batch and latency tests. Operations with `fetch`, as well as `swap` and `compare_swap`, return the old value before the operation; the other operations do not return the old value.

In the table, `K` indicates the number of times each target element is operated on within a formal sample: fixed to 1 for batch, and `--iterations` for latency. `ALL_ONES` is all 1s within the bit width of the selected type. Each sampling uses the initial values and parameters in the table. It also verifies the final target value and the old value returned by the last operation.

| `--op` | Initial Value | Operation Parameter | Final Target Value | Old Value Returned by the Last Operation | Underlying Operation Family |
| --- | --- | --- | --- | --- | --- |
| `inc` | 0 | Add 1 | K | — | FA |
| `fetch_inc` | 0 | Add 1 | K | K−1 | FA |
| `set` | 0 | value=1 | 1 | — | CAS |
| `add` | 0 | value=1 | K | — | FA |
| `fetch_add` | 0 | value=1 | K | K−1 | FA |
| `and` | ALL_ONES | value=ALL_ONES−1 | ALL_ONES−1 | — | CAS |
| `fetch_and` | ALL_ONES | value=ALL_ONES−1 | ALL_ONES−1 | ALL_ONES when K=1, otherwise ALL_ONES−1 | CAS |
| `or` | 0 | value=1 | 1 | — | CAS |
| `fetch_or` | 0 | value=1 | 1 | 0 when K=1, otherwise 1 | CAS |
| `xor` | 0 | value=1 | K%2 | — | FA |
| `fetch_xor` | 0 | value=1 | K%2 | (K−1)%2 | FA |
| `swap` | 0 | value=1 | 1 | 0 when K=1, otherwise 1 | CAS |
| `compare_swap` | 0 | compare=0, value=1 | 1 | 0 when K=1, otherwise 1 | CAS |

FA and CAS correspond to the `OP_ATOMIC_FA` and `OP_ATOMIC_CAS` operation families of the RDMA backend respectively. The specific semantics are jointly determined by the operands and masks. Batch `compare_swap` operates on each element only once, and all comparisons succeed. For latency, the first comparison on the same element succeeds, and the subsequent comparisons fail.

## Environment Requirements

The atomic operation test of this sample is currently enabled on the **Ascend950 + XSCALE (YunMai NIC) or HNS_1825 (1825 NIC)** configurations. For other backends, view `--help`. When you run the test, the system reports that the test case is not enabled and returns a non-zero status.

For RDMA network, driver, and runtime environment configuration, refer to [rdma_demo Environment Requirements](../../rdma_demo/README_en.md). Before running on the Ascend950 platform, set `IBV_EXTEND_DRIVERS` to point to the plug-in library of the corresponding NIC. For details, refer to [Environment Variable Description](../../rdma_demo/README_en.md#ibv_extend_drivers-environment-variable).

## Build Instructions

Run the following commands in the root directory of the SHMEM repository:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# YunMai NIC
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE
# 1825 NIC
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend HNS_1825
```

The build generates `build/bin/rdma_atomic_perftest`. The backend is selected at compilation time through `-rdma_backend`. After the backend configuration is changed, you must recompile. For other build options, refer to [Compilation and Build](../../../docs/compilation_build_guide_en.md#rdma-parameters).

## Usage

```bash
cd examples/shmem_perftest/rdma_atomic_perftest/
bash run.sh [options]
```

If `--pe-id` is not specified, the script starts PE0 and PE1, passes the same test parameters to both, and waits for the two processes to exit. If either PE returns a non-zero status, the configuration is judged as failed.

### Command-Line Options

| Parameter | Description | Default Value |
| --- | --- | --- |
| `--test <batch\|latency>` | Test mode | `latency` |
| `--op <op>` | Atomic operation. For available values, see "Supported Atomic Operations" | `inc` |
| `--data-type <type>` | Data type; batch supports only `uint64`, and latency supports the four integer types | `uint64` |
| `--iterations <n>` | Latency only: number of operations per formal sample, range 1~INT_MAX | `1000` |
| `--warmup-iterations <n>` | Number of warmup batches for batch, or number of warmup operations for latency; range 0~INT_MAX, where 0 means no warmup | `100` |
| `--repetitions <n>` | Number of formal sampling runs per test point, range 1~INT_MAX | `10` |
| `--min-batch <n>` | Minimum batch size, a positive integer | `1` |
| `--max-batch <n>` | Maximum batch size, which must not be smaller than min-batch; the required symmetric heap must be within the SHMEM capacity limit | `4096` |
| `--step-factor <n>` | Batch size growth factor, an integer greater than 1 | `2` |
| `--pes <n>` | Number of PEs, fixed to 2 | `2` |
| `--pe-id <id>` | PE ID of the current process, 0 or 1; when the script specifies this parameter, only the corresponding PE is started | Binary default `0` |
| `--ipport <tcp://ip:port>` | SHMEM initialization address | `tcp://127.0.0.1:8768` |
| `--gnpus <n>` | Number of NPUs in the local startup domain, a positive integer | `2` |
| `--fnpu <id>` | Starting local device ID, a non-negative integer | `0` |
| `--sync-id <id>` | RoCE synchronization event ID; the ID must be non-negative and supported by the target platform | `0` |
| `--all` | Runs all 65 test configurations, used only by `run.sh` | Disabled |
| `-h` / `--help` | Displays help information | — |

The actual device ID is `pe-id % gnpus + fnpu`. The batch size takes `min-batch × step-factor^k` in sequence until the next value exceeds `max-batch`. Setting the same min-batch and max-batch tests only one size. Latency always uses one scalar, and the batch range does not participate in the test. Invalid parameters are rejected before device initialization.

### Usage Examples

```bash
# uint64 add: batch size from 1 to 4096, with 10 samples per size
bash run.sh --test batch --op add

# uint64 fetch_add: each sample executes one batch of 4 operations, with 10 samples in total
bash run.sh --test batch --op fetch_add --min-batch 4 --max-batch 4 --repetitions 10

# uint64 fetch_add single-completion latency, with 1000 operations per sample
bash run.sh --test latency --op fetch_add --data-type uint64 --iterations 1000

# Full test matrix
bash run.sh --all --warmup-iterations 100 --repetitions 10
```

The full matrix contains 13 batch configurations (13 operations × uint64) and 52 latency configurations (13 operations × 4 types). Each batch configuration completes the size scan through the binary program. `--all` accepts additional sampling and device parameters, and `--iterations` affects only latency configurations. You cannot specify `--test`, `--op`, or `--data-type` at the same time. After any configuration fails, subsequent configurations stop.

To run a single configuration or `--all` across machines, specify `--pe-id 0` and `--pe-id 1` on the two host machines respectively. Use `--gnpus 1` and the same `--ipport tcp://<PE0 host IP>:8768` on both ends, and select the local device of each host machine through `--fnpu`. The script versions and test parameters on both ends must be consistent, and you must check the process exit status on each end separately. After one end fails, the other end may wait until the communication times out.

## Timing and Correctness Verification

Each test point performs warmup first and then samples independently according to `--repetitions`. The warmup phase repeatedly executes complete batches or latency operations; the next batch starts only after the previous batch completes, and the results are not counted as formal samples. The target values and result buffers are reinitialized before the warmup and each formal sampling.

| Mode and Operation | Timing Interval of a Formal Sample |
| --- | --- |
| Batch, non-fetch | Start timing → call atomic N times in sequence → one `aclshmemx_roce_quiet` → stop timing |
| Batch, fetch type | Start timing → call atomic N times in sequence, with each call completed before the API returns → stop timing |
| Latency, non-fetch | Start timing → repeat `atomic + aclshmemx_roce_quiet` iterations times in total → stop timing |
| Latency, fetch type | Start timing → repeat atomic iterations times in total, with each call completed before the API returns → stop timing |

The "fetch type" includes `fetch_*`, `swap`, and `compare_swap`. Batch uses the existing scalar atomic APIs, and the size indicates the number of calls in this batch; the wait inside the APIs still takes effect. The end-of-batch quiet of non-fetch operations allows multiple submissions to amortize the completion wait. However, the backend may still poll completions early when the queue space is insufficient, so the batch size is not equal to the actual number of requests in flight at the same time.

The device uses `AscendC::GetSystemCycle()` to collect the start and end cycles. Timing includes the loop, API calls, and the corresponding completion waits. PE synchronization, target initialization, result verification and transfer back, kernel launch, and host stream synchronization are all outside the timing interval.

After each formal sampling ends, PE1 scans all local target elements and transfers the verification result back to PE0. PE0 checks both the target result and the old value returned by the last fetch: batch corresponds to the last element of the batch, and latency corresponds to the last call. The verification is performed on the 32/64-bit bit pattern of the selected type, and the return value of each intermediate fetch is not checked.

When the target value, last returned value, operation count, and valid timing all meet the expectations, `Verified=true` is output. A verification failure is recorded as `Verified=false`, and error details are output to stderr; after the program completes the remaining sampling and synchronization, it returns a non-zero status.

## CSV Output

After the script runs, results are saved in the `output/` directory of this sample. Only PE0 writes the CSV; running the same test mode, operation, and type repeatedly overwrites the corresponding files.

### Per-Sampling Results

File name: `rdma_atomic_<test>_<op>_<data_type>_pe0.csv`.

| Column | Description |
| --- | --- |
| `Backend` | Transfer path, fixed to `ROCE` |
| `QPCount`, `PEs` | Number of QPs and number of PEs, fixed to 1 and 2 |
| `InitiatorPE`, `TargetPE` | Initiator and target, fixed to 0 and 1 |
| `Test`, `Op`, `DataType` | Test mode, atomic operation, and data type |
| `OperandBytes` | Number of bytes of a single operand |
| `BatchSize` | Number of atomic calls per batch; fixed to 1 for latency |
| `TotalOps` | Actual total number of operations of this formal sample; BatchSize for batch, and Iterations for latency |
| `Iterations` | Number of operations per element of this formal sample; fixed to 1 for batch, and the command-line configured value for latency |
| `WarmupIterations` | Number of warmup batches; each latency batch has only one operation |
| `Repetition` | Formal sampling sequence number, starting from 1 |
| `ElapsedCycles`, `ElapsedUs` | Total number of cycles and total elapsed time (us) of this sample |
| `BatchLatencyUs` | Whole-batch completion time (us/batch), filled only for batch |
| `AtomicRateMops` | Atomic operation rate (millions of operations per second), filled only for batch |
| `LatencyUs` | Average completion latency (us/op), filled only for latency |
| `Verified` | Joint verification result of the initiator and the target |

Metric fields that do not apply to the current mode are left empty. The metrics are calculated as follows:

```text
ElapsedUs       = ElapsedCycles / cycles_per_us
BatchLatencyUs  = ElapsedUs
AtomicRateMops  = BatchSize / ElapsedUs
LatencyUs       = ElapsedUs / Iterations
```

For Ascend950, `cycles_per_us` is 1000, and the startup log prints the actual conversion value used. A lower BatchLatencyUs indicates that the same number of operations completes faster, and a higher AtomicRateMops indicates a higher operation rate. Dividing the whole-batch elapsed time by BatchSize yields the time amortized to each operation, and this value includes the amortization effects of batch submission and completion wait.

### Summary Results

File name: `rdma_atomic_<test>_<op>_<data_type>_summary.csv`.

Rows are grouped by `Test`, `Op`, `DataType`, `BatchSize`, and `Metric`, and the same backend and PE metadata is retained. The two metrics of batch occupy one row each, and latency occupies one row.

| Column | Description |
| --- | --- |
| `Metric` | `BatchLatencyUs`, `AtomicRateMops`, or `LatencyUs` |
| `Samples` | Number of formal samples that pass verification |
| `Min`, `Max` | Minimum and maximum sample values |
| `Median` | Median; for an even number of samples, the average of the middle two values |
| `P95` | The `ceil(0.95 × Samples)`-th value after sorting, with sequence numbers starting from 1 |

Only samples with `Verified=true` participate in the summary. When no valid samples exist, `Samples=0` and the statistics are left empty. With the default 10 samples, P95 equals Max according to the preceding algorithm. The elapsed-time quantiles of batch reflect the fluctuation of the whole-batch elapsed time; the quantiles of latency reflect the fluctuation of multiple average latencies and cannot be used as the tail latency of a single request.

### `--all` Summary Tables

After `bash run.sh --all` successfully completes all configurations, two summary tables are generated in the `output/` directory on the host machine where PE0 resides. Each row corresponds to one atomic operation, arranged in the order of the preceding 13 operations, and values retain three decimal places.

| File | Column | Value Source |
| --- | --- | --- |
| `rdma_atomic_all_latency_us.csv` | `int32`, `uint32`, `int64`, `uint64` | `LatencyUs` / `Median` of each latency sub-item, in us/op |
| `rdma_atomic_all_mops.csv` | BatchSize values actually tested in this round, from small to large | `AtomicRateMops` / `Median` of each batch sub-item, with type uint64 and unit Mops/s |

## Known Constraints

1. Fixed 2 PEs, 1 AIV core, and 1 QP, with the communication direction PE0→PE1; parameters for bidirectional, multi-PE, multi-AIV, or multi-QP modes are not provided.
2. Only symmetric HBM memory allocated through `aclshmem_malloc` is tested; DRAM on the host side is not supported.
3. Batch calls the existing APIs element by element over consecutive elements, and WQE aggregation submission is not enabled. The results of fetch-type APIs include the overhead of the per-call completion wait.
4. When comparing test results, keep the hardware, CANN and driver versions, network topology, and test parameters consistent.
