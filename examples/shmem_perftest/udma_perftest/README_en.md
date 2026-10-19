# udma_perftest

## Overview

`udma_perftest` is a parameterized sample for **measuring SHMEM UDMA low-level API performance**, parallel to `mte_perftest` in the same directory for the MTE engine. It collects performance data through built-in profiling, covers `aclshmemx_udma_put_nbi`, `aclshmemx_udma_get_nbi`, and `aclshmemx_udma_put_signal_nbi`, and supports different payload sizes, batch submission, and QP configurations. **Results from this script are for reference only; performance in the actual workload is authoritative.**

## Test Scope

The sample measures the following UDMA data-transfer operations:

1. **Unidirectional Put** (`put`): Only the PE specified by `SHMEM_CYCLE_PROF_PE` calls `aclshmemx_udma_put_nbi` to transfer data to the peer PE.
2. **Bidirectional Put** (`bi_put`): Both PEs call Put concurrently and transfer data to each other.
3. **Unidirectional Get** (`get`): Only the profiling PE calls `aclshmemx_udma_get_nbi` to pull data from the peer PE.
4. **Bidirectional Get** (`bi_get`): Both PEs call Get concurrently and pull data from each other.
5. **Put + Signal** (`put_signal`): Only the profiling PE calls `aclshmemx_udma_put_signal_nbi` to transfer data and then write a remote signal. The signal value is verified after the test.

## Differences from `mte_perftest`

| Aspect | `mte_perftest` (MTE) | `udma_perftest` (UDMA) |
| --- | --- | --- |
| Engine | Default MTE | Explicit `ACLSHMEM_DATA_OP_UDMA` |
| Multi-core concurrency | Multiple cores for the same peer; data is split across 32 cores by default | `block_dim=qp_count`; each block owns one QP and independent data |
| `-b/--block-size`, `--block-range` | Controls core count | `-b` also controls QP count; `--block-range` is currently not used by UDMA tests |
| UB buffer | Required by MTE and affects transfer | This perftest explicitly tests low-level UDMA APIs. `--ub-size` is used as a low-level UDMA UB argument and is written to CSV. High-level UDMA RMA uses MTE staging and requires at least 128 B UB. |
| Test modes | put / bi_put / get / bi_get | put / bi_put / get / bi_get / **put_signal** |
| SoC restriction | General | **Ascend 950 only**; the device kernel aborts on non-950 SoCs |
| CSV name | `<test>_<dtype>_<pe>.csv` | `udma_<metric>_<test>_<dtype>[_qpN]_<pe>.csv` |

## Build

UDMA is available only on Ascend 950 and requires CANN 9.1.0 or later with the HCOMM resource APIs. For installation requirements and CANN resource APIs, see [UDMA Sample Version and Platform Support](../../udma_demo/README.md#版本和平台支持说明).

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -soc_type Ascend950
```

## Usage

### Basic Usage

```bash
cd examples/shmem_perftest/udma_perftest/
./run.sh [options]
```

### Multi-node Execution

Run `run.sh` once on each host. `-fpe` specifies the first global PE on the current host and `-ipport` must point to the host that owns PE 0. The following example uses 2 hosts × 8 NPUs = 16 PEs, where node0 is `192.168.1.10`:

```bash
# node0: global PEs 0 to 7
bash run.sh -t bi_put -pes 16 -gnpus 8 -fpe 0 -fnpu 0 -ipport tcp://192.168.1.10:8899
# node1: global PEs 8 to 15
bash run.sh -t bi_put -pes 16 -gnpus 8 -fpe 8 -fnpu 0 -ipport tcp://192.168.1.10:8899
```

> For multi-node execution, `bi_put` or `bi_get` is recommended because ring mode gives every PE both send and receive traffic.

### Command-line Options

| Option | Short form | Description | Default |
| --- | --- | --- | --- |
| `--test-type <type>` | `-t <type>` | Test type: put / bi_put / get / bi_get / put_signal / all | `put` |
| `--datatype <type>` | `-d <type>` | Data type: float / int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64 / char / all | `float` |
| `--block-size <size>` | `-b <size>` | Number of UDMA blocks and QPs. One block owns one QP. Range: `[1, 56]`. | 1 |
| `--block-range <min> <max>` | - | Accepted for compatibility but not used by UDMA tests. Use `-b` to set the block/QP count. | 1 1 |
| `--exponent <exponent>` | `-e <exponent>` | Per-QP payload exponent (`2^exponent` bytes). Total payload is this value times QP count for multi-QP runs. | - |
| `--exponent-range <min> <max>` | - | Range of per-QP payload exponents. | 3 17 |
| `--loop-count <count>` | - | Iteration count. | 1000 |
| `--ub-size <size>` | - | UB workspace size in KiB (`1 KiB = 1024 B`). The kernel uses this value to allocate its local UB workspace; low-level UDMA APIs receive the workspace address rather than this parameter value. CSV records the value in KiB. High-level UDMA RMA MTE staging requires at least 128 B UB. | 16 |
| `--metric <bw\|lat>` | - | Metric: `bw` for bandwidth, `lat` for API issue latency while traversing all API forms. See below. | bw |
| `--batch <n>` | - | For `bw`, number of operations before waiting for completion. For `lat`, aggregation window size. See below. | `0` |
| `-pes <size>` | - | Number of PEs. | 2 |
| `-ipport <ip:port>` | - | Communication address. | tcp://127.0.0.1:8768 |
| `-gnpus <num>` | - | Number of NPUs. | 2 |
| `-fnpu <id>` | - | First NPU ID. | 0 |
| `-fpe <id>` | - | First PE ID. | 0 |
| `-a/--analyse <mode>` | - | Analysis mode: none / plot / md. `perf_data_process.py` generates plots or reports for plot/md. In latency mode it produces latency curves and tables for four API forms. | none |

### DRAM / D2H Memory Constraint

This sample covers HBM (`DEVICE_SIDE`) memory paths only. It **does not support D2H / `HOST_SIDE` (DRAM)** because the UDMA engine currently provides no RMA path to host DRAM. To measure DRAM, use [rma_d2h_demo](../../rma_d2h_demo/README.md).

The default local memory size is 1 GB. For large payloads, the program automatically increases `local_mem_size` up to 40 GB.

### Metric Definitions

UDMA uses asynchronous NBI APIs. The benchmark measures `put_nbi/get_nbi` submission separately from `quiet` completion waiting. Both metrics use **one profiling window covering `loop_count` submissions**. Built-in profiling includes `pipe_barrier` overhead, so profiling is placed outside the loop and total window time is divided by `loop_count`. The metrics differ in whether `quiet` is inside the window:

- **`--metric bw` (default)**: `prof_start → loop(put_nbi) → quiet → prof_end`; the window includes `quiet`.
  - `Bandwidth/GB/s` = `datasize × qp_count / (window_us / loop_count)`, where `datasize` is the payload of one QP.
  - `CoreMaxTime/SingleCoreTime` = `window_us / loop_count`.
  - Applies to `put / bi_put / get / bi_get / put_signal`.
- **`--metric lat`**: Traverses four API forms to measure issue latency without completion waiting. See [Latency API Traversal](#latency-api-traversal---metric-lat). It supports only `-t put` and one QP.

CSV filenames include the metric prefix and append a QP-count suffix for multi-QP runs: `output/udma_<metric>_<test_type>_<dtype>[_qpN]_<pe>.csv`.

### Latency API Traversal (`--metric lat`)

`--metric lat` measures API issue latency without completion waiting and supports only `-t put` with one QP. One execution automatically traverses four API forms; `--batch` is not used to select a form:

| API form | Description | CSV column |
| --- | --- | --- |
| `plain` | Ordinary non-aggregation API `aclshmemx_udma_put_nbi`; every call submits immediately. | `PlainLat/us` |
| `qp` | QP-specific API `aclshmemx_udma_qp_put_nbi` with `qp_idx=0`; every call submits immediately. | `QpLat/us` |
| `defer` | Aggregation defer API; accumulates a batch without submitting it. | `DeferLat/us` |
| `submit` | Aggregation submit API; submits the accumulated batch. | `SubmitLat/us` |

Timing rules, all amortized to per-call issue latency with `quiet` excluded:

- `plain` / `qp` / `submit`: Call continuously for `loop_count` iterations, time the whole segment, then divide by `loop_count`.
- `defer`: Time N defer calls in one window and divide by N; issue one final submit to flush the batch, outside the timed window.
- Each form has independent warmup. Submit warmup uses an independent submit state for every call.
- `--batch N` is the aggregation window size, defaulting to 100. `run.sh` increases UB automatically when it is insufficient.

For example, `./run.sh -t put --metric lat` measures all four forms in one execution; `./run.sh -t put --metric lat --batch 200` sets the aggregation window to 200.

### Batch Submission (`--metric bw` Only)

The default `--metric bw` path is **fully asynchronous**: it submits `loop_count` ordinary `*_nbi` operations and calls `quiet` only once at the end of the window. `--batch <N>` controls both batch boundaries and submission mode:

- `--batch 0` (default): Fully asynchronous ordinary NBI; one final `quiet`, representing steady-state throughput.
- `--batch 1`: Call `quiet` immediately after every ordinary NBI, equivalent to synchronous submission and useful for end-to-end submit-plus-completion cost.
- `--batch N` (`N > 1`): The first N-1 operations in each batch use `defer`; the last uses `submit`, followed by `quiet`. If `loop_count` is not divisible by N, the remainder batch submits its actual operation count.

The profiling window is still collected once, so total window time divided by `loop_count` is the average per-operation cost including batched `quiet`. The `lat` path does not use BW batch semantics; `--batch` is only its aggregation window.

For `put / bi_put / get / bi_get`, `batch > 1` in `--metric bw` uses aggregation actions. `put_signal` always uses ordinary NBI but supports grouping followed by `quiet` according to batch boundaries.

### Examples

```bash
# Unidirectional PUT bandwidth, float, exponents 8 to 20.
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# Latency: traverse ordinary/QP/aggregation APIs once. Unsupported forms display `-`.
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --metric lat

# Latency with aggregation window 200.
./run.sh -t put -d float -e 10 --loop-count 2000 --metric lat --batch 200

# Bidirectional GET bandwidth, int32.
./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000

# Put-signal bandwidth.
./run.sh -t put_signal -d float -e 14 --loop-count 500

# Five modes × float bandwidth.
./run.sh -t all -d float --exponent-range 8 20 --loop-count 1000

# One mode × all data types.
./run.sh -t put -d all --exponent-range 8 20

# Synchronous submission (`batch=1`): quiet after every NBI.
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 1

# Semi-asynchronous: quiet after every 16 NBIs.
./run.sh -t get -d float --exponent-range 8 20 --loop-count 1000 --batch 16

# One-QP aggregation submission (`batch=8`).
./run.sh -t put -d float -e 14 --batch 8

# Four concurrent QPs; block count equals QP count.
./run.sh -t bi_put -d float -e 16 -b 4

# Four QPs with aggregation submission; every QP uses batch 8.
./run.sh -t bi_get -d float -e 16 -b 4 --batch 8
```

## Put-signal Behavior

The perftest manages signals automatically in `put_signal` mode:

- It allocates a symmetric signal buffer with `aclshmem_malloc(n_pes * sizeof(uint64_t))` and initializes it to zero.
- For every data point, it executes `warmup + loop_count` iterations and calls `aclshmemx_udma_put_signal_nbi(..., signal_base + i, peer_pe)`. Signal values increase linearly to avoid stale-data interference.
- After a data point, the host reads the peer signal slot and verifies `signal_base + (warmup + loop_count - 1)`.
- A validation failure prints an ERROR but does not stop subsequent data points.

## CSV Output

CSV columns align with the MTE version so `examples/utils/perf_data_process.py` can be reused. Both metrics share the first four columns, then use different metric columns:

| Column | Metric | Description |
| --- | --- | --- |
| `DataSize/B` | Common | Payload of one QP per transfer in bytes. Total payload is `DataSize × qp-count` in multi-QP mode. |
| `Npus` | Common | Number of NPUs. |
| `Blocks` | Common | `qp-count` in multi-QP mode and 1 otherwise; always 1 for `lat`. |
| `UBsize/KiB` | Common | UB size in KiB. |
| `Bandwidth/GB/s` | `bw` | Decimal GB/s, calculated from total traffic across all QPs. |
| `CoreMaxTime/us` | `bw` | Per-iteration time in microseconds. |
| `PlainLat/us` | `lat` | Issue latency of ordinary non-aggregation API in microseconds; `-` when the window is not applicable. |
| `QpLat/us` | `lat` | Issue latency of QP-specific API in microseconds. |
| `DeferLat/us` | `lat` | Per-call issue latency of aggregation defer API in microseconds. |
| `SubmitLat/us` | `lat` | Per-call issue latency of aggregation submit API in microseconds. |

For `put/bi_put/get/bi_get`, `Blocks` equals `--block-size`; `put_signal` remains one block. **`DataSize/B` is the payload of one QP per transfer, namely `2^e` bytes specified by `-e`; total payload is `DataSize × Blocks` for multiple QPs.** Bandwidth is calculated from total traffic and can be compared with single-QP results. When comparing engines with the same `-e`, include `Blocks` in the comparison. Filenames use `output/udma_<metric>_<test_type>_<dtype>[_qpN]_<pe>.csv`, for example `udma_bw_put_float_0.csv` or `udma_bw_put_float_qp4_0.csv`.

## Output Example

```bash
[INFO] udma_perftest start, pe=0, t=put, d=float, exp=10-10, loop=100, ub=16KiB, metric=lat, batch=100
pe: 0 size(per QP): 1024B, total: 1024B frame_id: 0
[LAT][plain] PE0: 0.85 us
[LAT][qp] PE0: 0.92 us
[LAT][defer] PE0: 0.31 us
[LAT][submit] PE0: 1.10 us
[Verification] put: checking...
[Verification] SUCCESS
[SUCCESS] udma_perftest done in pe 0
```

## Known Constraints

1. `include/device/gm2gm/engine/shmem_device_udma.h` states that concurrent RMA/AMO operations to the same PE are not supported. Single-QP mode keeps `block_dim=1`; multi-QP mode gives every block an explicit QP to prevent multiple blocks from contending for the same SQ.
2. UDMA aggregation actions require `PIPE_MTE3`. When `batch > 1`, the WQE count in each batch must be less than 4096 (256 KiB UB limit and 64 B per WQE), and UB scratch must be at least `64 * batch` bytes. QP-specific APIs used by multi-QP mode require `ACLSHMEM_RELAY_SUPPORT=OFF` and support direct UDMA builds only.
3. UDMA is enabled at build time only on Ascend 950. On other SoCs, the kernel reports an error and exits.
4. **D2H / `HOST_SIDE` (DRAM) is not supported.** The UDMA engine currently provides an RMA path only for HBM.
5. Atomic operations, such as `aclshmemx_udma_atomic_add`, are outside the scope of this perftest.
6. High-level UDMA RMA submits WQEs through default `PIPE_MTE3` staging with `offset = 189 * 1024`, `ub_size = 128` bytes, and `sync_id = 0`. If `aclshmemx_set_udma_config` changes this configuration, `ub_size` must be at least 128 bytes. This low-level perftest path continues to use its explicit local UB arguments.
