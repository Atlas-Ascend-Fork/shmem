# rdma_perftest

## Overview

`rdma_perftest` tests the transfer performance of the shmem RDMA low-level APIs (`aclshmemx_roce_put_nbi` / `aclshmemx_roce_get_nbi`) under different data sizes. It collects performance data through the [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) macros and supports two metrics: bandwidth (`bw`) and API issue latency (`lat`).

> The test results are for reference only; performance in actual scenarios is authoritative.

## Test Modes

| Mode | Description |
|------|------|
| `put` | One-way put: PE0 calls RDMA put to transfer data to PE1 |
| `bi_put` | Bidirectional put: two PEs put data to each other at the same time |
| `get` | One-way get: PE0 calls RDMA get to pull data from PE1 |
| `bi_get` | Bidirectional get: two PEs get data from each other at the same time |

## Differences from mte_perftest and udma_perftest

| Dimension | `mte_perftest` | `udma_perftest` | `rdma_perftest` |
|------|---------------|-----------------|-----------------|
| Engine | Default MTE | Explicit `ACLSHMEM_DATA_OP_UDMA` | RDMA engine |
| Concurrency | Multi-core on the same peer (data split across 32 cores by default) | Forced single core (UDMA does not allow concurrency on the same peer) | Single QP; XSCALE and HNS_1825 support multi-QP parallelism; IN_DIE/XSCALE support aggregation submission |
| `-b/--block-size` | Controls the number of cores | Compatibility input, forced to 1 | Compatibility input; any value has no effect, and the actual parallelism is determined by `-q/--qp` |
| UB buffer | Required for MTE | Required for UDMA | Required, at least 192B, 192B by default |
| Test modes | put / bi_put / get / bi_get | put / bi_put / get / bi_get / **put_signal** | put / bi_put / get / bi_get |
| SoC restriction | General | Ascend950 only | Ascend950 (XSCALE/HNS_1825 backend required) or A2/A3 |
| CSV file name | `<test>_<dtype>_<pe>.csv` | `udma_<test>_<dtype>_<pe>.csv` | `rdma_<metric>_<test>_<dtype>_<pe>.csv` |

## Environment Requirements

The environment requirements are the same as those in [rdma_demo](../../rdma_demo/README_en.md).

## Build Instructions

The RDMA function requires the `-enable_rdma` parameter to be enabled at compilation time, and the backend must be configured according to the SoC type. For details, refer to [Compilation and Build - RDMA Parameters](../../../docs/compilation_build_guide_en.md#rdma-parameters).

### RDMA Backend Capabilities

The following table describes whether the RDMA backend APIs themselves have the corresponding capabilities. It does not indicate that `rdma_perftest` has covered these capabilities.

| Backend Capability | IN_DIE | XSCALE | HNS_1825 |
|----------|--------|--------|----------|
| Applicable platform | A2/A3 | Ascend950 (YunMai NIC) | Ascend950 (1825 NIC) |
| Basic Put/Get NBI | Supported | Supported | Supported |
| Basic quiet and synchronization | Supported | Supported | Supported |
| Ordinary single-QP aggregation | Supported | Supported | Not supported |
| Multi-QP | Not supported | Supported | Supported |
| Multi-QP aggregation | Not supported | Supported | Not supported |
| RDMA atomic operations | Not supported | Supported | Supported |

### Capabilities Supported by rdma_perftest

The following table describes the functions actually opened by this test on each backend. If a backend has a capability but this test does not cover it, the capability is still marked as "not supported".

| Function of This Test | IN_DIE | XSCALE | HNS_1825 |
|------------|--------|--------|----------|
| Put/Get and one-way/bidirectional modes | Supported | Supported | Supported |
| Single-QP bandwidth test | Supported | Supported | Supported |
| Single-QP latency test | Supported | Supported | Supported |
| `batch` grouping and completion wait of the bandwidth test | Supported | Supported | Supported |
| Automatic traversal of API forms in the latency test | `plain` + `defer` + `submit` | `plain` + `qp` + `defer` + `submit` (`qp_num <= 2`) | `plain` + `qp` |
| Explicit QP count and QP index | Not supported; `run.sh` reports an error and exits | Supported, QP count range 1~56 | Supported, QP count range 1~56 |
| Multi-QP parallel bandwidth test | Not supported | Supported, QP count range 2~56 | Supported, QP count range 2~56 |
| Fixed-QP diagnostic mode | Not supported | Supported | Supported |
| Aggregation submission | Supported | Supported | Not supported |

## Usage

```bash
cd examples/shmem_perftest/rdma_perftest/
bash run.sh [options]
```

> On the Ascend950 platform, you need to set the `IBV_EXTEND_DRIVERS` environment variable. For details, refer to [Environment Variable Description](../../rdma_demo/README_en.md#ibv_extend_drivers-environment-variable).

### Command-Line Options

| Parameter | Abbreviation | Description | Default Value |
|------|------|------|--------|
| `--test-type <type>` | `-t` | Test type: `put` / `bi_put` / `get` / `bi_get` / `all` | `put` |
| `--datatype <type>` | `-d` | Data type: `float` / `int8` / `int16` / `int32` / `int64` / `uint8` / `uint16` / `uint32` / `uint64` / `char` / `all` | `float` |
| `--exponent <n>` | `-e` | Data size exponent (2^n bytes) | - |
| `--exponent-range <min> <max>` | - | Data size exponent range | `3 17` |
| `--loop-count <n>` | - | Number of loops | `1000` |
| `--metric <bw\|lat>` | - | Performance metric: `bw`=bandwidth, `lat`=API issue latency | `bw` |
| `--ub-size <n>` | - | UB size (B), automatically 64B aligned, range 192~131136 | `192` |
| `--batch <n>` | - | `bw`: number of operations to launch consecutively before waiting for completion; `lat`: aggregation window size (see the description below) | `0` |
| `-q/--qp/--qp-count <n>` | `-q` | QP count, that is, the number of queue pairs between two PEs, range 1~56 | `1` |
| `-i/--qp-index <n>` | `-i` | QP index fixed for use; `-1` means automatic allocation in multi-QP parallel mode | `-1` |
| `--sync-id <id>` | - | Synchronization ID explicitly passed to Put/Get/Quiet | `0` |
| `-b/--block-size <n>` | `-b` | Compatibility input; any value has no effect (only a WARN message is printed), and the actual parallelism is determined by `--qp` | `1` |
| `--block-range <min> <max>` | - | Compatibility input, same as above | `1 1` |
| `-pes <n>` | - | Number of PEs (forced to 2) | `2` |
| `-ipport <ip:port>` | - | Communication address | `tcp://127.0.0.1:8768` |
| `-gnpus <n>` | - | Number of NPUs | `2` |
| `-fnpu <id>` | - | First NPU ID | `0` |
| `-a/--analyse <mode>` | - | Analysis mode: `none` / `plot` / `md` | `none` |

### HBM and Symmetric Memory Constraints

- Only the HBM (`DEVICE_SIDE`) memory path is tested; **D2H / `HOST_SIDE` (DRAM) is not supported**.
- Allocate input/output buffers through `aclshmem_malloc`; both the local and remote operands of Put/Get must point to symmetric memory.
- The local memory is 1 GB by default and is automatically increased when the data size is large (up to 40 GB).

### Key Parameter Description

#### What Is a QP

A QP (Queue Pair) is the basic unit of RDMA communication, and each QP maintains an independent send queue and receive queue. `-q/--qp/--qp-count` specifies the number of QPs established between two PEs (range 1~56), and `-i/--qp-index` specifies which QP is used during the test.

#### Single-QP Mode (Default, Without `-q` or `-i`)

When no QP parameter is passed, the two PEs use 1 QP to complete all transfers. Each test performs `loop-count` data transfers, `DataSize/B` is the data size of each transfer, and the bandwidth = total data size / total elapsed time.

In bandwidth tests, IN_DIE automatically performs aggregation submission when a single QP is used and the message is strictly smaller than 32 KiB; XSCALE automatically performs aggregation submission when `qp_num <= 2` and the message is strictly smaller than 64 KiB; HNS_1825 always uses the ordinary QP-specific NBI path.

#### Multi-QP Parallel Mode (`--qp N --qp-index -1`)

N QPs transfer data in parallel to improve the total throughput. Each QP uses an independent data area and completes `loop-count` transfers respectively. `DataSize/B` is the data size of each transfer of a **single QP**, and the total data size of each test round is `DataSize × N`. The bandwidth is calculated based on the common time window in which all QPs complete these data.

- This mode must be used together with `--metric bw`. `-q N` specifies the QP count (range 1~56), and `-i -1` means that an independent data area is automatically allocated for each QP.
- For example, with `-q 4 -i -1`, `DataSize=4MB`, and `loop-count=1000`: the 4 QPs each submit 1000 4MB transfers, the total data size of a single concurrency is 16MB, and the cumulative total data size is 16GB.
- XSCALE performs aggregation submission when `N <= 2` and the message is strictly smaller than 64 KiB. HNS_1825 uses the ordinary QP-specific NBI API to submit immediately.

> **YunMai (XSCALE) and 1825 (HNS_1825)** both support multiple QPs. Other backends are intercepted by `run.sh`, which reports an error and exits.

#### Single-QP Diagnostic Mode (`--qp N --qp-index K`)

The K-th QP (0 ≤ K < N) is fixed to complete all tests, and the other QPs do not participate. This mode is used to compare the behavior of a single QP or troubleshoot problems of a specific QP.

- `-q N` declares the QP pool size, and `-i K` fixedly selects the K-th QP. `-i` must satisfy `-1 ≤ K < N`; otherwise, `run.sh` reports an error and exits.
- The latency test (`--metric lat`) requires `qp_num=1` and does not support `qp_num>1`; timing covers only the transfer time of the unique QP.
- In bandwidth tests, XSCALE performs aggregation submission when `N <= 2` and the message is strictly smaller than 64 KiB. HNS_1825 uses the ordinary QP-specific NBI API to submit immediately.

> **YunMai (XSCALE) and 1825 (HNS_1825)** both support the fixed-QP diagnostic mode. Other backends are intercepted by `run.sh`, which reports an error and exits.

#### Batch and Completion Semantics (`--batch`)

`--batch N` controls how operations are grouped on the bandwidth path (`--metric bw`): after N non-blocking operations are launched consecutively, the system waits for the whole group to complete before starting the next group.

- Completion semantics: after each group of operations completes, **the source buffer of Put can be safely reused** (the data has been received by the remote end), and **the data in the target buffer of Get is ready to read**.
- When you run the binary directly, `batch=0` or `batch > loop_count` is processed as `loop_count`, that is, a single unified wait is performed after all operations complete.

#### Latency API Traversal (`--metric lat`)

`--metric lat` measures the API issue latency (excluding the completion wait). One latency test automatically traverses **all API forms** supported by the current backend, and you do not need to select forms through `--batch`:

| API Form | Description | IN_DIE | XSCALE | HNS_1825 |
|----------|------|--------|--------|----------|
| `plain` | Ordinary non-aggregation API; each call to `put/get_nbi` submits immediately | Supported | Supported | Supported |
| `qp` | QP-specific API; each call goes through the QP-specific NBI API | Not supported | Supported | Supported |
| `defer` | Aggregation defer API (ordinary defer for IN_DIE, QP defer for XSCALE); accumulates a batch without submitting it | Supported | Supported (`qp_num <= 2`) | Not supported |
| `submit` | Aggregation submit API (ordinary submit for IN_DIE, QP submit for XSCALE); submits the accumulated batch | Supported | Supported (`qp_num <= 2`) | Not supported |

- Each form performs warmup and timing independently, and the execution order is `submit`, ordinary APIs, QP APIs, and `defer`. Results are still written to the `PlainLat/us`, `QpLat/us`, `DeferLat/us`, and `SubmitLat/us` columns of the CSV respectively. Forms not supported by the backend display `-`.
- The `submit` form uses the same timing method as the ordinary APIs: submit is called `loop_count` times consecutively, the start and end times of the whole segment are recorded, and the result is finally divided by `loop_count`; quiet is not included in timing. The `defer` form continuously times the N defer calls in the window as a whole and divides the result by N (the last submit completes and submits the aggregation group, outside timing). Both results are the API issue latencies amortized to a single call.
- The warmup of submit also uses an independent submit state for each call, preventing the queue state of one aggregation warmup from being different from that of the formal submit.
- `--batch N` serves only as the aggregation window size (100 automatically by default) and is not subject to the message size threshold of the bandwidth mode. The defer window contains N defer calls and one submit that carries its own WQE, N+1 entries in total; the warmup contains 100 entries in total. `run.sh` and the binary automatically increase the UB based on `64 + 128 × max(100, N+1)`: 12992B when N=100 and 25792B when N=200. The upper limit of the latency window is 1022, and the window automatically shrinks when the limit is exceeded, ensuring that a complete batch has at most 1023 entries, which is smaller than the SQ depth 1024; the batch count of the bandwidth mode does not append this WQE.

Example: `./run.sh -t put --metric lat` measures the latencies of the four API forms (submit, ordinary, QP, and defer) at one time; `./run.sh -t put --metric lat --batch 200` changes the aggregation window to 200.

#### Multi-QP Backend Identification

`run.sh` identifies XSCALE and HNS_1825 environments through the `IBV_EXTEND_DRIVERS` environment variable or the output of `ibv_devinfo`. Only these two backends accept an explicit QP count and QP index; other backends report an error and exit when explicit QP parameters are passed in. IN_DIE and XSCALE adjust the UB and batch parameters according to the aggregation conditions.

Bandwidth boundaries: for 16/32/64 KiB, IN_DIE with a single QP uses aggregation/ordinary/ordinary respectively, and XSCALE (`qp_num <= 2`) uses aggregation/aggregation/ordinary respectively.

`run.sh` automatically corrects the batch and UB on the BW aggregation path, with a capacity of at least `64 + 128 × max(100, batch_size)` bytes, covering the warmup and formal batches of Put/Get. When you start the binary directly, pass `--ub-size` that satisfies this convention and a valid `--batch`.

### Usage Examples

```bash
# Default compatibility mode: one-way PUT bandwidth test
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# Default compatibility mode: latency test that traverses ordinary/QP/aggregation APIs at one time (forms not supported by the backend display -)
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --metric lat

# Batch completion: wait for completion after every 100 operations, and then start the next group
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 100

# Multi-QP bandwidth test (XSCALE and HNS_1825 only): 2 QPs in parallel; XSCALE performs aggregation submission for small messages, and HNS_1825 submits NBI call by call
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 -q 2 -i -1 --metric bw --batch 100

# Single-QP diagnostic mode: tests only the QP with index 0, and waits for completion after every 100 operations
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 -q 4 -i 0 --metric bw --batch 100

# Bidirectional GET
./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000
```

## CSV Output

CSV files are output to the `output/` directory, named in the format of `rdma_<metric>_<test_type>_<dtype>_<pe>.csv`. The two metrics share the first 4 columns, and the subsequent columns differ:

| Column | Metric | Description |
|----|------|------|
| `DataSize/B` | General | Data size of each transfer of a single QP (byte); in multi-QP parallel mode, the total data size is `DataSize × qp-count` |
| `Npus` | General | Number of NPUs |
| `Blocks` | General | `qp-count` in multi-QP parallel mode, and 1 in other modes |
| `UBsize/KiB` | General | UB size (KiB) |
| `Bandwidth/GB/s` | `bw` | Bandwidth (GB/s, decimal) |
| `Bandwidth/GiB/s` | `bw` | Bandwidth (GiB/s, binary) |
| `CoreMaxTime/us` | `bw` | Time elapsed by a single iteration (us) |
| `PlainLat/us` | `lat` | Issue latency of the ordinary non-aggregation API (us); `-` when the backend does not support it |
| `QpLat/us` | `lat` | Issue latency of the QP-specific API (us); `-` when the backend does not support it |
| `DeferLat/us` | `lat` | Issue latency of the aggregation defer API amortized to a single call (us); `-` when the backend does not support it |
| `SubmitLat/us` | `lat` | Issue latency of the aggregation submit API amortized to a single call (us); `-` when the backend does not support it |

## Known Constraints

1. **Concurrency restriction**: The RDMA specification does not allow concurrent RMA/AMO operations on the same PE. The multi-QP parallel mode uses independent QPs to avoid this restriction; the single-QP diagnostic mode and the latency test use only a single QP.
2. **Compilation dependency**: `-enable_rdma` must be enabled; the Ascend950 platform additionally requires `-soc_type Ascend950` and `-rdma_backend XSCALE` (or `HNS_1825`).
3. **D2H is not supported**: The RDMA engine tests only HBM, and DRAM on the host side is not supported.
4. **Atomic operations are not supported**.
5. **put_signal is not supported**: The RDMA engine has no `aclshmemx_roce_put_signal_nbi` API.
