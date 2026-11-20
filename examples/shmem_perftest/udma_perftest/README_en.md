# udma_perftest

[中文](README.md)

## Overview

`udma_perftest` measures the low-level SHMEM UDMA interfaces `aclshmemx_udma_put_nbi`, `aclshmemx_udma_get_nbi`, and `aclshmemx_udma_put_signal_nbi`. It uses `SHMEMI_PROF_START/END` from [shmemi_prof.h](../../../src/device/utils/prof/shmemi_prof.h) and supports different transfer sizes, batches, and QP counts. Results are for reference; measure performance in the intended application.

The modes are `put`, `bi_put`, `get`, `bi_get`, and `put_signal`. Unidirectional modes submit from `SHMEM_CYCLE_PROF_PE`; bidirectional modes submit from both PEs. PutSignal also verifies the remote signal after the test.

## Differences from mte_perftest

| Item | MTE | UDMA |
| --- | --- | --- |
| Engine | Default MTE | Explicit `ACLSHMEM_DATA_OP_UDMA` |
| Concurrency | Multiple cores partition a peer's data | `block_dim=qp_count`; each block owns a QP and separate data |
| Block options | Select core count | `-b` selects QP count; `--block-range` is parsed but unused |
| UB | Required by MTE | Passed to low-level UDMA operations and recorded in CSV; high-level UDMA RMA staging requires at least 128 B |
| Modes | put / bi_put / get / bi_get | Also supports put_signal |
| Platform | General | Ascend950 only; other SOCs abort in the kernel |
| CSV filename | `<test>_<dtype>_<pe>.csv` | `udma_<metric>_<test>_<dtype>[_qpN]_<pe>.csv` |

## Build

UDMA requires Ascend950 and CANN 9.1.0 or later with the HCOMM resource interfaces. See the [UDMA example requirements](../../udma_demo/README_en.md) for installation details.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -soc_type Ascend950
```

## Usage

```bash
cd examples/shmem_perftest/udma_perftest/
./run.sh [options]
```

### Arguments

| Argument | Short form | Description | Default |
| --- | --- | --- | --- |
| `--test-type <type>` | `-t` | put / bi_put / get / bi_get / put_signal / all | put |
| `--datatype <type>` | `-d` | float / int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64 / char / all | float |
| `--block-size <size>` | `-b` | Blocks and QPs; one QP per block, in `[1, 56]` | 1 |
| `--block-range <min> <max>` | — | Parsed for compatibility but unused; use `-b` | 1 1 |
| `--exponent <exponent>` | `-e` | Bytes per QP as `2^exponent`; multiply by QP count for total bytes | Unspecified |
| `--exponent-range <min> <max>` | — | Per-QP size exponent range | 3 17 |
| `--loop-count <count>` | — | Iterations | 1000 |
| `--ub-size <size>` | — | UB in KB, passed to low-level operations and recorded in CSV; high-level UDMA RMA staging needs at least 128 B | 16 |
| `--metric <bw\|lat>` | — | Bandwidth, or submission latency for `-t put` only | bw |
| `--batch <N>` | — | BW batch size. 0/1 uses ordinary NBI. For N > 1, defer the first N−1 operations, submit the last, then quiet. N must be below 4096 and `N * 64 <= --ub-size * 1024`. Ignored for lat. | 0 |
| `-shared_jetty <0\|1>` | — | Enable sharing through `attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED` before initialization; all PEs must agree. See the sharing constraints below. | 0 |
| `-pes <size>` | — | PEs | 2 |
| `-ipport <ip:port>` | — | Bootstrap address | tcp://127.0.0.1:8768 |
| `-gnpus <num>` | — | NPUs | 2 |
| `-fnpu <id>` | — | First logical NPU | 0 |
| `-fpe <id>` | — | First PE | 0 |
| `-a/--analyse <mode>` | `-a` | none / plot / md | none |

### Shared jetty

Shared jetty is disabled by default. Enabling it makes peers resolving to the same `(local EID, QP slot)` share an underlying jetty, including one SQ/CQ ring and queue state. The runtime's `libhcomm.so` must provide the channel config interfaces; otherwise initialization returns `ACLSHMEM_NOT_SUPPORTED`.

Quiet completion covers the sharing group. Ordinary `quiet` drains the group for QP0; use `qp_quiet` to drain other QPs. The caller must serialize all queue-state mutations within a sharing group: put/get/atomic submissions, quiet, and capacity polling. Check the actual topology before combining sharing with concurrent multi-QP submission (`-b > 1`).

### Memory constraints

This benchmark tests HBM (`DEVICE_SIDE`) only. D2H / `HOST_SIDE` DRAM RMA is unsupported by UDMA here; use [rma_d2h_demo](../../rma_d2h_demo/README.md) for DRAM testing.

Local memory defaults to 1 GB and grows automatically for larger transfers, up to 40 GB.

### Metrics

The UDMA NBI interfaces are asynchronous. Each measurement window covers `loop_count` submissions and is divided by `loop_count`; profiling outside the loop avoids adding the profiling macros' `pipe_barrier` overhead to every operation.

- `bw`: `prof_start → submissions → quiet → prof_end`. Bandwidth in GB/s is `datasize × qp_count / (window_us / loop_count)`, where datasize is bytes per QP. Time columns contain `window_us / loop_count`. All five modes are supported.
- `lat`: `prof_start → submissions → prof_end → quiet`. The time columns report average `put_nbi` submission latency. Bandwidth is 0 because it is inapplicable. Only `-t put` is supported; other mode combinations fail.

### Batching

For `--metric bw`:

- `--batch 0`: ordinary NBI submissions followed by one final quiet.
- `--batch 1`: quiet after every ordinary NBI submission.
- `--batch N`, N > 1: defer the first N−1 operations, submit the last, then quiet. The last batch uses the remaining iteration count.

The single profiling window still reports average time per operation, including batched quiet. Aggregated actions apply to put / bi_put / get / bi_get with batch > 1. PutSignal always uses ordinary NBI but groups quiet by batch. `--metric lat` ignores batching.

### Examples

```bash
# Unidirectional PUT bandwidth.
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# PUT submission latency.
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --metric lat

./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000
./run.sh -t put_signal -d float -e 14 --loop-count 500
./run.sh -t all -d float --exponent-range 8 20 --loop-count 1000
./run.sh -t put -d all --exponent-range 8 20

# Quiet after each operation, then every 16 operations.
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 1
./run.sh -t get -d float --exponent-range 8 20 --loop-count 1000 --batch 16

# One QP with batching, four QPs, and four QPs with batching.
./run.sh -t put -d float -e 14 --batch 8
./run.sh -t bi_put -d float -e 16 -b 4
./run.sh -t bi_get -d float -e 16 -b 4 --batch 8

# Enable shared jetty through runtime_config.
./run.sh -t put -d float -e 16 -shared_jetty 1
```

## PutSignal verification

The benchmark allocates and zeroes `n_pes * sizeof(uint64_t)` symmetric signal memory. Each point runs `warmup + loop_count` calls to `aclshmemx_udma_put_signal_nbi(..., signal_base + i, peer_pe)`, increasing the signal to avoid stale matches. The host verifies the peer signal against `signal_base + (warmup + loop_count - 1)`. A mismatch prints ERROR and allows later points to continue.

## CSV output

The columns match the MTE benchmark and [perf_data_process.py](../../utils/perf_data_process.py):

```text
DataSize/B, Npus, Blocks, UBsize/KB, Bandwidth/GB/s, CoreMaxTime/us, SingleCoreTime/us
```

For put / bi_put / get / bi_get, `Blocks` is `--block-size`; PutSignal uses one block. `DataSize/B` is bytes per QP per transfer (`2^e`), and total bytes are `DataSize × Blocks`. Bandwidth uses total bytes. Include `Blocks` when comparing engines at the same exponent.

Files are named `output/udma_<metric>_<test_type>_<dtype>[_qpN]_<pe>.csv`, such as `udma_bw_put_float_0.csv` or `udma_bw_put_float_qp4_0.csv`. For lat, bandwidth is 0 and the time columns contain average submission latency.

```text
[INFO] udma_perftest start, pe=0, t=put, d=float, exp=10-10, loop=100, ub=16KB, metric=bw, batch=100
pe: 0 size(per QP): 1024B, total: 1024B frame_id: 0
[Verification] put: checking...
[Verification] SUCCESS
[SUCCESS] udma_perftest done in pe 0
```

## Limitations

1. Concurrent RMA/AMO operations to the same PE are unsupported by the UDMA interface. A single QP uses one block; each block in a multi-QP test owns an explicit QP and its SQ.
2. Aggregated actions require `PIPE_MTE3`. A batch must have fewer than 4096 WQEs, each using 64 B of UB, with a 256 KB UB limit. QP-specific interfaces require a direct UDMA build with `ACLSHMEM_RELAY_SUPPORT=OFF`.
3. UDMA is enabled only for Ascend950; other SOCs abort in the kernel.
4. Only HBM is tested; D2H / `HOST_SIDE` DRAM is unsupported.
5. Atomic operations are outside this benchmark's scope.
6. High-level UDMA RMA uses `PIPE_MTE3` staging with default `offset = 189 * 1024`, `ub_size = 128` bytes, and `sync_id = 0`. If changed through `aclshmemx_set_udma_config`, UB must remain at least 128 B. This benchmark's low-level path uses its explicit UB argument.
