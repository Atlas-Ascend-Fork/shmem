## Overview

This sample tests the bandwidth performance of the SIMT UDMA QP-specific APIs. It runs in parallel with `udma_perftest` in the same directory (which targets the SIMD/AICore low-level UDMA APIs). The sample collects performance data through the [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) macros, covering the bandwidth of `simt::aclshmemx_udma_qp_put_nbi_{warp|block}` / `simt::aclshmemx_udma_qp_get_nbi_{warp|block}` under both warp and block scopes. **The script test results are for reference only; performance should be evaluated in actual scenarios.**

## Test Objectives

Performance of the following SIMT UDMA QP-specific data transfer operations:

1. **Warp Put** (`warp_put`): The Active PE calls `simt::aclshmemx_udma_qp_put_nbi_warp` in the warp scope, with each warp cooperatively driving one QP.
2. **Warp Get** (`warp_get`): The Active PE calls `simt::aclshmemx_udma_qp_get_nbi_warp` in the warp scope, with each warp cooperatively driving one QP.
3. **Block Put** (`block_put`): The Active PE calls `simt::aclshmemx_udma_qp_put_nbi_block` in the block scope, with each kernel block driving one QP.
4. **Block Get** (`block_get`): The Active PE calls `simt::aclshmemx_udma_qp_get_nbi_block` in the block scope, with each kernel block driving one QP.
5. **Bidirectional Put** (`bi_put`): Two PEs simultaneously call put in the corresponding scope, transferring data to each other.
6. **Bidirectional Get** (`bi_get`): Two PEs simultaneously call get in the corresponding scope, pulling data from each other.

The first four are unidirectional modes (only the Active PE initiates transfers); the last two are bidirectional modes (both PEs initiate transfers simultaneously).

## Differences from `udma_perftest` (SIMD Version)

| Dimension | `udma_perftest` (SIMD/AICore) | `simt_udma_perftest` (SIMT) |
|------|------|------|
| Programming model | SIMD/AICore (`__global__ __vector__` + AIV block) | SIMT (`asc_vf_call` dispatching `__simt_vf__`) |
| API | `aclshmemx_udma_qp_put_nbi` / `aclshmemx_udma_qp_get_nbi` (requires UB scratch parameter) | `simt::aclshmemx_udma_qp_put_nbi_{warp\|block}` / `simt::aclshmemx_udma_qp_get_nbi_{warp\|block}` (no UB parameter) |
| Execution scope | Single (block = AIV) | Two (warp / block), selected at runtime via `--scope` |
| QP driving | block i drives QP i (`GetBlockIdx()`) | warp/block i drives QP i (depends on the scope) |
| Combination selection | `-t` selects the test type | `-t` (operation type) + `--scope` (execution scope); `all` expands 8 combinations sequentially within one process |
| UB buffer | Required (WQE staging) | Not used (SIMT API internally manages the DWQE window) |
| Test modes | put / bi_put / get / bi_get / put_signal | put / get / bi_put / bi_get (× warp/block = 8 combinations) |
| Data type | `-d` supports 10 types, participating in statistics by bit width | Fixed `uint8` (T = `uint8_t`, transferred by byte) |
| Metric | bw / lat (lat: single-phase window, measures put submission cost only) | bw / lat (lat: dual-phase baseline subtraction, per-call net latency) |
| SOC limitation | Ascend950 only | Ascend950 only |

## Version and Platform Support

- This sample supports Ascend950 (direct mode) only. Other platforms are not supported.
- CANN 9.2.0 and later provide the HCOMM resource APIs required by UDMA. Download and install the Ascend950 toolkit and ops packages from [CANN 9.2.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.2.0). CANN versions earlier than 9.2.0 are not supported by this sample.
- Before building, run `source /usr/local/Ascend/ascend-toolkit/set_env.sh`, and build with `-enable_simt -soc_type Ascend950`.

## Build Instructions

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
```

Build artifact: `build/bin/simt_udma_perftest` (single binary).

Each scope × operation-type combination is generated at compile time via template instantiation (each has its own VF form, `__launch_bounds__`, and QP driving method). At runtime, `--scope` / `-t` selects the corresponding combination; `all` runs all combinations sequentially within the same process. The build artifact is a single binary, consistent with other performance test cases in the same directory.

## Test Model

Fixed two-PE model:

| Role | PE ID | Responsibility |
|------|---------|------|
| Active PE | 0 | Initiates put/get transfers in unidirectional mode; collects prof data |
| Passive PE | 1 | Acts as the remote communication peer in unidirectional mode; in bidirectional mode, both PEs initiate transfers simultaneously |

Data direction and validating side (validation is performed by the data-receiving side):

| Operation | Data Direction | Validated By | Validation Content |
|------|---------|--------|---------|
| put | Active PE symmetric memory → Passive PE symmetric memory | Passive PE | Symmetric memory should contain the Active PE's data values |
| get | Passive PE symmetric memory → Active PE symmetric memory | Active PE | Symmetric memory should contain the Passive PE's data values |
| bi_put | Both PEs' symmetric memory → peer PE's symmetric memory | Both PEs | Each side's symmetric memory should contain the peer PE's data values |
| bi_get | Both PEs' symmetric memory ← peer PE's symmetric memory | Both PEs | Each side's symmetric memory should contain the peer PE's data values |

## Performance Test Method

### Submission and Draining

The transfer VF internally loops `loop_count` times. Each iteration issues one put/get per QP (without calling quiet), after which the quiet VF drains all outstanding WQEs across all QPs in one pass.

### Metric Definitions

`--metric` controls the coverage of the prof window. Both metric definitions take their markers outside the loop (`SHMEMI_PROF_START/END` include a `pipe_barrier`; per-iteration markers would accumulate measurement overhead into the result, so the whole window is divided by `loop_count`):

- **`bw` (default)**: `prof_start → loop(put/get_nbi) → quiet → SyncAll → prof_end`. The window includes quiet draining, reflecting end-to-end throughput:

  ```
  prof_start → asc_vf_call(transfer_vf: loop_count × put/get_nbi)
             → asc_vf_call(quiet_vf: per-QP quiet)
             → SyncAll → prof_end
  ```

- **`lat`**: The window covers only the submission loop; quiet is moved outside the window. A same-shape empty loop (volatile sink to prevent compiler elimination, with no API calls) is then executed as a baseline window. The difference yields the per-call net latency:

  ```
  prof_start → asc_vf_call(transfer_vf: loop_count × put/get_nbi) → prof_end
             → asc_vf_call(quiet_vf)                 # drain outside the window
  prof_start → asc_vf_call(baseline_vf: same-shape empty loop) → prof_end
  ```

  `net_per_call = (real_window − baseline_window) / loop_count`. After subtracting the baseline, the result is the pure API submission cost, which differs slightly from the `lat` metric in `udma_perftest` (single-phase submission cost without baseline subtraction).

### Buffer Design and Partitioning

Source and destination use independent symmetric buffers (`src` and `dst`): put reads from local `src` and writes to remote `dst`; get reads from remote `src` and writes to local `dst`. Using independent buffers avoids read/write contention in bidirectional mode when both PEs transfer simultaneously. Each QP occupies an independent slice based on `qp_idx * slice`, with no overlap.

### Warmup

Before timing, `warmup` warmup iterations are executed (outside the prof window). After warmup, the data is refilled before the timed rounds to eliminate cold-start effects.

### Bandwidth Calculation

`Bandwidth = slice × qp_count × loop_count / prof_time`, where `prof_time` takes the maximum window duration across all blocks (the slowest block determines the overall bandwidth). Only the `bw` metric computes bandwidth; under the `lat` metric, the bandwidth column is 0 and the time column shows the per-call net latency. In bidirectional mode, the statistics reflect the current PE's outgoing traffic.

### loop_count / warmup Upper Limit

The maximum number of outstanding calls per QP is 16378. `loop_count` and `warmup` must be below this value; otherwise, continuous submission within the VF triggers internal reclamation, contaminating the timing window.

## Usage

### Basic Usage

```bash
cd examples/shmem_perftest/simt_udma_perftest/
bash run.sh [options]
```

### Command-Line Options

The command-line option style is consistent with `udma_perftest`, `simt_rma_perftest`, and other sub-examples (long options use hyphen form uniformly). Run `bash run.sh --help` to view the built-in help.

| Option | Default | Description |
|------|--------|------|
| `--scope <type>` | `all` | Execution scope: `warp`, `block`, or `all` (both scopes run sequentially) |
| `-t\|--test-type <type>` | `all` | Operation type: `put`, `get`, `bi_put`, `bi_get`, or `all` (all four run sequentially) |
| `--qp <N>` | `4` | Number of QPs created for each remote PE. Range: `[1, 56]`. Must be identical on all PEs |
| `-e\|--exponent <E>` | - | Data volume exponent for a single QP (2^E bytes). Range: [1, 28] (E=0 yields only 1 byte, below the UDMA engine 2-byte lower limit; 2^28 = 256 MiB is the API single-request upper limit). When set, min=max=E |
| `--exponent-range <min> <max>` | `3 17` | Data volume exponent range for a single QP. min >= 1 (the UDMA engine requires each QP to transfer at least 2 bytes), max <= 28 (2^28 = 256 MiB, the SIMT UDMA single-request upper limit). Each exponent produces one row of statistics |
| `--loop-count <N>` | `1000` | Number of timed-loop iterations. Range: `[1, 16378)` |
| `--warmup <N>` | `100` | Number of warmup-loop iterations. Range: `[0, 16378)` |
| `--metric <bw\|lat>` | `bw` | Performance metric: `bw` measures end-to-end bandwidth; `lat` measures per-call net latency (dual-phase baseline subtraction) |
| `--heap-mb <N>` | `1024` | SHMEM symmetric heap size per PE in MiB. Must cover `2 × qp_count × 2^max_exponent` bytes (sym_src and sym_dst each occupy one segment), plus library internal overhead |
| `-ipport <tcp://IP:PORT>` | `tcp://127.0.0.1:8999` | Bootstrap initialization address |
| `-pes <N>` | `2` | Total number of PEs, fixed at 2 |
| `-gnpus <N>` | `2` | Number of processes / NPUs to start, fixed at 2 |
| `-fnpu <ID>` | `0` | First NPU ID; actual device id = `pe % gnpus + fnpu` |
| `-fpe <ID>` | `0` | First PE ID; kept for parameter consistency with other shmem_perftest sub-examples; currently unused |

### Usage Examples

```bash
# Default: all 8 combinations, 4 QPs, data volume 2^3~2^17 bytes, 1000 iterations
bash run.sh

# Test warp mode put only, 8 QPs
bash run.sh --scope warp -t put --qp 8

# Specify data volume range and iteration count
bash run.sh --exponent-range 10 20 --loop-count 500

# Test warp mode get only, single QP
bash run.sh --scope warp -t get --qp 1 -e 16

# Block mode put, 16 QPs, 2 MB data
bash run.sh --scope block -t put --qp 16 -e 21 --heap-mb 4096

# Warp mode bidirectional put
bash run.sh --scope warp -t bi_put --qp 8

# Warp mode per-call net latency (dual-phase)
bash run.sh --scope warp -t put --qp 8 --metric lat

# All 8 combinations, measure net latency
bash run.sh --metric lat --exponent-range 10 14
```

## Output Example

```text
[INFO] pe=0 pes=2 qp_count=4 exp=3-17 loop=1000 warmup=100 metric=bw scopes=2 ops=4
[INFO] scope=warp op=put pe=0 qp_count=4
pe: 0 size(per QP): 8B, total: 32B frame_id: 0
[Verification] SUCCESS
pe: 0 size(per QP): 16B, total: 64B frame_id: 1
[Verification] SUCCESS
...
[CSV] output/simt_udma_bw_put_warp_qp4_3-17_l1000.csv
[SUCCESS] simt_udma_perftest done in pe 0
```

With `--metric lat`, each data point additionally outputs the net latency result:

```text
[lat] pe=0 slice=1024B net=0.087 us/call (gross=0.112, baseline=0.025)
```

## CSV Output

One CSV file is produced per scope/operation combination. The file name format is (relative to the `simt_udma_perftest/` directory):

```
output/simt_udma_<metric>_<op>_<scope>_qp<qp_count>_<min_exp>-<max_exp>_l<loop_count>.csv
```

The CSV columns are consistent with `udma_perftest`, allowing reuse of `examples/utils/perf_data_process.py` for plotting performance charts. The columns are defined as follows:

| Column | Description |
|------|------|
| `DataSize/B` | Data volume per QP per transfer (bytes). With multiple QPs, the total data volume is `DataSize × QPs` |
| `Npus` | Number of NPUs participating in the test (`--gnpus`) |
| `QPs` | Number of QPs per remote PE (`--qp`) |
| `UBsize/KB` | Fixed at 0. The SIMT UDMA API does not use Unified Buffer; this column exists only to maintain column format consistency with `udma_perftest` |
| `Bandwidth/GB/s(1000)` | Bandwidth (decimal units). Under the `bw` metric, computed from the total traffic across all QPs; under the `lat` metric, set to 0 |
| `Bandwidth/GiB/s(1024)` | Bandwidth (binary units). Same as above |
| `CoreMaxTime/us` | Under the `bw` metric: per-iteration time of the slowest block (total window time / loop_count). Under the `lat` metric: per-call net latency (maximum across all blocks) |
| `SingleCoreTime_B{n}/us` | Per-iteration time (`bw`) or per-call net latency (`lat`) of the `n`-th block (0-indexed). The number of columns is dynamically generated based on the QP count; blocks without prof data are filled with `N/A`. Under the warp scope, only the `B0` column exists |

## Known Constraints

1. This sample tests the HBM (DEVICE_SIDE) memory path only; D2H / `HOST_SIDE` (DRAM) is not supported.
2. UDMA is enabled at compile time for Ascend950 only; on other SOCs, the kernel will error out.
3. `loop_count` and `warmup` must be less than 16378 (the maximum number of outstanding calls per QP); otherwise, continuous submission within the VF triggers internal reclamation, contaminating the timing window.
4. The execution scope and operation type are implemented via template instantiation at compile time. `--scope` / `-t` selects the specific combination; `all` runs multiple expanded combinations sequentially within the same process, sharing 1024 prof frames. When the total number of data points exceeds the limit, a warning is printed and the remaining combinations are stopped.
5. This sample does not test PutSignal; that is a capability of `udma_perftest`.
6. The exponent lower limit is 1 (2^1 = 2 bytes), corresponding to the API's minimum 2-byte per-transfer requirement.
7. The exponent upper limit is 28 (2^28 = 256 MiB), corresponding to the API's 256 MB single-transfer upper limit. Slices exceeding this upper limit exceed the API's single-request capability (rejected by the device-side range check), and 2^32 and above overflow to 0 when narrowed to `uint32_t` on the host side, so out-of-range values are directly rejected as invalid parameters.
