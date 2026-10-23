# sdma_perftest

## Overview

`sdma_perftest` is an example for testing SDMA data movement capability and performance. It uses the multi-QP (Queue Pair) SDMA interfaces and supports unidirectional or bidirectional get/put tests, bandwidth or latency metrics, and configurable QP/AIV counts through `--qp`. The results are for reference only; actual performance depends on the workload and runtime environment.

[中文版](README.md)

## Supported Tests

### Test Types

| Test type | Description | A2/A3 platforms | Ascend950 |
|-----------|-------------|-----------------|-----------|
| `get` | Unidirectional get. The profiling PE reads from the next PE's source buffer into the local destination buffer | Supported | Supported |
| `bi_get` | Bidirectional get. Both PEs read from their peer at the same time | Supported | Supported |
| `put` | Unidirectional put. The profiling PE writes its local source buffer to the next PE | Supported | Not supported |
| `bi_put` | Bidirectional put. Both PEs write to their peer at the same time | Supported | Not supported |

Ascend950 does not support SDMA remote WRITE, so `put` and `bi_put` are rejected before entering the kernel. `get` and `bi_get` use only the SDMA READ path.

### Metrics

| Option | Description | Applies to |
|--------|-------------|------------|
| `--metric bw` | Bandwidth test, the default mode | `get` / `bi_get` / `put` / `bi_put` |
| `--metric lat` | Latency test, reporting the average timing-window cost per nbi submit | `get` / `bi_get` / `put` / `bi_put` |
| `--batch <N>` | In bandwidth mode, call quiet after every N nbi submits; `0` means one quiet after all submits | Effective only with `--metric bw` |

## Environment Requirements

CANN version and ops package requirements follow the [CANN version description](../../../docs/quickstart_en.md#software-and-hardware-versions) in the quick start guide (see the [SDMA example](../../sdma/README_en.md) for details). Build from the repository root:

```bash
# A2/A3 platforms
bash scripts/build.sh -examples

# Ascend950
bash scripts/build.sh -examples -soc_type Ascend950
```

## Command-line Options

| Option | Shorthand | Description | Default |
|--------|-----------|-------------|---------|
| `--test-type <type>` | `-t <type>` | Test type: `get` / `bi_get` / `put` / `bi_put` | get |
| `--datatype <type>` | `-d <type>` | Data type: `float` / `int8` / `int16` / `int32` / `int64` / `uint8` / `uint16` / `uint32` / `uint64` / `char` | float |
| `--qp <count>` | - | Number of QPs/AIVs, from 1 up to the available AIV count on the device and capped at 72 | 2 |
| `--exponent <exponent>` | `-e <exponent>` | Transfer size is `2^exponent` bytes | - |
| `--exponent-range <min> <max>` | - | Transfer-size exponent range | 3-17 |
| `--loop-count <count>` | - | Loop count per data point | 1000 |
| `--ub-size <size>` | - | UB size in KB | 16 |
| `--metric <bw\|lat>` | - | Performance metric: `bw` or `lat` | bw |
| `--batch <N>` | - | Quiet interval for bandwidth tests. `0` means one quiet after all nbi submits, `1` means quiet after every nbi submit, and other values quiet per group | 0 |
| `-pes <size>` | - | Number of PEs, currently fixed at 2 | 2 |
| `-ipport <ip:port>` | - | Communication address | tcp://127.0.0.1:8769 |
| `-gnpus <num>` | - | Number of NPUs, currently fixed at 2 | 2 |
| `-fnpu <id>` | - | First NPU ID | 0 |
| `-fpe <id>` | - | First PE ID, currently fixed at 0 | 0 |

## Examples

Unidirectional get:

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get -d float -e 10 --loop-count 10 \
  -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28769
```

Bidirectional get:

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t bi_get -d float -e 10 --loop-count 10 \
  -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28775
```

Specify the QP count:

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get --qp 48 -d float \
  -e 20 --loop-count 100 -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28779
```

Latency test:

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get -d float -e 10 \
  --loop-count 1000 --metric lat -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28781
```

Quiet after every nbi submit in bandwidth mode:

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get -d float -e 10 \
  --loop-count 1000 --batch 1 -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28783
```

Run through the top-level dispatcher:

```bash
bash examples/shmem_perftest/run.sh -m sdma -t get -d float --qp 4 -e 10 --loop-count 10
```

## QP/AIV Configuration

- Each AIV maps to one QP; `AscendC::GetBlockIdx()` is used as both `aiv_idx` and `qp_idx`.
- Each active QP/AIV transfers one complete message per iteration and uses its own contiguous buffer region, so concurrent requests do not overlap.
- Before `aclshmemx_init_attr`, the program calls `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num)` to create the QPs required by the test.
- The requested QP/AIV count must not exceed the device vector-core count or 72. The available AIV count is queried and validated automatically at runtime: if `--qp` exceeds the device capacity, the program fails fast and reports the actual available count, so there is no need to look it up manually when the chip model is unknown. To check beforehand, see the `vector_core_cnt` field of the chip-specific `.ini` file under `${ASCEND_TOOLKIT_HOME}/<arch>-linux/data/platform_config/` (default `/usr/local/Ascend/ascend-toolkit/latest/<arch>-linux/`, where `arch` is `x86_64` or `aarch64`).

## Output

Results are written to `output/sdma_<metric>_<test_type>_<datatype>_qp<QP-count>_<prof_pe>.csv`, for example:

```text
output/sdma_bw_get_float_qp2_0.csv
output/sdma_lat_bi_get_float_qp4_0.csv
```

Only the profiling PE, specified by `SHMEM_CYCLE_PROF_PE` and defaulting to PE0, writes the CSV file.

The CSV file contains these fields:

| Field | Description |
|-------|-------------|
| `DataSize/B` | Per-transfer data size in bytes |
| `Npus` | Number of NPUs used |
| `QPs` | Actual number of QPs/AIVs used |
| `UBsize/KB` | UB size in KB |
| `Bandwidth/GB/s(1000)` | Bandwidth in decimal GB/s; 0 in `lat` mode |
| `Bandwidth/GiB/s(1024)` | Bandwidth in binary GiB/s; 0 in `lat` mode |
| `CoreMaxTime/us` | Maximum per-iteration time across active AIVs; average per-operation latency in `lat` mode |

## Notes

1. The transfer size (`2^exponent` bytes) must be a multiple of the selected data type size.
2. Local symmetric memory defaults to 1GB. For larger tests, the program increases it based on the maximum transfer size and QP count; make sure the device has enough memory.
3. Each data point runs 100 warm-up iterations before measurement.
4. Use a larger `--loop-count` for more stable performance data.
5. With `--metric lat`, `--batch` has no effect. All nbi submits are timed in one window, and quiet runs outside that window.
6. `--batch` defaults to 512 when unset: this amortizes the fixed cost of quiet while avoiding too many in-flight requests on one QP; if the effective value exceeds `--loop-count`, it is clamped to a single trailing quiet after all submits (e.g. the default 512 with `--loop-count 100` behaves as no grouping). If `--batch` is too large, multiple in-flight requests accumulate on one QP and may be limited by the SDMA channel SQ depth.
