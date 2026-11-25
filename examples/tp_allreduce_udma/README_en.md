# TP=2 AIV-UDMA AllReduce Sample

## Sample Overview

This sample launches multiple PEs on one machine and divides each pair of adjacent PEs into a
TP=2 communicator. Each TP group independently executes `reduce-scatter + local reduce +
all-gather`, and no AllReduce data is transferred between different TP groups.

The sample provides two communication modes through `--mode` and reuses the same host
initialization, Local Reduce, performance statistics, and golden data verification logic:

- `baseline`: Core 0 calls `aclshmemx_udma_put_nbi` to send data directly, and calls
  `aclshmemx_udma_quiet` to wait for completion.
- `tailcut`: Splits the communication shards by ratio into one direct path and two relay paths.
  Cores 0, 1, and 2 submit the direct, relay0, and relay1 paths respectively, and then Core 0
  calls `aclshmemx_udma_quiet` once.

Local Reduce is split by tile and computed by all launched AIV Cores. The tailcut mode regards
every four consecutive PEs as one box, and the two PEs outside the current TP pair serve as
relay PEs. The default traffic weight is `2:1:1`.


## Related Documentation

For the complete operator description, Tailcut relay design, and performance test method, refer
to [TP=2 AIV-UDMA AllReduce Operator Practice](https://gitcode.com/cann/shmem/wiki/operator-practice-tp-allreduce-udma).

## Compilation

Run the following commands in the shmem root directory.

For the baseline mode only:

```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

For both the baseline and tailcut modes:

```bash
bash scripts/build.sh -examples -soc_type Ascend950 -enable_relay
```

The tailcut mode depends on the relay interfaces, and `-enable_relay` must be enabled. When
comparing the performance of the two modes, use the same relay build configuration.

## Running

Running the script directly performs an FP16 functional test with the default baseline
parameters and compares the result with the golden data:

```bash
bash examples/tp_allreduce_udma/scripts/run.sh
```

Run the tailcut functional test (4 PEs, 3 AIV Cores, and `2:1:1` by default):

```bash
bash examples/tp_allreduce_udma/scripts/run.sh --mode tailcut
```

Run a performance comparison with the same configuration:

```bash
bash examples/tp_allreduce_udma/scripts/run.sh \
    --mode baseline -pes 4 -fnpu 0 -elements 1048576 -aiv 20 \
    --perf --warmup 1 --loops 20 --prof-pe 0 \
    --perf-csv output/baseline_perf.csv

bash examples/tp_allreduce_udma/scripts/run.sh \
    --mode tailcut -pes 4 -fnpu 0 -elements 1048576 -aiv 20 \
    --ratio 2:1:1 --perf --warmup 1 --loops 20 --prof-pe 0 \
    --perf-csv output/tailcut_perf.csv
```

## Common Parameters

| Parameter | Description | Default Value |
|---|---|---|
| `--mode` | Communication mode. `baseline` and `tailcut` are supported | `baseline` |
| `-pes` | Number of PEs. baseline supports 2, 4, 8, and 16, and tailcut supports 4, 8, and 16 | If not specified, baseline is 2 and tailcut is 4 |
| `-fnpu` | ID of the first NPU used | 0 |
| `-elements` | Number of elements in the input Tensor of each PE | 1024 |
| `-type` | Data type. `int32_t` and `float16_t` are supported | `float16_t` |
| `-aiv` | Number of AIV Cores used by the kernel. For tailcut, the value is at least 3 and cannot exceed the number of Vector Cores on the device | If not specified, baseline is 1 and tailcut is 3 |
| `-port` | SHMEM initialization port | 8899 |
| `--ratio` | Traffic weights of direct, relay0, and relay1 in tailcut mode | `2:1:1` |

## Performance Parameters

| Parameter | Description | Default Value |
|---|---|---|
| `--perf` | Enables the performance test mode | Disabled |
| `--warmup` | Number of warm-up rounds in the performance test. The value can be any non-negative integer | If not specified, 0 in functional mode and 1 in performance mode |
| `--loops` | Number of timed rounds in the performance test | 1 |
| `--prof-pe` | PE that outputs device-side segmented performance data | 0 |
| `--perf-csv` | CSV path of the performance results | `output/<mode>_perf.csv` |

Golden data and outputs are always placed in `golden/` and `output/` under the sample directory.
Each performance test creates or overwrites the CSV file specified by `--perf-csv`, and does
not append historical results.

## Result Description

The output `[SUCCESS]` on each PE indicates that kernel execution completes. Only the final
output `tp_allreduce_udma golden check passed` indicates that the results of all PEs pass the
golden data verification.


## Constraints

- The `baseline` and `tailcut` operators depend on UB-Mesh interconnection and the capability
  of AIV Cores to directly drive UDMA, and **currently support only Ascend 950PR&950DT
  products**.
- `tailcut` needs other NPUs to relay part of the communication data. **Currently, the relay
  function supports only the Atlas 950 SuperPoD**. In deployments other than this form, relay
  forwarding availability cannot be guaranteed. Do not use the `tailcut` mode.
- The TP size is fixed to 2.
- `elements` must be a positive number and integer-divisible by 2.
- The tailcut mode requires that the number of PEs be integer-divisible by 4, and the number of
  AIV Cores be at least 3.
- `--ratio` is used only in tailcut mode. The three weights must be non-negative integers, and
  their sum must be greater than 0.
- The traffic boundary of each stage is computed as
  `floor(shard_elements * cumulative weight / total weight)`, and paths with a weight of 0 do
  not send data.
