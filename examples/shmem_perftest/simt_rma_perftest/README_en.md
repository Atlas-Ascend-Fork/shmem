## Overview

This example demonstrates, in SIMD and SIMT hybrid compilation mode, how to run a performance test on the SIMT remote memory access (RMA) APIs.
The test evaluates the single-node, two-card Device-to-Device `gm2gm` (global memory to global memory) data transfer capability, covering the **one-directional** `put` and `get` operations, and reports bandwidth and latency statistics.

## Test Model

The test always uses two cards, with PE IDs 0 and 1:

- **Active PE (PE 0)**: The initiator of all communication operations (`put` / `get`).
- **Passive PE (PE 1)**: The remote endpoint of the communication. It never initiates an operation.

The data direction and the validating side of the two operations are as follows:

| Operation | Data Direction | Validated By |
| --- | --- | --- |
| `put` | Active PE's symmetric memory → Passive PE's symmetric memory | Passive PE |
| `get` | Passive PE's symmetric memory → Active PE's symmetric memory | Active PE |

## Performance Test Method

To obtain bandwidth and latency figures close to real usage, to avoid data cache hits inflating the results, and to keep the implementation simple, this example adopts the following design:

- **Only the Active PE initiates; source and destination share one address**: `put` / `get` are one-sided operations, all issued by the Active PE. The Passive PE only acts as the remote object and performs no computation (the two cards synchronize through host-side barriers). The source and destination address of each transfer is the **same symmetric memory offset**, so data moves between the same offset on the two cards.
- **Warmup and averaging over multiple rounds**: Each measurement first runs a fixed 100 transfers as warmup (excluded from the statistics), then runs `loops` transfers for sampling. The timing works as follows: after the warmup ends, the whole batch of `loops` transfers is timed as one window (the start marker is taken once before the first sampled transfer and the end marker once after the last). The resulting total duration is then divided by `loops` to obtain the per-transfer cost, which amortizes the marker overhead and rules out cold-start effects.
- **Logical segments and the logical ring**: Each Core is physically allocated a 1 MB block of symmetric memory. Let the size of a single transfer be $X$ bytes and let $N$ Cores be used. Each Core's **logical segment** size is defined as $L = \min(1\text{MB},\ (100 + loops) \times X)$ (never smaller than $X$, and always an integer multiple of $X$). The $N$ logical segments are placed end to end to form a **logical ring** of size $N \times L$, where Core $j$'s logical segment starts at offset $j \times L$ within the ring.
- **Sliding-window traversal**: On iteration $i$, all Cores together form a sliding window whose overall start offset within the ring is $(i \times X) \bmod (N \times L)$. Core $j$'s target offset for this transfer is $\big(i \times X + j \times L\big) \bmod (N \times L)$, that is, the Cores are staggered one logical segment apart from each other on the ring, each transferring one $X$-byte chunk. The window advances by $X$ bytes per round and wraps around within the ring. Because $X \le L$, the chunks of the different Cores never overlap within one round; conflicts across iterations are prevented by `SyncAll`.
- **Avoiding cache hits**: The addresses of consecutive transfers keep moving forward so that they land on different cache lines as much as possible. When the total transfer volume is small, the logical ring occupies just enough space; when it is large, the ring is reused cyclically up to the 1 MB cap. This prevents data cache hits from inflating the bandwidth readings. Validation walks the buffer with the same logical-segment layout, comparing segment by segment (the first $L$ bytes of each segment), so the validated regions exactly match the regions actually written.

## Source File Macro Configuration

Some test dimensions are controlled by constants defined at the top of `main.cpp`. You can modify these constants before compiling to change the API being called or the data width:

| Constant | Meaning and Effect | Allowed Values / Default |
| --- | --- | --- |
| `VF_TYPE` | Instruction computation mode framework. | `VfType::Simt` (default), `VfType::Simd` |
| `OP_TYPE` | The specific operation the performance test runs. | `OpType::Put` (default), `OpType::Get`, `OpType::None` (calls the API with `count = 0` only, measuring the call overhead without transferring real data) |
| `DATA_SIZE` | Data width, in bits, of the underlying RMA API being called. Changing it switches to the read/write API of the corresponding width (for example, `aclshmemx_get32_block` becomes `aclshmemx_get64_block`). | `8`, `16`, `32` (default), `64` |
| `THREAD_COUNT` | Number of threads launched per Core in SIMT mode, which determines the concurrency scale of the vector instruction stream. | Default `1024` |

> **Note**: After modifying the constants above, you must return to the root directory and recompile (see below) for the new configuration to take effect.
> For certain reasons, two vf functions that merely call similar SIMT RMA APIs within the same compilation unit currently cause a compilation problem (compilation itself does not report an error, but errors occur at runtime). This example therefore tests different SIMT RMA APIs by modifying the source, and provides the constants above to make that convenient.

## Supported Devices

- Ascend 950PR&950DT products

## Instructions

1. **Configure the CANN environment variables.**

   Before compiling, load the CANN environment variables (choose one according to your actual installation path):

   ```bash
   # Default installation path
   source /usr/local/Ascend/cann/bin/set_env.sh
   # Custom installation path
   source ${install_path}/cann/bin/set_env.sh
   ```

2. **Compile the project.**

   Run the compilation script in the `shmem/` root directory:

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

3. **Run the sample program.**

   Go to the example directory and run the execution script:

   ```bash
   cd examples/shmem_perftest/simt_rma_perftest
   bash run.sh [options]
   ```

### Script Options

`run.sh` supports the following options for adjusting the test scale and conditions:

| Option | Description | Default |
| --- | --- | --- |
| `-pes <int>` | Total number of PEs. This test uses a fixed two-card model, so it must be 2. | 2 |
| `-ipport <ip:port>` | ACLSHMEM initialization communication address. | `tcp://127.0.0.1:8760` |
| `-gnpus <int>` | Number of processes / NPUs started on this node. Fixed at 2 for this test; any other value reports an error. | 2 |
| `-fnpu <int>` | First NPU ID. The actual device id is `pe_id % gnpus + fnpu`. | 0 |
| `-fpe <int>` | First PE ID. Kept for CLI compatibility with the shmem_perftest options; it currently takes part in neither the rank nor the device computation. | 0 |
| `-t`/`--test-type <put\|get\|none>` | Optional consistency check. If given, it must match the compile-time `OP_TYPE`, otherwise the binary reports an error. | - |
| `-d`/`--datatype <type>` | Optional consistency check. The type name maps to a bit width, which must match the compile-time `DATA_SIZE`. Allowed values are `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, and `char`. | - |
| `-b`/`--block-size` | Number of Cores (blocks) used per PE. | 32 |
| `--block-range <min> <max>` | Sweep range for the number of Cores (blocks). Each core count produces its own statistics. | 32 32 |
| `--block-list <b1,b2,...>` | Explicitly specifies the core counts to test, separated by commas (for example, `2,4,8`). When given, it takes precedence over `-b`/`--block-range`. | - |
| `--loop-count` | Number of sampled iterations. Must be in $[1, 10000)$; out-of-range values report an error and exit. | 1000 |
| `-e`/`--exponent <exp>` | Exponent of the single-transfer data volume (a single value), as a power of 2 (for example, `10` means $2^{10} = 1024$ bytes). Must be in $[3, 20]$. | - |
| `--exponent-range <min> <max>` | Exponent range of the single-transfer data volume. Must be in $[3, 20]$; out-of-range values report an error and exit. The upper bound 20 corresponds to a single transfer of 1 MB, which is the physical buffer size of each Core. | 3 20 |
| `--ub-size` | Unified Buffer size used per Core, in KB. **Takes effect in SIMD mode only; the default SIMT mode does not use this option.** | 16 |

> This test uses a fixed two-card model (Active PE0 / Passive PE1). Both the number of started processes and the number of PEs inside the program are fixed at 2. `-pes` and `-gnpus` are kept for CLI compatibility with the shmem_perftest options, but passing any value other than 2 reports an error.
> The test sweeps the single-transfer data volume exponent by exponent from the `--exponent-range` min to max (that is, $2^{min}, 2^{min+1}, \dots, 2^{max}$ bytes), and iterates core count by core count over the set given by `--block-range` (or `--block-list`). Each (core count, data volume) combination produces one row of statistics.

For example, to test the performance of `4` Cores transferring $2^8$ to $2^{12}$ bytes:

```bash
bash run.sh -b 4 --exponent-range 8 12
```

### Performance Output

After the test finishes normally, the Active PE (PE 0) writes one `.csv` statistics file into the `output/` subdirectory of the example directory. The file name format is:

```bash
[DATA_SIZE]_[blocks]_[OpType]_[VfType]_[minExp]-[maxExp]_l[loop_count]_t[THREAD_COUNT].csv
```

The `[blocks]` field reflects the set of core counts actually tested. The naming rule depends only on the set itself, not on which option specified it: if the set happens to be one contiguous ascending range, it is written as `min-max` (for example, both `--block-range 1 4` and `--block-list 1,2,3,4` become `1-4`); otherwise the values are joined with `_` in test order (for example, `--block-list 2,4,8` becomes `2_4_8`, and `--block-list 8,2,4` becomes `8_2_4`).

The columns of the `.csv` file have the following meanings:

| Column | Description |
| --- | --- |
| `DataSize/B` | Data volume of a single RMA transfer, in bytes, corresponding to the $2^{exp}$ value sampled in this row. |
| `Npus` | Number of PEs taking part in the test, which is 2 for a two-card test. |
| `Blocks` | Number of Cores (blocks) taking part in the communication, that is, the core count used by this row. Under a core-count sweep (`--block-range` / `--block-list`), each row takes its value from the swept set. |
| `UBsize/KB` | Unified Buffer size used per Core, in KB, that is, `--ub-size`. |
| `Bandwidth/GB/s (1000)` | Average cross-card transfer bandwidth measured for this parameter set, converted using decimal units (divided by $1000^3$). Always 0 for the `none` operation. |
| `Bandwidth/GiB/s (1024)` | The same bandwidth converted using binary units (divided by $1024^3$). Always 0 for the `none` operation. |
| `CoreMaxTime/us` | Among all Cores, the latency of the Core with the longest average per-operation cost, in microseconds. This is the time used for the bandwidth computation. |
| `SingleCoreTime/us` | Average per-operation latency of each Core, in microseconds, obtained by dividing that Core's total duration over `loops` transfers by `loops`. There is one column per Core. |
