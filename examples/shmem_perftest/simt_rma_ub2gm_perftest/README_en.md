## Overview

This example demonstrates, in SIMD and SIMT hybrid compilation mode, how to run a performance test on the **ub2gm APIs** of SIMT remote memory access (RMA).
Unlike `simt_rma_perftest` (gm2gm), this example uses UB (Unified Buffer) as one end of the communication. The test evaluates the single-node, two-card Device-to-Device `ub2gm` data transfer capability, covering the **one-directional** `put` and `get` operations, and reports bandwidth and latency statistics.

| Operation | Data Path Under Test |
| --- | --- |
| `put` | Local UB → Remote GM |
| `get` | Remote GM → Local UB |

## Test Model

The test always uses two cards, with PE IDs 0 and 1:

- **Active PE (PE 0)**: The initiator of all communication operations (`put` / `get`).
- **Passive PE (PE 1)**: The remote endpoint of the communication. It never initiates an operation.

The data direction and the validating side of the two operations are as follows:

| Operation | Data Direction | Validated By |
| --- | --- | --- |
| `put` | Active PE's UB → Passive PE's symmetric memory | Passive PE |
| `get` | Passive PE's symmetric memory → Active PE's UB | Active PE |

## Performance Test Method

To keep the statistics focused as much as possible on the overhead of the ub2gm APIs, and to avoid data cache hits inflating the results, this example adopts the following design:

- **The UB buffer is allocated inside the kernel**: The UB array is defined in the `__global__` kernel and passed by `asc_vf_call` to the `__simt_vf__` function. Both data preparation and result write-back happen outside the timed region. Each iteration of the measurement loop contains only one ub2gm API call, plus one `VSSync` (S_V / V_S markers) and one `AscendC::SyncAll<true>()` all-core synchronization; these two synchronization costs fall inside the timing window and are not negligible at small transfer sizes.
- **Only the Active PE initiates; source and destination share one address**: `put` / `get` are one-sided operations, all issued by the Active PE. The Passive PE only acts as the remote object and performs no computation (the two cards synchronize through host-side barriers). The GM-side address lies at the same symmetric memory offset on both cards.
- **The UB data comes from local GM**: Before timing starts, the `put` test uses `DataCopyPad` to move data from the local symmetric memory into UB. Because the symmetric memory of one PE is filled with a single value, this move is only needed once, outside the measurement loop.
- **Warmup and averaging over multiple rounds**: Each measurement first runs a fixed 128 transfers as warmup (excluded from the statistics), then runs `loops` transfers for sampling. The timing works as follows: after the warmup ends, the whole batch of `loops` transfers is timed as one window (the start marker is taken once before the first sampled transfer and the end marker once after the last). The resulting total duration is then divided by `loops` to obtain the per-transfer cost, which amortizes the marker overhead and rules out cold-start effects.
- **Logical segments and the logical ring**: Each Core is physically allocated a 1 MB block of symmetric memory. Let the size of a single transfer be $X$ bytes and let $N$ Cores be used. Each Core's **logical segment** size is defined as $L = \min(1\text{MB},\ (128 + loops) \times X)$ (never smaller than $X$). The $N$ logical segments are placed end to end to form a **logical ring** of size $N \times L$, where Core $j$'s logical segment starts at offset $j \times L$ within the ring.
- **Sliding-window traversal**: On iteration $i$, all Cores together form a sliding window. The GM-side offset of Core $j$'s transfer for this iteration is $\big(i \times X + j \times L\big) \bmod (N \times L)$, that is, the Cores are staggered one logical segment apart from each other on the ring. The window advances by $X$ bytes per round and wraps around within the ring, so the addresses of consecutive transfers keep moving forward, which prevents data cache hits from inflating the bandwidth readings.
- **Result validation**: `put` is validated by the Passive PE. Because the sliding window fills the entire logical ring, validation compares the first $L$ bytes of each segment, segment by segment, following the logical-segment layout. `get` is validated by the Active PE: after timing ends, the kernel writes the UB contents back to the start address of the Core's own logical segment, so only the first $X$ bytes of each segment are compared.

## Source File Macro Configuration

Some test dimensions are controlled by constants defined at the top of `main.cpp`. You can modify the configurable ones before compiling to change the behavior of the API under test:

| Constant | Meaning and Effect | Allowed Values / Default |
| --- | --- | --- |
| `OP_TYPE` | The specific operation the performance test runs. | `OpType::Get` (default), `OpType::Put` |
| `T` | Element type of the transferred payload. `DATA_SIZE` is derived from it. | Fixed at `int32_t` |
| `DATA_SIZE` | Data width, in bits, of the underlying RMA API being called. Derived from `sizeof(T) * 8`; not configured separately. | Fixed at `32` |
| `THREAD_COUNT` | Number of threads launched per Core in SIMT mode, which determines the concurrency scale of the vector instruction stream. | Default `1024` |
| `UB_BUFFER_SIZE` | Number of elements the UB buffer can hold, which determines the upper limit of a single transfer. | Default `16384` (that is, 64 KB, corresponding to $2^{16}$ bytes) |
| `WARMUP_LOOPS` | Number of warmup rounds (excluded from the statistics). | Default `128` |

> **Note**: After modifying the constants above, you must return to the root directory and recompile (see below) for the new configuration to take effect.
> The data width is fixed at 32 bits: a `static_assert` in `main.cpp` rejects other values, and `transfer_vf_put` / `transfer_vf_get` directly call `simt::aclshmemx_int32_{put,get}_nbi_block`, so changing `T` alone does not switch to the APIs of another width.
> The size of a single transfer is limited by the UB capacity: with the default configuration the upper limit is $2^{16}$ bytes, so the values of `-e`/`--exponent-range` are restricted to $[3, 16]$ and anything outside reports an error and exits. To test larger data volumes, increase both `UB_BUFFER_SIZE` in `main.cpp` and `BYTES_IN_EXP_UPPER` in `argparser.h`, then recompile.
> This example supports SIMT mode only; it provides no SIMD mode for comparison.

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
   cd examples/shmem_perftest/simt_rma_ub2gm_perftest
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
| `-fpe <int>` | Kept for compatibility with the shmem_perftest options and unused by this example: the PE ranks are fixed at 0 and 1, and the device id is derived from `pe_id % gnpus + fnpu`, so neither is affected by this option. | 0 |
| `-t`/`--test-type <put\|get>` | Optional consistency check. If given, it must match the compile-time `OP_TYPE`, otherwise the binary reports an error. | - |
| `-b`/`--block-size <int>` | Number of Cores (blocks) used per PE. | 32 |
| `--block-range <min> <max>` | Sweep range for the number of Cores (blocks). Each core count produces its own statistics. | 32 32 |
| `--block-list <b1,b2,...>` | Explicitly specifies the core counts to test, separated by commas (for example, `1,8,16`). When given, it takes precedence over `-b`/`--block-size` and `--block-range`. | - |
| `--loop-count <int>` | Number of sampled iterations. Must be in $[1, 10000)$; out-of-range values report an error and exit. | 1000 |
| `-e`/`--exponent <exp>` | Exponent of the single-transfer data volume (a single value), as a power of 2 (for example, `10` means $2^{10} = 1024$ bytes). | - |
| `--exponent-range <min> <max>` | Exponent range of the single-transfer data volume. The values must fall within $[3, 16]$. | 3 16 |
| `-h`/`--help` | Print the option descriptions and exit. | - |

> This test uses a fixed two-card model (Active PE0 / Passive PE1). Both the number of started processes and the number of PEs inside the program are fixed at 2. `-pes` and `-gnpus` are kept for compatibility with the shmem_perftest options, but passing any value other than 2 reports an error.
> The core count must fall within $[1, 64]$: the upper bound is the number of Core slots contained in each PE's profiling buffer (`ACLSHMEM_CYCLE_PROF_MAX_BLOCK`), and exceeding it reports an error and exits.
> The test sweeps the single-transfer data volume exponent by exponent from the `--exponent-range` min to max (that is, $2^{min}, 2^{min+1}, \dots, 2^{max}$ bytes), and iterates core count by core count over the set given by `--block-list` (or `--block-range`). Each (core count, data volume) combination produces one row of statistics.

For example, to test the performance of `4` Cores transferring $2^8$ to $2^{12}$ bytes:

```bash
bash run.sh -b 4 --exponent-range 8 12
```

To sweep over several core counts:

```bash
bash run.sh --block-list 1,8,16,24,48 --exponent-range 4 16
```

### Performance Output

After the test finishes normally, the Active PE (PE 0) writes one `.csv` statistics file into the `output/` subdirectory of the example directory. The file name format is:

```bash
ub2gm_[DATA_SIZE]_[blocks]_[OpType]_simt_[minExp]-[maxExp]_l[loop_count]_t[THREAD_COUNT]_ub[UB_BUFFER_SIZE].csv
```

The `[blocks]` field reflects the set of core counts actually tested. The naming rule depends only on the set itself, not on which option specified it: if the set happens to be one contiguous ascending range, it is written as `min-max` (for example, both `--block-range 1 4` and `--block-list 1,2,3,4` become `1-4`); otherwise the values are joined with `_` in test order (for example, `--block-list 2,4,8` becomes `2_4_8`, and `--block-list 8,2,4` becomes `8_2_4`).

The columns of the `.csv` file have the following meanings:

| Column | Description |
| --- | --- |
| `DataSize/B` | Data volume of a single RMA transfer, in bytes, corresponding to the $2^{exp}$ value sampled in this row. |
| `Npus` | Number of PEs taking part in the test, which is 2 for a two-card test. |
| `Blocks` | Number of Cores (blocks) taking part in the communication, that is, the core count used by this row. Under a core-count sweep (`--block-range` / `--block-list`), each row takes its value from the swept set. |
| `UBsize/elements` | Number of elements the UB buffer can hold, that is, the compile-time `UB_BUFFER_SIZE`. |
| `Bandwidth/GB/s (1000)` | Average cross-card transfer bandwidth measured for this parameter set, converted using decimal units (divided by $1000^3$). |
| `Bandwidth/GiB/s (1024)` | The same bandwidth converted using binary units (divided by $1024^3$). |
| `CoreMaxTime/us` | Among all Cores, the latency of the Core with the longest average per-operation cost, in microseconds. This is the time used for the bandwidth computation. |
| `SingleCoreTime/us` | Average per-operation latency of each Core, in microseconds, obtained by dividing that Core's total duration over `loops` transfers by `loops`. There is one column per Core. |
