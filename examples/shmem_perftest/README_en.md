# Sample Introduction

`shmem_perftest` is a collection of parameterized test samples for testing the performance of AscendC::DataCopy, the shmem MTE/UDMA/RDMA engines, and SIMT RMA interfaces. It contains the following sub-samples:

- **ascendc_perftest**: Tests AscendC::DataCopy performance (not supported on Ascend950)
- **mte_perftest**: Tests shmem MTE engine performance
- **sdma_perftest**: Tests shmem SDMA put/get performance (supports `put`, `bi_put`, `get`, and `bi_get`; only `get` is supported on Ascend950)
- **udma_perftest**: Tests shmem UDMA low-level interface performance (Ascend950 only)
- **rdma_perftest**: Tests shmem RDMA low-level interface performance (compile with RDMA support enabled)
- **rdma_atomic_perftest**: Tests the batch completion time, operation rate, and single completion latency of single-QP RoCE atomic operations (fixed at 2 PEs; the Yunmai XSCALE case is currently enabled)
- **simt_rma_perftest**: Tests the performance of the SIMT RMA gm2gm interface (Ascend950 only; compile with SIMT support enabled)
- **simt_rma_ub2gm_perftest**: Tests the performance of the SIMT RMA ub2gm interface (Ascend950 only; compile with SIMT support enabled)
- **simt_udma_perftest**: Tests the performance of the SIMT UDMA QP-specific interface (Ascend950 only; compile with SIMT support enabled)

This sample helps you compare the performance of multiple data transfer methods. **The script test results are for reference only; actual performance depends on the actual scenario**

## Python Dependencies

If you need to generate performance charts and Markdown reports, install the following Python dependencies:

```bash
pip install pandas matplotlib seaborn numpy tabulate
```

## Usage

### Quick Start

Compile and run in the shmem root directory:

```bash
# Compile the samples
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# A2/A3 platforms
bash scripts/build.sh -examples
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples

# To run simt_rma_perftest / simt_rma_ub2gm_perftest, enable SIMT support and specify Ascend950
bash scripts/build.sh -examples -enable_simt -soc_type Ascend950

# To run rdma_perftest, enable RDMA support (for RDMA parameters, see docs/compilation_build_guide.md)
# A2/A3 platforms
bash scripts/build.sh -examples -enable_rdma
# Ascend950 platform (specify the RDMA backend: XSCALE or HNS_1825)
bash scripts/build.sh -soc_type Ascend950 -examples -enable_rdma -rdma_backend HNS_1825

# To run --memory-type dram, compile with CANN mode enabled
# A2/A3 platforms
bash scripts/build.sh -examples -cann
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples -cann

# Run the compiled samples (default mode all, generates CSV only)
# Note: the all mode includes ascendc/mte/udma/simt (simt includes the gm2gm and ub2gm sub-samples),
#     excluding rdma_perftest (which needs to be run separately)
cd examples/shmem_perftest
bash run.sh -m all -t put -d float -fpe 0

# Run the compiled samples and generate performance charts
bash run.sh -m all -t put -d float -fpe 0 -a plot

# Run the compiled samples and generate performance charts and a Markdown report
bash run.sh -m all -t put -d float -fpe 0 -a md
```

### Parameters of the Outer-Layer run.sh

The following table lists the parameters supported by the outer-layer `run.sh` and their default values. **Sub-samples may differ in the supported range and default value of the same parameter** (for example, the optional values of `-t`, whether `-b` takes effect, and the minimum value of `--ub-size`). For details, refer to the README of each sub-sample:

| Parameter | Description | Outer-Layer Default Value |
|------|------|---------|
| `-t\|--test-type <type>` | Test type; optional values vary by engine (see the description below) | put |
| `-d\|--datatype <type>` | Data type (float\|int8\|int16\|int32\|int64\|uint8\|uint16\|uint32\|uint64\|char\|all) | float |
| `-b\|--block-size <size>` | Sets the core count (in UDMA and RDMA, the core count equals the QP count) | 32 for AscendC/MTE/SIMT; 1 for UDMA/RDMA |
| `--block-range <min> <max>` | Sets the core count range (UDMA/RDMA currently does not perform range scanning) | 32-32 |
| `--block-list <b1,b2,...>` | Sets a discrete core count list (effective for ascendc/mte/simt_rma_perftest/simt_rma_ub2gm_perftest), for example `2,4,6,8`. After being specified, this parameter takes precedence over `-b` and `--block-range` | - |
| `--qp <count>` | Sets the number of SDMA QPs/AIVs, ranging from 1 to the available AIV count on the device and not exceeding 72 | 2 |
| `-e\|--exponent <exponent>` | Sets the exponent of the data volume | - |
| `--exponent-range <min> <max>` | Sets the exponent range of the data volume | 3-20 |
| `--loop-count <count>` | Sets the loop count | 1000 |
| `--ub-size <size>` | Sets the UB size in KB; rdma_perftest requires at least 192B. For details, refer to the README of the sub-sample | 16 |
| `--memory-type <hbm\|dram>` | Sets the SHMEM memory type used by mte_perftest (effective only for mte) | hbm |
| `--batch <N>` | Batch control of the BW path; in UDMA, N > 1 uses defer/submit aggregation and performs quiet after each batch; for the aggregation rules and restrictions of RDMA, refer to the README of the sub-sample | 0 |
| `-pes <size>` | Sets the PE size | 2 |
| `-ipport <ip:port>` | Sets the IP address and port | tcp://127.0.0.1:8760 |
| `-gnpus <num>` | Sets the number of NPUs | 2 |
| `-fnpu <id>` | Sets the first NPU ID | 0 |
| `-fpe <id>` | Sets the first PE ID | 0 |
| `-m\|--mode <ascendc\|mte\|sdma\|udma\|simt\|all>` | Sets the running mode (sdma=runs SDMA put/get only); for the meanings of other modes, see the following text | all |
| `-a\|--analyse <none\|plot\|md>` | Sets the analysis mode (none=generates nothing, plot=generates charts only, md=generates both charts and md) | none |
| `-h\|--help` | Prints the parameter description and exits | - |

> **Differences in optional `-t/--test-type` values across engines**:
>
> - **ascendc_perftest**: `put` / `get` / `ub2gm_local` / `ub2gm_remote` / `gm2ub_local` / `gm2ub_remote` / `all`
> - **mte_perftest / rdma_perftest**: `put` / `bi_put` / `get` / `bi_get` / `all`
> - **sdma_perftest**: `put` / `bi_put` / `get` / `bi_get` (only `get` is supported on Ascend950)
> - **udma_perftest**: `put` / `bi_put` / `get` / `bi_get` / `put_signal` / `all`
> - **simt_rma_perftest**: `put` / `get` (determined by the compile-time constant `OP_TYPE`; `-t` only performs a consistency check)
> - **simt_rma_ub2gm_perftest**: `put` / `get` (same as above, determined by the compile-time constant `OP_TYPE`; `none` is not supported)
>
> If the specified `-t` is not supported by an engine, the outer-layer script skips that engine and prints WARN. For the detailed description and constraints of each parameter and the parameters unique to each sub-sample (such as `--metric` of udma and `--sync-id` / `-q` of rdma), refer to the README of the corresponding sub-sample.

### Running Modes

- **all (default)**: Runs the ascendc, mte, sdma, udma, and simt performance test cases that support the test type. Engines that do not support the test type are skipped.
- **ascendc**: Runs only ascendc_perftest (not supported on Ascend950; running on Ascend950 reports an error)
- **mte**: Runs only mte_perftest
- **sdma**: Runs only sdma_perftest
- **udma**: Runs only udma_perftest
- **simt**: Runs the two SIMT RMA sub-samples — simt_rma_perftest (gm2gm) and simt_rma_ub2gm_perftest (ub2gm). In this mode, a missing executable causes either sub-sample to exit with an error (in `all` mode, a WARN is printed and the missing sub-sample is skipped)

> Note: `rdma_atomic_perftest` uses independent parameters and a run script and does not participate in the mode expansion of the top-level `run.sh`. For details, see [rdma_atomic_perftest/README.md](./rdma_atomic_perftest/README.md).

> Note: `rdma_perftest` is not included in the preceding running modes. You need to enter the `rdma_perftest/` directory separately to run it, and enable the `-enable_rdma` parameter at compile time. For details, see [rdma_perftest/README.md](./rdma_perftest/README.md).

> Note: `simt_udma_perftest` has an independent `--scope` (warp/block) execution domain dimension. The default `all` expands the 8 execution domain × operation combinations in sequence. Its parameter model differs from the `-t`/`-d`/`-b` expansion of the top-level `run.sh` and does not participate in the top-level mode expansion. You need to enter the `simt_udma_perftest/` directory separately to run it. For details, see [simt_udma_perftest/README_en.md](./simt_udma_perftest/README_en.md).

### SIMT RMA Test Constraints

`simt_rma_perftest` uses two cards for performance testing, and `-pes` and `-gnpus` must be 2. The `OP_TYPE` and `DATA_SIZE` of this sub-sample are determined by the compile-time constants in `simt_rma_perftest/main.cpp`. The outer-layer `-t` / `-d` values are forwarded to the SIMT sub-script for a consistency check only when you explicitly pass them. They do not change the actually tested operation type or data bit width.

Therefore, `-t all` or `-d all` does not expand and run the SIMT matrix, and the top-level script skips `simt_rma_perftest` and prints a message. To test other SIMT RMA operations or data bit widths, modify the compile-time constants in `simt_rma_perftest/main.cpp` and recompile.

`simt_rma_ub2gm_perftest` also belongs to SIMT RMA and uses the same fixed two-card model. It runs with the `simt`/`all` modes, and the preceding compile-time constant constraints also apply (its constants are in `simt_rma_ub2gm_perftest/main.cpp`). It differs from the gm2gm version in two aspects. The outer-layer script handles the differences automatically, so you do not need to adjust parameters manually:

| Difference | Handling by the Outer-Layer Script |
| --- | --- |
| The single transfer volume is limited by the UB capacity, with an exponent upper bound of 16 (20 for gm2gm) | The portion beyond the bound is shrunk to 16, and a WARN is printed to describe the shrunk range; if the min of `--exponent-range` is already greater than 16, the sub-sample is skipped |
| The bit width is fixed at 32 bit, the UB size is determined by a compile-time constant, and its `run.sh` has no `-d` / `--ub-size` parameters | These two parameters are not forwarded to it; if the explicitly passed `-d` does not correspond to 32 bit (that is, other than `float`/`int32`/`uint32`), the sub-sample is skipped and a WARN is printed |

### DRAM Memory Test Constraints

`--memory-type dram` applies only to mte_perftest and uses `aclshmemx_malloc(..., HOST_SIDE)` to allocate DRAM memory on the host side. This feature depends on the CANN mode. At compile time, you must use:

```bash
# A2/A3 platforms
bash scripts/build.sh -examples -cann
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples -cann
```

The DRAM test requires the runtime environment to support host-side DRAM memory access. For the related hardware and available memory constraints, refer to the "Restrictions" section of [rma_d2h_demo](../rma_d2h_demo/README_en.md). mte_perftest configures 1GB of local memory by default. When the test parameters require more local memory, the program increases the memory automatically based on the data volume. Before running, ensure that the available DRAM space is greater than the actual local memory configuration.

## Output

After the run completes, the results are placed in the `examples/shmem_perftest/output/` directory:

```text
examples/shmem_perftest/output/
├── ascendc_perftest/    # ascendc_perftest test results
│   ├── 0_put_float.csv
│   └── picture/          # Performance charts (generated when --plot is used)
│       └── 0_put_float/
│           ├── 0_put_float_UBsize_compare.png
│           ├── 0_put_float_Core_compare.png
│           ├── 0_put_float_bandwidth_max_heatmap.png
│           └── 0_put_float_bandwidth_mean_heatmap.png
├── mte_perftest/         # mte_perftest test results
│   ├── put_float_0.csv
│   └── picture/          # Performance charts (generated when --plot is used)
│       └── put_float_0/
│           ├── put_float_0_UBsize_compare.png
│           ├── put_float_0_Core_compare.png
│           ├── put_float_0_bandwidth_max_heatmap.png
│           └── put_float_0_bandwidth_mean_heatmap.png
├── sdma_perftest/        # sdma_perftest test results
│   └── sdma_get_float_0.csv
├── udma_perftest/        # udma_perftest test results
│   └── udma_bw_put_float[_qpN]_0.csv
├── rdma_perftest/        # rdma_perftest test results (copied here after running rdma_perftest separately)
│   └── rdma_put_float_0.csv
├── simt_rma_perftest/    # simt_rma_perftest test results
│   └── 32_32-32_put_simt_3-20_l1000_t1024.csv
├── simt_rma_ub2gm_perftest/  # simt_rma_ub2gm_perftest test results
│   └── ub2gm_32_32-32_get_simt_3-16_l1000_t1024_ub16384.csv
└── performance_report.md  # Performance test report (generated when --markdown is used)
```

## Testing Sub-Samples Separately

If you need to test a sub-sample separately, enter the corresponding subdirectory and view the detailed README:

- **ascendc_perftest**: For details, refer to [ascendc_perftest/README_en.md](./ascendc_perftest/README_en.md)
- **mte_perftest**: For details, refer to [mte_perftest/README_en.md](./mte_perftest/README_en.md)
- **sdma_perftest**: For details, refer to [sdma_perftest/README_en.md](./sdma_perftest/README_en.md)
- **udma_perftest**: For details, refer to [udma_perftest/README_en.md](./udma_perftest/README_en.md)
- **rdma_perftest**: For details, refer to [rdma_perftest/README.md](./rdma_perftest/README.md)
- **rdma_atomic_perftest**: For details, refer to [rdma_atomic_perftest/README.md](./rdma_atomic_perftest/README.md)
- **simt_rma_perftest**: For details, refer to [simt_rma_perftest/README_en.md](./simt_rma_perftest/README_en.md)
- **simt_rma_ub2gm_perftest**: For details, refer to [simt_rma_ub2gm_perftest/README_en.md](./simt_rma_ub2gm_perftest/README_en.md)
- **simt_udma_perftest**: For details, refer to [simt_udma_perftest/README_en.md](./simt_udma_perftest/README_en.md)
