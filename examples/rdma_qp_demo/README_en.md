# RDMA Multi-QP Demo

## Overview

This example demonstrates concurrent regular and aggregated Put/Get transfers between peer PEs using multiple RDMA QPs. It supports the XSCALE and HNS_1825 backends on Ascend950.

- `put`: write data from the local symmetric buffer to the next PE.
- `get`: read data from the next PE's symmetric buffer.
- `aggregate_put`: submit multiple Put operations as one batch on each QP.
- `aggregate_get`: submit multiple Get operations as one batch on each QP.
- `all`: run all four operations in sequence.

All PEs must use the same QP count, from 1 to 56. Each QP processes 512 KiB. Aggregate operations split each QP slice into eight chunks.

## Requirements

RDMA NIC, driver, CANN, and `IBV_EXTEND_DRIVERS` requirements are described in the [RDMA demo requirements](../rdma_demo/README_en.md#environment-requirements).

## Build and Run

Build the examples with an RDMA backend enabled:

```bash
# XSCALE
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE -examples
# HNS_1825
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend HNS_1825 -examples
```

Run the demo directly (profiling is disabled by default):

```bash
bash examples/rdma_qp_demo/run.sh
bash examples/rdma_qp_demo/run.sh -fnpu 0 -qp 4 -op put
```

To profile each PE with msprof, pass `-tool msprof`. `-n N` controls how many times each selected operation's kernel is launched; it defaults to 1. Initialization and buffer preparation happen once per operation, and the final iteration is checked.

```bash
bash examples/rdma_qp_demo/run.sh -qp 4 -op put -tool msprof -n 100
```

Capture data is written to `examples/rdma_qp_demo/output/<timestamp>/pe_<PE_ID>`. For multi-server runs, all servers must use the same `-pes`, `-qp`, `-op`, `-n`, and `-ipport`; `-ipport` must point to the server running PE 0.

Extract operation count, total time, and average time for all PEs in a capture:

```bash
python3 examples/utils/extract_totaltime.py \
    examples/rdma_qp_demo/output/<timestamp>
```

The result is saved as `totaltime.csv` in the capture directory, and the script prints the full output path when it finishes. Use `-o FILE` to choose another path. The output columns are `PE`, `Op Name`, `N` (call count), `Total Time`, and `Avg Time`; time units follow the msprof CSV. Python 3 is required. The script uses only the Python standard library, so no third-party packages need to be installed.

Per-PE rows also include `Min Time` and `Max Time` for individual kernel launches. An `ALL` row per operation reports `Mean PE Avg Time` (mean of PE averages), `Max PE Avg Time` (maximum PE average), and `Slowest PE`.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `-pes` | `2` | Global PE count; must be at least 2 |
| `-fpe` | `0` | First global PE started on this host |
| `-gnpus` | `2` | Number of local NPU/PE processes |
| `-fnpu` | `0` | First logical NPU ID used on this host |
| `-qp` | `2` | QPs per peer; range 1 to 56 |
| `-op` | `all` | `put`, `get`, `aggregate_put`, `aggregate_get`, or `all` |
| `-ipport` | Random local port | SHMEM bootstrap address |
| `-tool` | `none` | Use `msprof` to profile or `none` to run directly |
| `-n` | `1` | Kernel launches per selected operation; must be positive |

The program validates the received data and reports the first mismatch. The run script waits for all local PE processes and returns a non-zero status if any PE fails.
