# MoE Combine Sample Directory

This directory aggregates the combine-related samples. The shared data generation and verification scripts are stored in the parent `scripts/` directory, and the operator implementations are stored in separate subdirectories.

## Directory Structure

```text
examples/combine/
├── scripts/
│   ├── data_gen.py          # data generation script shared by classic and doubleplane
│   └── check_combine.py     # result verification script shared by classic and doubleplane
├── combine_classic/
│   ├── main.cpp
│   ├── combine_kernel.cpp
│   └── scripts/run.sh
└── combine_doubleplane/
    ├── main.cpp
    ├── combine_doubleplane_kernel.cpp
    └── scripts/run.sh
```

## How to Use

Classic combine:

```bash
cd examples/combine/combine_classic
bash scripts/run.sh -pes 2 -bs 8 -expertPerPe 2 -type int32_t
```

Double-plane combine:

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

Each `run.sh` generates `golden/` and `output/` in the current case directory, and calls the shared parent scripts `../scripts/data_gen.py` and `../scripts/check_combine.py`.

## Selection Recommendations

First use `combine_classic` to establish the correctness and performance baseline. When the shape is large and the returned payload contains obviously large segments, use `combine_doubleplane` to compare the `comm_only` and `full_op` metrics.

## Card Number Limitations

The sample starts the same number of AIVs as `pe_size` and configures the same number of SDMA QPs. The underlying implementation supports a maximum of 72 AIVs/QPs (`ACLSHMEM_MAX_AIV_PER_NPU`).

When performance profiling is enabled (by setting the environment variable `SHMEM_CYCLE_PROF_PE`), the `SHMEMI_PROF` profiling array has only 64 block slots (`ACLSHMEM_CYCLE_PROF_MAX_BLOCK`). Therefore, ensure that `pe_size` does not exceed 64. The AIV profiling data beyond this limit cannot be recorded, which may cause incomplete performance data. This restriction does not apply when performance profiling is disabled.
