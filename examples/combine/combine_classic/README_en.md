# Classic MoE Combine Sample

This sample implements the classic combine operator for non-quantized MoE, corresponding to the design document `DOC/moe_dispatch_combine_non_quant_architecture.md`. Combine consumes the expert outputs and auxiliary metadata generated in the dispatch stage, returns the result of each topK expert to the PE where the original token resides, and performs the weighted reduction.

## Function Description

The inputs include the dispatch outputs:

- `expand_x`: token data aggregated by expert after dispatch. The data generation script of this sample uses the identity expert computation, so `expand_x` can be directly regarded as the expert output.
- `assist_info_for_combine`: the source `[src_rank_id, src_token_id, src_topk_id]` of each expert output.
- `ep_recv_count`: the cumulative receive count of segments generated in the dispatch stage.

The inputs of combine itself are also required:

- `expert_ids`: the expert id corresponding to each token/topK pair.
- `expert_scales`: the weighting coefficient of each token/topK pair.

The output is as follows:

- `x_out`: the final combine result of each token.

The verification formula is as follows:

```text
x_out[token] = sum(topk_output[token, topk] * expert_scales[token, topk])
```

## Implementation Logic

The kernel starts `pe_size` AIV cores. In the send-back stage, the core with `core_id == src_rank` writes the expert outputs on the local PE back to the corresponding source rank.

The main procedure is as follows:

1. Traverse each `(local_expert_id, src_rank_id)` segment based on `ep_recv_count`.
2. Obtain the original `token_id` and `topk_id` from `assist_info_for_combine`.
3. Use the MTE `put_nbi` to write the expert outputs back to the combine window of the source PE.
4. Write the status ready flag to notify the source PE that the corresponding topK result is readable.
5. The local PE waits until all topK status ready flags of each local token are set.
6. Perform the weighted summation based on `expert_scales` and write out `x_out`.
7. Clear the status and perform inter-core synchronization.

The current classic version uses MTE for both the data plane and the control plane.

## Default Shape

```text
H = 7168
TopK = 8
BS, PEs, and expertPerPe are configurable
```

You can override the default values by using `-h`, `-topk`, or the performance sweep parameters.

## Build

Run the following command in the repository root directory:

- A2/A3 platforms:

```bash
bash scripts/build.sh -examples
```

- Ascend950 platform:

```bash
bash scripts/build.sh -soc_type Ascend950 -examples
```

## Run

Basic 2-card test:

```bash
cd examples/combine/combine_classic
bash scripts/run.sh -pes 2 -bs 8 -expertPerPe 2 -type int32_t
```

8-card, 64-expert test:

```bash
cd examples/combine/combine_classic
bash scripts/run.sh -pes 8 -bs 8 -expertPerPe 8 -type int32_t
```

float16 correctness test:

```bash
bash scripts/run.sh -pes 2 -bs 16 -expertPerPe 2 -type float16_t
```

TopK override:

```bash
bash scripts/run.sh -pes 2 -bs 8 -topk 4 -expertPerPe 2 -type int32_t
```

The script automatically generates the combine inputs and golden outputs, starts the process corresponding to each PE, writes the output to `output/x_out_<rank>.bin`, and runs the shared parent script `../scripts/check_combine.py` to verify the results.

## Common Parameters

```text
-pes <n>            Number of PE/NPUs. The single-machine sample requires -gnpus to be the same as -pes.
-bs <n>             Number of tokens per PE.
-h <n>              Token hidden size. Default: 7168.
-topk <n>           Number of experts that each token is routed to. Default: 8.
-expertPerPe <n>    Number of local experts per PE.
-type <dtype>       Data type. Supported values: int32_t and float16_t.
-fnpu <id>          Starting NPU id. Default: 0.
-ipport <url>       SHMEM bootstrap address. Default: tcp://127.0.0.1:8767.
```

`bfloat16_t` is currently not instantiated in the combine sample. The reason is that the CANN 9.0 beta backend does not support the scalar bf16 cast required by the combine accumulation path, so the script actively rejects `-type bfloat16_t`.

## Test Procedure

A functional test run performs the following steps:

1. Delete the old `golden/` and `output/` directories.
2. Use the shared parent script `../scripts/data_gen.py` to generate the inputs, dispatch metadata, and golden data.
3. Start the `combine` process of each PE.
4. Write out `output/x_out_<rank>.bin`.
5. Use the shared parent script `../scripts/check_combine.py` to compare the outputs of all ranks.

## Performance Test

Single-shape profiling:

```bash
cd examples/combine/combine_classic
bash scripts/run.sh --perf -pes 2 -bs 8 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

BS sweep:

```bash
bash scripts/run.sh --perf -pes 2 -type int32_t --bs-list 8,16,32 \
    --warmup 5 --loops 50
```

Multi-card, multi-shape sweep:

```bash
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16 \
    --expert-per-pe-list 2,8 -type int32_t --prof-pe all
```

The CSV metrics include:

- `full_op`: the complete combine, including the return communication, status wait, weighted reduction, cleanup, and synchronization.
- `comm_only`: the Stage 1 return communication and the necessary completion/status protocols.

The CSV files are written to `output/perf/`:

```text
combine_perf_rank0.csv
combine_perf_rank1.csv
combine_perf_summary.csv  # Generated when --prof-pe all is used
```

The first six columns are compatible with `examples/utils/perf_data_process.py`:

```text
DataSize/B,Npus,Blocks,UBsize/KB,Bandwidth/GB/s,CoreMaxTime/us
```

The additional columns include `Metric`, `GlobalDataSize/B`, `PerPeBandwidth/GB/s`, `BS`, `H`, `TopK`, `ExpertPerPe`, `Dtype`, `Warmup`, `Loops`, `ProfPe`, and `CaseId`.

You can use `--analyse plot` or `--analyse md` to call the unified performance report script after the performance test.
