# Classic MoE Dispatch Sample

This sample implements the classic dispatch operator for non-quantized MoE, corresponding to the design document `DOC/moe_dispatch_combine_non_quant_architecture.md`. Dispatch routes the token/topK pairs of the local PE to the PEs where the target experts reside based on `expert_ids`, and generates the auxiliary information required by the combine stage.

## Function Description

The inputs include:

- `x`: the token hidden states of the local PE.
- `expert_ids`: the global expert id corresponding to each token/topK pair.

The outputs include:

- `expand_x`: the token data aggregated by target expert.
- `assist_info_for_combine`: the auxiliary information used for combine return, in the format of `[src_rank_id, src_token_id, src_topk_id]`.
- `ep_recv_count`: the cumulative receive count arranged by `(local_expert_id, src_rank_id)`.
- `expert_token_nums`: the number of tokens actually received by each local expert.

The output order is fixed to `(local_expert_id, src_rank_id)`, facilitating subsequent combine and golden verification.

## Implementation Logic

The kernel starts `pe_size` AIV cores. In the send stage, the core with `core_id == dst_rank` writes to the symmetric window of the corresponding target PE.

The main procedure is as follows:

1. Each active core scans the `expert_ids` of the local PE and counts the number of tokens sent to each local expert of the target PE that the core is responsible for.
2. For each matched token/topK pair, write the payload to the SHMEM window of the target PE through the MTE `put_nbi`.
3. Write `assist_info_for_combine`, the token ready flags, and the segment count signals.
4. After all sending cores complete, the receiving end waits for the count signals of all source ranks.
5. Construct `ep_recv_count` and `expert_token_nums` based on the counts, and then compact the data into the final output in a deterministic order.

The current classic version prioritizes clear logic, verifiability, and a stable output order, and uses MTE for data plane transmission.

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
cd examples/dispatch/dispatch_classic
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

8-card, 64-expert test:

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh -pes 8 -bs 8 -h 16 -topk 2 -expertPerPe 8 -type int32_t
```

The script automatically generates the inputs, routing matrix, and golden data, starts the process corresponding to each PE, writes the output to `output/`, and verifies `expand_x`, `assist_info`, `ep_recv_count`, and `expert_token_nums`.

## Parameter Description

```text
-pes <n>              Number of PEs. The single-machine sample requires the value to be the same as -gnpus (default: 2).
-gnpus <n>            Number of NPUs started on the local machine. The value must be the same as -pes (default: 2).
-bs <n>               Number of tokens per PE (default: 8).
-h <n>                Token hidden size (default: 16).
-topk <n>             Number of experts that each token is routed to (default: 2).
-expertPerPe <n>      Number of local experts per PE, in the range [1, 1024] (default: 2).
-type <dtype>         Data type. Supported values: int32_t, float16_t, and bfloat16_t (default: int32_t).
-fpe <id>             Number of the first PE (default: 0).
-fnpu <id>            Starting NPU id (default: 0).
-ipport <url>         SHMEM bootstrap address (default: tcp://127.0.0.1:8766).
--perf                Performance test mode: retains the correctness verification and writes the performance CSV files to the --output-dir directory.
--warmup <n>          Number of warmup iterations of the performance test, which are not counted in the statistics (default: 5).
--loops <n>           Number of formal measurement iterations of the performance test (default: 50).
-pes-list <a,b,...>   List of PE counts scanned in performance test mode.
-bs-list <a,b,...>    List of token counts per PE scanned in performance test mode.
-h-list <a,b,...>     List of hidden sizes scanned in performance test mode.
--topk-list <a,b,...> List of topk values scanned in performance test mode.
--expert-per-pe-list <a,b,...>
                      List of local expert counts scanned in performance test mode.
--prof-pe <id|all>    PE ids for performance profiling. When the value is all, each PE is profiled in turn and the results are summarized (default: 0).
--output-dir <dir>    Output directory of the performance test CSV files (default: output/perf).
-a|--analyse <mode>   Performance result processing mode: plot (graphical display), md (generates a Markdown report), or none (no processing, default).
```

The upper limit of `expertPerPe` is 1024. The kernel allocates a fixed workspace on the AI core stack for each target local expert. Values exceeding this limit are rejected on the host side.

## Performance Test

`scripts/run.sh --perf` retains the correctness verification on each shape and writes the CSV files to `output/perf/`.

Single-shape profiling:

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh --perf -pes 2 -bs 8 -h 256 -topk 2 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

Multi-shape, multi-card sweep:

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16,32 --h-list 64,256,1024 \
    --topk-list 2 --expert-per-pe-list 2,8 -type int32_t --prof-pe all \
    --warmup 5 --loops 50
```

The CSV metrics include:

- `full_op`: the complete dispatch, including communication, metadata construction, compaction, and synchronization.
- `comm_only`: the Stage 1 payload communication and the necessary metadata/status protocols.

The single-rank file name is `dispatch_perf_rank<rank>.csv`. When `--prof-pe all` is used, the script profiles each PE in turn and generates `dispatch_perf_summary.csv`.
