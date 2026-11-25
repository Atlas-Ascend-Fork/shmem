# Double-Plane MoE Combine Sample

> **Ascend950 is not yet supported**: Currently, the sample does not yet support compiling and running on the matched Ascend950 platform.

This sample implements the double-plane version of the non-quantized MoE combine. It keeps the same input/output and verification semantics as `examples/combine/combine_classic`, but enables both MTE and SDMA in the expert output return stage and adaptively selects the transport path based on the segment size.

The double-plane path depends on the SHMEM SDMA capability. The SDMA function requires CANN 9.0.0 or later, and requires the toolkit and ops-legacy packages matching the hardware platform to be installed. For basic installation and the standalone SDMA demo, refer to `examples/sdma/README.md`.

## Why Use the Double-Plane Version

In the classic combine, all expert outputs are returned through MTE. For small shapes or uniform routing, the fixed overhead of MTE-only is low. However, when some source ranks correspond to many expert outputs, the returned payload forms large batches of remote writes.

The double-plane version aims to achieve the following:

- Small segments continue to use MTE to avoid the fixed SDMA overhead.
- Large segments use SDMA to improve the remote write efficiency of large payloads.
- Control signals such as status ready still use MTE to keep the synchronization protocol simple and reliable.
- The output `x_out` is fully consistent with that of the classic combine, facilitating functional verification and performance comparison.

Scenarios suitable for trying the double-plane version first:

- `bs`, `h`, and `topk` are large, and the total amount of data returned by combine is high.
- The `ep_recv_count` after dispatch shows that some `(local_expert_id, src_rank_id)` segments are obviously large.
- You focus on the `comm_only` performance of the communication stage and want to compare the difference between MTE-only and MTE+SDMA.

If every segment is small, the SDMA issue/event/quiet costs may offset the benefit.

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

## Design

The host enables the following at the same time when initializing SHMEM:

```text
ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA
```

The kernel still starts `pe_size` AIV cores, and each active core is responsible for one source rank. For each `(src_rank, local_expert)` segment, the core first obtains the token count of the segment through `ep_recv_count` and then determines whether to enable SDMA.

The SDMA decision logic can be understood as:

```text
segment_bytes = token_count * h * sizeof(T)

use_sdma =
    src_rank != my_rank &&
    token_count > 0 &&
    segment_bytes >= 2MB &&
    segment_bytes > the average remote segment size in bytes of the current PE
```

To avoid integer division truncation, the code uses cross multiplication:

```text
segment_bytes * threshold_den > threshold_num
```

where:

```text
threshold_num = remote_tokens * h * sizeof(T)
threshold_den = max(pe_size - 1, 1) * local_expert_num
```

That is, only a remote large segment that is larger than 2MB and larger than the average remote segment size of the current PE uses SDMA. Local segments, empty segments, small segments, and normal segments still use MTE.

## Implementation Logic

The main stages are as follows:

1. Each active core traverses all local expert segments of the source rank it is responsible for based on `ep_recv_count`.
2. Calculate the average remote segment size in bytes of the current PE and determine `use_sdma` for each segment.
3. For large segments, submit the SDMA payload first: use `aclshmemx_sdma_put_nbi` to write the payload back to the combine window of the source PE in a non-blocking manner.
4. Call `aclshmemx_sdma_quiet` after every 256 SDMA issues to prevent outstanding requests from accumulating without limit.
5. Write the payload of small segments through the MTE direct path and immediately write status ready.
6. After the SDMA payload stage is complete, call `sdma_quiet` in a unified manner, and then use MTE to write status ready for the SDMA segments.
7. The source PE waits until all topK status ready flags of each local token are set.
8. Perform the weighted summation based on `expert_scales`, write out `x_out`, and then clear the status and exit in synchronization.

Note: SDMA is responsible only for the payload data plane of large segments. The status ready control plane is still written by MTE. This ensures that the status does not become visible before the payload.

## Relationship with the Classic Combine

The inputs and outputs of the double-plane version are the same as those of the classic combine:

- It consumes `expand_x`, `assist_info_for_combine`, and `ep_recv_count`.
- It uses the same `expert_ids` and `expert_scales`.
- It produces `x_out` with the same semantics.
- It uses the same golden/check scripts to verify correctness.

Therefore, you can first establish the correctness baseline with the classic combine and then use the double-plane version to compare the performance of the same shapes.

## Build

Run the following command in the repository root directory:

```bash
bash scripts/build.sh -examples
```

## Run

Basic 2-card test:

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

8-card, 64-expert test:

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh -pes 8 -bs 8 -h 16 -topk 2 -expertPerPe 8 -type int32_t
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

`bfloat16_t` is currently not instantiated in the combine doubleplane sample. The reason is that the CANN 9.0 beta backend does not support the scalar bf16 cast required by the combine accumulation path, so the script actively rejects `-type bfloat16_t`.

## How to Choose

It is recommended that you compare the classic version and the double-plane version with the same set of shapes:

```bash
cd examples/combine/combine_classic
bash scripts/run.sh --perf -pes 8 -bs 32 -h 7168 -topk 8 -expertPerPe 8 -type int32_t --warmup 5 --loops 50

cd ../combine_doubleplane
bash scripts/run.sh --perf -pes 8 -bs 32 -h 7168 -topk 8 -expertPerPe 8 -type int32_t --warmup 5 --loops 50
```

Focus on `comm_only` first. If `comm_only` decreases, returning large segments through SDMA is effective for the communication stage. If the `full_op` benefit is small, analyze the result together with the status wait, weighted reduction, and synchronization overheads.

## Performance Test

Single-shape profiling:

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh --perf -pes 2 -bs 8 -h 256 -topk 2 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

Multi-shape, multi-card sweep:

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16,32 --h-list 64,256,1024 \
    --topk-list 2 --expert-per-pe-list 2,8 -type int32_t --prof-pe all \
    --warmup 5 --loops 50
```

The CSV metrics include:

- `full_op`: the complete combine, including the return communication, status wait, weighted reduction, cleanup, and synchronization.
- `comm_only`: the Stage 1 return communication and the necessary completion/status protocols.

The single-rank file name is `combine_doubleplane_perf_rank<rank>.csv`. When `--prof-pe all` is used, the script profiles each PE in turn and generates `combine_doubleplane_perf_summary.csv`.
