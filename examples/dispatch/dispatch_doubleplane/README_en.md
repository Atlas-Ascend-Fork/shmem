# Double-Plane MoE Dispatch Sample

> **Ascend950 is not yet supported**: Currently, the sample does not yet support compiling and running on the matched Ascend950 platform.

This sample implements the double-plane version of the non-quantized MoE dispatch. It keeps the same external semantics and output format as `examples/dispatch/dispatch_classic`, but enables both MTE and SDMA for payload transmission and adaptively selects the transport path based on the segment size.

The double-plane path depends on the SHMEM SDMA capability. The SDMA function requires CANN 9.0.0 or later, and requires the toolkit and ops-legacy packages matching the hardware platform to be installed. For basic installation and the standalone SDMA demo, refer to `examples/sdma/README.md`.

## Why Use the Double-Plane Version

In the classic dispatch, all token payloads are transmitted through MTE. For load balancing or small shapes, this method is simple and direct. However, MoE routing is naturally prone to skew, and some `(dst_rank, dst_local_expert)` segments concentrate a large number of tokens, forming large blocks of continuous remote data transmission.

The double-plane version aims to achieve the following:

- Small segments continue to use MTE to avoid the SDMA submission and synchronization overheads.
- Large segments use SDMA, leveraging the SDMA characteristic of being more suitable for moving large batches of remote data.
- The control plane still uses MTE to keep the ready/count/assist protocol simple and reliable.
- The output remains consistent with that of the classic dispatch externally, facilitating replacement and performance comparison.

Scenarios suitable for trying the double-plane version first:

- `bs`, `h`, and `topk` are large, and the total amount of remote payload is high.
- The expert routing distribution is uneven, and obviously large segments exist.
- You want to compare the benefits of the MTE-only path and the MTE+SDMA dual path in the communication stage.

If the shape is very small or the routing is very uniform, the double-plane version may not be significantly better than the classic version because the SDMA path itself has the fixed costs of issue, event, and quiet operations.

## Design

The host enables the following at the same time when initializing SHMEM:

```text
ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA
```

The kernel still starts `pe_size` AIV cores, and each active core is responsible for one target rank. For each `(dst_rank, dst_local_expert)` segment, the core first counts the tokens of the segment and then determines whether to enable SDMA.

The SDMA decision logic can be understood as:

```text
segment_bytes = token_count * h * sizeof(T)

use_sdma =
    dst_rank != my_rank &&
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

1. Each active core counts the `segment_counts` of the local expert segments sent to the target rank.
2. Calculate the average threshold based on the total number of remote tokens and mark `sdma_flags` for each segment.
3. Submit all SDMA payloads first: use `aclshmemx_sdma_put_nbi` to write the remote payload in a non-blocking manner.
4. Call `aclshmemx_sdma_quiet` after every 256 SDMA issues to prevent outstanding requests from accumulating without limit.
5. After the SDMA payload stage ends, if requests are still not quieted, call `aclshmemx_sdma_quiet` again in a unified manner.
6. The payload of small segments is transmitted through the MTE direct path.
7. Control plane signals are uniformly written by MTE, including `assist_info_for_combine`, the ready flags, and the count signals.
8. After the receiving end waits for count/ready, it constructs the final `expand_x`, `assist_info_for_combine`, `ep_recv_count`, and `expert_token_nums` in the order of the classic dispatch.

Note: SDMA is responsible only for the payload data plane of large segments. The control plane is still handled by MTE. This avoids the timing issue where the peer sees ready while the payload is not yet complete.

## Relationship with the Classic Dispatch

The inputs and outputs of the double-plane version are the same as those of the classic dispatch:

- A fixed-capacity symmetric SHMEM data window.
- Token-level ready flags.
- The output order is `(local_expert_id, src_rank_id)`.
- `assist_info_for_combine = [src_rank_id, src_token_id, src_topk_id]`.

Therefore, the same set of golden/check scripts can verify both paths. The functional correctness should be fully consistent with the classic dispatch, and the difference is mainly reflected in the communication stage performance.

## Build

Run the following command in the repository root directory:

```bash
bash scripts/build.sh -examples
```

## Run

Basic 2-card test:

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

8-card, 64-expert test:

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh -pes 8 -bs 8 -h 16 -topk 2 -expertPerPe 8 -type int32_t
```

The script automatically generates the inputs, routing matrix, and golden data, starts the process corresponding to each PE, writes the output to `output/`, and verifies the four outputs of the classic dispatch.

## Common Parameters

```text
-pes <n>            Number of PE/NPUs. The single-machine sample requires -gnpus to be the same as -pes.
-bs <n>             Number of tokens per PE.
-h <n>              Token hidden size.
-topk <n>           Number of experts that each token is routed to.
-expertPerPe <n>    Number of local experts per PE, in the range [1, 1024].
-type <dtype>       Data type. Supported values: int32_t, float16_t, and bfloat16_t.
-fnpu <id>          Starting NPU id. Default: 0.
-ipport <url>       SHMEM bootstrap address. Default: tcp://127.0.0.1:8766.
```

The upper limit of `expertPerPe` is 1024. The kernel allocates a fixed workspace on the AI core stack for each target local expert. Values exceeding this limit are rejected on the host side.

## How to Choose

It is recommended that you first establish the correctness baseline with the classic dispatch and then use the double-plane version to compare the performance of the same shapes.

The recommended comparison commands are as follows:

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh --perf -pes 8 -bs 32 -h 1024 -topk 2 -expertPerPe 8 -type int32_t --warmup 5 --loops 50

cd ../dispatch_doubleplane
bash scripts/run.sh --perf -pes 8 -bs 32 -h 1024 -topk 2 -expertPerPe 8 -type int32_t --warmup 5 --loops 50
```

If `comm_only` decreases, transmitting large payload segments through SDMA is effective for the communication stage. If the `full_op` benefit is not obvious, make a judgment based on the shape, the routing skew degree, and the subsequent compaction and synchronization costs.

## Performance Test

Single-shape profiling:

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh --perf -pes 2 -bs 8 -h 256 -topk 2 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

Multi-shape, multi-card sweep:

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16,32 --h-list 64,256,1024 \
    --topk-list 2 --expert-per-pe-list 2,8 -type int32_t --prof-pe all \
    --warmup 5 --loops 50
```

The CSV metrics include:

- `full_op`: the complete dispatch, including communication, metadata construction, compaction, and synchronization.
- `comm_only`: the Stage 1 payload communication and the necessary metadata/status protocols.

The single-rank file name is `dispatch_doubleplane_perf_rank<rank>.csv`. When `--prof-pe all` is used, the script profiles each PE in turn and generates `dispatch_doubleplane_perf_summary.csv`.
