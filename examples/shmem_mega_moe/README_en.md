# SHMEM MegaMoE Sample (Ascend950)

## Sample Introduction

This sample demonstrates a SHMEM-native MegaMoE forward pipeline for Ascend950. The operator places token routing, cross-Rank data dispatch, two expert matrix multiplications, SwiGLU activation, and multi-expert result merging in the same device-side pipeline. This avoids returning to the host machine or re-organizing communication between stages.

The responsibilities of the components in the implementation are as follows:

- ACLSHMEM provides multi-Rank symmetric memory, remote address access, and global synchronization.
- AscendC AIV cores generate routing masks, quantize tokens, perform SwiGLU post-processing, and merge the final results.
- Catlass AIC cores perform the two MXFP8 GEMMs of the expert network.
- AIC and AIV cores cooperate through flags and double buffering, allowing matrix computation, post-processing, and communication to execute in a pipeline.

This directory currently provides only the Ascend950 implementation. It supports the following two quantization modes for tokens and weights:

| Running Mode | FP8 Data Format | Scale Format |
| --- | --- | --- |
| `arch35_e4m3` | E4M3 | E8M0 |
| `arch35_e5m2` | E5M2 | E8M0 |

## Computation Definition

Assume that the current Rank has `token_count` input tokens and each token selects `experts_per_token` experts. The system has `rank_count` Ranks in total, and each Rank deploys `local_expert_count` experts. Therefore, the global expert count is:

```text
global_expert_count = rank_count * local_expert_count
```

The global expert ID relates to the Rank where the expert resides and the local expert ID within that Rank as follows:

```text
expert_rank  = global_expert_id / local_expert_count
local_expert = global_expert_id % local_expert_count
```

For the input `X_e` received by expert `e`, the expert forward computation is:

```text
U_e = X_e * W1_e
[Gate_e, Up_e] = Split(U_e, 2)
A_e = SiLU(Gate_e) .* Up_e
Z_e = A_e * W2_e
```

The first projection outputs a width of `ffn_dim` and splits it equally into the Gate and Up parts. SwiGLU fuses the two parts into an activation with a width of `ffn_dim / 2` and feeds it into the second projection. The final output of the original token `t` is the weighted sum of its hit Top-K expert results based on the routing weights:

```text
Y_t = Sum(routing_weights[t, k] * Z_expert_ids[t, k], k=0..experts_per_token-1)
```

The main data and shapes are as follows. The number of rows each expert actually computes is determined by the routing result and cannot exceed `max_received_tokens`.

| Data | Type | Logical Shape | Description |
| --- | --- | --- | --- |
| `x` | BF16 | `[token_count, model_dim]` | Input tokens of the current Rank |
| `expert_ids` | INT32 | `[token_count, experts_per_token]` | Global expert ID corresponding to each route |
| `routing_weights` | FP32 | `[token_count, experts_per_token]` | Top-K expert merging weights |
| `weight1` | FP8 | `[local_expert_count, model_dim, ffn_dim]` | Weight of the first expert projection |
| `weight1_scale` | E8M0 | Block-wise scale | MXFP scale of the first projection |
| `weight2` | FP8 | `[local_expert_count, ffn_dim / 2, model_dim]` | Weight of the second expert projection |
| `weight2_scale` | E8M0 | Block-wise scale | MXFP scale of the second projection |
| `y` | BF16 | `[token_count, model_dim]` | Output of the current Rank after weighted merging |
| `expert_token_counts` | INT32 | `[local_expert_count]` | Number of tokens actually processed by each expert on the current Rank |

## Overall Data Flow

The main flow of the operator is in `Process()` in `kernel/arch35/shmem_mega_moe_kernel.h`. The data path can be summarized as:

```text
BF16 tokens of the current Rank
        |
        v
Generate routing masks for target Ranks -----> SHMEM symmetric memory of remote Ranks
        |
        v
BF16 -> FP8(E4M3/E5M2) + E8M0 Scale
        |
        v
Target Ranks count and receive tokens, generating routing metadata
        |
        v
Catlass GEMM 1: model_dim -> ffn_dim
        |
        v
AIV SwiGLU + requantization: ffn_dim -> ffn_dim / 2
        |
        v
Catlass GEMM 2: ffn_dim / 2 -> model_dim
        |
        v
Write back to the combinedTokens of the source Rank based on routing metadata
        |
        v
The source Rank accumulates Top-K results by routing_weights -> BF16 output
```

### 1. Generating Cross-Rank Routing Masks

`BuildSendMasks` traverses the `expert_ids` of the current Rank. It converts each route of each token into a target Rank and a target local expert, and generates token bitmaps grouped by target Rank. The bitmaps and the token counts to be sent are written directly to the `inboundMask` symmetric memory of the target Rank. This allows the receiver to know which tokens to pull without host machine involvement.

### 2. Quantizing Local Tokens

`QuantizeLocalTokens` divides the tokens among AIV cores. Each AIV core converts the BF16 input into E4M3 or E5M2 data specified by the running mode, generates the E8M0 scale, and writes the result to the `quantizedTokens` symmetric memory of the current Rank. Subsequent receiving Ranks can directly read the quantized data through ACLSHMEM remote addresses.

### 3. Receiving Tokens and Building Routing Metadata

Each local expert first runs `CountReceivedTokens` to count the tokens sent to that expert by all source Ranks. Then `BuildRoutingMetadataAndDispatch` moves tokens under the `max_received_tokens` limit and records the source Rank, source token index, Top-K slot, and routing weight for each valid route. This metadata determines the input rows of the expert GEMM and is also used to write the results back to the original tokens after the second projection.

### 4. First Expert Projection and SwiGLU

`RunFirstProjection` uses Catlass to perform the first MXFP8 GEMM. AIC cores write the results to the UB in chunks, and AIV cores consume these chunks to complete the following operations:

1. Split the output with a width of `ffn_dim` into Gate and Up;
2. Compute `SiLU(Gate) * Up`;
3. Requantize the BF16 activation into FP8 and generate the E8M0 scale required by the second GEMM;
4. Notify the AIC through double-buffering flags that the corresponding chunk is ready.

Therefore, the first GEMM, SwiGLU, and requantization do not need to write the complete intermediate tensor back to the global memory before launching a new operator.

### 5. Second Expert Projection

`RunSecondProjection` uses Catlass to perform the second MXFP8 GEMM, mapping the activation with a width of `ffn_dim / 2` back to `model_dim`. The current sample always uses `COMBINE_NO_QUANT` in the Host Tiling: the second projection output is passed to the Combine process in BF16 format, and the Combine result does not undergo additional quantization.

### 6. Writing Expert Results Back to Remote Ranks

After the second projection, the AIV maps each row of expert results back to the source Rank, source token, and Top-K slot based on the routing metadata, and writes them to the `combinedTokens` symmetric memory of the source Rank. The write-back process uses ping-pong flags to synchronize data production and consumption, preventing the reuse of buffers in different rounds from overwriting data that has not been read.

### 7. Merging Top-K Results

After all expert results return, the source Rank runs `CombineRoutedTokens`. It reads all routing results of each token, multiplies them by the corresponding `routing_weights`, accumulates the products, and converts the final result into BF16 for writing to `y`.

## AIC/AIV Division of Work and Synchronization

| Processing Unit | Main Work |
| --- | --- |
| AIC | Two Catlass MXFP8 GEMMs and matrix chunk scheduling |
| AIV | Routing mask generation, token quantization, routing data movement, SwiGLU, requantization, and the final Combine |

Each local expert corresponds to one set of computing resources. The runtime queries the AIC/AIV counts on the device and writes the result to the tiling. You can also override the core counts through environment variables for controlled testing. AIC and AIV cores coordinate through the following mechanisms:

- The first projection uses double buffering: the AIC produces GEMM chunks, and the AIV consumes the chunks and performs SwiGLU;
- The input-ready, activation-ready, and Combine-ready states are passed through separate flag areas;
- Routing dispatch is processed in waves with a fixed number of rows, preventing the entire token batch from waiting serially;
- Multi-Rank stage boundaries use ACLSHMEM synchronization to ensure that remote writes are visible to the receiver.

Before running, the program checks that `local_expert_count` is not greater than the number of available AICs. If resources are insufficient, the program reports an error directly. This prevents multiple local experts from incorrectly competing for the same set of AIC pipeline resources.

## Memory Layout

### SHMEM Symmetric Memory

Each Rank creates ACLSHMEM symmetric memory of the same size and layout. The core regions include:

| Region | Purpose |
| --- | --- |
| Control and synchronization area | Stores control information such as cross-Rank barriers and flags |
| `inboundMask` | Receives token routing bitmaps and counts written by other Ranks |
| `quantizedTokens` | Stores the quantized tokens of the current Rank for remote reading by target Ranks |
| `combinedTokens` | Receives token results returned by remote experts that have not yet been accumulated by weight |

Main buffers are aligned to 512 bytes to meet the requirements of data movement and matrix computation. By default, the sample requests 1 GiB of symmetric memory. Before running a large shape, check the capacity based on the Rank count, token count, and `max_received_tokens`.

### Kernel Workspace

The operator workspace offsets are computed in a unified manner by `memory_layout.h`. The workspace mainly contains:

| Region | Purpose |
| --- | --- |
| `receivedTokenData` | Expert token data received by the current Rank |
| `receivedTokenScales` | E8M0 scales corresponding to the received tokens |
| `activatedTokenData` | Quantized activations after SwiGLU, used by the second GEMM |
| `activatedTokenScales` | E8M0 scales corresponding to the quantized activations |
| `receivedTokenCounts` | Number of tokens received by each local expert |
| `routingMetadata` | Source Rank, token, and routing slot information required for the result write-back |
| `inputReadyFlags` | Ready flags for routing inputs by wave |
| `activationReadyFlags` | Ready flags for SwiGLU activation chunks |

`memory_layout.h` also reserves the `matmulOutput` and `rowGroupReadyFlags` layouts for the quantized Combine mode. The current command-line sample always uses the BF16 non-quantized Combine path, so these two optional regions are not used.

## Directory Structure

```text
examples/shmem_mega_moe/
|-- CMakeLists.txt                         # Compilation entry of the Ascend950 sample
|-- README.md                              # This document
|-- include/
|   |-- shmem_mega_moe_host.h              # Host resources, parameter parsing, and validation
|   |-- shmem_mega_moe_main.h              # Host entry declaration
|   `-- shmem_mega_moe_main_arch35.h       # Ascend950 initialization, tiling, execution, and validation
|-- kernel/
|   |-- arch35/
|   |   |-- shmem_mega_moe_kernel.h        # Main operator pipeline
|   |   |-- shmem_mega_moe_tiling.h        # Tiling data structures
|   |   |-- shmem_mega_moe_memory_layout.h # Symmetric memory and workspace layouts
|   |   |-- shmem_mega_moe_catlass_pipeline.h
|   |   |                                      # Pipelines of the two Catlass GEMMs
|   |   |-- shmem_mega_moe_swiglu_epilogue.h # SwiGLU and activation quantization
|   |   `-- shmem_mega_moe_combine_pipeline.h # Result write-back and Combine pipelines
|   |-- common/                             # Common math functions
|   `-- dispatch/                           # Routing dispatch and quantization functions
|-- main.cpp                                # Executable entry point
`-- scripts/run_shmem_mega_moe_arch35.sh    # Multi-Rank launch script
```

## Compilation

### Environment Dependencies

- `shmem_mega_moe` supports only Ascend950. Other platforms do not generate a usable target for this sample.
- Before compilation, load the CANN Toolkit environment: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`. If the Toolkit is installed in another directory, replace the path with the actual path.
- `-examples` checks `3rdparty/catlass`. If the directory does not exist, the build script tries to download the dependency from the Catlass repository. If the download source is inaccessible, prepare Catlass source code compatible with Ascend950 in advance.

### Building

Run the following commands in the SHMEM repository root directory:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -soc_type Ascend950
```

After a successful build, the executable is located at `build/bin/shmem_mega_moe`. `scripts/build.sh` clears and regenerates the `build/` and `install/` directories. Confirm that these directories do not contain local build artifacts that need to be kept.

During the CMake configuration stage, the sample scans the actual function declarations in the current CANN AscendC header files and automatically selects the old interfaces `ReadGmByPassDCache`/`WriteGmByPassDCache` or the new interfaces `ReadGmBypassDCache`/`WriteGmBypassDCache`. Re-run the preceding build commands after switching the CANN version. If you use an existing CMake build directory, explicitly override the automatic detection:

```bash
cmake -S . -B build -DSHMEM_MEGA_MOE_BYPASS_DCACHE_MODE=OLD  # Old interfaces
cmake -S . -B build -DSHMEM_MEGA_MOE_BYPASS_DCACHE_MODE=NEW  # New interfaces
```

## Running

### Launch Script

It is recommended that you use the script provided with the sample to launch all Ranks at once. The following command applies to two NPUs on the same machine. The script starts one process for each Rank and maps Rank 0 and Rank 1 to device 0 and device 1:

Run the following commands in the SHMEM repository root directory:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source install/set_env.sh

BIN="$PWD/build/bin/shmem_mega_moe" \
    RANK_SIZE=2 NPU_NUM=2 FIRST_NPU=0 \
    MODE=arch35_e5m2 \
    TOKEN_COUNT=256 MODEL_DIM=4096 FFN_DIM=1024 \
    EXPERTS_PER_TOKEN=6 LOCAL_EXPERT_COUNT=4 \
    WARMUP=5 LOOP=20 \
    bash examples/shmem_mega_moe/scripts/run_shmem_mega_moe_arch35.sh
```

Before running, confirm the following:

- All Ranks use the same `RANK_SIZE`, `IP_PORT`, shape, and `MODE`;
- `BIN` points to the built executable and `source install/set_env.sh` has been completed, so that the SHMEM dynamic libraries can be loaded;
- In the single-machine multi-process mode where each process uses one NPU, `NPU_NUM` should cover at least `RANK_SIZE`, and the combination of `FIRST_NPU` and `NPU_NUM` must correspond to actually available device IDs;
- When running across host machines, set `IP_PORT` to the same address and port accessible to all Ranks, and ensure that the network and port are opened. The script currently only starts local processes and does not orchestrate cross-host processes.

### Parameter Description

The environment variables supported by the script and their default values are as follows:

| Environment Variable | Default Value | Description |
| --- | --- | --- |
| `BIN` | `./shmem_mega_moe` | Path of the executable; use an absolute path when calling the script from another directory |
| `RANK_SIZE` | `2` | Number of Ranks |
| `IP_PORT` | `tcp://127.0.0.1:8766` | ACLSHMEM initialization address |
| `NPU_NUM` | `8` | Number of consecutive NPUs used for Rank mapping; the device ID is `FIRST_NPU + rank % NPU_NUM` |
| `FIRST_NPU` | `0` | Starting NPU ID used for Rank mapping |
| `MODE` | `arch35_e4m3` | FP8 mode |
| `TOKEN_COUNT` | `4` | Number of tokens per Rank |
| `MODEL_DIM` | `4096` | Hidden dimension |
| `FFN_DIM` | `1024` | Output dimension of the first projection |
| `EXPERTS_PER_TOKEN` | `2` | Number of routed experts per token |
| `LOCAL_EXPERT_COUNT` | `1` | Number of experts per Rank |
| `MAX_RECEIVED_TOKENS` | `0` | Reception limit of each local expert; `0` means automatic calculation |
| `WARMUP` | `0` | Number of warmup rounds |
| `LOOP` | `1` | Number of timed rounds |
| `AIC_NUM` | Not set | Overrides the AIC count within the hardware limit |
| `AIV_NUM` | Not set | Overrides the AIV count within the hardware limit |

The script records the process IDs of all Ranks and waits for them to finish. If any Rank fails, the script terminates the other still-running Ranks and returns a non-zero status. This prevents a communication error from being masked by a successful process.

When setting `AIC_NUM` or `AIV_NUM`, the override values must not exceed the hardware core counts detected at runtime and must keep the `AIC_NUM:AIV_NUM=1:2` MIX Kernel ratio. It is recommended that you set both variables, for example `AIC_NUM=4 AIV_NUM=8`, to avoid a ratio validation failure caused by overriding only one variable.

## Parameter Constraints

Before running, the host performs shape, resource, and capacity validation. The main constraints are as follows:

- `mode` must be `arch35_e4m3` or `arch35_e5m2`;
- `ffn_dim` must be integer-divisible by 2 so that the Gate/Up split can be performed;
- `experts_per_token <= rank_size * local_expert_count`;
- `local_expert_count` cannot exceed the number of AICs available at runtime;
- The computed lengths of the input, weights, scales, output, and workspace must not overflow;
- All Ranks must use the same shape, running mode, and ACLSHMEM initialization address;
- When `max_received_tokens=0`, the capacity is automatically set as follows:

```text
token_count * rank_size * min(experts_per_token, local_expert_count)
```

When `max_received_tokens` is explicitly set, the current numerical verification program requires it to be no less than the preceding capacity. Otherwise, host parameter validation fails directly and the multi-Rank kernel will not start.

## Correctness Verification

To make the sample results reproducible, the host constructs inputs with deterministic data:

- BF16 elements of the input `x` are initialized to `1`;
- Expert IDs are deterministically generated based on the token, Top-K slot, and Rank;
- All routing weights are initialized to `1 / experts_per_token`;
- FP8 weights are initialized to E4M3/E5M2 encodings that exactly represent `1`;
- E8M0 scales are initialized to the encoding that represents `1`;
- All input tokens are active in the current sample.

Two types of verification are performed after each round:

1. **Expert token count verification**: checks that the number of tokens actually received by each local expert is consistent with the host routing result;
2. **Element-wise numerical verification**: the CPU computes the analytical golden result based on the two projections, SwiGLU, and routing weights, and compares each BF16 output element one by one.

With the current all-1 inputs and weights, the theoretical result of a single expert path is:

```text
first_projection = model_dim
silu_value       = first_projection / (1 + exp(-first_projection))
expert_output    = silu_value * first_projection * (ffn_dim / 2)
```

E4M3 and E5M2 use different relative error thresholds: 5% for E4M3 and 15% for E5M2. The absolute error threshold is 1 for both. When verification passes, the log contains:

```text
verify=pass(expert_token,numeric_golden)
```

This verification does not depend on a fixed Rank count and also covers token counts that are not tile-aligned. For acceptance in the target environment, it is recommended that you cover at least 2, 4, and 8 Ranks, both FP8 modes, and multiple token counts.

## Performance Metrics

The sample outputs both Kernel Event and end-to-end timing:

| Metric | Statistics Scope |
| --- | --- |
| `kernel_event_avg_ms` | Device-side operator execution time enclosed by ACL Events |
| `e2e_avg_ms` | Total time from operator launch to Stream synchronization and completion of the cross-Rank host barrier |

The `warmup` rounds are not counted in the statistics, and `loop` controls the number of officially timed rounds. Use `kernel_event_avg_ms` to compare the performance of the operator itself and `e2e_avg_ms` to evaluate the complete invocation overhead. In multi-Rank scenarios, confirm the verification results of all Ranks and compare with the same shape, core count, and running mode.

## FAQs

### CMake Does Not Generate the `shmem_mega_moe` Target

Check that `SOC_TYPE=Ascend950` and `USE_EXAMPLES=ON`, and check whether `CATLASS_ROOT` points to the source directory containing the required Catlass header files. Also ensure that the host, tiling, and kernel header files that this sample depends on are complete.

### AscendC Header Files Not Found

Set a correct `ASCEND_HOME_PATH` and confirm that the CANN Toolkit version supports Ascend950 and the AscendC interfaces used by the sample.

### Multi-Rank Execution Blocked

Confirm that all Ranks have started, `rank_size` and `IP_PORT` are exactly the same, and Rank IDs are not duplicated. It is recommended that you use the launch script provided with the sample. If any process fails, check the logs of the other Ranks at the same time. The Rank that reports an error first is usually the root cause.

### Insufficient Symmetric Memory or Workspace

Reduce the token count, dimensions, or Rank count, or check whether `max_received_tokens` is set too large. If the explicitly set reception capacity is smaller than the lower limit required by numerical verification, the host reports an error directly. Before running a large shape, estimate the actual usage of each Rank based on the memory layout.
