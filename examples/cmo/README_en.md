# Cache Maintenance Operation (CMO) Function Demonstration and Read Performance Test Example

## Function Description

This example demonstrates how to use the Cache Maintenance Operation (CMO) API of SHMEM to optimize the global memory (GM) access performance. The CMO API provides L2 cache management operations. It allows data to be prefetched from the GM to the L2 cache in advance, reducing data access latency and improving overall computing performance. The current implementation supports the existing A2/A3 platforms and Ascend950.

### L2 Cache Background

The Ascend AI Processor uses a multi-level cache architecture. The L2 cache is a level-2 cache located between the AI Core and the global memory (HBM) and has the following characteristics:

- **Capacity**: large-capacity high-speed cache (classic value for A2/A3: 192 MB)
- **Access speed**: The cache hit bandwidth is about 2 to 4 times the cache miss bandwidth.
- **Cache management**: Data can be loaded to the cache in advance to mask memory access latency.

By properly using CMO prefetch operations, the next batch of data can be prepared in advance while computation is ongoing, improving overall performance.

### Current Test Content

Each PE runs the following tests in sequence:

1. **Basic CMO demonstration without an explicit QP**
   - Allocates 1 MB of local GM.
   - Only AIV 0 calls `aclshmemx_cmo_nbi`, followed by `aclshmemx_sdma_quiet`.
   - This is an API usage demonstration only and is not written to the performance CSV files.

2. **GM-to-UB read bandwidth test**
   The following three prefetch modes are compared:
   - `NO_PREFETCH`: prefetches `trash_gm` to disturb/clear the cache; the actual copy target `cache_gm` is not prefetched.
   - `HOST_PREFETCH`: the host calls `aclrtCmoAsync` to prefetch the entire `cache_gm` region.
   - `DEVICE_BLOCK_PREFETCH`: each AIV prefetches its own `cache_gm` block inside the kernel.

   The current configuration copies 64 MB per loop and repeats the test 100 times. The current effective `n_blocks` value is 20. `copypad_size` ranges over powers of two from 8 B through 128 KB, with additional tests at 192 KB and 256 KB. The actual step size is at least 512 B. Results are written to `<PE_ID>_band.csv`.

3. **CMO latency test**
   The same set of prefetch sizes is used to test the single-core API and the explicit-QP API. The sizes are 512 B, 1 KB, 2 KB, ... 4 MB, plus an additional 96 MB case.
   - The single-core test fixes `nbi_blocks = 1`; only AIV 0 calls `aclshmemx_cmo_nbi`. Results are written to `<PE_ID>_cmo_nbi.csv`.
   - The explicit-QP test uses `aclshmemx_cmo_qp_nbi` with AIV counts `{1, 2, 4, 8, 16, 32, 40}`. Each AIV uses the QP with the same index. Results are written to `<PE_ID>_cmo_qp.csv`.

### Core APIs

#### CMO API (SHMEM Extension API)

```c
template <typename T>
void aclshmemx_cmo_qp_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                          __ubuf__ T *buf, uint32_t ub_size, uint32_t qp_idx, uint32_t sync_id);
```

- **Function**: Asynchronously triggers CMO operations on the device side and submits operation tasks to the STARS queue.
- **Parameter description**:
  - `src`: global memory address
  - `elem_size`: the number of elements
  - `cmo_type`: CMO operation type (Currently, only CMO_TYPE_PREFETCH is supported.)
  - `buf`: address of the temporary Unified Buffer
  - `ub_size`: Unified Buffer size (at least 64 bytes, 64-byte aligned)
  - `qp_idx`: explicitly selected SDMA QP; concurrent AIVs should use distinct QPs, and the index must be smaller than the configured QP count
  - `sync_id`: synchronization ID

##### CMO Operation Types

**Note**: Currently, SHMEM supports only the `CMO_TYPE_PREFETCH` operation.

- **CMO_TYPE_PREFETCH**: prefetch operation, which loads data from the global memory to the L2 cache in advance.
- **CMO_TYPE_WRITEBACK**: writeback operation, which writes the modified data in the L2 cache back to the global memory and retains a copy in the cache.
- **CMO_TYPE_INVALID**: invalidation operation, which discards the data blocks in the L2 cache.
- **CMO_TYPE_FLUSH**: flush operation, which forcibly writes the data in the L2 cache back to the global memory and removes the data from the cache.

#### SDMA Quiet API (SHMEM Extension API)

```c
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_quiet(AscendC::LocalTensor<T> &buf,
                                              uint32_t qp_idx, uint32_t sync_id);
```

- **Function**: Waits for the completion of operation tasks in the STARS queue for synchronization.
- **Parameter description**:
  - `buf`: address of the temporary Unified Buffer
  - `ub_size`: Unified Buffer size
  - `qp_idx`: must match the QP used for the corresponding CMO submission
  - `sync_id`: synchronization ID
- **Characteristics**: An SDMA flag task is delivered, and the flag is polled until the operations in the STARS queue are complete.

#### CMO API Without an Explicit QP (SHMEM Extension API)

```c
template <typename T>
void aclshmemx_cmo_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                       __ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);
```

- **Function**: Same as `aclshmemx_cmo_qp_nbi`, except that it always uses QP 0 and takes no `qp_idx` parameter.
- **Parameter description**: Same as `aclshmemx_cmo_qp_nbi`, but without the `qp_idx` parameter.
- **Completion wait**: Call `aclshmemx_sdma_quiet` (the version without a QP, which also drains QP 0 only).
- **Use case**: CMO operations executed by a single AIV. When multiple AIVs prefetch concurrently, use the explicit-QP API and assign a distinct QP to each AIV instead of contending for the fixed QP 0.

Before the performance test, this example runs a prefetch demo without a QP once (see the `cmo_pretech` kernel in `main.cpp`, invoked by `test_copy_perf`) to demonstrate the basic usage:

```c++
// In the kernel (AIV 0 only)
aclshmemx_cmo_nbi(src, size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
```

## Environment Requirements

The example does not scale a symmetric data region with the PE count; each process allocates its CMO test buffers locally. The current `run.sh` launch model supports up to 8 NPUs on one server. Each NPU supports up to 72 explicit QPs.

### Hardware Requirements
- Ascend AI Processor (Atlas 200I A2/A3, Atlas 300T A2/A3, Ascend950, etc.)
- Architecture compatibility: AArch64 and x86

### Software Dependencies
Refer to [CANN Version Description](../../docs/quickstart_en.md#software-and-hardware-versions) and [Compilation and Build Guide](../../docs/compilation_build_guide_en.md) to configure a CANN version that supports CMO.

| Platform | CANN Version Required for CMO | Toolkit Package | Ops Package |
| --- | --- | --- | --- |
| A2/A3 | CANN 9.0.0-beta.2 or later | Toolkit package 9.0.0-beta.2 or later: [Community Resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) | Ops package 9.0.0-beta.2 or later: [Community Resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) |
| Ascend950 | CANN 9.1.0 or later | Toolkit package 9.1.0: [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-toolkit_9.1.0_linux-x86_64.run) / [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-toolkit_9.1.0_linux-aarch64.run) | Ops package 9.1.0: [950 x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-950-ops_9.1.0_linux-x86_64.run) / [950 aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-950-ops_9.1.0_linux-aarch64.run) |

Install the toolkit and Ops packages in the same directory:

```bash
# Customize the CANN installation directory as required.
export INSTALL_PATH=/home/user/ascend
chmod +x Ascend-cann-toolkit_{cann_version}_linux-$(uname -m).run
chmod +x Ascend-cann-{soc_name}-ops_{cann_version}_linux-$(uname -m).run
./Ascend-cann-toolkit_{cann_version}_linux-$(uname -m).run --install --install-path=${INSTALL_PATH}
./Ascend-cann-{soc_name}-ops_{cann_version}_linux-$(uname -m).run --install --install-path=${INSTALL_PATH}
source ${INSTALL_PATH}/ascend-toolkit/set_env.sh
```

### Function Dependencies

**Important**: In this example, the CMO API `aclshmemx_cmo_nbi` on the device side depends on the SDMA function. You need to configure `attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA` by referring to example/sdma or example/cmo to start the SDMA engine.

### Platform Support

Ascend950 supports only CMO.

Read/write data transfers through the SDMA put/get interfaces are not currently supported. Therefore, SDMA put/get interfaces such as `aclshmemx_sdma_put_nbi` and `aclshmemx_sdma_get_nbi` are not applicable to Ascend950 and later platforms.

## Build Procedure

### 1. Build the example program

For general processes such as build environment configuration, source code compilation, and binary package installation, see [Compilation and Build](../../docs/compilation_build_guide_en.md).

```bash
cd shmem/
# A2/A3 platforms
bash scripts/build.sh -examples
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples
```

After the build is successful, the key outputs include:

- Executable file: `build/bin/cmo`
- SHMEM library: `build/lib/libshmem.so`

## Running Method

```bash
cd shmem/examples/cmo
bash run.sh -pes ${PEs} -type ${TYPE}
```

### Parameters

- PEs: the number of devices (NPUs) used for running the program, limited to a single server
- TYPE: type of the data to be transferred. Currently, the following data types are supported: int, uint8, int64, fp16, and fp32.

### Example: Using Two NPUs to Test int Data
```bash
bash run.sh -pes 2 -type int
```

## Output Results

### Console Output

When the program is running, the completion information of each PE is displayed:

```bash
PE 0 Finished!
PE 1 Finished!
[SUCCESS] demo run success in pe 0
[SUCCESS] demo run success in pe 1
```

### CSV File Output

The program generates the following CSV files in the `output/` directory:

#### 1. `{PE_ID}_band.csv` - Bandwidth Performance Test Results

The file contains the following columns:
- `loop_times`: number of loops (100 by default)
- `copy_size_per_loop`: size of data copied in each loop, currently 64 MB
- `blocks`: number of blocks used, currently 20
- `copypad_size`: data size of a single DataCopy operation
- `no_prefetch_time/us`, `host_prefetch_time/us`, `device_block_prefetch_time/us`: average copy time per AIV for each mode; the CSV stores the p50 of 100 samples
- `no_prefetch_band/Gbps`, `host_prefetch_band/Gbps`, `device_block_prefetch_band/Gbps`: sum of bandwidth across all AIVs; the CSV stores the p50 of 100 samples

#### 2. `{PE_ID}_cmo_nbi.csv` - Single-Core CMO Latency Results

- `loop_times`: number of measured loops, currently 100; one warmup iteration is excluded
- `blocks`: currently fixed at 1
- `cmo_size`: powers of two from 512 B through 4 MB, plus 96 MB
- `cmo_submit_time_p05/p50/p95/us`: submission time from entering `aclshmemx_cmo_nbi` until the API returns
- `cmo_execute_time_p05/p50/p95/us`: time from submission start until `aclshmemx_sdma_quiet` returns

Each row contains the p05, p50, and p95 of 100 samples. Since the non-QP path is issued only by AIV 0, its latency is measured only on AIV 0. The copy validation performed by the kernel is not an output metric in this CSV.

#### 3. `{PE_ID}_cmo_qp.csv` - Multi-QP CMO Latency Results

- `loop_times`: number of measured loops, currently 100; one warmup iteration is excluded
- `aiv_num`: number of concurrent AIVs/QPs: 1, 2, 4, 8, 16, 32, and 40
- `cmo_size`: powers of two from 512 B through 4 MB, plus 96 MB
- `cmo_qp_submit_time_avg/max/us`: each AIV first takes the p50 of its 100 submit samples; the average and maximum are then calculated across those per-AIV p50 values
- `cmo_qp_execute_time_avg/max/us`: each AIV first takes the p50 of its 100 execute samples; the average and maximum are then calculated across those per-AIV p50 values
- `cmo_qp_submit_time_p05/p50/p95/us` and `cmo_qp_execute_time_p05/p50/p95/us`: percentiles calculated from the participating AIVs' per-AIV p50 values
- `cmo_qp_*_core_<N>/us`: independent p05, p50, and p95 values for AIV/QP N; `N/A` is written for AIVs not participating in that row

Here, `submit` is the CMO API submission time, while `execute` is the time from submission start until the corresponding `aclshmemx_sdma_qp_quiet` returns. In the QP test, `qp_idx` equals the AIV index; applications must use the same QP for submission and quiet.

### Metric Collection

- All latency measurements use `AscendC::GetSystemCycle()` on the device and convert cycles to microseconds using the `cycle2us` value for the compiled target.
- The bandwidth kernel repeatedly performs `DataCopyPad` at a 512-byte granularity for each AIV. It calculates per-AIV copy time and bandwidth before aggregating across AIVs.
- Bandwidth time and bandwidth in the CSV are the p50 of 100 test samples, not the average of all raw samples.
- The NBI latency CSV directly calculates percentiles from 100 single-AIV samples.
- The QP latency CSV first calculates a p50 across 100 samples for each AIV, then calculates avg, max, p05, p50, and p95 from the per-AIV p50 values.

### Single-Core vs. Multi-QP API Recommendation

32 KB is an empirical usage guideline from the current tests, not a hard API limit. Final selection should be based on measurements on the target platform:

- For a single prefetch range of 32 KB or less, prefer `aclshmemx_cmo_nbi`, followed by `aclshmemx_sdma_quiet`. This path uses QP 0, has simpler call management, and is suitable for a single AIV or a range that cannot be split for concurrent execution.
- For a single prefetch range larger than 32 KB, when the data can be split into multiple independent ranges, consider having multiple AIVs call `aclshmemx_cmo_qp_nbi` concurrently. Each AIV should use a distinct `qp_idx` and call `aclshmemx_sdma_qp_quiet` with the same QP to wait for completion.
- Before using multiple QPs, configure enough QPs on the host with `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num)`. Do not use `aclshmemx_sdma_quiet` to wait for requests submitted to a non-zero QP, and do not let concurrent AIVs share a QP without coordination.
- If the data cannot be split into independent ranges, or concurrency overhead offsets the benefit, continue to use the single-core API. The 32 KB boundary is for API selection guidance only and does not guarantee that the multi-QP path is faster.

### Performance Metrics

- **Bandwidth**: used to measure the data transmission rate, in GB/s
- **Latency**: used to measure operation completion time, in microseconds
- **Percentile**: used to collect statistics on the distribution. p50 indicates the median.

## References

- [CANN Application Development API Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/index/index.html)
- [Memory Management aclrtCmoAsync](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/runtimeapi/aclcppdevg_03_0123.html)
