## Overview

This example demonstrates, in SIMD and SIMT hybrid compilation mode, how the SIMT remote memory access (RMA) APIs use UB (Unified Buffer) as an intermediate buffer for data movement. The sample code allocates a UB array on the device side inside a `__simt_vf__` function, then calls the SIMT RMA NBI APIs to transfer data between GM and UB.

This example mainly demonstrates the following API forms:

The UB↔GM RMA APIs are organized the same way as the GM↔GM ones: they come in three forms according to how the data length is specified (shown here for `get`; `put` moves data in the opposite direction):

1. `__simt_callee__ inline void aclshmem_{NAME}_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)`
2. `__simt_callee__ inline void aclshmem_get{BITS}_nbi(__ubuf__ void *dst, __gm__ void *src, size_t nelems, int32_t pe)`
3. `__simt_callee__ inline void aclshmem_getmem_nbi(__ubuf__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)`

- **The first form**: Specifies the length based on the concrete data type of each transferred element (such as `half`, `float`, and so on).
- **The second form**: Specifies the length based on the bit width of each transferred element (such as `8`, `16`, and so on).
- **The third form**: Directly specifies the total number of bytes to transfer.

The following table lists the allowed values of the placeholders.

| Placeholder | Allowed Values |
| --- | --- |
| `{NAME}` | `half`, `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `char`, and `bfloat16` |
| `{BITS}` | `8`, `16`, `32`, `64`, `128` |

Each form additionally comes in 3 thread-group variants, which differ in the range of threads that participate in the transfer:

| Variant | Description |
| --- | --- |
| `aclshmem_*` (no suffix) | Thread level. The calling thread transfers the whole region on its own. |
| `aclshmemx_*_block` | Block level. Threads within the same block cooperate on the transfer. |
| `aclshmemx_*_warp` | Warp level. Threads within the same warp (32 threads) cooperate on the transfer. |

In addition, every API has a synchronous (blocking) and an asynchronous version. The `_nbi` (non-blocking initiate) suffix marks the asynchronous version, which returns before the transfer has completed.

**This example uses the asynchronous, warp-level combination.** The kernel is launched with `dim3(32)`, so the 32 threads of a warp cooperate to move one copy of the data, rather than each thread independently traversing the whole region. The APIs actually called are:

```cpp
simt::aclshmemx_int32_get_nbi_warp(__ubuf__ int32_t *dst, __gm__ int32_t *src, size_t elem_size, int32_t pe);
simt::aclshmemx_int32_put_nbi_warp(__gm__ int32_t *dst, __ubuf__ int32_t *src, size_t elem_size, int32_t pe);
```

Where:

| Parameter | Description |
| --- | --- |
| `dst` | Destination address. In `get_nbi_warp`, this is a UB address; in `put_nbi_warp`, this is a GM address. |
| `src` | Source address. In `get_nbi_warp`, this is a GM address; in `put_nbi_warp`, this is a UB address. |
| `elem_size` | Number of `int32_t` elements to transfer. |
| `pe` | ID of the target or source PE. |

### Sample Execution Process

This example demonstrates the UB-to-GM RMA data path through the following process:

1. **Environment initialization**: Each PE initializes 3 symmetric memory blocks of the same size. The `origin` data is initialized to `[my_pe + 0, ..., my_pe + size - 1]`, while `res_prev` and `res_next` are initialized to `-1`.
2. **Local GM to UB**: Each PE uses `aclshmemx_int32_get_nbi_warp` to read the data in its own `origin` into the UB buffer.
3. **UB to remote GM**: Each PE uses `aclshmemx_int32_put_nbi_warp` to write the data in the UB buffer into the `res_next` of the **previous PE**. The target is the previous PE because, from its point of view, the current PE is its "next PE" — this way each PE's `res_next` ends up holding the `origin` data of its own next PE.
4. **Remote GM to UB**: Each PE uses `aclshmemx_int32_get_nbi_warp` to read the `origin` data that logically belongs to the previous PE into the UB buffer.
5. **UB to local GM**: Each PE uses `aclshmemx_int32_put_nbi_warp` to write the data in the UB buffer into its own `res_prev`.
6. **Result verification**: After the communication operations complete, each PE copies the data back to the host and automatically verifies the transfer result — `res_prev` must equal the previous PE's `origin`, and `res_next` must equal the next PE's `origin`.

Steps 2-3 demonstrate "UB staging + remote write", while steps 4-5 demonstrate "remote read + UB staging". Together they cover all four data paths between UB and GM.

## Supported Devices

- Ascend 950PR&950DT products

## Directory Structure

```text
examples/simt_rma_ub2gm/
├── CMakeLists.txt
├── README.md
├── main.cpp
└── run.sh
```

## Instructions

1. **Compile the project.**

   Run the compilation script in the root directory of SHMEM.

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **Run the simt_rma_ub2gm sample program.**

   Go to the example directory and run the execution script.

   ```bash
   cd examples/simt_rma_ub2gm
   bash run.sh
   ```

   By default, `run.sh` launches 2 independent processes, each corresponding to one PE bound to one device, and runs the sample using `build/bin/simt_rma_ub2gm`. Use the `-pes` option to specify the number of PEs (devices) to run with, for example, to run with 4 devices:

   ```bash
   bash run.sh -pes 4
   ```

   When running with multiple devices, make sure the number of NPU devices actually available in the environment is not less than the specified number of PEs.

3. **Check the results.**

   After the sample finishes running, it prints a summary of the `origin`, `res_prev`, and `res_next` data for each PE. If verification passes, you will see output similar to the following:

   ```text
   [SUCCESS] PE 0: Verification passed for RMA transfers.
   [SUCCESS] PE 1: Verification passed for RMA transfers.
   ```
