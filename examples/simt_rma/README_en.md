## Overview

This example demonstrates how to use the SIMT remote memory access (RMA) APIs in SIMD and SIMT hybrid compilation mode, and also demonstrates the SIMT atomic compare-and-swap (CAS) API. The RMA APIs mainly come in the following three forms:

1. `__simt_callee__ inline void aclshmem_{NAME}_{op}(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)`
2. `__simt_callee__ inline void aclshmem_{op}{BITS}(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe)`
3. `__simt_callee__ inline void aclshmem_{op}mem(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)`

The core function of all three forms is to transfer data over a contiguous memory region. They differ in how the data length is specified:

- **The first form**: Specifies the length based on the concrete data type of each transferred element (such as `half`, `float`, and so on).
- **The second form**: Specifies the length based on the bit width of each transferred element (such as `8`, `16`, and so on).
- **The third form**: Directly specifies the total number of bytes to transfer.

Each form additionally comes in 3 thread-group variants, which differ in the range of threads that participate in the transfer:

| Variant | Description |
| --- | --- |
| `aclshmem_*` (no suffix) | Thread level. The calling thread transfers the whole region on its own. |
| `aclshmemx_*_block` | Block level. Threads within the same block cooperate on the transfer. |
| `aclshmemx_*_warp` | Warp level. Threads within the same warp (32 threads) cooperate on the transfer. |

**This example calls the warp-level variants.** The kernel is launched with `dim3(32)`, so the 32 threads of a warp cooperate to move one copy of the data, rather than each thread independently traversing the whole region. The APIs actually called for each of the three forms are:

```cpp
// Form 1: length described by the element data type
simt::aclshmemx_int16_get_warp(__gm__ int16_t *dst, __gm__ int16_t *src, size_t elem_size, int32_t pe);
simt::aclshmemx_int16_put_warp(__gm__ int16_t *dst, __gm__ int16_t *src, size_t elem_size, int32_t pe);
// Form 2: length described by the element bit width
simt::aclshmemx_get128_warp(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe);
simt::aclshmemx_put128_warp(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe);
// Form 3: total number of bytes specified directly
simt::aclshmemx_getmem_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe);
simt::aclshmemx_putmem_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe);
```

The following table lists the allowed values of the placeholders `{}` in the API names.

| Placeholder | Allowed Values |
| --- | --- |
| `{op}` | `put`, `get` |
| `{NAME}` | `half`, `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `char`, and `bfloat16` |
| `{BITS}` | `8`, `16`, `32`, `64`, `128` |

### Sample Execution Process

This example demonstrates how the RMA APIs work through the following process:

1. **Environment initialization**: Each compute unit (PE) initializes 3 symmetric memory blocks of the same size. The first block is initialized to `[my_pe + 0, ..., my_pe + size - 1]`, while the second and third blocks are initialized to `-1`.
2. **GET operation demonstration**: Each PE calls the `get` API (warp-level variant) to pull data from the first memory block of the **previous PE** (in logical order) and write it into its own second memory block.
3. **PUT operation demonstration**: Each PE calls the `put` API (warp-level variant) to push data from its own first memory block to the third memory block of the **previous PE**. The target is the previous PE because, from its point of view, the current PE is its "next PE" — this way each PE's third block ends up holding the data of its own next PE.
4. **Result verification**: After the communication operations complete, each PE automatically compares the data in memory to verify the correctness of the data transfer — the second block must equal the previous PE's first block, and the third block must equal the next PE's first block.

> The sample code provides three functions — `test_put_get_type`, `test_put_get_bits`, and `test_put_get_mem` — corresponding to the three API forms above. `test_put_get_bits` is called by default; switch to another as needed.

## Atomic CAS Demonstration

After the RMA case, the sample appends an atomic CAS demonstration showing the following API:

```cpp
__simt_callee__ inline TYPE simt::aclshmem_NAME_atomic_compare_swap(
    __gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe);
```

`{NAME}` can be `int32`, `int64`, `uint32`, or `uint64`. Synchronous API: it atomically compares the value at `dest` with `cond`; if equal, `dest` is updated to `value`, otherwise it is left unchanged; the old value at `dest` is returned. `dest` may point to either the device symmetric memory or the HOST_SIDE symmetric memory.

The demonstration (using uint32) covers three checks:

| Case | Check |
| --- | --- |
| Slot semantics | A matching CAS returns the old value and replaces it; a mismatching CAS returns the current value without writing |
| Per-PE shared counter | All threads increment the same address through a CAS retry loop; final = threads × iterations (same-card atomicity) |
| PE0 global counter | All threads of all PEs increment the same address on PE0's heap; final = PEs × threads × iterations (cross-PE atomicity) |

The demonstration targets the device symmetric memory (the default of `aclshmemx_malloc`); the HOST_SIDE symmetric memory is equally supported by the API but is not covered by this sample. With `-pes 1` the global counter degrades to same-card contention; use `-pes 2` or more to cover the cross-PE case.

## Supported Devices

- Ascend 950

## Instructions

1. **Compile the project.**
   Run the compilation script in the root directory of SHMEM.

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **Run the simt_rma sample program.**
   Go to the example directory and run the execution script.

   ```bash
   cd examples/simt_rma
   bash run.sh
   ```

   By default, `run.sh` launches 2 independent processes, each corresponding to one PE bound to one device, and runs the sample using `build/bin/simt_rma`. Use the `-pes` option to specify the number of PEs (devices) to run with, for example, to run with 4 devices:

   ```bash
   bash run.sh -pes 4
   ```

   When running with multiple devices, make sure the number of NPU devices actually available in the environment is not less than the specified number of PEs.

On success each PE prints two `[SUCCESS]` verification lines, one for the RMA transfers and one for the atomic compare swap; any failed check prints the corresponding `[ERROR]` details followed by `[FAILURE]`.
