## Overview

This example demonstrates how to use the SIMT remote memory access (RMA) APIs in SIMD and SIMT hybrid compilation mode. It demonstrates the transmission of a single scalar. The APIs mainly include the following two forms:

1. `__simt_callee__ inline void aclshmem_{NAME}_p(__gm__ TYPE *dst, const TYPE value, int pe)`
2. `__simt_callee__ inline TYPE aclshmem_{NAME}_g(__gm__ TYPE *src, int32_t pe)`

The following table lists the allowed values of the placeholder `{NAME}` in the API names.

| Placeholder| Allowed Values|
| --- | --- |
| `{NAME}` | `half`, `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `char`, and `bfloat16`|

The core function of these APIs is to transfer a single scalar:
- **`_p` APIs**: Directly write the specified scalar value to the target memory address of a specified compute unit (PE).
- **`_g` APIs**: Read a single scalar value from the source memory address of a specified compute unit (PE) and return it.

Unlike the contiguous-region RMA APIs, `_p` and `_g` are thread-level only — they have no `_block` or `_warp` variants. Accordingly, this example launches the kernel with `dim3(1)`.

### Sample Execution Process

This example demonstrates how RMA single-scalar APIs work through the following process:

1. **Environment initialization**: Obtain the ID (`mype`) of the current compute unit and calculate the ID of the next PE as the target peer.
2. **PUT (write) operation demonstration**: The current PE calls the write API (`aclshmem_int32_p`) to directly write the ID of the target PE as a scalar to the symmetric memory corresponding to the target PE.
3. **GET (read) operation demonstration**: The current PE calls the read API (`aclshmem_int32_g`) to pull the scalar value stored in the symmetric memory of the target PE back to the current PE and store it in the local memory.
4.  **Result verification**: After the communication operation is complete, each PE automatically compares the data in the memory to verify the correctness of data transmission.


## Supported Devices

- Ascend 950PR&950DT products

## Instructions

1. **Compile a project**.
   Run the compilation script in the root directory of SHMEM.
   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **Run the simt_rma_scalar sample program**.
   Go to the `examples` directory and run the execution script.
   ```bash
   cd examples/simt_rma_scalar
   bash run.sh
   ```

   By default, `run.sh` launches 2 independent processes, each corresponding to one PE bound to one device, and runs the sample using `build/bin/simt_rma_scalar`. Use the `-pes` option to specify the number of PEs (devices) to run with, for example, to run with 4 devices:

   ```bash
   bash run.sh -pes 4
   ```

   When running with multiple devices, make sure the number of NPU devices actually available in the environment is not less than the specified number of PEs.
