## Overview

Based on the SHMEM project, this example describes how to use the SDMA batch data transfer APIs on the device kernel side to transfer data between regular device memory and host-side SHMEM symmetric memory.

The example supports three verification modes: `put`, `get`, and `all`.

- `put`: Verifies `aclshmemx_sdma_put_nbi`. Data is transferred from regular device memory on the local PE to `HOST_SIDE` SHMEM memory on the local PE and target PEs.
- `get`: Verifies `aclshmemx_sdma_get_nbi`. Data is transferred from `HOST_SIDE` SHMEM memory on the local PE and target PEs to regular device memory on the local PE.
- `all`: Executes `put` and then `get`.

## Supported Product Models

- Atlas A3 training products and Atlas A3 inference products

## Example Implementation

This example demonstrates how to use the SHMEM SDMA batch put/get APIs to transfer data between regular device memory and host-side SHMEM symmetric memory. These APIs are referred to as the SDMA `HOST_SIDE` put/get APIs in this document.

### Test Case Implementation

(1) Initialize [ACL](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_1945.html) and [SHMEM](../../README_en.md). Set the data path to `ACLSHMEM_DATA_OP_SDMA` during SHMEM initialization so that the device kernel can call the SDMA APIs.

(2) Allocate memory for the test data based on the runtime parameters. Each PE prepares a result area in the format `[PE0 segment][PE1 segment]...[PEn segment]`, with each segment storing data from one PE.

(3) In `put` mode, each PE prepares regular device source data `device_src`, whose value is `my_pe + 10`, and an all-zero `HOST_SIDE` SHMEM destination area `host_dst`. The kernel writes the local PE's `device_src` to the segment corresponding to the local PE in `host_dst` on the local PE and other PEs. The `pe == my_pe` case verifies D2H, and the `pe != my_pe` case verifies D2rH.

(4) In `get` mode, each PE prepares a `HOST_SIDE` SHMEM source area `host_src`, initializes only the segment corresponding to the local PE to `my_pe + 10`, and prepares an all-zero regular device destination area `device_dst`. The kernel reads the corresponding segments from `host_src` on the local PE and each remote PE and writes them to `device_dst` on the local PE. The `pe == my_pe` case verifies H2D, and the `pe != my_pe` case verifies rH2D.

(5) Call `aclshmem_barrier_all` before and after kernel execution to synchronize all PEs and ensure the correct ordering of data initialization, SDMA access, and result verification.

(6) After the kernel finishes, copy the results back to the host for verification. The result area of each PE must contain the segments from all PEs. For example, the PE0 segment contains 10, the PE1 segment contains 11, and the PE2 segment contains 12.

(7) Clean up and release SHMEM and ACL resources.

### Kernel Implementation

(1) On the kernel side, obtain the current PE ID `my_pe` and the total number of PEs `n_pes`.

(2) Calculate the number of bytes in each PE segment based on the number of elements and data type, and divide the data range among the kernel instances by block.

(3) In the `put` kernel, call `aclshmemx_sdma_put_nbi` to write data from regular device memory on the local PE to a `HOST_SIDE` SHMEM symmetric address on the local PE or a target PE. The API converts the host-side address based on the PE ID. The `pe == my_pe` case verifies local D2H.

(4) In the `get` kernel, call `aclshmemx_sdma_get_nbi` to read data from a `HOST_SIDE` SHMEM symmetric address on the local PE or a target PE and write the data to regular device memory on the local PE. The API converts the source host-side address based on the PE ID. The `pe == my_pe` case verifies local H2D.

(5) `aclshmemx_sdma_qp_put_nbi` and `aclshmemx_sdma_qp_get_nbi` are non-blocking APIs. In the kernel, call `aclshmemx_sdma_qp_quiet` with the same `qp_idx` to wait for completion.

## Build and Execution

For details about environment configuration, see [Quick Start](../../docs/quickstart_en.md). After the environment is configured, run the following commands to verify the function:

```bash
# Build the examples.
bash scripts/build.sh -examples -cann
cd examples/sdma_d2h_demo
# Run the default case in all mode.
bash run.sh
```

You can also specify the operation mode, data type, and number of elements:

```bash
bash run.sh -pes 2 -op all -type int -elems 1024
bash run.sh -pes 2 -op put -type uint8 -elems 1048576
bash run.sh -pes 2 -op get -type int64 -elems 262144
bash run.sh -pes 2 -op all -type fp32 -elems 262144
```

After the case is executed, if `[SUCCESS] put op pass in pe <my_pe>` is displayed, the `put` result verification on the current PE is successful. If `[SUCCESS] get op pass in pe <my_pe>` is displayed, the `get` result verification on the current PE is successful. If `[SUCCESS] sdma_d2h_demo run success in pe <my_pe>` is displayed, the example on the current PE is successfully executed and its resources are released properly.

The example also prints the first eight values in the segment for each PE. For example:

```text
[RESULT] pe 0 put segment 0 expected 10 first 8 values: 10 10 10 10 10 10 10 10
[RESULT] pe 0 put segment 1 expected 11 first 8 values: 11 11 11 11 11 11 11 11
```

If the first eight values in each segment match `expected`, the output shows that the data from each PE has been written to or read into the correct location.

### Runtime Parameters

`run.sh` supports the following parameters:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `-pes` | `2` | Total number of PEs. |
| `-op` | `all` | Operation mode. The options are `put`, `get`, and `all`. |
| `-type` | `int` | Data type. The options are `int`, `uint8`, `int64`, and `fp32`. |
| `-elems` | `1048576` | Number of elements on each PE. |

## Restrictions

### PE Count and Process Launch Requirements

`-pes` specifies the total number of PEs declared during SHMEM initialization. This example requires one process and one NPU for each PE. The script automatically starts the same number of processes as specified by `-pes` and binds each PE to the NPU with the same ID. The value of `-pes` cannot exceed the number of NPUs available in the current environment.

For example, to run four PEs, specify `-pes 4`:

```bash
bash run.sh -pes 4 -op put -type uint8 -elems 1048576
```

### HOST_SIDE Heap Size Constraint

The `HOST_SIDE` SHMEM heap of each PE must hold the segments of all PEs, with a total size of `-pes * -elems * sizeof(type)` bytes. The sample automatically derives the heap size from this value, rounding it up to 2 MB (huge-page aligned), with 16 MB minimum and 40 GB maximum (that is, `ACLSHMEM_MAX_LOCAL_SIZE`); no manual configuration is needed, and parameter validation fails when the upper limit is exceeded.

For example, with `-pes 8 -elems 1048576`, the heap for `int`/`fp32` and `int64` is automatically derived as 32 MB and 64 MB respectively (`uint8` requires 8 MB, and the 16 MB floor applies).
