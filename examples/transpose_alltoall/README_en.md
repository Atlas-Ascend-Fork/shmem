# TransposeAllToAll Sample (Ascend950)

## Sample Introduction

This sample demonstrates a fused operator for Ascend950: **local 4D transpose (BNSD → BSND) +
AllToAll along the B axis**.

## Computation Definition

Each Rank inputs a tensor in BNSD layout with shape `[B, N, S, D]` and data type `float16_t` /
`float32_t` / `int32_t` (default `float16_t`), and performs the following operations:

1. Performs a local transpose from BNSD to BSND (swaps the N and S axes) to obtain
   `[B, S, N, D]`;
2. Performs AllToAll along the B axis.

The output shape of each Rank is `[B, S, N, D]` (BSND), and the content is the concatenation of
the transposed corresponding B slices of all Ranks. `B % rankSize == 0` is required.

## Code Structure

```
transpose_alltoall/
├── CMakeLists.txt
├── main.cpp
├── transpose_alltoall_device.h
├── transpose_alltoall_kernel.hpp
└── scripts/
    ├── gen_data.py
    ├── verify_results.py
    └── run.sh
```

All kernels are shipped with this sample. The transpose and fused scheduling are implemented in
`transpose_alltoall_kernel.hpp` / `transpose_alltoall_device.h`, and cross-Rank data transfer is
implemented directly based on the shmem device-side interfaces (`aclshmemx_mte_get_nbi`,
`shmem_ptr`, `aclshmem_signal_wait_until`, `aclshmemx_signal_op`, `aclshmem_fence`, and so on),
without depending on catccos/catlass.

## Build and Run

Prerequisite: the SOC is Ascend950. This sample is self-contained and does not require an
additional third-party source tree.

```bash
# 1. Build shmem (including examples, Ascend950)
bash scripts/build.sh -examples -soc_type Ascend950

# 2. Correctness run (the device list determines rankSize, for example "0,1" means 2 Ranks)
cd examples/transpose_alltoall
bash scripts/run.sh "0,1"

# 3. Specify the data type (optional, default float16_t)
bash scripts/run.sh -type float16_t "0,1"

# 4. Detailed output (prints input/golden/actual tensors)
bash scripts/run.sh "0,1" -v

# 5. Skip golden verification (runs the kernel only and does not call verify_results.py)
bash scripts/run.sh --skip-check "0,1"   # equivalent form: -nc / --no-check
```

## Constraints

Before startup, the host side performs the following checks (`CheckTiling` in `main.cpp`):

- The batch dimension `B` must be integer-divisible by the Rank count `rankSize`.
- Input-side read stride constraint: `S × D × sizeof(dtype)` cannot exceed `UINT32_MAX`
  (about 4 GB). The GM-side stride of the device-side input `DataCopyExtParams` is held as a
  32-bit byte count. If the product of `S` and `D` is too large, the value is narrowed and
  truncated, causing incorrect strides across `S`. The host-side `CheckTiling` rejects such
  shapes.
- Communication scheduling block count constraint: the total block count
  `totalBlocks = ceil(chunkSize / commSizeM) × rankSize` cannot exceed `INT32_MAX`. The
  in-kernel `totalSteps` is held as `int32_t` (aligned with the int32 signal protocol). If the
  value exceeds the limit, it is narrowed and truncated. The host-side `CheckTiling` rejects
  such shapes.
- The IPC workspace size (`WORKSPACE_STAGES` × `AICORE_NUM` × `BLOCK_SIZE_BYTES`) cannot exceed
  half of the total size of the symmetric memory buffers.
