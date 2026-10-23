# Example of SHMEM Operator Integration with PyTorch

This directory provides an example of integrating selected SHMEM operators with PyTorch. This example is intended for demonstration purposes only and is not recommended for production environments!

The PyTorch extensions and the corresponding C++ operators share the same kernel implementation. **The running constraints, supported data types, and PE number limits are the same as those of the C++ examples.** Before using them, check the C++ example documents in the following table.

## Mapping Between Example Scripts and Operators

| Script| PyTorch Operator Class| Corresponding C++ Example Document|
|------|-------------|-------------------|
| `allgather.py` | `torch.classes.ShmemOps.AllGather` | [examples/allgather/README_en.md](../../allgather/README_en.md) |
| `kv_shuffle.py` | `torch.classes.ShmemOps.KVShuffle` | [examples/kv_shuffle/README_en.md](../../kv_shuffle/README_en.md) |

### Running Constraints of allgather.py

The constraints are the same as those of the [allgather C++ example](../../allgather/README_en.md). Key constraints are as follows:

- **PEs**: only `[2, 4, 8]` are supported.
- **dtype (supported by the operator)**: `torch.int32`, `torch.float16`, and `torch.bfloat16`. The example script uses `float16` by default and does not provide a data type switching option.

### Running Constraints of kv_shuffle.py

The constraints are the same as those of the [kv_shuffle C++ example](../../kv_shuffle/README_en.md). Key constraints are as follows:

- **PEs**: specified by `--pes`. The only constraint is that the value cannot exceed the number of available devices. The example uses 8 devices by default.
- **dtype (supported by the operator)**: The KV cache is `torch.int8`; `global_shuffle_tensor`, `src_block_tensor`, and `dst_block_tensor` are `torch.int64`. The example script uses the preceding types and does not provide a data type switching option.

## Build and Run

Run the following commands in the root directory of SHMEM:

```sh
# Build example operator cases and their PyTorch extensions.
# A2/A3 platforms
bash scripts/build.sh -python_example
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -python_example
source install/set_env.sh
cd examples/python_extension/torch_test
python xxx.py # Run the 8-device example by default.
python xxx.py --pes 2 # Run the 2-device example (--pes specifies the number of devices).
```

Note: Replace xxx with an actual file name.

## Parameter Description

### --tool

- `--tool 0`: Run directly without using the performance profiling tool. (This is the default value.)
- `--tool 1`: Use the msprof performance profiling tool.

**Note**: A memory check tool (for example, msSanitizer) and a profiling tool (for example, msprof) cannot be used simultaneously!

### --pes

- `--pes <N>`: the number of devices (PEs) to be used.
- **The constraints are the same as those of the corresponding C++ operators.** Before using them, check "Mapping Between Example Scripts and Operators" and the linked C++ example documents.

## Memory Check Tool Usage

To use the memory check tool (msSanitizer), perform the following operations:

1. **Add build options during the build**:

   ```sh
   # A2/A3 platforms
   bash scripts/build.sh -python_example -mssanitizer
   # Ascend950 platform
   bash scripts/build.sh -soc_type Ascend950 -python_example -mssanitizer
   ```

2. **Run the example using msSanitizer**:

   ```sh
   mssanitizer -- python xxx.py --pes 2
   ```

   Note: After adding the build options, you cannot directly run the example via `python xx.py`. You must run it via `mssanitizer -- python xx.py`.
