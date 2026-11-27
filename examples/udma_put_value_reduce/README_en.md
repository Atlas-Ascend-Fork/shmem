# Sample Description

## Version and Platform Support

This sample is supported only on Ascend950 and requires CANN 9.1.0 or later.

## Sample Overview

This sample demonstrates two low-level UDMA write operations:

- `aclshmemx_udma_put_value_nbi<T>`: writes one `value` to a remote symmetric address inline in the WRITE WQE. No local source buffer is required.
- `aclshmemx_udma_put_reduce_nbi<T, REDUCE_OP>`: reduces the local `src` vector into a remote symmetric address using `ADD`, `MAX`, or `MIN`.

Each PE writes one `uint64_t` inline value and 16 `int32_t` ADD/MAX/MIN reduction results into the symmetric memory of every other PE. The MAX/MIN destination is pre-seeded one below, equal to, or one above the source per element, so source-wins, equal, and destination-wins branches are all covered. After the kernel finishes, the host validates every element.

## Usage

1. Build in the `shmem/` directory:

    ```bash
    bash scripts/build.sh -examples -soc_type Ascend950
    ```

2. Run in the `shmem/` directory:

    ```bash
    bash examples/udma_put_value_reduce/run.sh
    ```

    The script starts 2 PEs by default. Set `NUM_PES` to use a different PE count
    (must not exceed the usable NPU count on the machine):

    ```bash
    NUM_PES=4 bash examples/udma_put_value_reduce/run.sh
    ```

    Equivalent manual launch:

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    export SHMEM_UID_SESSION_ID=127.0.0.1:8899
    ./build/bin/udma_put_value_reduce 2 0 tcp://127.0.0.1:8899 2 0 0 & # PE 0
    ./build/bin/udma_put_value_reduce 2 1 tcp://127.0.0.1:8899 2 0 0 & # PE 1
    ```

3. Command-line parameters:

    ```bash
    ./udma_put_value_reduce <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
    ```

    - n_pes: number of global PEs.
    - pe_id: PE ID of the current process.
    - ipport: IP address and port required for SHMEM initialization, in `tcp://<IP>:<port>` format.
    - g_npus: number of NPUs started on the current server.
    - f_pe: ID of the first PE used on the current server.
    - f_npu: ID of the first NPU used to run this sample on the current server.

## API Description

### aclshmemx_udma_put_value_nbi

```c++
template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemx_udma_put_value_nbi(__gm__ T* dst, T value, int pe);
```

Writes `value` to the remote symmetric address `dst` on PE `pe` asynchronously. The value is carried inline in the WRITE WQE using `sizeof(T)` bytes.

| Parameter | Description |
|-----------|-------------|
| dst       | Remote symmetric destination address on PE `pe` |
| value     | Inline value to write. Supported types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `half`, `bfloat16_t`, `int32_t`, `uint32_t`, `float`, `int64_t`, `uint64_t`, `double`; `sizeof(T)` must not exceed 8 bytes |
| pe        | Valid remote PE ID. Must satisfy `0 <= pe < aclshmem_n_pes()` and `pe != aclshmem_my_pe()`, and the UDMA connection to this PE must have been established during SHMEM initialization |

### aclshmemx_udma_put_reduce_nbi

```c++
template <
    typename T, aclshmemx_reduce_op_t REDUCE_OP,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemx_udma_put_reduce_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe);
```

Reduces `elem_size` elements from the local `src` into the existing elements of the remote symmetric address `dst` on PE `pe`, element by element, using `REDUCE_OP`.

| Parameter | Description |
|-----------|-------------|
| dst       | Remote symmetric destination address on PE `pe` |
| src       | Local source address |
| elem_size | Number of elements to reduce. One request must not exceed 256 MB |
| pe        | Valid remote PE ID. Must satisfy `0 <= pe < aclshmem_n_pes()` and `pe != aclshmem_my_pe()`, and the UDMA connection to this PE must have been established during SHMEM initialization |

`REDUCE_OP` values (the public `aclshmemx_reduce_op_t` enum is defined in `shmem_common_types.h` and shared with the other data-plane engines):

```c++
enum aclshmemx_reduce_op_t {
    ACLSHMEMX_REDUCE_ADD = 0, ///< Element-wise addition
    ACLSHMEMX_REDUCE_MAX,     ///< Element-wise maximum
    ACLSHMEMX_REDUCE_MIN,     ///< Element-wise minimum
};
```

Supported element types: `int8_t`, `int16_t`, `int32_t`, `uint32_t`, `half`, `float`, `bfloat16_t`. Unsupported `REDUCE_OP` values are rejected at compile time.

## Notes

- Both APIs are non-blocking. A normal return only means the WQE was submitted; it does not mean the data is visible on the remote PE. Call `aclshmemx_udma_quiet(pe)` before reading the destination or reusing the source buffer.
- `pe` must be a valid remote PE with an established UDMA connection and must differ from the calling PE. Both APIs return `void`, so an invalid `pe` cannot be reported at runtime: a debug build aborts the kernel on a self-send, while a release build relies on the caller to satisfy these preconditions. Validate the PE ID before calling.
- UDMA does not support concurrent RMA/AMO operations to the same PE. This sample calls `aclshmemx_udma_quiet(pe)` after each write to serialize them.
- The destination must reside in SHMEM symmetric memory. The inline value of `aclshmemx_udma_put_value_nbi` travels in the WQE and does not depend on a local source buffer.
- Remote reads use the existing `aclshmemx_udma_get_nbi` interface and are not covered by this sample.
