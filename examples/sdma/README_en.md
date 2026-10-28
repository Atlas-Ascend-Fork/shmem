# SDMA Usage Description

> **Platform limitation**: A2/A3 support SDMA put/get; Ascend950 supports only SDMA get, not SDMA put.
>
> **Example runtime limitation**: This example runs only on A2/A3 platforms. It includes SDMA put operations and therefore cannot run on Ascend950.

## Environment Requirements and Preparations

The SDMA put/get APIs require CANN 9.0.0-beta.2 or later and can be used for read/write data transfers. To download and install the toolkit package of the corresponding version, see [CANN Version Description](../../docs/quickstart_en.md#software-and-hardware-versions). To enable SDMA, you also need to install the ops package matching the toolkit version and device type.

## Supported Devices

The SDMA put/get APIs are available on A2/A3 platforms such as Atlas 200I A2/A3 and Atlas 300T A2/A3. Ascend950 supports only SDMA get; SDMA put is unavailable.

## Instructions for Using an Example
1. Build and install the software package in the `shmem/` directory:
```bash
bash scripts/build.sh -package
./install/*/SHMEM_1.0.0_linux-*.run --install
```

2. Build the examples in the `shmem/` directory:
```bash
bash scripts/build.sh -examples
```

3. Run the demo in the `shmem/examples/sdma` directory:
```bash
bash run.sh -pes ${PES} -type ${TYPES}
```
  - **Parameter description**:
      - PES: the number of devices (NPUs) used for running the demo; only 2, 4, or 8 cards are supported on one server
      - TYPES: type of the data to be transferred. Currently, the following data types are supported: int, uint8, int64, and fp32.

## SDMA API Usage Description

### aclshmemx_sdma_put_nbi

The ordinary put API is for a single submitting AIV and always uses QP 0:

```c++
template <typename T>
__global__ __aicore__ void sdma_put_single_qp(GM_ADDR gva, int elem_size, int target_pe)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __ubuf__ T* tmp = reinterpret_cast<__ubuf__ T*>(uint64_t(1024));
        __gm__ T* src = reinterpret_cast<__gm__ T*>(gva) + aclshmem_my_pe() * elem_size;
        aclshmemx_sdma_put_nbi(src, src, tmp, 64, elem_size, target_pe, EVENT_ID0);
        aclshmemx_sdma_quiet(tmp, 64, EVENT_ID0);
    }
}
```

The example first runs this kernel so every PE writes one complete segment to the next PE on QP 0 and verifies the segment received from the previous PE. It then runs the explicit-QP multi-AIV allgather. Multiple AIVs must not concurrently share the ordinary QP 0 API; use `aclshmemx_sdma_qp_put_nbi` for that case.

### aclshmemx_sdma_qp_put_nbi
For example, consider an API with pointer-type parameters:
```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                               uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
```
API function: This API transfers elements (with the number specified by `elem_size`) from the local `src` address of the current PE to the remote symmetric address `dst` on the target PE `pe`.
| Parameter      | Description                                                                |
|--------------|----------------------------------------------------------------------|
| dst          | Remote symmetric address on the target PE `pe` where data is written                                  |
| src          | Local source address on the current PE                                                      |
| buf          | Buffer address                                                          |
| ub_size      | Buffer size                                                          |
| elem_size    | Number of elements                                                            |
| pe           | Destination PE                                                              |
| qp_idx       | SDMA QP used by the current AIV, normally `GetBlockIdx()`                   |
| sync_id      | Synchronization ID                                                              |

### aclshmemx_sdma_qp_get_nbi
For example, consider an API with pointer-type parameters:
```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                               uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
```
API function: This API pulls elements (with the number specified by `elem_size`) from the remote symmetric address `src` on the target PE `pe` and writes them to the local `dst` address of the current PE.
| Parameter      | Description                                                                |
|--------------|----------------------------------------------------------------------|
| dst          | Local destination address on the current PE to which data is written                                  |
| src          | Remote symmetric address on the target PE `pe` from which data is read                                                      |
| buf          | Buffer address                                                          |
| ub_size      | Buffer size                                                          |
| elem_size    | Number of elements                                                            |
| pe           | Destination PE                                                              |
| qp_idx       | SDMA QP used by the current AIV, normally `GetBlockIdx()`                   |
| sync_id      | Synchronization ID                                                              |

## Precautions
The example allocates `128M * sizeof(T)` bytes of symmetric memory and uses `16M * sizeof(T)` bytes per PE. The documented support matrix is 2, 4, and 8 cards; actual use also depends on available symmetric-memory capacity. The SDMA infrastructure supports up to 72 AIVs/QPs.

To limit the number of explicit QPs, call `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num)` before `aclshmemx_init_attr`. The valid range of `qp_num` is 1 to the number of vector cores on the current device. If this API is not called, only one SDMA stream/QP is created. When multiple cores need to run concurrently, set the required number of QPs explicitly. In this example, each AIV uses `GetBlockIdx()` (the AIV-level global index) as `qp_idx`. Therefore, `qp_num` should be set to the number of blocks x the number of AIVs per block (currently `20 x 2 = 40`).

Both `aclshmemx_sdma_qp_put_nbi` and `aclshmemx_sdma_qp_get_nbi` are non-blocking APIs. They return control to the caller right away and do not wait for the data transfer to finish. To ensure that the data transfer has finished, you can adopt one of the following methods:
1. For each kernel that calls `aclshmemx_sdma_qp_put/get_nbi`, call `aclshmemx_sdma_qp_quiet` with the same `qp_idx` after the SDMA task ends, and wait until the operations on that QP are complete.
Scenario: Subsequent operations within the operator depend on the completion of the SDMA task. For example, subsequent operations need to use the data transferred by the SDMA.
2. For each kernel that calls `aclshmemx_sdma_qp_put/get_nbi`, call `aclshmemx_sdma_qp_notify_record` with the same `qp_idx` after the SDMA task ends, then call `aclrtWaitAndResetNotify` on the host and wait until the specified synchronization ID completes. For details, see [NotifyWait Mechanism Usage Description](../notifywait/README_en.md).
Scenario: Kernels on other streams must wait for the SDMA task to finish before continuing execution.
