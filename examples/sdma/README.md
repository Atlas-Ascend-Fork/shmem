# SDMA使用说明

> **平台限制**：A2/A3 支持 SDMA put/get；Ascend950 仅支持 SDMA get，不支持 SDMA put。
>
> **用例运行限制**：本用例仅支持在 A2/A3 平台运行。用例包含 SDMA put 操作，因此不支持在 Ascend950 上运行。

## 环境要求和准备

SDMA put/get接口需要CANN 9.0.0-beta.2及以上版本支持，可用于read/write数据搬运。请参考[CANN版本说明](../../docs/quickstart.md#431-cann-版本说明)下载并安装对应版本的toolkit包；使能SDMA时，还需要安装与toolkit版本和设备类型匹配的ops包。

## 支持设备

SDMA put/get 接口在 Atlas 200I A2/A3、Atlas 300T A2/A3 等 A2/A3 平台可用；Ascend950 仅支持 SDMA get，SDMA put 不可用。

## example使用方式

1. 在`shmem/`目录编译软件包并安装：

    ```bash
    bash scripts/build.sh -package
    ./install/*/SHMEM_1.0.0_linux-*.run --install
    ```

2. 在`shmem/`目录下编译examples：

    ```bash
    bash scripts/build.sh -examples
    ```

3. 在`shmem/examples/sdma`目录执行demo:

    ```bash
    bash run.sh -pes ${PES} -type ${TYPES}
    ```

    **参数说明**：
      - PES：指定用于运行的设备（NPU）数量，仅支持 2、4、8 卡，限定单台机器内。
      - TYPES：指定传输数据类型，当前支持：int，uint8，int64，fp32。

## SDMA接口使用说明

### aclshmemx_sdma_put_nbi

普通put接口用于单核提交并固定使用QP 0：

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

本用例会先通过该kernel让每个PE使用QP 0向下一PE写入一个完整segment，并校验上一PE写入的数据；随后再运行显式QP的多核allgather。普通接口不能由多个AIV并发共享QP 0，多核场景应使用`aclshmemx_sdma_qp_put_nbi`。

### aclshmemx_sdma_qp_put_nbi

以指针类型参数接口为例：

```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                               uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
```

接口功能：将当前 PE 本地 src 的数据，传输至目标 PE pe 的远端对称地址 dst，传输 elem_size 个元素。

| 参数名       | 含义                                                                 |
|--------------|----------------------------------------------------------------------|
| dst          | 目标 PE pe 上写入数据的远端对称地址                                  |
| src          | 当前 PE 本地的源地址                                                 |
| buf          | 缓冲区地址                                                           |
| ub_size      | 缓冲区大小                                                           |
| elem_size    | 元素个数                                                             |
| pe           | 目标PE                                                               |
| qp_idx       | 当前 AIV 使用的 SDMA QP，一般取 `GetBlockIdx()`                       |
| sync_id      | 同步ID                                                               |

### aclshmemx_sdma_qp_get_nbi

以指针类型参数接口为例：

```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                               uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id)
```

接口功能：从目标 PE pe 的远端对称地址 src 拉取数据，写入当前 PE 本地 dst，传输 elem_size 个元素。

| 参数名       | 含义                                                                 |
|--------------|----------------------------------------------------------------------|
| dst          | 当前 PE 本地的写入地址                                               |
| src          | 目标 PE pe 上读取数据的远端对称地址                                  |
| buf          | 缓冲区地址                                                           |
| ub_size      | 缓冲区大小                                                           |
| elem_size    | 元素个数                                                             |
| pe           | 目标PE                                                               |
| qp_idx       | 当前 AIV 使用的 SDMA QP，一般取 `GetBlockIdx()`                       |
| sync_id      | 同步ID                                                               |

## 注意事项

当前用例申请`128M * sizeof(T)`字节对称空间，每个PE占用`16M * sizeof(T)`字节。文档支持矩阵为2、4、8卡；实际可用卡数还需满足对称空间和运行环境容量条件。底层SDMA最多支持72个AIV/QP。

如需限制显式 QP 数量，可在 `aclshmemx_init_attr` 之前调用
`aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num)`。`qp_num` 的有效范围是 1 到当前设备 vector-core
数量；未调用时仅创建一个 SDMA stream/QP。需要多核并发时应显式设置所需 QP 数量。本用例每个 AIV 使用
`GetBlockIdx()`（AIV 级全局索引）作为 `qp_idx`，因此 `qp_num` 应设置为 block 数 × 每 block AIV 数
（当前为 `20 × 2 = 40`）。

`aclshmemx_sdma_qp_put_nbi`和`aclshmemx_sdma_qp_get_nbi`都是非阻塞接口，调用后立即返回，不等待数据传输完成。用户使用时，可通过以下两种方式确保数据传输完成：

1. 所有调用`aclshmemx_sdma_qp_put/get_nbi`的核，在sdma任务结束后，算子内调用相同`qp_idx`的`aclshmemx_sdma_qp_quiet`接口，等待该QP上的SDMA操作完成。
适用场景：算子内后续操作依赖sdma任务完成，例如后续算子需要使用sdma传输好的数据。
2. 所有调用`aclshmemx_sdma_qp_put/get_nbi`的核，在sdma任务结束后，算子内调用相同`qp_idx`的`aclshmemx_sdma_qp_notify_record`接口，然后在host侧调用`aclrtWaitAndResetNotify`接口，等待指定的同步ID完成（详细用法可查看[NotifyWait机制使用说明](../notifywait/README.md)）。

适用场景：其它stream上的kernel需要等待sdma任务完成后才能继续执行。
