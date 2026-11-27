# 样例介绍

## 版本和平台支持说明

本样例仅支持 Ascend950 平台，依赖 CANN 9.1.0 及以上版本。

## 样例说明

本样例展示 UDMA 低阶接口中的两种写操作：

- `aclshmemx_udma_put_value_nbi<T>`：将单个 `value` 内联在 WRITE WQE 中写入远端对称地址，调用方无需提供本地源缓冲区。
- `aclshmemx_udma_put_reduce_nbi<T, REDUCE_OP>`：将本地 `src` 向量按 `ADD` / `MAX` / `MIN` 归约写入远端对称地址。

每个 PE 向其他所有 PE 的对称内存写入 1 个 `uint64_t` 立即数，以及 16 个 `int32_t` 元素的 ADD/MAX/MIN 归约结果。MAX/MIN 的目的端初始值按元素分别取源值减一、等于源值、源值加一，使源端胜出、两值相等和目的端胜出三种分支都被覆盖。kernel 完成后由 Host 逐元素校验结果。

## 使用方式

1. 在 shmem/ 目录编译：

    ```bash
    bash scripts/build.sh -examples -soc_type Ascend950
    ```

2. 在 shmem/ 目录运行：

    ```bash
    bash examples/udma_put_value_reduce/run.sh
    ```

    默认启动 2 个 PE，可通过 `NUM_PES` 指定 PE 数（不超过本机可用 NPU 数）：

    ```bash
    NUM_PES=4 bash examples/udma_put_value_reduce/run.sh
    ```

    等价的手动启动方式：

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    export SHMEM_UID_SESSION_ID=127.0.0.1:8899
    ./build/bin/udma_put_value_reduce 2 0 tcp://127.0.0.1:8899 2 0 0 & # PE 0
    ./build/bin/udma_put_value_reduce 2 1 tcp://127.0.0.1:8899 2 0 0 & # PE 1
    ```

3. 命令行参数说明：

    ```bash
    ./udma_put_value_reduce <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
    ```

    - n_pes: 全局PE数量。
    - pe_id: 当前进程的PE号。
    - ipport: SHMEM初始化需要的IP及端口号，格式为`tcp://<IP>:<端口号>`。
    - g_npus: 当前机器上启动的NPU卡的数量。
    - f_pe: 当前机器上使用的第一个PE号。
    - f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号。

## 接口说明

### aclshmemx_udma_put_value_nbi

```c++
template <typename T, const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemx_udma_put_value_nbi(__gm__ T* dst, T value, int pe);
```

接口功能：将 `value` 按 `sizeof(T)` 字节内联在 WRITE WQE 中，异步写入目标 PE `pe` 的远端对称地址 `dst`。

| 参数名 | 含义 |
|--------|------|
| dst    | 目标 PE `pe` 上的远端对称目的地址 |
| value  | 写入远端的立即数值，支持 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float`、`int64_t`、`uint64_t`、`double`，且 `sizeof(T)` 不超过 8 字节 |
| pe     | 有效远端 PE 号，须满足 `0 <= pe < aclshmem_n_pes()` 且 `pe != aclshmem_my_pe()`，且该 PE 的 UDMA 连接已在 SHMEM 初始化时建立 |

### aclshmemx_udma_put_reduce_nbi

```c++
template <
    typename T, aclshmemx_reduce_op_t REDUCE_OP,
    const aclshmemx_udma_op_config_t& CONFIG = ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>
ACLSHMEM_DEVICE void aclshmemx_udma_put_reduce_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe);
```

接口功能：将当前 PE 本地 `src` 中的 `elem_size` 个元素，按 `REDUCE_OP` 与目标 PE `pe` 的远端对称地址 `dst` 中的现有元素逐元素归约。

| 参数名    | 含义 |
|-----------|------|
| dst       | 目标 PE `pe` 上的远端对称目的地址 |
| src       | 当前 PE 本地的源地址 |
| elem_size | 参与归约的元素个数，单次请求不超过 256 MB |
| pe        | 有效远端 PE 号，须满足 `0 <= pe < aclshmem_n_pes()` 且 `pe != aclshmem_my_pe()`，且该 PE 的 UDMA 连接已在 SHMEM 初始化时建立 |

`REDUCE_OP` 取值（公共枚举 `aclshmemx_reduce_op_t` 定义在 `shmem_common_types.h`，与其他数据面引擎共用）：

```c++
enum aclshmemx_reduce_op_t {
    ACLSHMEMX_REDUCE_ADD = 0, ///< 逐元素相加
    ACLSHMEMX_REDUCE_MAX,     ///< 逐元素最大值
    ACLSHMEMX_REDUCE_MIN,     ///< 逐元素最小值
};
```

归约元素类型支持 `int8_t`、`int16_t`、`int32_t`、`uint32_t`、`half`、`float`、`bfloat16_t`；不支持的 `REDUCE_OP` 会在编译期被拒绝。

## 注意事项

- 两个接口均为非阻塞接口，调用返回只表示 WQE 已提交，不代表数据已在远端可见。读取目的端数据或复用源缓冲区前，必须调用 `aclshmemx_udma_quiet(pe)` 等待该 PE 上的操作完成。
- 两个接口的 `pe` 必须是已建立 UDMA 连接的有效远端 PE，不能是当前 PE。`pe` 非法时接口返回 `void`，无法在运行期报告失败：debug 构建会对 self 发送直接终止 kernel，release 构建依赖调用方保证前置条件，因此调用前请自行校验 PE 号。
- UDMA 不支持同一 PE 上的 RMA/AMO 并发，本样例在每个写操作后调用 `aclshmemx_udma_quiet(pe)` 串行化。
- 目的地址必须位于 SHMEM 对称内存；`aclshmemx_udma_put_value_nbi` 的立即数由 WQE 携带，不依赖本地源缓冲区。
- 远端数据读取使用已有的 `aclshmemx_udma_get_nbi` 接口，本样例不涉及。
