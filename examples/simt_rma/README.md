# 样例介绍

本样例旨在展示 SIMD 与 SIMT 混合编译模式下，SIMT 远程内存访问（RMA）接口的典型使用方法，并演示 SIMT 原子 compare-and-swap（CAS）接口的用法。RMA 接口主要包含以下三种形式：

1. `__simt_callee__ inline void aclshmem_{NAME}_{op}(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)`
2. `__simt_callee__ inline void aclshmem_{op}{BITS}(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe)`
3. `__simt_callee__ inline void aclshmem_{op}mem(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)`

这三种形式的核心功能均为实现连续内存区域的数据传输，其区别在于数据长度的指定方式：

- **第一种接口**：基于每个传输元素的具体数据类型（如 `half`、`float` 等）进行描述。
- **第二种接口**：基于每个传输元素的比特位大小（如 `8`、`16` 等）进行描述。
- **第三种接口**：直接指定需要传输的总内存字节大小。

每种形式又各有 3 个线程组粒度的变体，区别在于参与本次搬运的线程范围：

| 变体 | 说明 |
| --- | --- |
| `aclshmem_*`（无后缀） | 线程级，由调用线程独立完成整块数据的搬运 |
| `aclshmemx_*_block` | block 级，由同一 block 内的线程协作完成搬运 |
| `aclshmemx_*_warp` | warp 级，由同一 warp（32 个线程）协作完成搬运 |

**本样例调用的是 warp 级变体**，kernel 以 `dim3(32)` 启动，由 warp 内 32 个线程协作搬运同一份数据，而非每个线程各自遍历整块内存。三种形式对应的实际调用接口如下：

```cpp
// 形式一：按元素数据类型描述长度
simt::aclshmemx_int16_get_warp(__gm__ int16_t *dst, __gm__ int16_t *src, size_t elem_size, int32_t pe);
simt::aclshmemx_int16_put_warp(__gm__ int16_t *dst, __gm__ int16_t *src, size_t elem_size, int32_t pe);
// 形式二：按元素比特位宽描述长度
simt::aclshmemx_get128_warp(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe);
simt::aclshmemx_put128_warp(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe);
// 形式三：直接指定传输的总字节数
simt::aclshmemx_getmem_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe);
simt::aclshmemx_putmem_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe);
```

上述接口名称中的占位符 `{}` 可选值如下表所示：

| 占位符 | 可选值 |
| --- | --- |
| `{op}` | `put`, `get` |
| `{NAME}` | `half`, `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `char`, `bfloat16` |
| `{BITS}` | `8`, `16`, `32`, `64`, `128` |

## 样例执行流程

本样例通过以下流程演示 RMA 接口的具体工作机制：

1. **环境初始化**：每个计算单元（PE）初始化 3 块大小相同的对称内存。其中，第一块内存的数据初始化为 `[my_pe + 0, ..., my_pe + size - 1]`，第二块和第三块内存的数据初始化为 `-1`。
2. **GET 操作演示**：每个 PE 均调用 `get` 接口（warp 级变体），将逻辑上属于**上一个 PE** 的第一块内存中的数据，拉取并写入至自身的第二块内存中。
3. **PUT 操作演示**：每个 PE 均调用 `put` 接口（warp 级变体），将自身第一块内存中的数据，推送并写入至**上一个 PE** 的第三块内存中。之所以推送给上一个 PE，是因为对它而言当前 PE 正是它的“下一个 PE”，这样每个 PE 的第三块内存最终存放的就是其下一个 PE 的数据。
4. **结果校验**：通信操作完成后，各 PE 将自动比对内存中的数据，验证数据传输的正确性——第二块内存应等于上一个 PE 的第一块内存，第三块内存应等于下一个 PE 的第一块内存。

> 样例代码中分别提供了 `test_put_get_type`、`test_put_get_bits`、`test_put_get_mem` 三个函数，对应上述三种接口形式；默认调用的是 `test_put_get_bits`，可按需切换。

## 原子 CAS 演示

RMA 用例之后，样例追加一段原子 CAS 演示，展示以下接口：

```cpp
__simt_callee__ inline TYPE simt::aclshmem_NAME_atomic_compare_swap(
    __gm__ TYPE* dest, TYPE cond, TYPE value, int32_t pe);
```

`{NAME}` 可选 `int32`、`int64`、`uint32`、`uint64`。同步接口：原子比较 `dest` 与 `cond`，相等则将 `dest` 更新为 `value`，否则保持不变；返回 `dest` 的旧值。`dest` 可指向 device 对称内存或 HOST_SIDE 对称内存。

CAS 演示（以 uint32 为例）包含三组校验：

| 用例 | 校验点 |
| --- | --- |
| 槽位语义 | 命中的 CAS 返回旧值并完成替换；未命中的 CAS 返回当前值且不改写 |
| 本 PE 共享计数器 | 全部线程经 CAS 重试循环并发自增同一地址，终值 = 线程数 × 轮数（同卡原子性）|
| PE0 全局计数器 | 所有 PE 的全部线程自增 PE0 堆上同一地址，终值 = PE 数 × 线程数 × 轮数（跨 PE 原子性）|

演示目标为 device 对称内存（`aclshmemx_malloc` 默认分配）；HOST_SIDE 对称内存接口同样支持，但不在本样例覆盖范围内。单 PE 运行（`-pes 1`）时全局计数器退化为同卡竞争，双卡及以上覆盖跨 PE 场景。

## 支持的设备

- Ascend950

## 使用方式

1. **编译项目**
   在 `shmem/` 根目录下执行编译脚本：

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **运行simt_rma示例程序**
   进入示例目录并执行运行脚本：

   ```bash
   cd examples/simt_rma
   bash run.sh
   ```

   `run.sh` 默认启动 2 个独立进程，每个进程对应一个 PE 并绑定一张卡，使用 `build/bin/simt_rma` 执行样例。可通过 `-pes` 参数指定参与运行的 PE（卡）数量，例如启动 4 卡运行：

   ```bash
   bash run.sh -pes 4
   ```

   使用多卡运行时，请确保执行环境中实际可用的 NPU 卡数不少于指定的 PE 数量。

运行成功时每个 PE 会输出 RMA 与 atomic compare swap 两行 `[SUCCESS]` 校验日志；任一校验失败将输出对应的 `[ERROR]` 明细与 `[FAILURE]`。
