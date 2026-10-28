## 概述

本样例基于 SHMEM 工程，介绍 device kernel 在直连 UDMA 场景下，使用 SIMT 编程模型通过多个 QP 并发完成同一对端 PE 的 Put/Get 数据传输。

样例覆盖 SIMT 的两种执行域（warp/block），每种执行域下每个 warp/block 各驱动一个 QP，验证 `simt::aclshmemx_udma_qp_{put,get}_nbi_{warp|block}` 与 `simt::aclshmemx_udma_qp_quiet_{warp|block}` 接口在多 QP 配置下的数据正确性。两种功能验证模式：

- `put`：当前 PE 使用多个 QP，将本地 SHMEM 对称内存中的数据写入下一个 PE 的 SHMEM 对称内存。
- `get`：当前 PE 使用多个 QP，从下一个 PE 的 SHMEM 对称内存读取数据到本地 SHMEM 对称内存。

## 版本和平台支持说明

- 本样例仅支持 Ascend950 平台（direct 模式），其他平台不支持运行该样例。
- CANN 9.2.0 及以上版本已提供 UDMA 所需的 HCOMM 资源接口。请从 [CANN 9.2.0 资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.2.0)下载并安装 Ascend950 对应的 toolkit 包和 ops 包；低于 9.2.0 的 CANN 版本不在本样例支持范围内。
- 多 QP 还要求当前 HCOMM 的 `HcommChannelDesc` 支持 `channelName`。
- 编译前需执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（自定义安装路径时使用对应的 `set_env.sh`），并使用 `-enable_simt -soc_type Ascend950` 编译；SIMT 支持需在编译时通过 `-enable_simt` 开启。

## 样例实现

本样例呈现多个 QP 在同一对端 PE 之间并发执行 SIMT UDMA Put/Get 的基本使用流程，覆盖 warp/block 两种执行域。

### 测试用例实现

（1）初始化 ACL 和 SHMEM，将数据通路配置为 `ACLSHMEM_DATA_OP_UDMA`，并在 SHMEM 初始化前通过进程级接口 `aclshmemx_set_qp_num` 设置每个对端 PE 使用的 QP 数量。

（2）每个 PE 对应一个进程和一张 NPU，并与环形拓扑中的下一个 PE 通信。Put 模式向下一个 PE 写数据，Get 模式从下一个 PE 读数据。

（3）每个 PE 分配三段 SHMEM 对称内存：`origin`（本端数据源）、`res_put`（Put 结果区）、`res_get`（Get 结果区），并准备包含 PE 编号和元素位置的测试数据。缓冲区只分配一次，warp 与 block 两种 mode 复用同一组。

（4）Host 通过 `<<<...>>>` 下发 SIMT kernel，kernel 内通过 `asc_vf_call` 调度 SIMT VF 执行数据传输。两种执行域下每个 warp/block 各驱动一个 QP，处理互不重叠的数据分片。

（5）kernel 执行完成后，将结果拷回 Host 逐字节校验。Put 结果（`res_put`）应来自上一个 PE 的 pattern，Get 结果（`res_get`）应来自下一个 PE 的 pattern。

（6）校验完成后由 `aclshmem_finalize` 统一释放 SHMEM 堆，并释放设备和 ACL 相关资源。校验失败只记录失败状态，两个 PE 继续执行相同的 mode 计划，最后统一清理并返回非零，避免单个 PE 提前退出导致对端阻塞在集合通信中。

### Kernel 实现

（1）两种执行域的 QP 驱动方式：

| 执行域 | kernel grid | VF 启动形态 | QP 驱动方式 |
|--------|------------|-------------|-------------|
| warp | `<<<1>>>` | `asc_vf_call<demo_warp_vf>(dim3(qp_count * 32), ...)` | warp i 驱动 QP i（warp 集合调用，warp 内线程协作提交 WQE；quiet 接口依赖 warp lockstep，无块级同步，VF 内全部线程须参与） |
| block | `<<<qp_count>>>` | `asc_vf_call<demo_block_vf>(dim3(128), ...)` 每 block | block i 驱动 QP i（VF 内读取 `blockIdx.x` 作为 QP 编号；仅 warp 0 组装并提交 WQE，其余 warp 不参与提交；quiet 接口含块级同步，128 线程须全部参与） |

（2）每个 driver（warp/block）根据自身的 `qp_idx` 按商和余数划分互不重叠的连续分片。

（3）Put 模式调用 `simt::aclshmemx_udma_qp_put_nbi_{warp|block}`，Get 模式调用 `simt::aclshmemx_udma_qp_get_nbi_{warp|block}`。每个 driver 只提交自己负责的数据分片。

（4）提交后调用 `simt::aclshmemx_udma_qp_quiet_{warp|block}`，等待当前对端 PE、当前 QP 上的操作完成。不同 QP 之间不提供隐式顺序。

（5）warp/block 变体为集合调用，VF 启动的全部线程必须都执行到接口。block quiet 接口内部使用 `asc_syncthreads` 同步，确保所有线程在完成确认后才返回；warp quiet 接口依赖 warp lockstep，无块级同步。put/get 的 warp/block 变体同样为集合调用，VF 全部线程必须参与。

## 编译执行

环境配置请参考[快速上手](../../docs/quickstart.md)。完成环境配置后，在仓库根目录执行：

```bash
# 编译 SIMT UDMA 样例（需启用 SIMT 支持）
bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
```

产物：`build/bin/simt_udma_qp_demo`。

### 运行示例

```bash
# 默认运行：2 个 PE、2 个 QP、两模式（warp/block）依次运行、1MB 数据
bash examples/simt_udma_qp_demo/run.sh

# 指定参数：4 个 PE、4 个 QP、仅 warp 模式、1MB 数据
bash examples/simt_udma_qp_demo/run.sh -pes 4 -qp_count 4 -mode warp -elems 1048576

# 8 个 QP 验证 block 模式
bash examples/simt_udma_qp_demo/run.sh -pes 2 -qp_count 8 -mode block -elems 1048576

# 指定对称堆大小（需容纳三段对称缓冲区 origin + res_put + res_get）
bash examples/simt_udma_qp_demo/run.sh -pes 2 -qp_count 4 -mode warp -heap_mb 2048
```

用例执行成功时，每个 PE 输出：

```text
[PASS] warp-put pe=0 elements=1048576
[PASS] warp-get pe=0 elements=1048576
[PASS] block-put pe=0 elements=1048576
[PASS] block-get pe=0 elements=1048576
[INFO] mode=all qp_count=2 pe=0 result=PASS
```

数据校验失败时，程序打印首个错误位置、实际值和期望值。`run.sh` 会等待所有 PE 进程，并在任意进程失败时返回非 0。

### 运行参数

`run.sh` 支持如下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-pes` | `2` | PE 总数，必须至少为 2；脚本启动相同数量的进程。 |
| `-qp_count` | `2` | 每个对端 PE 创建的 QP 数，范围为 `[1, 56]`，所有 PE 必须一致。 |
| `-mode` | `all` | 执行模式，支持 `warp`、`block`、`all`（两模式依次运行）。 |
| `-elems` | `1048576` | 每个 PE 搬运的字节数，至少为 QP 数的 2 倍（UDMA 引擎要求每个 QP 单次传输不低于 2 字节）；支持不均分。 |
| `-heap_mb` | `1024` | 每个 PE 的 SHMEM 对称堆大小，单位为 MiB；必须同时容纳三段对称缓冲区（origin + res_put + res_get）。 |
| `-first_npu` | `0` | PE i 绑定的逻辑 NPU 编号为 `first_npu + i`。 |
| `-ipport` | `tcp://127.0.0.1:8999` | 引导初始化地址。 |

## 使用约束（接口契约摘要）

1. QP 数由 Host 在 init 前经 `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num)` 下发，所有 PE 必须一致。
2. 同一 `(pe, qp_idx)` 同一时刻只允许一个驱动者（warp/block：一个线程组）；多线程并行请绑定不同 `qp_idx`。
3. `_nbi` 返回仅代表 WQE 已提交；读/写相关内存前必须调用对应的 quiet。
4. warp/block 变体为集合调用，VF 启动的全部线程必须都执行到接口。block quiet 变体内部使用 `asc_syncthreads` 同步，确保所有线程在完成确认后才返回；warp quiet 变体依赖 warp lockstep，无块级同步。put/get 的 warp/block 变体同样为集合调用，VF 全部线程必须参与。
5. 单请求传输 2~256MB，且每个 QP 单次传输不低于 2 字节。
6. 本样例不支持 Multi-link 链路模式和 PutSignal 操作（这些是 SIMD 版 `udma_qp_demo` 的能力）。
