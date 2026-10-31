# RDMA Multi-QP Demo

## 概述

本样例基于 SHMEM 工程，介绍 device kernel 在云脉（XSCALE）和 1825（HNS_1825）RDMA 场景下，使用多个 QP 并发完成同一对端 PE 的普通及聚合 Put/Get 数据传输。

样例在两个后端均覆盖普通和聚合 Put/Get：

- `put`：当前 PE 使用多个 QP，将本地 SHMEM 对称内存中的数据写入下一个 PE 的对称内存。
- `get`：当前 PE 使用多个 QP，从下一个 PE 的 SHMEM 对称内存读取数据到本地对称内存。
- `aggregate_put`：每个 QP 将多个 Put 操作聚合为一个批次后提交。
- `aggregate_get`：每个 QP 将多个 Get 操作聚合为一个批次后提交。

`-op all` 在 XSCALE 和 HNS_1825 上均执行四种模式。

## 环境要求

本样例的 RDMA 网卡、驱动、Ascend950 CANN 版本及 `IBV_EXTEND_DRIVERS` 配置要求，统一参考 [rdma_demo 环境要求](../rdma_demo/README.md#环境要求)。

- 本样例支持 Ascend950 平台的 XSCALE 和 HNS_1825 RDMA 后端。
- 所有 PE 必须配置相同的 QP 数量，取值范围为 `[1, 56]`。
- 每个 QP 固定处理 512 KiB。聚合模式将分片均分为 8 个 chunk，使用 1088B UB workspace。

## 样例实现

本样例呈现多个 QP 在同一对端 PE 之间并发执行 RDMA 普通及聚合 Put/Get。

### 测试用例实现

（1）初始化 ACL 和 SHMEM，将数据通路配置为 `ACLSHMEM_DATA_OP_ROCE`，并在 SHMEM 初始化前通过进程级接口 `aclshmemx_set_qp_num` 设置每个对端 PE 使用的 QP 数量。

（2）每个 PE 对应一个进程和一张 NPU，并与环形拓扑中的下一个 PE 通信。Put 模式向下一个 PE 写数据，Get 模式从下一个 PE 读取数据。

（3）每个 PE 分配一段 SHMEM 对称内存，前半部分作为本地源缓冲区，后半部分作为本地目标缓冲区，并准备包含 PE 编号和元素位置的测试数据。

（4）Host 按照 QP 数量启动相同数量的 AIV。各 AIV 分别使用一个 QP，处理互不重叠的数据分片；每个 QP 固定处理 512 KiB。

（5）kernel 执行完成后，将目标缓冲区拷回 Host 逐字节校验。Put 结果应来自环形拓扑中的上一个 PE，Get 结果应来自下一个 PE。

（6）两个后端的 `all` 模式都依次执行普通和聚合 Put/Get，其中聚合操作通过 submit/defer 接口提交。全部校验完成后释放资源。

### Kernel 实现

（1）每个 AIV 使用自身的任务块编号作为 QP 编号，并根据该编号定位自己负责的 512 KiB 数据分片：

```text
block 0 -> QP 0
block 1 -> QP 1
...
```

（2）普通 Put 模式调用 `aclshmemx_roce_qp_put_nbi`，普通 Get 模式调用 `aclshmemx_roce_qp_get_nbi`。每个 AIV 只提交一次自己负责的完整数据分片；两个后端都使用 QP 指定的聚合提交接口。

（3）聚合模式中，每个 AIV 将自己的数据分片拆成 8 个 chunk，并创建独立的 `aclshmemx_submit_state_t` 和 UB workspace。前 7 个操作使用 `aclshmemx_defer_t` 加入批次，最后一个操作使用 `aclshmemx_submit_t` 提交整个批次。

（4）每个聚合批次中的操作使用相同的对端 PE、QP、UB 基地址和内部同步事件 ID。不同 QP 可并发访问同一 PE，但每个 QP 必须由单一 AIV 使用独立的 `aclshmemx_submit_state_t` 和 UB workspace；同一 `(PE, QP)` 同时只能有一个活动批次。批次总操作数（含最后的 submit）不得超过 `ACLSHMEM_ROCE_QP_AGGREGATE_MAX_OPS`，超出时需拆分为多个批次。

（5）kernel 开始时，由 block 0 调用 `aclshmemx_roce_sync_all()` 完成跨 PE 同步，再通过 `SyncAll` 放行本机所有 AIV。各 AIV 提交操作后分别调用 `aclshmemx_roce_qp_quiet()` 等待自身 QP 完成，再通过 `SyncAll` 汇合；随后由 block 0 调用 `aclshmemx_roce_sync_all()` 执行跨 PE 同步。最后一次 `SyncAll` 确保所有 AIV 在跨 PE 同步完成后退出。

## 编译执行

在仓库根目录启用 RDMA XSCALE 或 HNS_1825 构建 examples，然后运行。两个后端均构建并运行普通及聚合 Put/Get 路径：

```bash
# XSCALE
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE -examples
# HNS_1825
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend HNS_1825 -examples
bash examples/rdma_qp_demo/run.sh
```

指定首个 NPU 和 QP 数。两个后端默认均运行四种模式：

```bash
bash examples/rdma_qp_demo/run.sh -fnpu 0 -qp 4
```

仅调试单个模式时可额外指定，例如 `-op put`。

跨机运行时，各机器的 `-pes`、`-qp` 和 `-ipport` 必须相同，`-ipport` 指向 PE0 所在机器，脚本会据此设置相同的 `SHMEM_UID_SESSION_ID`。假设两台机器各使用 2 张 NPU，机器 A 的 IP 为 `192.168.1.10`，在两台机器上分别执行：

```bash
# 机器 A：启动 PE 0~1，使用本机 NPU 0~1
bash examples/rdma_qp_demo/run.sh -pes 4 -fpe 0 -gnpus 2 -fnpu 0 -qp 4 \
    -ipport tcp://192.168.1.10:8899

# 机器 B：启动 PE 2~3，使用本机 NPU 0~1
bash examples/rdma_qp_demo/run.sh -pes 4 -fpe 2 -gnpus 2 -fnpu 0 -qp 4 \
    -ipport tcp://192.168.1.10:8899
```

默认用例使用 2 个 QP，总数据量为 1 MiB。执行成功时每个 PE 输出：

```text
[PASS] op=put pe=0 elements=1048576
[PASS] op=get pe=0 elements=1048576
[PASS] op=aggregate_put pe=0 elements=1048576
[PASS] op=aggregate_get pe=0 elements=1048576
```

数据校验失败时，程序打印首个错误位置、实际值和期望值。`run.sh` 会等待当前机器上的所有 PE 进程，并在任意进程失败时返回非 0。

### 运行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-pes` | `2` | 全局 PE 数量，至少为 2 |
| `-fpe` | `0` | 当前机器启动的首个全局 PE |
| `-gnpus` | `2` | 当前机器启动的 NPU/PE 进程数量 |
| `-fnpu` | `0` | 当前机器使用的首个逻辑 NPU 编号 |
| `-qp` | `2` | 每个对端 PE 的 QP 数，范围 1~56 |
| `-op` | `all` | `put`、`get`、`aggregate_put`、`aggregate_get` 或 `all` |
| `-ipport` | 动态本机端口 | SHMEM 初始化地址 |
