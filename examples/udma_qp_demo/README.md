## 概述

本样例基于 SHMEM 工程，介绍 device kernel 在直连 UDMA 场景下，使用多个 QP 并发完成同一对端 PE 的 Put/Get/PutSignal 数据传输。

样例覆盖三种功能验证模式：

- `put`：当前 PE 使用多个 QP，将本地普通 Device 内存中的数据写入下一个 PE 的 SHMEM 对称内存。
- `get`：当前 PE 使用多个 QP，从下一个 PE 的 SHMEM 对称内存读取数据到本地普通 Device 内存。
- `put_signal`：每个 QP 使用一个 `WRITE_WITH_NOTIFY` WQE 写入自己的数据分片和独立 signal word。

## 版本和平台支持说明

- UDMA 多 QP 样例仅支持 Ascend950 平台，其他平台不支持运行该样例。
- CANN 9.1.0 已提供 UDMA 所需的 `HcommEndpointCreate`、`HcommMemReg`、`HcommChannelCreate`、`HcommChannelGetStatus` 等 HCOMM 资源接口。请从 [CANN 9.1.0 资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0)下载并安装 Ascend950 对应的 toolkit 包和 ops 包；低于 9.1.0 的 CANN 版本不在本样例支持范围内。
- 多 QP 还要求当前 HCOMM 的 `HcommChannelDesc` 支持 `channelName`。
- 初始化时会加载并检查所需 HCOMM 符号。编译前需执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（自定义安装路径时使用对应的 `set_env.sh`），并使用 `-soc_type Ascend950` 编译；若 HCOMM 运行时库或符号不完整，初始化会失败。

## 样例实现

本样例呈现多个 QP 在同一对端 PE 之间并发执行 UDMA Put/Get/PutSignal 的基本使用流程。

### 测试用例实现

（1）初始化 ACL 和 SHMEM，将数据通路配置为 `ACLSHMEM_DATA_OP_UDMA`，并在 SHMEM 初始化前通过进程级接口 `aclshmemx_set_qp_num` 设置每个对端 PE 使用的 QP 数量。如需启用 shared jetty（解析到同一本地 eid_index 的多个对端 PE 共享一个底层 jetty），在初始化前再调用 `aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, true)`；该配置默认关闭，且所有 PE 必须一致。

（2）每个 PE 对应一个进程和一张 NPU，并与环形拓扑中的下一个 PE 通信。Put 和 PutSignal 模式向下一个 PE 写数据，Get 模式从下一个 PE 读数据。

（3）每个 PE 分配一段 SHMEM 对称内存和一段普通 Device 内存，并准备包含 PE 编号和元素位置的测试数据。PutSignal 模式额外为每个 QP 分配一个独占的对称 signal word。

（4）Host 按照 QP 数量启动相同数量的 AIV。各 AIV 分别使用一个 QP，处理互不重叠的数据分片。

（5）kernel 执行完成后，将结果拷回 Host 逐字节校验。Put 和 PutSignal 结果应来自上一个 PE，Get 结果应来自下一个 PE；PutSignal 还校验每个 QP 独立编码的 signal 值。

（6）校验完成后释放 SHMEM、设备和 ACL 相关资源。

### Kernel 实现

（1）每个 AIV 使用自身的任务块编号作为 QP 编号，并根据总字节数按商和余数划分互不重叠的连续分片。

（2）Put 模式调用 `aclshmemx_udma_qp_put_nbi`，Get 模式调用 `aclshmemx_udma_qp_get_nbi`，PutSignal 模式调用 `aclshmemx_udma_qp_put_signal_nbi`。每个 AIV 只提交自己负责的数据分片。

（3）提交后调用 `aclshmemx_udma_qp_quiet`，等待当前对端 PE、当前 QP 上的操作完成。PutSignal 的数据和 signal 位于同一个 WQE；不同 QP 之间不提供隐式顺序。

## 编译执行

环境配置请参考[快速上手](../../docs/quickstart.md)。完成环境配置后，在仓库根目录执行：

### 默认链路模式

不添加 `-multi_link` 时，多个 QP 使用默认链路策略。对于同时存在 MESH 和 CLOS 链路的对端，所有 QP 使用 MESH 链路；只有不存在 MESH 链路时，才在可用的 CLOS 链路中轮转。该模式可以通过 `-qp_count` 配置 QP 数量，但不能为某个 QP 指定链路，也不能传入 `-qp_net_layers`。

```bash
# 默认构建：未启用 multi-link；有 MESH 时所有 QP 使用 MESH，否则在 CLOS 链路中轮转
bash scripts/build.sh -examples -soc_type Ascend950

# 运行默认用例，默认使用 2 个 PE、每个对端 PE 使用 2 个 QP，并执行 put 模式
bash examples/udma_qp_demo/run.sh
```

也可以指定运行模式、QP 数和元素个数：

```bash
# 使用 4 个 QP 验证 put
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put -elems 1048576

# 使用 8 个 QP 验证 get
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 8 -op get -elems 1048576

# 使用 4 个 QP 验证 put_signal
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put_signal -elems 1048576

# 使用 4 个 QP 并启用 shared jetty 验证 put
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -shared_jetty 1 -op put -elems 1048576

# 从逻辑 NPU 4 开始绑定 4 个 PE
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 4 -op put -first_npu 4

# 使用非连续逻辑卡号；PE0/1/2/3 分别绑定 NPU1/3/2/4
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 4 -op put -device_ids 1,3,2,4
```

### Multi-link 链路模式

编译时添加 `-multi_link` 后，未传入 `-qp_net_layers` 时，QP 在可用的 MESH 和 CLOS 链路间按默认策略轮转。可以使用 `-qp_net_layers` 为指定 QP 配置链路，该配置会替换该 QP 的默认链路分配；未显式配置的 QP 保持默认链路分配，由默认轮转策略确定实际使用链路。

```bash
# 启用 multi-link 能力
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link

# 指定 QP0 使用 netLayer 0、QP2 使用 netLayer 1；QP1 保持默认链路分配
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 3 -qp_net_layers 0:0,2:1

# 相同 QP/层策略仍由所有 PE 执行；这里只改变 PE 到逻辑设备的映射
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 3 -device_ids 1,3,2,4 \
    -qp_net_layers 0:0,2:1
```

用例执行成功时，每个 PE 输出：

```text
[PASS] op=put pe=0 elements=1048576
[PASS] op=put pe=1 elements=1048576
```

PutSignal 还会输出 signal 校验结果：

```text
[PASS] op=put_signal pe=0 elements=1048576
[PASS] op=put_signal pe=0 signals=4
[PASS] op=put_signal pe=1 elements=1048576
[PASS] op=put_signal pe=1 signals=4
```

数据校验失败时，程序打印首个错误位置、实际值和期望值。`run.sh` 会等待所有 PE 进程，并在任意进程失败时返回非 0。

### 运行参数

`run.sh` 支持如下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-pes` | `2` | PE 总数，必须至少为 2；脚本启动相同数量的进程。 |
| `-qp_count` | `2` | 每个对端 PE 创建的 QP 数，范围为 `[1, 56]`，所有 PE 必须一致。 |
| `-qp_net_layers` | 未配置 | 仅在使用 `-multi_link` 编译时可用。以 `qp_idx:netLayer` 形式为指定 QP 配置 UDMA 链路，例如 `0:0,2:1`。每个 `qp_idx` 必须小于 `qp_count` 且不能重复；当前支持 layer 0/1。指定配置会替换对应 QP 的默认链路分配；未指定的 QP 保持默认链路分配，由默认轮转策略在可用 MESH 和 CLOS 链路中确定实际使用链路，所有 PE 必须使用相同配置。默认构建不支持该参数：有 MESH 时所有 QP 使用 MESH；无 MESH 时在可用 CLOS 链路中轮转。 |
| `-shared_jetty` | `0` | 是否启用 shared jetty（`0` 关闭，`1` 开启）；开启时在初始化前调用 `aclshmemx_set_shared_jetty`，所有 PE 必须一致。 |
| `-op` | `put` | 运行模式，支持 `put`、`get`、`put_signal`。 |
| `-elems` | `1048576` | 每个 PE 搬运的字节数，至少为 QP 数；支持不均分。 |
| `-heap_mb` | `1024` | 每个 PE 的 SHMEM 对称堆大小，单位为 MiB；PutSignal 模式必须同时容纳数据缓冲区和每 QP 一个 `uint64_t` signal word。 |
| `-peer_stride` | `1` | 环形通信的目标 PE 距离；实际目标、QP 网络层查询和日志均使用 `(pe + peer_stride) % pes`。取值范围为 `[1, pes)`。 |
| `-first_npu` | `0` | 本机设备起点；本机第 i 个进程使用 `first_npu + i`。 |
| `-device_ids` | 未配置 | 以逗号分隔的逻辑设备列表，数量必须等于 `pes`，且设备ID不能重复。第 i 项绑定 PE i，例如 `1,3,2,4`。配置后优先于 `first_npu`。 |
| `-ipport` | 动态本机端口 | 引导初始化地址；未指定时由脚本随机选择本机端口。 |
