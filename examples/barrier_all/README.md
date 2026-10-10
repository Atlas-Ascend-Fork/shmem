# barrier_all 示例

示例验证 `aclshmem_barrier_all()` 的通信完成语义，分两部分：

- **Device**：在 kernel 内用高阶接口 `aclshmem_putmem_nbi()` 向每个其它 PE 发起非阻塞传输；调用 Device `aclshmem_barrier_all()` 后，Host 回读并逐元素校验。
- **Host**：同样用高阶接口 `aclshmem_putmem_nbi()` 向每个其它 PE 传输；调用 Host `aclshmem_barrier_all()` 后，回读并逐元素校验。

启动脚本按拓扑设置初始化引擎，高阶接口再按 PE 的位置选择传输：

- **单机**：所有 PE 都在本机，只走 MTE。
- **双机**：同机 PE 走 MTE，跨机 PE 走 RDMA（即 MTE + RDMA）。

不传引擎参数时可执行文件仅初始化 MTE；跨机时启动脚本显式传入 `--roce`，初始化 MTE + ROCE。因此单机可使用默认构建，跨机需要开启 RDMA 支持。

本机多卡需支持 MTE 互访；双机需额外配置 RDMA，环境配置参考 [rdma_demo](../rdma_demo/README.md)。

## 编译

加载 CANN 环境后，在项目根目录执行：

```bash
# 单机（默认构建，仅 MTE）
bash scripts/build.sh -examples

# A2/A3 双机（MTE + RDMA）
bash scripts/build.sh -examples -enable_rdma

# Ascend950 单机（仅 MTE）
bash scripts/build.sh -examples -soc_type Ascend950

# Ascend950 双机：按网卡选择 XSCALE
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE
# Ascend950 双机：按网卡选择 HNS_1825
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend HNS_1825
```

## 运行

```text
bash examples/barrier_all/run.sh <n_pes> <n_pes_per_node> <node_id> <tcp://rank0_ip:port> [first_device_id:0]
```

- `n_pes`：PE 总数。
- `n_pes_per_node`：每台机器的卡数。
- `node_id`：本机编号（从 0 开始）。
- `first_device_id`：本机第一张卡的设备号（默认 0）。

### 单机 8 卡（只走 MTE）

```bash
bash examples/barrier_all/run.sh 8 8 0 "tcp://<rank0-IP>:8899"
```

### 双机每台 8 卡（MTE + RDMA）

在两台服务器分别执行，使用相同的节点 0 IP 和端口：

```bash
# 节点 0
bash examples/barrier_all/run.sh 16 8 0 "tcp://<节点0-IP>:8899"

# 节点 1
bash examples/barrier_all/run.sh 16 8 1 "tcp://<节点0-IP>:8899"
```

脚本等待本机全部进程完成，任一进程失败则返回非零；默认超时 120 秒，可通过 `BARRIER_ALL_TIMEOUT` 设置秒数。所有 PE 的 Device、Host 校验均通过才表示示例成功。

单个 PE 的可执行文件参数为：

```text
./build/bin/barrier_all <pe_id> <device_id> <n_pes> <tcp://rank0_ip:port> [--roce]
```

`--roce` 仅用于跨机运行，且要求使用 `-enable_rdma` 构建；不指定时只启用 MTE。

默认构建的双 PE 单机快速验证（需两张本机 NPU 和 CANN 环境）：

```bash
bash examples/barrier_all/run.sh 2 2 0 "tcp://<本机IP>:8899"
```

默认使用设备 0、1；如需指定起始设备号，在命令末尾添加 `first_device_id`。脚本仅初始化 MTE；两个 PE 的 Device、Host 校验均通过且脚本返回 0，表示验证通过。
