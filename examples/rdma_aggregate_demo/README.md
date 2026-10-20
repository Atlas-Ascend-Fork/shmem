# RDMA aggregate 接口示例

## 示例简介

本示例验证 RDMA aggregate NBI 的 8 个 Device API 重载，并额外验证一个 5 操作的 Put 指针 batch：

| 方向 | 参数形式 | action 形式 |
| --- | --- | --- |
| Get | `__gm__` 指针 | `aclshmemx_defer_t` / `aclshmemx_submit_t` |
| Get | `GlobalTensor` | `aclshmemx_defer_t` / `aclshmemx_submit_t` |
| Put | `__gm__` 指针 | `aclshmemx_defer_t` / `aclshmemx_submit_t` |
| Put | `GlobalTensor` | `aclshmemx_defer_t` / `aclshmemx_submit_t` |

此外，Put 指针路径还包含一个独立的 5 操作 batch：前 4 条使用 `defer`，第 5 条使用 `submit`。这 5 条操作使用独立的源/目的 slot，Host 会逐条校验传输结果。所有 batch 提交后调用 `aclshmemx_roce_barrier_all`，再由 Host 校验结果。

## 支持范围和约束

- 支持 `XSCALE`（Ascend950）和 `IN_DIE`（默认后端）两种 RDMA 后端，`HNS_1825` 不支持 aggregate。
- 示例中每个 PE 使用 1 个 AI Core，通信使用 QP0。
- 每个 aggregate batch 必须复用同一个 submit state、peer、操作方向、UB buffer 和 `sync_id`。
- 4 defer + 1 submit 的 Put 指针 batch 包含 5 条操作，需要 704B UB workspace，满足 `64 + 128 * n` 的容量要求。
- Put 的源数据和 Get 的目标数据在 submit 后不能立即复用，示例使用 barrier 保证可见性。

## 环境要求

需要已经配置好的 RDMA 环境、CANN 环境以及对应网卡 provider。Ascend950 平台运行前需要根据网卡类型设置 `IBV_EXTEND_DRIVERS`：

- XSCALE 使用云脉网卡对应的 provider。

具体 RDMA 网卡检查和 provider 配置请参考 [`rdma_demo`](../rdma_demo/README.md)。

## 编译

在项目根目录执行：

```bash
# A2/A3 IN_DIE
bash scripts/build.sh -enable_rdma -examples

# XSCALE
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE -examples
```

`rdma_aggregate_demo` 在 `IN_DIE` 和 `XSCALE` 后端配置下加入 examples 构建目标，`HNS_1825` 后端下不构建。

## 运行

在项目根目录执行：

```bash
bash examples/rdma_aggregate_demo/run.sh
```

也可以指定 bootstrap 使用的 ipport：

```bash
bash examples/rdma_aggregate_demo/run.sh -ipport tcp://127.0.0.1:8899
```

成功时两个 PE 都会输出：

```text
[SUCCESS] rdma aggregate pointer/tensor defer/submit overloads, including 4-defer pointer put, passed on PE 0
[SUCCESS] rdma aggregate pointer/tensor defer/submit overloads, including 4-defer pointer put, passed on PE 1
```

## 代码结构

- `rdma_aggregate_demo_kernel.cpp`：Device kernel，调用 8 个基础 aggregate API 重载以及 4 defer + 1 submit 的 Put 指针 batch。
- `rdma_aggregate_demo_common.h`：测试布局、slot 和校验值定义。
- `main.cpp`：Host 初始化 SHMEM/RDMA、准备 symmetric buffer、启动 kernel 和校验结果。
- `run.sh`：按参数在本地启动 PE 测试进程，默认启动 2 个 PE。
- `CMakeLists.txt`：复用仓库的 `aclshmem_add_collective_example` 构建模板。

## 多 PE、多卡与跨机运行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-ipport <addr>` | `tcp://127.0.0.1:8299` | bootstrap 地址及端口，格式为 `tcp://<IP>:<端口号>`；跨机时填 PE 0 所在机器的 IP |
| `-pes <count>` | `2` | 全局 PE 数量，至少为 2 且必须是 2 的幂（2、4、8、16……） |
| `-gnpus <count>` | 与 `-pes` 相同 | 当前机器上启动的 NPU 卡的数量，同时决定本机启动的 PE 数量；不指定时默认与 `-pes` 一致 |
| `-fnpu <id>` | `0` | 当前机器执行本样例使用的第一张 NPU 卡的卡号 |
| `-fpe <id>` | `0` | 当前机器上使用的第一个 PE 号 |

### 单机 4 PE

4 个 PE 时每个 PE 与环中的下一个 PE 交互：Get 拉取下一个 PE 的数据，Put 写给下一个 PE，本 PE 收到的 Put 来自上一个 PE。设备按 `PE 号 % 卡数 + 首卡号` 映射：

```bash
bash examples/rdma_aggregate_demo/run.sh -pes 4
```

### 两机 16 PE

下面示例为两台机器、每台 8 卡，共 16 个 PE，本机启动的 PE 数量由 `-gnpus 8` 决定：

```bash
# 机器 0（PE 0~7）
bash examples/rdma_aggregate_demo/run.sh -ipport tcp://10.0.0.1:8299 -pes 16 -gnpus 8 -fpe 0
# 机器 1（PE 8~15）
bash examples/rdma_aggregate_demo/run.sh -ipport tcp://10.0.0.1:8299 -pes 16 -gnpus 8 -fpe 8
```

每个 PE 与环中的下一个 PE 通信。两台机器必须使用同一个 bootstrap 地址和端口，并确保 RDMA 网卡、provider、端口及防火墙配置正确。
