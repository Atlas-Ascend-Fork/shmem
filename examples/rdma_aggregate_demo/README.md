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
- 示例固定使用 2 个 PE、1 个 AI Core 和 QP0。
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

也可以指定 bootstrap 使用的 IP 和端口：

```bash
bash examples/rdma_aggregate_demo/run.sh -ip 127.0.0.1 -port 8899
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
- `run.sh`：启动两个 PE 的本地测试进程。
- `CMakeLists.txt`：复用仓库的 `aclshmem_add_collective_example` 构建模板。
