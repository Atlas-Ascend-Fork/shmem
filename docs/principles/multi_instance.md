# 多实例支持

[English](multi_instance_en.md)

## 概述

SHMEM 多实例能力允许**在单个进程内创建多个独立 SHMEM 实例**。每个实例拥有独立的通信域、内存空间和同步资源，适用于以下场景：

- **多通信域共存**：单个进程中多个任务域需要独立 SHMEM 上下文。
- **资源隔离**：不同任务/模块之间隔离 SHMEM 资源，避免相互干扰。
- **弹性扩缩**：不同实例创建/释放互不影响。
- **动态实例管理**：运行时按 `instance_id` 创建/释放实例。

通信链路支持 MTE 与 UDMA

## 核心概念

### 实例（Instance）

一个 SHMEM 实例包含一组完整初始化状态，典型包括：

- 独立的通信域（`Team World`）。
- 独立的共享内存堆（`Heap`）。
- 独立的同步资源（`Sync Pool/Counter`）。
- 独立的 State/Bootstrap/MemoryManager 运行时快照。

### 实例标识（Instance ID）

当前仓库使用 `instance_id`（`uint64_t`）标识实例。实例生命周期由初始化属性和 `finalize` 接口管理。

### Context 切换

Host 通过**全局变量 swap 机制**完成实例切换：先回写当前实例的全局状态，再装载目标实例的全局状态。Host 与 Device 均使用 `aclshmemx_instance_ctx_set`，但二者选择各自执行域的上下文。Device 侧的 UDMA 提交和匹配完成等待必须使用同一实例，切换前完成旧实例的未完成操作。Host 在 quiet 入队并捕获实例后可以切换实例；旧实例、stream 和缓冲区必须保留到完成，并满足跨 stream 依赖和 Host 多线程串行化要求。

## 架构设计

```text
┌──────────────────────────────────────────────────────────┐
│                          Process                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │                    SHMEM Runtime                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│  │  │  Instance 0 │  │  Instance 1 │  │ Instance N │  │  │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌──────┐  │  │  │
│  │  │  │ Team  │  │  │  │ Team  │  │  │  │ Team │  │  │  │
│  │  │  │ Heap  │  │  │  │ Heap  │  │  │  │ Heap │  │  │  │
│  │  │  │ Sync  │  │  │  │ Sync  │  │  │  │ Sync │  │  │  │
│  │  │  │ State │  │  │  │ State │  │  │  │ State│  │  │  │
│  │  │  └───────┘  │  │  └───────┘  │  │  └──────┘  │  │  │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 设计点 | 当前实现方案 | 说明 |
|---|---|---|
| 实例隔离 | 独立 `aclshmem_context` 快照 | 每个实例保存独立的 `state`、`host_state`、`boot_handle`、`memory_manager` 等 |
| Context 管理 | 全局变量 swap | 通过 `aclshmemx_instance_ctx_set` 切换 |
| 实例标识 | `instance_id` (`uint64_t`) | 位于 `aclshmemx_init_attr_t`，并用于 `aclshmemx_finalize(instance_id)` |
| Device 侧路由 | `instance_id << 1` | kernel 侧通过硬编码地址访问对应实例的 `state` |

## 规格与限制（按代码实现）

| 项目 | 规格 |
|---|---|
| `instance_id` 类型 | `uint64_t` |
| 默认实例 | `instance_id = 0` |
| 实例 ID 槽位 | HBM-only 为 0～255；需要 Host DRAM 伴生实体时为 0～254，仍受端口范围和资源约束 |
| PE 上限 | `ACLSHMEM_MAX_PES = 16384` |
| 本地内存上限 | `ACLSHMEM_MAX_LOCAL_SIZE = 40GB` |
| 实例共享内存上限 | 目前为128GB， 超过128GB时底层虚拟地址冲突 |
| 非 0 实例 Team 行为 | 初始化时仅保留 world team 槽位；team split/translate/get_config 等返回 `ACLSHMEM_NOT_SUPPORTED` |
| default 多实例端口分配 | 依赖环境变量 `SHMEM_INSTANCE_PORT_RANGE=start:end` (start、end由用户自行指定，每个实例用到的实际端口为 `start_port + instance_id`) |
| 通信链路支持 | MTE、UDMA |
| UDMA QP 配置 | 首次初始化前通过 `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num)` 设置进程级 QP 数；存在活动实例时不能修改。队列资源、metadata 和完成状态按实例隔离 |
| Host UDMA quiet | `aclshmemx_quiet_on_stream(stream)`：入流时捕获当前实例；非零实例等待所有 QP（含 shared Jetty），实例 0 保持原有 QP0 完成范围，非零 QP 使用 `aclshmemx_udma_qp_quiet` |
| UDMA 实例 ID 范围 | HBM-only 初始化为 0～255；`USE_ACLRT_MEM_FABRIC_HANDLE` 创建 Host DRAM 伴生实体时最多 254。初始化超出 HBM 上界返回 `ACLSHMEM_INVALID_VALUE`；quiet 只接受已成功初始化且存活的实例，超出 255 在入流前拒绝 |

## 环境变量与运行要求

### 算子侧使用要求

- Device 接口（含 UDMA）：编译时添加 `MULTI_INSTANCE`，每个参与的 AIV 在操作前调用 `aclshmemx_instance_ctx_set(id)`，ID 必须对应已成功初始化且存活的实例。所有构建均在计算上下文地址前检查 ID 范围，越界立即 trap；错误日志仅在 Debug 输出。范围内的 ID 是否已初始化且存活仍由调用者保证，上下文一致性检查仅作 Debug 诊断。切换实例前完成旧实例的未完成操作，UDMA batch 的提交和完成等待须使用同一实例。
- UDMA 发送 kernel 返回前须完成本地 staging，并通过公开提交接口提交待发送的 batch；库负责内部队列状态发布。发送与 quiet 须在同一 stream 上有序执行，或显式建立跨 stream 的 event 依赖，不能在其他 stream 上同时提交或 quiet 同一队列。
- 显式使用非零 QP 时，用 `aclshmemx_udma_qp_quiet(peer, qp)` 完成等待。
- 通信完成前保持对应实例、stream 和内存有效。

### default 多实例模式

- 必须设置 `SHMEM_INSTANCE_PORT_RANGE`，格式 `start:end`。
- 初始化属性中的 `ip_port` 端口必须为 `0`，例如 `tcp://x.x.x.x:0`。

### unique id 模式

- 仍遵循现有 `SHMEM_UID_SESSION_ID` / `SHMEM_UID_SOCK_IFNAME` 规则。

## 注意事项

- 重复创建同一 `instance_id` 时：会有WARN日志提示并立即返回（不会重复创建）。
- `aclshmemx_finalize(instance_id)` 仅在目标实例非当前活动实例时才切换上下文，再执行释放；传入不存在的 `instance_id` 会返回错误，不会误伤当前活动实例。
- `aclshmem_finalize()` 释放当前活动实例，不会切换上下文；多实例下需配合 `aclshmemx_instance_ctx_set` 使用，或直接改用 `aclshmemx_finalize`。
- 实例上下文相关接口路径由互斥锁保护。
- Host 当前实例是进程级状态：多线程调用时，上层须用同一把锁保护 `aclshmemx_instance_ctx_set` 与 quiet 调用的完整序列，所有切换实例的线程都须遵守该约定。
- Host 侧 UDMA quiet 入队后实例已固定，后续切换 `instance_id` 不会改变已提交的 quiet；实例、stream 与通信缓冲区须保留到 quiet 完成，同步该 stream 并协调接收 PE 后才能读数据或释放/销毁实例。
- `aclshmemx_quiet_on_stream(NULL)` 使用该实例的默认 stream；`aclshmem_quiet()` 除入流外还会同步等待捕获的默认 stream。

## 快速上手（当前仓库 API）

### 1. 创建实例

```cpp
#include "shmem.h"
#include <cstdio>

aclshmemx_init_attr_t attr{};
attr.my_pe = pe_id;
attr.n_pes = pe_size;
attr.local_mem_size = 1024UL * 1024UL * 1024UL;
std::snprintf(attr.ip_port, sizeof(attr.ip_port), "%s", "tcp://x.x.x.x:0");       // 预先置为0,后续会分配
attr.comm_args = nullptr;               // default mode
attr.instance_id = 2;

int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

### 2. 切换到目标实例 Context

```cpp
aclshmemx_instance_ctx_set(2);

// 后续 SHMEM API 均作用于实例 2
void *ptr = aclshmem_malloc(1024);
aclshmem_free(ptr);
```

### 3. 获取当前实例

```cpp
aclshmem_instance_ctx *ctx = aclshmemx_instance_ctx_get();
// 当前主要用于读取 id，其他能力见上一节
uint64_t cur_id = (ctx == nullptr) ? 0 : ctx->id;
```

### 4. 释放实例

多实例场景推荐直接使用 `aclshmemx_finalize`，按 id 释放指定实例（内部会自动切换上下文，无需手动 set）：

```cpp
aclshmemx_finalize(2);
```

也可以用标准接口 `aclshmem_finalize()` 释放**当前活动实例**（即最近一次 `aclshmemx_instance_ctx_set` 设置的实例，未 set 过则为 instance 0）。若要用它释放某个特定实例，需先 set 再调用：

```cpp
aclshmemx_instance_ctx_set(2);
aclshmem_finalize();   // 释放当前活动实例 2
```

单实例场景下 `aclshmem_finalize()` 直接释放 instance 0，与历史无参用法完全一致。

### 5. UDMA 多实例

Host 创建实例时设置实例的 UDMA 引擎：

```cpp
attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

Device kernel 编译启用 `MULTI_INSTANCE`，每个参与 AIV 在入口设置实例，并在同一实例内完成 put 与 quiet：

```cpp
aclshmemx_instance_ctx_set(instance_id);
aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(dst, src, ub, bytes, peer, 0);
aclshmemx_udma_quiet(peer);
```

也可以在发送 kernel 之后，由 Host 选择对应实例并调用 `aclshmemx_quiet_on_stream` 完成该实例已提交的 UDMA 操作：

```cpp
aclshmemx_instance_ctx_set(2);
aclshmemx_quiet_on_stream(stream);   // 传入 NULL 使用该实例的默认 stream
aclrtSynchronizeStream(stream);
```

### 6. MTE 完整样例

[examples/multi_instance](../../examples/multi_instance/README.md)（基于 `aclshmemx_mte_put_nbi` / `aclshmemx_mte_get_nbi`）

## 与单实例模式对比

| 特性 | 单实例模式 | 多实例模式 |
|---|---|---|
| 初始化标识 | 默认 `instance_id=0` | `instance_id>0` |
| 资源隔离 | 全局共享 | 按实例隔离 |
| 上下文切换 | 默认实例 0 | Host/Device 使用 `aclshmemx_instance_ctx_set` |
| 生命周期管理 | `aclshmem_finalize()` | `aclshmemx_finalize(instance_id)` |
| 通信链路 | 支持平台配置下的 MTE、UDMA、RDMA/SDMA | 本文新增按实例路由的 MTE、UDMA |
| Team 高级操作 | 支持 | 仅 `instance_id=0` 支持 |

## 相关代码与文档

- 初始化/终止流程（含多实例 ctx 创建、端口分配与 finalize）：[init_finalize.md](init_finalize.md)
- Host   多实例切换：[src/host/init/shmem_init.cpp](../../src/host/init/shmem_init.cpp)
- Device 多实例切换：[src/device/shmemi_device_common.hpp](../../src/device/shmemi_device_common.hpp)
- Team   多实例限制逻辑：[src/host/team/shmem_team.cpp](../../src/host/team/shmem_team.cpp)
- MTE 示例：[examples/multi_instance](../../examples/multi_instance/README.md)
- UDMA 多实例用例：[multi_instance_routing_host_test.cpp](../../tests/unittest/host/multi_instance/multi_instance_routing_host_test.cpp)

---

最后更新：2026-11-12
