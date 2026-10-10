# SHMEM 同步/屏障接口使用说明

## 一、概述

本文档介绍 Host 侧同步/屏障接口，并说明 Device kernel 中的对应编程顺序。这些接口用于在多 PE 之间做集合同步，分为「阻塞式」与「基于 Stream」两类。返回 `int` 状态码的接口为 `aclshmem_barrier`、`aclshmemx_barrier_on_stream`、`aclshmem_team_sync` 与 `aclshmemx_sync_on_stream`；其余接口返回 `void`。

**接口一览：**

| 接口 | 返回类型 | 阻塞/流 | 说明 |
| ---- | ---- | ---- | ---- |
| aclshmem\_barrier | int | 阻塞 | 对指定 team 做屏障，返回状态 |
| aclshmem\_barrier\_all | void | 阻塞 | 对 ACLSHMEM\_TEAM\_WORLD 做屏障 |
| aclshmemx\_barrier\_on\_stream | int | 流 | 在指定流上排队 team 屏障，返回状态 |
| aclshmemx\_barrier\_all\_on\_stream | void | 流 | 在指定流上排队 ACLSHMEM\_TEAM\_WORLD 屏障 |
| aclshmem\_sync | void | 阻塞 | 对指定 team 做同步 |
| aclshmem\_team\_sync | int | 阻塞 | 对指定 team 做同步，返回状态 |
| aclshmem\_sync\_all | void | 阻塞 | 对 ACLSHMEM\_TEAM\_WORLD 做同步 |
| aclshmemx\_sync\_on\_stream | int | 流 | 在指定流上排队 team 同步，返回状态 |
| aclshmemx\_sync\_all\_on\_stream | void | 流 | 在指定流上排队 ACLSHMEM\_TEAM\_WORLD 同步 |

**阻塞与流的区别：**

- **阻塞式接口**（`aclshmem_barrier` / `aclshmem_barrier_all` / `aclshmem_sync` / `aclshmem_team_sync` / `aclshmem_sync_all`）：内部会等待流上操作完成（调用 `aclrtSynchronizeStream`），返回时集合同步已经完成。
- **流式接口**（`aclshmemx_barrier_on_stream` / `aclshmemx_barrier_all_on_stream` / `aclshmemx_sync_on_stream` / `aclshmemx_sync_all_on_stream`）：仅把同步操作排队到指定流上，返回不代表同步完成；调用者需自行 `aclrtSynchronizeStream` 或依赖后续流操作保证顺序。传入 `NULL` 时使用库管理的默认流，与 Host RMA/quiet 的 on-stream 接口一致。

**屏障、同步与通信完成：**

- sync 表示指定 team 的 PE 到达同一同步点；对参与核此前的普通对称内存存储提供适用范围内的同步与可见性，但不保证此前发出的非阻塞通信已完成。Host 的 aclshmem_team_sync 与流式 sync 均应按这一最低保证使用；不要以 sync 代替通信完成操作。
- quiet 完成调用 PE 在对应 Host 或 Device 执行范围内先前发出的通信，但不等待其他 PE 到达同步点。
- barrier 先完成相应执行范围内此前发出的通信，再等待指定 team 的所有 PE 到达同步点。所有集体调用必须由 team 内 PE 按相同顺序参与。
- 在 Device kernel 中，NBI 后如需读取远端结果或复用源缓冲区，使用 barrier；或者先 quiet 完成本 PE 通信，再 sync 等待其他 PE。fence 仅用于建立顺序，不作为通用的通信完成保证。

***

## 二、接口说明

### 1. aclshmem\_barrier

**功能描述：**

阻塞式团队屏障。所有参与者（指定 team 的每个 PE 与本地参与执行组）必须按相同顺序调用集合操作。返回时所有参与者已到达，且约定范围内此前发出的存储与远端内存更新（含 AMO、RMA 操作）均已完成，可读取目标数据并复用源缓冲区。

**函数原型：**

```cpp
ACLSHMEM_HOST_API int aclshmem_barrier(aclshmem_team_t team);
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | 需要同步的 team |

**返回值：**

- `0`：成功。
- 非 `0`：失败。

**使用示例：**

```cpp
// 伪代码示例：aclshmem_barrier 使用流程

// 1. 初始化
aclshmemx_init_attr_t attributes;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. 业务数据通信（put/get/amo 等）
// ...

// 3. 集合屏障
aclshmem_barrier(ACLSHMEM_TEAM_WORLD);

// 4. 后续读取其它 PE 已屏障前写入的数据
// ...
```

***

### 2. aclshmem\_barrier\_all

**功能描述：**

在 `ACLSHMEM_TEAM_WORLD` 上执行阻塞式屏障。语义与 `aclshmem_barrier(ACLSHMEM_TEAM_WORLD)` 一致；选择 WORLD 不改变本地参与范围。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmem_barrier_all(void);
```

**使用示例：**

```cpp
// 伪代码示例：aclshmem_barrier_all 使用流程
aclshmem_barrier_all();
// 屏障返回后，所有 PE 已在此前发出的通信操作均已完成
```

***

### 3. aclshmemx\_barrier\_on\_stream

**功能描述：**

在指定流上排队一次团队屏障操作。成功返回仅表示该屏障已被接受排队；完成保证在所选流执行到并完成该屏障时成立。需调用 `aclrtSynchronizeStream` 或依赖流内顺序观察完成。

**函数原型：**

```cpp
ACLSHMEM_HOST_API int aclshmemx_barrier_on_stream(aclshmem_team_t team, aclrtStream stream);
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | 需要同步的 team |
| stream | aclrtStream | 屏障操作排队的流（为 NULL 时使用默认流） |

**返回值：**

- `0`：成功。
- 非 `0`：失败。

**使用示例：**

```cpp
// 伪代码示例：aclshmemx_barrier_on_stream 使用流程
aclrtStream stream;
aclrtCreateStream(&stream);

aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream);
aclrtSynchronizeStream(stream);

aclrtDestroyStream(stream);
```

***

### 4. aclshmemx\_barrier\_all\_on\_stream

**功能描述：**

在指定流上排队一次 `ACLSHMEM_TEAM_WORLD` 屏障操作；选择 WORLD 不改变本地参与范围。成功返回仅表示该屏障已被接受排队；完成保证在所选流执行到并完成该屏障时成立。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmemx_barrier_all_on_stream(aclrtStream stream);
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| stream | aclrtStream | 屏障操作排队的流（为 NULL 时使用默认流） |

**使用示例：**

```cpp
// 伪代码示例：aclshmemx_barrier_all_on_stream 使用流程
aclshmemx_barrier_all_on_stream(stream);
aclrtSynchronizeStream(stream);
```

***

### 5. aclshmem\_sync

**功能描述：**

阻塞式团队同步。提供集合汇合及约定范围内普通存储的可见性，但不会完成尚未完成的非阻塞通信。调用后阻塞，直到指定 team 内的所有 PE 都调用了该同步才返回；需要完成先前异步通信时，请先调用 `aclshmem_quiet` 等完成接口。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmem_sync(aclshmem_team_t team);
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | 需要同步的 team |

**使用示例：**

```cpp
// 伪代码示例：aclshmem_sync 使用流程
aclshmem_sync(ACLSHMEM_TEAM_WORLD);
```

***

### 6. aclshmem\_team\_sync

**功能描述：**

阻塞式团队同步。提供集合汇合及约定范围内普通存储的可见性，但不会完成尚未完成的非阻塞通信。与 `aclshmem_sync` 的区别在于本接口返回状态码；需要完成先前异步通信时，请先调用 `aclshmem_quiet` 等完成接口。

**函数原型：**

```cpp
ACLSHMEM_HOST_API int aclshmem_team_sync(aclshmem_team_t team);
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | 需要同步的 team |

**返回值：**

- `0`：成功。
- 非 `0`：失败。

**使用示例：**

```cpp
// 伪代码示例：aclshmem_team_sync 使用流程
int status = aclshmem_team_sync(ACLSHMEM_TEAM_WORLD);
if (status != 0) {
    // 处理失败
}
```

***

### 7. aclshmem\_sync\_all

**功能描述：**

在 `ACLSHMEM_TEAM_WORLD` 上执行阻塞式同步。语义与 `aclshmem_sync(ACLSHMEM_TEAM_WORLD)` 一致；选择 WORLD 不改变本地参与范围。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmem_sync_all(void);
```

**使用示例：**

```cpp
// 伪代码示例：aclshmem_sync_all 使用流程
aclshmem_sync_all();
```

***

### 8. aclshmemx\_sync\_on\_stream

**功能描述：**

在指定流上排队一次团队同步操作。成功返回仅表示该同步已被接受排队；完成保证在所选流执行到并完成该同步时成立。需调用 `aclrtSynchronizeStream` 或依赖流内顺序观察完成。

**函数原型：**

```cpp
ACLSHMEM_HOST_API int aclshmemx_sync_on_stream(aclshmem_team_t team, aclrtStream stream);
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | 需要同步的 team |
| stream | aclrtStream | 同步操作排队的流（为 NULL 时使用默认流） |

**返回值：**

- `0`：成功。
- 非 `0`：失败。

**使用示例：**

```cpp
// 伪代码示例：aclshmemx_sync_on_stream 使用流程

// 1. 创建流
aclrtStream stream;
aclrtCreateStream(&stream);

// 2. 排队同步
int status = aclshmemx_sync_on_stream(ACLSHMEM_TEAM_WORLD, stream);
if (status != 0) {
    // 处理失败
}

// 3. 等待流上同步完成
aclrtSynchronizeStream(stream);

// 4. 释放资源
aclrtDestroyStream(stream);
```

***

### 9. aclshmemx\_sync\_all\_on\_stream

**功能描述：**

在指定流上排队一次 `ACLSHMEM_TEAM_WORLD` 同步操作；选择 WORLD 不改变本地参与范围。成功返回仅表示该同步已被接受排队；完成保证在所选流执行到并完成该同步时成立。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmemx_sync_all_on_stream(aclrtStream stream);
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| stream | aclrtStream | 同步操作排队的流（为 NULL 时使用默认流） |

**使用示例：**

```cpp
// 伪代码示例：aclshmemx_sync_all_on_stream 使用流程
aclshmemx_sync_all_on_stream(stream);
aclrtSynchronizeStream(stream);
```

***

## 三、Device kernel 中的使用顺序

以下两种顺序均要求 team 内所有 PE 和声明参与范围内的核按相同顺序调用集体接口：

    // 方式一：NBI -> barrier
    aclshmem_putmem_nbi(dst, src, bytes, peer);
    aclshmem_barrier_all();
    // barrier 返回后，先前通信完成；可读取对应目标数据、复用源缓冲区。

    // 方式二：NBI -> quiet -> sync
    aclshmem_putmem_nbi(dst, src, bytes, peer);
    aclshmem_quiet();     // 本 PE 的先前通信完成，可复用本地源缓冲区
    aclshmem_sync_all();  // 等待所有 PE 到达；之后再读取其他 PE 的写入结果

普通对称内存 store 的同步/可见性与通信 API 的完成范围不同：sync 同步参与核在受支持的 HCCS/UB 内存域中的普通存储，但不能代替 quiet 完成 NBI；barrier 同时提供通信完成与 PE 到达同步。对混合核，Cube 标量单元不在这些 API 的同步保证范围内；参与核范围见 Device public header。Host 发出的通信应使用 Host 对应接口完成，Device 发出的通信应在 Device 执行范围内完成。

***

## 四、注意事项

1. **初始化检查**：调用前需完成 `aclshmemx_init_attr` 初始化；未初始化时，返回状态码的接口会返回 `ACLSHMEM_NOT_INITED`，`void` 接口会打印错误日志后直接返回。
2. **team 有效性**：非成员传入哨兵值 `ACLSHMEM_TEAM_INVALID` 时，`aclshmem_barrier` / `aclshmem_sync` / `aclshmem_team_sync` / `aclshmemx_barrier_on_stream` / `aclshmemx_sync_on_stream` 为空操作并返回成功。
3. **阻塞式接口内部同步**：阻塞式接口内部会 `aclrtSynchronizeStream`，因此调用返回即代表集合同步完成；流式接口则需要调用者自行同步流。
4. **参与范围**：集合同步要求 team 内所有 PE 都调用相同接口，否则会阻塞等待直至超时或卡死。
5. **完成与缓冲区复用**：sync 仅保证 PE 到达同步点，不能单独保证 NBI 完成。完成本 PE 通信后才能复用该通信的源缓冲区；所有相关 PE 完成通信并到达同步点后，才读取其它 PE 写入的目标数据。Device 使用 barrier，或按 quiet 再 sync 的顺序调用。
6. **Host 流的等待范围**：Host 阻塞接口使用并等待库默认流；流式接口仅排队，须等待该集体操作所在流或建立明确的跨流依赖。对先前在其他流发出的工作，先建立到集体操作所在流的依赖；仅等待另一条流不能证明该集体操作已完成。单独同步启动 Device kernel 的流，也不能替代 kernel 内对异步通信的 quiet 或 barrier。
