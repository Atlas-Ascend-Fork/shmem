# rdma_perftest

## 概述

`rdma_perftest` 用于测试 shmem RDMA 低阶接口（`aclshmemx_roce_put_nbi` / `aclshmemx_roce_get_nbi`）在不同数据量下的传输性能，通过 [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) 宏采集性能数据，支持带宽（`bw`）和接口延迟（`lat`）两种指标。

> 该测试结果仅做参考，性能以实际场景为准。

## 测试模式

| 模式 | 说明 |
|------|------|
| `put` | 单向 Put：PE0 调用 RDMA put 将数据传输到 PE1 |
| `bi_put` | 双向 Put：两个 PE 同时互相 put |
| `get` | 单向 Get：PE0 调用 RDMA get 从 PE1 拉取数据 |
| `bi_get` | 双向 Get：两个 PE 同时互相 get |

## 与 mte_perftest、udma_perftest 的差异

| 维度 | `mte_perftest` | `udma_perftest` | `rdma_perftest` |
|------|---------------|-----------------|-----------------|
| 引擎 | 默认 MTE | 显式 `ACLSHMEM_DATA_OP_UDMA` | RDMA 引擎 |
| 并发能力 | 同 peer 多核（默认 32 核切分数据） | 强制单核（UDMA 不允许同 peer 并发） | 单 QP；XSCALE 和 HNS_1825 支持多 QP 并行；IN_DIE/XSCALE 支持聚合提交 |
| `-b/--block-size` | 控制核数 | 兼容入参，强制 1 | 兼容入参，设置任何值都不生效，实际并行度由 `-q/--qp` 决定 |
| UB 缓冲 | MTE 必需 | UDMA 必须 | 必须，至少 192B，默认 192B |
| 测试模式 | put / bi_put / get / bi_get | put / bi_put / get / bi_get / **put_signal** | put / bi_put / get / bi_get |
| SOC 限制 | 通用 | 仅 Ascend950 | Ascend950（需 XSCALE/HNS_1825 后端）或 A2/A3 |
| CSV 文件名 | `<test>_<dtype>_<pe>.csv` | `udma_<test>_<dtype>_<pe>.csv` | `rdma_<metric>_<test>_<dtype>_<pe>.csv` |

## 环境要求

同 [rdma_demo](../../rdma_demo/README.md) 中的环境要求。

## 编译说明

RDMA 功能需在编译时启用 `-enable_rdma` 参数，并根据 SOC 类型配置后端。详见 [编译与构建 - RDMA 参数使用说明](../../../docs/compilation_build_guide.md#rdma参数使用说明)。

### RDMA 后端能力

下表描述 RDMA 后端接口本身是否具备对应能力，不表示 `rdma_perftest` 已经覆盖该能力。

| 后端能力 | IN_DIE | XSCALE | HNS_1825 |
|----------|--------|--------|----------|
| 适用平台 | A2/A3 | Ascend950（云脉网卡） | Ascend950（1825 网卡） |
| 基础 Put/Get NBI | 支持 | 支持 | 支持 |
| 基础 Quiet 和同步 | 支持 | 支持 | 支持 |
| 普通单 QP 聚合功能 | 支持 | 支持 | 不支持 |
| 多 QP 功能 | 不支持 | 支持 | 支持 |
| 多 QP 聚合功能 | 不支持 | 支持 | 不支持 |
| RDMA 原子操作 | 不支持 | 支持 | 支持 |

### rdma_perftest 支持范围

下表描述本测试在各后端实际开放的功能。后端具备能力但本测试未覆盖时，仍标记为“不支持”。

| 本测试功能 | IN_DIE | XSCALE | HNS_1825 |
|------------|--------|--------|----------|
| Put/Get 与单向/双向模式 | 支持 | 支持 | 支持 |
| 单 QP 带宽测试 | 支持 | 支持 | 支持 |
| 单 QP 时延测试 | 支持 | 支持 | 支持 |
| 带宽测试的 `batch` 分组与完成等待 | 支持 | 支持 | 支持 |
| 时延测试自动遍历接口形态 | `plain` + `defer` + `submit` | `plain` + `qp` + `defer` + `submit`（`qp_num <= 2`） | `plain` + `qp` |
| 显式 QP 数量和 QP 编号 | 不支持，`run.sh` 报错退出 | 支持，QP 数量范围 1~32 | 支持，QP 数量范围 1~32 |
| 多 QP 并行带宽测试 | 不支持 | 支持，QP 数量范围 2~32 | 支持，QP 数量范围 2~32 |
| 固定 QP 诊断模式 | 不支持 | 支持 | 支持 |
| 聚合提交 | 支持 | 支持 | 不支持 |

## 使用方法

```bash
cd examples/shmem_perftest/rdma_perftest/
bash run.sh [选项]
```

> Ascend950 平台需设置 `IBV_EXTEND_DRIVERS` 环境变量，参见 [环境变量说明](../../rdma_demo/README.md#ibv_extend_drivers-环境变量)。

### 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t` | 测试类型：`put` / `bi_put` / `get` / `bi_get` / `all` | `put` |
| `--datatype <type>` | `-d` | 数据类型：`float` / `int8` / `int16` / `int32` / `int64` / `uint8` / `uint16` / `uint32` / `uint64` / `char` / `all` | `float` |
| `--exponent <n>` | `-e` | 数据量幂数（2^n 字节） | - |
| `--exponent-range <min> <max>` | - | 数据量幂数范围 | `3 17` |
| `--loop-count <n>` | - | 循环次数 | `1000` |
| `--metric <bw\|lat>` | - | 性能指标：`bw`=带宽，`lat`=接口延迟 | `bw` |
| `--ub-size <n>` | - | UB 大小（B），自动 64B 对齐，范围 192~131136 | `192` |
| `--batch <n>` | - | `bw`：连续发起多少次操作后等待完成；`lat`：聚合接口的窗口大小（详见下方说明） | `0` |
| `-q/--qp/--qp-count <n>` | `-q` | QP 数量，即两个 PE 之间的队列对数，范围 1~32 | `1` |
| `-i/--qp-index <n>` | `-i` | 固定使用的 QP 编号；`-1` 表示多 QP 并行模式自动分配 | `-1` |
| `--sync-id <id>` | - | 显式传给 Put/Get/Quiet 的同步 ID | `0` |
| `-b/--block-size <n>` | `-b` | 兼容入参，设置任何值都不生效（仅打印 WARN），实际并行度由 `--qp` 决定 | `1` |
| `--block-range <min> <max>` | - | 兼容入参，同上 | `1 1` |
| `-pes <n>` | - | PE 数量（强制为 2） | `2` |
| `-ipport <ip:port>` | - | 通信地址 | `tcp://127.0.0.1:8768` |
| `-gnpus <n>` | - | NPU 数量 | `2` |
| `-fnpu <id>` | - | 首个 NPU ID | `0` |
| `-a/--analyse <mode>` | - | 分析模式：`none` / `plot` / `md` | `none` |

### HBM 与对称内存约束

- 仅测试 HBM（`DEVICE_SIDE`）内存路径，**不支持 D2H / `HOST_SIDE` (DRAM)**。
- 通过 `aclshmem_malloc` 分配输入/输出缓冲区，Put/Get 的本地和远端操作数都必须指向对称内存。
- 默认 1 GB 本地内存，数据量较大时自动上调（最多 40 GB）。

### 关键参数说明

#### 什么是 QP

QP（Queue Pair，队列对）是 RDMA 通信的基本单元，每个 QP 维护独立的发送队列和接收队列。`-q/--qp/--qp-count` 指定两个 PE 之间建立的 QP 数量（范围 1~32），`-i/--qp-index` 指定测试时使用哪个 QP。

#### 单 QP 模式（默认，不传 `-q` 和 `-i`）

不传任何 QP 参数时，两个 PE 之间使用 1 个 QP 完成所有传输。每次测试 `loop-count` 次数据传输，`DataSize/B` 是每次传输的数据量，带宽 = 总数据量 / 总耗时。

IN_DIE 在单 QP 且消息严格小于 32 KiB、XSCALE 在 `qp_num <= 2` 且消息严格小于 64 KiB 的带宽测试中自动聚合提交；HNS_1825 始终使用普通 QP 指定 NBI 路径。

#### 多 QP 并行模式（`--qp N --qp-index -1`）

使用 N 个 QP 并行传输，提升总吞吐。每个 QP 使用独立的数据区，各自完成 `loop-count` 次传输。`DataSize/B` 是**单个 QP** 每次传输的数据量，每轮测试的总数据量为 `DataSize × N`。带宽按所有 QP 完成这些数据的共同时间窗口计算。

- 此模式需配合 `--metric bw` 使用。`-q N` 指定 QP 数量（范围 1~32），`-i -1` 表示自动为每个 QP 分配独立数据区。
- 例如 `-q 4 -i -1`、`DataSize=4MB`、`loop-count=1000`：4 个 QP 各自提交 1000 次 4MB 传输，单次并发总数据量 16MB，累计总数据量 16GB。
- IN_DIE 在单 QP 且消息严格小于 32 KiB、XSCALE 在 `N <= 2` 且消息严格小于 64 KiB 时聚合提交；HNS_1825 始终立即提交。

> **云脉（XSCALE）和 1825（HNS_1825）** 均支持多 QP。其他后端会被 `run.sh` 拦截并报错退出。

#### 单 QP 诊断模式（`--qp N --qp-index K`）

固定使用第 K 个 QP（0 ≤ K < N）完成全部测试，其他 QP 不参与。用于比较单个 QP 的行为或排查特定 QP 的问题。

- `-q N` 声明 QP 池大小，`-i K` 固定选择第 K 个 QP。`-i` 必须满足 `-1 ≤ K < N`，否则 `run.sh` 会报错退出。
- 时延测试（`--metric lat`）要求 `qp_num=1`，不支持 `qp_num>1`；计时只覆盖唯一 QP 的传输耗时。
- IN_DIE 使用单 QP 普通接口聚合；XSCALE 的单 QP、固定 QP 和多 QP 模式遵循聚合条件；HNS_1825 始终调用 QP 指定的普通 NBI 接口立即提交。

> **云脉（XSCALE）和 1825（HNS_1825）** 均支持固定 QP 诊断模式。其他后端会被 `run.sh` 拦截并报错退出。

#### 批量与完成语义（`--batch`）

`--batch N` 控制带宽路径（`--metric bw`）中操作的分组方式：连续发起 N 次非阻塞操作后，等待这一组全部完成，再开始下一组。

- 完成语义：每组操作完成后，**Put 的源缓冲区可安全复用**（数据已被远端接收），**Get 的目标缓冲区数据已就绪可读**。
- 直接执行二进制时，`batch=0` 或 `batch > loop_count` 按 `loop_count` 处理，即所有操作完成后统一等待一次。

#### 时延路径的接口遍历（`--metric lat`）

`--metric lat` 测量接口下发时延（不含完成等待）。一次时延测试自动遍历当前后端支持的**全部接口形态**，无需通过 `--batch` 选择：

| 接口形态 | 说明 | IN_DIE | XSCALE | HNS_1825 |
|----------|------|--------|--------|----------|
| `plain` | 普通非聚合接口，逐次调用 `put/get_nbi` 立即提交 | 支持 | 支持 | 支持 |
| `qp` | QP 指定接口，逐次调用 QP 指定 NBI 接口 | 不支持 | 支持 | 支持 |
| `defer` | 聚合 defer 接口（IN_DIE 普通 defer，XSCALE QP defer），攒批不提交 | 支持 | 支持（`qp_num <= 2`） | 不支持 |
| `submit` | 聚合 submit 接口（IN_DIE 普通 submit，XSCALE QP submit），提交攒批 | 支持 | 支持（`qp_num <= 2`） | 不支持 |

- 各形态独立 warmup + 计时，执行顺序为 `submit`、普通接口、QP 接口、`defer`；结果仍分别写入 CSV 的 `PlainLat/us`、`QpLat/us`、`DeferLat/us`、`SubmitLat/us` 列；后端不支持的形态显示 `-`。
- `submit` 形态与普通接口保持相同的计时方法：连续调用 `loop_count` 次 submit，记录整段开始和结束时间，最后除以 `loop_count`；quiet 不计入计时。`defer` 形态对窗口内 N 次 defer 调用连续整体计时后除以 N（最后 1 次 submit 补齐并提交聚合组，不计时）。结果均为摊薄到单次调用的接口下发时延。
- submit 的 warmup 也逐次使用独立 submit state，避免一次聚合 warmup 与正式 submit 的队列状态不同。
- `--batch N` 仅作为聚合接口的窗口大小（默认自动 100），不受带宽模式的消息大小阈值限制。defer 窗口包含 N 次 defer 和一次自带 WQE 的 submit，共 N+1 条；warmup 则共 100 条。`run.sh` 与二进制按 `64 + 128 × max(100, N+1)` 自动上调 UB：N=100 时为 12992B，N=200 时为 25792B。时延窗口上限为 1022，超过时自动缩小，保证完整批次最多 1023 条、小于 SQ 深度 1024；带宽模式的批次计数不追加这条 WQE。

示例：`./run.sh -t put --metric lat` 一次测出 submit、普通、QP、defer 四种接口时延；`./run.sh -t put --metric lat --batch 200` 将聚合窗口改为 200。

#### 多 QP 后端识别

`run.sh` 通过 `IBV_EXTEND_DRIVERS` 环境变量或 `ibv_devinfo` 输出识别 XSCALE 和 HNS_1825 环境。仅这两个后端接受显式 QP 数量和 QP 编号；其他后端传入显式 QP 参数会报错退出。IN_DIE 和 XSCALE 会按聚合条件调整 UB 和 batch 参数。

带宽边界：16/32/64 KiB 在 IN_DIE 单 QP 下分别走聚合/普通/普通，在 XSCALE（`qp_num <= 2`）下分别走聚合/聚合/普通。

`run.sh` 会在 BW 聚合路径自动校正 batch 和 UB，容量至少为 `64 + 128 × max(100, batch_size)` 字节，覆盖 Put/Get 的 warmup 和正式批次。直接启动二进制时，请传入满足该约定的 `--ub-size` 和合法 `--batch`。

### 使用示例

```bash
# 默认兼容模式：单向 PUT 带宽测试
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# 默认兼容模式：延迟测试，一次遍历普通/QP/聚合接口（后端不支持的形态显示 -）
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --metric lat

# 批量完成：每 100 次操作后等待完成，再开始下一组
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 100

# 多 QP 带宽测试：2 个 QP 并行；IN_DIE/XSCALE 小消息按批聚合，HNS_1825 逐次 NBI 提交
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 -q 2 -i -1 --metric bw --batch 100

# 单 QP 诊断模式：仅测试编号为 0 的 QP，每 100 次操作后等待完成
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 -q 4 -i 0 --metric bw --batch 100

# 双向 GET
./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000
```

## CSV 输出

CSV 文件输出到 `output/` 目录，命名格式为 `rdma_<metric>_<test_type>_<dtype>_<pe>.csv`。两种指标共用前 4 列，后续列不同：

| 列 | 指标 | 说明 |
|----|------|------|
| `DataSize/B` | 通用 | 单个 QP 每次传输的数据量（字节）；多 QP 并行模式总数据量为 `DataSize × qp-count` |
| `Npus` | 通用 | NPU 数量 |
| `Blocks` | 通用 | 多 QP 并行模式为 `qp-count`，其他模式为 1 |
| `UBsize/KB` | 通用 | UB 大小（KB） |
| `Bandwidth/GB/s` | `bw` | 带宽（GB/s，十进制） |
| `Bandwidth/GiB/s` | `bw` | 带宽（GiB/s，二进制） |
| `CoreMaxTime/us` | `bw` | 单次 iteration 耗时（us） |
| `PlainLat/us` | `lat` | 普通非聚合接口的下发时延（us），后端不支持时为 `-` |
| `QpLat/us` | `lat` | QP 指定接口的下发时延（us），后端不支持时为 `-` |
| `DeferLat/us` | `lat` | 聚合 defer 接口摊薄到单次调用的下发时延（us），后端不支持时为 `-` |
| `SubmitLat/us` | `lat` | 聚合 submit 接口摊薄到单次调用的下发时延（us），后端不支持时为 `-` |

## 已知约束

1. **并发限制**：RDMA 规范不允许同一 PE 的并发 RMA/AMO 操作。多 QP 并行模式使用独立 QP 规避此限制；单 QP 诊断模式和时延测试仅使用单 QP。
2. **编译依赖**：需启用 `-enable_rdma`；Ascend950 平台需额外指定 `-soc_type Ascend950` 及 `-rdma_backend XSCALE`（或 `HNS_1825`）。
3. **不支持 D2H**：RDMA 引擎仅测 HBM，不支持 Host 侧 DRAM。
4. **不支持原子操作**。
5. **不支持 put_signal**：RDMA 引擎无 `aclshmemx_roce_put_signal_nbi` 接口。
