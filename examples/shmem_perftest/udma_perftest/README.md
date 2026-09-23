# udma_perftest

## 示例概述

`udma_perftest` 是用于**测试 shmem UDMA 低阶接口性能**的参数化测试示例，平行于同目录下的 `mte_perftest`（针对 MTE 引擎）。该示例通过 [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) 宏采集性能数据，覆盖 `aclshmemx_udma_put_nbi` / `aclshmemx_udma_get_nbi` / `aclshmemx_udma_put_signal_nbi` 三个低阶接口，并支持不同数据量、batch 提交和 QP 配置。**该脚本测试结果仅做参考，性能以实际场景为准**。

## 测试目的

针对以下 UDMA 数据传输操作的性能：

1. **单向 Put** (`put`)：仅 `SHMEM_CYCLE_PROF_PE` 指定的 PE 调用 `aclshmemx_udma_put_nbi`，将数据传输到对端 PE。
2. **双向 Put** (`bi_put`)：两个 PE 同时调用 put，互相传输数据。
3. **单向 Get** (`get`)：仅 prof PE 调用 `aclshmemx_udma_get_nbi`，从对端 PE 拉取数据。
4. **双向 Get** (`bi_get`)：两个 PE 同时调用 get，互相拉取数据。
5. **Put + Signal** (`put_signal`)：仅 prof PE 调用 `aclshmemx_udma_put_signal_nbi`，传输数据后写一个远端信号；测试结束做信号值校验。

## 与 `mte_perftest` (MTE 版) 的差异

| 维度 | `mte_perftest` (MTE) | `udma_perftest` (UDMA) |
|------|----------------------|----------------------|
| 引擎 | 默认 MTE | 显式 `ACLSHMEM_DATA_OP_UDMA` |
| 多核并发 | 同 peer 多核 (默认 32 核切分数据) | `block_dim=qp_count`，每个 block 独占一个 QP 和一份独立数据 |
| `-b/--block-size`、`--block-range` | 控制核数 | `-b` 同时控制 QP 数；`--block-range` 当前不参与 UDMA 测试 |
| UB 缓冲 | MTE 必需，影响传输 | 本 perftest 显式测试 UDMA 低阶接口，`--ub-size` 仅用于低阶接口 UB 入参和 CSV；高阶 UDMA RMA 默认 MTE staging 需要至少 128 B UB |
| 测试模式 | put / bi_put / get / bi_get | put / bi_put / get / bi_get / **put_signal** |
| SOC 限制 | 通用 | **仅 Ascend950**：非 950 上 device kernel 内置 abort |
| CSV 文件名 | `<test>_<dtype>_<pe>.csv` | `udma_<metric>_<test>_<dtype>[_qpN]_<pe>.csv` |

## 编译说明

UDMA 仅在 Ascend950 上可用，需使用已提供 HCOMM 资源接口的 CANN 9.1.0 及以上版本。安装要求和 CANN 资源入口参见 [UDMA 样例版本和平台支持说明](../../udma_demo/README.md#版本和平台支持说明)。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -soc_type Ascend950
```

## 使用方法

### 基本用法

```bash
cd examples/shmem_perftest/udma_perftest/
./run.sh [选项]
```

### 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | 测试类型 (put / bi_put / get / bi_get / put_signal / all) | `put` |
| `--datatype <type>` | `-d <type>` | 数据类型 (float / int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64 / char / all) | `float` |
| `--block-size <size>` | `-b <size>` | UDMA 使用的 block 数，同时作为 QP 数（一个 block 对应一个 QP），范围 `[1, 56]` | 1 |
| `--block-range <min> <max>` | - | UDMA 兼容解析但不参与测试；使用 `-b` 指定 block/QP 数 | 1 1 |
| `--exponent <exponent>` | `-e <exponent>` | 单个 QP 的数据量幂数 (2^exponent 字节)；多 QP 时总数据量为该值乘 QP 数 | - |
| `--exponent-range <min> <max>` | - | 单个 QP 的数据量幂数范围 | 3 17 |
| `--loop-count <count>` | - | 循环次数 | 1000 |
| `--ub-size <size>` | - | UB size (KB)；本 perftest 传给低阶 UDMA 接口并写入 CSV。高阶 UDMA RMA 默认 MTE staging 的 UB 配置要求至少 128 B | 16 |
| `--metric <bw\|lat>` | - | 性能口径：`bw` 测带宽，`lat` 测单次 put_nbi 下发延时（仅 `-t put` 支持） | bw |
| `--batch <N>` | - | BW 测试批大小：`0` 或 `1` 使用普通 NBI；`N > 1` 时每批前 `N-1` 个操作使用 `defer`，最后一个使用 `submit`，随后调用 `quiet`。每个 WQE 在 UB 中占用 64 B，要求有效 batch 小于 4096 且 `N * 64 <= --ub-size * 1024`；`lat` 模式忽略此参数。 | 0 |
| `-pes <size>` | - | PE 数量 | 2 |
| `-ipport <ip:port>` | - | 通信地址 | tcp://127.0.0.1:8768 |
| `-gnpus <num>` | - | NPU 数量 | 2 |
| `-fnpu <id>` | - | 首个 NPU ID | 0 |
| `-fpe <id>` | - | 首个 PE ID | 0 |
| `-a/--analyse <mode>` | - | 分析模式 (none / plot / md) | none |

### DRAM / D2H 内存约束

本示例仅测试 HBM (DEVICE_SIDE) 内存路径，**不支持 D2H / `HOST_SIDE` (DRAM)**：UDMA 引擎当前未对 Host 侧 DRAM 提供 RMA 路径，相关测试不在本示例的覆盖范围。如需测 DRAM，请改用 [rma_d2h_demo](../../rma_d2h_demo/README.md)。

默认 1 GB 本地内存；当数据量较大时，程序会自动上调 `local_mem_size`（最多 40 GB）。

### Metric 口径说明

UDMA 是异步 NBI 接口，benchmark 把 `put_nbi/get_nbi` 提交与 `quiet` 等待分开计时。两种 metric 都是**单窗口覆盖 `loop_count` 次提交**（`SHMEMI_PROF_START/END` 自带 `pipe_barrier` 开销，每次循环打点会把测量本身的开销叠到延时数字上，所以打点放循环外、对总耗时除以 `loop_count`），区别只在 `quiet` 是否在窗口内：

- **`--metric bw`（默认）**：`prof_start → loop(put_nbi) → quiet → prof_end`，窗口包含 `quiet`。
  - `Bandwidth/GB/s` = `datasize × qp_count / (window_us / loop_count)`；`datasize` 是单个 QP 的数据量
  - `CoreMaxTime/SingleCoreTime` 列填 `window_us / loop_count`
  - 适用于 `put / bi_put / get / bi_get / put_signal`
- **`--metric lat`**：`prof_start → loop(put_nbi) → prof_end → quiet`，`quiet` 移出窗口，只测下发本身。
  - `CoreMaxTime/SingleCoreTime` 列填 `window_us / loop_count`，即单次 `put_nbi` 平均下发耗时
  - `Bandwidth/GB/s` 填 0，对延时口径无意义
  - **仅 `-t put` 支持**；其他 `-t` 与 `--metric lat` 组合会直接报错退出

CSV 文件名加 metric 前缀；多 QP 场景追加 QP 数后缀：`output/udma_<metric>_<test_type>_<dtype>[_qpN]_<pe>.csv`。

### Batch 提交（仅 `--metric bw`）

`--metric bw` 路径默认是**全异步**：`loop_count` 次普通 `*_nbi` 提交完后，仅在窗口末尾 `quiet` 一次。`--batch <N>` 同时决定 BW 路径的批边界和提交方式：

- `--batch 0`（默认）：普通 NBI 全异步，仅末尾一次 `quiet`，反映稳态吞吐。
- `--batch 1`：每次普通 NBI 后立刻 `quiet`，等价于同步提交，能反映"提交+完成"的端到端开销。
- `--batch N`（`N > 1`）：每批前 N-1 个操作使用 `defer`，最后一个使用 `submit`，然后 `quiet`；当 `loop_count` 不能整除 N 时，余数批次按实际操作数提交。

`SHMEMI_PROF_START/END` 仍只取一次（同 `--metric bw` 原本的实现），即窗口内总时间除以 `loop_count` 给出"含 batched quiet 的平均单次耗时"。`--metric lat` 路径不受 `--batch` 影响。

适用范围：`put / bi_put / get / bi_get` 在 `--metric bw` 下的 `batch > 1` 使用聚合 action；`put_signal` 始终使用普通 NBI，但支持按 batch 分组后调用 `quiet`。

### 使用示例

```bash
# 单向 PUT 带宽，float，幂数 8-20
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# 单向 PUT 单次下发延时
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --metric lat

# 双向 GET 带宽，int32
./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000

# put_signal 带宽
./run.sh -t put_signal -d float -e 14 --loop-count 500

# 五种模式 × float 带宽
./run.sh -t all -d float --exponent-range 8 20 --loop-count 1000

# 单一模式 × 全部数据类型
./run.sh -t put -d all --exponent-range 8 20

# 同步提交 (batch=1)：每次 nbi 后 quiet
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 1

# 半异步：每 16 次 nbi 后 quiet
./run.sh -t get -d float --exponent-range 8 20 --loop-count 1000 --batch 16

# 单 QP 聚合提交（batch=8）
./run.sh -t put -d float -e 14 --batch 8

# 4 QP 并发（核数与 QP 数相等）
./run.sh -t bi_put -d float -e 16 -b 4

# 4 QP + 聚合提交（每个 QP 的 batch=8）
./run.sh -t bi_get -d float -e 16 -b 4 --batch 8
```

## put_signal 行为说明

`put_signal` 模式由 perftest 自动管理信号：

- 测试启动时分配一段对称信号缓冲 `aclshmem_malloc(n_pes * sizeof(uint64_t))`，初始化为 0。
- 每个数据点循环 `warmup + loop_count` 次，每次调用 `aclshmemx_udma_put_signal_nbi(..., signal_base + i, peer_pe)`，信号值线性递增以避开脏数据干扰。
- 数据点结束后，host 端读回对端信号槽，校验是否等于 `signal_base + (warmup + loop_count - 1)`。
- 校验失败会打印 ERROR 但不终止后续数据点。

## CSV 输出

CSV 列与 MTE 版保持一致，便于复用 `examples/utils/perf_data_process.py` 出图：

```bash
DataSize/B, Npus, Blocks, UBsize/KB, Bandwidth/GB/s, CoreMaxTime/us, SingleCoreTime/us
```

`put/bi_put/get/bi_get` 的 `Blocks` 列等于 `--block-size`；`put_signal` 保持单 block。**`DataSize/B` 表示单个 QP 每次传输的数据量，也就是 `-e` 指定的 `2^e` 字节；多 QP 时总数据量为 `DataSize × Blocks`**。因此带宽统计会按所有 QP 的总流量计算，可直接与单 QP 结果对比。与 MTE 版的列结构一致：`Bandwidth = DataSize × Blocks / 时间` 表示总流量。跨引擎按同一 `-e` 对比时，应结合 `Blocks` 判断总流量。文件名格式为 `output/udma_<metric>_<test_type>_<dtype>[_qpN]_<pe>.csv`，例如 `udma_bw_put_float_0.csv` 或 `udma_bw_put_float_qp4_0.csv`。`--metric lat` 时 `Bandwidth/GB/s` 列填 0，`CoreMaxTime/SingleCoreTime` 列填测量窗口内单次操作的平均下发耗时。

## 输出示例

```bash
[INFO] udma_perftest start, pe=0, t=put, d=float, exp=10-10, loop=100, ub=16KB, metric=bw, batch=100
pe: 0 size(per QP): 1024B, total: 1024B frame_id: 0
[Verification] put: checking...
[Verification] SUCCESS
[SUCCESS] udma_perftest done in pe 0
```

## 已知约束

1. UDMA 头文件 `include/device/gm2gm/engine/shmem_device_udma.h` 注明：concurrent RMA/AMO operations to the same PE are not supported。单 QP 模式保持 `block_dim=1`；多 QP 模式由每个 block 独占一个显式 QP，避免多个 block 争用同一个 SQ。
2. UDMA 聚合 action 要求 `PIPE_MTE3`。当 `batch > 1` 时，每批 WQE 数必须小于 4096（UB 上限 256 KB，每 WQE 占 64 B），UB scratch 至少需要 `64 * batch` 字节；多 QP 依赖的 QP 专有接口要求 `ACLSHMEM_RELAY_SUPPORT=OFF`，仅直连 UDMA build 支持。
3. UDMA 仅在 Ascend950 编译期使能；在其他 SOC 上 kernel 会通过 `aclshmemi_kernel_abort` 报错退出。
4. **不支持 D2H / `HOST_SIDE` (DRAM)**：UDMA 引擎当前未对 Host 侧 DRAM 提供 RMA 路径，仅测 HBM。
5. 原子操作 (`aclshmemx_udma_atomic_add` 等) 不在本 perftest 范围。
6. 高阶 UDMA RMA 接口默认通过 `PIPE_MTE3` staging 下发 WQE，默认 UB 配置为 `offset = 189 * 1024`、`ub_size = 128` 字节、`sync_id = 0`。如果调用 `aclshmemx_set_udma_config` 修改配置，`ub_size` 必须不小于 128 字节。本 perftest 的低阶接口路径仍按显式入参使用本地 UB。
