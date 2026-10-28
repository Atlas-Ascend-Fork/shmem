## 示例概述

本样例用于测试 SIMT UDMA QP-specific 接口的带宽性能，平行于同目录下的 `udma_perftest`（针对 SIMD/AICore 低阶 UDMA 接口）。样例通过 [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) 宏采集性能数据，覆盖 `simt::aclshmemx_udma_qp_put_nbi_{warp|block}` / `simt::aclshmemx_udma_qp_get_nbi_{warp|block}` 在 warp/block 两种执行域下的带宽表现。**该脚本测试结果仅做参考，性能以实际场景为准**。

## 测试目的

针对以下 SIMT UDMA QP-specific 数据传输操作的性能：

1. **Warp Put** (`warp_put`)：Active PE 在 warp 执行域下调用 `simt::aclshmemx_udma_qp_put_nbi_warp`，每个 warp 协作驱动一个 QP。
2. **Warp Get** (`warp_get`)：Active PE 在 warp 执行域下调用 `simt::aclshmemx_udma_qp_get_nbi_warp`，每个 warp 协作驱动一个 QP。
3. **Block Put** (`block_put`)：Active PE 在 block 执行域下调用 `simt::aclshmemx_udma_qp_put_nbi_block`，每个 kernel block 驱动一个 QP。
4. **Block Get** (`block_get`)：Active PE 在 block 执行域下调用 `simt::aclshmemx_udma_qp_get_nbi_block`，每个 kernel block 驱动一个 QP。
5. **双向 Put** (`bi_put`)：两个 PE 同时在对应执行域下调用 put，互相传输数据。
6. **双向 Get** (`bi_get`)：两个 PE 同时在对应执行域下调用 get，互相拉取数据。

前 4 种为单向模式（仅 Active PE 发起传输），后 2 种为双向模式（两个 PE 同时发起传输）。

## 与 `udma_perftest` (SIMD 版) 的差异

| 维度 | `udma_perftest` (SIMD/AICore) | `simt_udma_perftest` (SIMT) |
|------|------|------|
| 编程模型 | SIMD/AICore（`__global__ __vector__` + AIV block） | SIMT（`asc_vf_call` 调度 `__simt_vf__`） |
| 接口 | `aclshmemx_udma_qp_put_nbi` / `aclshmemx_udma_qp_get_nbi`（需 UB scratch 入参） | `simt::aclshmemx_udma_qp_put_nbi_{warp\|block}` / `simt::aclshmemx_udma_qp_get_nbi_{warp\|block}`（无 UB 入参） |
| 执行域 | 单一（block = AIV） | 两种（warp / block），运行时 `--scope` 选择 |
| QP 驱动 | block i 驱动 QP i（`GetBlockIdx()`） | warp/block i 驱动 QP i（取决于执行域） |
| 组合选择 | `-t` 选择测试类型 | `-t`（操作类型）+ `--scope`（执行域）；`all` 在同一进程内依次展开 8 种组合 |
| UB 缓冲 | 必需（WQE staging） | 不使用（SIMT 接口内部管理 DWQE 窗口） |
| 测试模式 | put / bi_put / get / bi_get / put_signal | put / get / bi_put / bi_get（× warp/block = 8 组合） |
| 数据类型 | `-d` 可选 10 种，按位宽参与统计 | 固定 `uint8`（T = `uint8_t`，按字节搬运） |
| Metric | bw / lat（lat：单相窗口，仅测 put 下发耗时） | bw / lat（lat：双相位基线相减，单次净时延） |
| SOC 限制 | 仅 Ascend950 | 仅 Ascend950 |

## 版本和平台支持说明

- 本样例仅支持 Ascend950 平台（direct 模式），其他平台不支持运行该样例。
- CANN 9.2.0 及以上版本已提供 UDMA 所需的 HCOMM 资源接口。请从 [CANN 9.2.0 资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.2.0)下载并安装 Ascend950 对应的 toolkit 包和 ops 包；低于 9.2.0 的 CANN 版本不在本样例支持范围内。
- 编译前需执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并使用 `-enable_simt -soc_type Ascend950` 编译。

## 编译说明

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
```

产物：`build/bin/simt_udma_perftest`（单二进制）。

每种执行域 × 操作类型的组合在编译期以模板实例化生成（各自的 VF 形态、`__launch_bounds__` 与 QP 驱动方式不同），运行时经 `--scope` / `-t` 选择对应组合；`all` 会在同一进程内依次运行全部组合。产物与同目录其他性能用例一样为单个二进制。

## 测试模型

固定使用两个 PE：

| 角色 | PE 编号 | 职责 |
|------|---------|------|
| Active PE | 0 | 单向模式下发起 put/get 传输，采集 prof 数据 |
| Passive PE | 1 | 单向模式下作为远端通信对象；双向模式下两端 PE 同时发起传输 |

数据流向与校验方（校验由数据接收方执行）：

| 操作 | 数据流向 | 校验方 | 校验内容 |
|------|---------|--------|---------|
| put | Active PE 对称内存 → Passive PE 对称内存 | Passive PE | 对称内存应为 Active PE 的数据值 |
| get | Passive PE 对称内存 → Active PE 对称内存 | Active PE | 对称内存应为 Passive PE 的数据值 |
| bi_put | 两端 PE 对称内存 → 对端 PE 对称内存 | 两端 PE | 各端对称内存应为对端 PE 的数据值 |
| bi_get | 两端 PE 对称内存 ← 对端 PE 对称内存 | 两端 PE | 各端对称内存应为对端 PE 的数据值 |

## 性能测试方法

### 提交与排空

transfer VF 内部循环 `loop_count` 次，每次各 QP 发起一次 put/get（不调用 quiet），随后 quiet VF 一次性排空所有 QP 的未完成 WQE。

### Metric 口径说明

`--metric` 控制 prof 窗口的覆盖范围，两种口径的打点都在循环外（`SHMEMI_PROF_START/END` 自带 `pipe_barrier`，逐次打点会把测量开销累加到结果上，因此整窗打点后除以 `loop_count`）：

- **`bw`（默认）**：`prof_start → loop(put/get_nbi) → quiet → SyncAll → prof_end`，窗口包含 quiet 排空，反映端到端吞吐：

  ```
  prof_start → asc_vf_call(transfer_vf: loop_count × put/get_nbi)
             → asc_vf_call(quiet_vf: per-QP quiet)
             → SyncAll → prof_end
  ```

- **`lat`**：窗口只覆盖提交循环本身，quiet 移到窗口外；随后再执行一次同形空循环（volatile sink 防止循环被编译器优化消除，无接口调用）作为基线窗口，两者相减得到单次调用的净时延：

  ```
  prof_start → asc_vf_call(transfer_vf: loop_count × put/get_nbi) → prof_end
             → asc_vf_call(quiet_vf)                 # 窗口外排空
  prof_start → asc_vf_call(baseline_vf: 同形空循环) → prof_end
  ```

  `net_per_call = (real_window − baseline_window) / loop_count`，减去基线后即为纯接口下发耗时，与 `udma_perftest` 的 `lat`（未减基线的单相下发耗时）口径略有差异。

### 缓冲区设计与分片

源和目的使用独立的对称缓冲区（`src` 和 `dst`）：put 从本地 `src` 读取并写入远端 `dst`；get 从远端 `src` 读取并写入本地 `dst`。使用独立缓冲区可避免双向模式下两端同时传输时的读写竞争。各 QP 按 `qp_idx * slice` 分占独立分片，互不重叠。

### 预热

计时前执行 `warmup` 次预热（不在 prof 窗口内），预热后重新填充数据再执行计时轮，排除冷启动影响。

### 带宽计算

`Bandwidth = slice × qp_count × loop_count / prof_time`，其中 `prof_time` 取各 block 窗口耗时的最大值（最慢 block 决定整体带宽），仅 `bw` 口径计算带宽；`lat` 口径下带宽列为 0，时间列为单次净时延。双向模式下统计的是当前 PE 的发出流量。

### loop_count / warmup 上限

单 QP 最大未完成调用数为 16378。`loop_count` 和 `warmup` 必须小于此值，否则 VF 内连续提交会触发接口内部回收，污染计时窗口。

## 使用方法

### 基本用法

```bash
cd examples/shmem_perftest/simt_udma_perftest/
bash run.sh [选项]
```

### 命令行参数

命令行参数风格与 `udma_perftest`、`simt_rma_perftest` 等子示例保持一致（长选项统一使用连字符形式），`bash run.sh --help` 可查看脚本内置帮助。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scope <type>` | `all` | 执行域，支持 `warp`、`block`、`all`（两域依次运行） |
| `-t\|--test-type <type>` | `all` | 操作类型，支持 `put`、`get`、`bi_put`、`bi_get`、`all`（四种依次运行） |
| `--qp <N>` | `4` | 每个对端 PE 创建的 QP 数，范围为 `[1, 56]`，所有 PE 必须一致 |
| `-e\|--exponent <E>` | - | 单个 QP 的数据量幂数（2^E 字节），范围为 [1, 28]（E=0 时仅 1 字节，低于 UDMA 引擎 2 字节下限；2^28 = 256 MiB 为接口单次请求上限）；设置后 min=max=E |
| `--exponent-range <min> <max>` | `3 17` | 单个 QP 的数据量幂数范围，min >= 1（UDMA 引擎要求每 QP 传输不低于 2 字节），max <= 28（2^28 = 256 MiB，SIMT UDMA 单次请求上限），每个幂数各产出一行统计 |
| `--loop-count <N>` | `1000` | 计时轮循环次数，范围为 `[1, 16378)` |
| `--warmup <N>` | `100` | 预热轮循环次数，范围为 `[0, 16378)` |
| `--metric <bw\|lat>` | `bw` | 性能口径：`bw` 测端到端带宽；`lat` 测单次调用净时延（双相位基线相减） |
| `--heap-mb <N>` | `1024` | 每个 PE 的 SHMEM 对称堆大小，单位为 MiB；需覆盖 `2 × qp_count × 2^max_exponent` 字节（sym_src 与 sym_dst 各占一段），另需容纳库内部开销 |
| `-ipport <tcp://IP:PORT>` | `tcp://127.0.0.1:8999` | 引导初始化地址 |
| `-pes <N>` | `2` | PE 总数，固定为 2 |
| `-gnpus <N>` | `2` | 启动进程数 / NPU 数量，固定为 2 |
| `-fnpu <ID>` | `0` | 首个 NPU ID，实际 device id = `pe % gnpus + fnpu` |
| `-fpe <ID>` | `0` | 首个 PE ID，仅为与 shmem_perftest 其他子示例保持参数一致，当前未使用 |

### 使用示例

```bash
# 默认：全部 8 种组合，4 QP，数据量 2^3~2^17 字节，1000 次循环
bash run.sh

# 仅测试 warp 模式 put，8 QP
bash run.sh --scope warp -t put --qp 8

# 指定数据量范围和循环次数
bash run.sh --exponent-range 10 20 --loop-count 500

# 仅测试 warp 模式 get，单 QP
bash run.sh --scope warp -t get --qp 1 -e 16

# block 模式 put，16 QP，2MB 数据
bash run.sh --scope block -t put --qp 16 -e 21 --heap-mb 4096

# warp 模式双向 put
bash run.sh --scope warp -t bi_put --qp 8

# warp 模式单次调用净时延（双相位）
bash run.sh --scope warp -t put --qp 8 --metric lat

# 全部 8 种组合测净时延
bash run.sh --metric lat --exponent-range 10 14
```

## 输出示例

```text
[INFO] pe=0 pes=2 qp_count=4 exp=3-17 loop=1000 warmup=100 metric=bw scopes=2 ops=4
[INFO] scope=warp op=put pe=0 qp_count=4
pe: 0 size(per QP): 8B, total: 32B frame_id: 0
[Verification] SUCCESS
pe: 0 size(per QP): 16B, total: 64B frame_id: 1
[Verification] SUCCESS
...
[CSV] output/simt_udma_bw_put_warp_qp4_3-17_l1000.csv
[SUCCESS] simt_udma_perftest done in pe 0
```

`--metric lat` 时每个数据点额外输出净时延结果：

```text
[lat] pe=0 slice=1024B net=0.087 us/call (gross=0.112, baseline=0.025)
```

## CSV 输出

CSV 文件按每种执行域/操作组合各输出一份，文件名格式为（相对 `simt_udma_perftest/` 目录）：

```
output/simt_udma_<metric>_<op>_<scope>_qp<qp_count>_<min_exp>-<max_exp>_l<loop_count>.csv
```

CSV 列与 `udma_perftest` 保持一致，便于复用 `examples/utils/perf_data_process.py` 绘制性能图表。各列含义如下：

| 列名 | 说明 |
|------|------|
| `DataSize/B` | 单个 QP 每次传输的数据量（字节）；多 QP 时总数据量为 `DataSize × QPs` |
| `Npus` | 参与测试的 NPU 数量（`--gnpus`） |
| `QPs` | 每个对端 PE 的 QP 数量（`--qp`） |
| `UBsize/KB` | 固定为 0。SIMT UDMA 接口不使用 Unified Buffer，该列仅为保持与 `udma_perftest` 列格式一致 |
| `Bandwidth/GB/s(1000)` | 带宽（按 1000 进制换算）。`bw` 口径下按所有 QP 的总流量计算；`lat` 口径下填 0 |
| `Bandwidth/GiB/s(1024)` | 带宽（按 1024 进制换算）。同上 |
| `CoreMaxTime/us` | `bw` 口径下为最慢 block 的单轮耗时（窗口总耗时 / loop_count）；`lat` 口径下为单次净时延（取各 block 最大值） |
| `SingleCoreTime_B{n}/us` | 第 `n` 个 block（0 起）的单轮耗时（`bw`）或单次净时延（`lat`）。列数随 QP 数量动态生成；该 block 无 prof 数据时填 `N/A`。warp 口径下仅有 `B0` 一列 |

## 已知约束

1. 本样例仅测试 HBM (DEVICE_SIDE) 内存路径，不支持 D2H / `HOST_SIDE` (DRAM)。
2. UDMA 仅在 Ascend950 编译期使能；在其他 SOC 上 kernel 会报错退出。
3. `loop_count` 和 `warmup` 必须小于 16378（单 QP 最大未完成调用数），否则 VF 内连续提交会触发接口内部回收，污染计时窗口。
4. 执行域与操作类型在编译期以模板实例化实现，`--scope` / `-t` 选择具体组合；`all` 展开的多种组合在同一进程内依次执行，共享 1024 个 prof 帧，总数据点数超限时会打印告警并停止剩余组合。
5. 本样例不测 PutSignal，这是 `udma_perftest` 的能力。
6. 幂数下限为 1（2^1 = 2 字节），对应接口单次传输不低于 2 字节的下限要求。
7. 幂数上限为 28（2^28 = 256 MiB），对应接口单次传输 256 MB 的上限要求。超过该上限的切分会超出接口单次请求能力（设备侧范围检查拒绝），且 2^32 及以上在主机侧窄化为 `uint32_t` 时溢出为 0，故超出时直接判定参数非法。
