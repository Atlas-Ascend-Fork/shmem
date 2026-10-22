# rdma_atomic_perftest

## 概述

`rdma_atomic_perftest` 用于测试 SHMEM RDMA 原子操作接口（`aclshmemx_roce_atomic_*`）的性能，支持批量完成时间（`batch`）和单次完成时延（`latency`）两种测试模式。

测试固定使用两个 PE，每个 PE 对应一个进程和一张 NPU。PE0 发起操作，PE1 提供对称 HBM 目标内存；Device 侧使用一个 AIV core 和默认的单个 RDMA QP，所有原子操作通过 QP0 完成。

## 测试模式

| 模式 | 数据类型 | 每个正式样本的执行内容 | 指标 |
| --- | --- | --- | --- |
| `batch` | `uint64` | 单线程依次对连续的 N 个元素各执行一次 atomic，等待整批完成 | 整批完成时间（us/batch）、原子操作速率（Mops/s） |
| `latency` | `uint32`、`uint64`、`int32`、`int64` | 对一个标量重复执行 atomic，每次等完成后再执行下一次 | 平均完成时延（us/op） |

Batch size 表示一批中的 atomic 调用次数，默认依次测试 `1、2、4、8、16、32、64、128、256、512、1024、2048、4096`，共 13 个大小。每个大小独立采样 10 次，**每个正式样本只执行一批**。例如 batch size 为 4 时，一个样本包含对 `target[0]` 到 `target[3]` 各一次操作，完成后立即结束计时。

Latency 的每个正式样本默认执行 1000 次操作，以总耗时除以操作次数得到平均完成时延，同样独立采样 10 次。`--iterations` 仅控制 latency 的操作次数，不影响 batch 的正式采样。

### 支持的原子操作

以下 13 个操作均支持 batch 和 latency 测试。带 `fetch` 的操作以及 `swap`、`compare_swap` 返回操作前的旧值，其余操作不返回旧值。

表中 `K` 表示一个正式样本内每个目标元素被操作的次数：batch 固定为 1，latency 为 `--iterations`。`ALL_ONES` 为所选类型位宽的全 1。每次采样均使用表中的初始值和参数，并校验目标最终值与最后一次返回的旧值。

| `--op` | 初始值 | 操作参数 | 目标最终值 | 最后一次返回旧值 | 底层操作族 |
| --- | --- | --- | --- | --- | --- |
| `inc` | 0 | 加 1 | K | — | FA |
| `fetch_inc` | 0 | 加 1 | K | K−1 | FA |
| `set` | 0 | value=1 | 1 | — | CAS |
| `add` | 0 | value=1 | K | — | FA |
| `fetch_add` | 0 | value=1 | K | K−1 | FA |
| `and` | ALL_ONES | value=ALL_ONES−1 | ALL_ONES−1 | — | CAS |
| `fetch_and` | ALL_ONES | value=ALL_ONES−1 | ALL_ONES−1 | K=1 时为 ALL_ONES，否则为 ALL_ONES−1 | CAS |
| `or` | 0 | value=1 | 1 | — | CAS |
| `fetch_or` | 0 | value=1 | 1 | K=1 时为 0，否则为 1 | CAS |
| `xor` | 0 | value=1 | K%2 | — | FA |
| `fetch_xor` | 0 | value=1 | K%2 | (K−1)%2 | FA |
| `swap` | 0 | value=1 | 1 | K=1 时为 0，否则为 1 | CAS |
| `compare_swap` | 0 | compare=0, value=1 | 1 | K=1 时为 0，否则为 1 | CAS |

FA 和 CAS 分别对应 RDMA 后端的 `OP_ATOMIC_FA`、`OP_ATOMIC_CAS` 操作族，具体语义由操作数和掩码共同决定。Batch 的 `compare_swap` 对每个元素只操作一次，比较均成功；latency 对同一元素首次比较成功，后续比较失败。

## 环境要求

本示例的原子操作测试当前在 **Ascend950 + XSCALE（云脉网卡）或 HNS_1825（1825 网卡）** 配置下启用。其他后端可以查看 `--help`，执行测试时会报告用例未启用并返回非零状态。

RDMA 网络、驱动和运行环境配置参见 [rdma_demo 环境要求](../../rdma_demo/README.md)。Ascend950 平台运行前需设置 `IBV_EXTEND_DRIVERS`，指向对应网卡的插件库，详见 [环境变量说明](../../rdma_demo/README.md#ibv_extend_drivers-环境变量)。

## 编译说明

在 SHMEM 仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 云脉网卡
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE
# 1825 网卡
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend HNS_1825
```

构建生成 `build/bin/rdma_atomic_perftest`。后端在编译时通过 `-rdma_backend` 选择；更改后端配置后需要重新编译。其他构建选项参见 [编译与构建](../../../docs/compilation_build_guide.md#rdma参数使用说明)。

## 使用方法

```bash
cd examples/shmem_perftest/rdma_atomic_perftest/
bash run.sh [选项]
```

未指定 `--pe-id` 时，脚本启动 PE0 和 PE1，向两者传递相同的测试参数，并等待两个进程退出。任一 PE 返回非零状态时，该配置判为失败。

### 命令行参数

| 参数 | 描述 | 默认值 |
| --- | --- | --- |
| `--test <batch\|latency>` | 测试模式 | `latency` |
| `--op <op>` | 原子操作，取值见“支持的原子操作” | `inc` |
| `--data-type <type>` | 数据类型；batch 仅支持 `uint64`，latency 支持四种整数类型 | `uint64` |
| `--iterations <n>` | 仅 latency：每个正式样本的操作次数，范围 1~INT_MAX | `1000` |
| `--warmup-iterations <n>` | Batch 的预热批数，或 latency 的预热操作次数；范围 0~INT_MAX，0 表示不预热 | `100` |
| `--repetitions <n>` | 每个测试点的正式采样次数，范围 1~INT_MAX | `10` |
| `--min-batch <n>` | 最小 batch size，正整数 | `1` |
| `--max-batch <n>` | 最大 batch size，不小于 min-batch；所需对称堆须在 SHMEM 容量限制内 | `4096` |
| `--step-factor <n>` | Batch size 递增倍率，整数且大于 1 | `2` |
| `--pes <n>` | PE 数量，固定为 2 | `2` |
| `--pe-id <id>` | 当前进程的 PE 编号，取值 0 或 1；脚本指定此参数时只启动对应 PE | 二进制默认 `0` |
| `--ipport <tcp://ip:port>` | SHMEM 初始化地址 | `tcp://127.0.0.1:8768` |
| `--gnpus <n>` | 本地启动域 NPU 数，正整数 | `2` |
| `--fnpu <id>` | 起始本地 Device ID，非负整数 | `0` |
| `--sync-id <id>` | RoCE 同步事件 ID，须使用目标平台支持的非负 ID | `0` |
| `--all` | 运行全部 65 个测试配置，仅用于 `run.sh` | 关闭 |
| `-h` / `--help` | 显示帮助信息 | — |

实际 Device ID 为 `pe-id % gnpus + fnpu`。Batch size 依次取 `min-batch × step-factor^k`，直到下一项超过 `max-batch`。设置相同的 min-batch 和 max-batch 可只测一个大小。Latency 固定使用一个标量，batch 范围不参与测试。非法参数在设备初始化前被拒绝。

### 使用示例

```bash
# uint64 add：batch size 从 1 到 4096，每个大小采样 10 次
bash run.sh --test batch --op add

# uint64 fetch_add：每个样本执行一批 4 个操作，共采样 10 次
bash run.sh --test batch --op fetch_add --min-batch 4 --max-batch 4 --repetitions 10

# uint64 fetch_add 单次完成时延，每个样本执行 1000 次操作
bash run.sh --test latency --op fetch_add --data-type uint64 --iterations 1000

# 完整测试矩阵
bash run.sh --all --warmup-iterations 100 --repetitions 10
```

完整矩阵包含 13 个 batch 配置（13 个操作 × uint64）和 52 个 latency 配置（13 个操作 × 4 种类型）。每个 batch 配置由二进制程序完成大小扫描。`--all` 可附加采样和设备参数，其中 `--iterations` 只作用于 latency 配置；不能同时指定 `--test`、`--op` 或 `--data-type`。任一配置失败后停止后续配置。

跨机运行单个配置或 `--all` 时，在两台主机分别指定 `--pe-id 0` 和 `--pe-id 1`，使用 `--gnpus 1` 和相同的 `--ipport tcp://<PE0主机IP>:8768`，通过 `--fnpu` 选择各自的本地设备。两端脚本版本与测试参数须一致，并分别检查进程退出状态；任一端失败后，另一端可能等待通信超时。

## 计时与正确性校验

每个测试点先预热，再按 `--repetitions` 独立采样。预热阶段重复执行完整批次或 latency 操作，每批完成后再开始下一批，结果不计入正式样本。预热和每次正式采样前均重新初始化目标值及结果缓冲区。

| 模式与操作 | 一个正式样本的计时区间 |
| --- | --- |
| Batch，非 fetch | 开始计时 → 依次调用 N 次 atomic → 一次 `aclshmemx_roce_quiet` → 结束计时 |
| Batch，fetch 类 | 开始计时 → 依次调用 N 次 atomic，每次接口返回前完成 → 结束计时 |
| Latency，非 fetch | 开始计时 → 重复 `atomic + aclshmemx_roce_quiet` 共 iterations 次 → 结束计时 |
| Latency，fetch 类 | 开始计时 → 重复 atomic 共 iterations 次，每次接口返回前完成 → 结束计时 |

“Fetch 类”包括 `fetch_*`、`swap` 和 `compare_swap`。Batch 使用现有的标量 atomic 接口，大小表示本批调用次数；接口内的等待仍然生效。非 fetch 的批末 quiet 允许多次提交分摊完成等待，但后端仍可能在队列空间不足时提前轮询完成，batch size 因此不等于实际同时在途的请求数。

Device 使用 `AscendC::GetSystemCycle()` 采集开始和结束的周期。计时包括循环、接口调用和对应的完成等待。PE 间同步、目标初始化、结果校验与回传、kernel 启动及 Host stream 同步均位于计时区间外。

每次正式采样结束后，PE1 扫描本地全部目标元素，并将校验结果回传 PE0。PE0 同时检查目标结果和最后一次 fetch 返回值：batch 对应本批最后一个元素，latency 对应最后一次调用。校验按所选类型的 32/64 位位模式进行，不检查中间每次 fetch 的返回值。

目标值、最后返回值、操作计数和有效计时均符合预期时，输出 `Verified=true`。校验失败会记录 `Verified=false` 并在 stderr 输出错误详情；程序完成剩余采样和同步后返回非零状态。

## CSV 输出

运行脚本后，结果保存在本示例的 `output/` 目录。仅 PE0 写入 CSV；重复运行相同测试模式、操作和类型会覆盖对应文件。

### 逐次采样结果

文件名：`rdma_atomic_<test>_<op>_<data_type>_pe0.csv`。

| 列 | 说明 |
| --- | --- |
| `Backend` | 传输路径，固定为 `ROCE` |
| `QPCount`、`PEs` | QP 数和 PE 数，固定为 1、2 |
| `InitiatorPE`、`TargetPE` | 发起端和目标端，固定为 0、1 |
| `Test`、`Op`、`DataType` | 测试模式、原子操作和数据类型 |
| `OperandBytes` | 单个操作数的字节数 |
| `BatchSize` | 每批的 atomic 调用次数；latency 固定为 1 |
| `TotalOps` | 本次正式样本的实际操作总数；batch 为 BatchSize，latency 为 Iterations |
| `Iterations` | 本次正式样本每个元素的操作次数；batch 固定为 1，latency 为命令行配置值 |
| `WarmupIterations` | 预热批数；latency 每批只有一次操作 |
| `Repetition` | 正式采样序号，从 1 开始 |
| `ElapsedCycles`、`ElapsedUs` | 本次样本的总周期数和总耗时（us） |
| `BatchLatencyUs` | 整批完成时间（us/batch），仅 batch 填写 |
| `AtomicRateMops` | 原子操作速率（百万次/秒），仅 batch 填写 |
| `LatencyUs` | 平均完成时延（us/op），仅 latency 填写 |
| `Verified` | 发起端与目标端联合校验结果 |

不适用于当前模式的指标字段留空。指标计算如下：

```text
ElapsedUs       = ElapsedCycles / cycles_per_us
BatchLatencyUs  = ElapsedUs
AtomicRateMops  = BatchSize / ElapsedUs
LatencyUs       = ElapsedUs / Iterations
```

Ascend950 的 `cycles_per_us` 为 1000，启动日志打印实际使用的换算值。BatchLatencyUs 越低表示相同数量的操作完成得越快，AtomicRateMops 越高表示操作速率越高。整批耗时除以 BatchSize 可以得到摊销到每个操作的时间，该值包含批量提交和完成等待的分摊效果。

### 汇总结果

文件名：`rdma_atomic_<test>_<op>_<data_type>_summary.csv`。

按 `Test`、`Op`、`DataType`、`BatchSize` 和 `Metric` 分组，保留相同的后端与 PE 元数据。Batch 的两个指标各占一行，latency 占一行。

| 列 | 说明 |
| --- | --- |
| `Metric` | `BatchLatencyUs`、`AtomicRateMops` 或 `LatencyUs` |
| `Samples` | 校验通过的正式样本数 |
| `Min`、`Max` | 样本最小值和最大值 |
| `Median` | 中位数；偶数样本取中间两项平均 |
| `P95` | 排序后第 `ceil(0.95 × Samples)` 项，序号从 1 开始 |

仅 `Verified=true` 的样本参与汇总。无有效样本时 `Samples=0`，统计值留空。默认 10 个样本时，按上述算法 P95 等于 Max。Batch 的耗时分位数反映整批耗时的波动；latency 的分位数反映多次平均时延的波动，不能作为单次请求的尾延迟。

### `--all` 总表

`bash run.sh --all` 成功完成全部配置后，在 PE0 所在主机的 `output/` 目录生成两张总表。每行对应一个原子操作，按上述 13 个操作的顺序排列，数值保留三位小数。

| 文件 | 列 | 数值来源 |
| --- | --- | --- |
| `rdma_atomic_all_latency_us.csv` | `int32`、`uint32`、`int64`、`uint64` | 各 latency 分项的 `LatencyUs` / `Median`，单位 us/op |
| `rdma_atomic_all_mops.csv` | 本轮实际测试的 BatchSize，从小到大 | 各 batch 分项的 `AtomicRateMops` / `Median`，类型 uint64、单位 Mops/s |

## 已知约束

1. 固定 2 PE、1 AIV core、1 QP，通信方向为 PE0→PE1；不提供双向、多 PE、多 AIV 或多 QP 参数。
2. 仅测试通过 `aclshmem_malloc` 分配的对称 HBM 内存，不支持 Host 侧 DRAM。
3. Batch 按连续元素逐个调用现有接口，不启用 WQE 聚合提交。Fetch 类接口的结果包含逐次完成等待的开销。
4. 比较测试结果时，应保持硬件、CANN 与驱动版本、网络拓扑和测试参数一致。
