# TP=2 AIV-UDMA AllReduce 示例

## 示例概述

本示例在单机启动多个 PE，并将相邻的两个 PE 划分为一个 TP=2 通信组。每个 TP 组独立执行
`reduce-scatter + local reduce + all-gather`，不同 TP 组之间不传输 AllReduce 数据。

示例通过 `--mode` 提供两种通信模式，并复用相同的 Host 初始化、Local Reduce、性能统计和
golden 数据校验逻辑：

- `baseline`：Core 0 调用 `aclshmemx_udma_put_nbi` 直发数据，并调用 `aclshmemx_udma_quiet` 等待完成。
- `tailcut`：将通信分片按比例切分到一条直发路径和两条绕路路径，Core 0、1、2 分别提交 direct、relay0、relay1 路径，随后由 Core 0 调用一次 `aclshmemx_udma_quiet`。

Local Reduce 按 tile 切分给所有启动的 AIV Core 计算。Tailcut 模式将每四个连续 PE 视为一个
box，当前 TP 对之外的两个 PE 作为 relay PE，默认流量权重为 `2:1:1`。


## 相关文档

完整的算子说明、Tailcut 绕路设计和性能测试方法，请参见
[TP=2 AIV-UDMA AllReduce 算子实践](https://gitcode.com/cann/shmem/wiki/operator-practice-tp-allreduce-udma)。

## 编译

在 shmem 根目录执行：

仅使用 baseline 模式：

```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

同时使用 baseline 和 tailcut 模式：

```bash
bash scripts/build.sh -examples -soc_type Ascend950 -enable_relay
```

Tailcut 模式依赖 relay 接口，必须启用 `-enable_relay`。对比两种模式的性能时，应使用相同的
relay 编译配置。

## 运行

直接运行脚本将使用 baseline 默认参数执行一次 FP16 功能测试，并与 golden 数据比较：

```bash
bash examples/tp_allreduce_udma/scripts/run.sh
```

运行 tailcut 功能测试（默认使用 4 PE、3 个 AIV Core 和 `2:1:1`）：

```bash
bash examples/tp_allreduce_udma/scripts/run.sh --mode tailcut
```

使用相同配置进行性能对比：

```bash
bash examples/tp_allreduce_udma/scripts/run.sh \
    --mode baseline -pes 4 -fnpu 0 -elements 1048576 -aiv 20 \
    --perf --warmup 1 --loops 20 --prof-pe 0 \
    --perf-csv output/baseline_perf.csv

bash examples/tp_allreduce_udma/scripts/run.sh \
    --mode tailcut -pes 4 -fnpu 0 -elements 1048576 -aiv 20 \
    --ratio 2:1:1 --perf --warmup 1 --loops 20 --prof-pe 0 \
    --perf-csv output/tailcut_perf.csv
```

## 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mode` | 通信模式，支持 `baseline`、`tailcut` | `baseline` |
| `-pes` | PE 数量；baseline 支持 2、4、8、16，tailcut 支持 4、8、16 | 未指定时 baseline 为 2、tailcut 为 4 |
| `-fnpu` | 使用的第一张 NPU 卡号 | 0 |
| `-elements` | 每个 PE 输入 Tensor 的元素数量 | 1024 |
| `-type` | 数据类型，支持 `int32_t`、`float16_t` | `float16_t` |
| `-aiv` | Kernel 使用的 AIV Core 数量；tailcut 至少为 3，且不能超过设备的 Vector Core 数量 | 未指定时 baseline 为 1、tailcut 为 3 |
| `-port` | SHMEM 初始化端口 | 8899 |
| `--ratio` | Tailcut 的 direct、relay0、relay1 流量权重 | `2:1:1` |

## 性能参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--perf` | 开启性能测试模式 | 关闭 |
| `--warmup` | 性能测试的预热轮数，可设置为任意非负整数 | 未指定时功能模式为 0、性能模式为 1 |
| `--loops` | 性能测试的计时轮数 | 1 |
| `--prof-pe` | 输出设备侧分段性能数据的 PE | 0 |
| `--perf-csv` | 性能结果 CSV 路径 | `output/<mode>_perf.csv` |

golden 数据和输出固定放在示例目录下的 `golden/`、`output/`。每次性能测试都会新建或覆盖
`--perf-csv` 指定的 CSV 文件，不追加历史结果。

## 结果说明

每个 PE 输出 `[SUCCESS]` 表示 Kernel 执行完成；最后输出
`tp_allreduce_udma golden check passed` 才表示所有 PE 的结果均通过 golden 数据校验。


## 约束

- `baseline` 与 `tailcut` 算子依赖 UB-Mesh 互联，以及 AIV Core 直驱 UDMA 的能力，**当前仅支持 Ascend 950PR&950DT系列产品**。
- `tailcut` 需借助其他 NPU 中转部分通信数据，**当前 relay 功能仅支持 Atlas 950 SuperPoD**；非该形态部署下，无法保证绕路转发可用，请勿使用 `tailcut` 模式。
- TP 大小固定为 2。
- `elements` 必须为正数且能被 2 整除。
- Tailcut 模式要求 PE 数量能被 4 整除，且 AIV Core 数量至少为 3。
- `--ratio` 仅用于 Tailcut 模式，三个权重必须为非负整数且总和大于 0。
- 每阶段的流量边界按 `floor(shard_elements * 累计权重 / 总权重)` 计算，权重为 0 的路径不发送数据。
