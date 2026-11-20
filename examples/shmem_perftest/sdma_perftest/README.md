# sdma_perftest

## 示例概述

`sdma_perftest` 是用于测试 SDMA 数据搬运能力和性能的示例。该示例基于多 QP（Queue Pair）SDMA 接口，支持单向/双向、get/put、带宽/延迟等组合，并可通过 `--qp` 配置参与测试的 QP/AIV 数量。测试结果仅供参考，实际性能以业务场景为准。

[English version](README_en.md)

## 支持的测试

### 测试类型

| 测试类型 | 说明 | A2/A3 平台 | Ascend950 |
|---------|------|------------|-----------|
| `get` | 单向 get。profiling PE 从下一 PE 的 source 缓冲读取到本地 destination 缓冲 | 支持 | 支持 |
| `bi_get` | 双向 get。两个 PE 同时从对端读取数据 | 支持 | 支持 |
| `put` | 单向 put。profiling PE 将本地 source 缓冲写入下一 PE | 支持 | 不支持 |
| `bi_put` | 双向 put。两个 PE 同时向对端写入数据 | 支持 | 不支持 |

Ascend950 不支持 SDMA remote WRITE，因此 `put` 和 `bi_put` 会在进入 kernel 前被拒绝；`get` 和 `bi_get` 全流程只使用 SDMA READ 路径。

### 性能口径

| 参数 | 说明 | 适用测试类型 |
|------|------|--------------|
| `--metric bw` | 带宽测试，默认模式 | `get` / `bi_get` / `put` / `bi_put` |
| `--metric lat` | 延迟测试，统计每次 nbi 提交的平均窗口时间 | `get` / `bi_get` / `put` / `bi_put` |
| `--batch <N>` | 带宽测试中每提交 N 次 nbi 后 quiet；`0` 表示全部提交后 quiet 一次 | 仅 `--metric bw` 生效 |

## 环境要求

CANN 版本和 ops 包要求详见[CANN 版本说明](../../../docs/quickstart.md#431-cann-版本说明)。在仓库根目录完成编译：

```bash
# A2/A3 平台
bash scripts/build.sh -examples

# Ascend950 平台
bash scripts/build.sh -examples -soc_type Ascend950
```

## 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | 测试类型：`get` / `bi_get` / `put` / `bi_put` | get |
| `--datatype <type>` | `-d <type>` | 数据类型：`float` / `int8` / `int16` / `int32` / `int64` / `uint8` / `uint16` / `uint32` / `uint64` / `char` | float |
| `--qp <count>` | - | QP/AIV 数量，范围为 1 到当前设备可用 AIV 数，且不超过 72 | 2 |
| `--exponent <exponent>` | `-e <exponent>` | 单次传输数据量为 `2^exponent` 字节 | - |
| `--exponent-range <min> <max>` | - | 数据量幂数范围 | 3-17 |
| `--loop-count <count>` | - | 每个数据点的循环次数 | 1000 |
| `--ub-size <size>` | - | UB size，单位 KB | 16 |
| `--metric <bw\|lat>` | - | 性能口径：`bw` 或 `lat` | bw |
| `--batch <N>` | - | 带宽测试的 quiet 间隔。`0` 为全部 nbi 提交后 quiet 一次，`1` 为每次 nbi 后 quiet，其他值按组 quiet | `run.sh` 为 `512`，直接运行二进制为 `0` |
| `-pes <size>` | - | PE 数量，当前固定为 2 | 2 |
| `-ipport <ip:port>` | - | 通信地址 | tcp://127.0.0.1:8769 |
| `-gnpus <num>` | - | NPU 数量，当前固定为 2 | 2 |
| `-fnpu <id>` | - | 起始 NPU ID | 0 |
| `-fpe <id>` | - | 起始 PE ID，当前固定为 0 | 0 |

## 使用示例

单向 get：

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get -d float -e 10 --loop-count 10 \
  -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28769
```

双向 get：

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t bi_get -d float -e 10 --loop-count 10 \
  -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28775
```

指定 QP 数量：

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get --qp 48 -d float \
  -e 20 --loop-count 100 -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28779
```

延迟测试：

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get -d float -e 10 \
  --loop-count 1000 --metric lat -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28781
```

带宽测试中每次 nbi 后 quiet：

```bash
bash examples/shmem_perftest/sdma_perftest/run.sh -t get -d float -e 10 \
  --loop-count 1000 --batch 1 -pes 2 -gnpus 2 -ipport tcp://127.0.0.1:28783
```

也可通过顶层调度脚本运行：

```bash
bash examples/shmem_perftest/run.sh -m sdma -t get -d float --qp 4 -e 10 --loop-count 10
```

## QP/AIV 配置

- 每个 AIV 对应一个 QP，`AscendC::GetBlockIdx()` 作为 `aiv_idx` 和 `qp_idx`。
- 每个活跃 QP/AIV 每轮搬运一份完整消息，并使用独立的连续缓冲区区域，避免并发请求互相覆盖。
- 程序会在 `aclshmemx_init_attr` 前调用 `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num)` 创建本次测试所需的 QP。
- 请求的 QP/AIV 数量不能超过当前设备 vector core 数量，且不能超过 72。设备可用 AIV 数由程序运行时自动查询校验，`--qp` 超出容量时程序会直接报错并提示设备实际可用数量，因此不确定芯片型号时无需手工查询；如需事先确认，可在 `${ASCEND_TOOLKIT_HOME}/<arch>-linux/data/platform_config/`（默认 `/usr/local/Ascend/ascend-toolkit/latest/<arch>-linux/`，`arch` 为 `x86_64` 或 `aarch64`）下对应芯片型号的 `.ini` 文件中查看 `vector_core_cnt` 字段。

## 输出说明

测试结果写入 `output/sdma_<metric>_<test_type>_<datatype>_qp<QP数>_<prof_pe>.csv`，例如：

```text
output/sdma_bw_get_float_qp2_0.csv
output/sdma_lat_bi_get_float_qp4_0.csv
```

只有 profiling PE（环境变量 `SHMEM_CYCLE_PROF_PE` 指定，默认 PE0）会输出 CSV 文件。

CSV 文件字段如下：

| 字段 | 描述 |
|------|------|
| `DataSize/B` | 单次传输数据大小，单位字节 |
| `Npus` | 使用的 NPU 数量 |
| `QPs` | 实际使用的 QP/AIV 数量 |
| `UBsize/KB` | UB size，单位 KB |
| `Bandwidth/GB/s(1000)` | 带宽，按 1000 进制；`lat` 模式下为 0 |
| `Bandwidth/GiB/s(1024)` | 带宽，按 1024 进制；`lat` 模式下为 0 |
| `CoreMaxTime/us` | 所有活跃 AIV 中最大的单次执行时间；`lat` 模式下为平均单次延迟 |

## 注意事项

1. 数据量（`2^exponent` 字节）必须是所选数据类型大小的整数倍。
2. 本地对称内存默认配置 1GB；当测试数据量较大时，程序会按最大数据量和 QP 数自动上调，运行前需确保设备内存充足。
3. 每个数据点在正式采集前会先执行 100 次预热迭代。
4. 建议使用较大的 `--loop-count` 获取更稳定的性能数据。
5. `--metric lat` 中 `--batch` 不生效，所有 nbi 提交位于同一个计时窗口，quiet 在窗口外执行。
6. `--batch` 过大时，同一 QP 上会累积多笔在途请求，可能受到 SDMA 通道 SQ 深度限制。
