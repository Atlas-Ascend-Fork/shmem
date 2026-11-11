# 样例介绍

本样例展示在 SIMD 与 SIMT 混合编译模式下，如何对 SIMT 远程内存访问（RMA）的 **ub2gm 接口**进行性能测试（Performance Test）。
与 `simt_rma_perftest`（gm2gm）不同，本样例以 UB（Unified Buffer）作为通信的一端，测试用于评估单机两卡间 Device-to-Device 的 `ub2gm` 数据传输能力，覆盖**单向**的 `put` 与 `get` 操作（通过 `-t` 参数选择），并输出带宽与时延统计。

| 操作 | 被测数据路径 |
| --- | --- |
| `put` | 本卡 UB → 对端 GM |
| `get` | 对端 GM → 本卡 UB |

## 测试模型

测试固定使用两张卡，PE 号分别为 0 和 1：

- **Active PE（PE 0）**：所有通信操作（`put` / `get`）的发起方。
- **Passive PE（PE 1）**：通信的对端，不主动发起操作。

两种操作的数据流向与校验方分别为：

| 操作 | 数据流向 | 校验方 |
| --- | --- | --- |
| `put` | Active PE 的 UB → Passive PE 的对称内存 | Passive PE |
| `get` | Passive PE 的对称内存 → Active PE 的 UB | Active PE |

## 性能测试方法

为使统计结果尽可能聚焦于 ub2gm 接口的开销，并避免数据缓存（Data Cache）命中导致结果虚高，本样例采用以下设计：

- **UB 缓冲区在核函数内申请**：UB 数组定义在 `__global__` 核函数中，由 `asc_vf_call` 传入 `__simt_vf__` 函数使用。数据准备与结果回写都在计时区间之外完成，测量循环的每次迭代内只有一次 ub2gm 接口调用，另有一次 `VSSync`（S_V / V_S 打点）与一次 `AscendC::SyncAll<true>()` 全核同步；这两项同步开销计入计时窗口，在小数据量下占比不可忽略。
- **仅 Active PE 发起、源与目的同址**：`put` / `get` 是单边（one-sided）操作，全部由 Active PE 发起，Passive PE 仅作为远端对象、不参与计算（两卡通过 host 侧 barrier 同步）。GM 一侧的地址在两卡的相同对称内存偏移上。
- **UB 数据来自本卡 GM**：`put` 测试在计时前，用 `DataCopyPad` 将本卡对称内存中的数据搬运到 UB。由于同一 PE 的对称内存被填充为同一值，只需在测量循环外搬运一次。
- **预热与多轮取均值**：单次测量会先执行固定 128 次传输进行预热（不计入统计），再执行 `loops` 次传输进行采样。计时方式为：预热结束后，对这 `loops` 次传输整体计时（仅在首次采样前打点开始、末次采样后打点结束），统计得到的总时长再除以 `loops` 得到单次耗时，从而摊薄打点开销、排除冷启动的影响。
- **逻辑段与逻辑环**：每个 Core 物理上分配一块 1MB 的对称内存。设单次传输大小为 $X$ 字节、使用 $N$ 个 Core，定义每个 Core 的**逻辑段**大小为 $L = \min(1\text{MB},\ (128 + loops) \times X)$（且不小于 $X$）。$N$ 个逻辑段首尾相接，构成一个大小为 $N \times L$ 的**逻辑环**，Core $j$ 的逻辑段起始于环内偏移 $j \times L$。
- **滑动窗口遍历**：第 $i$ 次迭代时，所有 Core 共同构成一个滑动窗口，Core $j$ 本次传输的 GM 侧偏移为 $\big(i \times X + j \times L\big) \bmod (N \times L)$，即各 Core 在环上彼此错开一个逻辑段。窗口每轮前进 $X$ 字节并在环内回绕，使连续传输的地址不断前移，避免数据缓存命中抬高带宽读数。
- **结果校验**：`put` 由 Passive PE 校验。滑动窗口会写满整个逻辑环，因此按逻辑段布局逐段比对每段的前 $L$ 字节。`get` 由 Active PE 校验，核函数在计时结束后将 UB 内容回写到本 Core 逻辑段的首地址，故只比对每段的前 $X$ 字节。

## 源文件宏定义配置

被测操作（`put` / `get`）通过 `-t/--test-type` 参数在运行时选择（见下文）。其余测试维度通过 `main.cpp` 开头的常量定义控制，在编译前可直接修改其中的可配置项来改变被测接口的行为：

| 常量定义 | 含义与作用 | 可选值 / 默认值 |
| --- | --- | --- |
| `T` | 传输负载的元素类型，`DATA_SIZE` 由其推导。 | 固定为 `int32_t` |
| `DATA_SIZE` | 所调用 RMA 底层接口的数据位宽（单位：bit），由 `sizeof(T) * 8` 得出，不单独配置。 | 固定为 `32` |
| `THREAD_COUNT` | SIMT 模式下单 Core 启动的线程数，决定向量指令流的并发规模。 | 默认 `1024` |
| `UB_BUFFER_SIZE` | UB 缓冲区可容纳的元素个数，决定单次传输大小的上限。 | 默认 `16384`（即 64KB，对应 $2^{16}$ 字节） |
| `WARMUP_LOOPS` | 预热轮数（不计入统计）。 | 默认 `128` |

> **提示**：修改上述常量后，需重新回到根目录执行编译（见下文），新配置才会生效。
> 数据位宽固定为 32 bit：`main.cpp` 中的 `static_assert` 会拦住其他取值，且 `transfer_vf_put` / `transfer_vf_get` 里直接调用的是 `simt::aclshmemx_int32_{put,get}_nbi_block`，仅改 `T` 并不会切换到其他位宽的接口。
> 单次传输大小受 UB 容量限制：按默认配置上限为 $2^{16}$ 字节，因此 `-e`/`--exponent-range` 的取值被限制在 $[3, 16]$，超出会报错退出。需要测试更大数据量时，请同时增大 `main.cpp` 中的 `UB_BUFFER_SIZE` 与 `argparser.h` 中的 `BYTES_IN_EXP_UPPER` 后重新编译。
> 本样例仅支持 SIMT 模式，不提供 SIMD 模式对照。

## 支持的设备

- Ascend950

## 使用方式

1. **配置 CANN 环境变量**
   编译前需先加载 CANN 的环境变量（按实际安装路径选择其一）：

   ```bash
   # 默认安装路径
   source /usr/local/Ascend/cann/bin/set_env.sh
   # 自定义安装路径
   source ${install_path}/cann/bin/set_env.sh
   ```

2. **编译项目**
   在 `shmem/` 根目录下执行编译脚本：

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

3. **运行示例程序**
   进入示例目录并执行运行脚本：

   ```bash
   cd examples/shmem_perftest/simt_rma_ub2gm_perftest
   bash run.sh [options]
   ```

### 脚本参数说明

`run.sh` 支持以下参数，用于调节测试规模与条件：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-pes <int>` | PE 总数。本测试固定两卡模型，必须为 2。 | 2 |
| `-ipport <ip:port>` | ACLSHMEM 初始化通信地址。 | `tcp://127.0.0.1:8760` |
| `-gnpus <int>` | 本机启动进程数 / NPU 数量。本测试固定为 2，传入其他值会报错。 | 2 |
| `-fnpu <int>` | 首个 NPU ID，实际 device id 为 `pe_id % gnpus + fnpu`。 | 0 |
| `-fpe <int>` | 保留为与 shmem_perftest 参数兼容，本样例不使用：PE rank 固定为 0 和 1，device id 由 `pe_id % gnpus + fnpu` 得出，均不受该参数影响。 | 0 |
| `-t`/`--test-type <put\|get>` | 选择被测操作类型：`put` 或 `get`。 | `get` |
| `-b`/`--block-size <int>` | 每个 PE 使用的 Core（Block）数量。 | 32 |
| `--block-range <min> <max>` | Core（Block）数量扫描范围，每个核数各产出统计结果。 | 32 32 |
| `--block-list <b1,b2,...>` | 以逗号分隔显式指定要测试的核数（如 `1,8,16`）。同时指定时优先于 `-b`/`--block-size` 与 `--block-range`。 | - |
| `--loop-count <int>` | 正式采样的循环次数，取值须落在 $[1, 10000)$，越界会报错退出。 | 1000 |
| `-e`/`--exponent <exp>` | 单次传输数据量的指数（单值），取值为 2 的指数（例如 `10` 表示 $2^{10} = 1024$ 字节）。 | - |
| `--exponent-range <min> <max>` | 单次传输数据量的指数范围，取值须落在 $[3, 16]$。 | 3 16 |
| `-h`/`--help` | 打印参数说明并退出。 | - |

> 本测试为固定的两卡（Active PE0 / Passive PE1）模型，启动进程数与程序内 PE 数均固定为 2。`-pes` 和 `-gnpus` 保留为与 shmem_perftest 参数兼容，但传入非 2 的值会报错。
> 核数须落在 $[1, 64]$：上界为每个 PE 的 profiling 缓冲区所含的 Core 槽位数（`ACLSHMEM_CYCLE_PROF_MAX_BLOCK`），超出会报错退出。
> 测试会从 `--exponent-range` 的 min 到 max 逐个指数遍历单次传输数据量（即 $2^{min}, 2^{min+1}, \dots, 2^{max}$ 字节），并在 `--block-list`（或 `--block-range`）指定的核数集合上逐个核数遍历，每个（核数, 数据量）组合各产出一行统计结果。

例如，测试 `4` 个 Core 在传输 $2^8$ 到 $2^{12}$ 字节数据时的性能表现：

```bash
bash run.sh -b 4 --exponent-range 8 12
```

测试 `put` 操作（默认测 `get`）：

```bash
bash run.sh -t put -b 4 --exponent-range 8 12
```

在多个核数上遍历：

```bash
bash run.sh --block-list 1,8,16,24,48 --exponent-range 4 16
```

### 性能输出说明

测试正常结束后，Active PE（PE 0）会在示例目录下的 `output/` 子目录中输出一个 `.csv` 性能统计文件，文件名格式为：

```bash
ub2gm_[DATA_SIZE]_[blocks]_[OpType]_simt_[minExp]-[maxExp]_l[loop_count]_t[THREAD_COUNT]_ub[UB_BUFFER_SIZE].csv
```

其中 `[blocks]` 段反映本次实际测试的核数集合，命名规则只取决于该集合本身、与它是由哪个参数指定的无关：若集合恰好是一段连续升序区间，记为 `min-max`（如 `--block-range 1 4` 与 `--block-list 1,2,3,4` 都记为 `1-4`）；否则按测试顺序以 `_` 连接（如 `--block-list 1,8,16` 记为 `1_8_16`，`--block-list 16,1,8` 记为 `16_1_8`）。

`.csv` 文件中各列含义如下：

| 列名 | 说明 |
| --- | --- |
| `DataSize/B` | 单次 RMA 通信传输的数据量（字节），对应本行采样的 $2^{exp}$ 取值。 |
| `Npus` | 参与测试的 PE 数量，两卡测试下为 2。 |
| `Blocks` | 参与通信的 Core（Block）数量，即本行采样所用的核数。核数扫描（`--block-range` / `--block-list`）下逐行取自扫描集合。 |
| `UBsize/elements` | UB 缓冲区可容纳的元素个数，即编译期 `UB_BUFFER_SIZE`。 |
| `Bandwidth/GB/s (1000)` | 本组参数测得的跨卡平均传输带宽，按十进制单位换算（除以 $1000^3$）。 |
| `Bandwidth/GiB/s (1024)` | 同一带宽按二进制单位换算（除以 $1024^3$）。 |
| `CoreMaxTime/us` | 多个 Core 中，单次操作平均耗时最长的那个 Core 的时延（微秒），即带宽计算所用的时间。 |
| `SingleCoreTime/us` | 各 Core 单次操作的平均时延（微秒），由该 Core `loops` 次传输的总时长除以 `loops` 得到，每个 Core 一列。 |
