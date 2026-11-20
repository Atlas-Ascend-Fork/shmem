# CMO (Cache Maintenance Operation) 功能演示与读性能测试示例

## 功能简介

本示例演示了如何使用Shmem的CMO（Cache Maintenance Operation）接口来优化GM（Global Memory）内存访问性能。该CMO接口提供L2缓存管理操作，可以通过预取（Prefetch）将数据从GM提前加载到L2缓存中，从而减少数据访问延迟，提升整体计算性能。当前实现支持现有A2/A3平台以及Ascend950。

### L2缓存背景知识

昇腾AI处理器采用多级缓存架构，L2缓存是二级缓存，位于AI Core和全局内存（HBM）之间，具有以下特点：

- **容量**：大容量高速缓存（A2/A3 经典值192MB）
- **访问速度**：缓存命中带宽约为缓存miss带宽的2~4倍
- **缓存管理**：支持提前将数据加载到缓存，掩盖内存访问延迟

通过合理使用CMO预取操作，可以在计算进行的同时提前准备下一批数据，提升整体性能。

### 当前测试内容

程序每个 PE 依次执行以下内容：

1. **不带 QP 的 CMO 基本用法演示**
   - 分配一块 1MB 的本地 GM。
   - 只由 0 号 AIV 调用 `aclshmemx_cmo_nbi`，随后调用 `aclshmemx_sdma_quiet`。
   - 该部分只用于演示接口调用，不写入性能 CSV。

2. **GM 到 UB 的读带宽测试**
   对比以下三种预取方式：
   - `NO_PREFETCH`：对 `trash_gm` 做预取以扰动/清理缓存，实际拷贝目标 `cache_gm` 不做预取。
   - `HOST_PREFETCH`：Host 侧调用 `aclrtCmoAsync`，对整块 `cache_gm` 预取。
   - `DEVICE_BLOCK_PREFETCH`：kernel 内每个 AIV 对自己负责的 `cache_gm` 分块调用 CMO 预取。

   当前代码配置为每次循环拷贝 64MB、重复 100 次，`n_blocks` 的当前有效值为 20。`copypad_size` 从 8B 到 128KB 的 2 次幂，并额外测试 192KB 和 256KB；实际步长不小于 512B。结果写入 `<PE_ID>_band.csv`。

3. **CMO 延迟测试**
   使用同一组 CMO 预取大小分别测试单核接口和显式 QP 接口。预取大小为 512B、1KB、2KB ... 4MB，以及额外的 96MB。
   - 单核接口测试固定 `nbi_blocks = 1`，仅由 0 号 AIV 调用 `aclshmemx_cmo_nbi`，结果写入 `<PE_ID>_cmo_nbi.csv`。
   - 显式 QP 测试使用 `aclshmemx_cmo_qp_nbi`，AIV 数量依次为 `{1, 2, 4, 8, 16, 32, 40}`。每个 AIV 使用与自身编号对应的 QP，结果写入 `<PE_ID>_cmo_qp.csv`。

### 核心接口

#### CMO接口（SHMEM扩展接口）

```c
template <typename T>
void aclshmemx_cmo_qp_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                          __ubuf__ T *buf, uint32_t ub_size, uint32_t qp_idx, uint32_t sync_id);
```

- **功能**：在Device侧异步触发CMO操作，向STARS队列提交操作任务
- **参数说明**：
  - `src`：全局内存地址
  - `elem_size`：元素数量
  - `cmo_type`：CMO操作类型（当前仅支持CMO_TYPE_PREFETCH）
  - `buf`：临时UB缓冲区地址
  - `ub_size`：UB缓冲区大小（至少64字节，64字节对齐）
  - `qp_idx`：显式选择的SDMA QP，多核并发调用时各 AIV 应使用独立 QP，且必须小于初始化时配置的 QP 数
  - `sync_id`：同步ID

##### CMO操作类型

**注意**：当前SHMEM实现仅支持`CMO_TYPE_PREFETCH`操作。

- **CMO_TYPE_PREFETCH**：预取操作，将数据从全局内存提前加载到L2缓存
- **CMO_TYPE_WRITEBACK**：写回操作，将L2缓存中的修改数据写回全局内存，同时在缓存中保留副本
- **CMO_TYPE_INVALID**：失效操作，丢弃L2缓存中的数据块
- **CMO_TYPE_FLUSH**：刷新操作，强制将L2缓存数据写回全局内存并从缓存中移除

#### SDMA Quiet接口（SHMEM扩展接口）

```c
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_quiet(AscendC::LocalTensor<T> &buf,
                                              uint32_t qp_idx, uint32_t sync_id);
```

- **功能**：等待STARS队列中的操作任务完成，用于同步
- **参数说明**：
  - `buf`：临时UB缓冲区地址
  - `qp_idx`：必须与对应CMO提交使用的QP一致
  - `sync_id`：同步ID
- **特点**：通过下发SDMA的Flag任务，并轮询Flag等待STARS队列中的操作完成

#### 不带QP的CMO接口（SHMEM扩展接口）

```c
template <typename T>
void aclshmemx_cmo_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                       __ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);
```

- **功能**：与`aclshmemx_cmo_qp_nbi`一致，区别是固定使用QP 0，无需传入`qp_idx`
- **参数说明**：同`aclshmemx_cmo_qp_nbi`，但没有`qp_idx`参数
- **完成等待**：调用`aclshmemx_sdma_quiet`（不带QP版本，同样固定等待QP 0）
- **适用场景**：单个 AIV 执行的 CMO 操作。多个 AIV 并发预取时，应使用显式 QP 接口，各 AIV 使用独立 QP，避免同时争用固定的 QP 0。

本示例在性能测试前会执行一次不带QP的预取演示（见`main.cpp`中的`cmo_pretech`内核，由`test_copy_perf`调用），仅演示基本用法：

```c++
// kernel内（仅0号AIV执行）
aclshmemx_cmo_nbi(src, size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
```

## 环境要求

当前用例不按PE数量扩展对称数据区，CMO测试数据均为每个进程本地申请；当前`run.sh`采用单机最多8个NPU的启动模型。每个NPU的显式QP上限为72。

### 硬件要求

- 昇腾AI处理器（Atlas 200I A2/A3、Atlas 300T A2/A3、Ascend950等）
- 架构兼容：aarch64、x86

### 软件依赖

参考仓内[CANN版本说明](../../docs/quickstart.md#43-cann)和[编译与构建](../../docs/compilation_build_guide.md)，配置支持CMO功能的CANN版本。

| 平台 | CMO功能CANN版本要求 | toolkit包 | ops包 |
| --- | --- | --- | --- |
| A2/A3 | CANN 9.0.0-beta.2及以上 | 9.0.0-beta.2及以上 toolkit包：[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) | 9.0.0-beta.2及以上 ops包：[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) |
| Ascend950 | CANN 9.1.0及以上 | 9.1.0 toolkit 包：[x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-toolkit_9.1.0_linux-x86_64.run) / [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-toolkit_9.1.0_linux-aarch64.run) | 9.1.0 ops包：[950 x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-950-ops_9.1.0_linux-x86_64.run) / [950 aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-950-ops_9.1.0_linux-aarch64.run) |

toolkit包和ops包需要安装到同一目录：

```bash
# 自定义CANN安装目录，可按实际环境修改
export INSTALL_PATH=/home/user/ascend
chmod +x Ascend-cann-toolkit_{cann_version}_linux-$(uname -m).run
chmod +x Ascend-cann-{soc_name}-ops_{cann_version}_linux-$(uname -m).run
./Ascend-cann-toolkit_{cann_version}_linux-$(uname -m).run --install --install-path=${INSTALL_PATH}
./Ascend-cann-{soc_name}-ops_{cann_version}_linux-$(uname -m).run --install --install-path=${INSTALL_PATH}
source ${INSTALL_PATH}/ascend-toolkit/set_env.sh
```

### 功能依赖说明

**重要**：本示例中的Device侧CMO接口`aclshmemx_cmo_nbi`依赖SDMA功能，需要参考`examples/sdma`或`examples/cmo`，配置`attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA`以启动SDMA引擎。

### 平台支持

Ascend950仅支持CMO功能。

暂不支持通过SDMA put/get接口使用read/write数据搬运能力。因此，`aclshmemx_sdma_put_nbi`、`aclshmemx_sdma_get_nbi`等SDMA put/get接口不适用于Ascend950及以上平台。

## 编译步骤

### 1. 编译示例程序

编译环境配置、源码编译、二进制包安装等通用流程请参考[编译与构建](../../docs/compilation_build_guide.md)。

```bash
cd shmem/
# A2/A3 平台
bash scripts/build.sh -examples
# Ascend950 平台
bash scripts/build.sh -soc_type Ascend950 -examples
```

编译成功后，关键产物包括：

- 可执行文件：`build/bin/cmo`
- SHMEM库：`build/lib/libshmem.so`

## 运行方法

```bash
cd shmem/examples/cmo
bash run.sh -pes ${PEs} -type ${TYPE}
```

### 参数说明

- PEs：指定用于运行的设备（NPU）数量，限定单台机器内。
- TYPE：指定传输数据类型，当前支持：int，uint8，int64，fp16，fp32。

### 运行示例：使用2个NPU测试int类型数据

```bash
bash run.sh -pes 2 -type int
```

## 输出结果

### 控制台输出

程序运行时会输出每个PE的完成信息：

```bash
PE 0 Finished!
PE 1 Finished!
[SUCCESS] demo run success in pe 0
[SUCCESS] demo run success in pe 1
```

### CSV文件输出

程序会在`output/`目录下生成以下CSV文件：

#### 1. `{PE_ID}_band.csv` - 带宽性能测试结果

包含以下列：

- `loop_times`: 循环次数（默认100次）
- `copy_size_per_loop`: 每次循环拷贝的数据大小，当前为 64MB
- `blocks`: 使用的 block 数量，当前为 20
- `copypad_size`: 单次DataCopy操作的数据大小
- `no_prefetch_time/us`、`host_prefetch_time/us`、`device_block_prefetch_time/us`：各模式每个 AIV 的平均拷贝时间；CSV 中取 100 次样本的 p50
- `no_prefetch_band/Gbps`、`host_prefetch_band/Gbps`、`device_block_prefetch_band/Gbps`：各模式所有 AIV 的带宽之和；CSV 中取 100 次样本的 p50

#### 2. `{PE_ID}_cmo_nbi.csv` - 单核 CMO 延迟结果

- `loop_times`：统计循环次数，当前为 100 次；另有 1 次 warmup，不参与统计
- `blocks`：当前固定为 1
- `cmo_size`：512B 到 4MB 的 2 次幂，以及 96MB
- `cmo_submit_time_p05/p50/p95/us`：从调用 `aclshmemx_cmo_nbi` 开始到接口返回的提交耗时
- `cmo_execute_time_p05/p50/p95/us`：从提交开始到 `aclshmemx_sdma_quiet` 返回的耗时

该文件的每一行是 100 次样本的 p05、p50 和 p95。由于不带 QP 的路径只由 0 号 AIV 发起，延迟只统计 0 号 AIV；拷贝校验仍由测试 kernel 执行，但不作为该 CSV 的输出指标。

#### 3. `{PE_ID}_cmo_qp.csv` - 多 QP CMO 延迟结果

- `loop_times`：统计循环次数，当前为 100 次；另有 1 次 warmup，不参与统计
- `aiv_num`：并发 AIV/QP 数量，依次为 1、2、4、8、16、32、40
- `cmo_size`：512B 到 4MB 的 2 次幂，以及 96MB
- `cmo_qp_submit_time_avg/max/us`：各 AIV 先分别取 100 次 submit 样本的 p50，再对这些 AIV 的 p50 求平均值和最大值
- `cmo_qp_execute_time_avg/max/us`：各 AIV 先分别取 100 次 execute 样本的 p50，再对这些 AIV 的 p50 求平均值和最大值
- `cmo_qp_submit_time_p05/p50/p95/us`、`cmo_qp_execute_time_p05/p50/p95/us`：对所有参与 AIV 的“每核 p50”再次计算分位数
- `cmo_qp_*_core_<N>/us`：第 N 个 AIV/QP 的独立 p05、p50、p95；未参与本行测试的 core 写为 `N/A`

这里的 `submit` 是 CMO 接口提交耗时，`execute` 是从提交开始到对应 `aclshmemx_sdma_qp_quiet` 返回的耗时。QP 测试中 `qp_idx` 与 AIV 编号相同，应用实际使用时也必须保证提交和 quiet 使用同一个 QP。

### 指标统计方式

- 所有延迟由 Device 侧 `AscendC::GetSystemCycle()` 计时，并按当前编译目标的 `cycle2us` 换算为微秒。
- 带宽测试的 kernel 对每个 AIV 重复执行 512B 粒度的 `DataCopyPad`，先得到该 AIV 的拷贝时间和带宽，再汇总所有 AIV。
- 带宽 CSV 的时间和带宽均为 100 次测试样本的 p50，不是所有原始采样的平均值。
- NBI 延迟 CSV 直接对 100 次单 AIV 样本计算分位数。
- QP 延迟 CSV 先按 AIV 对 100 次样本计算 p50，再基于各 AIV 的 p50 计算 avg、max、p05、p50、p95。

### 单核接口与多 QP 接口选择建议

32KB 是当前测试和使用建议中的经验分界，不是接口的硬性限制，最终应以目标平台实测结果为准：

- 当单次预取范围不超过 32KB 时，推荐使用单核接口 `aclshmemx_cmo_nbi`，完成后调用 `aclshmemx_sdma_quiet`。该路径固定使用 QP 0，调用关系简单，适合单 AIV 或无法拆分并发的预取。
- 当单次预取范围超过 32KB，且数据可以拆分为多个相互独立的区间时，建议由多个 AIV 并发调用 `aclshmemx_cmo_qp_nbi`。每个 AIV 使用独立的 `qp_idx`，完成时调用同一个 QP 对应的 `aclshmemx_sdma_qp_quiet`。
- 使用多 QP 前，Host 侧需要通过 `aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num)` 配置足够数量的 QP。不能使用 `aclshmemx_sdma_quiet` 等待非 QP 0 的请求，也不能让多个并发 AIV 无约束地共享同一个 QP。
- 如果数据无法拆分为独立区间，或并发管理成本抵消了收益，仍推荐使用单核接口。32KB 分界仅用于指导选型，不代表超过 32KB 就必然更快。

### 性能指标说明

- **带宽**：衡量数据传输速率，单位为GB/s
- **延迟**：衡量操作完成时间，单位为微秒
- **分位数**：用于统计分布情况，p50为中位数。

## 参考文档

- [CANN应用开发接口文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/index/index.html)
- [内存管理 aclrtCmoAsync](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/runtimeapi/aclcppdevg_03_0123.html)
