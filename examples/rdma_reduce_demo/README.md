# 样例介绍

本样例使用 RoCE Put Reduce 接口进行 ADD、MAX、MIN 分片归约。每个 PE 向其他所有 PE 归约写对应分片，并校验本地归约结果。样例还使用 `aclshmemx_half_sat_t` 标签演示 FP16 饱和加法，正向溢出结果钳位到最大有限值 65504。

## 环境要求

- 当前支持 Ascend950 平台的 1825 网卡（HNS_1825），不支持云脉网卡（XSCALE）。
- 至少使用两张 NPU 卡，RDMA 网卡及驱动已正确安装配置。
- CANN 版本要求、RDMA 环境检查及网卡插件库配置参见 [RDMA 样例说明](../rdma_demo/README.md#环境要求)。

## 使用方式

### 编译

参考 [RDMA 样例的编译说明](../rdma_demo/README.md#编译)，选择 **Ascend950 + HNS_1825** 的编译命令。

### 运行

按 [RDMA 样例的环境变量说明](../rdma_demo/README.md#ibv_extend_drivers-环境变量) 设置 1825 网卡对应的 `IBV_EXTEND_DRIVERS`。手动运行时，按 [RDMA 样例的运行说明](../rdma_demo/README.md#运行) 配置 `LD_LIBRARY_PATH`。

#### 单机运行

在 `shmem/` 目录执行：

```bash
bash examples/rdma_reduce_demo/run.sh
```

`run.sh` 默认启动 2 个 PE，使用本机设备 0、1。可通过 `-pes` 指定 PE 数量：

```bash
bash examples/rdma_reduce_demo/run.sh -pes 8
```

#### 跨机运行

两台机器各使用一张 NPU 卡。假设机器 A 的 IP 为 `ip1`，在两台机器的 `shmem/` 目录设置：

```bash
export SHMEM_UID_SESSION_ID=ip1:8899
```

在机器 A 启动 PE0：

```bash
./build/bin/rdma_reduce_demo 2 0 tcp://ip1:8899 1 0 0
```

同时，在机器 B 启动 PE1：

```bash
./build/bin/rdma_reduce_demo 2 1 tcp://ip1:8899 1 1 0
```

参数含义参见 [RDMA 样例的参数说明](../rdma_demo/README.md#命令行参数说明)。

#### 预期结果

运行成功后，每个 PE 输出校验与执行成功信息。双 PE 示例：

```text
check transport result success, relative pe=0
[SUCCESS] demo run success in relative pe 0
check transport result success, relative pe=1
[SUCCESS] demo run success in relative pe 1
```
