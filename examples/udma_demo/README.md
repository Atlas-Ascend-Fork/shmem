# 样例介绍

## 版本和平台支持说明

- UDMA 样例仅支持 Ascend950 平台，其他平台不支持运行该样例。
- CANN 9.1.0 已提供 UDMA 所需的 `HcommEndpointCreate`、`HcommMemReg`、`HcommChannelCreate`、`HcommChannelGetStatus` 等 HCOMM 资源接口。请从 [CANN 9.1.0 资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0)下载并安装 Ascend950 对应的 toolkit 包和 ops 包；低于 9.1.0 的 CANN 版本不在本样例支持范围内。
- 初始化时会加载并检查所需 HCOMM 符号。编译前需执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（自定义安装路径时使用对应的 `set_env.sh`），并使用 `-soc_type Ascend950` 编译；若 HCOMM 运行时库或符号不完整，初始化会失败。
- 样例会在初始化后启用 `aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG)`，并在 kernel 同步后调用 `aclshmemx_report_exception()`，用于展示 UDMA 异常上报接口的调用示例。

使用方式:

0.运行前确认环境已使能UDMA：
- 单机：执行`npu-smi info -t topo`查看NPU卡间拓扑，卡间连接类型为`UB`即表示支持UDMA：
![npu-smi info -t topo 输出示例](../../docs/images/udma_demo_env.png)
- 跨机：server和pod机型支持跨机UDMA通信，运行前需先确认节点间UB链路连通，可使用`hccn_tool`在节点间发起UB ping验证（也可使用CANN自带的hccl_test工具在节点间运行集合通信用例验证连通性，详见[HCCL 性能测试工具](https://www.hiascend.com/document/redirect/CANNCommunityToolHcclTest)）；定制机型需结合具体硬件配置判断是否支持。
  `hccn_tool`随Ascend HDK驱动包安装，位于`<驱动安装路径>/driver/tools/`目录（默认驱动安装路径为`/usr/local/Ascend`）。
  1. 分别在两个节点上执行以下命令，查询本节点NPU的EID（输出中的字段`ipourma`的`IP Address`为可取的`eid`）：
  ```bash
  hccn_tool -g -dev_info -i <devid>
  ```
  2. 在任一节点上以对端EID为目的发起UB ping（`src_eid`为本端EID，`dst_eid`为对端EID）：
  ```bash
  hccn_tool -t -ub_ping -i <devid> -src_eid <src_eid> -dst_eid <dst_eid> [-pkt <size>] [-cnt <size>] [-interval <size>]
  ```
  其中`pkt`为ping的数据包大小（单位：字节），`cnt`为发包数量，`interval`为发送间隔。
  报文全部收到应答即节点间UB链路连通。

1.在shmem/目录编译:
```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

2.在shmem/目录运行:
```bash
bash examples/udma_demo/run.sh 0 # allgather测试
bash examples/udma_demo/run.sh 1 # put signal 测试
bash examples/udma_demo/run.sh 0 -shared_jetty 1 # allgather测试，启用shared jetty
```
默认按单机8卡启动，脚本依次拉起`PE 0`到`PE 7`，并等待所有进程退出。

`run.sh`会自动设置`PROJECT_ROOT`、`LD_LIBRARY_PATH`，并根据`ipport`参数设置`SHMEM_UID_SESSION_ID`，无需手动导出这些环境变量。

UDMA 高阶 RMA 接口默认使用 `PIPE_MTE3` 下发 WQE，需要一段 UB
scratch。默认配置为 `offset = 189 * 1024`、`ub_size = 128` 字节、
`sync_id = 0`；如果样例或业务 kernel 需要复用这段 UB，可通过
`aclshmemx_set_udma_config(offset, ub_size, sync_id)` 调整。`ub_size`
必须不小于 128 字节，用于容纳当前 UDMA 数据搬移操作的一块完整
WQE staging block。

3.run.sh脚本命令行参数说明

脚本支持一个可选的命名开关，可以出现在命令行任意位置，其余参数仍按位置解析（均可选，缺省时使用默认值）：

```bash
bash examples/udma_demo/run.sh [-shared_jetty 0|1] <test_type> <n_pes> <g_npus> <ipport> <f_pe> <local_pes> <f_npu>
```

- -shared_jetty: 可选命名开关，`1`表示启用shared jetty，`0`（默认，不传即关闭）表示关闭。开启时通过初始化参数`attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED`设置；该配置必须在初始化前设置，且所有PE必须一致。shared jetty会使解析到同一本地EID的对端PE共享底层jetty（一条SQ/CQ环和一个队列状态），开启时要求运行环境中的libhcomm.so提供channel config接口，否则初始化会失败并返回`ACLSHMEM_NOT_SUPPORTED`；同时在共享组内发送操作需由调用方串行下发。
- test_type: 测试类型，0表示运行all-gather测试（默认），1表示运行put signal测试。
- n_pes: 跨所有机器的全局PE总数（默认8）。
- g_npus: 本机使用的NPU卡数量（默认8）。
- ipport: bootstrap节点地址，格式为`IP:PORT`（默认`127.0.0.1:8899`）。必须是node0的IP，且所有节点均可达。
- f_pe: 本机使用的第一个全局PE号（默认0）。
- local_pes: 本机要拉起的PE进程数（默认等于n_pes）。
- f_npu: 本机使用的第一张NPU卡的卡号（默认0）。

多机启动示例（2机 × 8卡 = 16 PE，假设node0的IP为192.168.1.10）：
```bash
# node0：f_pe=0，拉起全局 PE 0~7
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 0 8 0
# node1：f_pe=8，拉起全局 PE 8~15
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 8 8 0
```
多机场景如需启用shared jetty，在两个节点上均追加`-shared_jetty 1`（所有节点必须一致）。

### mssanitizer 检测

执行 mssanitizer 前，使用 `-mssanitizer` 重新编译：

```bash
bash scripts/build.sh -examples -soc_type Ascend950 -mssanitizer
```

当前实验在 MTE3 路径恢复结构体字段填充，并由 all-gather 与 put signal 两个 kernel 在首次调用前对完整 128 字节 scratch 做一次性清零（`udma_demo_kernel.cpp` 中的 `init_udma_wqe_scratch`）。这既初始化结构体位域读改写可能读取的原值，也覆盖 put signal 的 32 字节尾部填充，用于验证调用方全量预初始化能否消除 initcheck 未初始化读。

```bash
mssanitizer -- bash examples/udma_demo/run.sh 0
mssanitizer -- bash examples/udma_demo/run.sh 1
```

4.底层二进制命令行参数说明
```bash
./udma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [test_type] [shared_jetty]
```

- n_pes: 全局PE数量。
- pe_id: 当前进程对应的PE号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。
- g_npus: 当前机器上启动的NPU卡的数量。
- f_pe: 当前机器上使用的第一个PE号。
- f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号。
- test_type: 测试类型（可选），0表示运行all-gather测试（默认），1表示运行put signal测试。
- shared_jetty: 是否启用shared jetty（可选），0表示关闭（默认），1表示开启；所有PE必须一致。
