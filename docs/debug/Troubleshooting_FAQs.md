# SHMEM 使用限制

1. GM2GM的高阶 RMA 操作使用默认buffer，不支持并发操作，否则可能造成数据覆盖。若有并发需求，建议使用低阶接口。
2. barrier接口当前必须在Mix Kernel（包含mmad和GM2UB/UB2GM操作）中使用，可参考example样例。该限制待编译器更新后移除。
3. 使用RDMA的高阶接口前，需要先使用`aclshmemx_rdma_config`接口配置UB Buffer和sync_id等信息，且需要保证预留UB Buffer大小大于等于128字节。若不配置，则使用默认的190KB处的UB Buffer和EVENT_ID0作为接口内部的同步EVENT_ID。RDMA相关接口内部使用`PipeBarrier<PIPE_MTE3>`阻塞MTE3流水以确保RDMA任务下发完成。RDMA 引擎专用 Put/Get 接口的 `src`、`dst` 都必须指向对称内存，且各自的完整传输范围不得越过其所在的内存分配。通用 Put/Get 接口仍保留标准 RMA 操作数语义；但通过 `ACLSHMEM_DATA_OP_ROCE` 使能 RDMA 后，运行时可能按目标 PE 自动选择 RDMA，因此调用者必须预先保证两端都指向对称内存，无需判断单次调用实际使用的引擎。
4. 使用SDMA的高阶接口前，需要先使用`aclshmemx_sdma_config`接口配置UB Buffer和sync_id等信息，且需要保证预留UB Buffer大小大于等于64字节。若不配置，则使用默认的191KB处的UB Buffer和EVENT_ID0作为接口内部的同步EVENT_ID。
5. 使用UDMA的高阶接口前，可使用`aclshmemx_set_udma_config`接口配置UB Buffer和sync_id等信息，且需要保证预留UB Buffer大小大于等于128字节。若不配置，则使用默认的189KB处的UB Buffer和EVENT_ID0作为接口内部的同步EVENT_ID。UDMA高阶接口默认通过`PIPE_MTE3` staging下发WQE，该UB Buffer用于暂存一块完整WQE。
6. A2 16卡机型：NPU分为前八卡后八卡两个8P Fullmesh组，每个8P组内通过HCCS总线完成两两互联，两个8P Fullmesh组之间通过PCIe-SW完成互连，因此不支持直接使用MTE接口在跨组NPU间完成数据搬运。部分example用例使用MTE搬运接口，单机用例请勿跨组指定NPU，以防发生未知报错（如流同步失败等）。
7. A3 D2H/D2rH等功能，需要确保Host内存（DRAM）可用空间大于aclshmemx_init_attr_t初始化过程中pe分配的local_mem_size大小。因为HCCS总线上DRAM地址范围是固定的，部分环境上并不是所有DRAM都在HCCS固定的总线地址范围内，只有和HCCS总线固定的地址交集部分才是可用的DRAM空间。确认DRAM可用空间方法：通过lsmem查询本机物理地址范围，和如下4个地址区间取交集（0x29580000000-0x34000000000， 0xa9580000000-0xb4000000000， 0x129580000000-0x134000000000， 0x1a9580000000-0x1b4000000000），得到具体可用的DRAM容量。如果没有交集，表示当前没有可用DRAM空间或可用空间小于配置的local_mem_size，则不支持该功能。

# SHMEM 常见问题

## 内存分配相关问题

### aclshmem_malloc多卡分配非对称共享内存

#### Q: 算子精度问题，无error日志，发现共享内存访问到的数据异常

![算子精度报错](./images/trouble-shooting/shmem_alloc_asymmetric_alloc_no_error.png)

错误示例代码:

以`example`目录下的`sdma`为例，以下为非对称共享内存分配的简单示例场景：

```cpp
// Inappropriate calling of aclshmem_malloc
void *symmTest = nullptr;
symmTest = aclshmem_malloc(((rank_id + 1) * 1024 * 1024) * sizeof(__fp16));

void *symmPtr = aclshmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
uint8_t *gmSymmetric = (uint8_t *)symmPtr;

... ...

aclshmem_free(symmPtr);
if (symmTest != nullptr) {
    aclshmem_free(symmTest);
}
```

#### A: 可使用debug模式排查共享内存分配对称性问题

`debug`模式开启方法：在代码仓根目录下执行：

- A2/A3 平台：`bash scripts/build.sh -examples -debug`
- Ascend950 平台：`bash scripts/build.sh -soc_type Ascend950 -examples -debug`

此时执行代码获得如下报错，确认错误为使用`aclshmem_malloc`接口分配了非对称的共享内存

![开启debug模式后错误日志](./images/trouble-shooting/shmem_alloc_asymmetric_alloc_log.png)

错误原因分析示意图:

修正方式: 确保每个rank分配相同大小的共享内存

![共享内存非对称分配](./images/trouble-shooting/shmem_alloc_asymmetric_alloc.png)

### aclshmemx_set_attr_uniqueid_args对每个pe设置了不同的local_mem_size

#### Q: 提示local size diffs

错误调用代码片段:

```cpp
aclshmemx_init_attr_t attributes;
aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;

int64_t local_mem_size = (1024 + pe * 2) * 1024 * 1024;
if (pe == 0) {
    status = aclshmemx_get_uniqueid(&uid);
}

MPI_Bcast(&uid, sizeof(aclshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
status = aclshmemx_set_attr_uniqueid_args(pe, pe_size, local_mem_size, &uid, &attributes);
status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);
```

错误日志:

![图片](./images/trouble-shooting/local_mem_size_asymmetric_alloc.png)

注意:

1. 日志中显示的实际分配大小和local_mem_size大小有`6MB`的差异为shmem框架内部使用空间
2. 此处`local_mem_size`大小为`2MB`对齐，若尝试分配其他大小如`1025 * 1024 * 1024`可能会出现不同的错误信息:

    ![图片](./images/trouble-shooting/local_mem_size_asymmetric_alloc_not_2mb.png)

#### A: 应保证aclshmemx_init_attr_t初始化过程中每pe分配的local_mem_size大小一致

## IP/PORT配置相关问题

### 绑定端口被占用

#### Q: 尝试使用的ip/port已被占用，错误日志如图

1. 端口被占用错误日志:

    ![端口被占用](./images/trouble-shooting/port_being_used.png)

2. ip不可用错误日志1:

    ![ip不可用1](./images/trouble-shooting/ip_unaccessible.png)

3. ip不可用错误日志2:

    ![ip不可用2](./images/trouble-shooting/ip_unaccessible2.png)

#### A: 逐步排查ip及端口可用情况

1. 确认ip是否符合预期
2. 检查端口是否被占用，`netstat -tuln | grep <端口号>`
3. 调整环境变量`SHMEM_UID_SESSION_ID`及实际执行文件所使用的ip及端口号

### default 模式同一端口多次初始化失败

#### Q: 使用 default 模式（`ACLSHMEMX_INIT_WITH_DEFAULT`）时，对同一端口进行多次初始化（例如在同一进程中创建多个实例，或重复调用初始化接口而未更换端口），出现端口已被占用的错误

错误日志特征：

- `address in use for bind listen on ...` 或 `ACC_LINK_ADDRESS_IN_USE`
- `startup acc tcp server on port: ... already in use.`
- `bind failed` / `bind socket fail`

#### A: default 模式下每次初始化都会绑定一个独立端口。若端口已被前一个实例占用（未 finalize），则再次初始化会因端口冲突而失败。解决方案

1. **多实例场景**：设置 `SHMEM_INSTANCE_PORT_RANGE=start:end`，并将 `ip_port` 中的端口设为 `0`（如 `tcp://127.0.0.1:0`），框架会根据 `instance_id` 自动分配 `start_port + instance_id` 作为实际端口，保证各实例使用不同的端口。
2. **单实例多次初始化场景**：确保每次初始化时使用不同的可用端口，或确保上一次初始化已完成 `finalize` 释放端口后再进行下一次初始化。

### 未通过环境变量配置ip/port，且使用自动搜索可用网口时失败

#### Q: `SHMEM_UID_SESSION_ID`和`SHMEM_UID_SOCK_IFNAME`均未配置时自动搜索可用网口（IPv4/IPv6均可，排除lo/docker/veth/br-/virbr/tun/tap等虚拟接口），查询失败时错误日志如下

![session_id_sock_ifnam_not_set](./images/trouble-shooting/session_id_sock_ifnam_not_set.png)

#### A: 应手动配置可用的`SHMEM_UID_SESSION_ID`或`SHMEM_UID_SOCK_IFNAME`

配置示例:

- SHMEM_UID_SESSION_ID:

    `SHMEM_UID_SESSION_ID=127.0.0.1:1234`

    `SHMEM_UID_SESSION_ID=localhost:8888`
- SHMEM_UID_SOCK_IFNAME:

    `SHMEM_UID_SOCK_IFNAME=[6666:6666:6666:6666:6666:6666:6666:6666]:886`

    `SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4` 取ipv4

    `SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6` 取ipv6

    `SHMEM_UID_SOCK_IFNAME=eth0` 自动探测可用协议（优先ipv4）

注意: 同时配置时只读取`SHMEM_UID_SESSION_ID`

## RDMA相关问题

### 同端口通信需开启端口桥

#### Q: 1825 网卡单机多 NPU 共用同一物理端口，RDMA 通信失败

#### A: 1825 网卡在单机多 NPU 场景下，若多个 NPU 共用同一物理端口，两个 NPU 间的 RDMA 流量会从该端口发出、经交换机后从同一端口返回。交换机默认丢弃此类**同源同宿报文**，导致通信失败。需在交换机对应接口开启**端口桥**（port bridge）功能

> 详细说明：[配置端口桥功能](https://support.huawei.com/hedex/hdx.do?docid=EDOC1100558302&id=ZH-CN_TASK_0000001176744393)

**操作步骤：**

1. 确认 NPU 所连交换机端口，首先需要了解物理连线，明确网卡各端口与 NPU 卡的对应关系（即哪些 NPU 共用同一网卡端口）。

   **常用排布参考（8卡机型）：**

   | NPU 卡 | 网卡设备 |
   |--------|----------|
   | 0、1   | hrn5_1   |
   | 2、3   | hrn5_0   |
   | 4、5   | hrn5_3   |
   | 6、7   | hrn5_2   |

   例如，若需要 NPU 6 和 NPU 7 之间通信，二者共用 `hrn5_2` 端口，需在交换机对应的 `hrn5_2` 端口上开启端口桥。

   - 通过 `ibv_devinfo` 或 `hiroce5 gids` 查看网络设备（`net_dev` 列）：

     ```bash
     ibv_devinfo
     # 或
     hiroce5 gids
     ```

     ![](../images/net_dev_query.png)

   - 通过 `ifconfig <net_dev>` 或 `ip link show <net_dev>` 查看对应接口的 MAC 地址：

     ```bash
     ifconfig <net_dev>
     # 或
     ip link show <net_dev>
     ```

     ![](../images/mac_address_query.png)

   - 根据该 MAC 地址，在交换机上查询对应端口号：

     ```bash
     display mac-address | include <mac-address>
     ```

     ![](../images/port_number_query.png)

   > 注：上述命令中的 `<net_dev>` 需替换为上一步查询到的实际网络设备名（如 `enp3s0`），`<mac-address>` 需替换为上一步查询到的实际 MAC 地址（格式如 `aabb-ccdd-eeff`）。
   >
   > 注：若需查询 NPU 6、7（共用 `hrn5_2`，需端口桥）或 NPU 5、6（跨 `hrn5_3` 与 `hrn5_2`，无需端口桥）对应的交换机端口，需先在这两组 NPU 之间各进行一次 RDMA 通信（例如运行 `rdma_demo`），使交换机学到对应的 MAC 表项，否则 `display mac-address` 可能查不到对应记录。

2. 登录交换机，进入系统视图：`system-view`
3. 进入对应接口（以 100GE1/0/1 为例，按实际替换）：

   ```bash
   interface 100GE1/0/1
   ```

4. 开启端口桥：

   ```bash
   port bridge enable
   ```

5. 提交：`commit`
6. 验证：`display current-configuration | include port bridge`，输出含 `port bridge enable` 即成功。
7. 多个接口需同端口通信时，对每个接口重复 3-5。

配置完成后，在使用相同网卡端口的 NPU 卡上运行 `rdma_demo` 即可验证通信是否正常。

### 通信丢包

#### Q: 使能RDMA(RoCEv2)后，网络出现丢包现象

#### A: 检查交换机和端侧的TC与SL配置是否正确，如果不一致会出现丢包现象。可以参考[环境变量说明](../api/env_vars_intro.md)对环境变量[HCCL_RDMA_TC](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0089.html)和[HCCL_RDMA_SL](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0090.html)进行设置

### RDMA 端口管理规则

Ascend950 RDMA V2 建链监听端口由系统自动分配，用户无需配置 RDMA 建链端口。

- 同一 RDMA IP 上启动多个进程或多个 SHMEM instance 时，无需为每个进程手工分配 RDMA 建链端口。
- RDMA 建链端口与初始化通信端口不同；初始化通信端口仍按初始化方式配置，例如 default 模式下的实例端口或 unique id 模式下的 `SHMEM_UID_SESSION_ID`。
- 若建链失败，请优先检查 RDMA 网卡连通性、GID index、交换机 PFC/ECN 等网络配置。

### 建链失败检查

#### Q: RDMA 建链报错 `HcommChannelCreate failed: 19`（HCCL_E_NETWORK）

#### A: 该错误通常为 RDMA QP 状态迁移（INIT→RTR）超时，与端口分配无关。请依次检查

1. 确认 GID index 两端一致：日志中 `gid_idx` 字段
2. 确认 RDMA 网卡间 IP 层可达：`ping <对端RDMA_IP>`
3. 检查交换机 PFC/ECN 无损网络配置

### 卡间同步Device侧执行卡死

#### Q: 在算子代码中调用RDMA卡间同步接口 `aclshmemx_roce_sync_all, aclshmemx_roce_barrier_all, aclshmemx_roce_barrier, aclshmemx_roce_team_sync`出现某几个PE执行卡死

#### A: 确保当前调用上述接口的算子只在AIV上执行

### 同步超时上报 SHMEM_SYNC_TIMEOUT

#### Q: 同步报出 `SHMEM_SYNC_TIMEOUT`，如何解读和处理？

#### A: 同步接口的跨 PE 信号等待及卡内软件汇合支持超时检测，默认关闭，需显式设置阈值开启

在初始化属性中设置 `attr.runtime_config.sync_timeout_us`，单位为微秒。例如 `attr.runtime_config.sync_timeout_us = 5000000` 设置为 5 秒，随后调用初始化接口并检查返回值。0 表示未指定：首个实例关闭超时检测，后续实例继承进程中已冻结的生效值。显式值的有效范围为 `[1, ACLSHMEM_SYNC_TIMEOUT_MAX_US]`，字段类型为 `uint32_t`，上限为 `UINT32_MAX` 微秒（约 71 分 35 秒），赋值前应确保原始值不超过该范围。实例存活期间，新实例请求冲突的显式阈值返回 `ACLSHMEM_NOT_SUPPORTED`。初始化失败不提交新配置；最后一个实例释放后恢复关闭状态。每个 PE 分别填写该属性。该配置不支持 RoCE 同步超时上报，也不修改 RoCE 后端已有超时和错误处理；无法中断硬件 `SyncAll`，库与设备内核须一起重新编译。每核从进入同步起独立计时，后续阶段共用起始时刻；分批检查失败轮询，因此阈值不是严格的退出时间上界，内部检查间隔不打印。

| 字段 | 含义 |
| --- | --- |
| instance、team、pe | 实例编号、该实例内的 team handle、上报者在该实例内的 world PE；非多实例构建的 instance 为 0。 |
| core_type、core | 1=AIV、2=AIC、0=未知；core 是 GetBlockIdx() 的逻辑编号，不是物理核编号。 |
| generation、round、slot、peer | team 同步轮次、从 0 开始的内部 round、信号槽位、等待或发送的 world PE。 |
| expected、observed | 期望与实际信号。 |
| elapsed_us、timeout_us | 从该核进入同步起已等待时间和阈值，单位均为微秒。 |
| algorithm、phase | algorithm 是内部同步诊断标识；独立软件卡内汇合为 0；phase=0/1/2/3/4 分别表示进入时汇合、同步轮次发布汇合、阶段结束汇合、发送、等待信号。 |
| wait_core、local_round | 软件汇合等待的逻辑 AIV 编号和本地 round，此时 peer 为本 PE。集合同步内部汇合保留上层算法编号。 |

**phase 速查：日志只打印数字，不打印枚举名称。** 这些枚举是诊断标记，不参与同步信号判定。

| 日志值 | 代码枚举 | 执行位置 |
| --- | --- | --- |
| `phase=0` | `LOCAL_ENTER` | 进入同步时的卡内汇合。 |
| `phase=1` | `LOCAL_PUBLISH` | AIV0 更新共享同步轮次 counter 后的卡内汇合；汇合后各 AIV 读取该轮次，不是向其他 PE 发送信号。 |
| `phase=2` | `LOCAL_EXIT` | 同步退出前或内部 round 结束时的卡内汇合。 |
| `phase=3` | `SEND` | 发送同步信号前设置的标记；不能据此保证阻塞的硬件发送会被超时检查中断。 |
| `phase=4` | `WAIT` | 等待其他 PE 的同步信号。 |

建议按以下顺序读日志：

1. 用 `instance/team/pe/core_type/core` 确认哪个参与者上报；例如 `core_type=1 core=0` 表示逻辑 AIV0。
2. 用 `algorithm/phase` 确认算法和位置；例如 `algorithm=4 phase=1 wait_core=3` 表示内部算法标识为 4，在轮次发布后的**软件卡内汇合**中等待逻辑 AIV3 超时。`peer` 此时是本 PE，不是 PE3。
3. 跨 PE 等待时看 `peer`，软件汇合时看 `wait_core/local_round`；再结合 `generation/round/slot` 定位轮次和槽位。
4. 对照 `expected/observed` 和 `elapsed_us/timeout_us`，确认信号差异及已等待时间。`generation` 是 team 同步轮次，`round` 是该次同步内部的算法步骤，两者不同。

不适用或未知字段为 -1。下面是显式设置 60 秒阈值后的格式示例，不是实际故障测试记录：

```text
SHMEM_SYNC_TIMEOUT instance=0 team=0 pe=0 core_type=1 core=1 algorithm=3 phase=4 generation=1 round=0 slot=0 peer=1 wait_core=-1 local_round=-1 expected=1 observed=0 elapsed_us=60000508 timeout_us=60000000
```

示例表示 PE0 的 AIV1 等待 PE1 的信号 1，但实际仍为 0，等待约 60.000508 秒后超时。round 标识内部同步步骤；同步轮次发布前 generation 可能未知。打印后设备执行 `trap()`，不会作为同步成功返回。主机应检查 stream/device 同步错误，退出或重建通信上下文，不要继续使用已失配的状态；具体错误编号和上报耗时取决于运行时。

先核对所有参与者是否使用一致的 team、轮次及调用顺序，排查提前返回或异常退出。根据 peer、wait_core 定位缺席参与者。软件轮询检查不能中断已阻塞的硬件 SyncAll、事件等待或内存访问。RoCE 接口、独立 quiet、普通点对点等待保留原行为；软件汇合通过不代表硬件汇合具备缺核超时能力。完整字段说明见 [DEVICE API 文档](../api/device_api.rst)。

## 调试相关

### 编译问题

#### Q:自行添加"-O0 -g"编译选项调试，编译出错，"bisheng: error: xxxxx will be ignored. [-Werror -Woption-ignored]"

#### A:SHMEM根目录的CMakeLists.txt的-Werror选项导致编译器告警被作为错误处理，注释掉-Werror编译选项即可解决

### 算子问题

#### Q:算子使用"-O0 -g"编译选项编译后，运行出错，"min stack size is xxx, larger than current process default size 32768. Please modify aclInit json, and reboot process."

#### A:在aclInit()接口传入的json文件中，配置更大的栈空间

[配置参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0145.html)
