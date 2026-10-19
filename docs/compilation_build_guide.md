# 编译与构建

## SHMEM编译

### 下载SHMEM源码

```shell
git clone https://gitcode.com/cann/shmem.git
```

您可自行选择需要的分支。

### 编译

进入shmem的根目录，编译

```shell
cd shmem
# A2/A3 平台
bash scripts/build.sh
# Ascend950 平台
bash scripts/build.sh -soc_type Ascend950
```

更多命令介绍可查看shmem仓主目录下的`README.md`和`scripts/build.sh`文件。

### SHMEM编译相关说明

SHMEM的基本编译命令是`bash build.sh`，默认构建模式下生成版本信息，并创建安装包(默认情况下不会编译RDMA能力、示例、测试、python接口)。后可跟参数，实现不同功能：

- `-use_cxx11_abi1`：启用 `C++11 ABI`，默认
- `-use_cxx11_abi0`：禁用 `C++11 ABI`
- `-cann`：CANN 8.5以上版本可以使用CANN开放接口编译
- `-uttests`：构建`tests/unittest`目录下所有ut用例
- `-examples`：构建examples目录下所有用例
- `-python_example`：对部分examples目录下用例提供torch接入能力
- `-enable_rdma`：构建并启用RDMA相关能力。A2/A3 默认配置 RDMA 后端类型，Ascend950 需要配合 `-rdma_backend` 来指定后端类型。
- `-rdma_backend`：指定RDMA后端类型（A2/A3 不支持该选项），可选值为 `XSCALE`（使用云脉网卡）或 `HNS_1825`（使用 1825 网卡）。**必须配合 `-enable_rdma` 使用**，否则会报错。参数顺序不限。
- `-enable_ascendc_dump`：启用`AscendC_Dump`模式，用于对算子内核代码进行调测
- `-package`：构建交付安装包，同时生成 run 包和 Python wheel（包含 `-python_extension` 行为）
    * 生成 run 包 `SHMEM_{version}_linux-{arch}.run`，路径为 `{project_root}/package/{arch}/`
    * 生成同时包含 910/950 后端的 Python wheel `cann_shmem-*.whl`，路径为 `{project_root}/package/{arch}/`
- `-python_extension`：生成同时包含 910/950 后端的 Python wheel `cann_shmem-*.whl`，路径为 `{project_root}/dist/`
- `-gendoc`：生成文档
- `-onlygendoc`: 生成文档，不构建源码
- `-debug`：设置构建类型为 `Debug` 模式
- `-mssanitizer`：example启用mssanitizer内存检测工具，脚本执行需用mssanitizer拉起任务才能实际生效，allgather样例运行脚本提供了tool选项使用工具。工具使用方法参见[异常检测工具（msSanitizer，MindStudio Sanitizer）](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/optool/atlasopdev_16_0039.html) 注：如果开启该选项请确保使用mssanitizer工具拉起算子，如使用其他方式拉起算子可能导致未知错误！
- `-soc_type`：如果SOC是Ascend950类别，需要增加`-soc_type Ascend950`参数，可以通过`npu-smi info`命令查看，其他SOC可不加该参数。
- `-enable_simt`：使能simt编程模式接口。
- `-enable_relay`：构建并启用 UDMA relay（绕路/detour）能力。relay 构建在 UDMA 之上，仅 Ascend950 提供 UDMA，因此该选项**必须配合 `-soc_type Ascend950`** 使用；在多 SOC 的 wheel/包构建（`-python_extension` / `-package` / `-full`）中也可使用，此时 relay 仅应用于 950 后端。详见下文[Relay参数使用说明](#relay参数使用说明)。
- `-multi_link`：构建并启用 UDMA 多链路能力，支持按 QP 配置网络层路由策略。仅支持 Ascend950 的直连 UDMA，不能与 `-enable_relay` 同时使用。详见下文[Multi-link参数使用说明](#multi-link参数使用说明)。
- `-clean`：构建前删除已有 `build` 目录，用于避免 CMake 缓存和历史构建产物影响本次配置。
- `-full`：编译 package + python_extension + uttests + examples。SHMEM自动编译脚本，会自动完成依赖库的下载，工程编译，UT用例编译，库打包。

### RDMA参数使用说明

若需要使用 RDMA 功能，需要在编译时开启 `-enable_rdma` 参数。之后用户需要根据服务器类型与使用的网卡种类进行参数配置来配置后端

- A2/A3 平台
  - 仅需要使用 `-enable_rdma` 参数，不需要额外参数，后端使用默认配置
- Ascend950 平台
  - 使用 `-enable_rdma` 参数
  - 显式指定 `-soc_type Ascend950`
  - 指定 `-rdma_backend` 参数，否则构建脚本将会报错

**参数依赖规则：**

- `-rdma_backend` 参数在使用时**必须**同时指定 `-enable_rdma` 参数
- `-rdma_backend` 参数仅在 `-soc_type Ascend950` 时有效，否则构建脚本将会报错
- 如果指定了 `-rdma_backend` 但未指定 `-enable_rdma`，构建将失败并提示错误信息
- `-rdma_backend`, `-enable_rdma`, `-soc_type xxx` 三个参数顺序不限，但所有依赖参数必须最终都被指定

**有效命令示例：**

```shell
# Ascend950 平台启用RDMA并指定后端为 XSCALE（参数顺序示例1）
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE

# Ascend950 平台启用RDMA并指定后端为 HNS_1825
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend HNS_1825

# Ascend950 平台启用RDMA并指定后端为 XSCALE（参数顺序示例2）
bash scripts/build.sh -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE

# Ascend950 平台启用RDMA并指定后端为 XSCALE（参数顺序示例3）
bash scripts/build.sh -rdma_backend XSCALE -enable_rdma -soc_type Ascend950

# A2/A3 启用RDMA（不需要指定backend）
bash scripts/build.sh -enable_rdma
```

**无效命令示例：**

```shell
# 错误：指定了-rdma_backend但未启用-enable_rdma
bash scripts/build.sh -soc_type Ascend950 -rdma_backend XSCALE
# 将报错：Error: -rdma_backend requires -enable_rdma to be specified.

# 错误：在非Ascend950平台使用-rdma_backend
bash scripts/build.sh -enable_rdma -rdma_backend XSCALE
# 将报错：Error: -rdma_backend can only be specified when SOC_TYPE is Ascend950.
```

**RDMA 后端类型说明：**

`-rdma_backend` 参数支持以下后端类型：

- `XSCALE`：云脉网卡后端（仅 Ascend950 支持）
- `HNS_1825`：1825 网卡后端（仅 Ascend950 支持）

**⚠️ 重要提示：**

1. **自动生成的编译定义**：
   - CMake 会根据 `-rdma_backend` 的值自动生成以下编译定义，用户不应手动设置：
     - 当 `-rdma_backend XSCALE` 时，自动添加 `-DACLSHMEMI_RDMA_K_BACKEND_XSCALE=1`
     - 当 `-rdma_backend HNS_1825` 时，自动添加 `-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1`
     - 未指定后端时，不添加上述后端宏，设备侧默认选择 `IN_DIE`
   - 手动定义这些宏可能导致编译冲突或功能异常

2. **禁止手动定义的内部宏**：
   - 用户不应手动添加名为 `ACLSHMEMI_K_RDMA_BACKEND` 的编译定义
   - 这是一个设备端代码使用的内部宏，由系统在 `src/device/gm2gm/engine/shmem_device_rdma.hpp` 中根据上述自动生成的定义进行设置
   - 手动定义此宏会导致编译错误或运行时功能异常

### Relay参数使用说明

relay（绕路/detour）能力允许将通信流量在直发路径之外，经由其他 PE 绕路转发，用于提升特定拓扑下的带宽利用率（例如 `tp_allreduce_udma` 样例的 tailcut 模式）。relay 构建在 UDMA 传输之上，而 UDMA 仅在 Ascend950 上提供，因此使用 `-enable_relay` 时需满足对应的平台依赖。

启用 `-enable_relay` 后，CMake 会自动添加 `ACLSHMEM_RELAY_SUPPORT` 编译宏，用户无需手动定义。

**参数依赖规则：**

- 单 SOC 构建时，`-enable_relay` **必须**同时指定 `-soc_type Ascend950`，否则构建脚本会报错。
- 在多 SOC 的 wheel/包构建（`-python_extension` / `-package` / `-full`）中允许使用 `-enable_relay`，此时脚本会自动只对 950 后端启用 relay，910 后端不启用。
- `-enable_relay` 与 `-soc_type Ascend950` 参数顺序不限。

**有效命令示例：**

```shell
# Ascend950 平台构建 examples 并启用 relay
bash scripts/build.sh -examples -soc_type Ascend950 -enable_relay

# 参数顺序调换同样有效
bash scripts/build.sh -enable_relay -soc_type Ascend950 -examples

# 多 SOC wheel 构建，relay 仅应用于 950 后端
bash scripts/build.sh -python_extension -enable_relay
```

**无效命令示例：**

```shell
# 错误：在非 Ascend950 平台（未指定 -soc_type Ascend950）使用 -enable_relay
bash scripts/build.sh -examples -enable_relay
# 将报错：Error: -enable_relay requires UDMA support, which is only enabled for Ascend950.
```

> 说明：tailcut 等依赖 relay 的样例功能，必须在启用 `-enable_relay` 的前提下编译，详见 `examples/tp_allreduce_udma/README.md`。

### Multi-link参数使用说明

Multi-link 用于为同一对端 PE 的不同 UDMA QP 选择网络层路由。启用后，应用可以通过
`aclshmemx_set_qp_net_layer` 为指定 QP 设置显式网络层策略；未设置策略的 QP 使用默认路由选择。
当前仅支持 UDMA 的 layer 0 和 layer 1。实际可用链路由运行时 rootinfo 和拓扑决定：同一 layer-0
网络实例内可选择 MESH 或 CLOS；跨实例对端只会从可达的 CLOS 路由中选择。

启用 `-multi_link` 后，CMake 会自动添加 `ACLSHMEM_MULTI_LINK_SUPPORT` 编译宏，用户无需手动定义。

**参数依赖与限制：**

- `-multi_link` **必须**配合 `-soc_type Ascend950` 使用。
- multi-link 使用直连 UDMA，**不能**与 `-enable_relay` 同时使用。
- multi-link 不依赖 `-enable_rdma` 或 `-rdma_backend`；后两者用于 RoCE/RDMA 后端配置。
- 构建启用 multi-link 后，仍需由运行环境提供可用的 UDMA 通道和 rootinfo/拓扑信息；编译成功不表示所有 PE
  或所有网络层均可建立路由。
- 使用 `aclshmemx_set_qp_num` 和 `aclshmemx_set_qp_net_layer` 时，必须在 SHMEM 初始化前调用；所有参与 PE
  必须配置相同的 QP 数和相同的显式 QP 网络层策略。最后一个 SHMEM 实例析构后，这些配置会恢复默认值。
- 显式策略只对同一 layer-0 网络实例内的对端生效；跨实例通信会忽略显式 MESH/layer 策略并选择可达的 CLOS
  路由。因此，同一个 QP 面向不同对端最终使用的网络层可能不同。

**有效命令示例：**

```shell
# 编译 Ascend950 多链路 UDMA 能力及示例
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link

# 可与 RDMA 后端能力一同构建；仅限配备 XSCALE 网卡的 Ascend950 环境，multi-link 本身不依赖 RDMA
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link -enable_rdma -rdma_backend XSCALE
```

**无效命令示例：**

```shell
# 错误：multi-link 仅支持 Ascend950
bash scripts/build.sh -examples -multi_link
# 将报错：ACLSHMEM_MULTI_LINK_SUPPORT requires SOC_TYPE=Ascend950.

# 错误：multi-link 要求直连 UDMA，不能与 relay 同时启用
bash scripts/build.sh -examples -soc_type Ascend950 -multi_link -enable_relay
# 将报错：ACLSHMEM_MULTI_LINK_SUPPORT currently requires direct UDMA; disable relay support.
```

运行时 QP 配置和多链路验证示例请参考 `examples/udma_qp_demo/README.md`；该样例支持通过
`-qp_net_layers qp_idx:netLayer[,qp_idx:netLayer...]` 指定 QP 策略，并可使用
`aclshmemx_get_qp_net_layer` 查询已建立路由的实际网络层。

## SHMEM关键文件介绍

### `scripts`目录

   - `install.sh`: 安装脚本
   - `uninstall.sh`: 卸载脚本
   - `build.sh`: 编译脚本
   - `release.sh`：全自动构建与打包脚本
   - `set_env.sh`：SHMEM的环境变量设置文件
   - `preinstall_check.sh`：芯片、版本、拓扑、通信能力和包产物环境检测脚本

### run.sh脚本使用

UT（单元测试）测试用例运行脚本。

```sh
bash scripts/run.sh
```

提供多种参数支持自定义用例执行

```sh
-ranks          # 总rank数
-frank          # 该服务器第一个rank
-ipport         # host:port（脚本自动拼接 tcp://）
-fnpu           # 每个服务器起的第一个npu
-gnpus          # 单机使用的卡数
-test_filter    # gtest_filter

# 例如
bash scripts/run.sh -ranks 4 -fnpu 2 -gnpus 4 -test_filter ScalarP # 会在2-6卡
bash scripts/run.sh -ranks 8 -ipport 127.0.0.1:8666 -test_filter Init

```

### install.sh

打包生成的run包的安装/卸载脚本，提供安装和卸载的功能。

- `软件包名.run --check`：仅执行环境和包能力检查。
- `软件包名.run --install --check`：先执行环境检查，再安装。
- `软件包名.run --install`：直接安装。
- `软件包名.run --install-path=/absolute/path`：安装到指定绝对路径下的 `shmem` 目录。

`--check` 输出的 `WARN` 用于提示不满足或需要人工确认的能力。当前检测以输出内容和最终汇总为判断依据，命令退出成功不代表所有可选通信能力均通过。

安装目录

```sh
${INSTALL_PATH}
    |--shmem
        |--latest
        |--${version}
            |--shmem
                |--include  (头文件)
                |--lib      (so库)
            |--scripts      (卸载脚本)
```

注：`INSTALL_PATH`表示用户设置的安装目录。

### uninstall.sh

卸载脚本，可以卸载对应路径安装的shmem库或通过run包的--uninstall卸载默认路径下的shmem。

### release.sh

出包脚本，编译后使用。对编译产物打包后会删除install目录下其他文件。推荐使用build.sh完成打包。

### 编译文件`build.sh`

文件名：`scripts/build.sh`
SHMEM编译文件，一般无需更改。

### 环境变量设置文件`set_env.sh`

​**文件名**​：`scripts/set_env.sh`
SHMEM安装完成后，提供进程级环境变量设置脚本`set_env.sh`，以自动完成环境变量设置，用户进程结束后自动失效。
