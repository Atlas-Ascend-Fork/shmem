# 动态 metadata 设计与接入

## 功能与构建模式

SHMEM 默认将 metadata 放在固定地址。开启设备符号能力后，Runtime 自动分配一块 32 MiB 内存。
每个设备二进制中的 `g_meta_root` 保存这块内存的地址。同一进程内的 SHMEM 多实例共用这块内存，
继续通过现有实例选择接口访问各自的 state。

当前支持编译器管理设备镜像的混编算子，不提供自定义 binary loader 接入。
构建维护者负责开启功能、调整链接配置并重新编译算子。普通 kernel 作者继续使用原 SHMEM 接口，
无需增加参数或调用注册方法。

| 项目 | 固定地址模式（FIXED，默认） | 动态地址模式（ROOT） |
| --- | --- | --- |
| 构建参数 | 无新增参数 | `-enable_device_symbol` |
| metadata 寻址 | 固定地址 | 读取设备变量 `g_meta_root` |
| 混编算子链接 | 保持原配置 | 增加同包的 `libshmem_device.a` 和设备链接选项 |
| 新增 Runtime 依赖 | 无 | `aclrtGetSymbolAddress` |

**动态模式的 Runtime 接口要求为 CANN 9.1.0 正式版及以上。** 9.1.0-beta.1～beta.3
不满足当前方案要求。公开 Runtime 仓的 `v9.1.0.pre` 已包含该接口，正式发布基线采用 `v9.1.0`；
该 tag 中可核对接口的[公开声明](https://gitcode.com/cann/runtime/blob/v9.1.0/include/external/acl/acl_rt.h#L2403)、
[导出包装](https://gitcode.com/cann/runtime/blob/v9.1.0/src/acl/aclrt_impl/acl_rt_wrapper.h#L333)和
[实现](https://gitcode.com/cann/runtime/blob/v9.1.0/src/acl/aclrt_impl/memory.cpp#L2113)。
`v9.1.0-beta.2` 已提供 `aclrtBinaryGetGlobal`，但尚无 `aclrtGetSymbolAddress`，不能据此前者判断兼容。
当前方案使用 `aclrtGetSymbolAddress + aclrtMemcpy`，不依赖 `aclrtMemcpyToSymbol` 或 `aclrtBinaryGetGlobal`。

以上是运行时接口的最低版本要求，不替代目标芯片的 CANN 配套要求和编译器能力要求。
本功能已完成实测的环境为 A5 / Ascend950PR / CANN 9.2.0-beta.2；CANN 9.1.0 的接口支持由公开源码确认，
尚未完成该版本的真机验证。固定地址模式不新增此版本要求。

```bash
bash scripts/build.sh -soc_type Ascend950 -enable_device_symbol
```

直接使用 CMake 时，设置 `-DACLSHMEM_ENABLE_DEVICE_SYMBOL=ON`。
通过 `build.sh` 构建时，不传开关会恢复固定地址模式，包括复用旧构建目录的情况。

显式开启功能后会告警提示重编算子和部署要求，不按构建机的 ACL 接口声明拦截。
部署时从实际加载的 ACL 库解析接口；缺少接口会导致初始化失败，不会自动回退到固定地址模式。
编译器仍须支持目标 SoC、GM 全局变量和设备链接，开关不能补足旧工具链缺少的能力。

源码中的 `include/host_device/shmem_common_macros.h` 默认设置 `ACLSHMEM_DYNAMIC_META=0`。
CMake 用编译宏选择仓内目标的模式，安装时生成同名配置副本，记录库的模式。
外部仓使用安装包的头文件即可继承配置。直接使用源码头或手动设置宏时，须与所链接的库保持一致。
**Host、SIMD 和 SIMT 必须使用同一模式，固定地址算子不能与动态地址库混用。**
切换到动态模式时，必须重新编译 SHMEM，并使用新安装包重新编译所有使用 SHMEM Device 接口的算子包
（包括 C++ 算子库、Torch 扩展及其独立 kernel 库）。不能只替换 `libshmem.so` 而继续使用旧算子二进制。
清理旧模式的构建产物和相关 JIT 缓存后重新构建；切回固定模式也遵循相同要求。

## 混编算子接入

使用 bisheng `<<<>>>` 启动 kernel 的工程，由编译器管理设备镜像。仓内 examples 已自动接入；
外部仓将以下配置加入自己的构建文件即可，无需携带注册源码。

示例假设 `operator_kernel` 已配置 `-xasc`、匹配的 `NPU_ARCH`，以及原有 SHMEM 头文件和库路径。
`SHMEM_ROOT` 指向安装包，`shmem` 为同包库目标。
头文件路径应包含安装包的 `include`、`src` 和需要的 Device/SIMT 目录。

```cmake
file(STRINGS "${SHMEM_ROOT}/include/host_device/shmem_common_macros.h" shmem_root_enabled
    REGEX "^#define ACLSHMEM_DYNAMIC_META 1$")
if(shmem_root_enabled)
    target_compile_options(operator_kernel PRIVATE -dc)
    target_link_options(operator_kernel PRIVATE
        --cce-fatobj-link --npu-arch=${NPU_ARCH})
    target_link_libraries(operator_kernel PRIVATE
        "${SHMEM_ROOT}/lib/libshmem_device.a"
        shmem)
endif()
```

OBJECT/STATIC 中间设备目标也要启用 `-dc`。每个最终设备二进制链接一次静态库。
静态库须与同包 SDK、SoC 和设备编译 ABI 配套；wheel 使用所选后端的
`backends/<backend>/libshmem_device.a`。固定地址模式下，上述条件块不增加依赖。

普通链接不需要 `--whole-archive` 或 `-u`。`include/host_device/shmem_common_macros.h`
仅在动态模式的混编 Host 阶段引用 `src/host_device/shmem_device_symbol.h`。
该实现头声明隐藏链接锚点 `aclshmemx_device_library_anchor`，并通过保留引用带入静态库中的模块对象。
公共头不直接声明该入口；实现头随 SDK 安装到 `src/host_device`，不纳入自动接口文档。
模块对象再引用 Host 注册对象，保证编译器先登记设备变量，SHMEM 再登记入口。
这个隐藏的构建入口无需用户调用；注册、注销和地址发布方法保留在 `src/` 内部实现。
完成上述编译和链接配置后，模块登记自动完成，接入者无需编写初始化函数或管理内部注册顺序。

**使用动态模式时，必须在 SHMEM 初始化前加载所有相关算子库；本版本不支持初始化后再加载新的
SHMEM 算子库。** 提前加载时只登记符号，地址发布失败由 `aclshmemx_init_attr` 返回；调用方须检查
返回值，失败后不得启动 SHMEM kernel。现有代码仍保留晚加载发布路径，但不作为支持承诺：
若其符号查询或地址拷贝失败，构造函数打印原因并调用 `abort()`，无法通过普通 `dlopen` 返回码处理。

### 模块卸载与重新加载

动态模式不支持在同一进程中反复真正卸载、重新加载同一个算子 `.so`。正常使用应加载一次并保持
模块驻留，反复调用其中的 kernel。检测到曾注销的 Host 符号再次登记时，SHMEM 输出
`META_MODULE_RELOAD_SUSPECT` warning；这是排查提示，不是可靠的重载检测或恢复机制。

在已验证的 CANN 9.2.0-beta.2 环境，同一进程内真正 `dlclose` 算子模块后再 `dlopen`，
可能复用旧的 Host 符号映射。若 Host 查询和新 kernel 使用的设备变量地址不同，会导致地址发布失效；
现有日志尚未独立证实这种地址错配。
反复启动 kernel、模块驻留期间的 SHMEM finalize/init，以及退出后重启进程，不属于这种情况。

存在卸载重载的工程，推荐给对应算子模块添加以下选项。普通接入不强制添加：

```cmake
target_link_options(operator_kernel PRIVATE -Wl,-z,nodelete)
```

此选项让模块驻留至进程退出，`dlclose` 不立即回收模块代码和全局变量。
metadata 仍按 SHMEM 生命周期管理：最后一个实例释放时清零入口并释放内存，重新初始化时发布新地址。

Runtime 主线已有 [Program 释放时清理变量符号的修复](https://gitcode.com/cann/runtime/commit/b372e14919c5ba4c3ae412b2d213da934b04568c)。
但复现日志未显示 Host `dlclose` 执行了对应的 binary unload，因此尚不能确认该修复解决了混编模块重载。
`nodelete` 是当前环境已验证的规避方式，不代表支持真正卸载重载。

## 地址发布与生命周期

每个设备二进制有一个 8 B 的 `g_meta_root`，共同指向本地 metadata 内存。
混编模块提供编译器登记的 Host 变量标识，SHMEM 用 `aclrtGetSymbolAddress` 查询设备变量地址。
随后通过 `aclrtMemcpy` 写入 metadata 地址；SIMD 直接读取 `g_meta_root`，SIMT 使用 `asc_ldcg`。

metadata 归属于初始化时的设备。多实例共享这块内存，切换实例不会覆盖 root。
发布和释放检查当前 device ID，不保存、比较或切换初始化时的 Context。

1. 加载所有算子库，设置目标设备后初始化 SHMEM，成功后再启动使用 SHMEM 的 kernel。
2. finalize 前停止提交相关 kernel，并完成通信所需的 quiet 及所有相关 stream/graph 的同步。
3. 同步、清零或释放失败时，finalize 返回失败。保持设备运行环境，排除故障后再次调用 init/finalize，
   重试待清理资源。初始化失败也遵循此约束；新一轮 init 成功前不得恢复 SHMEM kernel 执行。

graph capture 前须完成模块加载、登记和 SHMEM 初始化。capture/replay 期间，模块、metadata、实例 state
和业务 heap 均须有效。跨 finalize 的旧 graph 必须重新构建。

## 日志定位

启动进程前设置以下变量，可查看登记和地址发布日志：

```bash
export SHMEM_LOG_LEVEL=DEBUG
export SHMEM_LOG_TO_STDOUT=1
```

日志级别使用大写名称，只看告警可设为 `WARN`。未开启标准输出时，日志默认写入 `~/shmem/log`，
可用 `SHMEM_LOG_PATH` 修改。也可在每次初始化成功后调用 `aclshmemx_set_log_level(0)` 开启 DEBUG。

| 日志标记 | 级别 | 含义 |
| --- | --- | --- |
| `META_MODULE_RELOAD_SUSPECT` | WARN | 已注销的 Host token 再次登记，提示检查模块重载和旧映射 |
| `META_BIND_SYMBOL` | DEBUG / ERROR | Host token、查询所得设备变量地址、root；失败时附接口和返回码 |
| `META_PUBLISH` | DEBUG / ERROR | 发布或清零时的 device ID、root、模块数量；失败返回码 |
| `META_DEVICE_QUERY` / `META_DEVICE_MISMATCH` | ERROR | 设备查询失败或与 metadata 所属设备不符 |

重载提示保留最近 64 次注销的 token，普通 finalize/init 不触发提示。
不同模块复用 Host 地址可能触发提示；token 改变或历史淘汰也可能漏报。因此，告警本身不能证明地址错配。
DEBUG 中的设备变量地址来自 Host 查询，仍须结合 Runtime plog 的
`already registered, skip duplicate registration` 和 Program 加载信息排查。
`507035` 或 GM 访问异常也不能单独证明发生了旧映射问题。

## 共享 heap 与固定 metadata

本功能只调整 metadata 的分配和寻址。共享 heap 的预留地址、连续布局、映射及释放沿用 `dev` 原有实现，
不改变 ACL/Fabric、VMM 或旧驱动路径的分配策略。

固定 metadata 保留原有的地址和预留 flag，不处理固定地址被占用的问题。
动态 metadata 由 `aclrtMalloc` 选择可用地址，不依赖预设地址空闲。

## 验证方法与范围

回归用例已接入 gtest 套件 `DynamicDeviceAddress`，覆盖初始化前模块登记、SIMD/MIX/SIMT、
同设备不同 Context 的符号查询与 kernel 读取、finalize 后入口清零、多实例、通信、Host 占用旧 meta 地址和重新初始化，
以及 Host 故障恢复和打包检查。

```bash
bash scripts/build.sh -soc_type Ascend950 -uttests -enable_device_symbol
bash scripts/run.sh -ranks 2 -gnpus 2 -fnpu 2 -test_filter 'DynamicDeviceAddress.*'
```

示例使用 2、3 号卡，运行前替换为实际空闲的连续两张卡。设备测试要求单机双 PE；
动态模式预期 4 项通过，固定地址模式预期 4 项 skipped。Host 夹具依赖 Python3、C++ 编译器和 setuptools。
examples 使用 `-examples -enable_device_symbol` 构建，再按各自 README 验证。

本次移植以 `dev` 为基线，包含 SIMD 和 SIMT 多实例的动态 metadata 寻址。
本次已通过 CANN 9.2.0-beta.2 下动态、固定两种模式的库及完整 UT 编译，
安装产物的 umbrella、直接 Device 头文件和 SIMT 三种 `.a` 链接检查，以及纯 Host 编译检查。
注册故障恢复、meta 清理重试和 wheel 打包的 Host 夹具也已通过。
A5 环境当前 `npu-smi` 无法正常查询设备，本次尚未完成设备运行回归。
历史 master 版本在 A5 / Ascend950PR / CANN 9.2.0-beta.2 上完成过模块发布、
双 PE MTE 通信、重新初始化及 Host 占用旧 meta 地址验证，记录见
[PR #953](https://gitcode.com/cann/shmem/pull/953)。这些历史结果不替代本次 dev 分支验证。
A2/A3 真机、真实旧 SDK、跨编译器 archive 和完整多 SoC wheel 尚未完成验收。
