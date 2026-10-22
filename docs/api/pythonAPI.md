# SHMEM Python API Reference

## shmem.core API

### 对外接口

1. **get_version** — 获取当前库版本。返回 ACLSHMEM 库的版本信息。

    ```python
    def get_version() -> str
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|版本信息组成的字符串，格式为 ``"libaclshmem_version=X.Y"``|

2. **get_unique_id** — 生成用于 UID 初始化的唯一 ID。应由单个进程（如 rank 0）调用，并通过广播分发给其他进程。

    ```python
    def get_unique_id(empty: bool=False) -> bytes
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |empty|[in]|预留参数，无实际意义|
    |返回值|[out]|原生唯一 ID 的序列化 ``bytes``，可直接通过对象广播分发；生成失败时引发 ``AclshmemError``|

3. **init** — 使用唯一 ID 初始化 ACLSHMEM 运行时。这是一个集合（collective）操作，所有 PE 必须调用。

    ```python
    def init(device: int=None, uid: Optional[Union[UniqueID, bytes]]=None, rank: int=None, nranks: int=None, mpi_comm=None, initializer_method: str="", mem_size: int=None, instance_id: int=0) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |device|[in]|预留参数，无实际意义|
    |uid|[in]|用于初始化的唯一标识符，必填|
    |rank|[in]|当前进程在 ACLSHMEM 作业中的排名（0-based），必填|
    |nranks|[in]|参与 ACLSHMEM 作业的总进程数，必填|
    |mpi_comm|[in]|预留参数，无实际意义|
    |initializer_method|[in]|指定初始化方法，必须为 ``"uid"``|
    |mem_size|[in]|每个 PE 分配的对称内存大小（字节），必填|
    |instance_id|[in]|实例 ID，必须为 0 到 254 范围内的非 bool 整数，默认 0|
    |返回值|-|无返回值。参数缺失引发 ``AclshmemInvalid``，初始化失败引发 ``AclshmemError``|

4. **finalize** — 销毁 ACLSHMEM 运行时，释放所有资源。每个进程在完成所有 ACLSHMEM 操作后应调用一次。

    ```python
    def finalize(instance_id: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |instance_id|[in]|不传时销毁当前活动实例；传入时按 ID 销毁指定实例，ID 必须为 0 到 254 范围内的非 bool 整数|
    |返回值|-|无返回值。ID 非法时引发 ``AclshmemInvalid``，销毁失败时引发 ``AclshmemError``|

5. **buffer** — 分配一个由 ACLSHMEM 支持的 NPU 缓冲区。这是一个集合（collective）操作，所有 PE 必须同步调用。

    ```python
    def buffer(size, release=False, except_on_del=True, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |size|[in]|要分配的缓冲区大小（字节）|
    |release|[in]|预留参数，无实际意义|
    |except_on_del|[in]|预留参数，无实际意义|
    |mem_type|[in]|对称堆类型：``MemType.DEVICE_SIDE`` 或 ``MemType.HOST_SIDE``，默认 Device 侧|
    |返回值|[out]|通过地址和字节长度表示的原始内存缓冲区。若分配失败则引发 ``AclshmemError``|

6. **free** — 释放由 ``buffer()`` 分配的缓冲区。这是一个集合（collective）操作。

    ```python
    def free(buf: Buffer) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |buf|[in]|需要释放的缓冲区|
    |返回值|-|无返回值|

7. **get_peer_buffer** — 将本地对称地址换算为指定 PE 上对应的对称地址。返回地址支持的访问方式取决于传输引擎和拓扑。

    ```python
    def get_peer_buffer(buf: Buffer, pe: int) -> Buffer
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |buf|[in]|本地 PE 上的对称地址|
    |pe|[in]|PE 编号|
    |返回值|[out]|指定 PE 上对应的对称地址缓冲区。若输入地址或 PE 非法则引发 ``AclshmemError``|

8. **put_signal** — 从本地 PE 复制连续数据到指定 PE 的对称内存地址，并在复制完成后更新远程信号变量。该接口在返回前完成本次操作；所有地址必须在调用期间保持有效。

    ```python
    def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int, stream=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 上目标数据的对称地址|
    |src|[in]|本地内存中的源数据地址|
    |signal_var|[in]|远程 PE 上待更新信号字的对称地址；Buffer 至少 4 字节且地址按 4 字节对齐|
    |signal_val|[in]|用于更新信号变量的值|
    |signal_operation|[in]|必须为 ``SignalOp`` 类型，支持：``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``|
    |remote_pe|[in]|必填的远程 PE 编号，必须为非 bool 整数且满足 ``0 <= remote_pe < pe_count()``|
    |stream|[in]|预留参数，忽略；该接口不提供显式流顺序|
    |返回值|-|无返回值。目标容量不足、信号字非法或 PE 编号非法时引发 ``AclshmemInvalid``|

9. **put** — 将本地 PE 的连续数据复制到指定 PE 的对称内存地址，并按指定流排序。Host 调用只将操作排入流；调用者须同步该流后才能观测完成。

    ```python
    def put(dst: Buffer, src: Buffer, remote_pe: int, stream: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 上目标数据的对称地址|
    |src|[in]|本地内存中的源数据地址|
    |remote_pe|[in]|必填的远程 PE 编号，必须为非 bool 整数且满足 ``0 <= remote_pe < pe_count()``|
    |stream|[in]|非 bool 的非负整数 ACL 流地址；传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值。目标容量、PE 编号或 stream 非法时引发 ``AclshmemInvalid``|

10. **get** — 将指定 PE 对称内存中的连续数据复制到本地缓冲区，并按指定流排序。Host 调用只将操作排入流；调用者须同步该流后才能读取结果。

    ```python
    def get(dst: Buffer, src: Buffer, remote_pe: int, stream: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地内存中的目标数据地址|
    |src|[in]|远程 PE 上源数据的对称地址|
    |remote_pe|[in]|必填的远程 PE 编号，必须为非 bool 整数且满足 ``0 <= remote_pe < pe_count()``|
    |stream|[in]|非 bool 的非负整数 ACL 流地址；传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值。目标容量、PE 编号或 stream 非法时引发 ``AclshmemInvalid``|

11. **signal_op** — 在指定 PE 上更新远程信号变量，并按指定流排序。Host 调用只将操作排入流；调用者须同步该流后才能观测完成。

    ```python
    def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int, stream: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |signal_var|[in]|目标 PE 可访问的信号变量的本地地址；Buffer 至少 4 字节且地址按 4 字节对齐|
    |signal_val|[in]|用于原子操作的值|
    |signal_operation|[in]|必须为 ``SignalOp`` 类型，支持：``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``|
    |remote_pe|[in]|必填的目标 PE 编号，必须为非 bool 整数且满足 ``0 <= remote_pe < pe_count()``|
    |stream|[in]|必须为非 bool 的非负整数 ACL 流地址；``0`` 表示默认流，不接受 ``None``|
    |返回值|-|无返回值。信号字、枚举、PE 编号或 stream 非法时引发 ``AclshmemInvalid``|

12. **signal_wait** — 等待对称信号变量满足指定比较条件。等待操作在给定流上执行，Host 调用立即返回；同步该流后，条件 ``signal_var`` ``cmp`` ``signal_val`` 保证为真。

    ```python
    def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |signal_var|[in]|源信号变量的本地地址；Buffer 至少 4 字节且地址按 4 字节对齐|
    |signal_val|[in]|与 signal_var 所指向值进行比较的值|
    |signal_operation|[in]|必须为 ``ComparisonType`` 类型，支持：``ComparisonType.CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |stream|[in]|必须为非 bool 的非负整数 ACL 流地址；``0`` 表示默认流，不接受 ``None``|
    |返回值|-|无返回值。信号字、枚举或 stream 非法时引发 ``AclshmemInvalid``|

13. **quiet** — 将完成点排入指定流，使该完成点之前由当前调用上下文发出的对称数据操作先于完成点结束。Host 调用立即返回；调用者须同步该流后才能观测完成。

    ```python
    def quiet(stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |stream|[in]|必须为非 bool 的非负整数 ACL 流地址；``0`` 表示默认流，不接受 ``None``|
    |返回值|-|无返回值。stream 非法时引发 ``AclshmemInvalid``|

14. **my_pe** — 获取本地 PE 编号。

    ```python
    def my_pe() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|本地 PE 编号|

15. **team_my_pe** — 获取当前进程在指定 team 中的 PE 编号。

    ```python
    def team_my_pe(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|目标 team 的 ID|
    |返回值|[out]|指定 team 中的 PE 编号，若 team 无效则返回 -1|

16. **n_pes** — 获取程序中运行的 PE 总数（world team 维度）。

    ```python
    def n_pes() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|PE 总数|

17. **team_n_pes** — 获取指定 team 中的 PE 总数。

    ```python
    def team_n_pes(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|目标 team 的 ID|
    |返回值|[out]|指定 team 中的 PE 数目，若 team 无效则返回 -1|

18. **init_status** — 查询共享内存模块的当前初始化状态。

    ```python
    def init_status() -> InitStatus
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|返回初始化状态枚举值：``NOT_INITIALIZED`` / ``SHM_CREATED`` / ``INITIALIZED`` / ``INVALID``|

19. **calloc** — 按元素数量从指定对称堆分配并清零内存。这是一个集合（collective）操作。

    ```python
    def calloc(count: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |count|[in]|元素个数，必须为正整数|
    |size|[in]|单个元素的字节数，必须为正整数；``count * size`` 不得溢出 ``size_t``|
    |mem_type|[in]|对称堆类型：``MemType.DEVICE_SIDE`` 或 ``MemType.HOST_SIDE``|
    |返回值|[out]|已清零的拥有型 ``Buffer``，长度为 ``count * size``；分配失败时引发 ``AclshmemError``|

20. **align** — 从指定对称堆按 2 的幂对齐分配内存。这是一个集合（collective）操作。

    ```python
    def align(alignment: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |alignment|[in]|对齐字节数，必须为正的 2 的幂|
    |size|[in]|分配字节数，必须为正整数且不超过 ``size_t`` 范围|
    |mem_type|[in]|对称堆类型：``MemType.DEVICE_SIDE`` 或 ``MemType.HOST_SIDE``|
    |返回值|[out]|满足指定对齐的拥有型 ``Buffer``；分配失败时引发 ``AclshmemError``|

21. **barrier** — 阻塞当前 Host 线程，直到指定 team 中所有 PE 到达 barrier。这是一个集合（collective）操作。

    ```python
    def barrier(team: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|0 到 2047 范围内且当前仍存活的 team ID|
    |返回值|-|无返回值；team 未创建、已销毁或超出范围时引发 ``AclshmemInvalid``|

22. **barrier_all** — 阻塞当前 Host 线程，直到 world team 中所有 PE 到达 barrier。这是一个集合（collective）操作。

    ```python
    def barrier_all() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值；所有 PE 必须按相同顺序调用|

23. **sync** — 同步指定 team 中所有 PE，保证此前 Host store 的完成与可见性。这是一个集合（collective）操作。

    ```python
    def sync(team: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|0 到 2047 范围内且当前仍存活的 team ID|
    |返回值|-|无返回值；team 未创建、已销毁或超出范围时引发 ``AclshmemInvalid``|

24. **sync_all** — 同步 world team 中所有 PE，保证此前 Host store 的完成与可见性。这是一个集合（collective）操作。

    ```python
    def sync_all() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值；所有 PE 必须按相同顺序调用|

25. **barrier_on_stream** — 将指定 team 的 barrier 操作排入 ACL Stream。该接口不隐式同步流。

    ```python
    def barrier_on_stream(team: int, stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|0 到 2047 范围内且当前仍存活的 team ID|
    |stream|[in]|非 bool 的非负整数 ``aclrtStream`` 地址；``0`` 表示默认流|
    |返回值|-|无返回值；team 或 stream 非法时引发 ``AclshmemInvalid``|

26. **barrier_all_on_stream** — 将 world team 的 barrier 操作排入 ACL Stream。该接口不隐式同步流。

    ```python
    def barrier_all_on_stream(stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |stream|[in]|非 bool 的非负整数 ``aclrtStream`` 地址；``0`` 表示默认流|
    |返回值|-|无返回值；stream 非法时引发 ``AclshmemInvalid``|

27. **current_instance** — 获取当前活动的 SHMEM 实例 ID。

    ```python
    def current_instance() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|当前活动实例 ID；未获取到实例上下文时引发 ``AclshmemError``|

28. **set_instance** — 将当前活动上下文切换到指定 SHMEM 实例。

    ```python
    def set_instance(instance_id: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |instance_id|[in]|目标实例 ID，必须为 0 到 254 范围内的非 bool 整数且实例已初始化|
    |返回值|-|无返回值；ID 非法时引发 ``AclshmemInvalid``，切换失败时引发 ``AclshmemError``|

29. **multi_instance** — 在上下文管理器代码块内切换实例，并在退出代码块时恢复原实例。

    ```python
    @contextmanager
    def multi_instance(instance_id: int)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |instance_id|[in]|代码块内使用的实例 ID，必须为 0 到 254 范围内的非 bool 整数且实例已初始化|
    |返回值|[out]|上下文管理器；进入代码块时切换到目标实例，退出时恢复先前实例|

30. **set_mte_config** — 为当前活动实例配置 MTE 工作区和同步标识。应在实例初始化并切换为当前实例后、相关操作发出前调用。

    ```python
    def set_mte_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|工作区偏移/地址，单位为字节；必须为 ``[0, 2**64 - 1]`` 范围内的非 bool 整数|
    |ub_size|[in]|工作区大小，单位为字节；必须为 ``[0, 2**32 - 1]`` 范围内的非 bool 整数|
    |sync_id|[in]|同步标识；必须为 ``[0, 2**32 - 1]`` 范围内的非 bool 整数|
    |返回值|-|无返回值；参数非法时引发 ``AclshmemInvalid``，当前实例无法应用配置时引发 ``AclshmemError``|

31. **set_sdma_config** — 为当前活动实例配置 SDMA 工作区和同步标识。应在实例初始化并切换为当前实例后、相关操作发出前调用；可用性取决于当前平台和运行时。

    ```python
    def set_sdma_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|工作区偏移/地址，单位为字节；必须为 ``[0, 2**64 - 1]`` 范围内的非 bool 整数|
    |ub_size|[in]|工作区大小，单位为字节；必须为 ``[0, 2**32 - 1]`` 范围内的非 bool 整数|
    |sync_id|[in]|同步标识；必须为 ``[0, 2**32 - 1]`` 范围内的非 bool 整数|
    |返回值|-|无返回值；参数非法时引发 ``AclshmemInvalid``，当前实例无法应用配置时引发 ``AclshmemError``|

32. **set_rdma_config** — 为当前活动实例配置 RDMA 工作区和同步标识。应在实例初始化并切换为当前实例后、相关操作发出前调用；可用性取决于当前平台和运行时。

    ```python
    def set_rdma_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|工作区偏移/地址，单位为字节；必须为 ``[0, 2**64 - 1]`` 范围内的非 bool 整数|
    |ub_size|[in]|工作区大小，单位为字节；必须为 ``[128, 2**32 - 1]`` 范围内的非 bool 整数|
    |sync_id|[in]|同步标识；必须为 ``[0, 2**32 - 1]`` 范围内的非 bool 整数|
    |返回值|-|无返回值；参数非法时引发 ``AclshmemInvalid``，当前实例无法应用配置时引发 ``AclshmemError``|

33. **set_udma_config** — 为当前活动实例配置 UDMA 工作区和同步标识。应在实例初始化并切换为当前实例后、相关操作发出前调用；可用性取决于当前平台和运行时。

    ```python
    def set_udma_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|工作区偏移/地址，单位为字节；必须为 ``[0, 2**64 - 1]`` 范围内的非 bool 整数|
    |ub_size|[in]|工作区大小，单位为字节；必须为 ``[128, 2**32 - 1]`` 范围内的非 bool 整数|
    |sync_id|[in]|同步标识；必须为 ``[0, 2**32 - 1]`` 范围内的非 bool 整数|
    |返回值|-|无返回值；参数非法时引发 ``AclshmemInvalid``，当前实例无法应用配置时引发 ``AclshmemError``|

34. **handle_wait** — 将 Handle 绑定 team 的操作完成等待和成员会合排入指定流。team 中所有 PE 必须在发出需要覆盖的异步操作后，以匹配顺序参与调用；仅 ``ACLSHMEM_TEAM_WORLD`` 对应的 Handle 要求所有 PE 参与。Host 调用只负责入流，调用者须同步该流后才能观测完成。

    ```python
    def handle_wait(handle: Handle, stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |handle|[in]|绑定当前仍存活 team 的 ``Handle``；调用成员范围由该 team 决定|
    |stream|[in]|非 bool 的非负整数 ``aclrtStream`` 地址；``0`` 表示默认流|
    |返回值|-|无返回值；Handle、team 或 stream 非法时引发 ``AclshmemInvalid``。参与成员或调用顺序不匹配可能导致永久等待|

35. **get_prof** — 获取当前实例、当前 PE 的 profiling 数据深拷贝。

    ```python
    def get_prof(verbose: bool=False) -> Optional[ProfData]
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |verbose|[in]|是否同时打印 profiling 数据，默认 ``False``|
    |返回值|[out]|``SHMEM_CYCLE_PROF_PE`` 指定的 PE 返回 ``ProfData``，其他 PE 返回 ``None``；数据按实例隔离|

36. **show_prof** — 打印当前实例的 profiling 数据。

    ```python
    def show_prof() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值；兼容接口，新代码优先使用 ``get_prof``|

### 类

1. **UniqueId** 类 — 用于 UID 初始化的唯一标识符句柄。

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |version|[out]|版本信息|
    |my_pe|[out]|当前进程的 PE 编号|
    |n_pes|[out]|所有进程的 PE 总数|
    |internal|[out]|UID 的内部信息（字节）|

2. **InitStatus** 枚举类 — 共享内存模块的初始化状态。

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |枚举值|含义|
    |-|-|
    |NOT_INITIALIZED|未初始化|
    |SHM_CREATED|共享内存已创建|
    |INITIALIZED|初始化完成|
    |INVALID|无效状态|

3. **SignalOp** 枚举类 — 信号变量原子操作类型。

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |枚举值|含义|
    |-|-|
    |SIGNAL_SET|原子设置：将给定值写入远程信号|
    |SIGNAL_ADD|原子加：将给定值加到远程信号现有值上|

4. **ComparisonType** 枚举类 — 信号等待比较操作类型。

    ```python
    class ComparisonType(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |枚举值|含义|
    |-|-|
    |CMP_EQ|等于（==）|
    |CMP_NE|不等于（!=）|
    |CMP_GT|大于（>）|
    |CMP_GE|大于等于（>=）|
    |CMP_LT|小于（<）|
    |CMP_LE|小于等于（<=）|

5. **MemType** 枚举类 — 对称堆类型。

    ```python
    class MemType(Enum):
        HOST_SIDE
        DEVICE_SIDE
    ```

    |枚举值|含义|
    |-|-|
    |HOST_SIDE|Host 侧对称堆；是否可用取决于 CANN 运行时能力|
    |DEVICE_SIDE|Device 侧对称堆，默认值|

6. **Buffer** 类 — 使用地址、长度和对称堆类型描述一段可供 SHMEM 接口使用的内存。

    ```python
    class Buffer:
        def __init__(
            self,
            addr: int,
            length: int,
            mem_type: MemType=MemType.DEVICE_SIDE,
        ) -> None:
    ```

    **属性**

    |属性|方向|含义|
    |-|-|-|
    |addr|[in/out]|内存起始地址|
    |length|[in/out]|内存长度，单位为字节|
    |mem_type|[in/out]|对称堆类型：``MemType.DEVICE_SIDE`` 或 ``MemType.HOST_SIDE``|
    |owned|[out]|只读；表示该对象是否拥有内存释放权|
    |instance_id|[out]|只读；工厂分配的 Buffer 所属实例 ID，外部地址描述对象为 ``None``|
    |release_called|[out]|只读；表示是否已经对该内存发起过释放|

    **功能与约束**

    - 直接构造 ``Buffer(addr, length)`` 用于描述外部地址，不拥有内存，不能传给 ``free``。通过 ``buffer``、``calloc`` 或 ``align`` 创建的 Buffer 拥有内存释放权。
    - ``addr`` 和 ``length`` 必须为非 bool 整数，并分别满足 ``0 < addr <= INTPTR_MAX`` 和 ``0 < length <= SIZE_T_MAX``。
    - 工厂分配的 Buffer 及其 peer 视图只能在所属实例为当前实例时用于 ``free``、peer 地址换算和 RMA/Signal 操作；实例不匹配时引发 ``AclshmemInvalid``。
    - 内存发起释放后，原 Buffer 及其 peer 视图均不可再用于释放、peer 地址换算或 RMA/Signal 操作；重复释放会被拒绝。
    - Buffer 析构时不会自动释放对称内存。释放是集合操作，必须由所有 PE 以相同顺序显式调用 ``free``。

7. **InstanceContext** 类 — 当前活动实例上下文的只读值快照。

    ```python
    class InstanceContext:
        @property
        def id(self) -> int:
    ```

    |属性|方向|含义|
    |-|-|-|
    |id|[out]|只读；实例 ID|

8. **Handle** 类 — team 作用域的异步操作句柄。

    ```python
    class Handle:
        def __init__(self, team_id: int=ACLSHMEM_TEAM_WORLD) -> None:

        @property
        def team_id(self) -> int:
    ```

    |属性/参数|方向|含义|
    |-|-|-|
    |team_id|[in/out]|team ID，默认 ``ACLSHMEM_TEAM_WORLD``；构造和赋值时必须引用 0 到 2047 范围内且当前仍存活的 team|

    ``team_id`` 为 bool、超出范围、未创建或已销毁时引发 ``ValueError``。

9. **ProfData** 类 — 当前 PE 的 profiling 数据深拷贝快照，不依赖 NumPy。

    ```python
    class ProfData:
        @property
        def pe_id(self) -> int:

        @property
        def ccount(self) -> list[list[int]]:

        @property
        def cycles(self) -> list[list[int]]:
    ```

    |属性|方向|含义|
    |-|-|-|
    |pe_id|[out]|只读；profiling 数据所属 PE 编号|
    |ccount|[out]|只读；尺寸为 ``64 x 1024`` 的计数二维整数列表|
    |cycles|[out]|只读；尺寸为 ``64 x 1024`` 的周期二维整数列表|

## shmem._pyshmem API

### 对外接口

1. **aclshmem_init** — 初始化共享内存模块。这是一个集合（collective）操作。

    ```python
    def aclshmem_init(attributes: InitAttr) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |attributes|[in]|初始化属性（`InitAttr` 类型），包含 `my_pe`（本地 PE 索引，范围 0 ~ n_pes-1）、`n_pes`（PE 总数）、`local_mem_size`（每个 PE 分配的内存大小，字节）等字段|
    |返回值|[out]|成功返回 0，失败返回 -1|

2. **aclshmem_finalize** — 销毁共享内存模块，释放所有资源。

    ```python
    def aclshmem_finalize() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |无参数|[in]|-|
    |返回值|[out]|状态码：成功返回 0，失败返回错误码；如需销毁指定实例，请使用 ``aclshmemx_finalize(instance_id)``|

3. **aclshmemx_init_status** — 查询共享内存模块的当前初始化状态。

    ```python
    def aclshmemx_init_status() -> InitStatus
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|返回初始化状态枚举值。返回 ``INITIALIZED`` 表示初始化已完成|

4. **set_conf_store_tls_key** — 设置 TLS 私钥与密码，并注册解密回调函数。

    ```python
    def set_conf_store_tls_key(tls_pk, tls_pk_pw, py_decrypt_func:Callable[[str], str]) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |tls_pk|[in]|TLS 私钥内容|
    |tls_pk_pw|[in]|TLS 私钥密码内容|
    |py_decrypt_func|[in]|解密回调函数，接受 ``(str cipher_text)`` 返回 ``(str plain_text)``|
    |返回值|[out]|成功返回 0，失败返回非零错误码|

5. **set_conf_store_tls** — 设置是否开启 TLS 加密，并可指定 TLS 配置信息。

    ```python
    def set_conf_store_tls(enable, tls_info) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |enable|[in]|是否开启 TLS 加密，``True`` 开启，``False`` 关闭|
    |tls_info|[in]|TLS 配置信息字符串，可为空字符串|
    |返回值|[out]|成功返回 0，失败返回非零错误码|

6. **aclshmem_malloc** — 分配对称内存。这是一个集合（collective）操作，内嵌隐式 barrier。

    ```python
    def aclshmem_malloc(size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |size|[in]|分配内存大小（字节）|
    |返回值|[out]|成功返回指向已分配内存的指针（int），若 size 为 0 或分配失败返回 0 并引发异常|

7. **aclshmem_calloc** — 分配零初始化的对称内存。集合操作，内嵌隐式 barrier。

    ```python
    def aclshmem_calloc(nmemb, size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |nmemb|[in]|元素数量|
    |size|[in]|每个元素的大小（字节）|
    |返回值|[out]|成功返回指向已分配内存的指针（int），若 nmemb 或 size 为 0 返回 0 并引发异常|

8. **aclshmem_align** — 分配指定对齐方式的对称内存。集合操作，内嵌隐式 barrier。

    ```python
    def aclshmem_align(alignment, size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |alignment|[in]|内存对齐要求（必须是 2 的幂）|
    |size|[in]|要分配的字节数|
    |返回值|[out]|成功返回指向已分配内存的指针（int），若分配失败则引发异常|

9. **aclshmem_free** — 释放由对称内存分配函数分配的内存空间。集合操作，内嵌隐式 barrier。

    ```python
    def aclshmem_free(ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |ptr|[in]|要释放的内存指针|
    |返回值|-|无返回值|

10. **aclshmem_ptr** — 将本地对称地址换算为指定 PE 上对应的对称地址。返回地址支持的访问方式取决于传输引擎和拓扑。

    ```python
    def aclshmem_ptr(ptr, peer) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |ptr|[in]|本地 PE 上的对称地址|
    |peer|[in]|PE 编号|
    |返回值|[out]|成功返回指定 PE 上对应的对称地址（int），若输入地址或 PE 非法则返回 0|

11. **aclshmemx_get_heap_base** — 获取本地对称内存堆的起始地址。

    ```python
    def aclshmemx_get_heap_base(mem_type: MemType=None) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |mem_type|[in]|对称内存分配位置：``MemType.HOST_SIDE``（主机侧）/ ``MemType.DEVICE_SIDE``（设备侧，默认）|
    |返回值|[out]|成功返回对称内存堆的起始地址指针（int），未初始化时返回 0|

12. **my_pe** — 获取 PE 编号（world team 维度）。

    ```python
    def my_pe() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|本地 PE 编号|

13. **team_my_pe** — 获取指定 team 中的 PE 编号。

    ```python
    def team_my_pe(team_id) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team_id|[in]|team 的句柄|
    |返回值|[out]|指定 team 中 PE 的编号，出错返回 -1|

14. **pe_count** — 获取 PE 总数（world team 维度）。

    ```python
    def pe_count() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|PE 总数|

15. **team_n_pes** — 获取指定 team 中的 PE 数量。

    ```python
    def team_n_pes(team_id) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team_id|[in]|team 的句柄|
    |返回值|[out]|指定 team 中 PE 的数量，出错返回 -1|

16. **team_split_strided** — 从现有父 team 中按步长拆分子 team。这是一个集合（collective）操作。

    ```python
    def team_split_strided(parent, start, stride, size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |parent|[in]|父 team ID|
    |start|[in]|子 team 的起始 PE 编号|
    |stride|[in]|PE 编号之间的步长|
    |size|[in]|子 team 包含的 PE 数量|
    |返回值|[out]|成功且当前 PE 属于子 team 时返回新 team ID；当前 PE 不属于子 team 时返回 `ACLSHMEM_TEAM_INVALID`；底层返回错误码时引发 `RuntimeError`|

17. **team_split_2d** — 基于二维笛卡尔空间从父 team 中拆分 team。集合操作。

    ```python
    def team_split_2d(parent, x_range) -> tuple
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |parent|[in]|父 team 句柄|
    |x_range|[in]|第一维度的元素数量|
    |返回值|[out]|成功返回 (x_team_id, y_team_id) 元组，失败引发异常|

18. **aclshmem_team_get_config** — 获取创建 team 时传入的 team 配置。

    ```python
    def aclshmem_team_get_config(team) -> TeamConfig
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|team ID|
    |返回值|[out]|成功返回 ``TeamConfig`` 对象，失败引发异常|

19. **aclshmem_putmem** — 将连续数据从本地 PE 复制到指定 PE 的对称地址。同步（blocking）接口。

    ```python
    def aclshmem_putmem(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

20. **aclshmem_getmem** — 将对称内存中指定 PE 上的连续数据复制到本地 PE。同步（blocking）接口。

    ```python
    def aclshmem_getmem(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

21. **aclshmem_{TYPE}_iput** — 将本地内存中按 sst 步距排列的数据复制到指定 PE 对称内存的 dst 步距位置。同步（blocking）接口。

    ```python
    def aclshmem_{TYPE}_iput(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``double`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |dest|[in]|远程目标数据的对称内存指针|
    |source|[in]|本地源数据的指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

22. **aclshmem_{TYPE}_iget** — 将远程 PE 对称内存中按 sst 步距排列的数据复制到本地 dst 步距位置。同步（blocking）接口。

    ```python
    def aclshmem_{TYPE}_iget(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``double`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |dest|[in]|本地目标数据的指针|
    |source|[in]|远程源数据的对称内存指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

23. **aclshmem_put{BITS}** — 将连续数据从本地 PE 复制到指定 PE 的对称内存地址。同步（blocking）接口。

    ```python
    def aclshmem_put{BITS}(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

24. **aclshmem_get{BITS}** — 将对称内存中指定 PE 上的连续数据复制到本地 PE。同步（blocking）接口。

    ```python
    def aclshmem_get{BITS}(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

25. **aclshmem_iput{BITS}** — 将本地内存中按 sst 步距排列的数据复制到指定 PE 的 dst 步距位置（位宽版本）。同步（blocking）接口。

    ```python
    def aclshmem_iput{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dest|[in]|远程目标数据的对称内存指针|
    |source|[in]|本地源数据的指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

26. **aclshmem_iget{BITS}** — 将远程 PE 对称内存中按 sst 步距排列的数据复制到本地 dst 步距位置（位宽版本）。同步（blocking）接口。

    ```python
    def aclshmem_iget{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dest|[in]|本地目标数据的指针|
    |source|[in]|远程源数据的对称内存指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

27. **aclshmem_info_get_version** — 返回库的主版本号和次版本号。

    ```python
    def aclshmem_info_get_version() -> tuple
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|返回 (major, minor) 元组|

28. **aclshmem_info_get_name** — 返回供应商定义的名称字符串。

    ```python
    def aclshmem_info_get_name() -> str
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|供应商定义的名称字符串|

29. **team_translate_pe** — 将一个 team 中的给定 PE 编号转换为另一个 team 中的对应 PE 编号。

    ```python
    def team_translate_pe(src_team, src_pe, dest_team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |src_team|[in]|源 team ID|
    |src_pe|[in]|源 PE 编号|
    |dest_team|[in]|目标 team ID|
    |返回值|[out]|成功返回目标 team 中对应的 PE 编号，出错返回 -1|

30. **team_destroy** — 销毁一个 team。

    ```python
    def team_destroy(team) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|要销毁的 team ID|
    |返回值|-|无返回值|

31. **get_ffts_config** — 获取运行时 FFTS 配置。

    ```python
    def get_ffts_config() -> str
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|FFTS 配置字符串|

32. **set_log_level** — 设置所有模块的日志级别。

    ```python
    def set_log_level(level) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |level|[in]|日志级别|
    |返回值|[out]|成功返回 0，失败返回非零错误码|

33. **set_extern_logger** — 注册外部日志回调函数，用于接管所有模块的日志输出。

    ```python
    def set_extern_logger(func) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |func|[in]|日志回调函数，签名为 ``func(level: int, msg: str)``|
    |返回值|[out]|成功返回 0，失败返回非零错误码|

34. **aclshmem_putmem_nbi** — 将本地 PE 上的连续数据复制到指定 PE 的对称地址。异步（non-blocking）接口。

    ```python
    def aclshmem_putmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

35. **aclshmem_getmem_nbi** — 将对称内存中指定 PE 上的连续数据复制到本地 PE。异步（non-blocking）接口。

    ```python
    def aclshmem_getmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

36. **aclshmem_put{BITS}_nbi** — 将连续数据从本地 PE 复制到指定 PE 的对称内存地址（位宽版本）。异步（non-blocking）接口。

    ```python
    def aclshmem_put{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

37. **aclshmem_get{BITS}_nbi** — 将对称内存中指定 PE 上的连续数据复制到本地 PE（位宽版本）。异步（non-blocking）接口。

    ```python
    def aclshmem_get{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

38. **aclshmemx_putmem_signal_nbi** — 从本地 PE 复制连续数据到指定 PE 的对称地址，并在完成后更新远程信号变量。异步（non-blocking）接口。

    ```python
    def aclshmemx_putmem_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``SIGNAL_SET`` / ``SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

39. **aclshmemx_putmem_signal** — 从本地 PE 复制连续数据到指定 PE 的对称地址，并更新远程信号变量。同步（blocking）接口。

    ```python
    def aclshmemx_putmem_signal(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``SIGNAL_SET`` / ``SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

40. **aclshmemx_put{BITS}_signal_nbi** — 从本地 PE 复制连续数据到指定 PE 对称地址并更新远程信号（位宽版本）。异步（non-blocking）接口。

    ```python
    def aclshmemx_put{BITS}_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``SIGNAL_SET`` / ``SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

41. **aclshmemx_put{BITS}_signal** — 从本地 PE 复制连续数据到指定 PE 对称地址并更新远程信号（位宽版本）。同步（blocking）接口。

    ```python
    def aclshmemx_put{BITS}_signal(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``SIGNAL_SET`` / ``SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

42. **aclshmem_global_exit** — 所有 PE 通过广播调用 exit() 退出进程。

    ```python
    def aclshmem_global_exit(status) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |status|[in]|传递给 exit() 的状态值|
    |返回值|-|无返回值|

43. **my_pe** — 获取指定 team 中的 PE 编号。

    ```python
    def my_pe(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|team ID|
    |返回值|[out]|指定 team 中 PE 的编号，出错返回 -1|

44. **pe_count** — 获取指定 team 中的 PE 数量。

    ```python
    def pe_count(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|team ID|
    |返回值|[out]|指定 team 中 PE 的数目，出错返回 -1|

45. **aclshmem_signal_wait_until** — 等待信号变量满足比较条件（``*sig_addr cmp cmp_val``）时返回。阻塞（blocking）接口。

    ```python
    def aclshmem_signal_wait_until(sig_addr, cmp, cmp_val) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |sig_addr|[in]|源信号变量的本地地址|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|比较值|
    |返回值|[out]|满足条件时 ``sig_addr`` 的值|

46. **aclshmem_{TYPE}_wait_until** — 等待单个元素满足比较条件 ``ivar cmp cmp_val`` 后返回（类型化版本）。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until(ivar, cmp, cmp_val) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivar|[in]|对称内存中信号变量的指针|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|比较值|
    |返回值|-|无返回值|

47. **aclshmem_{TYPE}_wait** — 等待信号变量不等于给定值时返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait(ivar, cmp_val) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivar|[in]|对称内存中信号变量的指针|
    |cmp_val|[in]|比较值|
    |返回值|-|无返回值|

48. **aclshmem_{TYPE}_wait_until_all** — 等待数组中所有元素均满足比较条件 ``ivars[i] cmp cmp_val`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_all(ivars_ptr, nelems, status_ptr, cmp, cmp_val) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|比较值|
    |返回值|-|无返回值|

49. **aclshmem_{TYPE}_wait_until_any** — 等待数组中至少有一个元素满足比较条件 ``ivars[i] cmp cmp_val`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_any(ivars_ptr, nelems, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|比较值|
    |res_out_ptr|[out]|接收满足比较条件的元素索引值|
    |返回值|-|无返回值|

50. **aclshmem_{TYPE}_wait_until_some** — 等待数组中至少有一个元素满足比较条件，并返回所有满足条件的元素索引。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|比较值|
    |res_out_ptr|[out]|接收满足比较条件的元素个数|
    |返回值|-|无返回值|

51. **aclshmem_{TYPE}_wait_until_all_vector** — 等待数组中所有元素均满足向量比较条件 ``ivars[i] cmp cmp_values[i]`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_values_ptr|[in]|比较值数组|
    |返回值|-|无返回值|

52. **aclshmem_{TYPE}_wait_until_any_vector** — 等待数组中至少有一个元素满足向量比较条件 ``ivars[i] cmp cmp_values[i]`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|接收满足比较条件的元素索引值|
    |返回值|-|无返回值|

53. **aclshmem_{TYPE}_wait_until_some_vector** — 等待数组中至少有一个元素满足向量比较条件，并返回所有满足条件的元素索引。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|接收满足比较条件的元素个数|
    |返回值|-|无返回值|

54. **aclshmem_{TYPE}_test** — 检查单个元素是否满足比较条件 ``ivar cmp cmp_value``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test(ivar, cmp, cmp_value, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivar|[in]|对称内存中信号变量的指针|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_value|[in]|比较值|
    |res_out_ptr|[out]|满足条件返回 1，否则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

55. **aclshmem_{TYPE}_test_any** — 检查数组中是否至少有一个元素满足比较条件 ``ivars[i] cmp cmp_value``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_any(ivars_ptr, nelems, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_value|[in]|比较值|
    |res_out_ptr|[out]|满足条件的元素索引值；若无元素满足或测试集为空则返回 ``SIZE_MAX``|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

56. **aclshmem_{TYPE}_test_some** — 检查数组中是否至少有一个元素满足比较条件，并返回所有满足条件的元素索引。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_value|[in]|比较值|
    |res_out_ptr|[out]|满足条件的元素个数；若测试集为空则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 和 ``indices_ptr`` 返回|

57. **aclshmem_{TYPE}_test_all_vector** — 检查数组中所有元素是否均满足向量比较条件 ``ivars[i] cmp cmp_values[i]``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|全部满足或 nelems 为 0 返回 1，否则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

58. **aclshmem_{TYPE}_test_any_vector** — 检查数组中是否至少有一个元素满足向量比较条件 ``ivars[i] cmp cmp_values[i]``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|满足条件的第一个元素索引值；若无元素满足或测试集为空则返回 ``SIZE_MAX``|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

59. **aclshmem_{TYPE}_test_some_vector** — 检查数组中是否至少有一个元素满足向量比较条件，并返回所有满足条件的元素索引。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|满足条件的元素个数；若测试集为空则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 和 ``indices_ptr`` 返回|

60. **aclshmemx_putmem_on_stream** — 在指定流上将连续数据从本地 PE 复制到指定 PE 的对称地址。非阻塞（non-blocking）接口。

    ```python
    def aclshmemx_putmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |stream|[in]|ACL 流的整数地址；传入 ``0`` 使用默认流，不接受 ``None``|
    |返回值|-|无返回值|

61. **aclshmemx_getmem_on_stream** — 在指定流上将对称内存中指定 PE 上的连续数据复制到本地 PE。非阻塞（non-blocking）接口。

    ```python
    def aclshmemx_getmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |stream|[in]|ACL 流的整数地址；传入 ``0`` 使用默认流，不接受 ``None``|
    |返回值|-|无返回值|

62. **aclshmemx_signal_op_on_stream** — 将远程信号变量更新操作排入指定流。Host 调用立即返回；调用者须同步该流后才能观测完成。

    ```python
    def aclshmemx_signal_op_on_stream(sig, signal, sig_op, pe, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |sig|[in]|目标 PE 上可访问的信号变量的本地地址|
    |signal|[in]|用于原子操作的值|
    |sig_op|[in]|对远程信号执行的操作。支持：``SIGNAL_SET`` / ``SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |stream|[in]|ACL 流的整数地址；传入 ``0`` 使用默认流，不接受 ``None``|
    |返回值|-|无返回值|

63. **aclshmemx_signal_wait_until_on_stream** — 将信号条件等待排入指定流。Host 调用立即返回；调用者须同步该流后才能确认条件已满足。

    ```python
    def aclshmemx_signal_wait_until_on_stream(sig, cmp, cmp_val, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |sig|[in]|源信号变量的本地地址|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|与 sig 所指向值进行比较的值|
    |stream|[in]|ACL 流对象，用于执行排序。必须传入有效流|
    |返回值|-|无返回值|

64. **aclshmemx_quiet_on_stream** — 将完成点排入指定流，使该完成点之前由当前调用上下文发出的对称数据操作先于完成点结束。Host 调用立即返回；调用者须同步该流后才能观测完成。

    ```python
    def aclshmemx_quiet_on_stream(stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |stream|[in]|执行 quiet 操作的 ACL 流整数地址；传入 ``0`` 使用默认流，不接受 ``None``|
    |返回值|-|无返回值|

#### 扩展 Host API 绑定

以下接口提供与同名 Host 操作一致的 Python 调用。地址和 stream 参数使用非 bool 整数表示；``0`` 表示默认流，非零 stream 必须在调用期间保持有效。包含集合语义的接口会在各自条目中明确参与范围、匹配顺序和完成条件。

65. **aclshmemx_finalize** — 销毁指定 ID 的 SHMEM 实例并释放相关资源。

    ```python
    def aclshmemx_finalize(instance_id: int) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |instance_id|[in]|实例 ID；仅接受 0 到 254 范围内的非 bool 整数|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

66. **aclshmemx_set_qp_num** — 配置数据操作引擎创建的 QP 数量。该配置为进程级配置，必须在 ACLSHMEM 初始化前调用，并且所有 PE 必须使用相同的参数。

    ```python
    def aclshmemx_set_qp_num(engine: OpEngineType, qp_num: int) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |engine|[in]|数据操作引擎。``OpEngineType.SDMA`` 的取值范围为 ``[1, min(device vector-core count, ACLSHMEM_MAX_AIV_PER_NPU)]``，每个 QP 对应一个本地 device-only stream 且不绑定对端；``OpEngineType.UDMA`` 在 Ascend950 上可用，``OpEngineType.ROCE`` 要求 XSCALE 或 HNS_1825 后端|
    |qp_num|[in]|SDMA 的取值范围为 ``[1, min(device vector-core count, ACLSHMEM_MAX_AIV_PER_NPU)]``；UDMA 和 ROCE 的取值范围为 ``[1, ACLSHMEM_MAX_QP_NUM]``（当前最大值为 56），每个对端连接创建一个 QP 组|
    |返回值|[out]|成功返回 ``0``；参数非法返回 ``ACLSHMEM_INVALID_VALUE``；引擎或当前配置不支持时返回 ``ACLSHMEM_NOT_SUPPORTED``|

    配置在任一 ACLSHMEM 实例初始化后被冻结；最后一个实例 ``aclshmem_finalize`` 后，QP 数量重置为 1。启用 UDMA relay（``ACLSHMEM_RELAY_SUPPORT=ON``）时仅支持 ``qp_num == 1``，多 QP 配置会在初始化阶段返回 ``ACLSHMEM_NOT_SUPPORTED``。

67. **aclshmemx_set_attr_uniqueid_args** — 使用唯一 ID 参数填充 ``InitAttr``。

    ```python
    def aclshmemx_set_attr_uniqueid_args(my_pe: int, n_pes: int, local_mem_size: int, uid: UniqueId, attr: InitAttr) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |my_pe|[in]|当前 PE 编号，满足 ``0 <= my_pe < n_pes``|
    |n_pes|[in]|实例中的 PE 总数|
    |local_mem_size|[in]|每个 PE 的本地对称内存大小，单位为字节|
    |uid|[in]|用于实例初始化的 ``UniqueId``；调用成功后，即使删除原 ``uid`` 变量，填充后的 ``attr`` 仍可用于初始化|
    |attr|[in/out]|待填充的 ``InitAttr`` 对象|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

68. **aclshmemx_init_attr** — 使用初始化模式和 ``InitAttr`` 初始化 SHMEM 实例。这是一个集合（collective）操作。

    ```python
    def aclshmemx_init_attr(bootstrap_flags: InitMode, attributes: InitAttr) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |bootstrap_flags|[in]|初始化方式，UID 多实例场景使用 ``InitMode.UNIQUEID``|
    |attributes|[in]|包含 PE、内存、实例 ID 和引擎选项的初始化属性|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

69. **aclshmemx_instance_ctx_get** — 获取当前实例上下文的值快照。

    ```python
    def aclshmemx_instance_ctx_get() -> Optional[InstanceContext]
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|返回包含当前实例 ID 的 ``InstanceContext`` 快照；无上下文时返回 ``None``|

70. **aclshmemx_instance_ctx_set** — 将当前运行时上下文切换到指定实例。

    ```python
    def aclshmemx_instance_ctx_set(instance_id: int) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |instance_id|[in]|目标实例 ID；仅接受 0 到 254 范围内的非 bool 整数|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

71. **aclshmemx_malloc** — 从指定类型的对称堆分配内存。这是一个集合（collective）操作。

    ```python
    def aclshmemx_malloc(size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |size|[in]|分配字节数|
    |mem_type|[in]|对称堆类型，默认 ``MemType.DEVICE_SIDE``|
    |返回值|[out]|成功返回整数地址，失败返回 0|

72. **aclshmemx_calloc** — 从指定类型的对称堆分配并清零内存。这是一个集合（collective）操作。

    ```python
    def aclshmemx_calloc(count: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |count|[in]|元素个数|
    |size|[in]|单个元素的字节数|
    |mem_type|[in]|对称堆类型，默认 ``MemType.DEVICE_SIDE``|
    |返回值|[out]|成功返回已清零内存的整数地址，失败返回 0|

73. **aclshmemx_align** — 从指定类型的对称堆对齐分配内存。这是一个集合（collective）操作。

    ```python
    def aclshmemx_align(alignment: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |alignment|[in]|对齐字节数，必须满足底层对齐要求|
    |size|[in]|分配字节数|
    |mem_type|[in]|对称堆类型，默认 ``MemType.DEVICE_SIDE``|
    |返回值|[out]|成功返回满足对齐要求的整数地址，失败返回 0|

74. **aclshmemx_free** — 释放指定对称堆中的内存。这是一个集合（collective）操作。

    ```python
    def aclshmemx_free(ptr: int, mem_type: MemType=MemType.DEVICE_SIDE) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |ptr|[in]|由对应分配接口返回的整数地址|
    |mem_type|[in]|必须与分配时使用的对称堆类型一致|
    |返回值|-|无返回值|

75. **aclshmemx_set_mte_config** — 设置 MTE 引擎的 UB workspace 与同步 ID。

    ```python
    def aclshmemx_set_mte_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|UB 偏移|
    |ub_size|[in]|UB 大小|
    |sync_id|[in]|同步 ID|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

76. **aclshmemx_set_sdma_config** — 设置 SDMA 引擎的 UB workspace 与同步 ID。

    ```python
    def aclshmemx_set_sdma_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|UB 偏移|
    |ub_size|[in]|UB 大小|
    |sync_id|[in]|同步 ID|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

77. **aclshmemx_set_rdma_config** — 设置 RDMA 引擎的 UB workspace 与同步 ID。

    ```python
    def aclshmemx_set_rdma_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|UB 偏移|
    |ub_size|[in]|UB 大小，不得小于 128 字节|
    |sync_id|[in]|同步 ID|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

78. **aclshmemx_set_udma_config** — 设置 UDMA 引擎的 UB workspace 与同步 ID。

    ```python
    def aclshmemx_set_udma_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |offset|[in]|UB 偏移|
    |ub_size|[in]|UB 大小，不得小于 128 字节|
    |sync_id|[in]|同步 ID|
    |返回值|[out]|成功返回 ``ACLSHMEM_SUCCESS``，失败返回底层错误码|

79. **aclshmem_barrier** — 阻塞 Host，直到指定 team 中所有 PE 到达 barrier。这是一个集合（collective）操作。

    ```python
    def aclshmem_barrier(team: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|0 到 2047 范围内且当前仍存活的 team ID|
    |返回值|-|无返回值；非法或失效 team 引发 ``ValueError``|

80. **aclshmem_barrier_all** — 阻塞 Host，直到 world team 中所有 PE 到达 barrier。这是一个集合（collective）操作。

    ```python
    def aclshmem_barrier_all() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值|

81. **aclshmem_sync** — 同步指定 team 中的所有 PE。这是一个集合（collective）操作。

    ```python
    def aclshmem_sync(team: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|0 到 2047 范围内且当前仍存活的 team ID|
    |返回值|-|无返回值；非法或失效 team 引发 ``ValueError``|

82. **aclshmem_sync_all** — 同步 world team 中的所有 PE。这是一个集合（collective）操作。

    ```python
    def aclshmem_sync_all() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值|

83. **aclshmemx_barrier_on_stream** — 将指定 team 的 barrier 排入 ACL Stream。

    ```python
    def aclshmemx_barrier_on_stream(team: int, stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|0 到 2047 范围内且当前仍存活的 team ID|
    |stream|[in]|``aclrtStream`` 的整数地址；``0`` 表示默认流|
    |返回值|-|无返回值；非法或失效 team 引发 ``ValueError``|

84. **aclshmemx_barrier_all_on_stream** — 将 world team 的 barrier 排入 ACL Stream。

    ```python
    def aclshmemx_barrier_all_on_stream(stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |stream|[in]|``aclrtStream`` 的整数地址；``0`` 表示默认流|
    |返回值|-|无返回值|

85. **aclshmemx_handle_wait** — 将 Handle 绑定 team 的操作完成等待和成员会合排入指定流。team 中所有 PE 必须在发出需要覆盖的异步操作后，以匹配顺序参与调用；仅 world-team Handle 要求所有 PE 参与。Host 调用只负责入流，调用者须同步该流后才能观测完成。

    ```python
    def aclshmemx_handle_wait(handle: Handle, stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |handle|[in]|绑定当前仍存活 team 的 ``Handle``；该 team 决定参与的 PE 范围|
    |stream|[in]|``aclrtStream`` 的整数地址；``0`` 表示默认流|
    |返回值|-|无返回值；Handle 中的 team 已失效时引发 ``ValueError``。参与成员或调用顺序不匹配可能导致永久等待|

86. **aclshmemx_get_prof** — 获取当前实例、当前 PE 的 profiling 数据深拷贝。

    ```python
    def aclshmemx_get_prof(verbose: bool=False) -> Optional[ProfData]
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |verbose|[in]|是否同时打印 profiling 数据，默认 ``False``|
    |返回值|[out]|目标 PE 返回 ``ProfData``，无可用数据时返回 ``None``；数据按实例隔离|

87. **aclshmemx_show_prof** — 打印当前实例的 profiling 数据。

    ```python
    def aclshmemx_show_prof() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值|

#### 类

1. **OpEngineType** 枚举类 — 数据传输引擎类型。

    ```python
    class OpEngineType(Enum):
        MTE
        SDMA
        ROCE
        UDMA
    ```

    |枚举值|含义|
    |-|-|
    |MTE|Memory Transfer Engine|
    |SDMA|System DMA|
    |ROCE|RDMA over Converged Ethernet|
    |UDMA|Unified DMA|

2. **MemType** 枚举类 — 对称内存分配位置。

    ```python
    class MemType(Enum):
        HOST_SIDE
        DEVICE_SIDE
    ```

    |枚举值|含义|
    |-|-|
    |HOST_SIDE|主机侧内存|
    |DEVICE_SIDE|设备侧内存（默认）|

3. **OptionalAttr** 类 — 初始化可选属性配置。

    ```python
    class OptionalAttr:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |version|[in]|配置版本号|
    |data_op_engine_type|[in]|数据传输引擎类型（``OpEngineType`` 枚举值）|
    |shm_init_timeout|[in]|init 操作的超时时间|
    |shm_create_timeout|[in]|create 操作的超时时间|
    |control_operation_timeout|[in]|控制操作的超时时间|
    |sockFd|[in]|socket 文件描述符，默认 -1|

4. **InitAttr** 类 — 初始化属性配置。

    ```python
    class InitAttr:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |my_rank|[in]|当前进程的 PE 编号|
    |n_ranks|[in]|PE 总数|
    |ip_port|[in]|通信服务器的 IP 和端口|
    |local_mem_size|[in]|当前 PE 分配的对称内存大小（字节）|
    |option_attr|[in]|``OptionalAttr`` 可选属性配置|
    |instance_id|[in]|多实例 ID，绑定层仅接受 0 到 254 范围内的非 bool 整数，默认 0|

5. **TeamConfig** 类 — Team 配置。

    ```python
    class TeamConfig:
    ```

    |属性|方向|含义|
    |-|-|-|
    |num_contexts|[in]|一个 team 中可以同时运行的上下文数量|

6. **UniqueId** 类 — UID 初始化的唯一标识符句柄。

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |version|[out]|版本信息|
    |my_pe|[out]|当前进程的 PE 编号|
    |n_pes|[out]|所有进程的 PE 总数|
    |internal|[out]|UID 的内部信息（字节）|
    |``to_bytes()``|[out]|按原生结构大小序列化 UID|
    |``from_bytes(data)``|[in]|从长度严格匹配的字节串恢复 UID|

7. **InitStatus** 枚举类 — 共享内存模块初始化状态。

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |枚举值|含义|
    |-|-|
    |NOT_INITIALIZED|未初始化|
    |SHM_CREATED|共享内存已创建|
    |INITIALIZED|初始化完成|
    |INVALID|无效状态|

8. **SignalOp** 枚举类 — 信号变量原子操作类型。

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |枚举值|含义|
    |-|-|
    |SIGNAL_SET|原子设置：将给定值写入远程信号|
    |SIGNAL_ADD|原子加：将给定值加到远程信号现有值上|

9. **CmpOp** 枚举类 — 信号比较操作类型。

    ```python
    class CmpOp(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |枚举值|含义|
    |-|-|
    |CMP_EQ|等于（==）|
    |CMP_NE|不等于（!=）|
    |CMP_GT|大于（>）|
    |CMP_GE|大于等于（>=）|
    |CMP_LT|小于（<）|
    |CMP_LE|小于等于（<=）|

10. **InstanceContext** 类 — 当前活动实例上下文的只读值快照。

    ```python
    class InstanceContext:
        @property
        def id(self) -> int:
    ```

    |属性|方向|含义|
    |-|-|-|
    |id|[out]|只读；实例 ID|

11. **Handle** 类 — 可构造的 team 作用域异步操作句柄。

    ```python
    class Handle:
        def __init__(self, team_id: int=ACLSHMEM_TEAM_WORLD) -> None:

        @property
        def team_id(self) -> int:
    ```

    |属性/参数|方向|含义|
    |-|-|-|
    |team_id|[in/out]|team ID，默认 ``ACLSHMEM_TEAM_WORLD``；构造和赋值时必须引用 0 到 2047 范围内且当前仍存活的 team|

12. **ProfData** 类 — 当前 PE 的 profiling 数据深拷贝快照。

    ```python
    class ProfData:
        @property
        def pe_id(self) -> int:

        @property
        def ccount(self) -> list[list[int]]:

        @property
        def cycles(self) -> list[list[int]]:
    ```

    |属性|方向|含义|
    |-|-|-|
    |pe_id|[out]|只读；profiling 数据所属 PE 编号|
    |ccount|[out]|只读；尺寸为 ``64 x 1024`` 的计数二维整数列表|
    |cycles|[out]|只读；尺寸为 ``64 x 1024`` 的周期二维整数列表|

13. **InitMode** 枚举类 — ``aclshmemx_init_attr`` 使用的初始化方式。

    ```python
    class InitMode(Enum):
        DEFAULT
        MPI
        UNIQUEID
    ```

    |枚举值|含义|
    |-|-|
    |DEFAULT|使用默认初始化方式|
    |MPI|使用 MPI 初始化方式|
    |UNIQUEID|使用唯一 ID 初始化方式；支持多实例初始化|

## Python 接口约束与限制

- ``buffer`` / ``calloc`` / ``align`` / ``free``、``barrier`` / ``sync`` 及初始化、销毁均含集合语义；所有参与 PE 必须以相同参数和相同顺序调用。测试应使用 ``torchrun``，并设置外部超时以发现失配或死锁。
- ``barrier`` 只完成 Host 发起的通信操作；若要从 Host 等待 NPU 发起的操作，需要同步相应 ACL Stream，或使用 stream 版本的 barrier。
- stream 接口只将操作排入流，不隐式同步。高层接口拒绝 bool、负数和非整数；``0`` 表示默认流。非零 stream 整数必须指向调用期间仍有效的 ACL stream。
- 公开 ``Buffer`` 拒绝零地址、bool、负值以及超过当前平台地址或长度上限的参数。带实例归属的 Buffer 只能在其分配实例处用于 peer 地址换算、RMA/Signal 和释放；跨实例调用引发 ``AclshmemInvalid``。
- ``handle_wait`` 的参与范围由 Handle 绑定的 team 决定；所有 team 成员必须以匹配顺序参与。该操作只入流，须同步相应 stream 后才能观测完成。
- stream RMA 的具体传输路径由初始化配置、目标 PE 和运行时能力共同决定；调用者不应依赖某次调用选择的具体引擎。``signal_wait`` 等待本地地址，不提供远程等待语义。
- 多实例当前最多 255 个，高层接口及对应低层入口均只接受 0 到 254 范围内的非 bool ``instance_id``；单实例 heap 上限 128GB；非 0 实例仅保留 world team，team split/translate/get_config 等高级 team 操作不支持。
- ``HOST_SIDE`` 对称堆依赖包含 ``HAS_ACLRT_MEM_FABRIC_HANDLE`` 能力的 CANN 运行时；不支持时分配返回失败，高层接口引发 ``AclshmemError``。

## shmem API（Python 顶层封装）

以下 Tensor 接口由 ``shmem`` 包顶层提供，可直接通过 ``shmem.<接口名>`` 调用。它们负责对称内存中 Tensor 的创建、释放和元信息维护；调用者须遵守各条目说明的集合参与和生命周期约束。

### 对外接口

1. **aclshmem_create_tensor** — 在对称内存上分配指定形状和数据类型的 torch.Tensor。这是一个集合（collective）操作，底层调用 ``aclshmem_malloc``，所有 PE 必须以相同的顺序参与调用，且各 PE 分配的 shape/dtype（即等价的内存大小）必须相同，否则会破坏对称堆布局。

    ```python
    def aclshmem_create_tensor(shape, dtype: torch.dtype=torch.float32, device_id=0) -> torch.Tensor
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |shape|[in]|张量形状，如 ``(2, 3)``|
    |dtype|[in]|张量数据类型（``torch.dtype``），默认 ``torch.float32``|
    |device_id|[in]|NPU 设备编号，默认 0|
    |返回值|[out]|建立在对称内存上的 ``torch.Tensor``，分配失败引发异常|

2. **aclshmem_free_tensor** — 释放由 ``aclshmem_create_tensor`` 分配的张量所对应的对称内存。这是一个集合（collective）操作，底层调用 ``aclshmem_free``，所有 PE 必须以与分配时相同的顺序参与调用；释放完成后不得再访问该张量及其地址。

    ```python
    def aclshmem_free_tensor(tensor: torch.Tensor) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |tensor|[in]|待释放的对称内存张量|
    |返回值|-|无返回值|
