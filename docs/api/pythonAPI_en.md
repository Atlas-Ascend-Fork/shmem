# SHMEM Python API Reference
## shmem.core API
### External APIs
1. **get_version** — Obtain the current library version. The ACLSHMEM library version is returned.

    ```python
    def get_version() -> str
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|A string containing the version information, in the format of `"libaclshmem_version=X.Y"`|

2. **get_unique_id** — Generate a unique ID for UID initialization. The API should be called by a single process (for example, rank 0) and broadcast to other processes.

    ```python
    def get_unique_id(empty: bool=False) -> bytes
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |empty|[in]|Reserved. It has no actual meaning.|
    |Return Value|[out]|Serialized native unique ID as `bytes`, suitable for object broadcast. If generation fails, `AclshmemError` is raised.|

3. **init** — Initialize the ACLSHMEM runtime using a unique ID. This is a collective operation and must be called by all PEs.

    ```python
    def init(device: int=None, uid: Optional[Union[UniqueID, bytes]]=None, rank: int=None, nranks: int=None, mpi_comm=None, initializer_method: str="", mem_size: int=None, instance_id: int=0) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |device|[in]|Reserved. It has no actual meaning.|
    |uid|[in]|(Required) Unique identifier for initialization.|
    |rank|[in]|(Required) Rank (0-based) of the current process in the ACLSHMEM job.|
    |nranks|[in]|(Required) Total number of processes involved in the ACLSHMEM job.|
    |mpi_comm|[in]|Reserved. It has no actual meaning.|
    |initializer_method|[in]|Initialization method. The value must be `"uid"`.|
    |mem_size|[in]|(Required) Size of the symmetric memory allocated to each PE, in bytes.|
    |instance_id|[in]|Instance ID. It must be a non-bool integer in the range 0 to 254. The default is 0.|
    |Return Value|-|No return value. `AclshmemInvalid` is thrown due to missing parameters, and `AclshmemError` is thrown due to an initialization failure.|

4. **finalize** — Destroy the ACLSHMEM runtime and resource allocations. Each process should call this API once after completing all ACLSHMEM operations.

    ```python
    def finalize(instance_id: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |instance_id|[in]|If omitted, finalize the active instance. If supplied, finalize that instance; it must be a non-bool integer in the range 0 to 254.|
    |Return Value|-|No return value. Invalid IDs raise `AclshmemInvalid`; finalization failures raise `AclshmemError`.|

5. **buffer** — Allocate an NPU buffer supported by ACLSHMEM. This is a collective operation and must be called by all PEs synchronously.

    ```python
    def buffer(size, release=False, except_on_del=True, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |size|[in]|Size of the buffer to be allocated, in bytes|
    |release|[in]|Reserved. It has no actual meaning.|
    |except_on_del|[in]|Reserved. It has no actual meaning.|
    |mem_type|[in]|Symmetric heap type: `MemType.DEVICE_SIDE` or `MemType.HOST_SIDE`. The default is device-side memory.|
    |Return Value|[out]|Original memory buffer represented by the address and byte length If the allocation fails, `AclshmemError` is thrown.|

6. **free** — Free the buffer allocated by `buffer()`. This is a collective operation.

    ```python
    def free(buf: Buffer) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |buf|[in]|Buffer to be freed|
    |Return Value|-|No return value.|

7. **get_peer_buffer** — Convert a local symmetric address to the corresponding symmetric address on a specified PE. The access mode supported by the returned address depends on the transfer engine and topology.

    ```python
    def get_peer_buffer(buf: Buffer, pe: int) -> Buffer
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |buf|[in]|Symmetric address on the local PE|
    |pe|[in]|PE ID|
    |Return Value|[out]|Corresponding symmetric address buffer on the specified PE. If the input address or PE is invalid, `AclshmemError` is thrown.|

8. **put_signal** — Copy contiguous data from the local PE to the symmetric memory address of a specified PE and update the remote signal variable after the copy completes. This call completes the operation before returning; all addresses must remain valid for the duration of the call.

    ```python
    def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int, stream=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Symmetric address of the target data on the remote PE|
    |src|[in]|Address of the source data in the local memory|
    |signal_var|[in]|Symmetric address of the signal word to be updated on the remote PE; at least 4 bytes and 4-byte aligned|
    |signal_val|[in]|Value of the signal variable to be updated|
    |signal_operation|[in]|Signal variable update operation. Supported operation: `SignalOp.SIGNAL_SET` or `SignalOp.SIGNAL_ADD`|
    |remote_pe|[in]|Required non-bool integer satisfying `0 <= remote_pe < pe_count()`|
    |stream|[in]|Reserved and ignored; this interface does not provide explicit stream ordering.|
    |Return Value|-|No return value.|

9. **put** — Copy contiguous data from the local PE to symmetric memory on a specified PE, ordered on the supplied stream. The host call only enqueues the operation; synchronize that stream before observing completion.

    ```python
    def put(dst: Buffer, src: Buffer, remote_pe: int, stream: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Symmetric address of the target data on the remote PE|
    |src|[in]|Address of the source data in the local memory|
    |remote_pe|[in]|Required non-bool integer satisfying `0 <= remote_pe < pe_count()`|
    |stream|[in]|ACL stream object, which is used for sorting. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

10. **get** — Copy contiguous data from symmetric memory on a specified PE to a local buffer, ordered on the supplied stream. The host call only enqueues the operation; synchronize that stream before reading the result.

    ```python
    def get(dst: Buffer, src: Buffer, remote_pe: int, stream: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Address of the target data in the local memory|
    |src|[in]|Symmetric address of the source data on the remote PE|
    |remote_pe|[in]|Required non-bool integer satisfying `0 <= remote_pe < pe_count()`|
    |stream|[in]|ACL stream object, which is used for sorting. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

11. **signal_op** — Update a remote signal variable on a specified PE, ordered on the supplied stream. The host call only enqueues the operation; synchronize that stream before observing completion.

    ```python
    def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int, stream: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |signal_var|[in]|Local address of the signal variable that can be accessed on the target PE|
    |signal_val|[in]|Value used for atomic operations|
    |signal_operation|[in]|Operation performed on a remote signal. Supported operation: `SignalOp.SIGNAL_SET` or `SignalOp.SIGNAL_ADD`|
    |remote_pe|[in]|Required non-bool integer satisfying `0 <= remote_pe < pe_count()`|
    |stream|[in]|Non-bool, non-negative ACL stream address. `0` is the default stream; `None` is rejected.|
    |Return Value|-|No return value. Invalid buffers, enum values, PE IDs, or streams raise `AclshmemInvalid`.|

12. **signal_wait** — Wait until a symmetric signal variable satisfies the specified comparison. The wait is ordered on the supplied stream and the host call returns immediately; after synchronizing that stream, `signal_var` `cmp` `signal_val` is guaranteed to be true.

    ```python
    def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |signal_var|[in]|Local address of the source signal variable|
    |signal_val|[in]|Value to be compared with the value pointed to by signal_var|
    |signal_operation|[in]|Comparison operator. Supported comparison operator: `ComparisonType.CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`, `CMP_LT`, or `CMP_LE`|
    |stream|[in]|Non-bool, non-negative ACL stream address. `0` is the default stream; `None` is rejected.|
    |Return Value|-|No return value. Invalid buffers, comparison values, or streams raise `AclshmemInvalid`.|

13. **quiet** — Enqueue a completion point so that symmetric-data operations previously issued by the calling context finish before that point on the supplied stream. The host call returns immediately; synchronize the stream before observing completion.

    ```python
    def quiet(stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |stream|[in]|Non-bool, non-negative ACL stream address. `0` is the default stream; `None` is rejected.|
    |Return Value|-|No return value. Invalid streams raise `AclshmemInvalid`.|

14. **my_pe** — Obtain the local PE ID.

    ```python
    def my_pe() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Local PE ID|

15. **team_my_pe** — Obtain the PE ID of the current process in a specified team.

    ```python
    def team_my_pe(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Target team ID|
    |Return Value|[out]|PE ID in the specified team. If the team is invalid, `-1` is returned.|

16. **n_pes** — Obtain the total number of PEs running in the program (world team dimension).

    ```python
    def n_pes() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Total number of PEs|

17. **team_n_pes** — Obtain the total number of PEs in a specified team.

    ```python
    def team_n_pes(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Target team ID|
    |Return Value|[out]|The number of PEs in the specified team. If the team is invalid, `-1` is returned.|

18. **init_status** — Query the current initialization status of the shared memory module.

    ```python
    def init_status() -> InitStatus
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|The initialization status is returned. The enumerated values are as follows: `NOT_INITIALIZED`, `SHM_CREATED`, `INITIALIZED`, and `INVALID`.|

19. **calloc** — Allocate zero-initialized memory by element count from a selected symmetric heap. This is a collective operation.

    ```python
    def calloc(count: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |count|[in]|Number of elements; must be a positive integer|
    |size|[in]|Size of one element in bytes; must be positive and `count * size` must fit in `size_t`|
    |mem_type|[in]|Symmetric heap type: `MemType.DEVICE_SIDE` or `MemType.HOST_SIDE`|
    |Return Value|[out]|Zero-initialized owning `Buffer` with length `count * size`; allocation failure raises `AclshmemError`|

20. **align** — Allocate power-of-two-aligned memory from a selected symmetric heap. This is a collective operation.

    ```python
    def align(alignment: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |alignment|[in]|Alignment in bytes; must be a positive power of two|
    |size|[in]|Allocation size in bytes; must be positive and fit in `size_t`|
    |mem_type|[in]|Symmetric heap type: `MemType.DEVICE_SIDE` or `MemType.HOST_SIDE`|
    |Return Value|[out]|Owning `Buffer` with the requested alignment; allocation failure raises `AclshmemError`|

21. **barrier** — Block the current host thread until all PEs in a specified team reach the barrier. This is a collective operation.

    ```python
    def barrier(team: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Currently live team ID in the range 0 to 2047|
    |Return Value|-|No return value; a never-created, destroyed, or out-of-range team raises `AclshmemInvalid`|

22. **barrier_all** — Block the current host thread until all PEs in the world team reach the barrier. This is a collective operation.

    ```python
    def barrier_all() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value; all PEs must call in the same order|

23. **sync** — Synchronize all PEs in a specified team and make prior host stores complete and visible. This is a collective operation.

    ```python
    def sync(team: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Currently live team ID in the range 0 to 2047|
    |Return Value|-|No return value; a never-created, destroyed, or out-of-range team raises `AclshmemInvalid`|

24. **sync_all** — Synchronize all PEs in the world team and make prior host stores complete and visible. This is a collective operation.

    ```python
    def sync_all() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value; all PEs must call in the same order|

25. **barrier_on_stream** — Enqueue a barrier for a specified team on an ACL stream. The API does not synchronize the stream implicitly.

    ```python
    def barrier_on_stream(team: int, stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Currently live team ID in the range 0 to 2047|
    |stream|[in]|Non-bool, nonnegative integer `aclrtStream` address; `0` selects the default stream|
    |Return Value|-|No return value; an invalid team or stream raises `AclshmemInvalid`|

26. **barrier_all_on_stream** — Enqueue a world-team barrier on an ACL stream. The API does not synchronize the stream implicitly.

    ```python
    def barrier_all_on_stream(stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |stream|[in]|Non-bool, nonnegative integer `aclrtStream` address; `0` selects the default stream|
    |Return Value|-|No return value; an invalid stream raises `AclshmemInvalid`|

27. **current_instance** — Obtain the currently active SHMEM instance ID.

    ```python
    def current_instance() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Active instance ID; raises `AclshmemError` if no instance context is available|

28. **set_instance** — Switch the active context to a specified SHMEM instance.

    ```python
    def set_instance(instance_id: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |instance_id|[in]|Initialized target instance ID; must be a non-bool integer in the range 0 to 254|
    |Return Value|-|No return value; invalid IDs raise `AclshmemInvalid` and switch failures raise `AclshmemError`|

29. **multi_instance** — Switch instances within a context-manager scope and restore the original instance when leaving the scope.

    ```python
    @contextmanager
    def multi_instance(instance_id: int)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |instance_id|[in]|Initialized instance ID used inside the scope; must be a non-bool integer from 0 to 254|
    |Return Value|[out]|Context manager that switches to the target instance on entry and restores the previous instance on exit|

30. **set_mte_config** — Configure MTE workspace parameters for the active instance. Call it after initializing and selecting that instance and before issuing operations that use the configuration.

    ```python
    def set_mte_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|Workspace offset/address in bytes; a non-bool integer in `[0, 2**64 - 1]`|
    |ub_size|[in]|Workspace size in bytes; a non-bool integer in `[0, 2**32 - 1]`|
    |sync_id|[in]|Synchronization identifier; a non-bool integer in `[0, 2**32 - 1]`|
    |Return Value|-|No return value; invalid arguments raise `AclshmemInvalid`, and failure to apply the configuration to the active instance raises `AclshmemError`|

31. **set_sdma_config** — Configure SDMA workspace parameters for the active instance. Call it after initializing and selecting that instance and before issuing operations that use the configuration. Availability depends on the current platform and runtime.

    ```python
    def set_sdma_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|Workspace offset/address in bytes; a non-bool integer in `[0, 2**64 - 1]`|
    |ub_size|[in]|Workspace size in bytes; a non-bool integer in `[0, 2**32 - 1]`|
    |sync_id|[in]|Synchronization identifier; a non-bool integer in `[0, 2**32 - 1]`|
    |Return Value|-|No return value; invalid arguments raise `AclshmemInvalid`, and failure to apply the configuration to the active instance raises `AclshmemError`|

32. **set_rdma_config** — Configure RDMA workspace parameters for the active instance. Call it after initializing and selecting that instance and before issuing operations that use the configuration. Availability depends on the current platform and runtime.

    ```python
    def set_rdma_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|Workspace offset/address in bytes; a non-bool integer in `[0, 2**64 - 1]`|
    |ub_size|[in]|Workspace size in bytes; a non-bool integer in `[128, 2**32 - 1]`|
    |sync_id|[in]|Synchronization identifier; a non-bool integer in `[0, 2**32 - 1]`|
    |Return Value|-|No return value; invalid arguments raise `AclshmemInvalid`, and failure to apply the configuration to the active instance raises `AclshmemError`|

33. **set_udma_config** — Configure UDMA workspace parameters for the active instance. Call it after initializing and selecting that instance and before issuing operations that use the configuration. Availability depends on the current platform and runtime.

    ```python
    def set_udma_config(offset: int, ub_size: int, sync_id: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|Workspace offset/address in bytes; a non-bool integer in `[0, 2**64 - 1]`|
    |ub_size|[in]|Workspace size in bytes; a non-bool integer in `[128, 2**32 - 1]`|
    |sync_id|[in]|Synchronization identifier; a non-bool integer in `[0, 2**32 - 1]`|
    |Return Value|-|No return value; invalid arguments raise `AclshmemInvalid`, and failure to apply the configuration to the active instance raises `AclshmemError`|

34. **handle_wait** — Enqueue a completion wait and member rendezvous for the Handle's team on the supplied stream. Every PE in that team must participate in matching order after issuing the asynchronous operations to be covered; only a Handle for `ACLSHMEM_TEAM_WORLD` requires every PE. The host call only enqueues the operation, so synchronize the stream before observing completion.

    ```python
    def handle_wait(handle: Handle, stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |handle|[in]|`Handle` bound to a currently active team; that team defines the participating PEs|
    |stream|[in]|Non-bool, nonnegative integer `aclrtStream` address; `0` selects the default stream|
    |Return Value|-|No return value; an invalid Handle, team, or stream raises `AclshmemInvalid`. Mismatched participation or call order can wait indefinitely|

35. **get_prof** — Obtain a deep copy of profiling data for the current instance and PE.

    ```python
    def get_prof(verbose: bool=False) -> Optional[ProfData]
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |verbose|[in]|Whether to also print profiling data; default is `False`|
    |Return Value|[out]|The PE selected by `SHMEM_CYCLE_PROF_PE` returns `ProfData`; other PEs return `None`; data is isolated per instance|

36. **show_prof** — Print profiling data for the current instance.

    ```python
    def show_prof() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value; compatibility API, with `get_prof` preferred for new code|
1. UniqueId class: unique identifier handle for UID initialization

### Class
1. **UniqueId** class — Unique identifier handle for UID initialization

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |version|[out]|Version|
    |my_pe|[out]|PE ID of the current process|
    |n_pes|[out]|Total number of PEs of all processes|
    |internal|[out]|Internal information about the UID (bytes)|

2. **InitStatus** enumeration class — Initialization statuses of the shared memory module

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |NOT_INITIALIZED|Not initialized|
    |SHM_CREATED|Shared memory created|
    |INITIALIZED|Initialization completed|
    |INVALID|Invalid|

3. **SignalOp** enumeration class — Atomic operation types of signal variables

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |SIGNAL_SET|Atomic setting: writes a given value to a remote signal.|
    |SIGNAL_ADD|Atomic addition: adds a given value to the existing value of a remote signal.|

4. **ComparisonType** enumeration class — Signal waiting comparison operation types

    ```python
    class ComparisonType(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |CMP_EQ|Equal to (==)|
    |CMP_NE|Not equal to (!=)|
    |CMP_GT|Greater than (>)|
    |CMP_GE|Greater than or equal to (>=)|
    |CMP_LT|Less than (<)|
    |CMP_LE|Less than or equal to (<=)|

5. **MemType** enumeration class — Symmetric heap type.

    ```python
    class MemType(Enum):
        HOST_SIDE
        DEVICE_SIDE
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |HOST_SIDE|Host-side symmetric heap; availability depends on the CANN runtime|
    |DEVICE_SIDE|Device-side symmetric heap and the default value|

6. **Buffer** class — Describe memory used by SHMEM APIs through its address, length, and symmetric heap type.

    ```python
    class Buffer:
        def __init__(
            self,
            addr: int,
            length: int,
            mem_type: MemType=MemType.DEVICE_SIDE,
        ) -> None:
    ```

    **Attributes**

    |Attribute|Direction|Meaning|
    |-|-|-|
    |addr|[in/out]|Start address of the memory|
    |length|[in/out]|Memory length in bytes|
    |mem_type|[in/out]|Symmetric heap type: `MemType.DEVICE_SIDE` or `MemType.HOST_SIDE`|
    |owned|[out]|Read-only; whether this object owns the right to free the memory|
    |instance_id|[out]|Read-only; instance ID for a factory-allocated Buffer, or `None` for an external-address descriptor|
    |release_called|[out]|Read-only; whether a free operation has been initiated for the memory|

    **Functionality and Constraints**

    - Direct `Buffer(addr, length)` construction describes an external address and does not own the memory, so it cannot be passed to `free`. Buffers returned by `buffer`, `calloc`, or `align` own the right to free their memory.
    - `addr` and `length` must be non-bool integers satisfying `0 < addr <= INTPTR_MAX` and `0 < length <= SIZE_T_MAX`.
    - A factory-allocated Buffer and its peer views can be used for `free`, peer-address translation, and RMA/Signal operations only while their allocation instance is active. An instance mismatch raises `AclshmemInvalid`.
    - After a free operation is initiated, neither the original Buffer nor its peer views can be reused for free, peer-address translation, or RMA/Signal operations. Repeated free calls are rejected.
    - Buffer destruction does not automatically free symmetric memory. Free is collective and must be called explicitly by all PEs in the same order.

7. **InstanceContext** class — Read-only value snapshot of the active instance context.

    ```python
    class InstanceContext:
        @property
        def id(self) -> int:
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |id|[out]|Read-only instance ID|

8. **Handle** class — Team-scoped asynchronous-operation handle.

    ```python
    class Handle:
        def __init__(self, team_id: int=ACLSHMEM_TEAM_WORLD) -> None:

        @property
        def team_id(self) -> int:
    ```

    |Attribute/Parameter|Direction|Meaning|
    |-|-|-|
    |team_id|[in/out]|Team ID; defaults to `ACLSHMEM_TEAM_WORLD` and must refer to a currently live team in the range 0 to 2047 on construction and assignment|

    A bool, out-of-range, never-created, or destroyed `team_id` raises `ValueError`.

9. **ProfData** class — NumPy-independent deep-copy snapshot of profiling data for the current PE.

    ```python
    class ProfData:
        @property
        def pe_id(self) -> int:

        @property
        def ccount(self) -> list[list[int]]:

        @property
        def cycles(self) -> list[list[int]]:
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |pe_id|[out]|Read-only PE ID associated with the profiling data|
    |ccount|[out]|Read-only ``64 x 1024`` two-dimensional integer list of counts|
    |cycles|[out]|Read-only ``64 x 1024`` two-dimensional integer list of cycles|

## shmem._pyshmem API
### External APIs
1. **aclshmem_init** — Initialize the shared memory module. This is a collective operation.
    ```python
    def aclshmem_init(attributes: InitAttr) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |attributes|[in]|Initialization attributes (of the `InitAttr` type), including `my_pe` (local PE index, ranging from `0` to `n_pes-1`), `n_pes` (total number of PEs), and `local_mem_size` (size of the memory allocated to each PE, in bytes)|
    |Return Value|[out]|Result code. Success: 0. Failure: -1.|

2. **aclshmem_finalize** — Destroy the shared memory module and release all resources.
    ```python
    def aclshmem_finalize() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |No parameter|[in]|-|
    |Return Value|[out]|Result code. Success: 0. Failure: error code. Use `aclshmemx_finalize(instance_id)` to finalize a selected instance.|

3. **aclshmemx_init_status** — Query the current initialization status of the shared memory module.
    ```python
    def aclshmemx_init_status() -> InitStatus
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Enumeration value of the initialization status. `INITIALIZED` indicates that the initialization is complete.|

4. **set_conf_store_tls_key** — Set the TLS private key and password, and register the decryption callback function.
    ```python
    def set_conf_store_tls_key(tls_pk, tls_pk_pw, py_decrypt_func:Callable[[str], str]) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |tls_pk|[in]|TLS private key content|
    |tls_pk_pw|[in]|TLS private key password|
    |py_decrypt_func|[in]|Decryption callback function. It receives `(str cipher_text)` and returns `(str plain_text)`.|
    |Return Value|[out]|Result code. Success: 0. Failure: error code.|

5. **set_conf_store_tls** — Set whether to enable TLS encryption, and specify the TLS configuration information.
    ```python
    def set_conf_store_tls(enable, tls_info) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |enable|[in]|Whether to enable TLS encryption. `True` indicates enabling, and `False` indicates disabling.|
    |tls_info|[in]|TLS configuration information string, which can be an empty string|
    |Return Value|[out]|Result code. Success: 0. Failure: error code.|

6. **aclshmem_malloc** — Allocate symmetric memory. This is a collective operation with an implicit barrier.
    ```python
    def aclshmem_malloc(size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |size|[in]|Size of the allocated memory, in bytes|
    |Return Value|[out]|If the operation is successful, a pointer (int) to the allocated memory is returned. If `size` is `0` or the allocation fails, `0` is returned, and an exception is thrown.|

7. **aclshmem_calloc** — Allocate zero-initialized symmetric memory. This is a collective operation with an implicit barrier.
    ```python
    def aclshmem_calloc(nmemb, size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |nmemb|[in]|The number of elements|
    |size|[in]|Size of each element, in bytes|
    |Return Value|[out]|If the operation is successful, a pointer (int) to the allocated memory is returned. If `nmemb` or `size` is `0`, `0` is returned, and an exception is thrown.|

8. **aclshmem_align** — Allocate symmetric memory with a specified alignment mode. This is a collective operation with an implicit barrier.
    ```python
    def aclshmem_align(alignment, size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |alignment|[in]|Memory alignment requirement (The value must be a power of 2.)|
    |size|[in]|The number of bytes to be allocated|
    |Return Value|[out]|If the operation is successful, a pointer (int) to the allocated memory is returned. If the allocation fails, an exception is thrown.|

9. **aclshmem_free** — Free the memory allocated by the symmetric memory allocation function. This is a collective operation with an implicit barrier.
    ```python
    def aclshmem_free(ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |ptr|[in]|Pointer to the memory to be freed|
    |Return Value|-|No return value.|

10. **aclshmem_ptr** — Convert a local symmetric address to the corresponding symmetric address on a specified PE. The access mode supported by the returned address depends on the transfer engine and topology.
    ```python
    def aclshmem_ptr(ptr, peer) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |ptr|[in]|Symmetric address on the local PE|
    |peer|[in]|PE ID|
    |Return Value|[out]|If the operation is successful, the corresponding symmetric address (int) on the specified PE is returned. If the input address or PE is invalid, `0` is returned.|

11. **aclshmemx_get_heap_base** — Obtain the start address of the local symmetric memory heap.
    ```python
    def aclshmemx_get_heap_base(mem_type: MemType=None) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |mem_type|[in]|Symmetric memory allocation location: `MemType.HOST_SIDE` (host side) / `MemType.DEVICE_SIDE` (device side, default)|
    |Return Value|[out]|If the operation is successful, the pointer (int) to the start address of the symmetric memory heap is returned. If the module is not initialized, `0` is returned.|

12. **my_pe** — Obtain the PE ID (world team dimension).
    ```python
    def my_pe() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Local PE ID|

13. **team_my_pe** — Obtain the PE ID in a specified team.
    ```python
    def team_my_pe(team_id) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team_id|[in]|Handle of the team|
    |Return Value|[out]|ID of the PE in the specified team. If an error occurs, `-1` is returned.|

14. **pe_count** — Obtain the total number of PEs (world team dimension).
    ```python
    def pe_count() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Total number of PEs|

15. **team_n_pes** — Obtain the number of PEs in a specified team.
    ```python
    def team_n_pes(team_id) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team_id|[in]|Handle of the team|
    |Return Value|[out]|The number of PEs in the specified team. If an error occurs, `-1` is returned.|

16. **team_split_strided** — Split child teams from the existing parent team by stride. This is a collective operation.
    ```python
    def team_split_strided(parent, start, stride, size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |parent|[in]|Parent team ID|
    |start|[in]|Start PE ID of a child team|
    |stride|[in]|Stride between PE IDs|
    |size|[in]|The number of PEs in a child team|
    |Return Value|[out]|Returns the new team ID when successful and the current PE is a member; returns `ACLSHMEM_TEAM_INVALID` for a non-member PE; raises `RuntimeError` when the native call returns an error code.|

17. **team_split_2d** — Split a team from the parent team based on the two-dimensional Cartesian space. This is a collective operation.
    ```python
    def team_split_2d(parent, x_range) -> tuple
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |parent|[in]|Handle of the parent team|
    |x_range|[in]|The number of elements in the first dimension|
    |Return Value|[out]|If the operation is successful, the (x_team_id, y_team_id) tuple is returned. If the operation fails, an exception is thrown.|

18. **aclshmem_team_get_config** — Obtain the team configuration passed during team creation.
    ```python
    def aclshmem_team_get_config(team) -> TeamConfig
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|team ID|
    |Return Value|[out]|If the operation is successful, the `TeamConfig` object is returned. If the operation fails, an exception is thrown.|

19. **aclshmem_putmem** — Copy contiguous data from the local PE to the symmetric address of a specified PE. This is a synchronous (blocking) API.
    ```python
    def aclshmem_putmem(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

20. **aclshmem_getmem** — Copy contiguous data from the symmetric memory of a specified PE to the local PE. This is a synchronous (blocking) API.
    ```python
    def aclshmem_getmem(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

21. **aclshmem_{TYPE}_iput** — Copy the data arranged by the sst stride in the local memory to the corresponding position in the symmetric memory of a specified PE by the dst stride. This is a synchronous (blocking) API.
    ```python
    def aclshmem_{TYPE}_iput(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `double`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |dest|[in]|Symmetric memory pointer to the remote target data|
    |source|[in]|Pointer to the local source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

22. **aclshmem_{TYPE}_iget** — Copy the data arranged by the sst stride in the symmetric memory of a remote PE to the corresponding position of the local memory by the dst stride. This is a synchronous (blocking) API.
    ```python
    def aclshmem_{TYPE}_iget(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `double`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |dest|[in]|Pointer to the local target data|
    |source|[in]|Symmetric memory pointer to the remote source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

23. **aclshmem_put{BITS}** — Copy contiguous data from the local PE to the symmetric memory address of a specified PE. This is a synchronous (blocking) API.
    ```python
    def aclshmem_put{BITS}(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Number of BITS-bit elements; the transferred byte count is elem_size * (BITS / 8)|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

24. **aclshmem_get{BITS}** — Copy contiguous data from the symmetric memory of a specified PE to the local PE. This is a synchronous (blocking) API.
    ```python
    def aclshmem_get{BITS}(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Number of BITS-bit elements; the transferred byte count is elem_size * (BITS / 8)|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

25. **aclshmem_iput{BITS}** — Copy the data arranged by the sst stride in the local memory to the corresponding position (bit width version) of a specified PE by the dst stride. This is a synchronous (blocking) API.
    ```python
    def aclshmem_iput{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dest|[in]|Symmetric memory pointer to the remote target data|
    |source|[in]|Pointer to the local source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

26. **aclshmem_iget{BITS}** — Copy the data arranged by the sst stride in the symmetric memory of a remote PE to the corresponding position of the local memory by the dst stride (bit width version). This is a synchronous (blocking) API.
    ```python
    def aclshmem_iget{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dest|[in]|Pointer to the local target data|
    |source|[in]|Symmetric memory pointer to the remote source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

27. **aclshmem_info_get_version** — Return the major and minor version numbers of the library.
    ```python
    def aclshmem_info_get_version() -> tuple
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|(major, minor) tuple|

28. **aclshmem_info_get_name** — Return the vendor-defined name string.
    ```python
    def aclshmem_info_get_name() -> str
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Vendor-defined name string|

29. **team_translate_pe** — Convert a given PE ID in one team to the corresponding PE ID in another team.
    ```python
    def team_translate_pe(src_team, src_pe, dest_team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |src_team|[in]|Source team ID|
    |src_pe|[in]|Source PE ID|
    |dest_team|[in]|Target team ID|
    |Return Value|[out]|If the operation is successful, the ID of the corresponding PE in the target team is returned. If an error occurs, `-1` is returned.|

30. **team_destroy** — Destroy a team.
    ```python
    def team_destroy(team) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|ID of the team to be destroyed|
    |Return Value|-|No return value.|

31. **get_ffts_config** — Obtain the runtime FFTS configuration.
    ```python
    def get_ffts_config() -> str
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|FFTS configuration string|

32. **set_log_level** — Set the log level of all modules.
    ```python
    def set_log_level(level) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |level|[in]|Log level|
    |Return Value|[out]|Result code. Success: 0. Failure: error code.|

33. **set_extern_logger** — Register an external log callback function to take over the log output of all modules.
    ```python
    def set_extern_logger(func) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |func|[in]|Log callback function. The signature is `func(level: int, msg: str)`|
    |Return Value|[out]|Result code. Success: 0. Failure: error code.|

34. **aclshmem_putmem_nbi** — Copy contiguous data from the local PE to the symmetric address of a specified PE. This is an asynchronous (non-blocking) API.
    ```python
    def aclshmem_putmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

35. **aclshmem_getmem_nbi** — Copy contiguous data from the symmetric memory of a specified PE to the local PE. This is an asynchronous (non-blocking) API.
    ```python
    def aclshmem_getmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

36. **aclshmem_put{BITS}_nbi** — Copy contiguous data from the local PE to the symmetric memory address of a specified PE (bit width version). This is an asynchronous (non-blocking) API.
    ```python
    def aclshmem_put{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Number of BITS-bit elements; the transferred byte count is elem_size * (BITS / 8)|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

37. **aclshmem_get{BITS}_nbi** — Copy contiguous data from the symmetric memory of a specified PE to the local PE (bit width version). This is an asynchronous (non-blocking) API.
    ```python
    def aclshmem_get{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Number of BITS-bit elements; the transferred byte count is elem_size * (BITS / 8)|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

38. **aclshmemx_putmem_signal_nbi** — Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal variable after the operation is complete. This is an asynchronous (non-blocking) API.
    ```python
    def aclshmemx_putmem_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `SIGNAL_SET` / `SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

39. **aclshmemx_putmem_signal** — Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal variable. This is a synchronous (blocking) API.
    ```python
    def aclshmemx_putmem_signal(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `SIGNAL_SET` / `SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

40. **aclshmemx_put{BITS}_signal_nbi** — Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal (bit width version). This is an asynchronous (non-blocking) API.
    ```python
    def aclshmemx_put{BITS}_signal_nbi(dst, src, nelems, sig, signal, sig_op, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |nelems|[in]|Number of BITS-bit elements; the transferred byte count is nelems * (BITS / 8)|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `SIGNAL_SET` / `SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

41. **aclshmemx_put{BITS}_signal** — Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal (bit width version). This is a synchronous (blocking) API.
    ```python
    def aclshmemx_put{BITS}_signal(dst, src, nelems, sig, signal, sig_op, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |nelems|[in]|Number of BITS-bit elements; the transferred byte count is nelems * (BITS / 8)|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `SIGNAL_SET` / `SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

42. **aclshmem_global_exit** — All PEs call exit() through broadcast to exit the process.
    ```python
    def aclshmem_global_exit(status) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |status|[in]|Status value passed to exit()|
    |Return Value|-|No return value.|

43. **my_pe** — Obtain the PE ID in a specified team.
    ```python
    def my_pe(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|team ID|
    |Return Value|[out]|ID of the PE in the specified team. If an error occurs, `-1` is returned.|

44. **pe_count** — Obtain the number of PEs in a specified team.
    ```python
    def pe_count(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|team ID|
    |Return Value|[out]|The number of PEs in the specified team. If an error occurs, `-1` is returned.|

45. **aclshmem_signal_wait_until** — Wait until the signal variable meets the comparison condition `*sig_addr cmp cmp_val`. This is a blocking API.
    ```python
    def aclshmem_signal_wait_until(sig_addr, cmp, cmp_val) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |sig_addr|[in]|Local address of the source signal variable|
    |cmp|[in]|Comparison operator. Supported comparison operator: `CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`, `CMP_LT`, or `CMP_LE`|
    |cmp_val|[in]|Comparison value|
    |Return Value|[out]|Value of `sig_addr` when the condition is met|

46. **aclshmem_{TYPE}_wait_until** — Wait until a single element meets the comparison condition `ivar cmp cmp_val`. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait_until(ivar, cmp, cmp_val) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivar|[in]|Pointer to the signal variable in the symmetric memory|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |Return Value|-|No return value.|

47. **aclshmem_{TYPE}_wait** — Wait until the signal variable is not equal to the given value. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait(ivar, cmp_val) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivar|[in]|Pointer to the signal variable in the symmetric memory|
    |cmp_val|[in]|Comparison value|
    |Return Value|-|No return value.|

48. **aclshmem_{TYPE}_wait_until_all** — Wait until all elements in the array meet the comparison condition `ivars[i] cmp cmp_val`. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait_until_all(ivars_ptr, nelems, status_ptr, cmp, cmp_val) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |Return Value|-|No return value.|

49. **aclshmem_{TYPE}_wait_until_any** — Wait until at least one element in the array meets the comparison condition `ivars[i] cmp cmp_val`. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait_until_any(ivars_ptr, nelems, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |res_out_ptr|[out]|Index values of elements that meet the comparison condition|
    |Return Value|-|No return value.|

50. **aclshmem_{TYPE}_wait_until_some** — Wait until at least one element in the array meets the comparison condition and return the indexes of all elements that meet the condition. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait_until_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |res_out_ptr|[out]|The number of elements that meet the comparison condition|
    |Return Value|-|No return value.|

51. **aclshmem_{TYPE}_wait_until_all_vector** — Wait until all elements in the array meet the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait_until_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |Return Value|-|No return value.|

52. **aclshmem_{TYPE}_wait_until_any_vector** — Wait until at least one element in the array meets the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait_until_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|Index values of elements that meet the comparison condition|
    |Return Value|-|No return value.|

53. **aclshmem_{TYPE}_wait_until_some_vector** — Wait until at least one element in the array meets the vector comparison condition and return the index values of all elements that meet the condition. This is a blocking API.
    ```python
    def aclshmem_{TYPE}_wait_until_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|The number of elements that meet the comparison condition|
    |Return Value|-|No return value.|

54. **aclshmem_{TYPE}_test** — Check whether a single element meets the comparison condition `ivar cmp cmp_value`. This is a non-blocking query API.
    ```python
    def aclshmem_{TYPE}_test(ivar, cmp, cmp_value, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivar|[in]|Pointer to the signal variable in the symmetric memory|
    |cmp|[in]|Comparison operator|
    |cmp_value|[in]|Comparison value|
    |res_out_ptr|[out]|If the condition is met, `1` is returned. Otherwise, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

55. **aclshmem_{TYPE}_test_any** — Check whether at least one element in the array meets the comparison condition `ivars[i] cmp cmp_value`. This is a non-blocking query API.
    ```python
    def aclshmem_{TYPE}_test_any(ivars_ptr, nelems, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_value|[in]|Comparison value|
    |res_out_ptr|[out]|Index of the element that meets the condition. If no element meets the condition or the test dataset is empty, `SIZE_MAX` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

56. **aclshmem_{TYPE}_test_some** — Check whether at least one element in the array meets the comparison condition and return the index values of all elements that meet the condition. This is a non-blocking query API.
    ```python
    def aclshmem_{TYPE}_test_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_value|[in]|Comparison value|
    |res_out_ptr|[out]|The number of elements that meet the condition. If the test dataset is empty, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr` and `indices_ptr`.|

57. **aclshmem_{TYPE}_test_all_vector** — Check whether all elements in the array meet the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a non-blocking query API.
    ```python
    def aclshmem_{TYPE}_test_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|If all elements meet the condition or `nelems` is `0`, `1` is returned. Otherwise, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

58. **aclshmem_{TYPE}_test_any_vector** — Check whether at least one element in the array meets the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a non-blocking query API.
    ```python
    def aclshmem_{TYPE}_test_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|Index of the first element that meets the condition. If no element meets the condition or the test dataset is empty, `SIZE_MAX` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

59. **aclshmem_{TYPE}_test_some_vector** — Check whether at least one element in the array meets the vector comparison condition and return the indexes of all elements that meet the condition. This is a non-blocking query API.
    ```python
    def aclshmem_{TYPE}_test_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|The number of elements that meet the condition. If the test dataset is empty, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr` and `indices_ptr`.|

60. **aclshmemx_putmem_on_stream** — Copy contiguous data from the local PE to the symmetric address of a specified PE through a specified stream. This is a non-blocking API.
    ```python
    def aclshmemx_putmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |stream|[in]|Integer ACL stream address; pass `0` for the default stream. `None` is not accepted.|
    |Return Value|-|No return value.|

61. **aclshmemx_getmem_on_stream** — Copy contiguous data from the symmetric memory of a specified PE to the local PE through a specified stream. This is a non-blocking API.
    ```python
    def aclshmemx_getmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |stream|[in]|Integer ACL stream address; pass `0` for the default stream. `None` is not accepted.|
    |Return Value|-|No return value.|

62. **aclshmemx_signal_op_on_stream** — Enqueue a remote signal update on the supplied stream. The host call returns immediately; synchronize that stream before observing completion.
    ```python
    def aclshmemx_signal_op_on_stream(sig, signal, sig_op, pe, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |sig|[in]|Local address of the signal variable that can be accessed on the target PE|
    |signal|[in]|Value used for atomic operations|
    |sig_op|[in]|Operation performed on a remote signal. Supported operation: `SIGNAL_SET` / `SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |stream|[in]|Integer ACL stream address; pass `0` for the default stream. `None` is not accepted.|
    |Return Value|-|No return value.|

63. **aclshmemx_signal_wait_until_on_stream** — Enqueue a signal-condition wait on the supplied stream. The host call returns immediately; synchronize that stream before assuming the condition has been satisfied.
    ```python
    def aclshmemx_signal_wait_until_on_stream(sig, cmp, cmp_val, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |sig|[in]|Local address of the source signal variable|
    |cmp|[in]|Comparison operator. Supported comparison operator: `CMP_EQ` / `CMP_NE` / `CMP_GT` / `CMP_GE` / `CMP_LT` / `CMP_LE`|
    |cmp_val|[in]|Value to be compared with the value pointed to by `sig`|
    |stream|[in]|ACL stream object, which is used for sorting. A valid stream must be passed.|
    |Return Value|-|No return value.|

64. **aclshmemx_quiet_on_stream** — Enqueue a completion point so that symmetric-data operations previously issued by the calling context finish before that point on the supplied stream. The host call returns immediately; synchronize the stream before observing completion.
    ```python
    def aclshmemx_quiet_on_stream(stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |stream|[in]|Integer ACL stream address for the quiet operation; pass `0` for the default stream. `None` is not accepted.|
    |Return Value|-|No return value.|

##### Extended Host API Bindings

The following APIs expose the same operations as their Host counterparts. Address and stream parameters are represented by non-bool integers; `0` selects the default stream, and a nonzero stream must remain valid during the call. Entries with collective semantics state their participant set, matching-order requirement, and completion condition explicitly.

65. **aclshmemx_finalize** — Finalize a specified SHMEM instance and release its resources.

    ```python
    def aclshmemx_finalize(instance_id: int) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |instance_id|[in]|Instance ID; accepts only non-bool integers from 0 to 254|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

66. **aclshmemx_set_qp_num** — Configure the number of QPs created for a data operation engine before initialization.

    ```python
    def aclshmemx_set_qp_num(engine: OpEngineType, qp_num: int) -> int
    ```

    The process-wide setting must be identical on every PE. SDMA accepts `[1, min(device vector-core count, ACLSHMEM_MAX_AIV_PER_NPU)]` and creates local device-only streams that are not bound to a peer. UDMA and ROCE create one QP group per peer connection and accept `[1, ACLSHMEM_MAX_QP_NUM]` (currently 56); unsupported engines or backends return `ACLSHMEM_NOT_SUPPORTED`.

67. **aclshmemx_set_attr_uniqueid_args** — Populate an `InitAttr` object from unique-ID initialization arguments.

    ```python
    def aclshmemx_set_attr_uniqueid_args(my_pe: int, n_pes: int, local_mem_size: int, uid: UniqueId, attr: InitAttr) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |my_pe|[in]|Current PE ID satisfying `0 <= my_pe < n_pes`|
    |n_pes|[in]|Total number of PEs in the instance|
    |local_mem_size|[in]|Local symmetric-memory size per PE in bytes|
    |uid|[in]|`UniqueId` used for initialization; after success, the populated `attr` remains usable even if the original `uid` variable is deleted|
    |attr|[in/out]|`InitAttr` object to populate|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

68. **aclshmemx_init_attr** — Initialize a SHMEM instance from an initialization mode and `InitAttr`. This is a collective operation.

    ```python
    def aclshmemx_init_attr(bootstrap_flags: InitMode, attributes: InitAttr) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |bootstrap_flags|[in]|Initialization method; use `InitMode.UNIQUEID` for UID multi-instance initialization|
    |attributes|[in]|Initialization attributes containing PE, memory, instance-ID, and engine options|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

69. **aclshmemx_instance_ctx_get** — Obtain a value snapshot of the active instance context.

    ```python
    def aclshmemx_instance_ctx_get() -> Optional[InstanceContext]
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|`InstanceContext` snapshot containing the current instance ID, or `None` if unavailable|

70. **aclshmemx_instance_ctx_set** — Switch the active runtime context to a specified instance.

    ```python
    def aclshmemx_instance_ctx_set(instance_id: int) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |instance_id|[in]|Target instance ID; accepts only non-bool integers from 0 to 254|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

71. **aclshmemx_malloc** — Allocate memory from a selected symmetric heap. This is a collective operation.

    ```python
    def aclshmemx_malloc(size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |size|[in]|Allocation size in bytes|
    |mem_type|[in]|Symmetric heap type; default is `MemType.DEVICE_SIDE`|
    |Return Value|[out]|Integer address on success or 0 on failure|

72. **aclshmemx_calloc** — Allocate and zero memory from a selected symmetric heap. This is a collective operation.

    ```python
    def aclshmemx_calloc(count: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |count|[in]|Number of elements|
    |size|[in]|Size of one element in bytes|
    |mem_type|[in]|Symmetric heap type; default is `MemType.DEVICE_SIDE`|
    |Return Value|[out]|Integer address of zero-initialized memory on success or 0 on failure|

73. **aclshmemx_align** — Allocate aligned memory from a selected symmetric heap. This is a collective operation.

    ```python
    def aclshmemx_align(alignment: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |alignment|[in]|Alignment in bytes satisfying native alignment requirements|
    |size|[in]|Allocation size in bytes|
    |mem_type|[in]|Symmetric heap type; default is `MemType.DEVICE_SIDE`|
    |Return Value|[out]|Aligned integer address on success or 0 on failure|

74. **aclshmemx_free** — Free memory from a selected symmetric heap. This is a collective operation.

    ```python
    def aclshmemx_free(ptr: int, mem_type: MemType=MemType.DEVICE_SIDE) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |ptr|[in]|Integer address returned by the corresponding allocation API|
    |mem_type|[in]|Symmetric heap type, which must match the allocation|
    |Return Value|-|No return value|

75. **aclshmemx_set_mte_config** — Configure the MTE UB workspace and synchronization ID.

    ```python
    def aclshmemx_set_mte_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|UB offset|
    |ub_size|[in]|UB size|
    |sync_id|[in]|Synchronization ID|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

76. **aclshmemx_set_sdma_config** — Configure the SDMA UB workspace and synchronization ID.

    ```python
    def aclshmemx_set_sdma_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|UB offset|
    |ub_size|[in]|UB size|
    |sync_id|[in]|Synchronization ID|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

77. **aclshmemx_set_rdma_config** — Configure the RDMA UB workspace and synchronization ID.

    ```python
    def aclshmemx_set_rdma_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|UB offset|
    |ub_size|[in]|UB size, at least 128 bytes|
    |sync_id|[in]|Synchronization ID|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

78. **aclshmemx_set_udma_config** — Configure the UDMA UB workspace and synchronization ID.

    ```python
    def aclshmemx_set_udma_config(offset: int, ub_size: int, sync_id: int) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |offset|[in]|UB offset|
    |ub_size|[in]|UB size, at least 128 bytes|
    |sync_id|[in]|Synchronization ID|
    |Return Value|[out]|Returns `ACLSHMEM_SUCCESS` on success or a native error code on failure|

79. **aclshmem_barrier** — Block the host until all PEs in a specified team reach the barrier. This is a collective operation.

    ```python
    def aclshmem_barrier(team: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Currently live team ID in the range 0 to 2047|
    |Return Value|-|No return value; an invalid or stale team raises `ValueError`|

80. **aclshmem_barrier_all** — Block the host until all PEs in the world team reach the barrier. This is a collective operation.

    ```python
    def aclshmem_barrier_all() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value|

81. **aclshmem_sync** — Synchronize all PEs in a specified team. This is a collective operation.

    ```python
    def aclshmem_sync(team: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Currently live team ID in the range 0 to 2047|
    |Return Value|-|No return value; an invalid or stale team raises `ValueError`|

82. **aclshmem_sync_all** — Synchronize all PEs in the world team. This is a collective operation.

    ```python
    def aclshmem_sync_all() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value|

83. **aclshmemx_barrier_on_stream** — Enqueue a barrier for a specified team on an ACL stream.

    ```python
    def aclshmemx_barrier_on_stream(team: int, stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Currently live team ID in the range 0 to 2047|
    |stream|[in]|Integer `aclrtStream` address; `0` selects the default stream|
    |Return Value|-|No return value; an invalid or stale team raises `ValueError`|

84. **aclshmemx_barrier_all_on_stream** — Enqueue a world-team barrier on an ACL stream.

    ```python
    def aclshmemx_barrier_all_on_stream(stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |stream|[in]|Integer `aclrtStream` address; `0` selects the default stream|
    |Return Value|-|No return value|

85. **aclshmemx_handle_wait** — Enqueue a completion wait and member rendezvous for the Handle's team on the supplied stream. Every PE in that team must participate in matching order after issuing the asynchronous operations to be covered; only a world-team Handle requires every PE. The host call only enqueues the operation, so synchronize the stream before observing completion.

    ```python
    def aclshmemx_handle_wait(handle: Handle, stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |handle|[in]|`Handle` bound to a currently active team; that team defines the participating PEs|
    |stream|[in]|Integer `aclrtStream` address; `0` selects the default stream|
    |Return Value|-|No return value; a stale team in the Handle raises `ValueError`. Mismatched participation or call order can wait indefinitely|

86. **aclshmemx_get_prof** — Obtain a deep copy of profiling data for the current instance and PE.

    ```python
    def aclshmemx_get_prof(verbose: bool=False) -> Optional[ProfData]
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |verbose|[in]|Whether to also print profiling data; default is `False`|
    |Return Value|[out]|Returns `ProfData` on the selected PE or `None` when unavailable; data is isolated per instance|

87. **aclshmemx_show_prof** — Print profiling data for the current instance.

    ```python
    def aclshmemx_show_prof() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value|

#### Class
1. **OpEngineType** enumeration class — Data transfer engine types

    ```python
    class OpEngineType(Enum):
        MTE
        SDMA
        ROCE
        UDMA
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |MTE|Memory Transfer Engine|
    |SDMA|System DMA|
    |ROCE|RDMA over Converged Ethernet|
    |UDMA|Unified DMA|

2. **MemType** enumeration class — Symmetric memory allocation location

    ```python
    class MemType(Enum):
        HOST_SIDE
        DEVICE_SIDE
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |HOST_SIDE|Host-side memory|
    |DEVICE_SIDE|Device-side memory (default)|

3. **OptionalAttr** class — Optional attribute configurations for initialization

    ```python
    class OptionalAttr:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |version|[in]|Configuration version|
    |data_op_engine_type|[in]|Data transfer engine type (an enumerated value of `OpEngineType`)|
    |shm_init_timeout|[in]|Timeout period of the init operation|
    |shm_create_timeout|[in]|Timeout period of the create operation|
    |control_operation_timeout|[in]|Timeout period of the control operation|
    |sockFd|[in]|Socket file descriptor. The default value is `-1`.|

4. **InitAttr** class — Attribute configurations for initialization

    ```python
    class InitAttr:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |my_rank|[in]|PE ID of the current process|
    |n_ranks|[in]|Total number of PEs|
    |ip_port|[in]|IP address and port number of the communication server|
    |local_mem_size|[in]|Size of the symmetric memory allocated to the current PE, in bytes|
    |option_attr|[in]|`OptionalAttr` optional attribute configurations|
    |instance_id|[in]|Multi-instance ID; the binding accepts only non-bool integers from 0 to 254. The default is 0.|

5. **TeamConfig** class — Team configurations

    ```python
    class TeamConfig:
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |num_contexts|[in]|The number of contexts that can run simultaneously in a team|

6. **UniqueId** class — Unique identifier handle for UID initialization

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |version|[out]|Version|
    |my_pe|[out]|PE ID of the current process|
    |n_pes|[out]|Total number of PEs of all processes|
    |internal|[out]|Internal information about the UID (bytes)|
    |`to_bytes()`|[out]|Serialize the UID using the exact native structure size|
    |`from_bytes(data)`|[in]|Restore a UID from an exactly sized byte string|

7. **InitStatus** enumeration class — Initialization statuses of the shared memory module

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |NOT_INITIALIZED|Not initialized|
    |SHM_CREATED|Shared memory created|
    |INITIALIZED|Initialization completed|
    |INVALID|Invalid|

8. **SignalOp** enumeration class — Atomic operation types of signal variables

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |SIGNAL_SET|Atomic setting: writes a given value to a remote signal.|
    |SIGNAL_ADD|Atomic addition: adds a given value to the existing value of a remote signal.|

9. **CmpOp** enumeration class — Signal comparison operation types

    ```python
    class CmpOp(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |CMP_EQ|Equal to (==)|
    |CMP_NE|Not equal to (!=)|
    |CMP_GT|Greater than (>)|
    |CMP_GE|Greater than or equal to (>=)|
    |CMP_LT|Less than (<)|
    |CMP_LE|Less than or equal to (<=)|

10. **InstanceContext** class — Read-only value snapshot of the active instance context.

    ```python
    class InstanceContext:
        @property
        def id(self) -> int:
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |id|[out]|Read-only instance ID|

11. **Handle** class — Constructible team-scoped asynchronous-operation handle.

    ```python
    class Handle:
        def __init__(self, team_id: int=ACLSHMEM_TEAM_WORLD) -> None:

        @property
        def team_id(self) -> int:
    ```

    |Attribute/Parameter|Direction|Meaning|
    |-|-|-|
    |team_id|[in/out]|Team ID; defaults to `ACLSHMEM_TEAM_WORLD` and must refer to a currently live team in the range 0 to 2047 on construction and assignment|

12. **ProfData** class — Deep-copy snapshot of profiling data for the current PE.

    ```python
    class ProfData:
        @property
        def pe_id(self) -> int:

        @property
        def ccount(self) -> list[list[int]]:

        @property
        def cycles(self) -> list[list[int]]:
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |pe_id|[out]|Read-only PE ID associated with the profiling data|
    |ccount|[out]|Read-only ``64 x 1024`` two-dimensional integer list of counts|
    |cycles|[out]|Read-only ``64 x 1024`` two-dimensional integer list of cycles|

13. **InitMode** enumeration class — Initialization methods used by `aclshmemx_init_attr`.

    ```python
    class InitMode(Enum):
        DEFAULT
        MPI
        UNIQUEID
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |DEFAULT|Use the default initialization method|
    |MPI|Use MPI initialization|
    |UNIQUEID|Use unique-ID initialization, including multi-instance initialization|

## Python API Constraints and Limitations

- `buffer` / `calloc` / `align` / `free`, `barrier` / `sync`, initialization, and finalization have collective semantics. All participating PEs must call them with matching parameters and in the same order. Use `torchrun` and an external timeout to detect mismatches or deadlocks.
- `barrier` completes host-issued communication. To wait on NPU-issued operations from the host, synchronize the relevant ACL stream or use the stream barrier APIs.
- Stream APIs enqueue work without implicit synchronization. High-level APIs reject bool, negative, and non-integer values; `0` represents the default stream. A nonzero stream integer must identify a live ACL stream for the duration of the call.
- Public `Buffer` construction rejects zero addresses, bool values, negative values, and arguments beyond the current platform's address or length limits. A Buffer with an allocation instance can only be used for peer lookup, RMA/Signal, and free while that instance is active; cross-instance use raises `AclshmemInvalid`.
- `handle_wait` participation is scoped to the Handle's team, and every team member must call it in matching order. The operation is only enqueued; synchronize the corresponding stream before observing completion.
- The runtime chooses the stream-RMA transport path from initialization, target-PE, and platform capabilities; callers must not rely on a particular engine being selected for an individual call. `signal_wait` waits on a local address and does not provide remote-wait semantics.
- Multi-instance mode currently supports at most 255 instances. High-level APIs and corresponding low-level entry points accept only non-bool `instance_id` values from 0 to 254. Each instance heap is limited to 128 GB; nonzero instances retain only the world team and do not support advanced team split/translate/get-config operations.
- The `HOST_SIDE` symmetric heap requires a CANN runtime with `HAS_ACLRT_MEM_FABRIC_HANDLE`; otherwise allocation fails and the high-level API raises `AclshmemError`.

## shmem API (Python Top-Level Wrapper)

The following Tensor APIs are available from the `shmem` package top level and can be called as `shmem.<API name>`. They create and release Tensors in symmetric memory and maintain their metadata; callers must follow the collective-participation and lifetime requirements stated for each entry.

### External APIs
1. **aclshmem_create_tensor** — Allocate a torch.Tensor of a specified shape and data type on the symmetric memory. This is a collective operation. It calls `aclshmem_malloc` internally, so all PEs must participate in the call in the same order, and the shape/dtype (that is, the equivalent memory size) allocated by each PE must be the same. Otherwise, the symmetric heap layout will be corrupted.

    ```python
    def aclshmem_create_tensor(shape, dtype: torch.dtype=torch.float32, device_id=0) -> torch.Tensor
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |shape|[in]|Tensor shape, for example, `(2, 3)`|
    |dtype|[in]|Tensor data type (`torch.dtype`). The default value is `torch.float32`|
    |device_id|[in]|NPU device ID. The default value is `0`|
    |Return Value|[out]|`torch.Tensor` created on the symmetric memory. If the allocation fails, an exception is thrown.|

2. **aclshmem_free_tensor** — Release the symmetric memory corresponding to the tensor allocated by `aclshmem_create_tensor`. This is a collective operation. It calls `aclshmem_free` internally, so all PEs must participate in the call in the same order as the allocation. After the release is complete, do not access the tensor or its address again.

    ```python
    def aclshmem_free_tensor(tensor: torch.Tensor) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |tensor|[in]|Symmetric memory tensor to be released|
    |Return Value|-|No return value.|
