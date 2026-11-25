# ACLSHMEM Atomic Operation API Usage Semantics Summary

> This document is written based on the actual code implementation. It covers the APIs at each layer: the standard APIs `aclshmem_<type>_atomic_<op>`, the MTE backend `aclshmemx_mte_atomic_<op>`, and the RDMA backend `aclshmemx_roce_atomic_<op>`.

---

## 1. API Layer Relationships

The atomic APIs are divided into two layers:

```text
aclshmem_<type>_atomic_<op>()          ← standard API
    │
    ├── topo == MTE  → aclshmemx_mte_atomic_<op>()   ← MTE backend
    └── topo == ROCE → aclshmemx_roce_atomic_<op>()  ← RDMA backend
```

The standard API automatically dispatches calls to the MTE or RDMA backend based on `device_state->topo_list[pe]`. It additionally adds synchronization guarantees on top of the backend calls.

---

## 2. MTE Backend APIs: `aclshmemx_mte_atomic_<op>`

> Header file: `include/device/gm2gm/engine/shmem_device_mte.h`
> Implementation file: `src/device/gm2gm/engine/shmem_device_mte.hpp`

**Core feature: All MTE backend APIs are asynchronous**, regardless of whether they return a value. Each API only initiates the hardware atomic operation and then returns. No quiet/fence is performed.

Synchronization primitives: `aclshmemx_mte_quiet()` or Kernel Runtime-level `SetFlag/WaitFlag`.

### 2.1 API List

| Full API Name | Return Value | Supported Types | Hardware Platform |
|---|---|---|---|
| `aclshmemx_mte_atomic_fetch` | T | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_set` | void | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_swap` | T | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_compare_swap` | T | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_inc` | void | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_add` | void | int8, int16, bf16, half, int32, float (A2/A3/950) + uint32, uint64, int64 (950) | A2/A3/950 |
| `aclshmemx_mte_atomic_fetch_inc` | T | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_fetch_add` | T | int32, uint32, float, int64, uint64 | 950 |

### 2.2 Synchronization Requirements

After you call an MTE backend API, check for **data dependencies between the scalar computation unit (Scalar) and the data move units (MTE2/MTE3)**. If such dependencies exist, insert synchronization according to the actual data flow:

| Timing | Scenario | Synchronization Method |
|---|---|---|
| Before the call | Another unit (such as MTE2) is writing to the same UB area | `SetFlag/WaitFlag` on `MTE2_S` |
| After the call | A subsequent operation depends on the result of the atomic operation in the GM | `SetFlag/WaitFlag` on `MTE3_MTE2` |
| After the call | The same UB area needs to be reused to write new data | `SetFlag/WaitFlag` on `MTE3_S` |

### 2.3 Type Differences of `aclshmemx_mte_atomic_add` (Important)

On Ascend950, `aclshmemx_mte_atomic_add` has two implementation paths depending on the type:

| Type Group | Supported Types | Implementation | Synchronization Notes |
|---|---|---|---|
| UB + MTE3 path | int8, int16, bf16, half, int32, float | Scalar writes the operand to the UB → MTE3 reads the UB → MTE3 performs an atomic add and writes to the GM | Pay attention to UB-related fences (MTE2_S and MTE3_S) and the GM fence (MTE3_MTE2) |
| Scalar AtomicAdd path | uint32, uint64, int64 | The Scalar unit directly performs a scalar AtomicAdd and writes to the GM, without passing through the UB and MTE3 | UB/MTE3-related fences are not required; synchronize according to the general rules only when a GM data dependency exists between Scalar↔MTE2/MTE3 |

---

## 3. RDMA Backend APIs: `aclshmemx_roce_atomic_<op>`

> Header file: `include/device/gm2gm/engine/shmem_device_rdma.h`
> Implementation file: `src/device/gm2gm/engine/shmem_device_rdma.hpp`

**Core feature: APIs that return a value are synchronous (quiet is performed internally), and APIs without a return value are asynchronous (quiet is required from the caller).**

Synchronization primitive: `aclshmemx_roce_quiet(pe, buf, sync_id)`.

The target/source parameters of an RDMA AMO are translated to the specified PE as a local symmetric address. Therefore, these parameters must point to symmetric memory. Do not directly use an arbitrary local GM address as this parameter. You can prepare the target object through symmetric memory allocation APIs such as `aclshmem_malloc`.

### 3.1 API List (Synchronous APIs — Quiet Performed Internally)

| Full API Name | Return Value | Supported Types | Hardware Platform |
|---|---|---|---|
| `aclshmemx_roce_atomic_fetch` | T | 32/64-bit integers | 950 |
| `aclshmemx_roce_atomic_fetch_add` | T | 32/64-bit integers | 950 |
| `aclshmemx_roce_atomic_fetch_inc` | T | 32/64-bit integers | 950 |
| `aclshmemx_roce_atomic_fetch_and` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_fetch_or` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_fetch_xor` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_compare_swap` | T | 32/64-bit integers | 950 |
| `aclshmemx_roce_atomic_swap` | T | 32/64-bit integers | 950 |

> Each of the preceding APIs internally calls `aclshmemx_roce_quiet` before returning. **The caller does not need additional synchronization.**

### 3.2 API List (Asynchronous APIs — Quiet Required from the Caller)

| Full API Name | Return Value | Supported Types | Hardware Platform |
|---|---|---|---|
| `aclshmemx_roce_atomic_set` | void | 32/64-bit integers | 950 |
| `aclshmemx_roce_atomic_add` | void | 32/64-bit integers | 950 |
| `aclshmemx_roce_atomic_inc` | void | 32/64-bit integers | 950 |
| `aclshmemx_roce_atomic_and` | void | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_or` | void | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_xor` | void | int32, uint32, int64, uint64 | 950 |

> Each of the preceding APIs only initiates the operation and then returns. **The caller must call `aclshmemx_roce_quiet(pe, buf, sync_id)` before depending on the result.**

### 3.3 Concurrency Restrictions

> **WARNING**: When RDMA is used as the underlying transport, concurrent RMA/AMO operations to the same PE are not supported. Use the `sync_id` in `device_state.rdma_config` to perform pipeline synchronization.

---

## 4. Standard APIs: `aclshmem_<type>_atomic_<op>`

> Header file: `include/device/gm2gm/shmem_device_amo.h`
> Implementation file: `src/device/gm2gm/shmem_device_amo.hpp`

The standard API automatically dispatches calls to the MTE or RDMA backend based on the topology. It **additionally adds synchronization guarantees** on top of the backend calls.

The standard AMO APIs pass the user-provided `dst`, `dest`, or `source` to `aclshmem_ptr(..., pe)` to translate them into the target address on the specified PE. Therefore, these parameters must point to symmetric memory. This requirement comes from the address translation method of the standard APIs. It does not mean that the MTE engine itself can only access symmetric memory. Automatic dispatch by the transport layer does not change the API semantics.

```text
aclshmem_<type>_atomic_<op>(dst, val, pe)
    │
    ├── topo_list[pe] & MTE
    │   ├── fetch variant:     calls aclshmemx_mte_atomic_fetch_<op>() + PipeBarrier<PIPE_ALL>
    │   ├── swap/cas:          calls aclshmemx_mte_atomic_swap/compare_swap() + PipeBarrier<PIPE_ALL>
    │   ├── add/inc:           passes through aclshmemx_mte_atomic_add/inc(), without PipeBarrier
    │   └── set/and/or/xor:    passes through the underlying asynchronous APIs, without PipeBarrier
    │
    └── topo_list[pe] & ROCE
        ├── fetch variant:     passes through aclshmemx_roce_atomic_fetch_<op>() (the backend already performs quiet)
        ├── swap/cas:          passes through aclshmemx_roce_atomic_swap/compare_swap() (the backend already performs quiet)
        ├── add/inc:           passes through aclshmemx_roce_atomic_add/inc() (the backend is asynchronous, no quiet added)
        └── set/and/or/xor:    passes through the underlying asynchronous APIs (no quiet added)
```

### 4.1 Synchronous APIs (With a Return Value)

Each function performs synchronization internally before returning. **The caller does not need additional synchronization.**

| Full API Name (int32 as an Example) | Return Value | MTE Path Synchronization Method | RDMA Path Synchronization Method |
|---|---|---|---|
| `aclshmem_int32_atomic_fetch_add` | int32_t | `aclshmemx_mte_atomic_fetch_add` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_fetch_add` (quiet performed internally) |
| `aclshmem_int32_atomic_fetch_inc` | int32_t | `aclshmemx_mte_atomic_fetch_inc` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_fetch_inc` (quiet performed internally) |
| `aclshmem_int32_atomic_fetch_and` | int32_t | ✗ MTE not supported | `aclshmemx_roce_atomic_fetch_and` (quiet performed internally) |
| `aclshmem_int32_atomic_fetch_or` | int32_t | ✗ MTE not supported | `aclshmemx_roce_atomic_fetch_or` (quiet performed internally) |
| `aclshmem_int32_atomic_fetch_xor` | int32_t | ✗ MTE not supported | `aclshmemx_roce_atomic_fetch_xor` (quiet performed internally) |
| `aclshmem_int32_atomic_fetch` | int32_t | `aclshmemx_mte_atomic_fetch` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_fetch` (quiet performed internally) |
| `aclshmem_uint32_atomic_swap` | uint32_t | `aclshmemx_mte_atomic_swap` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_swap` (quiet performed internally) |
| `aclshmem_uint32_atomic_compare_swap` | uint32_t | `aclshmemx_mte_atomic_compare_swap` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_compare_swap` (quiet performed internally) |

### 4.2 Asynchronous APIs (Without a Return Value, Explicit Synchronization Required)

| Full API Name (int32 as an Example) | Return Value | MTE Path | RDMA Path | Synchronization Required from the Caller |
|---|---|---|---|---|
| `aclshmem_int32_atomic_add` | void | `aclshmemx_mte_atomic_add` (asynchronous) | `aclshmemx_roce_atomic_add` (asynchronous) | MTE: SetFlag/WaitFlag (as needed); RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_inc` | void | `aclshmemx_mte_atomic_inc` (asynchronous) | `aclshmemx_roce_atomic_inc` (asynchronous) | MTE: SetFlag/WaitFlag (as needed); RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_and` | void | ✗ MTE not supported | `aclshmemx_roce_atomic_and` (asynchronous) | RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_or` | void | ✗ MTE not supported | `aclshmemx_roce_atomic_or` (asynchronous) | RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_xor` | void | ✗ MTE not supported | `aclshmemx_roce_atomic_xor` (asynchronous) | RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_uint32_atomic_set` | void | `aclshmemx_mte_atomic_set` (asynchronous) | `aclshmemx_roce_atomic_set` (asynchronous) | MTE: SetFlag/WaitFlag (as needed); RDMA: `aclshmemx_roce_quiet` |

> For the MTE path synchronization details of `atomic_add` (two paths divided by data type), refer to [Section 2.3](#23-type-differences-of-aclshmemx_mte_atomic_add-important).

---

## 5. Synchronization Rules Quick Reference (For the MTE and RDMA Paths)

```text
Using the aclshmem_<type>_atomic_<op> APIs
│
├── fetch_add / fetch_inc / fetch_and / fetch_or / fetch_xor / fetch / swap / compare_swap
│   └── Synchronous APIs (with a return value); the operation completes when the call returns, and the return value is reliable
│       MTE: fetch_add/fetch_inc/fetch/swap/compare_swap have PipeBarrier
│       RDMA: quiet is performed internally for all APIs
│       (The PipeBarrier on the MTE path already guarantees transfer completion; if another unit writes to the UB before the call, SetFlag/WaitFlag is still required)
│
├── add / inc / and / or / xor / set
│   └── Asynchronous APIs (without a return value); explicit synchronization is required:
│       RDMA: aclshmemx_roce_quiet(pe, buf, sync_id)
│       MTE: insert SetFlag/WaitFlag based on the actual data dependencies (and/or/xor does not support MTE)
```

---

## 6. Supported Types and Hardware Platforms of Each Standard API

### 6.1 `aclshmem_<type>_atomic_add`

| Transport Path | Supported Types | MTE Implementation | Hardware Platform |
|---|---|---|---|
| MTE | int8, int16, bf16, half, int32, float | UB + MTE3 | A2/A3/950 |
| MTE | uint32, uint64, int64 | Scalar AtomicAdd | 950 |
| RDMA | int32, uint32, int64, uint64 | — | 950 |

### 6.2 `aclshmem_<type>_atomic_fetch_add`

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| MTE | int32, uint32, uint64, int64, float | 950 |
| RDMA | int32, uint32, int64, uint64 | 950 |

### 6.3 `aclshmem_<type>_atomic_inc`

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| MTE | int32, uint32, uint64, int64, float | 950 |
| RDMA | int32, uint32, int64, uint64 | 950 |

### 6.4 `aclshmem_<type>_atomic_fetch_inc`

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| MTE | int32, uint32, uint64, int64, float | 950 |
| RDMA | int32, uint32, int64, uint64 | 950 |

### 6.5 Bitwise Operation Series (`atomic_and/or/xor` and Fetch Variants)

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| RDMA | int32, uint32, int64, uint64 | 950 |

> **NOTE**: The bitwise operation APIs do not support the MTE path.

### 6.6 `aclshmem_<type>_atomic_fetch`

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

### 6.7 `aclshmem_<type>_atomic_set`

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

> CAST variants (`ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST`): int32→uint32, int64→uint64, and float→uint32 are implemented through type conversion.

### 6.8 `aclshmem_<type>_atomic_swap`

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

> The CAST variants are the same as above.

### 6.9 `aclshmem_<type>_atomic_compare_swap`

| Transport Path | Supported Types | Hardware Platform |
|---|---|---|
| MTE | uint32, uint64, int32, int64 | 950 |
| RDMA | uint32, uint64, int32, int64 | 950 |

> CAST variants (`ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_CAST`): int32→uint32 and int64→uint64.

---

## 7. Cross-PCIe (Cross-Node) Restrictions

**The MTE path does not support cross-PCIe communication.** In cross-node scenarios, use the RDMA path.

---

## 8. Code Examples

In the following examples, `dst` must come from symmetric allocation. On the host side, all PEs call the allocation API collectively with the same size. Pass the returned address to the kernel that executes the Device AMO:

```c
void *symmetric_target = aclshmem_malloc(sizeof(uint64_t));
// Pass symmetric_target to the kernel and use the typed pointer as dst.
```

You cannot use an ordinary local GM address allocated by `aclrtMalloc` to replace the target of a standard AMO. This restriction comes from remote address translation, not from an inherent limitation of the MTE or RDMA engine on GM addresses.

### 8.1 Asynchronous API + RDMA (`aclshmem_int32_atomic_add`)

```c
// Asynchronous atomic add, RDMA path
aclshmem_int32_atomic_add(dst, 42, pe);

// ... Other computations that do not depend on dst ...

// Quiet is required before reading the result
aclshmemx_roce_quiet(pe, ub_buf, sync_id);

// Now you can safely use the value of dst
int32_t result = *dst;
```

### 8.2 Synchronous API (`aclshmem_int32_atomic_fetch_add`)

```c
// Synchronous API; the return value can be used directly once the call returns
int32_t old_val = aclshmem_int32_atomic_fetch_add(dst, 42, pe);
// old_val is the reliable old value; no additional quiet is required
```

### 8.3 MTE Path + UB Reuse (`aclshmem_int32_atomic_add`, UB+MTE3 Path)

```c
aclshmemx_set_mte_config(0, 256, sync_id);

// Fence until MTE2 finishes writing to the UB (if MTE2 previously wrote to the same UB area)
SetFlag<HardEvent::MTE2_S>(sync_id);
WaitFlag<HardEvent::MTE2_S>(sync_id);

// Asynchronous atomic add (Scalar writes to the UB → MTE3 reads the UB → MTE3 writes to the GM)
aclshmem_int32_atomic_add(dst, 1, pe);

// Fence until MTE3 finishes writing to the GM — the result in the GM is needed later
SetFlag<HardEvent::MTE3_MTE2>(sync_id);
WaitFlag<HardEvent::MTE3_MTE2>(sync_id);

// Fence until MTE3 finishes reading the UB — if you need to reuse the UB
SetFlag<HardEvent::MTE3_S>(sync_id);
WaitFlag<HardEvent::MTE3_S>(sync_id);

// Now you can safely read the GM result and reuse the UB
```

### 8.4 Synchronous API + MTE Path (`aclshmem_int32_atomic_swap`)

```c
// The MTE path of the atomic_swap standard API already adds PipeBarrier; the operation completes upon return
int32_t old = aclshmem_int32_atomic_swap(dst, new_val, pe);
// old is the reliable old value; no additional synchronization is required

// If another unit is writing to the same UB area before the call, SetFlag/WaitFlag is still required:
SetFlag<HardEvent::MTE2_S>(sync_id);
WaitFlag<HardEvent::MTE2_S>(sync_id);
```

### 8.5 Asynchronous API + RDMA (`aclshmem_int32_atomic_and`, RDMA Path)

```c
// atomic_and has no return value; the standard API passes through roce_atomic_and (asynchronous)
aclshmem_int32_atomic_and(dst, mask, pe);

// Quiet is required before reading the result
aclshmemx_roce_quiet(pe, ub_buf, sync_id);

// Now you can safely read the result
int32_t val = *dst;
```

### 8.6 `aclshmem_uint32_atomic_set` (Explicit Synchronization Required)

```c
// The underlying implementation of atomic_set is asynchronous (regardless of MTE or RDMA)
aclshmem_uint32_atomic_set(dst, 99, pe);

// The RDMA path requires quiet
aclshmemx_roce_quiet(pe, ub_buf, sync_id);

// Or the MTE path requires inserting SetFlag/WaitFlag based on data dependencies
```

---

## 9. Function Name Mapping

Macro definition mappings for the standard SHMEM style:

```c
#define shmem_int8_atomic_add      aclshmem_int8_atomic_add
#define shmem_int16_atomic_add     aclshmem_int16_atomic_add
#define shmem_int32_atomic_add     aclshmem_int32_atomic_add
#define shmem_half_atomic_add      aclshmem_half_atomic_add
#define shmem_bfloat16_atomic_add  aclshmem_bfloat16_atomic_add
#define shmem_float_atomic_add     aclshmem_float_atomic_add
```

Other types follow the same rule: `shmem_<type>_atomic_<op>` → `aclshmem_<type>_atomic_<op>`.

---

## 10. Synchronization Summary Table for Each API Layer

### MTE Backend (`aclshmemx_mte_atomic_<op>`)

| API | Return Value | Synchronization | Synchronization Method |
|---|---|---|---|
| `mte_atomic_fetch` | T | Asynchronous | SetFlag/WaitFlag or `aclshmemx_mte_quiet` is required |
| `mte_atomic_set` | void | Asynchronous | Same as above |
| `mte_atomic_swap` | T | Asynchronous | Same as above |
| `mte_atomic_compare_swap` | T | Asynchronous | Same as above |
| `mte_atomic_inc` | void | Asynchronous | Same as above |
| `mte_atomic_add` | void | Asynchronous | Same as above (two paths depending on the type, see 2.3) |
| `mte_atomic_fetch_inc` | T | Asynchronous | Same as above |
| `mte_atomic_fetch_add` | T | Asynchronous | Same as above |

### RDMA Backend (`aclshmemx_roce_atomic_<op>`)

| API | Return Value | Synchronization | Synchronization Method |
|---|---|---|---|
| `roce_atomic_fetch` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_fetch_add` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_fetch_inc` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_fetch_and` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_fetch_or` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_fetch_xor` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_compare_swap` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_swap` | T | Synchronous | Internal `roce_quiet` |
| `roce_atomic_set` | void | Asynchronous | `aclshmemx_roce_quiet` is required |
| `roce_atomic_add` | void | Asynchronous | `aclshmemx_roce_quiet` is required |
| `roce_atomic_inc` | void | Asynchronous | `aclshmemx_roce_quiet` is required |
| `roce_atomic_and` | void | Asynchronous | `aclshmemx_roce_quiet` is required |
| `roce_atomic_or` | void | Asynchronous | `aclshmemx_roce_quiet` is required |
| `roce_atomic_xor` | void | Asynchronous | `aclshmemx_roce_quiet` is required |

### Standard APIs (`aclshmem_<type>_atomic_<op>`)

| API | Return Value | MTE Path | RDMA Path | Caller Synchronization Required? |
|---|---|---|---|---|
| `atomic_fetch_add` | T | Synchronous (+PipeBarrier) | Synchronous (pass-through) | No |
| `atomic_fetch_inc` | T | Synchronous (+PipeBarrier) | Synchronous (pass-through) | No |
| `atomic_fetch` | T | Synchronous (+PipeBarrier) | Synchronous (pass-through) | No |
| `atomic_fetch_and` | T | ✗ Not supported | Synchronous (pass-through) | No |
| `atomic_fetch_or` | T | ✗ Not supported | Synchronous (pass-through) | No |
| `atomic_fetch_xor` | T | ✗ Not supported | Synchronous (pass-through) | No |
| `atomic_swap` | T | Synchronous (+PipeBarrier) | Synchronous (pass-through) | No |
| `atomic_compare_swap` | T | Synchronous (+PipeBarrier) | Synchronous (pass-through) | No |
| `atomic_and` | void | ✗ Not supported | Asynchronous (pass-through) | **Yes** |
| `atomic_or` | void | ✗ Not supported | Asynchronous (pass-through) | **Yes** |
| `atomic_xor` | void | ✗ Not supported | Asynchronous (pass-through) | **Yes** |
| `atomic_add` | void | Asynchronous | Asynchronous | **Yes** |
| `atomic_inc` | void | Asynchronous | Asynchronous | **Yes** |
| `atomic_set` | void | Asynchronous | Asynchronous | **Yes** |

---

## 11. Source Code Reference

| Layer | Header File | Implementation File |
|---|---|---|
| Standard API declaration | `include/device/gm2gm/shmem_device_amo.h` | — |
| Standard API implementation | — | `src/device/gm2gm/shmem_device_amo.hpp` |
| MTE backend APIs | `include/device/gm2gm/engine/shmem_device_mte.h` | `src/device/gm2gm/engine/shmem_device_mte.hpp` |
| RDMA backend APIs | `include/device/gm2gm/engine/shmem_device_rdma.h` | `src/device/gm2gm/engine/shmem_device_rdma.hpp` |
