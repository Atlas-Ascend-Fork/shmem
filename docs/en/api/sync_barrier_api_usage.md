# SHMEM Sync/Barrier API Usage

## 1. Overview

This document describes the Host-side sync/barrier APIs and the corresponding ordering inside a Device kernel. These APIs perform collective synchronization across multiple PEs and fall into two categories: blocking and stream-based. The APIs returning an `int` status code are `aclshmem_barrier`, `aclshmemx_barrier_on_stream`, `aclshmem_team_sync`, and `aclshmemx_sync_on_stream`; the remaining APIs return `void`.

**API summary:**

| API | Return type | Blocking/Stream | Description |
| ---- | ---- | ---- | ---- |
| aclshmem\_barrier | int | Blocking | Barriers over a team and returns status |
| aclshmem\_barrier\_all | void | Blocking | Barriers over ACLSHMEM\_TEAM\_WORLD |
| aclshmemx\_barrier\_on\_stream | int | Stream | Enqueues a team barrier on a stream and returns status |
| aclshmemx\_barrier\_all\_on\_stream | void | Stream | Enqueues an ACLSHMEM\_TEAM\_WORLD barrier on a stream |
| aclshmem\_sync | void | Blocking | Synchronizes a team |
| aclshmem\_team\_sync | int | Blocking | Synchronizes a team and returns status |
| aclshmem\_sync\_all | void | Blocking | Synchronizes ACLSHMEM\_TEAM\_WORLD |
| aclshmemx\_sync\_on\_stream | int | Stream | Enqueues a team sync on a stream and returns status |
| aclshmemx\_sync\_all\_on\_stream | void | Stream | Enqueues an ACLSHMEM\_TEAM\_WORLD sync on a stream |

**Blocking vs. stream:**

- **Blocking APIs** (`aclshmem_barrier` / `aclshmem_barrier_all` / `aclshmem_sync` / `aclshmem_team_sync` / `aclshmem_sync_all`): wait for the stream operation to finish internally (call `aclrtSynchronizeStream`); the collective sync is complete when they return.
- **Stream APIs** (`aclshmemx_barrier_on_stream` / `aclshmemx_barrier_all_on_stream` / `aclshmemx_sync_on_stream` / `aclshmemx_sync_all_on_stream`): only enqueue the sync on the specified stream; returning does not mean the sync is complete. The caller must call `aclrtSynchronizeStream` or rely on subsequent stream ordering. NULL selects the library-managed default stream, as it does for Host on-stream RMA and quiet.

**Barrier, sync, and communication completion:**

- sync waits for the PEs in a team to reach the same collective point. It synchronizes ordinary symmetric-memory stores from participating cores within the applicable visibility scope, but does not guarantee completion of outstanding nonblocking communication. Use Host aclshmem_team_sync and stream sync according to this minimum guarantee; do not use sync as a communication-completion operation.
- quiet completes communication previously issued by the calling PE in the corresponding Host or Device execution scope, but does not wait for other PEs to reach a collective point.
- barrier completes preceding communication in the corresponding execution scope and then waits for every PE in the team to arrive. All participants must call collectives in the same order.
- Inside a Device kernel, after NBI, use barrier before reading remote results or reusing a source buffer; alternatively, use quiet to complete the calling PE's communication before sync. fence establishes ordering and is not a general communication-completion guarantee.

***

## 2. API Reference

### 1. aclshmem\_barrier

**Description:**

Blocking team barrier. All participants (every PE in the team and the local participating execution group) must call collectives in the same order. On return, every participant has arrived, and previously issued stores and remote memory updates (including AMOs and RMA operations) within the applicable visibility scope are complete, so the destination may be read and the source buffer reused.

**Prototype:**

```cpp
ACLSHMEM_HOST_API int aclshmem_barrier(aclshmem_team_t team);
```

**Parameters:**

| Parameter | Type | Description |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | Team to synchronize |

**Return value:**

- `0`: success.
- Nonzero: failure.

**Example:**

```cpp
// aclshmem_barrier usage flow (pseudocode)

// 1. Initialize
aclshmemx_init_attr_t attributes;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. Data communication (put/get/amo, etc.)
// ...

// 3. Collective barrier
aclshmem_barrier(ACLSHMEM_TEAM_WORLD);

// 4. Read data written by other PEs before the barrier
// ...
```

***

### 2. aclshmem\_barrier\_all

**Description:**

Performs a blocking barrier over `ACLSHMEM_TEAM_WORLD`. Semantically equivalent to `aclshmem_barrier(ACLSHMEM_TEAM_WORLD)`; selecting WORLD does not change the local participation scope.

**Prototype:**

```cpp
ACLSHMEM_HOST_API void aclshmem_barrier_all(void);
```

**Example:**

```cpp
// aclshmem_barrier_all usage flow (pseudocode)
aclshmem_barrier_all();
// When the barrier returns, communication issued before it by all PEs is complete.
```

***

### 3. aclshmemx\_barrier\_on\_stream

**Description:**

Enqueues a team barrier on the specified stream. A successful return only means the barrier has been accepted for queueing; the completion guarantee takes effect when the stream reaches and completes the barrier. Call `aclrtSynchronizeStream` or rely on in-stream ordering to observe completion.

**Prototype:**

```cpp
ACLSHMEM_HOST_API int aclshmemx_barrier_on_stream(aclshmem_team_t team, aclrtStream stream);
```

**Parameters:**

| Parameter | Type | Description |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | Team to synchronize |
| stream | aclrtStream | Stream on which to enqueue the barrier (NULL uses the default stream) |

**Return value:**

- `0`: success.
- Nonzero: failure.

**Example:**

```cpp
// aclshmemx_barrier_on_stream usage flow (pseudocode)
aclrtStream stream;
aclrtCreateStream(&stream);

aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream);
aclrtSynchronizeStream(stream);

aclrtDestroyStream(stream);
```

***

### 4. aclshmemx\_barrier\_all\_on\_stream

**Description:**

Enqueues an `ACLSHMEM_TEAM_WORLD` barrier on the specified stream; selecting WORLD does not change the local participation scope. A successful return only means the barrier has been accepted for queueing; the completion guarantee takes effect when the stream reaches and completes the barrier.

**Prototype:**

```cpp
ACLSHMEM_HOST_API void aclshmemx_barrier_all_on_stream(aclrtStream stream);
```

**Parameters:**

| Parameter | Type | Description |
| ---- | ---- | ---- |
| stream | aclrtStream | Stream on which to enqueue the barrier (NULL uses the default stream) |

**Example:**

```cpp
// aclshmemx_barrier_all_on_stream usage flow (pseudocode)
aclshmemx_barrier_all_on_stream(stream);
aclrtSynchronizeStream(stream);
```

***

### 5. aclshmem\_sync

**Description:**

Blocking team synchronization. Provides collective rendezvous and visibility of ordinary stores within the agreed scope, but does not complete outstanding nonblocking communication. Blocks until all PEs in the specified team have called the sync; use a completion API such as `aclshmem_quiet` first when prior NBI communication must be finished.

**Prototype:**

```cpp
ACLSHMEM_HOST_API void aclshmem_sync(aclshmem_team_t team);
```

**Parameters:**

| Parameter | Type | Description |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | Team to synchronize |

**Example:**

```cpp
// aclshmem_sync usage flow (pseudocode)
aclshmem_sync(ACLSHMEM_TEAM_WORLD);
```

***

### 6. aclshmem\_team\_sync

**Description:**

Blocking team synchronization. Provides collective rendezvous and visibility of ordinary stores within the agreed scope, but does not complete outstanding nonblocking communication. Unlike `aclshmem_sync`, this API returns a status code; use a completion API such as `aclshmem_quiet` first when prior NBI communication must be finished.

**Prototype:**

```cpp
ACLSHMEM_HOST_API int aclshmem_team_sync(aclshmem_team_t team);
```

**Parameters:**

| Parameter | Type | Description |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | Team to synchronize |

**Return value:**

- `0`: success.
- Nonzero: failure.

**Example:**

```cpp
// aclshmem_team_sync usage flow (pseudocode)
int status = aclshmem_team_sync(ACLSHMEM_TEAM_WORLD);
if (status != 0) {
    // handle failure
}
```

***

### 7. aclshmem\_sync\_all

**Description:**

Performs a blocking synchronization over `ACLSHMEM_TEAM_WORLD`. Semantically equivalent to `aclshmem_sync(ACLSHMEM_TEAM_WORLD)`; selecting WORLD does not change the local participation scope.

**Prototype:**

```cpp
ACLSHMEM_HOST_API void aclshmem_sync_all(void);
```

**Example:**

```cpp
// aclshmem_sync_all usage flow (pseudocode)
aclshmem_sync_all();
```

***

### 8. aclshmemx\_sync\_on\_stream

**Description:**

Enqueues a team synchronization on the specified stream. A successful return only means the sync has been accepted for queueing; the completion guarantee takes effect when the stream reaches and completes the sync. Call `aclrtSynchronizeStream` or rely on in-stream ordering to observe completion.

**Prototype:**

```cpp
ACLSHMEM_HOST_API int aclshmemx_sync_on_stream(aclshmem_team_t team, aclrtStream stream);
```

**Parameters:**

| Parameter | Type | Description |
| ---- | ---- | ---- |
| team | aclshmem\_team\_t | Team to synchronize |
| stream | aclrtStream | Stream on which to enqueue the sync (NULL uses the default stream) |

**Return value:**

- `0`: success.
- Nonzero: failure.

**Example:**

```cpp
// aclshmemx_sync_on_stream usage flow (pseudocode)

// 1. Create a stream
aclrtStream stream;
aclrtCreateStream(&stream);

// 2. Enqueue the sync
int status = aclshmemx_sync_on_stream(ACLSHMEM_TEAM_WORLD, stream);
if (status != 0) {
    // handle failure
}

// 3. Wait for the stream sync to complete
aclrtSynchronizeStream(stream);

// 4. Release resources
aclrtDestroyStream(stream);
```

***

### 9. aclshmemx\_sync\_all\_on\_stream

**Description:**

Enqueues an `ACLSHMEM_TEAM_WORLD` synchronization on the specified stream; selecting WORLD does not change the local participation scope. A successful return only means the sync has been accepted for queueing; the completion guarantee takes effect when the stream reaches and completes the sync.

**Prototype:**

```cpp
ACLSHMEM_HOST_API void aclshmemx_sync_all_on_stream(aclrtStream stream);
```

**Parameters:**

| Parameter | Type | Description |
| ---- | ---- | ---- |
| stream | aclrtStream | Stream on which to enqueue the sync (NULL uses the default stream) |

**Example:**

```cpp
// aclshmemx_sync_all_on_stream usage flow (pseudocode)
aclshmemx_sync_all_on_stream(stream);
aclrtSynchronizeStream(stream);
```

***

## 3. Ordering inside a Device kernel

Both sequences require every PE in the team and every core in the declared participation scope to enter the collectives in the same order:

    // Option 1: NBI -> barrier
    aclshmem_putmem_nbi(dst, src, bytes, peer);
    aclshmem_barrier_all();
    // Prior communication is complete; the destination may be read and the source reused.

    // Option 2: NBI -> quiet -> sync
    aclshmem_putmem_nbi(dst, src, bytes, peer);
    aclshmem_quiet();     // This PE's prior communication is complete; the local source may be reused.
    aclshmem_sync_all();  // Wait for all PEs before reading data written by other PEs.

Visibility of ordinary symmetric-memory stores and completion of communication APIs have different scopes: sync synchronizes ordinary stores from participating cores in supported HCCS/UB memory domains, but cannot replace quiet for NBI completion. barrier provides both communication completion and PE arrival. In a mixed kernel, the Cube scalar unit is outside these APIs' synchronization guarantee; see the Device public header for the participating cores. Complete Host-issued communication with the corresponding Host APIs and Device-issued communication within the Device execution scope.

***

## 4. Notes

1. **Initialization check**: call after `aclshmemx_init_attr` completes. When not initialized, APIs returning a status code return `ACLSHMEM_NOT_INITED`; `void` APIs log an error and return directly.
2. **Team validity**: a non-member passes the sentinel `ACLSHMEM_TEAM_INVALID`; `aclshmem_barrier` / `aclshmem_sync` / `aclshmem_team_sync` / `aclshmemx_barrier_on_stream` / `aclshmemx_sync_on_stream` are then no-ops that return success.
3. **Blocking API internal sync**: blocking APIs call `aclrtSynchronizeStream` internally, so returning means the collective sync is complete; stream APIs require the caller to synchronize the stream.
4. **Participation scope**: collective sync requires all PEs in the team to call the same API; otherwise it blocks until timeout or hangs.
5. **Completion and buffer reuse**: sync only waits for PE arrival and does not by itself complete NBI. Reuse a source buffer only after the calling PE's communication has completed. Read a destination written by another PE only after the relevant PEs have completed their communication and reached the collective point. On Device, use barrier or quiet followed by sync.
6. **Host stream wait scope**: blocking Host APIs use and wait for the library-managed default stream; stream APIs only enqueue work. Wait for the stream containing the collective or establish an explicit cross-stream dependency. Work issued on another stream needs a dependency before the collective; waiting for a different stream alone does not prove that the collective has completed. Synchronizing the stream that launched a Device kernel does not replace quiet or barrier inside that kernel for asynchronous communication.
