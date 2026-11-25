# SHMEM Python Extension Tests

This directory covers the original Python interfaces, and the newly added UID attributes,
multi-instance, ``mem_type`` symmetric heap, engine configuration, collective synchronization,
profiling, and ``handle_wait`` interfaces.

## 1. Prerequisites

- A SHMEM wheel matching the current CANN, Python, and CPU architecture is installed.
- Each local process exclusively occupies one NPU, and the CANN and SHMEM environment variables
  are loaded before running.
- Multi-PE test cases are launched through ``torchrun``, and all PEs must execute collective
  operations in the same order.

## 2. Running All Test Cases

Two NPUs are used by default:

```bash
bash examples/python_extension/run.sh
```

Single-NPU smoke test (for binding, single-PE collective paths, and multi-instance isolation
checks):

```bash
NPROC_PER_NODE=1 bash examples/python_extension/run.sh
```

If the CANN runtime supports the host symmetric heap, you can also enable the ``HOST_SIDE``
allocation test:

```bash
SHMEM_TEST_HOST_HEAP=1 NPROC_PER_NODE=2 bash examples/python_extension/run.sh
```

By default, the script runs the MTE ``handle_wait`` smoke test on a separate single PE, which
avoids using this interface with multiple PEs when ROCE is not enabled. You can override this
with ``HANDLE_WAIT_NPROC_PER_NODE``. However, multi-PE ``handle_wait`` should preferably be
verified in a real ROCE environment as described in the next section.

## 3. ``putmem_on_stream`` Python/C++ Performance Comparison

The ``perf`` directory uses the same SHMEM instance, the same set of symmetric buffers, the
same data length, the same PE ring routing, and the same explicit ACL stream to compare:

- C++ reference path: the auxiliary dynamic library directly calls
  ``aclshmemx_putmem_on_stream`` in the C++ loop;
- Python path: the Python loop calls the high-level ``shmem.core.put``, which enters the same
  ``aclshmemx_putmem_on_stream`` through pybind11.

Both paths synchronize the same stream at the end of the timing interval. After warm-up, the
test alternates between the C++ and Python paths. The maximum elapsed time among all PEs in
each round is used as the completion time of the group, and the median across multiple rounds
is taken. After the test, the first and last bytes of the receive buffers are sampled for
verification to ensure that the RMA results are correct. By default, the threshold requires the
Python overhead to be no more than 5%.

Run with 2, 4, and 8 PEs:

```bash
NPROC_PER_NODE=2 bash examples/python_extension/perf/run.sh
NPROC_PER_NODE=4 bash examples/python_extension/perf/run.sh
NPROC_PER_NODE=8 bash examples/python_extension/perf/run.sh
```

The default configuration is 8 MiB, 100 iterations, 10 warm-up iterations, and 7 measurement
rounds, and can be overridden through environment variables:

```bash
PERF_BYTES=8388608 \
PERF_ITERATIONS=100 \
PERF_WARMUP=10 \
PERF_ROUNDS=7 \
PERF_THRESHOLD_PERCENT=5 \
NPROC_PER_NODE=4 \
bash examples/python_extension/perf/run.sh
```

JSON results are written to ``examples/python_extension/perf/results/`` by default. Only
environments that actually have the corresponding number of NPUs can serve as acceptance
evidence for the corresponding 2/4/8-NPU configurations. Results obtained with fewer NPUs must
not be used as substitutes.

## 4. Cross-Machine ROCE handle_wait

``test/core/test_handle_wait.py`` uses the same completion sequence as C++
``examples/rdma_handlewait_test/use_handlewait``:

```text
stream RMA -> handle_wait -> stream synchronize -> data visibility check
```

On two machines with ROCE configured, start the same script with standard multi-machine
``torchrun`` parameters. For example, one NPU per node:

```bash
SHMEM_TEST_ENGINE=ROCE torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=${NODE_RANK} \
  --master-addr=${MASTER_ADDR} \
  --master-port=${MASTER_PORT} \
  examples/python_extension/test/core/test_handle_wait.py
```

``NODE_RANK`` is 0 and 1 on the two machines respectively. ``MASTER_ADDR`` must be the address
of rank 0 that is reachable by both machines. The CANN/driver, ROCE NICs, and SHMEM routing
must be configured in advance.

## 5. Test Cases and Covered Interfaces

|Test Case|Covered Content|
|-|-|
|``test/core/test_memory.py``|``aclshmemx_malloc/calloc/align/free``, ``MemType``, peer buffer|
|``test/core/test_sync_config_prof.py``|4 configuration interfaces, 4 host collective synchronization interfaces, 2 stream barriers, ``get_prof/show_prof``|
|``test/core/test_multi_instance.py``|UID attribute construction chain, ``instance_ctx_get/set``, heap isolation, finalize of a specified instance|
|``test/core/test_handle_wait.py``|``Handle``, stream RMA, ``handle_wait``, data visibility after completion|
|Original ``test/core`` test cases|Initialization, direct, RMA/Signal, and compatibility regression|

By default, ``run.sh`` sets a 10-minute timeout for each ``torchrun`` test case, and you can
adjust it through ``SHMEM_TEST_TIMEOUT``. For example:

```bash
SHMEM_TEST_TIMEOUT=20m NPROC_PER_NODE=2 bash examples/python_extension/run.sh
```

A timeout usually indicates that PEs call in an inconsistent order, a process exits abnormally
early, or the communication links of the running environment are not ready.
