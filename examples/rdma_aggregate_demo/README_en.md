# RDMA Aggregate Interface Sample

## Sample Introduction

This sample verifies the 8 Device API overloads of the RDMA aggregate NBI, and additionally
verifies a 5-operation Put pointer batch:

| Direction | Parameter Form | Action Form |
| --- | --- | --- |
| Get | `__gm__` pointer | `aclshmemx_defer_t` / `aclshmemx_submit_t` |
| Get | `GlobalTensor` | `aclshmemx_defer_t` / `aclshmemx_submit_t` |
| Put | `__gm__` pointer | `aclshmemx_defer_t` / `aclshmemx_submit_t` |
| Put | `GlobalTensor` | `aclshmemx_defer_t` / `aclshmemx_submit_t` |

In addition, the Put pointer path contains an independent 5-operation batch. The first 4
operations use `defer`, and the 5th operation uses `submit`. These 5 operations use independent
source/destination slots, and the host machine verifies the transfer result of each operation.
After all batches are submitted, `aclshmemx_roce_barrier_all` is called, and then the host
machine verifies the results.

## Supported Scope and Constraints

- Three RDMA backends are supported: `IN_DIE` (default backend), `XSCALE`, and `HNS_1825`.
- There are 2 PEs by default. Each PE uses 1 AI Core, and communication uses QP0.
- Each aggregate batch must reuse the same submit state, peer, operation direction, UB buffer,
  and `sync_id`.
- The Put pointer batch of 4 defer + 1 submit contains 5 operations and requires a 704B UB
  workspace, which satisfies the capacity requirement of `64 + 128 * n`.
- The source data of Put and the target data of Get cannot be reused immediately after submit.
  The sample uses a barrier to ensure visibility.

## Environment Requirements

A configured RDMA environment, CANN environment, and the corresponding NIC provider are
required. Before running on the Ascend950 platform, set `IBV_EXTEND_DRIVERS` based on the NIC
type:

- XSCALE uses the provider corresponding to the XSCALE NIC.
- HNS_1825 uses the provider corresponding to the 1825 NIC.

For details about RDMA NIC checks and provider configuration, refer to
[`rdma_demo`](../rdma_demo/README_en.md).

## Compilation

Run the following commands in the project root directory:

```bash
# A2/A3 IN_DIE
bash scripts/build.sh -enable_rdma -examples

# XSCALE
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE -examples

# HNS_1825
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend HNS_1825 -examples
```

`rdma_aggregate_demo` is added to the examples build target under the `IN_DIE`, `XSCALE`, and
`HNS_1825` backend configurations.

## Running

Run the following command in the project root directory:

```bash
bash examples/rdma_aggregate_demo/run.sh
```

You can also specify the ipport used by bootstrap:

```bash
bash examples/rdma_aggregate_demo/run.sh -ipport tcp://127.0.0.1:8899
```

On success, both PEs output:

```text
[SUCCESS] rdma aggregate pointer/tensor defer/submit overloads, including 4-defer pointer put, passed on PE 0
[SUCCESS] rdma aggregate pointer/tensor defer/submit overloads, including 4-defer pointer put, passed on PE 1
```

## Code Structure

- `rdma_aggregate_demo_kernel.cpp`: Device kernel, which calls the 8 basic aggregate API
  overloads and the Put pointer batch of 4 defer + 1 submit.
- `rdma_aggregate_demo_common.h`: Definitions of the test layout, slots, and verification
  values.
- `main.cpp`: Host initialization of SHMEM/RDMA, symmetric buffer preparation, kernel launch,
  and result verification.
- `run.sh`: Starts local PE test processes based on the parameters, and starts 2 PEs by
  default.
- `CMakeLists.txt`: Reuses the `aclshmem_add_collective_example` build template of the
  repository.

## Multi-PE, Multi-NPU, and Cross-Machine Running Parameters

| Parameter | Default Value | Description |
| --- | --- | --- |
| `-ipport <addr>` | `tcp://127.0.0.1:8299` | Bootstrap address and port, in the format `tcp://<IP>:<port>`. For cross-machine running, enter the IP of the machine where PE 0 resides |
| `-pes <count>` | `2` | Global number of PEs, at least 2 and a power of 2 (2, 4, 8, 16, and so on) |
| `-gnpus <count>` | Same as `-pes` | Number of NPUs launched on the current machine, which also determines the number of PEs launched on the machine. If not specified, the value defaults to the same as `-pes` |
| `-fnpu <id>` | `0` | ID of the first NPU used by the current machine to run this sample |
| `-fpe <id>` | `0` | First PE ID used on the current machine |

### 4 PEs on One Machine

With 4 PEs, each PE interacts with the next PE in the ring. Get pulls data from the next PE,
Put writes to the next PE, and the Put received by this PE comes from the previous PE. Devices
are mapped according to `PE ID % NPU count + first NPU ID`:

```bash
bash examples/rdma_aggregate_demo/run.sh -pes 4
```

### 16 PEs on Two Machines

The following example uses two machines with 8 NPUs each, 16 PEs in total. The number of PEs
launched on each machine is determined by `-gnpus 8`:

```bash
# Machine 0 (PE 0~7)
bash examples/rdma_aggregate_demo/run.sh -ipport tcp://10.0.0.1:8299 -pes 16 -gnpus 8 -fpe 0
# Machine 1 (PE 8~15)
bash examples/rdma_aggregate_demo/run.sh -ipport tcp://10.0.0.1:8299 -pes 16 -gnpus 8 -fpe 8
```

Each PE communicates with the next PE in the ring. The two machines must use the same bootstrap
address and port, and ensure that the RDMA NIC, provider, port, and firewall configurations are
correct.
