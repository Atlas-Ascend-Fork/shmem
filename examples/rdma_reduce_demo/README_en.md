# Sample Introduction

This sample uses the RoCE Put Reduce APIs to perform ADD, MAX, and MIN reductions on data partitions. Each PE contributes the corresponding partition to every other PE and checks its local reduction results. It also uses the `aclshmemx_half_sat_t` tag for FP16 saturating addition, clamping positive overflow to the maximum finite value of 65504.

## Environment Requirements

- Ascend950 with an HNS_1825 NIC. XSCALE NICs are not currently supported.
- At least two NPU devices with the RDMA NIC and driver configured.
- See the [RDMA sample](../rdma_demo/README_en.md#environment-requirements) for CANN version requirements, RDMA environment checks, and NIC provider configuration.

## Usage

### Build

Follow the [RDMA sample build instructions](../rdma_demo/README_en.md#build) and select the **Ascend950 + HNS_1825** command.

### Run

Set `IBV_EXTEND_DRIVERS` for the HNS_1825 NIC as described in the [RDMA sample environment variables](../rdma_demo/README_en.md#ibv_extend_drivers-environment-variable). For manual execution, configure `LD_LIBRARY_PATH` as described in the [RDMA sample run instructions](../rdma_demo/README_en.md#run).

#### Single-Machine Execution

Run in the `shmem/` directory:

```bash
bash examples/rdma_reduce_demo/run.sh
```

The script starts two PEs on local devices 0 and 1 by default. Use `-pes` to specify the PE count:

```bash
bash examples/rdma_reduce_demo/run.sh -pes 8
```

#### Multi-Machine Execution

Use one NPU device on each of two machines. Assuming machine A has IP address `ip1`, set the following in the `shmem/` directory on both machines:

```bash
export SHMEM_UID_SESSION_ID=ip1:8899
```

Start PE0 on machine A:

```bash
./build/bin/rdma_reduce_demo 2 0 tcp://ip1:8899 1 0 0
```

At the same time, start PE1 on machine B:

```bash
./build/bin/rdma_reduce_demo 2 1 tcp://ip1:8899 1 1 0
```

See the [RDMA sample command-line parameters](../rdma_demo/README_en.md#command-line-parameters) for argument descriptions.

#### Expected Output

Each PE prints validation and execution success messages. A successful two-PE run prints:

```text
check transport result success, relative pe=0
[SUCCESS] demo run success in relative pe 0
check transport result success, relative pe=1
[SUCCESS] demo run success in relative pe 1
```
