# Sample Introduction

This sample uses RDMA atomic APIs for remote Fetch-and-Add (FAA) and Compare-and-Swap (CAS). Each PE performs one FAA, one successful CAS, and one unsuccessful CAS on the next PE, then checks the returned values and target value.

Each PE uses a single AIV to call the RDMA atomic APIs sequentially. Do not concurrently use QP0 for the same target PE during these calls.

## Environment Requirements

- Ascend950 with an XSCALE or HNS_1825 NIC.
- At least two NPU devices with the RDMA NIC and driver configured.
- See the [RDMA sample](../rdma_demo/README_en.md#environment-requirements) for CANN version requirements, RDMA environment checks, and NIC provider configuration.

## Usage

### Build

Follow the [RDMA sample build instructions](../rdma_demo/README_en.md#instructions) and select the Ascend950 command for your NIC.

### Run

Set `IBV_EXTEND_DRIVERS` as described in the [RDMA sample environment variables](../rdma_demo/README_en.md#ibv_extend_drivers-environment-variable). For manual execution, configure `LD_LIBRARY_PATH` as described in the [RDMA sample run instructions](../rdma_demo/README_en.md#instructions).

#### Single-Machine Execution

Run in the `shmem/` directory:

```bash
bash examples/rdma_atomic_demo/run.sh
```

The script starts two PEs on local devices 0 and 1 by default. Use `-pes` to specify the PE count:

```bash
bash examples/rdma_atomic_demo/run.sh -pes 4
```

#### Multi-Machine Execution

Use one NPU device on each machine. Assuming machine A has IP address `ip1`, set the following in the `shmem/` directory on both machines:

```bash
export SHMEM_UID_SESSION_ID=ip1:8899
```

Start PE0 on machine A:

```bash
./build/bin/rdma_atomic_demo 2 0 tcp://ip1:8899 1 0 0
```

At the same time, start PE1 on machine B:

```bash
./build/bin/rdma_atomic_demo 2 1 tcp://ip1:8899 1 1 0
```

#### Expected Output

A successful two-PE run on one or two machines prints:

```text
pe=0 peer=1 faa_old=11 cas_old=21 cas_miss_old=31 target=30
[SUCCESS] demo run success in relative pe 0
pe=1 peer=0 faa_old=10 cas_old=20 cas_miss_old=30 target=31
[SUCCESS] demo run success in relative pe 1
```

### Command-Line Arguments

```text
./rdma_atomic_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
```

- `n_pes`: Global PE count.
- `pe_id`: PE ID of the current process.
- `ipport`: SHMEM initialization address in the format `tcp://<IP>:<port>`. Use the Host IP of PE0 for multi-machine execution.
- `g_npus`: Number of NPU devices used on the local machine.
- `f_pe`: First PE ID on the local machine.
- `f_npu`: First Runtime-visible NPU device ID used on the local machine.
