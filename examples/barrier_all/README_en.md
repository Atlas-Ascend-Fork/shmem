# barrier_all Example

The example verifies the completion semantics of `aclshmem_barrier_all()` in two parts:

- **Device**: inside the kernel, it issues a non-blocking transfer to every other PE using the high-level `aclshmem_putmem_nbi()` interface. After calling the Device `aclshmem_barrier_all()`, the host reads the data back and verifies it element by element.
- **Host**: it likewise transfers to every other PE using the high-level `aclshmem_putmem_nbi()` interface. After calling the Host `aclshmem_barrier_all()`, it reads the data back and verifies it element by element.

The launch script selects the engines to initialize based on the topology, and the high-level interface selects the transport for each PE:

- **Single server**: all PEs are on the local machine, so only MTE is used.
- **Dual server**: PEs on the same server use MTE, and PEs on different servers use RDMA (i.e. MTE + RDMA).

With no engine flag, the executable initializes MTE only. For cross-server runs, the launch script explicitly passes `--roce` to initialize MTE and ROCE. Single-server runs work with the default build; cross-server runs require RDMA support.

Multi-card communication on a single server requires MTE support; cross-server communication additionally requires RDMA. Refer to [rdma_demo](../rdma_demo/README.md) for environment configuration.

## Environment Requirements

- For dual-server runs, ensure that the RDMA environment is available (the RDMA NIC and driver have been correctly installed and configured). Single-server runs require only MTE.

### Ascend950 CANN Version Requirement

RDMA examples on Ascend950 are verified with CANN 9.1.0. CANN 9.1.0 and later are supported; refer to the version table in [quickstart](../../docs/quickstart.md) for the full requirement. Download the corresponding package from the [CANN resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0).

## Instructions

### Build

After loading the CANN environment, run the following commands in the `shmem/` directory:

```bash
# Single server (default build, MTE only)
bash scripts/build.sh -examples

# A2/A3 dual server (MTE + RDMA)
bash scripts/build.sh -examples -enable_rdma

# Ascend950 single server (MTE only)
bash scripts/build.sh -examples -soc_type Ascend950

# Ascend950 dual server (XSCALE NIC)
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE

# Ascend950 dual server (HNS_1825 NIC)
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend HNS_1825
```

### Run

```text
bash examples/barrier_all/run.sh <n_pes> <n_pes_per_node> <node_id> <tcp://rank0_ip:port> [first_device_id:0]
```

- `n_pes`: total number of PEs.
- `n_pes_per_node`: number of cards on each server.
- `node_id`: index of the current server (starting from 0).
- `first_device_id`: device ID of the first card on the current server (default 0).

#### Single server with 8 cards (MTE only)

```bash
bash examples/barrier_all/run.sh 8 8 0 "tcp://<rank0-IP>:8899"
```

#### Dual server with 8 cards per server (MTE + RDMA)

Run the following commands on the two servers respectively, using the same node 0 IP and port:

```bash
# Node 0
bash examples/barrier_all/run.sh 16 8 0 "tcp://<node0-IP>:8899"

# Node 1
bash examples/barrier_all/run.sh 16 8 1 "tcp://<node0-IP>:8899"
```

The script waits for all local processes to finish and returns a non-zero exit code if any process fails. The default timeout is 120 seconds and can be configured via `BARRIER_ALL_TIMEOUT` (in seconds). The example succeeds only when both the Device and Host verifications of all PEs pass.

#### Command Line Parameters

The executable for a single PE takes the following parameters:

```text
./build/bin/barrier_all <pe_id> <device_id> <n_pes> <tcp://rank0_ip:port> [--roce]
```

- `pe_id`: PE ID of the current process.
- `device_id`: device ID used by the current process.
- `n_pes`: total number of PEs.
- `ipport`: IP address and port number required for SHMEM initialization, in the format `tcp://<IP>:<port>`. For cross-server tests, set the IP address to the IP of the host where PE0 is located.

Use `--roce` only for cross-server runs with an `-enable_rdma` build. Without it, only MTE is initialized.

Quick two-PE single-server check with the default build (requires two local NPUs and CANN):

```bash
bash examples/barrier_all/run.sh 2 2 0 "tcp://<local-IP>:8899"
```

By default, the script uses devices 0 and 1. To select another starting device ID, append `first_device_id` to the command. The script initializes MTE only; verification passes when the Device and Host checks pass on both PEs and the script exits with status 0.
