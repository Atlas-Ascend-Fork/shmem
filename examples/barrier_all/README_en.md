# barrier_all Example

This example verifies the communication completion semantics of `aclshmem_barrier_all()` in two parts:

- **Device**: inside the kernel, it issues a non-blocking transfer to every other PE using the high-level `aclshmem_putmem_nbi()` API. After the Device `aclshmem_barrier_all()` is called, the Host reads the data back and verifies it element by element.
- **Host**: it likewise transfers data to every other PE using the high-level `aclshmem_putmem_nbi()` API. After the Host `aclshmem_barrier_all()` is called, it reads the data back and verifies it element by element.

The launch script sets the initialization engines based on the topology, and the high-level API selects the transport based on the PE position:

- **Single server**: all PEs are on the local server, so only MTE is used.
- **Dual server**: PEs on the same server use MTE, and PEs on different servers use RDMA (that is, MTE + RDMA).

Without engine parameters, the executable initializes MTE only. For cross-server runs, the launch script explicitly passes `--roce` to initialize MTE + ROCE. Therefore, single-server runs can use the default build, and cross-server runs require RDMA support.

Multi-card communication on a single server requires MTE mutual access support. Cross-server communication additionally requires RDMA. For environment configuration, see [rdma_demo](../rdma_demo/README_en.md).

## Build

After loading the CANN environment, run the following commands in the project root directory:

```bash
# Single server (default build, MTE only)
bash scripts/build.sh -examples

# A2/A3 dual server (MTE + RDMA)
bash scripts/build.sh -examples -enable_rdma

# Ascend950 single server (MTE only)
bash scripts/build.sh -examples -soc_type Ascend950

# Ascend950 dual server: select XSCALE based on the NIC
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE
# Ascend950 dual server: select HNS_1825 based on the NIC
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend HNS_1825
```

## Run

```text
bash examples/barrier_all/run.sh <n_pes> <n_pes_per_node> <node_id> <tcp://rank0_ip:port> [first_device_id:0]
```

- `n_pes`: total number of PEs.
- `n_pes_per_node`: number of cards on each server.
- `node_id`: ID of the current server (starting from 0).
- `first_device_id`: device ID of the first card on the current server (default 0).

### Single server with 8 cards (MTE only)

```bash
bash examples/barrier_all/run.sh 8 8 0 "tcp://<rank0-IP>:8899"
```

### Dual server with 8 cards per server (MTE + RDMA)

Run the following commands on the two servers respectively, using the same node 0 IP address and port:

```bash
# Node 0
bash examples/barrier_all/run.sh 16 8 0 "tcp://<node0-IP>:8899"

# Node 1
bash examples/barrier_all/run.sh 16 8 1 "tcp://<node0-IP>:8899"
```

The script waits for all local processes to complete and returns a non-zero value if any process fails. The default timeout is 120 seconds, and the number of seconds can be set through `BARRIER_ALL_TIMEOUT`. The example succeeds only when the Device and Host verifications of all PEs pass.

The executable of a single PE uses the following parameters:

```text
./build/bin/barrier_all <pe_id> <device_id> <n_pes> <tcp://rank0_ip:port> [--roce]
```

`--roce` is used only for cross-server runs and requires the build with `-enable_rdma`. If not specified, only MTE is enabled.

Quick verification of two PEs on a single server with the default build (requires two local NPUs and the CANN environment):

```bash
bash examples/barrier_all/run.sh 2 2 0 "tcp://<local-IP>:8899"
```

Devices 0 and 1 are used by default. To specify the starting device ID, append `first_device_id` to the end of the command. The script initializes MTE only. The verification passes when the Device and Host verifications of both PEs pass and the script returns 0.
