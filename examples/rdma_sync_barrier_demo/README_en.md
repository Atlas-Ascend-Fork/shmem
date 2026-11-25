# Example Scenario

This example demonstrates how to use the low-level API `aclshmemx_roce_put_nbi` over the RDMA transmission path, together with synchronization/barrier primitives, to perform all-gather data communication between multiple PEs and verify data correctness.

Specifically, this example provides the following 8 synchronization modes:

- **sync_all**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_sync_all` to perform global synchronization.
- **sync_all_buf**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_sync_all(buf, sync_id)`, explicitly passing in the UB buffer and sync_id.
- **barrier_all**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier_all` to perform global synchronization (quiet is automatically performed inside barrier).
- **barrier_all_buf**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier_all(buf, sync_id)`, explicitly passing in the UB buffer and sync_id.
- **sync_team**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_team_sync(team)` to perform synchronization within the team.
- **sync_team_buf**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_team_sync(team, buf, sync_id)`, explicitly passing in parameters within the team.
- **barrier_team**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier(team)` to perform synchronization within the team.
- **barrier_team_buf**: Uses `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier(team, buf, sync_id)`, explicitly passing in parameters.

## Environment Requirements

Same as the environment requirements in [rdma_demo](../rdma_demo/README_en.md).

## Instructions

### Build

Build in the shmem/ directory. For details about the RDMA build parameters (A2/A3, and the `XSCALE` / `HNS_1825` backends of Ascend950), see [Compilation and Build - RDMA Parameters](../../docs/compilation_build_guide_en.md#rdma-parameters).

### Run

> Note: On the Ascend950 platform, the `IBV_EXTEND_DRIVERS` environment variable must be set. See [Environment Variable Description](../rdma_demo/README_en.md#ibv_extend_drivers-environment-variable).

#### Method 1: Run `bash run.sh` in the `examples/rdma_sync_barrier_demo` directory

`run.sh` supports specifying the demo type using a parameter. The default value is `sync_all`.

```bash
bash run.sh sync_all           # Run the sync_all demo (default)
bash run.sh sync_all_buf       # Run the sync_all(buf, sync_id) demo
bash run.sh barrier_all        # Run the barrier_all demo
bash run.sh barrier_all_buf    # Run the barrier_all(buf, sync_id) demo
bash run.sh sync_team          # Run the sync_team demo
bash run.sh sync_team_buf      # Run the sync_team(buf, sync_id) demo
bash run.sh barrier_team       # Run the barrier_team demo
bash run.sh barrier_team_buf   # Run the barrier_team(buf, sync_id) demo
```

#### Method 2: Run commands manually in the `shmem/` directory

- Commands for single-server dual-device execution:

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # Required only on the Ascend950 platform. Set this based on the NIC type. See the environment variable description.
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
./build/bin/rdma_sync_barrier_demo 2 0 tcp://127.0.0.1:8899 2 0 0 sync_all & # PE 0
./build/bin/rdma_sync_barrier_demo 2 1 tcp://127.0.0.1:8899 2 0 0 sync_all & # PE 1
```

> Note: \<shmem-root-directory\> indicates the root directory of the SHMEM project, and \<path_to_plugin.so\> indicates the plugin library path set based on the NIC type.

- Commands for cross-server dual-device execution:

Assume that the IP address of server A is ip1 and that of server B is ip2.
Run the following command on server A:

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # Required only on the Ascend950 platform. Set this based on the NIC type. See the environment variable description.
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
./build/bin/rdma_sync_barrier_demo 2 0 tcp://ip1:8765 1 0 0 sync_all # PE 0
```

At the same time, run the following command on server B:

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # Required only on the Ascend950 platform. Set this based on the NIC type. See the environment variable description.
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
./build/bin/rdma_sync_barrier_demo 2 1 tcp://ip1:8765 1 1 0 sync_all # PE 1
```

> Note: \<shmem-root-directory\> indicates the root directory of the SHMEM project, and \<path_to_plugin.so\> indicates the plugin library path set based on the NIC type.
>
> To run a cross-server test in a container, specify the `--net=host` mode when starting the container.

- Example commands for dual-server 16-device (8 devices per server) execution:

Assume that the IP address of server A is ip1 and that of server B is ip2.
Run the following command on server A:

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # Required only on the Ascend950 platform. Set this based on the NIC type. See the environment variable description.
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
pids=()
for pe in $(seq 0 7); do
    ./build/bin/rdma_sync_barrier_demo 16 $pe tcp://ip1:8765 8 0 0 sync_all &
    pids+=($!)
done
for pid in ${pids[@]}; do wait $pid; done
```

Run the following command on server B:

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # Required only on the Ascend950 platform. Set this based on the NIC type. See the environment variable description.
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
pids=()
for pe in $(seq 8 15); do
    ./build/bin/rdma_sync_barrier_demo 16 $pe tcp://ip1:8765 8 8 0 sync_all &
    pids+=($!)
done
for pid in ${pids[@]}; do wait $pid; done
```

### Command Line Parameters

```bash
./rdma_sync_barrier_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [demo_type]
```

| Parameter | Description |
|------|------|
| n_pes | Number of global PEs |
| pe_id | PE ID of the current process |
| ipport | IP address and port number required for SHMEM initialization, in the format `tcp://<IP_address>:<port_number>`. To perform a cross-server test, set the IP address to that of the host where PE 0 is located |
| g_npus | Number of NPUs started on the current server |
| f_pe | ID of the first PE used on the current server |
| f_npu | ID of the first NPU used to run this sample on the current server |
| demo_type | Optional. Specifies the demo type. The default value is `sync_all`. For supported values, see the table below |

**Values of demo_type:**

| Value | Description |
|------|------|
| `sync_all` | Performs global synchronization using `aclshmemx_roce_sync_all` |
| `sync_all_buf` | Performs global synchronization using `aclshmemx_roce_sync_all(buf, sync_id)` |
| `barrier_all` | Performs global synchronization using `aclshmemx_roce_barrier_all` |
| `barrier_all_buf` | Performs global synchronization using `aclshmemx_roce_barrier_all(buf, sync_id)` |
| `sync_team` | Performs synchronization within the team using `aclshmemx_roce_team_sync(team)` |
| `sync_team_buf` | Performs synchronization within the team using `aclshmemx_roce_team_sync(team, buf, sync_id)` |
| `barrier_team` | Performs synchronization within the team using `aclshmemx_roce_barrier(team)` |
| `barrier_team_buf` | Performs synchronization within the team using `aclshmemx_roce_barrier(team, buf, sync_id)` |

### Expected Output

Each PE sends its own data (with the value of `pe_id + 10`) to all other PEs. After synchronization is complete, each PE checks whether the received data is correct. If the check passes, the following output is displayed:

```sh
[PASS] check success, pe=<pe_id>
```

If the check fails, the specific mismatch information is printed:

```sh
[FAIL] pe=<pe_id> offset=<offset> got=<actual> expected=<expected>
```
