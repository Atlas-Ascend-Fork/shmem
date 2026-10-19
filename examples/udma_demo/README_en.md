# Sample Introduction

## Version and Platform Support

- The UDMA sample supports only the Ascend950 platform. Other platforms do not support this sample.
- CANN 9.1.0 provides the HCOMM resource APIs required by UDMA, such as `HcommEndpointCreate`, `HcommMemReg`, `HcommChannelCreate`, and `HcommChannelGetStatus`. Download and install the toolkit and ops packages corresponding to Ascend950 from the [CANN 9.1.0 resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0). CANN versions earlier than 9.1.0 are not supported by this sample.
- During initialization, the required HCOMM symbols are loaded and checked. Before building, run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (use the corresponding `set_env.sh` for a custom installation path) and build with `-soc_type Ascend950`. If the HCOMM runtime library or symbols are incomplete, the initialization fails.
- After initialization, the sample enables `aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG)` and calls `aclshmemx_report_exception()` after kernel synchronization to demonstrate how to call the UDMA exception reporting APIs.

Instructions:

0. Before running, confirm that UDMA is enabled in the environment:
- Single server: run `npu-smi info -t topo` to check the inter-NPU topology. If the connection type between NPUs is `UB`, UDMA is supported:
![Example output of npu-smi info -t topo](../../docs/images/udma_demo_env.png)
- Cross-server: server and pod models support cross-server UDMA communication. Before running, confirm that the UB links between nodes are connected. You can use `hccn_tool` to send a UB ping between the nodes (or use the hccl_test tool shipped with CANN to run collective communication test cases across nodes, see [HCCL performance test tool](https://www.hiascend.com/document/redirect/CANNCommunityToolHcclTest)). For customized models, determine whether UDMA is supported based on the specific hardware configuration.
  `hccn_tool` is installed with the Ascend HDK driver package and is located in the `<driver installation path>/driver/tools/` directory (the default driver installation path is `/usr/local/Ascend`).
  1. Run the following command on each of the two nodes to query the EID of the local NPU (the `IP Address` in the `ipourma` field of the output is the available `eid`):
  ```bash
  hccn_tool -g -dev_info -i <devid>
  ```
  2. On either node, send a UB ping to the peer EID (`src_eid` is the local EID and `dst_eid` is the peer EID):
  ```bash
  hccn_tool -t -ub_ping -i <devid> -src_eid <src_eid> -dst_eid <dst_eid> [-pkt <size>] [-cnt <size>] [-interval <size>]
  ```
  In the command, `pkt` specifies the ping packet size (in bytes), `cnt` specifies the number of packets, and `interval` specifies the sending interval.
  If all packets receive replies, the UB link between the nodes is connected.

1. Build in the `shmem/` directory:
```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

2. Run in the `shmem/` directory:
```bash
bash examples/udma_demo/run.sh 0 # AllGather test
bash examples/udma_demo/run.sh 1 # put signal test
bash examples/udma_demo/run.sh 0 -shared_jetty 1 # AllGather test with shared jetty enabled
```
By default, the script launches with 8 devices on a single server, sequentially starting `PE 0` to `PE 7`, and waits for all processes to exit.

`run.sh` automatically sets `PROJECT_ROOT`, `LD_LIBRARY_PATH`, and `SHMEM_UID_SESSION_ID` (based on the `ipport` parameter). You do not need to manually export these environment variables.

The UDMA advanced RMA APIs use `PIPE_MTE3` to submit WQEs by default, which requires a UB scratch area. The default configuration is `offset = 189 * 1024`, `ub_size = 128` bytes, and `sync_id = 0`. If the sample or a service kernel needs to reuse this UB area, adjust it via `aclshmemx_set_udma_config(offset, ub_size, sync_id)`. `ub_size` must be at least 128 bytes to hold a complete WQE staging block for the current UDMA data movement operation.

3. Command-line parameters of the run.sh script

The script supports one optional named flag, which may appear anywhere on the command line; the remaining parameters are positional and all optional, with default values used if they are not specified.

```bash
bash examples/udma_demo/run.sh [-shared_jetty 0|1] <test_type> <n_pes> <g_npus> <ipport> <f_pe> <local_pes> <f_npu>
```

- -shared_jetty: optional named flag. Value `1` enables shared jetty; value `0` (the default when the flag is absent) disables it. When enabled, it is set through the initialization parameter `attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED`; the configuration must be set before initialization and must be identical on every PE. Shared jetty makes peers resolving to the same local EID share one underlying jetty (one SQ/CQ ring and one queue state). Enabling it requires the channel config APIs of the running libhcomm.so; otherwise initialization fails with `ACLSHMEM_NOT_SUPPORTED`. In addition, send operations within one sharing group must be issued serially by the caller.
- test_type: test type. Value 0 (default) indicates the all-gather test, and value 1 indicates the put signal test.
- n_pes: total number of global PEs across all servers (default 8).
- g_npus: number of NPU devices used on the local server (default 8).
- ipport: bootstrap node address, in the format `IP:PORT` (default `127.0.0.1:8899`). It must be the IP address of node0 and reachable from all nodes.
- f_pe: first global PE ID used on the local server (default 0).
- local_pes: number of PE processes to start on the local server (default: equal to n_pes).
- f_npu: ID of the first NPU device used on the local server (default 0).

Multi-server launch example (2 servers x 8 devices = 16 PEs; assume the IP address of node0 is 192.168.1.10):
```bash
# node0: f_pe=0, starting global PEs 0 to 7
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 0 8 0
# node1: f_pe=8, starting global PEs 8 to 15
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 8 8 0
```
To enable shared jetty in a multi-server launch, append `-shared_jetty 1` on every node (the value must be identical on all nodes).

### mssanitizer Check

Before running mssanitizer, rebuild with `-mssanitizer`:

```bash
bash scripts/build.sh -examples -soc_type Ascend950 -mssanitizer
```

Currently, the experiment restores the struct field padding on the MTE3 path, and both the all-gather and put signal kernels perform a one-time zeroing of the entire 128-byte scratch area before their first call (`init_udma_wqe_scratch` in `udma_demo_kernel.cpp`). This initializes the original values that may be read during the read-modify-write of the struct bit fields and also covers the 32-byte tail padding of put signal. It is used to verify whether full pre-initialization by the caller can eliminate uninitialized reads reported by initcheck.

```bash
mssanitizer -- bash examples/udma_demo/run.sh 0
mssanitizer -- bash examples/udma_demo/run.sh 1
```

4. Command-line parameters of the underlying binary
```bash
./udma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [test_type] [shared_jetty]
```

- n_pes: number of global PEs.
- pe_id: PE ID of the current process.
- ipport: IP address and port number required for SHMEM initialization, in the format tcp://`<IP_address>:<port_number>`.
- g_npus: number of NPUs started on the current server.
- f_pe: ID of the first PE used on the current server.
- f_npu: ID of the first NPU used to run this sample on the current server.
- test_type: test type (optional). Value 0 (default) indicates the all-gather test, and value 1 indicates the put signal test.
- shared_jetty: whether to enable shared jetty (optional). Value 0 (default) disables it and value 1 enables it; the value must be identical on every PE.
