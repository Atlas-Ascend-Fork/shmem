# Sample Introduction

## Environment Requirements

Same as the environment requirements in [rdma_demo](../rdma_demo/README_en.md).

## Usage

1. Compile in the shmem/ directory.
    For the RDMA compilation parameters (A2/A3, and the `XSCALE` / `HNS_1825` backends of
    Ascend950), refer to [Compilation and Build - RDMA Parameters](../../docs/compilation_build_guide_en.md#rdma-parameters).

2. Run `bash run.sh` directly in the `examples/rdma_perftest_demo` directory, or run in the shmem/ directory.
    > Note: The Ascend950 platform requires setting the `IBV_EXTEND_DRIVERS` environment variable. For details, see [Environment Variable Description](../rdma_demo/README_en.md#ibv_extend_drivers-environment-variable).

    - Commands for two NPUs on one machine

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_plugin.so>  # Only the Ascend950 platform requires setting this environment variable based on the NIC type. For details, see the environment variable description
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_perftest_demo 2 0 tcp://127.0.0.1:8899 2 0 0 highlevel_put_pingpong_latency 64 # PE 0
    ./build/bin/rdma_perftest_demo 2 1 tcp://127.0.0.1:8899 2 0 0 highlevel_put_pingpong_latency 64 # PE 1
    ```

    > Note: \<shmem-root-directory\> is the root directory of the SHMEM project, and \<path_to_plugin.so\> is the plugin library path set based on the NIC type.
    - Commands for two NPUs on two machines

    Assume that the IP of machine A is ip1 and the IP of machine B is ip2.
    Run the following commands on machine A:

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_plugin.so>  # Only the Ascend950 platform requires setting this environment variable based on the NIC type. For details, see the environment variable description
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_perftest_demo 2 0 tcp://ip1:8765 1 0 0 highlevel_put_pingpong_latency 64 # PE 0
    ```

    Meanwhile, run the following commands on machine B:

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_plugin.so>  # Only the Ascend950 platform requires setting this environment variable based on the NIC type. For details, see the environment variable description
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_perftest_demo 2 1 tcp://ip1:8765 1 1 0 highlevel_put_pingpong_latency 64 # PE 1
    ```

    > Note: \<shmem-root-directory\> is the root directory of the SHMEM project, and \<path_to_plugin.so\> is the plugin library path set based on the NIC type.

3. Command line parameter description

    ```bash
    ./rdma_perftest_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> <test_type> <msg_len>
    ```

    - n_pes: Global number of PEs. Only 2 PEs are supported.
    - pe_id: PE ID of the current process.
    - ipport: IP address and port required for SHMEM initialization, in the format
      `tcp://<IP>:<port>`. For a cross-machine test, set the IP to the IP of the host machine
      where PE 0 resides.
    - g_npus: Number of NPUs launched on the current machine.
    - f_pe: First PE ID used on the current machine.
    - f_npu: ID of the first NPU used by the current machine to run this sample.
    - test_type: Test type.
        - highlevel_put_pingpong_latency: Tests the pingpong latency of the high-level Put
          interface.
        - postsend_cost: Tests the time consumed by the postsend interface.
        - highlevel_put_bw: Tests the bandwidth of the high-level Put interface.
        - rdma_mte_bw: Tests the bandwidth when MTE and RDMA are issued in parallel.
    - msg_len: Size of the data transferred in the test, in bytes (Byte).
