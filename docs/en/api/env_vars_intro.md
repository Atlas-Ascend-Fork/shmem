# SHMEM Env Vars

## Initialization Related

When the unique ID API is used for initialization, you need to manually configure the environment variable `SHMEM_UID_SESSION_ID` or `SHMEM_UID_SOCK_IFNAME`. If both are configured, only `SHMEM_UID_SESSION_ID` is read. When specifying `SHMEM_UID_SESSION_ID`, ensure that the IP address is reachable and the port is idle. When specifying `SHMEM_UID_SOCK_IFNAME`, ensure that the network protocol address specified by the network port exists.

* `SHMEM_UID_SESSION_ID`: Directly specifies the IP address and port number of the listening socket of PE 0. The following formats are supported:
  - **IPv4 literal**: `ip:port`, for example `SHMEM_UID_SESSION_ID=192.168.1.100:1234`
  - **IPv6 literal**: `[ip]:port`, for example `SHMEM_UID_SESSION_ID=[::1]:886`
  - **Host name**: `hostname:port`, for example `SHMEM_UID_SESSION_ID=my-server:5555` (the host name is resolved to an actual IP address through the system name resolution service (DNS, /etc/hosts, and so on))
SHMEM_UID_SESSION_ID configuration example (select one):

```bash
export SHMEM_UID_SESSION_ID=127.0.0.1:1234
export SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
export SHMEM_UID_SESSION_ID=localhost:8888
```

* `SHMEM_UID_SOCK_IFNAME`: Specifies the network port name and network layer protocol of the listening socket of PE 0. The following two formats are supported:
  - **Network port name + protocol**: `<ifname>:<inet4|inet6>`. The address family is specified explicitly. For example, `SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4` uses IPv4, and `SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6` uses IPv6.
  - **Network port name only**: `<ifname>`. If the protocol family is not specified, the available address family of the network port is automatically detected (the available IPv4/IPv6 addresses are printed in the log). If both are available, IPv4 is preferred. If only IPv6 is available, IPv6 is selected. If neither is available, an error is reported. For example, `SHMEM_UID_SOCK_IFNAME=eth0`.
SHMEM_UID_SOCK_IFNAME configuration example (select one):

```bash
export SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  # Use IPv4
export SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  # Use IPv6
export SHMEM_UID_SOCK_IFNAME=eth0           # Automatically detect the available protocol (prefer IPv4)
```

If neither of the preceding two environment variables is configured, available network ports are automatically searched (both IPv4 and IPv6 are supported, and virtual network ports such as lo/docker/veth/br-/virbr/tun/tap are skipped).

### RDMA Scenario

In RDMA scenarios, configure TC and SL.

* `HCCL_RDMA_TC`: Reuses the HCCL environment variable to configure the traffic class of the RDMA NIC. The value range is [0, 255] and must be an integer multiple of 4. Default value: 132. [Reference](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0089.html)
* `HCCL_RDMA_SL`: Reuses the HCCL environment variable to configure the service level of the RDMA NIC. This value must be consistent with the PFC priority configured on the NIC. An inconsistency may cause performance deterioration. The value must be an integer in the range [0, 7]. Default value: 4. [Reference](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0090.html)

## Multi-instance Related

Since each instance has an independent bootstrap, an available port must be provided when each bootstrap is built.

> **Precautions**: In default mode, each initialized instance exclusively occupies a port. Re-initialization is not supported while the port is occupied. Concurrent instances must use different ports. If a single instance is initialized repeatedly, ensure that the previous `finalize` has released the port before re-initialization to prevent port conflicts.

* `SHMEM_INSTANCE_PORT_RANGE`: Specifies the available port range.
SHMEM_INSTANCE_PORT_RANGE configuration example:
export SHMEM_INSTANCE_PORT_RANGE=1024:2047

## Log Related

For details about log-related environment variables, see [SHMEM Logs](../debug/log_debug.md).

## Profiling Related

SHMEM provides a profile data tracing tool. By collecting the number of system clock cycles and converting it into actual time, the tool accurately quantifies the MTE movement performance in different blocks (compute cores) and frames (trace point IDs). For details, see [Using the Profiler to Collect Profile Data in a Sample Project](../debug/profiling.md).

* `SHMEM_CYCLE_PROF_PE`: Sets PEs on which profiling is to be performed. The value range of `pe_id` is [0, PEs-1]. To cancel profiling, run the `unset SHMEM_CYCLE_PROF_PE` command.
