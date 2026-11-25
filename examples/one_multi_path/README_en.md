# One Path + Multi Path Single-Core Sharded Transfer Sample

## Sample Introduction

This sample verifies that a single AIV core on Ascend950 accesses two independent remote HBM
blocks through the `one_path` and `multi_path` paths respectively. It focuses on fabric handle
exchange, dual-path mapping, and the single-core sharded transfer flow.

Each PE allocates and exports two local physical memory blocks, and exchanges the two fabric
handles through TCP allgather. Each PE then selects the next PE based on the ring relationship:

```text
peer_pe = (pe_id + 1) % n_pes
```

Each PE imports the two fabric handles of its peer once each, using the fixed `one_path`
`linkType=2` and `multi_path` `linkType=3`, and maps them to two remote VAs. The two remote
memory blocks store the same test data. The kernel splits the data into two halves at a
32-byte boundary in the middle, and completes the transfer with one kernel launch and one
AIV block:

```text
A single AIV block
├── one_path remote VA -> data[0, split)
└── multi_path remote VA -> data[split, size)
```

The two transfer segments execute sequentially in the same kernel. This sample only verifies
data access correctness of the two link mappings. It does not verify concurrency between
`one_path` and `multi_path`, and does not collect or compare bandwidth.

## Supported Platforms and Prerequisites

- An Ascend950 runtime environment.
- CANN provides `aclrtMemExportToShareableHandleV2`, `aclrtMemImportFromShareableHandleV2`, and
  `aclrtMemMapSetLink`.
- For cross-machine running, the two consecutive ports after `ipport` must be reachable.
  `port+1` is used to exchange fabric handles, and `port+2` is used to exchange status.
- Each pair of adjacent PEs in the ring supports `one_path` `linkType=2` and
  `multi_path` `linkType=3`.

This sample does not support A2/A3. The actual routing of `linkType` is determined by the CANN
runtime and the hardware network topology. Before running, confirm that the CANN and driver
versions in use support the preceding APIs and the corresponding link types.

## Directory Structure

```text
examples/one_multi_path/
├── CMakeLists.txt                    # Sample build configuration
├── main.cpp                          # Initialization, fabric handle exchange, mapping, and result verification
├── one_multi_path_kernel.cpp         # Single AIV block dual-path sharded transfer kernel
├── one_multi_path_kernel.h           # Kernel launch interface
├── README.md                         # Usage instructions
└── run.sh                            # Single-machine or multi-machine multi-process launch script
```

## Call Flow

1. Each PE completes `aclInit`, `aclrtSetDevice`, and `aclshmemx_init_attr`.
2. Each PE allocates two local HBM blocks using `aclrtMallocPhysical`, and fills the test data
   of both memory blocks with `pe_id + 10`.
3. All PEs exchange the two fabric handles through TCP gather/broadcast.
4. Each PE selects the next PE in the ring, and imports its two fabric handles once each.
5. `aclrtMemMapSetLink(2)` and `aclrtMemMapSetLink(3)` are called on the two imported handles
   respectively.
6. The two imported handles are mapped to the `one_path` remote VA and the `multi_path` remote
   VA respectively.
7. The data is split into two halves at a 32-byte boundary, and one AIV block is launched. If a
   single transfer exceeds 16 KB, the kernel loops over chunks internally.
8. The same kernel first transfers the first half through the `one_path` remote VA, and then
   transfers the second half through the `multi_path` remote VA.
9. The results are copied to the host machine. After each `int32_t` element is verified, the
   sample exchanges the final status and releases resources.

## Compilation

```bash
bash scripts/build.sh -cann -examples -soc_type Ascend950
```

Compilation output:

```text
build/bin/one_multi_path
```

The sample is included in the build when examples is enabled and `SOC_TYPE=Ascend950`.

`-cann` enables the build with CANN public interfaces, `-examples` builds samples, and
`-soc_type Ascend950` selects the Ascend950 backend. For the complete compilation parameters,
refer to [Compilation and Build](../../docs/compilation_build_guide_en.md).

## Running

Two NPUs on one machine:

```bash
cd examples/one_multi_path
bash run.sh -pes 2 -gnpus 2 -fpe 0 -fnpu 0 -size 2048
```

Four NPUs on one machine:

```bash
bash run.sh -pes 4 -gnpus 4 -fpe 0 -fnpu 0 -size 2048
```

When two machines each use two NPUs, both sides must use the same bootstrap address.

Machine A:

```bash
bash run.sh -pes 4 -ipport tcp://<A_IP>:8899 -gnpus 2 -fpe 0 -fnpu 0 -size 2048
```

Machine B:

```bash
bash run.sh -pes 4 -ipport tcp://<A_IP>:8899 -gnpus 2 -fpe 2 -fnpu 0 -size 2048
```

The script requires each machine to own a contiguous PE range, and maps the local PEs to
consecutive logical NPUs in order:

```text
local_index = pe_id - fpe
device_id   = fnpu + local_index
```

In multi-machine running, `-pes` and `-ipport` must be identical on all machines, the
`[fpe, fpe + gnpus)` ranges must not overlap, and the PE ranges of all machines combined must
cover `[0, pes)`. The IP in `ipport` must be an address of the machine that launches PE 0 and
is accessible to the other machines.

## Parameter Description

| Parameter | Default Value | Description |
| --- | --- | --- |
| `-pes` | `2` | Total number of PEs, at least 2 |
| `-ipport` | `tcp://127.0.0.1:8899` | SHMEM bootstrap address |
| `-gnpus` | `2` | Number of PEs/NPUs launched on the current machine |
| `-fpe` | `0` | First PE ID on the current machine |
| `-fnpu` / `-dev` | `0` | First logical NPU ID on the current machine |
| `-size` | `2048` | Actual data size for verification, in KB. The value must be a positive integer |

`one_path` `linkType=2` and `multi_path` `linkType=3` are fixed inside the sample and do not
need to be passed through the command line. The sample rounds the mapping size of each physical
memory block up to the 2 MB huge page size. `-size` still indicates the amount of data actually
transferred and verified.

## Expected Output

```text
PE 0: path=one_path, offset=0, size=1048576
PE 0: path=multi_path, offset=1048576, size=1048576
PE 0: single-core one_path/multi_path copy from PE 1 PASSED
PE 1: single-core one_path/multi_path copy from PE 0 PASSED
```

A passed verification only proves that both remote VAs are accessible correctly. Whether the
physical traffic passes the expected link according to `linkType` is still determined by the
CANN runtime and the actual network topology. If necessary, confirm this by using platform link
counters.

## FAQ

- No `build/bin/one_multi_path` after compilation: Confirm that both `-examples` and
  `-soc_type Ascend950` are passed, and check `SOC_TYPE`, `USE_EXAMPLES`, and
  `ACLSHMEM_ONE_MULTI_PATH` in the CMake output. CMake compiles this sample only when the CANN
  headers and `libascendcl.so` both provide the Fabric Handle APIs and `aclrtMemMapSetLink`.
- Initialization fails or processes keep waiting: Confirm that all PEs use the same and
  reachable `-ipport`, that the PE ranges are complete without overlap, and that the bootstrap
  ports are not occupied by other tasks.
- `aclrtMemMapSetLink` or remote handle mapping fails: Confirm that CANN, the driver, and the
  network support `linkType=2/3` between the target PEs, and check the actual topology of
  adjacent PEs in the ring.
- Data verification passes but the physical link cannot be confirmed: The sample does not read
  link counters. Use platform-side link observation tools for further confirmation.
