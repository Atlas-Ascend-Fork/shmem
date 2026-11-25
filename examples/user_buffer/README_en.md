# User Buffer Sample

This sample demonstrates the multi-buffer usage of `aclshmemx_init_attr_with_buffers`. The
caller creates and maps two independent Device buffers, and passes them to SHMEM as the
symmetric heap prefix in the order of the descriptor array. The sample then allocates source
data from the SHMEM-managed tail, and performs put operations to the two user buffers of
adjacent PEs through MTE or UDMA respectively.

The two source buffers use different PE and buffer characteristic values. After communication
completes, each PE copies the two user buffers back to the host machine respectively, and
performs a byte-by-byte exact comparison on the 64 KiB payload. The sample returns a failure if
any buffer has a mismatch. Unit tests also cover combinations including VA-only input,
intra-buffer offsets and out-of-bound queries, single/dual PE, tail allocation, and resource
release.

## Compilation

```bash
source /path/to/cann/set_env.sh
bash scripts/build.sh -examples
```

UDMA requires an Ascend950 build:

```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

The outputs are `build/bin/user_buffer` and `build/lib/libuser_buffer_kernel.so`.

## Running

By default, MTE runs on NPU 0 and NPU 1:

```bash
bash examples/user_buffer/run.sh
```

Run UDMA on Ascend950:

```bash
bash examples/user_buffer/run.sh -engine udma
```

You can also adjust the PEs, devices, and control plane address through `-pes`, `-gnpus`,
`-fnpu`, and `-ipport`. On success, each PE outputs:

```text
[SUCCESS] PE 0 verified 2 caller-owned buffers with mte
```

The two sets of source VA mappings and physical handles created by the user are always owned by
the user. After `aclshmem_finalize` returns, the sample unmaps the mappings and releases the
handles in reverse order. The SHMEM-owned tail is released by SHMEM itself.
