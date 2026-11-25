# shmem-config Command Reference

`shmem-config` is a configuration query tool bundled with the SHMEM Python package. It is available after `cann-shmem` is installed. It queries the installation path, backend type, compilation and linking parameters, environment detection, and diagnostic information. For the complete environment preparation and installation steps for A2/A3/Ascend950, refer to [Quick Start](../../quickstart_en.md#5-getting-started).

## Command Overview

| Command | Purpose | Example Output |
|---|---|---|
| `shmem-config --version` | Queries the SHMEM version number | `1.0.0` |
| `shmem-config --backend` | Current SoC backend selection result | `910` or `950` |
| `shmem-config --include` | C/C++ header file include path | `/usr/local/lib/python3.x/.../shmem/include` |
| `shmem-config --lib` | Backend .so library directory path | `/usr/local/lib/python3.x/.../shmem/backends/910` |
| `shmem-config --ldflags` | Recommended linking parameters (including `-I`, `-L`, `-l`) | `-I.../include -L.../backends/910 -lshmem` |
| `shmem-config --rpath` | Recommended rpath linking parameter | `-Wl,-rpath,.../backends/910` |
| `shmem-config --root` | SHMEM package root directory path | `/usr/local/lib/python3.x/.../shmem` |
| `shmem-config --runtime-root` | Root path of the libshmem.so loaded at runtime | `/usr/local/lib/python3.x/.../shmem` |
| `shmem-config --diagnose` | Structured diagnostic information (JSON) | See the example below |
| `shmem-config --check` | Runs the environment and package check script | Terminal output, including chip/card count/MTE/SDMA/UDMA/RDMA status |

## Command Details

### --version

This command outputs the version number of the SHMEM package. The data source is the first line of `shmem/version.info`.

```bash
$ shmem-config --version
1.0.0
```

### --backend

This command automatically detects the current SoC model through the `aclrtGetSocName()` API of `libascendcl.so` and returns a backend identifier.

- `Ascend910` / `DAV-220*` series → `910`
- `Ascend950` / `DAV-3510` series → `950`
- Returns the fallback value `910` if detection fails

```bash
$ shmem-config --backend
910
```

### --include / --lib / --root

These commands query key paths in the SHMEM installation directory. These paths are commonly referenced in CMake or Makefile.

```bash
$ shmem-config --include
/usr/local/lib/python3.11/site-packages/shmem/include

$ shmem-config --lib
/usr/local/lib/python3.11/site-packages/shmem/backends/910

$ shmem-config --root
/usr/local/lib/python3.11/site-packages/shmem
```

### --ldflags / --rpath

These commands output the compilation and linking flags of SHMEM. You can use these flags in `gcc` / `g++` command lines or CMake variables. The SHMEM public header files depend on CANN header files. Before compilation, load the CANN environment and additionally pass in paths required by the current CANN installation, such as `${ASCEND_HOME_PATH}/include`.

```bash
$ shmem-config --ldflags
-I/usr/local/lib/python3.11/site-packages/shmem/include \
-L/usr/local/lib/python3.11/site-packages/shmem/backends/910 -lshmem

$ shmem-config --rpath
-Wl,-rpath,/usr/local/lib/python3.11/site-packages/shmem/backends/910
```

Some package versions do not embed `$ORIGIN` in their internal shared libraries. To ensure compatibility with these versions, you also need to add the directory output by `shmem-config --lib` to `LD_LIBRARY_PATH`. Perform this step before running an external C/C++ program.

### --runtime-root

This command queries the package root path that contains the `libshmem.so` actually loaded by the current process. The command first reads `/proc/self/maps` and then searches `LD_LIBRARY_PATH`. If `libshmem.so` is found in neither location, the output is empty.

```bash
$ shmem-config --runtime-root
/usr/local/lib/python3.11/site-packages/shmem
```

### --diagnose

This command outputs structured diagnostic information in JSON format. The information is suitable for CI integration or automated inspection. For a fresh installation, the environment check runs first when the package is loaded for the first time. If you need to directly parse the JSON, you should first run `shmem-config --check` separately once, and then collect the standard output of `--diagnose`. The output fields include:

| Field | Description |
|---|---|
| `version` | SHMEM package version |
| `backend.selected` | Currently selected backend |
| `backend.auto_detected_soc` | Automatically detected SoC model |
| `release_build` | Whether the build is a Release build |
| `native_load` | Whether the native extension and its dependent libraries are loaded successfully |
| `multi_so_conflict` | Whether conflicts exist among multiple libshmem.so versions |
| `multi_so_conflict.loaded_paths` | List of loaded libshmem.so paths |
| `backend_artifacts` | Backend .so file integrity check result |
| `runtime_root` | Whether the runtime loading path is consistent with the package path |
| `degraded` | Whether the package runs in a degraded mode |
| `next_steps` | Diagnostic conclusion and recommended actions |

Example output:

```json
{
  "version": "9.0.0.beta.2",
  "backend": {
    "selected": "910",
    "auto_detected_soc": "Ascend910"
  },
  "release_build": true,
  "native_load": {
    "ok": true,
    "error": null
  },
  "multi_so_conflict": {
    "detected": false,
    "loaded_paths": []
  },
  "backend_artifacts": {
    "backend": "910",
    "complete": true,
    "missing": []
  },
  "runtime_root": {
    "path": "/usr/local/lib/python3.11/site-packages/shmem",
    "matches_package_root": true,
    "package_root": "/usr/local/lib/python3.11/site-packages/shmem"
  },
  "degraded": false,
  "next_steps": [
    "No issues detected. SHMEM is ready."
  ]
}
```

### --check [--package &lt;path&gt;]

This command runs the environment and package check script `preinstall_check.sh` bundled with the wheel. The check covers the following items:

1. Chip platform identification
2. CANN / HDK version baseline
3. Topology links and MTE support
4. SDMA support (910B/C platforms)
5. UDMA support (Ascend950 platform)
6. RDMA NIC and network health status
7. Package content integrity (when `--package` is specified)

```bash
# Check the current environment
shmem-config --check

# Check the current environment and the specified package directory
shmem-config --check --package /absolute/path/to/package/root
```

A `WARN` in the check result indicates that the corresponding capability needs confirmation or does not meet the conditions. The current check is used for diagnosis and provides handling suggestions. The check may still return success even if capability warnings exist. Use the final summary and the output of each check item as the judgment basis.

The expected results on each platform are as follows:

| Check Item | A2/A3 | Ascend950 |
|---|---|---|
| Chip identification | `910B/C` | `950` |
| MTE | Checks HCCS/SIO links based on the topology | Supported by default |
| SDMA | Determined based on the CANN version and environment | Displays `N/A` |
| UDMA | Displays `N/A` | Checks the UB networking and UDMA artifacts in the package |
| RDMA | Checks the RoCE network | Checks the XSCALE or HNS_1825 NICs and the package backend |

## Automatic Check on First Import

After the wheel is installed, an environment check automatically runs once when the current user loads the `shmem` package for the first time. Running `import shmem` or most `shmem-config` subcommands triggers the package loading:

```bash
python3 -c "import shmem; print('import shmem success')"
```

The check result is recorded by SHMEM version in `~/.cache/shmem/.env_checked`. Subsequent imports of the same version do not trigger the check again. The automatic check is skipped when it takes more than 15 seconds, so it does not block module import. If you need to run the check again, directly run `shmem-config --check`.

If you need to re-verify the first import process, delete the check marker of the current user and import the package again:

```bash
rm -f ~/.cache/shmem/.env_checked
python3 -c "import shmem"
```

## Installation Acceptance and Troubleshooting

Run the following commands after the installation is complete:

```bash
shmem-config --version
shmem-config --backend
shmem-config --diagnose
```

The following conditions indicate that the basic Python wheel installation is normal:

- `native_load.ok` is `true`, and `degraded` is `false`.
- The `--backend` output is `910` on A2/A3 and `950` on Ascend950.
- `backend_artifacts.complete` is `true`.
- `multi_so_conflict.detected` is `false`.

If Ascend950 is selected as `910`, confirm that the driver is loaded and the device is visible to the current user. The system falls back to the default 910 backend when automatic identification fails.

If `multi_so_conflict.detected` is `true`, check `LD_LIBRARY_PATH` against `loaded_paths`. Remove the old installation directories or duplicate `libshmem.so` paths, and restart the process. A Release package may also directly report `Multiple libshmem.so instances detected` and exit during the startup check.

If `shmem-config` is not found, confirm that the wheel is installed in the current Python environment. Also confirm that the executable directory of this environment is in `PATH`:

```bash
python3 -m pip show cann-shmem
command -v shmem-config
```

## Use Cases

- **Compiling external projects**: Use `--ldflags` and `--rpath` to assemble compilation commands or CMake variables
- **CI inspection**: Use the JSON output of `--diagnose` for automated health checks
- **Troubleshooting**: Use `--runtime-root` to confirm the actually loaded SHMEM version, and use `--diagnose` to check .so conflicts and integrity
- **Pre-deployment verification**: Use `--check` to check whether the environment meets the running conditions at one time
