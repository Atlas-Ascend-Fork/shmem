# SHMEM Coding Style Guide

This guide is intended for contributors to the SHMEM open source repository. It defines the basic requirements for public APIs, public header files, implementation files, and sample code. It also covers comments, error handling, memory alignment, Device-side checks, environment variables, namespace usage, and verifiable checks.

This guide serves as the rule input for code reviews and automated checks. Manual reviews and automated tools should first check the objectives, naming summary table, and hard constraints in Chapter 1. Then they should verify each item against the subsequent chapters.

All new code should first comply with this guide. When modifying existing code, align with this guide step by step, provided that source code compatibility, ABI compatibility, and the user migration path are not broken. Once public APIs, public types, public macros, and public structure fields are released, do not rename them at will. Do not change their parameter meanings, return code semantics, or synchronization completion semantics.

> This guide applies to the SHMEM repository (`include/`, `src/`, `examples/`, `tests/`, `tools/`, `scripts/`, and code snippets in external-facing documentation). Unless otherwise specified, paths in this guide are relative to the repository root directory.

---

## 1. Objectives, Naming Summary Table, and Hard Constraints

### 1.1 Objectives

The core objective of this guide is to define the external interface boundary of SHMEM and prevent internal implementations from leaking into user-visible areas.

External API prefix rules:

- Standard interfaces use `aclshmem_*`.
- Extended interfaces use `aclshmemx_*`.

The following are explicitly prohibited:

- `aclshmemi_*` appearing in external interfaces.
- `aclshmemi_*` appearing in external data structures.
- Calling `aclshmemi_*` in sample code.
- Describing `aclshmemi_*` in user documentation as an interface that users can depend on.
- Exposing the `ACLSHMEMI_*` public macro in public header files.

`aclshmemi_*` and `ACLSHMEMI_*` can only be used as internal implementation symbols, and should be restricted to `src/`, internal tests, or internal implementation files that are not installed.

### 1.2 Naming Rules Summary Table

The following table serves as a quick reference for the naming rules. For details, see Chapter 4.

| Object | Rule | Allowed Examples | Prohibited or Not Recommended Examples |
|---|---|---|---|
| Standard public functions | `snake_case`, using `aclshmem_*` | `aclshmem_malloc`, `aclshmem_putmem_nbi` | `aclshmemInit`, `shmem_put` |
| Extended public functions | `snake_case`, using `aclshmemx_*` | `aclshmemx_barrier_on_stream` | `aclshmemi_barrier_on_stream` |
| Internal functions | `snake_case`, `aclshmemi_*` allowed, internal implementation only | `aclshmemi_prepare_and_post_rma` | Appearing in `include/**` or `examples/**` |
| Public types | `snake_case` + `_t`, using `aclshmem_*_t` or `aclshmemx_*_t` | `aclshmem_team_t`, `aclshmemx_init_attr_t` | `aclshmemi_context_t` |
| Public structures/unions | `snake_case` + `_t`, fields in `snake_case` | `aclshmem_handle_t` | Field types exposing `aclshmemi_*` |
| Enum types | `snake_case` + `_t` | `aclshmem_error_code_t` | `AclshmemErrorCode` |
| Public enum values | All uppercase with underscores, using `ACLSHMEM_*` or `ACLSHMEMX_*` | `ACLSHMEM_SUCCESS` | `SHMEM_SUCCESS`, `AclshmemSuccess` |
| Standard public macros | `ACLSHMEM_<module/capability>_<semantic name>` | `ACLSHMEM_MAX_TEAMS`, `ACLSHMEM_SYNCBIT_SIZE` | `CANN_SHMEM_TEAM_MAX_SIZE` |
| Extended public macros | `ACLSHMEMX_<module/capability>_<semantic name>` | Name new extended macros following this template | `SHMEMX_STREAM_MAX_EVENTS` |
| Internal macros | `ACLSHMEMI_<module/capability>_<semantic name>`, internal implementation only | `ACLSHMEMI_*` in internal implementation files | Appearing in public header files |
| Environment variables | `SHMEM_<module/capability>_<semantic name>` | `SHMEM_LOG_LEVEL`, `SHMEM_UID_SESSION_ID` | `ACLSHMEM_LOG_LEVEL` |
| Local variables | `snake_case` | `local_mem_size` | `localMemSize` |
| Boolean variables | `is_` / `has_` / `can_` prefixes | `is_initialized`, `has_error` | `initialized`, `errorFlag` |
| File names | `snake_case` | `shmem_host_rma.h` | `ShmemHostRma.h` |
| Header guard macros | All uppercase with underscores, corresponding to the file path | `SHMEM_API_H`, `SHMEM_HOST_RMA_H` | `#pragma once` (inconsistent) |
| C++ namespace usage | Global `using namespace` is prohibited in public header files | `std::vector<int>` | `using namespace std;` |
| Device-side defensive checks | Unnecessary fallback checks are allowed only in debug mode | Use the unified debug check mechanism introduced by the project | Repeated `if (ptr == NULL) return;` in release hot paths |

### 1.3 Hard Constraints

#### 1.3.1 include Directory Constraints

New or modified `include/**` files must not introduce any `aclshmemi_*` interfaces or data structures.

This rule covers the following paths (including similar directories added in the future):

- `include/*.h`
- `include/host/**/*.h`
- `include/device/**/*.h`
- `include/host_device/**/*.h`
- `include/internal/**/*.h`
- `include/detail/**/*.h`
- `include/private/**/*.h`

It is not allowed to bypass the public header file check by adding directories such as `include/internal`, `include/detail`, or `include/private`. All files under `include/**` are regarded as user-visible header files. Legacy compatibility aliases should be cleaned up step by step according to the compatibility policy, but they must not be used as a basis for new code.

#### 1.3.2 Sample Constraints

New or modified `examples/**` files are allowed to use external APIs only.

Sample code must not:

- Call `aclshmemi_*`.
- Declare or depend on `aclshmemi_*` data structures.
- Use `ACLSHMEMI_*` macros.
- Include internal header files.
- Depend on undocumented environment variables.
- Depend on internal debugging entry points.
- Depend on internal allocators, internal transports, or internal synchronization implementations.

Sample code must demonstrate recommended usage, including initialization, error handling, communication completion, and resource release procedures.

#### 1.3.3 Macro Definition Constraints

Macro definitions in public header files use:

- `ACLSHMEM_<module/capability>_<semantic name>`
- `ACLSHMEMX_<module/capability>_<semantic name>`

Internal macro definitions use:

- `ACLSHMEMI_<module/capability>_<semantic name>`

Constraints:

- `ACLSHMEM_*` is used for standard public capabilities.
- `ACLSHMEMX_*` is used for extended public capabilities.
- `ACLSHMEMI_*` can only be used for internal implementation.
- `ACLSHMEMI_*` must not appear in `include/**` as a public macro.
- `ACLSHMEMI_*` must not appear in recommended usage in `examples/**`.
- `ACLSHMEMI_*` must not be presented as a dependable macro in user documentation.

Existing public macros such as `ACLSHMEM_MAX_TEAMS` and `ACLSHMEM_SYNCBIT_SIZE` can be used as naming references. Name new extended public macros by using the `ACLSHMEMX_*` template, and name new internal macros by using the `ACLSHMEMI_*` template. Restrict new internal macros to internal implementation files.

#### 1.3.4 Environment Variable Constraints

SHMEM environment variables use the `SHMEM_*` prefix, not the `ACLSHMEM_*` prefix.

Recommended:

```bash
SHMEM_LOG_LEVEL=INFO
SHMEM_UID_SESSION_ID=127.0.0.1:8899
SHMEM_CYCLE_PROF_PE=0
```

Not recommended:

```bash
ACLSHMEM_LOG_LEVEL=INFO
ACLSHMEM_UID_SESSION_ID=127.0.0.1:8899
```

Environment variables belong to runtime configuration and are not part of the C API symbol system. APIs, types, enums, and macros use `aclshmem_*`, `aclshmemx_*`, `ACLSHMEM_*`, and `ACLSHMEMX_*`; environment variables use `SHMEM_*`.

#### 1.3.5 Memory Alignment Constraints

Memory in SHMEM communication paths must meet the alignment requirements of the corresponding scenario.

When the following objects are involved, you must specify the alignment granularity, allocation source, lifecycle, and failure handling:

- Symmetric memory.
- Device memory.
- Host page-locked memory.
- UB workspace.
- LocalTensor.
- Synchronization objects such as signal, sync, flag, doorbell, and notify.
- Shared metadata accessed across PEs.
- Communication buffers and transport engine workspaces.

Do not write only generalized descriptions such as "aligned to the cache line" or "aligned according to hardware requirements".

#### 1.3.6 Memory Detection Constraints

Some addresses are not allocated by `aclrt`. If such an address enters a SHMEM communication path, a Device access path, or a cross-PE memory access path, new code must be assessed against the support scope of the memory detection tool (msSanitizer). New code must also be verified by using instrumented compilation and launching the tool.

Key scenarios to check include:

- Host addresses allocated by `malloc` / `calloc` / `realloc`.
- Host addresses allocated by `new` / `new[]`.
- Addresses allocated by `mmap` or other system interfaces.
- Addresses returned by project-specific custom allocators.
- Addresses that are not allocated by `aclrt` interfaces such as `aclrtMalloc` and `aclrtMallocHost` but are accessed by communication or the Device later.

Requirements:

- New logic where non-`aclrt` addresses enter asynchronous communication, RMA, DMA, or Device access paths should be included in the msSanitizer verification scope.
- Before submitting a PR, run local tests with msSanitizer to cover the new or modified memory paths.

#### 1.3.7 Device-Side Check Constraints

Device-side code is in performance-sensitive paths. Do not perform fallback checks unless necessary.

Device-side code in release paths should meet the following requirements:

- Do not perform unnecessary parameter fallback checks.
- Do not perform unnecessary state fallback checks.
- Do not add unnecessary platform fallback branches.
- Do not add unnecessary slow-path fallbacks.
- Do not add validity checks that can be completed earlier on the Host side to performance paths.
- Do not use complex `if-else` fallbacks on the Device side to handle incorrect user input.

All defensive or fallback `if-else` checks on the Device side should be placed in debug mode, including:

- Null pointer checks.
- PE range checks.
- Size range checks.
- Alignment checks.
- Enum validity checks.
- Address space checks.
- Workspace size checks.
- Validity checks for sync_id, event_id, and notify addresses.
- Fallback checks for unsupported chips, unsupported transport engines, and unsupported thread domains.

Note: Branches required by service semantics are not fallback checks. For example, selecting different execution paths based on block, warp, vector, AIV, AIC, or the transport engine can be retained. However, compile-time dispatch or lightweight branches are preferred to avoid repeated parameter defense in hot paths.

Recommended practices:

- Complete parameter validation before Host API calls, Host wrapper calls, or kernel launch.
- Keep only branches required by service semantics in the Device release path.
- Use the unified debug check mechanism that the project has introduced and verified in the Device debug path.
- Use `#if`, `if constexpr`, template specialization, or compile-time constant dispatch for capabilities that can be determined at compile time.
- For unsupported capabilities, return an error on the Host side instead of falling back in Device hot paths.

In the Device release path, repeated fallback checks on null pointers, PE ranges, alignment, or capabilities are not recommended. Do not directly return or enter a slow path after such checks. When adding such checks, move them to the Host entry, the Host wrapper, or a point before the kernel launch first. If a check must be retained on the Device side, explain why the service semantics require it. Before the project has uniformly introduced and verified a debug check wrapper, the external specification does not preset specific debug macro names.

#### 1.3.8 Namespace Constraints

Global `using namespace` is prohibited in public header files, including but not limited to:

```cpp
using namespace std;
using namespace shmem;
using namespace acl;
```

The prohibition covers:

- `include/**/*.h`
- `include/**/*.hpp`
- `examples/**/include/*.h`
- `examples/**/include/*.hpp`
- Public header file code snippets recommended for copying in user documentation

Public header files should use explicit fully qualified names:

```cpp
std::vector<int> values;
std::string name;
```

Source files (`.cpp`) should also avoid file-level `using namespace`. Prefer explicit fully qualified names or local namespace aliases.

When you need to shorten a long qualified name, use a namespace alias in a function or a local scope. Do not introduce aliases or `using namespace` in the global scope of a public header file.

### 1.4 Overall Principles

1. Stability of external interfaces comes first. Functions, types, macros, enum values, and structure fields in public header files are directly depended on by user code. You must consider compatibility before adding or modifying them.

2. Internal implementations must not be exposed. Internal prefixes, internal types, internal file paths, internal debugging entry points, and experimental capabilities should not appear in public header files, sample code, or user documentation.

3. Treat all of `include/**` as user-visible areas. Even if the directory name contains `internal`, `detail`, or `private`, do not place `aclshmemi_*` external interfaces or data structures in it.

4. Samples demonstrate recommended usage only. Sample code should use officially released public APIs and should not depend on internal symbols, internal header files, or undocumented environment variables.

5. Keep the boundary between Host APIs and Device APIs clear. Distinguish Host-side resource management, initialization, synchronization, and task dispatch from Device-side remote memory access, data movement, atomic operations, and synchronization operations. Keep the distinction clear in naming, comments, header file paths, and execution domain descriptions.

6. Keep the Device-side release path lightweight. Parameter checks, state checks, and capability checks that can be completed on the Host side should not be repeatedly performed as fallbacks in Device hot paths. By default, defensive `if-else` checks on the Device side are enabled only in debug mode.

7. Asynchronous semantics must be stated clearly. Successful submission, successful enqueuing, local completion, remote visibility, and communication completion are different concepts. Interface comments, samples, and tests must distinguish them.

8. Memory and alignment requirements must be specific. Code involving symmetric memory, Device memory, Host page-locked memory, UB workspace, cache line, or page must specify allocation source, alignment granularity, unit, lifecycle, and failure handling.

9. Memory that is not allocated by `aclrt` and enters communication or Device access paths must be included in the msSanitizer verification scope.

10. Environment variables use the `SHMEM_*` prefix, not the `ACLSHMEM_*` prefix.

11. Public header files must not pollute the user namespace. Global `using namespace` is prohibited.

12. Keep abstractions minimal and necessary. New wrappers should serve stable ABIs, cross-module reuse, clear boundaries, or testability, instead of creating extra layers for local code.

---

## 2. Applicable Scope

This guide applies to the C/C++ code, public header files, sample code, test code, and code snippets in external-facing documentation in the SHMEM repository.

Key constraints include:

- Naming of external APIs, external types, public macros, and public enum values.
- The boundary between `include/` and `src/`.
- Declarations, comments, and execution domain descriptions of Host APIs and Device APIs.
- The release/debug check boundary on the Device side.
- Environment variable naming.
- C++ namespace usage.
- Error codes, logs, resource release, and failure path handling.
- Semantics of asynchronous submission, stream enqueuing, local completion, remote visibility, and communication completion.
- Alignment of symmetric memory, AscendCL memory, Ascend C LocalTensor/UB, cache lines, and pages.
- msSanitizer memory detection.
- Sample code and CI style checks.

Python wrappers (`src/host/python_wrapper/` and `src/python/`) follow PEP 8, use 4 spaces and 120 columns, and keep naming consistent with C/C++ interfaces. This guide does not cover Python details.

---

## 3. File Organization

### 3.1 Directory Boundaries

The directory layout of the SHMEM repository is as follows:

- `include/`: Public declarations, intended for user installation and reference.
- `include/host/`: Host-side public declarations.
- `include/device/`: Device-side public declarations.
- `include/host_device/`: Public types, macros, and declarations shared by the Host and the Device.
- `src/`: Implementation files and internal implementation details.
- `examples/`: Recommended usage samples, which use only public APIs.
- `tests/`: Unit tests, system tests, regression tests, and performance tests.
- `docs/`: User documentation, development documentation, and description materials.
- `scripts/` and `tools/`: Build, check, generation, and auxiliary tools.

Treat all of `include/**` as user-visible areas. Do not place internal interfaces, internal data structures, or internal macros in directories such as `include/internal`, `include/detail`, or `include/private`.

The public entry header file `include/shmem.h` aggregates only stable header files, and does not directly or indirectly export internal implementation symbols.

### 3.2 Header Files and Implementation Files

- Public C/C++ declarations preferably use `.h`.
- C++ implementations use `.cpp`.
- Device-side template or inline implementations may use `.hpp`, but only within clear implementation boundaries (such as `src/**/*.hpp`). They do not constitute a stable ABI commitment to users.
- Header files should be self-contained: when users include any public header file, they should not need to depend on an extra implicit include order.
- New header files use include guards (see section 3.4) to keep the entire repository consistent.
- Public header files must not include files under `src/`.
- Code snippets in sample code and user documentation must not include internal header files.
- Public header files must not define non-`inline` function entities or non-`constexpr` global objects, to avoid One Definition Rule (ODR) and duplicate definition issues.
- Public header files should avoid introducing unnecessary heavyweight dependencies; prefer forward declarations when a forward declaration is possible.
- Global `using namespace` is prohibited in public header files.

### 3.3 Copyright Header

New source files must have a copyright statement added at the beginning of the file. C/C++ files and header files use the block comment format, and Python/Shell files use the line comment format. Keep the copyright header template consistent with [`CONTRIBUTING_en.md`](https://gitcode.com/cann/shmem/blob/master/CONTRIBUTING_en.md).

C/C++ and header file template:

```c
/**
 * Copyright (c) [Name of the copyright owner]. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

Python and Shell file template:

```bash
# Copyright (c) [Name of the copyright owner]. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ================================================================================================================
```

Notes:

- `[Name of the copyright owner]`: For personal contributions, enter your personal signature. For contributions on behalf of an employer, enter the employer name. If you have questions about copyright ownership, consult a legal advisor or the legal department of your employer.
- `2026` is the year when the file is created or modified. Change it according to the actual time.
- The copyright header in a public header file can be wrapped with Doxygen `@cond IGNORE_COPYRIGHT` / `@endcond` to avoid interfering with API documentation generation.

### 3.4 Header Guard Macros

New header files use include guards (`#ifndef` / `#define` / `#endif`). Name the macro in the all-uppercase underscore form that corresponds to the file path. Do not use `#pragma once`, to keep the entire repository consistent.

```c
#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

/* ... declarations ... */

#endif // SHMEM_HOST_RMA_H
```

Naming rules:

- The macro name is the file path with the file extension removed, converted to uppercase, and with separators changed to `_`.
- For example, `include/host/data_plane/shmem_host_rma.h` uses `SHMEM_HOST_RMA_H` in the repository.
- The entry header `include/shmem.h` uses `SHMEM_API_H`.

---

## 4. Naming Conventions

### 4.1 Public API Prefixes

SHMEM public APIs are classified into standard interfaces and extended interfaces:

- Standard interfaces use `aclshmem_*`.
- Extended interfaces use `aclshmemx_*`.
- Internal implementations may use `aclshmemi_*`, but such symbols can exist only within the internal implementation scope.

`aclshmemi_*` must not appear in public header files, sample code, or user documentation as a user-callable interface or a type that users can depend on.

Do not bypass this rule in the following ways:

```c
/* Not allowed */
typedef struct aclshmemi_context aclshmem_context_t;

/* Not allowed */
#define aclshmem_create_context aclshmemi_create_context

/* Not allowed */
static inline int aclshmem_do_something(...)
{
    return aclshmemi_do_something(...);
}
```

### 4.2 Macro Definition Naming

New macro definitions should distinguish between the public visibility scope and the internal implementation scope.

Standard public macros use:

```c
ACLSHMEM_<module/capability>_<semantic name>
```

Extended public macros use:

```c
ACLSHMEMX_<module/capability>_<semantic name>
```

Internal macros use:

```c
ACLSHMEMI_<module/capability>_<semantic name>
```

Requirements:

- New public macros in public header files use `ACLSHMEM_*` or `ACLSHMEMX_*`.
- New internal macros use `ACLSHMEMI_*`.
- `ACLSHMEMI_*` must not appear in `include/**`.
- `ACLSHMEMI_*` must not appear in recommended usage in `examples/**`.
- Both public macros and internal macros use all-uppercase names with underscores, instead of camel case or names without a project prefix.
- Debug switches, path macros, and transport detail macros that are used only for internal implementation must not be placed in public header files.
- This rule is mandatory for newly added macro definitions. If legacy compatibility macros have completed rectification, they should be removed from public header files, samples, and user documentation.

Existing public macros such as `ACLSHMEM_MAX_TEAMS` and `ACLSHMEM_SYNCBIT_SIZE` can be used as naming references. Do not use specific names that do not exist in the repository to disguise new macro examples as available interfaces. When you need to explain the naming rules, use template expressions such as `ACLSHMEM_<module/capability>_<semantic name>`, `ACLSHMEMX_<module/capability>_<semantic name>`, and `ACLSHMEMI_<module/capability>_<semantic name>`.

### 4.3 Environment Variable Naming

Environment variables use the `SHMEM_*` prefix.

Recommended:

```bash
SHMEM_LOG_LEVEL=INFO
SHMEM_UID_SESSION_ID=127.0.0.1:8899
SHMEM_CYCLE_PROF_PE=0
```

Not recommended:

```bash
ACLSHMEM_LOG_LEVEL=INFO
ACLSHMEM_UID_SESSION_ID=127.0.0.1:8899
```

Requirements for new environment variables:

- Use the `SHMEM_*` prefix.
- Use all-uppercase names with underscores.
- Do not use camel case.
- Do not use `ACLSHMEM_*`.
- Do not use internal implementation prefixes.
- You must document the default value, value range, unit, and effective time in the user documentation.
- For environment variables involving performance, memory size, transport paths, or debugging behavior, you must document the compatibility and security impact.

Example description format:

```markdown
### SHMEM_LOG_LEVEL

Controls SHMEM runtime log level.

Values:
- ERROR
- WARN
- INFO
- DEBUG

Default: WARN.
Effective time: read during SHMEM initialization, for example
`aclshmemx_init_attr`.
```

### 4.4 Standard Interfaces and Extended Interfaces

The standard interfaces `aclshmem_*` are used for stable, general, and cross-platform SHMEM capabilities, for example:

- Initialization, finalization, and basic information queries.
- Symmetric memory management.
- PE and team queries.
- RMA, atomic, synchronization, and signal operations.

The extended interfaces `aclshmemx_*` are used for platform-specific, execution domain-specific, or enhanced capabilities, for example:

- Interfaces with stream semantics.
- Device thread domain interfaces such as block / warp / vector / AIV / AIC.
- Interfaces for configuring or optimizing a specific transport engine.
- Non-standard initialization and context extension interfaces.
- Interfaces that depend on a specific chip, a specific CANN version, or experimental capabilities.

When adding a new interface, if the capability belongs to general SHMEM semantics, design it as `aclshmem_*` first. If the capability depends on a specific execution model, backend, stream, transport engine, or experimental extension, design it as `aclshmemx_*`.

---

## 5. C/C++ Formatting and Namespaces

SHMEM C/C++ code adopts the basic constraints of the Google C++ Style Guide and combines them with the existing conventions of the project. The repository root directory provides `.clang-format`, and the repository configuration takes precedence; new code must not work against clang-format through manual formatting.

Key `.clang-format` configurations:

- `BasedOnStyle: Google`
- `ColumnLimit: 120`
- `IndentWidth: 4`, `UseTab: Never`
- `PointerAlignment: Left` (pointers aligned left, for example `int *ptr`)
- `BreakBeforeBraces: Custom`, `AfterFunction: true` (function opening braces on a new line, the rest kept compact)
- `SortIncludes: false` (include order is organized semantically by developers, and alphabetical order is not enforced)

Basic requirements:

- Use 4 spaces for indentation.
- A single line must not exceed 120 columns.
- Pointers aligned left, for example `int *ptr`.
- Use `snake_case` for function names, variable names, and file names.
- Use uppercase with underscores for enum values, public macros, and public constants.
- Do not use tabs.
- When a function has many parameters, break lines by semantic grouping instead of squeezing everything onto one line.
- Intentionally unused parameters should be explicitly marked following the existing project convention, for example using `(void)param;`.
- When an exception is truly required, first discuss whether to adjust the formatting configuration.

Example:

```c
ACLSHMEM_HOST_API void aclshmemx_putmem_on_stream(
    void *dst,
    void *src,
    size_t elem_size,
    int32_t pe,
    aclrtStream stream);
```

### 5.1 Namespaces

The following is forbidden in public headers:

```cpp
using namespace std;
```

Not recommended:

```cpp
// public header
using namespace std;

class Foo {
public:
    vector<int> values;
};
```

Recommended:

```cpp
// public header
class Foo {
public:
    std::vector<int> values;
};
```

File-level `using namespace` should also be avoided in source files.

Not recommended:

```cpp
using namespace std;

void foo()
{
    vector<int> values;
}
```

Recommended:

```cpp
void foo()
{
    std::vector<int> values;
}
```

Local namespace aliases inside functions are allowed:

```cpp
void create_endpoint()
{
    namespace local_alias = fully::qualified::namespace_name;
    local_alias::Endpoint endpoint;
}
```

---

## 6. Public Types and ABI

Public structs, enums, and type aliases must remain readable, stable, and evolvable.

### 6.1 Public Type Requirements

- Use `aclshmem_*_t` or `aclshmemx_*_t` for type names.
- Use `ACLSHMEM_*` or `ACLSHMEMX_*` for enum values.
- Use `snake_case` for public struct fields.
- Public struct fields should have comments describing their purpose, lifetime, units, and valid ranges.
- It is recommended that ABI-facing structs contain `version`, `size`, or an equivalent version field.
- When adding fields to external structs, prefer appending them at the end and do not reorder existing fields.
- Pointer fields must state ownership, whether they can be null, whether symmetric memory is required, and whether alignment is required.
- Enum fields must state how invalid values are handled.
- Struct initialization should provide an initialization macro, an initialization function, or a clear zero-initialization convention.
- Public data structures must not contain `aclshmemi_*` field types, internal handles, or internal context pointers.

Example of the public initialization attribute struct in the repository:

```cpp
typedef struct aclshmemx_init_attr {
    int my_pe;
    int n_pes;
    char ip_port[ACLSHMEM_MAX_IP_PORT_LEN] = {};
    uint64_t local_mem_size;
    aclshmem_init_optional_attr_t option_attr = {
        (1 << 16) + sizeof(aclshmem_init_optional_attr_t),
        ACLSHMEM_DATA_OP_MTE,
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT};
    void *comm_args = nullptr;
    uint64_t instance_id = 0;
} aclshmemx_init_attr_t;
```

### 6.2 Internal Type Requirements

Internal types must not become objects that users need to declare, pass in, store, or interpret.

Not allowed:

```c
typedef struct aclshmemi_context aclshmem_context_t;
```

When it is truly necessary to expose state or handles, encapsulate them as public opaque types. The form is as follows; this snippet is a naming template, not an existing interface in the repository:

```c
typedef struct <public_context> *<public_context_t>;
```

And clarify the following in the documentation:

- Creation interface.
- Destruction interface.
- Whether it can be copied.
- Whether it can be used across threads.
- Whether it can be used across PEs.
- Whether it becomes invalid after finalize.
- Whether the handle remains valid after a failure.

Users should not depend on the internal layout of public handles. The creation, destruction, copying, concurrent access, and invalidation rules of handles must be described by the public API documentation.

### 6.3 Legacy Compatibility Strategy

The rules in this section are mandatory constraints on new public APIs, public types, and public macros. Existing public symbols that do not comply with the rules must not have their scope of use expanded.

Historical compatibility symbols that truly need to be retained should meet the following conditions:

- Not used in new code.
- Not recommended in samples or new documentation.
- Have a clear replacement interface.
- Have migration instructions.
- Have a removal plan or compatibility notes.

If historical internal-prefix symbols have completed remediation, for example migrated from `aclshmemi_*` to `aclshmemx_*`, the old internal-prefix aliases should no longer be kept in public headers, samples, or user documentation.

---

## 7. Host API and Device API

### 7.1 Host API

The Host API refers to interfaces called by the Host CPU, responsible for initialization, configuration, resource management, synchronization, and task dispatch.

The Host API should meet the following requirements:

- Use `ACLSHMEM_HOST_API` or the unified export macro of the project.
- Use `aclshmem_*` or `aclshmemx_*` for naming.
- State clearly in comments that the execution domain is Host.
- State clearly whether it is thread-safe.
- State clearly whether it blocks.
- Document completion conditions for asynchronous submission interfaces.
- Document enqueue success, local completion, and stream synchronization requirements for stream interfaces.
- Document special return values for allocation, deallocation, and query interfaces.
- Perform necessary validity checks on parameters that will enter the Device path to avoid redundant fallback checks in the Device release path.

The Host API does not mandate that all interfaces return `int`. The rules are as follows:

- In principle, management, configuration, submission, and new feature interfaces return `int` or `aclshmem_error_code_t`.
- Memory allocation interfaces may return pointers, but the semantics of returning `NULL` on failure must be explicit.
- Deallocation interfaces may return `void`, but null pointer behavior and call preconditions must be explicit.
- Query interfaces may return results through return values or output parameters, but failure handling must be explicit.
- Public APIs do not throw exceptions.
- Public APIs do not return undocumented special values.

Example:

```c
ACLSHMEM_HOST_API int aclshmemx_init_attr(
    aclshmemx_bootstrap_t bootstrap_flags,
    aclshmemx_init_attr_t *attributes);
ACLSHMEM_HOST_API void *aclshmem_malloc(size_t size);
ACLSHMEM_HOST_API int aclshmem_finalize(void);
ACLSHMEM_HOST_API void aclshmem_free(void *ptr);
```

### 7.2 Device API

The Device API refers to interfaces called in the device execution context, responsible for data movement, remote memory access, atomics, and synchronization.

The Device API should meet the following requirements:

- Use `ACLSHMEM_DEVICE` or the unified device marker of the project.
- State clearly in comments the calling domain, for example device-only, block, warp, vector, AIV, AIC, or host/device, all of which are acceptable.
- Do not introduce unnecessary runtime branches in the release path.
- Do not depend on undocumented combinations of compilation macros.
- Document memory visibility, synchronization requirements, and thread domain restrictions clearly.
- Document dependencies on transport engines, chip models, and CANN versions clearly.
- Document the lifetime and reuse rules for resources such as the UB workspace, sync_id, event_id, and notify addresses clearly.
- Do not perform unnecessary fallback checks in the Device release path.
- Defensive `if-else` checks on the Device side should be placed in debug mode.

Device API comments should make the following explicit:

- Supported data types.
- Pointer address space requirements, for example global memory, UB, and LocalTensor.
- Symmetric address requirements.
- Valid range of the PE parameter.
- Whether it returns synchronously.
- Whether `quiet`, `fence`, `barrier`, `sync`, or engine-specific quiet is required.
- Concurrency restrictions, for example whether the same PE, the same remote address, or the same sync_id supports concurrency.
- Behavior when unsupported types, platforms, or thread domains are used.
- Which errors should be checked on the Host side, and which debug checks are enabled only in Device debug mode.

### 7.3 Device-Side Release/Debug Check Boundary

Device-side code should distinguish three types of branches.

| Type | Allowed in the release path | Description |
|---|---:|---|
| Business semantic branches | Allowed | For example, selecting different implementations by thread domain, data type, or transport engine |
| Compile-time capability dispatch | Allowed | For example, `#if`, `if constexpr`, and template specialization |
| Defensive parameter checks | Not allowed by default | Should be placed in debug mode or checked on the Host side |
| State fallback checks | Not allowed by default | Should be handled on the Host side, during initialization, or during submission |
| Slow path fallback | Not allowed by default | Unless proven to be necessary semantics with a performance evaluation |
| Platform compatibility fallback | Not allowed by default | Should be handled by Host-side capability queries or during initialization |

Not recommended in the Device release path:

```c
if (ptr == NULL) {
    return;
}

if (!is_supported_engine(engine)) {
    fallback_to_other_engine(...);
}

if (!is_aligned(ptr, 32)) {
    slow_path(...);
}
```

The Device debug path should cover defensive conditions such as null pointers, alignment, and PE validity through the unified check mechanism already published by the project. The external specification does not preset specific debug macro names.

Requirements for Device-side check macros:

- If the macro is located in a public header, use the `ACLSHMEM_*` or `ACLSHMEMX_*` prefix.
- If the macro is located only in internal implementation files, use the `ACLSHMEMI_*` prefix.
- Debug checks must be empty implementations or fully optimized away by the compiler in release builds.
- Debug checks must not change release behavior.
- Debug checks must not introduce error return semantics that users could depend on.

---

## 8. API Comments

Public APIs use Doxygen-style comments. When you add or modify a public API, the comment must contain at least the following items:

- `@brief`: A one-sentence description of the interface behavior.
- Parameter descriptions: Mark each parameter as `[in]`, `[out]`, or `[in,out]`.
- Return values: Describe the success value and failure error codes.
- Execution domain: Host, Device, stream, block, warp, vector, AIV, or AIC.
- Blocking semantics: blocking, non-blocking, enqueue, or asynchronous submission.
- Completion semantics: When the operation completes locally, when data becomes visible to the remote side, and whether `quiet`, `barrier`, `sync`, engine-specific quiet, or stream synchronization is required.
- Thread safety: Whether the API can be called concurrently and whether external serialization is required.
- Memory requirements: Whether symmetric memory is required, the address space, alignment, size unit, and lifecycle.
- Device check boundary: Which validity conditions the Host side guarantees, and which debug checks are enabled only in debug mode.
- Failure behavior: Whether output parameters, handles, stream tasks, and memory states remain valid after a failure.

Avoid restating the literal meaning of the code in comments. For example, do not comment `count` as "the count". Describe the unit, range, and behavioral impact.

Recommended template:

```c
/**
 * @brief Submit a non-blocking put operation to a remote PE.
 *
 * @param[in] dst Destination symmetric address on the remote PE. Must not be NULL.
 * @param[in] src Source buffer on the local PE. Must not be NULL.
 * @param[in] size Number of bytes to transfer.
 * @param[in] pe Remote PE number. Must be in the valid PE range.
 *
 * @note Address requirements: dst points to symmetric memory on pe. src is
 *       used as supplied and may be any valid local device GM buffer.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting
 *       ACLSHMEM_DATA_OP_ROCE, complete transfer ranges for both operands must
 *       remain within their respective symmetric memory allocations. Runtime
 *       dispatch may select RDMA
 *       for the target PE; callers need not determine the per-operation engine.
 * @note Execution domain: Host.
 * @note Blocking semantics: non-blocking. Returning from this void API only
 *       means the submission path has returned; it is not a completion signal.
 * @note Completion semantics: the caller must call aclshmem_quiet(),
 *       aclshmem_barrier(), aclshmem_sync(), an engine-specific quiet API,
 *       or synchronize the associated stream as documented before reusing
 *       resources or assuming remote visibility.
 * @note Failure behavior: this existing API does not return a status code.
 *       Implementations must document diagnostics and error propagation
 *       through the established project mechanism.
 * @note Thread safety: thread-safe after successful initialization unless
 *       otherwise documented.
 */
ACLSHMEM_HOST_API void aclshmem_putmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe);
```

Device API comment example:

```c
/**
 * @brief Put data from the current device execution context to a remote PE.
 *
 * @param[in] dst Destination symmetric address on the remote PE.
 * @param[in] src Source address in device global memory.
 * @param[in] size Number of bytes to transfer.
 * @param[in] pe Remote PE number.
 *
 * @note Address requirements: dst points to symmetric memory on pe. src is
 *       used as supplied and may be any valid address in device global memory.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting
 *       ACLSHMEM_DATA_OP_ROCE, complete transfer ranges for both operands must
 *       remain within their respective symmetric memory allocations. Runtime
 *       dispatch may select RDMA
 *       for the target PE; callers need not determine the per-operation engine.
 * @note Execution domain: Device.
 * @note The caller must ensure that dst and src are non-NULL, properly aligned,
 *       and that pe is valid. These conditions may be checked only in debug
 *       builds. Release device code does not perform defensive fallback checks.
 * @note Completion semantics: the caller must use the documented quiet or sync
 *       operation before assuming remote visibility.
 */
/* device-side public API declaration */
```

---

## 9. Error Handling and Logging

### 9.1 Return Codes

- Return `0` or `ACLSHMEM_SUCCESS` on success.
- Return a stable negative error code on failure, or return the failure value explicitly defined in the documentation.
- Public APIs do not throw exceptions.
- Public APIs do not return undocumented special values.
- Do not leak underlying dependency error codes directly as public error codes, unless the error code is already part of the public contract.

Error codes should be layered by semantics:

- Parameter errors: NULL pointers, out-of-range values, invalid enum values, invalid sizes, and invalid alignments.
- State errors: Uninitialized state, repeated initialization, calls after finalize, and context mismatch.
- Resource errors: Insufficient memory, invalid handles, insufficient sync_ids, insufficient events, and unavailable underlying dependencies.
- Runtime errors: Timeout, unreachable remote side, device execution failure, and stream submission failure.
- Capability errors: The operation is not supported on the current platform, chip, CANN version, transport engine, or backend.
- Internal errors: Broken invariants or states that should not occur. Log them and return a stable error code.

### 9.2 Caller-Side Handling

After you call a public API that returns a status value, you should check the return value. On a failure path, you should exit promptly or enter the unified error handling logic. Do not continue to use the following items:

- Invalid pointers.
- Invalid handles.
- Invalid teams.
- Invalid streams.
- Contexts whose initialization is not complete.
- `NULL` values returned by failed allocations.
- Asynchronous tasks whose submission failed but whose existence is still assumed.

Example:

```c
int ret = aclshmemx_set_sdma_config(offset, ub_size, sync_id);
if (ret != ACLSHMEM_SUCCESS) {
    SHM_LOG_ERROR("aclshmemx_set_sdma_config failed, ret=" << ret);
    return ret;
}

void *buf = aclshmem_malloc(size);
if (buf == NULL) {
    SHM_LOG_ERROR("aclshmem_malloc failed, size=" << size);
    aclshmem_finalize();
    return ACLSHMEM_MALLOC_FAILED;
}
```

### 9.3 Logging

- External logs should use the unified project logger or a logger callback.
- Error logs should contain the API name, the error code, and key context.
- Logs must not leak sensitive information, keys, user data content, full memory dumps, or unmasked network addresses.
- Avoid repeated printing on the same failure path to prevent log noise from masking the root cause.
- By default, public APIs should not output a large number of INFO logs.
- Logs in performance paths must be disableable or controlled by log level.
- Device-side debug output must not enter the default release path.
- Device-side release paths must not replace error handling with logs.
- Error codes in logs must be consistent with the return codes to avoid misleading troubleshooting.

Recommended format:

```c
SHM_LOG_ERROR("aclshmem_team_split_strided failed, ret=" << ret
              << ", start=" << start
              << ", stride=" << stride
              << ", size=" << size);
```

---

## 10. Asynchronous and Synchronous Semantics

Interfaces with `nbi`, `on_stream`, or other asynchronous semantics must clearly specify the following:

- Whether a successful return only means that the submission succeeded.
- Whether a successful return means that the task has been enqueued.
- When the data can be reused locally.
- When the remote data becomes visible.
- Which completion interface users need to call.
- Whether Host-initiated and Device-initiated operations belong to the same completion domain.
- The relationships among stream synchronization, device synchronization, SHMEM quiet, barrier, and sync.

### 10.1 States That Must Be Distinguished

| State | Meaning |
|---|---|
| Submission succeeded | The API parameter check passed, and the task has been submitted to the runtime, an internal queue, or the transport engine |
| Enqueue succeeded | The task has entered the stream, SQ, WQ, or engine queue |
| Local completion | The local buffer can be reused, or the local write has completed |
| Remote visibility | The remote PE can observe the data update |
| Communication completion | The communication operation initiated by the current execution domain has completed |
| Global synchronization completion | The specified team or all PEs have reached the synchronization point |

### 10.2 Completion Interfaces

An asynchronous interface must state which completion interface is required, for example:

- `aclshmem_quiet`
- `aclshmem_fence`
- `aclshmem_barrier`
- `aclshmem_sync`
- The stream synchronization interface
- The device synchronization interface
- Engine-specific quiet, for example the quiet or wait interfaces corresponding to MTE, SDMA, RDMA, and UDMA

### 10.3 Sample Code Requirements

In sample code, if you use an asynchronous interface, you must explicitly show the completion operation, for example:

```c
aclshmem_putmem_nbi(dst, src, size, pe);

/* Required before reusing src or assuming remote visibility. */
aclshmem_quiet();
```

The stream sample must show stream synchronization or explicitly delegate synchronization to the caller:

```c
aclshmemx_putmem_on_stream(dst, src, size, pe, stream);

aclError acl_ret = aclrtSynchronizeStream(stream);
if (acl_ret != ACL_SUCCESS) {
    return ACLSHMEM_INNER_ERROR;
}
```

---

## 11. Memory and Alignment

The SHMEM communication path is sensitive to memory layout, symmetry, cache line, page, UB workspace, and hardware DMA alignment. When adding public structures, shared state, communication buffers, symmetric heap allocation, and memory pool splitting logic, alignment requirements should be described separately based on the allocation source and usage scenario.

Do not write only generalized descriptions such as "aligned to the cache line" or "aligned as required by the hardware".

### 11.1 Symmetric Heap Allocation

- Use "symmetric memory" uniformly for user/API semantics; use "symmetric heap" only when describing the underlying reserved heap, `heap_base`, `heap_size`,
  or the allocator implementation.
- Symmetric memory that is communicable by users should be obtained through public memory APIs, for example `aclshmem_malloc`, `aclshmem_calloc`, `aclshmem_align`, `aclshmemx_malloc`, `aclshmemx_calloc`, or `aclshmemx_align`.
- Example code and user documentation must not require users to depend on the internal heap implementation.
- When using `aclshmem_align(alignment, size)`, `alignment` must be a power of 2.
- The caller must handle the case where allocation fails and returns `NULL`.
- The semantics of `size == 0`, allocation failure, and returning `NULL` must be reflected in interface comments or the handling logic on the caller side.
- Symmetric memory on each PE must maintain consistent allocation relationships. All PEs should allocate in the same order, with the same size, or in a manner explicitly allowed by the documentation.
- Do not pass Host addresses from ordinary `malloc/new` to RMA, AMO, signal, or synchronization APIs as SHMEM symmetric addresses.
- Do not expose internal allocator layout, chunk header, free list, page table, or remote address translation details in the public ABI.

### 11.2 Initialization Attributes and Symmetric Memory Size

When `local_mem_size`, heap size, page size, or internal mapping size is involved:

- The unit must be explicitly stated as byte.
- Whether consistency across all PEs is required must be clearly stated.
- Whether alignment is rounded up to the page must be clearly stated.
- Bare numbers must not be scattered in business code.
- A unified macro or helper should be used for alignment, for example `ALIGN_TO(size, page)` or the unified wrapper of the project.
- Alignment failure, integer overflow, and size exceeding the upper limit must have error handling.
- Fields representing memory sizes in public structures should use `size_t`, `uint64_t`, or the unified type of the project, and must not use `int` of ambiguous width.

### 11.3 Secondary Allocation of AscendCL Device/Host Memory

When a large memory block is obtained through `aclrtMalloc`, `aclrtMallocHost`, or other AscendCL memory allocation APIs and split by the project itself, each sub-segment must satisfy the size and starting address alignment requirements of the corresponding AscendCL API.

Secondary allocation of Device memory should comply with the following:

- The sub-segment size is aligned up to 32 bytes with an additional 32 bytes reserved, that is, a pattern like `ALIGN_UP(len, 32) + 32`.
- The starting address of the sub-segment satisfies 64-byte alignment.
- The splitting logic for large memory blocks must be centrally encapsulated; hand-written `ALIGN_UP` of different versions is not allowed in business code.
- Do not assume that a Host address returned by ordinary `malloc/new` can directly satisfy the requirements of the Device DMA, RMA, or asynchronous copy paths.

Alignment and overflow check requirements:

- `alignment` must be a power of 2; otherwise, it is regarded as a parameter error.
- Alignment computation (align_up) must check the validity of `alignment` and return an error or 0 when it is invalid.
- Pointer alignment checks must verify both the validity of `alignment` and address alignment.
- Integer overflow must be checked when size addition or multiplication is involved.
- Helpers for alignment and overflow checks are part of the internal implementation and must not be placed in `include/**`; the unified wrapper of the project should be used, and hand-written versions of different implementations are not allowed in business code.

### 11.4 Ascend C LocalTensor, UB, and Data Movement

Code related to Ascend C LocalTensor, UB workspace, and data movement should clarify the following requirements:

- Buffers related to LocalTensor/UB are designed and checked at a 32-byte granularity by default.
- If a specific API requires 512B, 64B, 128B, or another granularity, it must be clearly stated in the API comments, the config structure, and the examples.
- The UB workspace field must describe:
  - Address space.
  - Unit of size.
  - Minimum capacity.
  - Alignment requirements.
  - Lifecycle.
  - Whether it can be reused across calls.
  - Whether it can be used concurrently by multiple cores.
- Synchronization resources such as `sync_id`, `event_id`, notify addresses, and flag addresses must have their allocation source and reuse rules described.
- Alignment requirements may differ among different TPosition values, different transport engines, and different chip models; the 32B or 512B constraint of one API must not be generalized into the sole rule for the entire repository.

### 11.5 cache line, L2 cache line, and page

- When scalar cache line alignment is required, use the unified constants of the project and do not hard-code `64`.
- When L2 cache line alignment is required, use the unified constants of the project and do not hard-code `512`.
- When SHMEM page alignment is required, use the unified constants of the project and do not hard-code `2 * 1024 * 1024`.
- Shared state, synchronization flags, signal, doorbell, notify addresses, and metadata structures accessed across PEs must avoid false sharing.
- Structures with fixed layouts must be accompanied by `static_assert(sizeof(...))`, field offset checks, or ABI tests.
- The public ABI must not depend on undocumented compiler padding, temporary structure layouts, or internal allocator implementations.

Example:

```c
static_assert(SCALAR_DATA_CACHELINE_SIZE == 64, "Unexpected scalar cache line size");
static_assert(L2_CACHELINE_SIZE == 512, "Unexpected L2 cache line size");
```

### 11.6 Alignment Checks

Public Host APIs involving alignment should check the following at the entry point:

- Whether alignment is a power of 2.
- Whether the pointer satisfies the alignment.
- Whether the size satisfies the minimum granularity.
- Whether the size alignment computation overflows.
- Whether the symmetric address comes from a valid symmetric heap.
- Whether the UB workspace satisfies the size and address space requirements.

If a check fails, a parameter error should be returned, and submission of communication operations must not continue.

The Device API release path should not repeat the above non-essential fallback checks. The Device side keeps the corresponding `DCHECK` or assertions only in debug mode.

---

## 12. msSanitizer Constraints

### 12.1 Check Scope

When new code involves the following paths, it must be evaluated whether msSanitizer needs to be used for memory detection or race detection:

- SHMEM RMA paths.
- SHMEM AMO paths.
- signal / wait / sync paths.
- Device kernel access paths.
- Transport paths such as DMA / SDMA / RDMA / UDMA / MTE within the tool support scope.
- Cross-PE remote access paths.
- Internal asynchronous copy paths of the runtime.

When new transport paths such as DMA / SDMA / RDMA / UDMA / MTE or related interfaces are added later, the msSanitizer support scope check must be proactively initiated to confirm whether the tool covers the path, and the check conclusion, verification commands, and results must be recorded in the PR. If the tool does not support the path yet, the reason should be stated and equivalent verification evidence or a follow-up verification plan should be provided.

### 12.2 Usage

The msSanitizer verification of SHMEM is completed through compile-time instrumentation and runtime tool launching.

New code should follow the usage described below:

- Use `bash scripts/build.sh -mssanitizer` or combined sample parameters, for example `bash scripts/build.sh -examples -mssanitizer`.
- When `USE_MSSANITIZER=ON`, CMake sets the sanitizer compilation options according to `SOC_TYPE` and the bisheng version: for non-Ascend950, `-g --cce-enable-sanitizer` is always used; for Ascend950, the same option is enabled under newer CANN versions, while older versions add only `-g` (in that case, memory detection related to the AscendC API is unavailable, see the cmake configuration log for details).
- At runtime, the target program must be launched using mssanitizer, for example:

```sh
mssanitizer -- application parameter1 parameter2 ...
```

For sample verification, refer to the allgather approach:

```sh
bash scripts/build.sh -examples -mssanitizer
cd examples/allgather
mssanitizer -- bash run.sh -pes 2
```

### 12.3 PR Requirements

When the following changes are involved, the PR description should state the msSanitizer verification results:

- Adding an allocator.
- Modifying the symmetric heap.
- Modifying Host buffer management.
- Modifying Device buffer management.
- Modifying the communication buffer lifecycle.
- Modifying the RMA, AMO, signal, sync, and DMA paths.
- Modifying the logic for non-`aclrt` addresses entering the Device or communication paths.

Before submitting a PR, local tests should be launched with msSanitizer. Paths that cannot be covered by msSanitizer should have the reason stated in the PR, along with equivalent verification evidence or a follow-up verification plan.

---

## 13. Sample Code

Sample code is the first entry point for users to learn APIs. It should follow stricter external boundaries than ordinary tests.

Sample code must meet the following requirements:

- Use only the `aclshmem_*` and `aclshmemx_*` public APIs.
- Do not include internal header files.
- Do not call internal functions.
- Do not use `aclshmemi_*` or `ACLSHMEMI_*`.
- Do not rely on undocumented environment variables.
- Environment variable examples use `SHMEM_*`.
- Show the complete initialization process.
- Show return value checks.
- Show communication completion operations.
- Show the resource release process.
- Asynchronous samples must include completion operations.
- Device samples must describe the execution domain, the launch method, and Host-side parameter checks.
- Multi-PE samples must describe the run command, the number of PEs, the expected output, and the entry point for troubleshooting common failures.

Recommended sample structure:

```c
int main(int argc, char **argv)
{
    /* Build attributes with aclshmemx_set_attr_uniqueid_args or project helper first. */
    int ret = aclshmemx_init_attr(bootstrap_flags, &attributes);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    int my_pe = aclshmem_my_pe();
    int npes = aclshmem_n_pes();

    void *buf = aclshmem_malloc(kSize);
    if (buf == NULL) {
        aclshmem_finalize();
        return ACLSHMEM_MALLOC_FAILED;
    }

    if (my_pe == 0 && npes > 1) {
        aclshmem_putmem_nbi(remote_dst, buf, kSize, 1);
        aclshmem_quiet();
    }

    aclshmem_barrier_all();
    aclshmem_free(buf);
    aclshmem_finalize();
    return 0;
}
```

In Device samples, check parameter validity on the Host side:

```c
if (dst == NULL || src == NULL || size == 0) {
    return ACLSHMEM_INVALID_PARAM;
}

if (!is_aligned(dst, 32) || !is_aligned(src, 32)) {
    return ACLSHMEM_INVALID_PARAM;
}

/* Launch kernel after validation. */
```

It is not recommended to repeat fallback checks in Device kernels:

```c
if (dst == NULL || src == NULL) {
    return;
}
```

---

## 14. Testing and Verification

New code should include at least the verification that matches its risks:

- Naming or header file boundary changes: add static checks or script checks.
- Public API behavior changes: add unit tests or system tests.
- Asynchronous semantics changes: cover submission success, enqueue success, completion, remote visibility, and exception paths.
- Device API changes: cover different thread domains, compilation macro combinations, and chip capability combinations.
- Device release/debug check changes: confirm that the release path does not introduce unnecessary `if-else` fallback checks.
- Memory allocation or alignment changes: cover size 0, invalid alignment, unaligned addresses, overflows, allocation failures, and release paths.
- msSanitizer-related changes: cover verification scenarios where non-`aclrt` addresses enter communication or Device access paths.
- ABI-related changes: check the structure size, field order, field offsets, and initialization compatibility.
- Sample changes: ensure that samples compile, run, and produce verifiable output.
- Transfer engine changes: cover the corresponding paths of MTE, SDMA, RDMA, UDMA, and so on.
- Performance path changes: in addition to correctness tests, retain performance baseline comparisons.

Test code can use internal helpers, but public samples and user documents must not use internal symbols.

Tests for Device-side performance paths should confirm at least the following:

- Debug checks are eliminated in release builds.
- The release path contains no new unnecessary defensive branches.
- Invalid parameters can be captured by debug checks in debug builds.
- Host-side parameter checks cover the preconditions on which the Device release path depends.

---

## 15. Pre-release Checklist

Before submitting public APIs, public header files, Device-side performance paths, or samples, check at least the following items:

- New or modified `include/**` files do not introduce `aclshmemi_*` interfaces or data structures.
- New or modified `include/**` files do not introduce `ACLSHMEMI_*` public macros.
- New or modified `examples/**` files and external documents contain no recommended usage of internal prefixes.
- New public APIs use `aclshmem_*` or `aclshmemx_*`.
- New public types use `aclshmem_*_t` or `aclshmemx_*_t`.
- Enumerated values use `ACLSHMEM_*` or `ACLSHMEMX_*`.
- New public macros use `ACLSHMEM_*` or `ACLSHMEMX_*`.
- New internal macros use `ACLSHMEMI_*` and must not appear in public header files, samples, or user documents.
- Environment variables use `SHMEM_*`, not `ACLSHMEM_*`.
- Public header files and public include header files under examples contain no global `using namespace`.
- New header files use include guards (`#ifndef`/`#define`/`#endif`), and their names correspond to the file paths.
- New source files contain a copyright header with the correct year and attribution.
- Sample code includes only public header files and calls only public APIs.
- Comments and samples of asynchronous interfaces clearly specify the completion operations.
- The calling scopes of Host APIs and Device APIs are clearly stated.
- The Device release path contains no unnecessary fallback `if-else` checks.
- Device defensive checks are placed in debug mode.
- Users can reliably depend on the semantics of return values and error codes.
- The version, lifecycle, field units, and ABI impact of public structures are documented.
- Memory alignment requirements specify the allocation source, the alignment granularity, and failure handling.
- New logic involving non-`aclrt` addresses entering communication or Device access paths has passed msSanitizer verification.
- Related tests, static checks, or equivalent verification has been performed.

### 15.1 Check Commands for Incremental Commits

The commands in this section check the code that is newly added or modified in this change. Historical code may contain compatibility aliases, internal samples, or legacy usages being migrated. Do not directly equate repository-wide hits with violations in new code. For a normal PR or patch, check the staged diff or review diff first.

```bash
# New public areas must not introduce recommended usages of aclshmemi_* or ACLSHMEMI_*.
git diff --cached -U0 -- include examples docs \
  | rg '^\+.*\baclshmemi_|\+.*\bACLSHMEMI_'

# New public APIs should avoid camel case naming. Manual judgment based on the context is required.
git diff --cached -U0 -- include \
  | rg '^\+.*\b[A-Za-z0-9]+[A-Z][A-Za-z0-9_]*\s*\('

# Samples and external documents must not add includes of internal paths.
git diff --cached -U0 -- examples docs \
  | rg '^\+.*#\s*include\s*[<"].*(internal|detail|private|src)/'

# New environment variables must not use ACLSHMEM_*.
git diff --cached -U0 -- src examples docs \
  | rg '^\+.*(getenv\("ACLSHMEM_|ACLSHMEM_[A-Z0-9_]+=)'

# Public header files and public include header files under examples must not add global using namespace.
git diff --cached -U0 -- include 'examples/**/include' \
  | rg '^\+.*\busing\s+namespace\b'

# Asynchronous samples require manual review of completion operations.
git diff --cached -U0 -- examples docs \
  | rg '^\+.*(nbi|on_stream|quiet|barrier|sync|Synchronize)'
```

### 15.2 Auxiliary Commands for the Full Baseline

The commands in this section are only used to evaluate the historical baseline of the repository and the cleanup scope. If the current baseline still contains historical compatibility symbols or hits in legacy samples, create a cleanup plan or an allowlist first. Do not directly use repository-wide hits to reject normal new code.

```bash
rg -n "\baclshmemi_" include examples docs \
  --glob '!**/coding_style_guide.md'

rg -n "\bACLSHMEMI_" include examples docs \
  --glob '!**/coding_style_guide.md'

rg -n "\busing\s+namespace\b" include 'examples/**/include'
```

### 15.3 Device-Side Check Commands

For Device-side release/debug checks, you can use auxiliary commands to discover new branches, and manual review is required:

```bash
# Auxiliary discovery of new if branches on the Device side. Manual judgment is required on whether they are business semantic branches.
git diff --cached -U0 -- include/device src \
  | rg '^\+.*\bif\s*\('

# Auxiliary discovery of newly added fallback semantics.
git diff --cached -U0 -- include/device src \
  | rg '^\+.*\b(fallback|slow_path|return;|return\s+0;)\b'
```

Check requirements:

- If a new `if` on the Device side is a parameter check, a state check, an alignment check, or a capability fallback, put it into a debug check or migrate it to the Host side.
- If a new `if` on the Device side is a business semantic branch, explain its necessity in the code or the PR.
- For a new fallback on the Device side, you must state whether it is a required path for functional semantics and provide a performance impact assessment.

### 15.4 Automated Check Prompt

You can provide the following content together with this file to automated check tools as check requirements:

```text
Check this change according to docs/coding_style_guide.md, and focus on the following items first:
1. Whether new or modified include/** introduces aclshmemi_* interfaces, data structures, or ACLSHMEMI_* public macros.
2. Whether public APIs use only the aclshmem_* or aclshmemx_* prefixes.
3. Whether public macros use ACLSHMEM_* or ACLSHMEMX_* and internal macros use ACLSHMEMI_*.
4. Whether new or modified examples/** uses only public APIs, and whether it contains internal header files or internal symbols.
5. Whether environment variables use SHMEM_* and whether ACLSHMEM_* is incorrectly used.
6. Whether public header files contain using namespace.
7. Whether the Device release path adds unnecessary if-else fallback checks, and whether defensive checks are in debug mode.
8. Whether memory paths describe alignment requirements, and whether scenarios where non-aclrt addresses enter communication or Device paths are included in msSanitizer verification.
9. Whether asynchronous interfaces and samples specify completion operations such as quiet, barrier, sync, stream synchronize, or engine-specific quiet.
10. Whether public structures, public types, error codes, and ABI impacts have comments and compatibility descriptions.
```

---

## 16. Compatibility Policy

Historical compatibility aliases can be retained when necessary, but new capabilities should be released first with formal public prefixes. When you must deprecate an old interface, provide migration instructions, alternative interfaces, and a sufficient transition period.

Once a public API is released, do not rename it arbitrarily, change the meanings of its parameters, or change the semantics of its return codes. If you must break compatibility, clearly inform users through version descriptions and release notes.

Compatibility handling principles:

- New code must not continue to use deprecated interfaces.
- New samples must not demonstrate deprecated interfaces.
- New documents must not recommend deprecated interfaces.
- Before deleting deprecated symbols, confirm the impact scope and provide migration instructions.
- Source compatibility breakages, ABI breakages, and behavioral semantic changes should be described separately.
- Changes to user-visible synchronous completion semantics must be treated as compatibility risks.
- Modifications to the field order, size, alignment, and initialization method of public structures must undergo ABI impact assessment.
- Behavior changes in the Device release path must undergo performance and compatibility impact assessment.
- When migrating Device-side fallback checks to the Host side, you must confirm that the Host side covers all user entry points.

If historical aliases with internal prefixes have already been converted to `aclshmemx_*`, `ACLSHMEM_*`, or `ACLSHMEMX_*`, do not keep any `aclshmemi_*` or `ACLSHMEMI_*` compatibility entries. This requirement applies to public header files, sample code, and user documents.

---

## 17. References

- Repository root directory: [`SHMEM`](https://gitcode.com/cann/shmem)
- Contribution guide: [`CONTRIBUTING.md`](https://gitcode.com/cann/shmem/blob/master/CONTRIBUTING.md)
- License: `LICENSE` (CANN Open Software License Agreement Version 2.0)
- Format configuration: `.clang-format` in the repository root directory (`BasedOnStyle: Google`, 120 columns, 4 spaces, pointers left-aligned)
- Benchmark reference: NVSHMEM API naming conventions (`nvshmem_*` / `nvshmemx_*` / `NVSHMEM_*`, with calling scopes distinguished by host-only / gpu-only / host+gpu)

> This guide evolves with each version. Submit revisions through Issues or PRs. New rules should be accompanied by verifiable check commands or review steps.
