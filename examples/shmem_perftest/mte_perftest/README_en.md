# mte_perftest

## Sample Overview

`mte_perftest` is a parameterized test sample for **testing the performance of the shmem MTE (Memory Transfer Engine)**. It supports multiple test types, multiple data types, and flexible parameter configuration. The sample collects performance data through the [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) macros and tests the MTE transfer bandwidth under different core counts and data volumes, helping you evaluate MTE data transfer performance. **The script test results are for reference only; actual performance depends on the actual scenario**

## Test Objectives

This sample is mainly used to test the performance of the following MTE data transfer operations:

1. **Unidirectional put**: Only PE0 performs the put operation and transfers data to the remote PE
2. **Bidirectional put**: Both PEs perform the put operation at the same time and transfer data to each other
3. **Unidirectional get**: Only PE0 performs the get operation and reads data from the remote PE
4. **Bidirectional get**: Both PEs perform the get operation at the same time and read data from each other

Based on the test results, you can understand the performance of MTE interfaces in different scenarios.

## Features

### Supported Test Types

- **put**: Unidirectional put. Only the PE specified by the environment variable `SHMEM_CYCLE_PROF_PE` performs data movement, and PE0 is used by default when it is not specified
- **bi_put**: Bidirectional put. Both PEs perform data movement
- **get**: Unidirectional get. Only the PE specified by the environment variable `SHMEM_CYCLE_PROF_PE` performs data movement, and PE0 is used by default when it is not specified
- **bi_get**: Bidirectional get. Both PEs perform data movement
- **all**: Runs all test types

### Supported Data Types

- **Floating-point**: `float`
- **Signed integer**: `int8`, `int16`, `int32`, `int64`
- **Unsigned integer**: `uint8`, `uint16`, `uint32`, `uint64`
- **Character**: `char`
- **all**: Tests all data types

### Other Features

- ✅ Configurable core count range (32 cores by default)
- ✅ Configurable data volume range (powers of 2 are supported)
- ✅ Loop count can be set to obtain stable performance data
- ✅ Uses SHMEMI_PROF to collect performance data of each core
- ✅ Generates performance reports in CSV format
- ✅ Supports batch testing (the "all" option)

## Compilation

Compile the sample in the shmem root directory:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# A2/A3 platforms
bash scripts/build.sh -examples
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples

# To run --memory-type dram, compile with CANN mode enabled
# A3 platform (A2 does not support the DRAM test)
bash scripts/build.sh -examples -cann
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples -cann
```

## Usage

### Basic Usage

```bash
# Enter the sample directory
cd examples/shmem_perftest/mte_perftest/

# Run the test script
./run.sh [options]
```

### Command-line Options

| Option | Shorthand | Description | Default |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | Test type (put/bi_put/get/bi_get/all) | put |
| `--datatype <type>` | `-d <type>` | Data type (float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char/all) | float |
| `--block-size <size>` | `-b <size>` | Sets a single core count | 32 |
| `--block-range <min> <max>` | - | Sets the core count range | 32-32 |
| `--block-list <b1,b2,...>` | - | Sets a core count list | - |
| `--exponent <exponent>` | `-e <exponent>` | Sets the exponent of the data volume (2^exponent) | - |
| `--exponent-range <min> <max>` | - | Sets the exponent range of the data volume | 3-17 |
| `--loop-count <count>` | - | Sets the loop count | 1000 |
| `--memory-type <hbm\|dram>` | - | Sets the SHMEM memory type. dram uses HOST_SIDE memory, supports A3 and Ascend950, and does not support A2 | hbm |
| `-pes <size>` | - | Sets the number of PEs, not yet modifiable | 2 |
| `-ipport <ip:port>` | - | Sets the communication address | tcp://127.0.0.1:8764 |
| `-gnpus <num>` | - | Sets the number of NPUs, not yet modifiable | 2 |
| `-fnpu <id>` | - | Sets the first NPU ID, controlling the starting NPU | 0 |
| `-fpe <id>` | - | Sets the first PE ID, not yet modifiable | 0 |

### DRAM Memory Test Constraints

`--memory-type dram` uses `aclshmemx_malloc(..., HOST_SIDE)` to allocate DRAM memory on the host side. The test involves remote host machine memory access, supports A3 and Ascend950, and **does not support A2**. For A2, use the default `--memory-type hbm`.

The DRAM test also requires the CANN mode. At compile time, use:

```bash
# A3 platform (A2 does not support the DRAM test)
bash scripts/build.sh -examples -cann
# Ascend950 platform
bash scripts/build.sh -soc_type Ascend950 -examples -cann
```

The DRAM test requires the runtime environment to support host-side DRAM memory access. For the related hardware and available memory constraints, refer to the "Restrictions" section of [rma_d2h_demo](../../rma_d2h_demo/README_en.md). mte_perftest configures 1GB of local memory by default. When the test parameters require more local memory, the program increases the memory automatically based on the data volume. Before running, ensure that the available DRAM space is greater than the actual local memory configuration.

### Examples

#### 1. Testing a Single Operation Type and Data Type

```bash
# Test the unidirectional PUT operation with the float type, 32 cores, and a data volume from 2^8 to 2^20 bytes
./run.sh -t put -d float --block-size 32 --exponent-range 8 20 --loop-count 1000

# Test the bidirectional GET operation with the int32 type
./run.sh -t bi_get -d int32 --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 2. Testing All Operation Types (Specifying the Data Type)

```bash
# Test all operation types (put/bi_put/get/bi_get) with the float data type
./run.sh -t all -d float --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 3. Testing All Data Types (Specifying the Operation Type)

```bash
# Test the PUT operation with all data types
./run.sh -t put -d all --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 4. Testing All Operation Types and Data Types

```bash
# Test all operation types and all data types
./run.sh -t all -d all --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 5. Testing Specified Core Counts

```bash
# Test all operation types and all data types with the specified core counts (2, 4, 6, 8, and 16 cores).
./run.sh -t all -d all --block-list 2,4,6,8,16 --exponent-range 8 20 --loop-count 1000
```

#### 6. Running with Default Parameters

```bash
# Run the put test with default parameters
./run.sh

# Run all test types
./run.sh -t all
```

#### 7. Visualizing Data

```bash
# Use perf_data_process.py -d to pass in the path where the CSV data is stored. The visualized results are stored in the picture folder under the CSV storage path.
python ../utils/perf_data_process.py -d output
```

## Output Description

### CSV File Format

After the test runs, the `output` folder is generated in the current directory. It contains the CSV files of test results, and the file name format is:

```bash
{test_type}_{data_type}_{prof_pe}.csv
```

For example:

- `put_float_0.csv` - Unidirectional PUT operation with the float type
- `bi_get_int32_0.csv` - Bidirectional GET operation with the int32 type

### CSV File Content

The CSV file contains the following fields:

| Field | Description |
|------|------|
| `DataSize/B` | Data size (bytes) |
| `Npus` | Number of NPUs used |
| `Blocks` | Number of cores |
| `Bandwidth/GBps` | Bandwidth (GB/s) |
| `CoreMaxTime/us` | Maximum core time (microseconds) |
| `SingleCoreTime/us` | Single-core time (microseconds), one column per core |

## Performance Data Collection

This sample uses the `SHMEMI_PROF_START/END` macros to collect performance data:

- **Profiling PE**: Specified by the environment variable `SHMEM_CYCLE_PROF_PE`, 0 by default
- **Maximum recorded core count**: `ACLSHMEM_CYCLE_PROF_MAX_BLOCK` (64 cores)
- **Maximum recorded frame count**: `ACLSHMEM_CYCLE_PROF_FRAME_CNT` (1024 frames)

Only the PE specified by the environment variable outputs the CSV file, and the data of other PEs is not collected.

## Precautions

1. **Environment configuration**: Before running, ensure that the Ascend environment variables are correctly set:

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. **Data volume calculation**: The test data volume is calculated as `2^exponent` bytes

3. **Parameter range**:
   - The core count range beyond `ACLSHMEM_CYCLE_PROF_MAX_BLOCK` (64 cores) is not recorded
   - The frame count cannot exceed `ACLSHMEM_CYCLE_PROF_FRAME_CNT` (1024 frames)

4. **Memory requirement**: Testing a large data volume requires sufficient device memory

5. **Performance stability**: It is recommended that you use a larger loop count (such as 1000) to obtain stable performance data

6. **Difference between unidirectional and bidirectional operations**:
   - Unidirectional operations (put/get): only the PE specified by `SHMEM_CYCLE_PROF_PE` performs data movement
   - Bidirectional operations (bi_put/bi_get): both PEs perform data movement

### Common Problems

1. **Memory allocation failure**

   ```bash
   Error: aclrtMalloc failed
   ```

   Solution: Reduce the data volume range or core count and ensure that the device has sufficient memory

2. **CSV file contains no data**
   Solution: Check whether the `SHMEM_CYCLE_PROF_PE` environment variable is correctly set and ensure that the specified PE executes the test

3. **frame_id exceeds the maximum value**

   ```bash
   Warning: frame_id exceeds the maximum value 1024, stopping the test
   ```

   Solution: Reduce the test range and ensure that the test frame count does not exceed 1024.

   The test frame count is calculated as:

   ```bash
   Test frame count = (max_block_size - min_block_size + 1) × (max_exponent - min_exponent + 1)
   ```

   For example, the test frame count of `--block-range 1 64 --exponent-range 10 30` is:

   ```bash
   (64 - 1 + 1) × (30 - 10 + 1) = 64 × 21 = 1344 > 1024 (exceeds the limit)
   ```
