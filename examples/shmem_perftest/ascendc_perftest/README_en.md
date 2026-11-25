# ascendc_perftest

> **Ascend950 is not yet supported**: Building and running with the Ascend950 platform is currently not yet supported.

## Sample Overview

`ascendc_perftest` is a parameterized test sample for **testing the performance of AscendC::DataCopy and comparing it with the shmem put/get interfaces**. It supports multiple test types, multiple data types, and flexible parameter configuration. The sample tests performance under different core counts and data volumes, helping you evaluate and compare the performance difference between the two data transfer methods and providing a reference for performance optimization. **The script test results are for reference only; actual performance depends on the actual scenario**

## Test Objectives

This sample is mainly used to compare the performance of the following two data transfer methods:

1. **AscendC::DataCopy**: A data transfer interface based on Ascend C
2. **shmem put/get interfaces**: Data transfer interfaces of the shared memory

By comparing the test results, you can understand which data transfer method performs better in different scenarios.

## Features

### Supported Test Types

- **put**: Tests the performance of cross-device PUT operations (transfers data from the local device memory to the remote device memory)
- **get**: Tests the performance of cross-device GET operations (transfers data from the remote device memory to the local device memory)
- **ub2gm_local**: Tests the data transfer performance from the local device UB to GM
- **ub2gm_remote**: Tests the data transfer performance from the remote device UB to GM
- **gm2ub_local**: Tests the data transfer performance from the local device GM to UB
- **gm2ub_remote**: Tests the data transfer performance from the remote device GM to UB
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
- ✅ Device IDs can be specified for cross-device testing
- ✅ Loop count can be set to obtain stable performance data
- ✅ Automatic data correctness verification
- ✅ Generates performance reports in CSV format
- ✅ Supports batch testing (the "all" option)

## Compilation

Compile the sample in the shmem root directory:

```bash
# Method 1: Use the build.sh script to compile all samples
bash scripts/build.sh -examples

# Method 2: Compile manually
mkdir -p build
cd build
cmake -DUSE_EXAMPLES=ON ..
source /usr/local/Ascend/ascend-toolkit/set_env.sh
make -j
```

## Usage

### Basic Usage

```bash
# Enter the sample directory
cd examples/shmem_perftest/ascendc_perftest

# Run the test script
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./run.sh [options]
```

### Command-line Options

| Option | Shorthand | Description | Default |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | Test type (put/get/ub2gm_local/ub2gm_remote/gm2ub_local/gm2ub_remote/all) | put |
| `--datatype <type>` | `-d <type>` | Data type (float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char/all) | float |
| `--block-size <size>` | `-b <size>` | Sets a single core count | 32 |
| `--block-range <min> <max>` | - | Sets a continuous core count range | 32-32 |
| `--block-list <b1,b2,...>` | - | Sets a discrete core count list, for example `2,4,6,8` | - |
| `--exponent <exponent>` | `-e <exponent>` | Sets the exponent of the data volume (2^exponent) | - |
| `--exponent-range <min> <max>` | - | Sets the exponent range of the data volume | 3-20 |
| `--device1 <id>` | - | Sets the ID of device 1 | 1 |
| `--device2 <id>` | - | Sets the ID of device 2 | 2 |
| `--loop-count <count>` | - | Sets the loop count | 1000 |
| `--ub-size <size>` | - | Sets the UB size in KB, 16 by default | 16 |

### Examples

#### 1. Testing a Single Operation Type and Data Type

```bash
# Test the PUT operation with the float type, 32 cores, a data volume from 2^8 to 2^20 bytes, and 100 loops
./run.sh -t put -d float --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000

# Test the GET operation with the int32 type
./run.sh -t get -d int32 --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 2. Testing All Operation Types (Specifying the Data Type)

```bash
# Test all operation types (PUT/GET/UB2GM/GM2UB) with the float data type
./run.sh -t all -d float --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 3. Testing All Data Types (Specifying the Operation Type)

```bash
# Test the PUT operation with all data types
./run.sh -t put -d all --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 4. Testing All Operation Types and Data Types

```bash
# Test all operation types and all data types
./run.sh -t all -d all --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 5. Testing Specified Core Counts

```bash
# Test all operation types and all data types with the specified core counts (2, 4, 6, 8, and 16 cores)
./run.sh -t all -d all --block-list 2,4,6,8,16 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
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
{rank}_{test_type}_{data_type}.csv
```

For example:

- `0_put_float.csv` - PUT operation with the float type
- `0_get_int32.csv` - GET operation with the int32 type
- `0_ub2gm_local_float.csv` - Local UB-to-GM operation with the float type
- `0_gm2ub_remote_int32.csv` - Remote GM-to-UB operation with the int32 type

### CSV File Content

The CSV file contains the following fields:

| Field | Description |
|------|------|
| `DataSize/B` | Data size (bytes) |
| `Npus` | Number of NPUs used |
| `Blocks` | Number of cores |
| `UBsize/KB` | UB size (KB) |
| `Bandwidth/GB/s` | Bandwidth (GB/s) |
| `CoreMaxTime/us` | Maximum core time (microseconds) |
| `SingleCoreTime/us` | Single-core time (microseconds) |

## Data Verification

During the test, only PUT/GET automatically performs data correctness verification:

- **PUT operation**: Verifies whether the source data is completely transferred to the destination address
- **GET operation**: Verifies whether the data read from the remote end is correct

When verification succeeds, the following is displayed:

```bash
[Verification] SUCCESS: All cores' first values transferred correctly!
```

## Precautions

1. **Environment configuration**: Before running, ensure that the Ascend environment variables are correctly set:

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. **Device configuration**: Ensure that the specified device IDs (device1 and device2) are available and support Peer Access

3. **Data volume calculation**: The test data volume is calculated as `2^exponent` bytes

4. **Parameter range**: The core count range cannot be greater than the device maximum

5. **Memory requirement**: Testing a large data volume requires sufficient device memory

6. **Data type support**: Some data types (such as uint64 and char) may have restrictions in specific scenarios

7. **Performance stability**: It is recommended that you use a larger loop count (such as 1000) to obtain stable performance data

## Performance Analysis Suggestions

1. Compare the performance differences between test types (put/get/ub2gm_local/ub2gm_remote/gm2ub_local/gm2ub_remote)
2. Compare the performance of different data types
3. Adjust the core count parameters based on the actual hardware configuration and observe the performance changes
4. Analyze performance bottlenecks for different data volume ranges
5. Select the data transfer method and data type most suitable for your actual application scenario based on the test results
6. Compare the performance differences between local operations (_local) and remote operations (_remote)

## Troubleshooting

### Common Problems

1. **Device unavailable error**

   ```bash
   Error: The device ID is invalid
   ```

   Solution: Use `npu-smi info` to check available devices and ensure that the device1 and device2 parameters are correct

2. **Memory allocation failure**

   ```bash
   Error: aclrtMalloc failed
   ```

   Solution: Reduce the data volume range or core count and ensure that the device has sufficient memory

3. **Peer Access failure**

   ```bash
   Error: aclrtDeviceEnablePeerAccess failed
   ```

   Solution: Ensure that Peer Access is supported between the two devices

4. **Data verification failure**

   ```bash
   [Verification] FAILED: Data mismatch
   ```

   Solution: Check the device status. You may need to reset the device or run the test again.
