#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

EXEC_BIN=${PROJECT_ROOT}/build/bin/udma_perftest

# 默认测试类型为put
TEST_TYPE="put"
# 默认数据类型为float
DATA_TYPE="float"
# UDMA 中 block 数即 QP 数，通过 -b/--block-size 设置
BLOCK_SIZE_INPUT="1"
# 默认幂数范围
MIN_EXPONENT="3"
MAX_EXPONENT="17"
# 默认循环次数
LOOP_COUNT="1000"
# 默认UB size(KB)
UB_SIZE="16"
# 默认 metric: bw|lat (lat 仅支持 -t put)
METRIC="bw"
# 默认 batch：0 表示按 loop_count 全异步提交，仅末尾一次 quiet
# 1：每次普通 NBI 提交后调用 quiet；
# >1：按 N 次操作分组，普通 put/get 每组执行 N-1 次 defer 和 1 次 submit，随后调用 quiet；put_signal 始终使用普通 NBI。
BATCH="0"
# 默认RANK配置
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8768"
GNPU_NUM="2"
FIRST_NPU="0"
FIRST_PE="0"
# 分析模式: none/plot/md
ANALYSE_MODE="none"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-type)
            if [ -n "$2" ]; then TEST_TYPE="$2"; shift 2; else echo "Error: -t requires a value."; exit 1; fi
            ;;
        -d|--datatype)
            if [ -n "$2" ]; then DATA_TYPE="$2"; shift 2; else echo "Error: -d requires a value."; exit 1; fi
            ;;
        --loop-count)
            if [ -n "$2" ]; then LOOP_COUNT="$2"; shift 2; else echo "Error: --loop-count requires a value."; exit 1; fi
            ;;
        --ub-size)
            if [ -n "$2" ]; then UB_SIZE="$2"; shift 2; else echo "Error: --ub-size requires a value."; exit 1; fi
            ;;
        --metric)
            if [ -n "$2" ]; then METRIC="$2"; shift 2; else echo "Error: --metric requires a value (bw|lat)."; exit 1; fi
            ;;
        --batch)
            if [ -n "$2" ]; then BATCH="$2"; shift 2; else echo "Error: --batch requires a value."; exit 1; fi
            ;;
        -b|--block-size)
            if [ -n "$2" ]; then
                BLOCK_SIZE_INPUT="$2"
                shift 2
            else echo "Error: -b requires a value."; exit 1; fi
            ;;
        --block-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                echo "WARN: UDMA ignores --block-range; use -b|--block-size N to set block/QP count"
                shift 3
            else echo "Error: --block-range requires two values."; exit 1; fi
            ;;
        -e|--exponent)
            if [ -n "$2" ]; then MIN_EXPONENT="$2"; MAX_EXPONENT="$2"; shift 2;
            else echo "Error: -e requires a value."; exit 1; fi
            ;;
        --exponent-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                MIN_EXPONENT="$2"; MAX_EXPONENT="$3"; shift 3
            else echo "Error: --exponent-range requires two values."; exit 1; fi
            ;;
        -pes)
            if [ -n "$2" ]; then
                PE_SIZE="$2"
                if [[ "$GNPU_NUM" -gt "$PE_SIZE" ]]; then
                    GNPU_NUM="$PE_SIZE"
                fi
                shift 2
            else echo "Error: -pes requires a value."; exit 1; fi
            ;;
        -ipport)
            if [ -n "$2" ]; then IPPORT="$2"; shift 2; else echo "Error: -ipport requires a value."; exit 1; fi
            ;;
        -gnpus)
            if [ -n "$2" ]; then
                GNPU_NUM="$2"
                if [[ "$GNPU_NUM" -gt "$PE_SIZE" ]]; then GNPU_NUM="$PE_SIZE"; fi
                shift 2
            else echo "Error: -gnpus requires a value."; exit 1; fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then FIRST_NPU="$2"; shift 2; else echo "Error: -fnpu requires a value."; exit 1; fi
            ;;
        -fpe)
            if [ -n "$2" ]; then FIRST_PE="$2"; shift 2; else echo "Error: -fpe requires a value."; exit 1; fi
            ;;
        -a|--analyse)
            if [ -n "$2" ]; then ANALYSE_MODE="$2"; shift 2; else echo "Error: -a requires a value."; exit 1; fi
            ;;
        *)
            echo "Error: Unknown option $1."
            echo "使用方法: $0 [选项]"
            echo "  -t|--test-type <type>           put|bi_put|get|bi_get|put_signal|all"
            echo "  -d|--datatype <type>            float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all"
            echo "  -b|--block-size <size>          UDMA block 数，同时作为 QP 数（一个 block 对应一个 QP）"
            echo "  --block-range <min> <max>       UDMA 忽略范围，不做扫描；使用 -b|--block-size 指定 QP 数"
            echo "  -e|--exponent <exponent>        单个 QP 的数据量幂数"
            echo "  --exponent-range <min> <max>    单个 QP 的数据量幂数范围"
            echo "  --loop-count <count>            循环次数 (默认 1000)"
            echo "  --ub-size <size>                UB size(KB)，用于低阶 UDMA 入参和 CSV；高阶默认 MTE staging 需至少 128B"
            echo "  --metric <bw|lat>               性能口径 (lat 仅支持 -t put)"
            echo "  --batch <N>                     BW 测试：0=普通 NBI 全异步，1=每次后 quiet，>1=每批 defer...submit 后 quiet"
            echo "  -pes <size>                     PE 数量"
            echo "  -ipport <ip:port>               通信地址"
            echo "  -gnpus <num>                    NPU 数量"
            echo "  -fnpu <id>                      首个 NPU ID"
            echo "  -fpe <id>                       首个 PE ID"
            echo "  -a|--analyse <none|plot|md>     分析模式"
            exit 1
            ;;
    esac
done

VALID_TEST_TYPES="put bi_put get bi_get put_signal all"
if [[ ! " $VALID_TEST_TYPES " =~ " $TEST_TYPE " ]]; then
    echo "错误: 测试类型必须是 'put' / 'bi_put' / 'get' / 'bi_get' / 'put_signal' 或 'all'"
    exit 1
fi

VALID_DATATYPES="float int8 int16 int32 int64 uint8 uint16 uint32 uint64 char all"
if [[ ! " $VALID_DATATYPES " =~ " $DATA_TYPE " ]]; then
    echo "错误: 数据类型不在支持列表中"
    exit 1
fi

VALID_METRICS="bw lat"
if [[ ! " $VALID_METRICS " =~ " $METRIC " ]]; then
    echo "错误: --metric 必须是 'bw' 或 'lat'"
    exit 1
fi
if [[ "$METRIC" == "lat" && "$TEST_TYPE" != "put" ]]; then
    echo "错误: --metric lat 仅支持 -t put (got -t $TEST_TYPE)"
    exit 1
fi

if ! [[ "$BATCH" =~ ^[0-9]+$ ]]; then
    echo "错误: --batch 必须是非负整数 (got '$BATCH')"
    exit 1
fi
if ! [[ "$BLOCK_SIZE_INPUT" =~ ^[0-9]+$ ]] || [ "$BLOCK_SIZE_INPUT" -lt 1 ] || [ "$BLOCK_SIZE_INPUT" -gt 56 ]; then
    echo "错误: --block-size 必须是 [1,56] 内的整数"
    exit 1
fi
if [[ "$BLOCK_SIZE_INPUT" -gt 1 && ( "$TEST_TYPE" == "put_signal" || "$TEST_TYPE" == "all" ) ]]; then
    echo "错误: 多 QP 仅支持 put/bi_put/get/bi_get；put_signal 保持单 QP"
    exit 1
fi

echo "测试类型: $TEST_TYPE"
echo "数据类型: $DATA_TYPE"
echo "幂数范围: $MIN_EXPONENT-$MAX_EXPONENT"
echo "循环次数: $LOOP_COUNT"
echo "UB size(KB): $UB_SIZE"
echo "Metric: $METRIC"
echo "Batch: $BATCH"
echo "Block/QP count: $BLOCK_SIZE_INPUT"
echo "PE_SIZE: $PE_SIZE, GNPU_NUM: $GNPU_NUM"
echo "FIRST_NPU: $FIRST_NPU, FIRST_PE: $FIRST_PE"

export SHMEM_CYCLE_PROF_PE=${SHMEM_CYCLE_PROF_PE:-0}

ALL_TEST_TYPES=("put" "bi_put" "get" "bi_get" "put_signal")
ALL_DATATYPES=("float" "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64" "char")

run_test() {
    local test_type="$1"
    local data_type="$2"
    local -a pids=()
    for (( idx =0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
        ${EXEC_BIN} --pes "$PE_SIZE" --pe-id "$idx" --ipport "$IPPORT" --gnpus "$GNPU_NUM" \
            --fpe "$FIRST_PE" --fnpu "$FIRST_NPU" -t "$test_type" -d "$data_type" \
            --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" \
            --ub-size "$UB_SIZE" --metric "$METRIC" --batch "$BATCH" \
            --block-size "$BLOCK_SIZE_INPUT" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    return "$failed"
}

overall_failed=0

if [[ "$TEST_TYPE" == "all" && "$DATA_TYPE" == "all" ]]; then
    for type in "${ALL_TEST_TYPES[@]}"; do
        for dtype in "${ALL_DATATYPES[@]}"; do
            echo -e "\n=== 运行测试类型: $type, 数据类型: $dtype ==="
            run_test "$type" "$dtype" || overall_failed=1
        done
    done
elif [[ "$TEST_TYPE" == "all" ]]; then
    for type in "${ALL_TEST_TYPES[@]}"; do
        echo -e "\n=== 运行测试类型: $type, 数据类型: $DATA_TYPE ==="
        run_test "$type" "$DATA_TYPE" || overall_failed=1
    done
elif [[ "$DATA_TYPE" == "all" ]]; then
    for dtype in "${ALL_DATATYPES[@]}"; do
        echo -e "\n=== 运行测试类型: $TEST_TYPE, 数据类型: $dtype ==="
        run_test "$TEST_TYPE" "$dtype" || overall_failed=1
    done
else
    run_test "$TEST_TYPE" "$DATA_TYPE" || overall_failed=1
fi

cd ${CURRENT_DIR}

PERF_SCRIPT="${SCRIPT_DIR}/../../utils/perf_data_process.py"
if [ -f "${PERF_SCRIPT}" ]; then
    if [ "$ANALYSE_MODE" = "plot" ] || [ "$ANALYSE_MODE" = "md" ]; then
        echo -e "\n========== Generating performance charts =========="
        cmd_args=("-d" "${SCRIPT_DIR}/output")
        if [ "$ANALYSE_MODE" = "plot" ]; then
            cmd_args+=("--no-markdown")
        fi
        python3 "${PERF_SCRIPT}" "${cmd_args[@]}"
    fi
fi

exit "$overall_failed"
