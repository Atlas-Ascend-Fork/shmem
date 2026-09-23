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

EXEC_BIN=${PROJECT_ROOT}/build/bin/rdma_perftest

# 默认测试类型为put
TEST_TYPE="put"
# 默认数据类型为float
DATA_TYPE="float"
# RDMA block-size 仅作为入参兼容；多 QP 带宽自动模式按 QP 数决定 kernel block 数
BLOCK_SIZE_INPUT="1"
# 默认幂数范围
MIN_EXPONENT="3"
MAX_EXPONENT="17"
# 默认循环次数
LOOP_COUNT="1000"
# 默认UB size(B) - RDMA需要至少192B的UB空间
UB_SIZE="192"
# RDMA特有参数
BATCH="0"
MAX_BATCH_SIZE="1024"
AGGREGATE_CACHE_SIZE="64"
AGGREGATE_ITEM_SIZE="128"
AGGREGATE_THRESHOLD=$((32 * 1024))
AGGREGATE_THRESHOLD_EXPONENT="15"
PERFTEST_WARMUP_ITERS="100"
DEFAULT_BATCH_SIZE="100"
SYNC_ID="0"
QP_NUM="1"
QP_INDEX="-1"
QP_SPECIFIED="0"
METRIC="bw"
# 默认RANK配置 - RDMA目前强制PE数量为2
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8768"
GNPU_NUM="2"
FIRST_NPU="0"
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
        --batch)
            if [ -n "$2" ]; then BATCH="$2"; shift 2; else echo "Error: --batch requires a value."; exit 1; fi
            ;;
        --sync-id)
            if [ -n "$2" ]; then SYNC_ID="$2"; shift 2; else echo "Error: --sync-id requires a value."; exit 1; fi
            ;;
        -q|--qp|--qp-count)
            if [ -n "$2" ]; then QP_NUM="$2"; QP_SPECIFIED="1"; shift 2; else echo "Error: -q requires a value."; exit 1; fi
            ;;
        -i|--qp-index)
            if [ -n "$2" ]; then QP_INDEX="$2"; QP_SPECIFIED="1"; shift 2; else echo "Error: --qp-index requires a value."; exit 1; fi
            ;;
        --metric)
            if [ -n "$2" ]; then METRIC="$2"; shift 2; else echo "Error: --metric requires a value."; exit 1; fi
            ;;
        -b|--block-size)
            if [ -n "$2" ]; then
                BLOCK_SIZE_INPUT="$2"
                if [[ "$BLOCK_SIZE_INPUT" != "1" ]]; then
                    echo "WARN: RDMA perftest ignores -b $BLOCK_SIZE_INPUT; bw + qp-count>1 + qp-index=-1 uses QP-derived block_dim"
                fi
                shift 2
            else echo "Error: -b requires a value."; exit 1; fi
            ;;
        --block-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                if [[ "$2" != "1" || "$3" != "1" ]]; then
                    echo "WARN: RDMA perftest ignores --block-range $2 $3; bw + qp-count>1 + qp-index=-1 uses QP-derived block_dim"
                fi
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
        -a|--analyse)
            if [ -n "$2" ]; then ANALYSE_MODE="$2"; shift 2; else echo "Error: -a requires a value."; exit 1; fi
            ;;
        *)
            echo "Error: Unknown option $1."
            echo "使用方法: $0 [选项]"
            echo "  -t|--test-type <type>           put|bi_put|get|bi_get|all"
            echo "  -d|--datatype <type>            float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all"
            echo "  -b|--block-size <size>          RDMA 兼容入参；实际 block_dim 由 QP 自动模式决定"
            echo "  --block-range <min> <max>       RDMA 兼容入参；实际 block_dim 由 QP 自动模式决定"
            echo "  -e|--exponent <exponent>        数据量幂数"
            echo "  --exponent-range <min> <max>    数据量幂数范围"
            echo "  --loop-count <count>            循环次数 (默认 1000)"
            echo "  --ub-size <size>                UB size(B), 192B~131136B, 自动对齐；聚合路径会警告并自动修正到最小值 (默认 192)"
            echo "  --batch <count>                 连续发起多少次操作后等待完成；bw 聚合路径自动修正到可运行值；lat 下作为聚合接口的窗口大小（默认自动 100），lat 会遍历所有接口形态 (默认 0)"
            echo "  --sync-id <id>                  显式传给 Put、Get、Quiet 的同步 ID (默认 0)"
            echo "  -q|--qp|--qp-count <num>        每个 PE 对的 QP 个数，范围 1~56；XSCALE 和 HNS_1825 支持 (默认 1)"
            echo "  -i|--qp-index <index>           固定使用的 QP 编号；-1 表示多 QP 并行模式自动分配 (默认 -1)"
            echo "  --metric <bw|lat>              性能指标: bw=带宽, lat=接口延迟 (默认 bw)"
            echo "  -pes <size>                     PE 数量 (目前强制为 2)"
            echo "  -ipport <ip:port>               通信地址"
            echo "  -gnpus <num>                    NPU 数量"
            echo "  -fnpu <id>                      首个 NPU ID"
            echo "  -a|--analyse <none|plot|md>     分析模式"
            exit 1
            ;;
    esac
done

# 辅助函数：验证是否为非负整数
validate_non_negative_int() {
    local param_name="$1"
    local param_value="$2"
    if ! [[ "$param_value" =~ ^[0-9]+$ ]]; then
        echo "错误: $param_name 必须是非负整数 (got '$param_value')"
        exit 1
    fi
}

# 辅助函数：验证是否为正整数
validate_positive_int() {
    local param_name="$1"
    local param_value="$2"
    if ! [[ "$param_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误: $param_name 必须是正整数 (got '$param_value')"
        exit 1
    fi
}

# 辅助函数：验证 IPPORT 格式
validate_ipport() {
    local param_value="$1"
    if ! [[ "$param_value" =~ ^tcp://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$ ]]; then
        echo "错误: -ipport 格式无效，应为 'tcp://IP:PORT' 格式 (got '$param_value')"
        exit 1
    fi
}

is_xscale_runtime() {
    if [[ "${IBV_EXTEND_DRIVERS:-}" =~ [Xx][Ss][Cc][Aa][Ll][Ee] ]] || \
        [[ "${IBV_EXTEND_DRIVERS:-}" =~ libxscale_nda\.so ]]; then
        return 0
    fi

    if command -v ibv_devinfo &>/dev/null && ibv_devinfo 2>/dev/null | grep -qi "xscale"; then
        return 0
    fi

    return 1
}

is_hns_1825_runtime() {
    if [[ "${IBV_EXTEND_DRIVERS:-}" =~ [Hh][Nn][Ss].*1825 ]] || \
        [[ "${IBV_EXTEND_DRIVERS:-}" =~ libhrn5-rdma ]]; then
        return 0
    fi

    if command -v ibv_devinfo &>/dev/null && ibv_devinfo 2>/dev/null | grep -Eqi "hns.*1825"; then
        return 0
    fi

    return 1
}

is_roce_multi_qp_runtime() {
    is_xscale_runtime || is_hns_1825_runtime
}

get_aggregate_required_ub_size() {
    local batch="$1"
    echo $((AGGREGATE_CACHE_SIZE + AGGREGATE_ITEM_SIZE * batch))
}

aggregate_qp_supported() {
    if is_xscale_runtime; then
        [[ "$QP_NUM" -le "2" ]]
    else
        [[ "$QP_NUM" -eq "1" ]]
    fi
}

uses_aggregate() {
    # lat 模式自动遍历所有接口形态（普通/QP/聚合），必然经过聚合路径；bw 模式保持小消息聚合
    ! is_hns_1825_runtime && aggregate_qp_supported && \
        { [[ "$METRIC" == "lat" ]] || \
          { [[ "$METRIC" == "bw" ]] && [[ "$MIN_EXPONENT" -lt "$AGGREGATE_THRESHOLD_EXPONENT" ]]; } }
}

# 验证测试类型
VALID_TEST_TYPES="put bi_put get bi_get all"
if [[ ! " $VALID_TEST_TYPES " =~ " $TEST_TYPE " ]]; then
    echo "错误: 测试类型必须是 'put' / 'bi_put' / 'get' / 'bi_get' 或 'all'"
    exit 1
fi

# 验证数据类型
VALID_DATATYPES="float int8 int16 int32 int64 uint8 uint16 uint32 uint64 char all"
if [[ ! " $VALID_DATATYPES " =~ " $DATA_TYPE " ]]; then
    echo "错误: 数据类型不在支持列表中"
    exit 1
fi

# 验证分析模式
VALID_ANALYSE_MODES="none plot md"
if [[ ! " $VALID_ANALYSE_MODES " =~ " $ANALYSE_MODE " ]]; then
    echo "错误: 分析模式必须是 'none' / 'plot' / 'md' (got '$ANALYSE_MODE')"
    exit 1
fi

# 验证metric参数
VALID_METRICS="bw lat"
if [[ ! " $VALID_METRICS " =~ " $METRIC " ]]; then
    echo "错误: metric 必须是 'bw' / 'lat' (got '$METRIC')"
    exit 1
fi

# 验证数值参数
validate_non_negative_int "MIN_EXPONENT" "$MIN_EXPONENT"
validate_non_negative_int "MAX_EXPONENT" "$MAX_EXPONENT"
validate_positive_int "LOOP_COUNT" "$LOOP_COUNT"
validate_positive_int "UB_SIZE" "$UB_SIZE"
validate_non_negative_int "BATCH" "$BATCH"
validate_non_negative_int "SYNC_ID" "$SYNC_ID"
validate_positive_int "QP_NUM" "$QP_NUM"
validate_positive_int "PE_SIZE" "$PE_SIZE"
validate_positive_int "GNPU_NUM" "$GNPU_NUM"
validate_non_negative_int "FIRST_NPU" "$FIRST_NPU"
if ! [[ "$QP_INDEX" =~ ^-1$|^[0-9]+$ ]]; then
    echo "错误: QP_INDEX 必须是 -1 或非负整数 (got '$QP_INDEX')"
    exit 1
fi

# 验证 IPPORT 格式
validate_ipport "$IPPORT"

# 验证幂数范围
if [[ "$MIN_EXPONENT" -gt "$MAX_EXPONENT" ]]; then
    echo "错误: MIN_EXPONENT ($MIN_EXPONENT) 不能大于 MAX_EXPONENT ($MAX_EXPONENT)"
    exit 1
fi

# 验证 QP_NUM
if [[ "$QP_NUM" -lt "1" || "$QP_NUM" -gt "56" ]]; then
    echo "错误: QP_NUM 必须在 1~56 范围内"
    exit 1
fi

if ! is_roce_multi_qp_runtime && [[ "$QP_SPECIFIED" == "1" ]]; then
    echo "错误: 当前环境不支持 QP 设置，仅云脉（XSCALE）和 1825 网卡支持 -q/--qp 与 -i/--qp-index。"
    exit 1
fi

if [[ "$QP_INDEX" -ge "$QP_NUM" ]]; then
    echo "错误: --qp-index 的值必须为 -1 或小于 --qp 的值（当前 --qp=$QP_NUM, --qp-index=$QP_INDEX）"
    exit 1
fi

if [[ "$METRIC" == "lat" && "$QP_NUM" -gt "1" ]]; then
    echo "错误: latency 测试不支持 qp_num > 1，请使用 --qp 1"
    exit 1
fi

# 验证并强制 PE_SIZE
if [[ "$PE_SIZE" != "2" ]]; then
    echo "WARN: RDMA perftest 目前强制 PE 数量为 2，忽略 -pes $PE_SIZE"
    PE_SIZE="2"
fi

if [[ "$UB_SIZE" -lt "192" ]]; then
    echo "警告: --ub-size $UB_SIZE 过小，已修正为 192"
    UB_SIZE="192"
elif [[ "$UB_SIZE" -gt "131136" ]]; then
    echo "警告: --ub-size $UB_SIZE 过大，已修正为 131136"
    UB_SIZE="131136"
fi

if is_xscale_runtime; then
    RDMA_NIC_TYPE="XSCALE"
    AGGREGATE_THRESHOLD=$((64 * 1024))
    AGGREGATE_THRESHOLD_EXPONENT="16"
elif is_hns_1825_runtime; then
    RDMA_NIC_TYPE="HNS_1825"
else
    RDMA_NIC_TYPE="IN_DIE/default"
fi

if uses_aggregate; then
    # Normalize the window before deriving the full batch size and UB capacity.
    if [[ "$BATCH" -eq 0 ]]; then
        echo "提示: --batch 未指定，自动设为 $DEFAULT_BATCH_SIZE"
        BATCH="$DEFAULT_BATCH_SIZE"
    elif [[ "$METRIC" == "lat" ]]; then
        if [[ "$BATCH" -gt "$LOOP_COUNT" ]]; then
            BATCH="$LOOP_COUNT"
        fi
    elif [[ "$BATCH" -gt "$LOOP_COUNT" || "$BATCH" -ge "$MAX_BATCH_SIZE" ]]; then
        CORRECT_BATCH="$DEFAULT_BATCH_SIZE"
        if [[ "$LOOP_COUNT" -lt "$CORRECT_BATCH" ]]; then
            CORRECT_BATCH="$LOOP_COUNT"
        fi
        echo "警告: --batch 的值 $BATCH 无效（需不大于 loop-count $LOOP_COUNT 且小于 1024），已修正为 $CORRECT_BATCH"
        BATCH="$CORRECT_BATCH"
    fi

    if [[ "$METRIC" == "lat" ]]; then
        # N defer calls append one submit WQE; N+1 must remain below SQ depth.
        MAX_LAT_WINDOW=$((MAX_BATCH_SIZE - 2))
        if [[ "$BATCH" -gt "$MAX_LAT_WINDOW" ]]; then
            echo "提示: 时延窗口缩小为 $MAX_LAT_WINDOW，为 submit 预留一条 WQE"
            BATCH="$MAX_LAT_WINDOW"
        fi
        FULL_BATCH_COUNT=$((BATCH + 1))
    else
        # In bandwidth mode, submit is the last of N operations, not an extra one.
        FULL_BATCH_COUNT="$BATCH"
    fi

    MAX_AGGREGATE_COUNT="$PERFTEST_WARMUP_ITERS"
    if [[ "$FULL_BATCH_COUNT" -gt "$MAX_AGGREGATE_COUNT" ]]; then
        MAX_AGGREGATE_COUNT="$FULL_BATCH_COUNT"
    fi
    if [[ "$MAX_AGGREGATE_COUNT" -ge "$MAX_BATCH_SIZE" ]]; then
        echo "错误: 完整聚合批次 $MAX_AGGREGATE_COUNT 必须小于 QP 队列深度 $MAX_BATCH_SIZE"
        exit 1
    fi

    REQUIRED_UB_SIZE=$(get_aggregate_required_ub_size "$MAX_AGGREGATE_COUNT")
    if [[ "$REQUIRED_UB_SIZE" -gt "131136" ]]; then
        echo "错误: 所需 UB 大小 ${REQUIRED_UB_SIZE}B 超过上限 131136B，请减小 --batch"
        exit 1
    fi
    if [[ "$UB_SIZE" -lt "$REQUIRED_UB_SIZE" ]]; then
        echo "警告: 聚合场景要求 --ub-size >= ${REQUIRED_UB_SIZE}B，当前为 ${UB_SIZE}B，已自动修正"
        UB_SIZE="$REQUIRED_UB_SIZE"
    fi
fi

echo "测试类型: $TEST_TYPE"
echo "数据类型: $DATA_TYPE"
echo "幂数范围: $MIN_EXPONENT-$MAX_EXPONENT"
echo "循环次数: $LOOP_COUNT"
echo "UB size(B): $UB_SIZE"
echo "Batch size: $BATCH"
echo "Sync ID: $SYNC_ID"
echo "QP count: $QP_NUM"
echo "QP index: $QP_INDEX"
echo "QP specified: $QP_SPECIFIED"
echo "Metric: $METRIC"
echo "RDMA NIC type: $RDMA_NIC_TYPE"
if uses_aggregate; then
    echo "RDMA bandwidth aggregation: qp <= 2 and message below ${AGGREGATE_THRESHOLD}B"
fi
echo "PE_SIZE: $PE_SIZE, GNPU_NUM: $GNPU_NUM"
echo "FIRST_NPU: $FIRST_NPU"

ALL_TEST_TYPES=("put" "bi_put" "get" "bi_get")
ALL_DATATYPES=("float" "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64" "char")

run_test() {
    local test_type="$1"
    local data_type="$2"
    local -a pids=()
    for (( idx =0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
        ${EXEC_BIN} --pes "$PE_SIZE" --pe-id "$idx" --ipport "$IPPORT" --gnpus "$GNPU_NUM" \
            --fnpu "$FIRST_NPU" -t "$test_type" -d "$data_type" \
            --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" \
            --ub-size "$UB_SIZE" --batch "$BATCH" --sync-id "$SYNC_ID" \
            --metric "$METRIC" --qp-count "$QP_NUM" --qp-index "$QP_INDEX" --qp-specified "$QP_SPECIFIED" &
        pids+=($!)
    done
    local failed=0
    local exit_code=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    return $failed
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
