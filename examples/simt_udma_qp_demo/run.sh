#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

EXAMPLE=simt_udma_qp_demo

cd ${SCRIPT_DIR}

GNPU_NUM="2"
MODE="all"
QP_COUNT="2"
ELEMS="1048576"
HEAP_MB="1024"
FIRST_NPU="0"
IPPORT="tcp://127.0.0.1:8999"

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

while [[ $# -gt 0 ]]; do
    case "$1" in
        -pes)
            if [ -n "$2" ] && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                GNPU_NUM="$2"; shift 2
            else
                echo "Error: -pes requires a positive integer value."; exit 1
            fi
            ;;
        -mode)
            if [ -n "$2" ]; then
                MODE="$2"; shift 2
            else
                echo "Error: -mode requires a value (warp|block|all)."; exit 1
            fi
            ;;
        -qp_count)
            if [ -n "$2" ] && [[ "$2" =~ ^[1-9][0-9]*$ ]] && [ "$2" -le 56 ]; then
                QP_COUNT="$2"; shift 2
            else
                echo "Error: -qp_count requires an integer in [1, 56]."; exit 1
            fi
            ;;
        -elems)
            if [ -n "$2" ] && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                ELEMS="$2"; shift 2
            else
                echo "Error: -elems requires a positive integer value."; exit 1
            fi
            ;;
        -heap_mb)
            if [ -n "$2" ] && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                HEAP_MB="$2"; shift 2
            else
                echo "Error: -heap_mb requires a positive integer value."; exit 1
            fi
            ;;
        -first_npu)
            if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
                FIRST_NPU="$2"; shift 2
            else
                echo "Error: -first_npu requires a non-negative integer value."; exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"; shift 2
            else
                echo "Error: -ipport requires a value."; exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."; exit 1
            ;;
    esac
done

pids=()
for (( idx = 0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
    ${PROJECT_ROOT}/build/bin/${EXAMPLE} -pes ${GNPU_NUM} -pe ${idx} -mode ${MODE} -qp_count ${QP_COUNT} \
        -elems ${ELEMS} -heap_mb ${HEAP_MB} -first_npu ${FIRST_NPU} -ipport ${IPPORT} &
    pid=$!
    pids+=("$pid")
    echo "$pid background process recorded"
done

ret=0
for pid in ${pids[@]}; do
    wait $pid
    cur_ret=$?
    echo "wait process $pid done"
    if [[ $cur_ret -ne 0 ]]; then
        ret=$cur_ret
    fi
done

cd ${CURRENT_DIR}
exit $ret
