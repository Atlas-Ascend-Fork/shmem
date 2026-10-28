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
export SHMEM_CYCLE_PROF_PE=${SHMEM_CYCLE_PROF_PE:-0}

BIN="${PROJECT_ROOT}/build/bin/simt_udma_perftest"

# --- Defaults ---
SCOPE="all"
TEST_TYPE="all"
QP_COUNT="4"
MIN_EXP="3"
MAX_EXP="17"
LOOP_COUNT="1000"
WARMUP="100"
METRIC="bw"
HEAP_MB="1024"
IPPORT="tcp://127.0.0.1:8999"
PE_SIZE="2"
GNPU_NUM="2"
FIRST_NPU="0"
FIRST_PE="0"

usage() {
    cat <<'EOF'
Usage: run.sh [options]

Launches the SIMT UDMA QP-specific perf test with a fixed 2-card
(Active PE0 / Passive PE1) model.

Options:
  --scope <type>                Execution scope: warp|block|all.              (default: all)
  -t|--test-type <type>         Operation type: put|get|bi_put|bi_get|all.    (default: all)
  --qp <N>                      QP count per peer, range [1, 56].             (default: 4)
  -e|--exponent <E>             Single transfer-size exponent; size = 2^E B, range [1, 28].
  --exponent-range <min> <max>  Transfer-size exponent range, range [1, 28].   (default: 3 17)
  --loop-count <N>              Timed iterations per data point.              (default: 1000)
  --warmup <N>                  Warmup iterations before timing.              (default: 100)
  --metric <bw|lat>             bw = bandwidth; lat = per-call latency.       (default: bw)
  --heap-mb <N>                 SHMEM symmetric heap per PE in MiB.           (default: 1024)
  -ipport <ip:port>             Bootstrap communication address.              (default: tcp://127.0.0.1:8999)
  -pes <N>                      Number of PEs, must be 2.                     (default: 2)
  -gnpus <N>                    Number of NPUs used on this node, must be 2.  (default: 2)
  -fnpu <id>                    First NPU id; device = pe % gnpus + fnpu.     (default: 0)
  -fpe <id>                     First PE id. Kept for CLI compatibility; unused. (default: 0)
  -h|--help                     Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scope)
            if [ -n "$2" ]; then SCOPE="$2"; shift 2; else echo "Error: --scope requires a value."; exit 1; fi
            ;;
        -t|--test-type)
            if [ -n "$2" ]; then TEST_TYPE="$2"; shift 2; else echo "Error: -t requires a value."; exit 1; fi
            ;;
        --qp)
            if [ -n "$2" ] && [[ "$2" =~ ^[1-9][0-9]*$ ]] && [ "$2" -le 56 ]; then
                QP_COUNT="$2"; shift 2
            else
                echo "Error: --qp requires an integer in [1, 56]."; exit 1
            fi
            ;;
        -e|--exponent)
            if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then MIN_EXP="$2"; MAX_EXP="$2"; shift 2
            else echo "Error: -e requires a non-negative integer."; exit 1; fi
            ;;
        --exponent-range)
            if [ -n "$2" ] && [ -n "$3" ] && [[ "$2" =~ ^[0-9]+$ ]] && [[ "$3" =~ ^[0-9]+$ ]]; then
                MIN_EXP="$2"; MAX_EXP="$3"; shift 3
            else
                echo "Error: --exponent-range requires two non-negative integers."; exit 1
            fi
            ;;
        --loop-count)
            if [ -n "$2" ] && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then LOOP_COUNT="$2"; shift 2
            else echo "Error: --loop-count requires a positive integer."; exit 1; fi
            ;;
        --warmup)
            if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then WARMUP="$2"; shift 2
            else echo "Error: --warmup requires a non-negative integer."; exit 1; fi
            ;;
        --metric)
            if [ -n "$2" ]; then METRIC="$2"; shift 2
            else echo "Error: --metric requires a value (bw|lat)."; exit 1; fi
            ;;
        --heap-mb)
            if [ -n "$2" ] && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then HEAP_MB="$2"; shift 2
            else echo "Error: --heap-mb requires a positive integer."; exit 1; fi
            ;;
        -ipport)
            if [ -n "$2" ]; then IPPORT="$2"; shift 2; else echo "Error: -ipport requires a value."; exit 1; fi
            ;;
        -pes)
            if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then PE_SIZE="$2"; shift 2
            else echo "Error: -pes requires a non-negative integer."; exit 1; fi
            ;;
        -gnpus)
            if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then GNPU_NUM="$2"; shift 2
            else echo "Error: -gnpus requires a non-negative integer."; exit 1; fi
            ;;
        -fnpu)
            if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then FIRST_NPU="$2"; shift 2
            else echo "Error: -fnpu requires a non-negative integer."; exit 1; fi
            ;;
        -fpe)
            if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then FIRST_PE="$2"; shift 2
            else echo "Error: -fpe requires a non-negative integer."; exit 1; fi
            ;;
        -h|--help)
            usage; exit 0
            ;;
        *)
            echo "Error: Unknown option $1."
            usage
            exit 1
            ;;
    esac
done

VALID_SCOPES="warp block"
VALID_OPS="put get bi_put bi_get"
VALID_METRICS="bw lat"

if [[ "$SCOPE" != "all" ]] && [[ ! " $VALID_SCOPES " =~ " $SCOPE " ]]; then
    echo "Error: --scope must be warp, block, or all (got '$SCOPE')"; exit 1
fi
if [[ "$TEST_TYPE" != "all" ]] && [[ ! " $VALID_OPS " =~ " $TEST_TYPE " ]]; then
    echo "Error: -t must be put, get, bi_put, bi_get, or all (got '$TEST_TYPE')"; exit 1
fi
if [[ ! " $VALID_METRICS " =~ " $METRIC " ]]; then
    echo "Error: --metric must be bw or lat (got '$METRIC')"; exit 1
fi
if [[ "$MIN_EXP" -gt "$MAX_EXP" ]]; then
    echo "Error: exponent range must satisfy min <= max (got $MIN_EXP-$MAX_EXP)"; exit 1
fi
if [[ "$MIN_EXP" -lt 1 ]]; then
    echo "Error: min exponent must be >= 1 (2^0 = 1 byte is below the UDMA engine 2-byte minimum per QP)."; exit 1
fi
if [[ "$MAX_EXP" -gt 28 ]]; then
    echo "Error: max exponent must be <= 28 (2^28 = 256 MiB is the SIMT UDMA single-request upper bound)."; exit 1
fi
# Fixed two-card Active(PE0)/Passive(PE1) model: -pes and -gnpus must both be 2.
if [[ "$PE_SIZE" -ne 2 || "$GNPU_NUM" -ne 2 ]]; then
    echo "Error: -pes and -gnpus must both be 2 for the simt_udma_perftest fixed 2-card model."
    exit 1
fi

echo "Scope: $SCOPE  TestType: $TEST_TYPE  QP=$QP_COUNT  Exp=$MIN_EXP-$MAX_EXP " \
    "Loop=$LOOP_COUNT  Warmup=$WARMUP  Metric=$METRIC"

if [ ! -f "${BIN}" ]; then
    echo "[ERROR] Binary not found: ${BIN}. Build with:" \
        "bash scripts/build.sh -examples -enable_simt -soc_type Ascend950"
    exit 1
fi

overall_failed=0
pids=()
for (( idx = 0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
    "${BIN}" --pes "${PE_SIZE}" --pe-id "${idx}" --ipport "${IPPORT}" --gnpus "${GNPU_NUM}" \
        --fpe "${FIRST_PE}" --fnpu "${FIRST_NPU}" --scope "${SCOPE}" -t "${TEST_TYPE}" --qp "${QP_COUNT}" \
        --exponent-range "${MIN_EXP}" "${MAX_EXP}" --loop-count "${LOOP_COUNT}" \
        --warmup "${WARMUP}" --metric "${METRIC}" --heap-mb "${HEAP_MB}" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || overall_failed=1
done

cd ${CURRENT_DIR}
exit ${overall_failed}
