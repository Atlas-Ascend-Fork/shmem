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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BIN="${RDMA_QP_DEMO_BIN:-${PROJECT_ROOT}/build/bin/rdma_qp_demo}"

PES=2
GNPUS=2
FPE=0
FNPU=0
QP=2
OP=all
N=1
TOOL=none
IPPORT="${IPPORT:-tcp://127.0.0.1:$((27010 + RANDOM % 900))}"

usage() {
    echo "Usage: $0 [-pes N] [-fpe ID] [-gnpus N] [-fnpu ID] [-qp N]" >&2
    echo "          [-op put|get|aggregate_put|aggregate_get|all] [-ipport tcp://IP:PORT]" >&2
    echo "          [-tool none|msprof] [-n N]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || usage
    case "$1" in
        -pes) PES="$2"; shift 2 ;;
        -fpe) FPE="$2"; shift 2 ;;
        -gnpus) GNPUS="$2"; shift 2 ;;
        -fnpu) FNPU="$2"; shift 2 ;;
        -qp) QP="$2"; shift 2 ;;
        -op) OP="$2"; shift 2 ;;
        -n) N="$2"; shift 2 ;;
        -tool) TOOL="$2"; shift 2 ;;
        -ipport) IPPORT="$2"; shift 2 ;;
        *) usage ;;
    esac
done

for value in "${PES}" "${FPE}" "${GNPUS}" "${FNPU}" "${QP}" "${N}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]{0,8})$ ]] || usage
done
if ((N < 1 || PES < 2 || GNPUS < 1 || QP < 1 || QP > 56 || FPE + GNPUS > PES)); then
    usage
fi
case "${OP}" in
    put|get|aggregate_put|aggregate_get|all) ;;
    *) usage ;;
esac

case "${TOOL}" in
    none|msprof) ;;
    *) usage ;;
esac
[[ -n "${IPPORT}" ]] || usage
if [[ "${TOOL}" == "msprof" ]] && ! command -v msprof >/dev/null 2>&1; then
    echo "msprof not found; source the CANN environment before using -tool msprof." >&2
    exit 1
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not found: ${BIN}" >&2
    echo "Build with RDMA XSCALE or HNS_1825 enabled before running this demo." >&2
    exit 1
fi

export PROJECT_ROOT
export SHMEM_UID_SESSION_ID="${SHMEM_UID_SESSION_ID:-${IPPORT#tcp://}}"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH:-}/lib64:${LD_LIBRARY_PATH:-}"
pids=()
wrapper_dir=""

cleanup() {
    if ((${#pids[@]} > 0)); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
    exit 130
}
trap cleanup INT TERM
cleanup_wrappers() {
    if [[ -n "${wrapper_dir}" ]]; then
        rm -rf -- "${wrapper_dir}"
    fi
}
trap cleanup_wrappers EXIT

if [[ "${TOOL}" == "msprof" ]]; then
    RUN_OUTPUT_DIR="${SCRIPT_DIR}/output/$(date +%Y%m%d_%H%M%S_%N)"
    mkdir -p "${RUN_OUTPUT_DIR}"
    # Keep the wrapper path free of spaces for msprof's application parser.
    wrapper_dir=$(mktemp -d /tmp/rdma_qp_demo_msprof.XXXXXX)
    echo "msprof output: ${RUN_OUTPUT_DIR}"
fi

for ((local_rank = 0; local_rank < GNPUS; ++local_rank)); do
    pe=$((FPE + local_rank))
    app=("${BIN}" -pe "${pe}" -pes "${PES}" -gnpus "${GNPUS}" -fpe "${FPE}" -fnpu "${FNPU}"
        -qp "${QP}" -op "${OP}" -ipport "${IPPORT}" -n "${N}")
    if [[ "${TOOL}" == "msprof" ]]; then
        wrapper="${wrapper_dir}/pe_${pe}.sh"
        {
            printf '#!/bin/bash\nexec'
            printf ' %q' "${app[@]}"
            printf '\n'
        } > "${wrapper}"
        chmod +x "${wrapper}"
        msprof --application="${wrapper}" --output="${RUN_OUTPUT_DIR}/pe_${pe}" &
    else
        "${app[@]}" &
    fi
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
trap - INT TERM
exit "${status}"
