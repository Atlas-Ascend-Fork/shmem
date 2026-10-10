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

usage() {
    echo "Usage: bash $0 <n_pes> <n_pes_per_node> <node_id> <tcp://rank0_ip:port> [first_device_id:0]"
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage >&2
    exit 1
fi

n_pes=$1
n_pes_per_node=$2
node_id=$3
ipport=$4
first_device_id=${5:-0}
timeout_seconds=${BARRIER_ALL_TIMEOUT:-120}

# n_pes: total PEs. n_pes_per_node: cards on this machine. node_id: this machine's index.
# Single machine: n_pes_per_node == n_pes. Dual machine: n_pes_per_node == n_pes / 2.
# Validate the raw argument format first (digits only). Leading zeros are allowed
# here but normalized to decimal below, because Bash arithmetic would otherwise
# interpret them as octal (and reject "08"/"09").
if [[ ! "$n_pes" =~ ^[0-9]+$ ]] ||
   [[ ! "$n_pes_per_node" =~ ^[0-9]+$ ]] ||
   [[ ! "$node_id" =~ ^[0-9]+$ ]] ||
   [[ ! "$ipport" =~ ^tcp://[^/[:space:]]+:[0-9]+$ ]] ||
   [[ ! "$first_device_id" =~ ^[0-9]{1,6}$ ]] ||
   [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    usage >&2
    echo "[ERROR] Invalid arguments or BARRIER_ALL_TIMEOUT." >&2
    exit 1
fi

# Normalize all counts/indices to decimal (10#) so leading zeros are not read as
# octal, keeping the launcher consistent with main's strtol(..., 10) parsing.
n_pes=$((10#$n_pes))
n_pes_per_node=$((10#$n_pes_per_node))
node_id=$((10#$node_id))
first_device_id=$((10#$first_device_id))

if [[ "$n_pes" -lt 2 ]] ||
   [[ "$n_pes_per_node" -lt 1 ]] ||
   [[ $((n_pes % n_pes_per_node)) -ne 0 ]] ||
   [[ "$node_id" -ge $((n_pes / n_pes_per_node)) ]]; then
    usage >&2
    echo "[ERROR] Invalid arguments or BARRIER_ALL_TIMEOUT." >&2
    exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] GNU timeout is required." >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${script_dir}/../.." && pwd)"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SHMEM_UID_SESSION_ID="${ipport#tcp://}"
binary="${PROJECT_ROOT}/build/bin/barrier_all"
if [[ ! -x "$binary" ]]; then
    echo "[ERROR] Build the barrier_all example first: $binary" >&2
    exit 1
fi

engine_args=()
if (( n_pes > n_pes_per_node )); then
    engine_args=(--roce)
fi

pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Launch this machine's local PEs.
for ((local_pe = 0; local_pe < n_pes_per_node; local_pe++)); do
    pe_id=$((node_id * n_pes_per_node + local_pe))
    device_id=$((first_device_id + local_pe))
    timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" \
        "$binary" "$pe_id" "$device_id" "$n_pes" "$ipport" "${engine_args[@]}" &
    pids+=("$!")
done

ret=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        status=0
    else
        status=$?
    fi
    pids[$index]=""
    if [[ $status -ne 0 ]]; then
        echo "[ERROR] PE $((node_id * n_pes_per_node + index)) exited with status $status." >&2
        if [[ $ret -eq 0 ]]; then
            ret=$status
        fi
    fi
done
exit "$ret"
