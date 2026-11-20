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

num_pes=2
if [[ $# -ne 0 ]]; then
    if [[ $# -ne 2 || "$1" != "-pes" ]]; then
        echo "Usage: bash $0 [-pes <number_of_pes>]"
        exit 1
    fi
    num_pes="$2"
fi
if ! [[ "$num_pes" =~ ^[1-9][0-9]*$ ]] || ((num_pes < 2)); then
    echo "number_of_pes must be at least 2"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
export PROJECT_ROOT="${project_root}"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
export SHMEM_UID_SESSION_ID="${SHMEM_UID_SESSION_ID:-127.0.0.1:8899}"
binary="${PROJECT_ROOT}/build/bin/rdma_reduce_demo"
if [[ ! -x "$binary" ]]; then
    echo "Build rdma_reduce_demo with Ascend950 and HNS_1825 first: $binary"
    exit 1
fi
cd "${PROJECT_ROOT}" || exit 1

pids=()
for ((pe=0; pe<num_pes; ++pe)); do
    "$binary" "$num_pes" "$pe" "tcp://${SHMEM_UID_SESSION_ID}" "$num_pes" 0 0 &
    pids+=("$!")
done

ret=0
for pid in "${pids[@]}"; do
    wait "$pid" || ret=1
done
exit "$ret"
