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

set -e

usage() {
    echo "Usage: $0 [-ipport tcp://ip:port] [-pes N] [-gnpus N] [-fpe ID] [-fnpu ID]"
    exit 1
}

ipport=tcp://127.0.0.1:8299
pes=2
gnpus=""
fpe=0
fnpu=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ipport)
            ipport="$2"
            shift 2
            ;;
        -pes)
            pes="$2"
            shift 2
            ;;
        -gnpus)
            gnpus="$2"
            shift 2
            ;;
        -fpe)
            fpe="$2"
            shift 2
            ;;
        -fnpu)
            fnpu="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# 未指定 -gnpus 时默认与 -pes 相同（单机全量场景）
[[ -z "$gnpus" ]] && gnpus=$pes
[[ "$pes" =~ ^[1-9][0-9]*$ && "$gnpus" =~ ^[1-9][0-9]*$ && "$fpe" =~ ^[0-9]+$ && "$fnpu" =~ ^[0-9]+$ ]] || usage
(( pes >= 2 )) || usage
(( (pes & (pes - 1)) == 0 )) || { echo "ERROR: pes must be a power of 2 (2, 4, 8, 16, ...), got $pes"; exit 1; }
(( fpe + gnpus <= pes )) || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../" && pwd)"
export PROJECT_ROOT="$project_root"
export LD_LIBRARY_PATH="$PROJECT_ROOT/build/lib:$LD_LIBRARY_PATH"
cd "$project_root"

pids=()
for ((offset = 0; offset < gnpus; offset++)); do
    pe_id=$((fpe + offset))
    ./build/bin/rdma_aggregate_demo "$pes" "$pe_id" "$ipport" "$gnpus" "$fpe" "$fnpu" &
    pids+=("$!")
done

ret=0
for pid in "${pids[@]}"; do
    wait "$pid" || ret=1
done
exit "$ret"
