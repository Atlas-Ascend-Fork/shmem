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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd ${script_dir}/../../ && pwd)"
export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
cd ${PROJECT_ROOT}

num_pes="${NUM_PES:-2}"
if ! [[ "${num_pes}" =~ ^[0-9]+$ ]] || (( num_pes < 2 )); then
    echo "NUM_PES must be an integer >= 2" >&2
    exit 1
fi

pids=()
for ((pe = 0; pe < num_pes; ++pe)); do
    ./build/bin/udma_put_value_reduce "${num_pes}" "${pe}" tcp://127.0.0.1:8899 "${num_pes}" 0 0 &
    pids+=("$!")
done

ret=0
for pid in ${pids[@]}; do
    wait $pid
    r=$?
    if [[ $r -ne 0 ]]; then
        ret=$r
    fi
    echo "wait $pid finished"
done
exit $ret
