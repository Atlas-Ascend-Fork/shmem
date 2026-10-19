#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
readonly CURRENT_DIR=$(pwd)
readonly SCRIPT_DIR=$(dirname $(readlink -f "$0"))
readonly PROJECT_ROOT=${SCRIPT_DIR}/../../
readonly NPROC_PER_NODE=${NPROC_PER_NODE:-2}
readonly HANDLE_WAIT_NPROC_PER_NODE=${HANDLE_WAIT_NPROC_PER_NODE:-1}
readonly TEST_TIMEOUT=${SHMEM_TEST_TIMEOUT:-10m}

function run_torchrun()
{
    timeout --signal=TERM "${TEST_TIMEOUT}" torchrun "$@"
}

function pre_check()
{
    cd $PROJECT_ROOT
    pip show cann-shmem >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        echo "begin uninstall old cann-shmem wheel package"
        pip uninstall --yes cann-shmem
    fi

    echo "begin install cann-shmem wheel package"

    whl_file=$(find "${PROJECT_ROOT}/dist" -maxdepth 1 -name "cann_shmem-*.whl" -type f | sort | tail -n 1)

    if [[ -z "$whl_file" ]]; then
        echo "No cann-shmem wheel found in ${PROJECT_ROOT}/dist."
        echo "Execute 'bash scripts/build.sh -python_extension' in the project root directory."
        exit 1
    fi

    echo "Found wheel: $whl_file"
    # The runtime dependencies (torch/torch-npu) are environment prerequisites;
    # do not replace the CANN-matched versions while reinstalling this wheel.
    pip install --force-reinstall --no-deps "$whl_file"
    [[ $? -eq 0 ]] || exit 1
}

function run_py_test()
{
    cd $SCRIPT_DIR/test/
    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" init_test.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" qp_num_test.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" runtime_config_test.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" tls_test.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" unique_id_test.py
    [[ $? -eq 0 ]] || return 1
}

function run_core_py_test()
{
    cd $SCRIPT_DIR/test/core
    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" test_init_final.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" test_memory.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" test_rma.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" test_direct.py
    [[ $? -eq 0 ]] || return 1

    SHMEM_CYCLE_PROF_PE=0 run_torchrun --nproc-per-node "${NPROC_PER_NODE}" test_sync_config_prof.py
    [[ $? -eq 0 ]] || return 1

    run_torchrun --nproc-per-node "${NPROC_PER_NODE}" test_multi_instance.py
    [[ $? -eq 0 ]] || return 1

    # handle_wait targets ROCE completion. The default MTE smoke uses one PE;
    # use the documented direct torchrun command for cross-host ROCE validation.
    run_torchrun --nproc-per-node "${HANDLE_WAIT_NPROC_PER_NODE}" test_handle_wait.py
    [[ $? -eq 0 ]] || return 1
}

function main()
{
    pre_check

    run_py_test
    [[ $? -eq 0 ]] || return 1

    run_core_py_test
    [[ $? -eq 0 ]] || return 1

    echo "All Python tests passed!"
}

main
exit $?
