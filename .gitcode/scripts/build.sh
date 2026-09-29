#!/bin/bash

set -ex

if [ -f "${WORKSPACE}/.gitcode/scripts/common.sh" ];then
    source ${WORKSPACE}/.gitcode/scripts/common.sh
else
    source ${WORKSPACE}/common.sh
fi
check_docs_changes ${WORKSPACE}/pr_filelist.txt

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ACLSHMEM_LOG_TO_STDOUT=1

test_import_shmem() {
    echo "开始 import shmem 测试"

    local package=$(find ./package/ -type f \( -name "shmem*.whl" -o -name "cann_shmem*.whl" \) | head -1)

    if [ -z "$package" ]; then
        echo "FAIL: 未找到 shmem*.whl 文件"
        return 1
    fi

    echo "安装 wheel: $package"
    if ! pip install "$package"; then
        echo "FAIL: pip install 失败"
        return 1
    fi

    echo "测试导入..."
    if python -c "import shmem"; then
        echo "SUCCESS: import shmem 成功"
        return 0
    else
        echo "FAIL: import shmem 出错"
        return 1
    fi
}

if [ "${task_name}" = "Compile_950" ]; then
    bash scripts/build.sh -soc_type Ascend950 -enable_simt -examples -enable_rdma -rdma_backend XSCALE -uttests
else
    env_info

    bash -x scripts/build.sh -package

    echo "执行 import shmem 测试..."
    if ! test_import_shmem; then
        echo "FAIL: import shmem 测试失败，CI 退出"
        exit 1
    fi

    bash -x scripts/build.sh -examples
    bash -x scripts/run_examples.sh
fi
