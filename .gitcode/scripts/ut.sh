#!/bin/bash

set -ex

if [ -f "${WORKSPACE}/.gitcode/scripts/common.sh" ];then
    source ${WORKSPACE}/.gitcode/scripts/common.sh
else
    source ${WORKSPACE}/common.sh
fi
# 检查变更文件是否全部为文档（docs/ 开头 或 .md 结尾）
check_docs_changes ${WORKSPACE}/pr_filelist.txt

# 加载cann 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
unset ASCEND_RT_VISIBLE_DEVICES
# 打印环境信息
env_info

export ACLSHMEM_LOG_TO_STDOUT=1 #开启日志
bash -x scripts/build.sh -uttests
if [ ${task_name} = "UT_part1" ];then
    export GTEST_TOTAL_SHARDS=2
    export GTEST_SHARD_INDEX=0
    bash scripts/run.sh -ranks 2 -ipport 127.0.0.1:$((RANDOM % 60000 + 1024))
else
    export GTEST_TOTAL_SHARDS=2
    export GTEST_SHARD_INDEX=1
    bash scripts/run.sh -ranks 2 -ipport 127.0.0.1:$((RANDOM % 60000 + 1024))
fi
