#!/bin/bash

set -ex

function check_docs_changes() {
    # ========== 从 pr_filelist.txt 读取变更文件，仅文档（docs/ 或 .md）时跳过 ==========
    local file_list="${1:-pr_filelist.txt}"
    if [ ! -f "${file_list}" ]; then
        echo "pr_filelist.txt not found, skip doc-only check."
        return 0
    fi
    CHANGED_FILES=$(grep -v '^\s*$' "${file_list}" | grep -v '^\s*#' || true)
    if [ -z "${CHANGED_FILES}" ]; then
        echo "pr_filelist.txt is empty, skip doc-only check."
        return 0
    fi
    ONLY_DOCS=true
    while IFS= read -r file; do
        [ -z "${file}" ] && continue
        if [[ ! "${file}" =~ ^docs/ ]] && [[ ! "${file}" =~ \.md$ ]]; then
            ONLY_DOCS=false
            break
        fi
    done <<< "${CHANGED_FILES}"
    if [ "${ONLY_DOCS}" = true ]; then
        echo "----------------------------------------"
        echo "Only documentation files (docs/ or .md) changed. Skipping Build and UT."
        echo "----------------------------------------"
        exit 0
    fi
    # ========== 检测结束 ==========
}

function env_info() {
    echo "=======================================================/usr/local/Ascend/==============================="
    ls -la /usr/local/Ascend/ || echo "no /usr/local/Ascend/"

    echo "=======================================================/dev/davinci====================================="
    ls -la /dev/davinci* 2>/dev/null || echo "no /dev/davinci* devices"

    echo "=======================================================driver info====================================="
    cat /usr/local/Ascend/driver/version.info 2>/dev/null || echo "no /usr/local/Ascend/driver/version.info"

    npu-smi info 2>/dev/null || echo "npu-smi not available"
}
