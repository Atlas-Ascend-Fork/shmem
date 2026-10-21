#!/bin/bash

set -o pipefail

cd ${WORKSPACE}

if [ -f "${WORKSPACE}/.gitcode/scripts/common.sh" ];then
    source ${WORKSPACE}/.gitcode/scripts/common.sh
else
    source ${WORKSPACE}/common.sh
fi
check_docs_changes ${WORKSPACE}/pr_filelist.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_SCRIPT="$SCRIPT_DIR/build.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASSED=0
FAILED=0
TOTAL=0
declare -a FAILED_TESTS

SKIP_HEAVY=false

for arg in "$@"; do
    case "$arg" in
        --skip-heavy) SKIP_HEAVY=true ;;
    esac
done

clean_build_dirs() {
    local build_dir="$PROJECT_ROOT/build"
    local thirdparty_dir="$PROJECT_ROOT/3rdparty"

    if [ -d "$build_dir" ]; then
        echo -e "  ${YELLOW}清理 build 目录: $build_dir${NC}"
        rm -rf "$build_dir"
    fi

    if [ -d "$thirdparty_dir" ]; then
        echo -e "  ${YELLOW}清理 3rdparty 目录: $thirdparty_dir${NC}"
        rm -rf "$thirdparty_dir"
    fi
}

run_test() {
    local name="$1"
    if [ $# -gt 0 ]; then
        shift
    fi
    local options=("$@")

    ((++TOTAL))

    echo ""
    echo -e "${CYAN}===== [$TOTAL] $name =====${NC}"
    echo -e "${CYAN}  build.sh ${options[*]}${NC}"

    cd "$PROJECT_ROOT"

    clean_build_dirs

    # 针对 onlygendoc 特殊处理：先安装并检查 sphinx
    if [[ "$name" == "-onlygendoc" ]]; then
        echo -e "${YELLOW}准备 Sphinx 环境...${NC}"
        export PATH="/opt/buildtools/python-3.10.2/bin:$PATH"

        if ! pip install sphinx; then
            echo -e "${RED}✗ FAIL: pip install sphinx 失败${NC}"
            exit 1
        fi

        echo -e "${CYAN}  pip show sphinx${NC}"
        pip show sphinx || true

        echo -e "${CYAN}  python -m sphinx --version${NC}"
        python -m sphinx --version || true

        echo -e "${CYAN}  sphinx-build --version${NC}"
        sphinx-build --version || true
    fi

    local log_file="/tmp/build_test_${TOTAL}.log"

    if bash "$BUILD_SCRIPT" "${options[@]}" > "$log_file" 2>&1; then
        echo -e "  ${GREEN}✓ PASS${NC}"
        rm -f "$log_file"
        ((++PASSED))
        return 0
    else
        local exit_code=$?
        echo -e "  ${RED}✗ FAIL (exit code: $exit_code)${NC}"
        echo -e "  ${YELLOW}完整日志如下:${NC}"
        echo "----------------------------------------"
        cat "$log_file"
        echo "----------------------------------------"
        ((++FAILED))
        FAILED_TESTS+=("$name")
        exit 1   # 直接中断整个脚本
    fi
}

echo ""
echo "============================================"
echo "  Phase 1: 独立选项测试"
echo "============================================"

run_test "default (无选项)"
run_test "-debug" "-debug"
run_test "-enable_ascendc_dump" "-enable_ascendc_dump"
run_test "-use_cxx11_abi0" "-use_cxx11_abi0"
export branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || git rev-parse HEAD 2> /dev/null || echo "unknown")
run_test "-onlygendoc" "-onlygendoc"

echo ""
echo "============================================"
echo "  Phase 2: 依赖第三方库的选项"
echo "============================================"

run_test "-python_extension" "-python_extension"
run_test "-package" "-package"
run_test "-examples" "-examples"
run_test "-python_example" "-python_example"

echo ""
echo "============================================"
echo "  Phase 3: 组合测试"
echo "============================================"

run_test "-cann -examples" \
    "-cann" "-examples"
run_test "-full" "-full"

echo ""
echo "============================================"
echo "  结果汇总"
echo "============================================"
echo -e "  总计:  $TOTAL"
echo -e "  通过:  ${GREEN}$PASSED${NC}"
echo -e "  失败:  ${RED}$FAILED${NC}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo -e "  ${RED}失败列表:${NC}"
    for t in "${FAILED_TESTS[@]}"; do
        echo -e "    ${RED}✗ $t${NC}"
    done
    exit 1
fi

echo ""
echo -e "${GREEN}全部通过!${NC}"
exit 0
