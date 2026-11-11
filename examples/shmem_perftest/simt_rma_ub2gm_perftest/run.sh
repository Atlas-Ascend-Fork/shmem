#!/bin/bash

# Default parameters
NPES=2
GNPUS=2
FIRST_NPU=0
FIRST_PE=0
IPPORT="tcp://127.0.0.1:8760"
BLOCK_SIZE=32
LOOP_COUNT=1000
EXP_MIN=3
EXP_MAX=16

# Reject an option whose value is missing or is itself an option, so a typo fails
# with a clear message instead of silently consuming the next flag as a value.
require_value() {
    local opt="$1" val="${2-}"
    if [ -z "$val" ] || [ "${val:0:1}" = "-" ]; then
        echo "[ERROR] Option $opt requires a value." >&2
        echo "Run 'bash run.sh --help' for the list of supported options." >&2
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -pes)
            require_value "$1" "${2-}"
            NPES="$2"
            shift 2
            ;;
        -gnpus)
            require_value "$1" "${2-}"
            GNPUS="$2"
            shift 2
            ;;
        -fnpu)
            require_value "$1" "${2-}"
            FIRST_NPU="$2"
            shift 2
            ;;
        -fpe)
            require_value "$1" "${2-}"
            FIRST_PE="$2"
            shift 2
            ;;
        -ipport)
            require_value "$1" "${2-}"
            IPPORT="$2"
            shift 2
            ;;
        -b|--block-size)
            require_value "$1" "${2-}"
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --block-range)
            require_value "$1" "${2-}"
            require_value "$1" "${3-}"
            BLOCK_RANGE_MIN="$2"
            BLOCK_RANGE_MAX="$3"
            shift 3
            ;;
        --block-list)
            require_value "$1" "${2-}"
            BLOCK_LIST="$2"
            shift 2
            ;;
        --loop-count)
            require_value "$1" "${2-}"
            LOOP_COUNT="$2"
            shift 2
            ;;
        -e|--exponent)
            require_value "$1" "${2-}"
            EXP_MIN="$2"
            EXP_MAX="$2"
            shift 2
            ;;
        --exponent-range)
            require_value "$1" "${2-}"
            require_value "$1" "${3-}"
            EXP_MIN="$2"
            EXP_MAX="$3"
            shift 3
            ;;
        -t|--test-type)
            require_value "$1" "${2-}"
            TEST_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: run.sh [options]

Launches a fixed 2-card (Active PE0 / Passive PE1) ub2gm RMA perf test.

Options:
  -pes <int>                Number of PEs, must be 2.                    (default: 2)
  -ipport <ip:port>         Bootstrap communication address.             (default: tcp://127.0.0.1:8760)
  -gnpus <int>              Number of NPUs used on this node, must be 2. (default: 2)
  -fnpu <int>               First NPU id; device = pe_id % gnpus + fnpu. (default: 0)
  -fpe <int>                First PE id. Kept for CLI compatibility; unused. (default: 0)
  -t|--test-type <put|get>  Operation to measure: put or get.        (default: get)
  -b|--block-size <int>     Cores (blocks) used per PE.                  (default: 32)
  --block-range <min> <max> Cores (blocks) sweep range, one CSV row per count. (default: 32 32)
  --block-list <b1,b2,...>  Explicit core counts, comma-separated (e.g. 1,8,16).
                            Overrides -b/--block-size and --block-range.
  --loop-count <int>        Sampled iterations, averaged for results.    (default: 1000)
  -e|--exponent <int>       Single transfer-size exponent; size = 2^e bytes.
  --exponent-range <min> <max>  Transfer-size exponent range; size = 2^exp bytes. (default: 3 16)
  -h|--help                 Show this message.

The test sweeps 2^min .. 2^max bytes, one CSV row per (core count, size) pair.
Transfer sizes are capped by the UB buffer, so exponents must stay within [3, 16].
EOF
            exit 0
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            echo "Run 'bash run.sh --help' for the list of supported options." >&2
            exit 1
            ;;
        *)
            echo "[ERROR] Unexpected positional argument: '$1'" >&2
            echo "This script takes options only; values must follow their option name" >&2
            echo "(e.g. 'bash run.sh -b 4 --exponent-range 8 12')." >&2
            echo "Run 'bash run.sh --help' for the list of supported options." >&2
            exit 1
            ;;
    esac
done

# Check the integer shape first: a non-numeric value makes the -ne tests below error
# out and fall through instead of reporting the bad argument.
if ! [[ "$NPES" =~ ^[0-9]+$ && "$GNPUS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] -pes and -gnpus must be non-negative integers." >&2
    exit 1
fi
if [ "$NPES" -ne 2 ] || [ "$GNPUS" -ne 2 ]; then
    echo "[ERROR] This test requires exactly 2 PEs and 2 NPUs." >&2
    exit 1
fi

# Environment setup
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export ACLSHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

cd "${SCRIPT_DIR}"

# Build arguments
ARGS="-pes $NPES -gnpus $GNPUS -fnpu $FIRST_NPU -fpe $FIRST_PE -ipport $IPPORT"
ARGS="$ARGS --loop-count $LOOP_COUNT --exponent-range $EXP_MIN $EXP_MAX"

# --block-list wins over --block-range and -b/--block-size, matching the binary's
# own precedence (see parse_args in argparser.h).
if [ -n "$BLOCK_LIST" ]; then
    ARGS="$ARGS --block-list $BLOCK_LIST"
elif [ -n "$BLOCK_RANGE_MIN" ] && [ -n "$BLOCK_RANGE_MAX" ]; then
    ARGS="$ARGS --block-range $BLOCK_RANGE_MIN $BLOCK_RANGE_MAX"
else
    ARGS="$ARGS -b $BLOCK_SIZE"
fi

if [ -n "$TEST_TYPE" ]; then
    ARGS="$ARGS -t $TEST_TYPE"
fi

# Find binary
BINARY="${PROJECT_ROOT}/build/bin/simt_rma_ub2gm_perftest"

if [ ! -f "$BINARY" ]; then
    echo "[ERROR] Binary not found. Please compile first with:" >&2
    echo "  bash scripts/build.sh -examples -enable_simt -soc_type Ascend950" >&2
    exit 1
fi

# CSV output dir (see main.cpp). Created only after the checks above pass, so a
# failed run leaves no empty directory behind.
mkdir -p output

# Launch processes. PE ranks are always 0..NPES-1: the binary only accepts rank 0
# (ACTIVE) or 1 (PASSIVE), so -fpe must not shift them. It is forwarded for CLI
# parity with the other shmem_perftest samples but does not affect the rank or the
# device id (device = pe_id % gnpus + fnpu).
echo "[INFO] Launching $NPES processes..."
for ((i=0; i<$NPES; i++)); do
    $BINARY $ARGS --pe-id $i &
    PIDS[$i]=$!
done

# Wait for all processes
echo "[INFO] Waiting for processes to complete..."
EXIT_CODE=0
for ((i=0; i<$NPES; i++)); do
    wait ${PIDS[$i]}
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo "[ERROR] PE $i exited with code $RESULT"
        EXIT_CODE=$RESULT
    fi
done

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Test completed successfully."
    echo "[INFO] Results saved in output/"
else
    echo "[FAILURE] Test failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
