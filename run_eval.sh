#!/bin/bash

set -euo pipefail

ORIG_CWD="$(pwd)"
REAL_T_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REAL_T_ROOT}/env_setup.sh"

usage() {
    cat <<'EOF'
Usage:
  bash ./run_eval.sh --output-dir <path> --test-set <DEV|EVAL> --cuda <id> [1] [2]

Modes:
  1    Run all evaluation sub-scripts
  2    Aggregate existing CSV results into <output_name>_summary.txt

If no mode is provided, the default is: 1 2
EOF
}

OUTPUT_DIR=""
TEST_SET=""
CUDA_ID=""
MODES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --test-set)
            TEST_SET="${2:-}"
            shift 2
            ;;
        --cuda)
            CUDA_ID="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            MODES+=("$1")
            shift
            ;;
    esac
done

if [ ${#MODES[@]} -eq 0 ]; then
    MODES=(1 2)
fi

for mode in "${MODES[@]}"; do
    if [ "$mode" != "1" ] && [ "$mode" != "2" ]; then
        echo "Invalid mode: $mode"
        usage
        exit 1
    fi
done

if [ -z "$OUTPUT_DIR" ] || [ -z "$TEST_SET" ] || [ -z "$CUDA_ID" ]; then
    usage
    exit 1
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(cd "$ORIG_CWD" && cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"
fi

if [ "$TEST_SET" != "EVAL" ] && [ "$TEST_SET" != "DEV" ]; then
    echo "--test-set must be DEV or EVAL."
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory not found: $OUTPUT_DIR"
    exit 1
fi

DATASET_ROOT="./datasets/REAL-T-$(echo "$TEST_SET" | tr '[:upper:]' '[:lower:]')"
TEST_SET_DIR="${DATASET_ROOT}/${TEST_SET}"
if [ ! -d "$TEST_SET_DIR" ]; then
    echo "Test set directory not found: $TEST_SET_DIR"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_ID"
export OUTPUT_DIRS="$OUTPUT_DIR"
export TEST_SET_DIR
export MAPPING_CSV="${DATASET_ROOT}/mapping.csv"
export USE_GPU=1
export ASR_DEVICE="cuda:0"
export WESPEAKER_PROVIDER="cuda"
export DNSMOS_PROVIDER="cuda"
export EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR:-eval_metrics}"

run_stage() {
    local label="$1"
    shift
    echo
    echo "===== $label ====="
    "$@"
}

run_pipeline() {
    echo "Running full eval pipeline"
    echo "  output_dir : $OUTPUT_DIR"
    echo "  test_set : $TEST_SET"
    echo "  cuda     : $CUDA_VISIBLE_DEVICES"
    echo "  metrics  : ${EVAL_METRICS_SUBDIR:-.}"

    run_stage "TER" bash "${REAL_T_ROOT}/eval/transcribe_and_evaluation.sh" 1 2
    run_stage "TSE_TIMING" bash "${REAL_T_ROOT}/eval/vad_and_evaluation.sh" 1 2
    run_stage "SPK_SIM_TSE_ENROL" bash "${REAL_T_ROOT}/eval/compute_spk_similarity.sh" 1 2
    run_stage "SPK_SIM_MIXTURE_ENROL" env SPK_SIM_PAIR_MODE=mixture_enrol bash "${REAL_T_ROOT}/eval/compute_spk_similarity.sh" 1 2
    run_stage "DNSMOS" bash "${REAL_T_ROOT}/eval/compute_dnsmos.sh" 1 2

    echo
    echo "Full eval pipeline completed successfully."
}

run_summary() {
    echo
    echo "===== AGGREGATED SUMMARY ====="
    python3 "${REAL_T_ROOT}/utils/aggregate_eval_summary.py" \
        --output_dir "$OUTPUT_DIR" \
        --metrics_subdir "${EVAL_METRICS_SUBDIR:-}"
    echo "Aggregated summary completed successfully."
}

for mode in "${MODES[@]}"; do
    if [ "$mode" = "1" ]; then
        run_pipeline
    elif [ "$mode" = "2" ]; then
        run_summary
    else
        echo "Unexpected mode: $mode"
        exit 1
    fi
done
