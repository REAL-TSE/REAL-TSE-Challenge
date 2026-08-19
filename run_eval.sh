#!/bin/bash

set -euo pipefail

ORIG_CWD="$(pwd)"
REAL_T_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REAL_T_ROOT}/env_setup.sh"
source "${REAL_T_ROOT}/utils/dataset_paths.sh"

usage() {
    cat <<'EOF'
Usage:
  bash ./run_eval.sh --output-dir <path> --test-set <DEV|EVAL1|EVAL2> --cuda <id> \
                     [--dataset-root <path>] [--chinese-asr <name>] [--english-asr <name>] [1] [2]

Modes:
  1    Run all evaluation sub-scripts
  2    Aggregate existing CSV results into <output_name>_summary.txt

If no mode is provided, the default is: 1 2

Local evaluation is available for DEV, EVAL1, and EVAL2. Each split uses
the same pipeline and expects the same ground-truth layout as DEV
(meta CSVs, transcripts, and overlap JSON).

ASR backend selection (optional):
  --chinese-asr     ASR model for Chinese datasets (AISHELL-4, AliMeeting, unseen_CN).
                    Default: zipformer-zh.
                    Other supported: FireRedASR-AED-L
  --english-asr     ASR model for English datasets (AMI, CHiME6, DipCo, unseen_EN).
                    Default: zipformer-en.
                    Other supported: whisper-large-v2

Note: the reported TER metrics in the README use the default Zipformer
pair (zipformer-zh / zipformer-en). Whisper / FireRedASR remain available
behind these flags for reproducing the historical numbers, but they can
hallucinate (repeated n-grams / loops) on long real-world conversation
audio, which is why Zipformer is the default for the final metric.

Note: before the metric stages, a pre-scan lists dataset dirs under
--output-dir that lack a language mapping (they are skipped by SPK_SIM /
DNSMOS / TSE_TIMING). Set DATASET_LANG_STRICT=1 to abort on them instead.
EOF
}

OUTPUT_DIR=""
TEST_SET=""
CUDA_ID=""
DATASET_ROOT=""
CHINESE_ASR_OVERRIDE=""
ENGLISH_ASR_OVERRIDE=""
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
        --dataset-root)
            DATASET_ROOT="${2:-}"
            shift 2
            ;;
        --chinese-asr)
            CHINESE_ASR_OVERRIDE="${2:-}"
            shift 2
            ;;
        --english-asr)
            ENGLISH_ASR_OVERRIDE="${2:-}"
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

validate_test_set "$TEST_SET" || exit 1

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory not found: $OUTPUT_DIR"
    exit 1
fi

resolve_dataset_paths "$TEST_SET" "$DATASET_ROOT"
verify_test_set_dir "$TEST_SET_DIR" "$DATASET_ROOT" || exit 1

if [[ "$TEST_SET_DIR" != /* ]]; then
    TEST_SET_DIR="$(cd "$ORIG_CWD" && cd "$TEST_SET_DIR" && pwd)"
fi
if [[ "$DATASET_ROOT" != /* ]]; then
    DATASET_ROOT="$(cd "$ORIG_CWD" && cd "$DATASET_ROOT" && pwd)"
fi
MAPPING_CSV="${DATASET_ROOT}/mapping.csv"
verify_eval_assets "$TEST_SET_DIR" "$MAPPING_CSV" || exit 1

export CUDA_VISIBLE_DEVICES="$CUDA_ID"
export OUTPUT_DIRS="$OUTPUT_DIR"
export TEST_SET_DIR
export MAPPING_CSV
export USE_GPU=1
export ASR_DEVICE="cuda:0"
export WESPEAKER_PROVIDER="cuda"
export DNSMOS_PROVIDER="cuda"
export EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR:-eval_metrics}"

# Forward optional ASR backend overrides to transcribe_and_evaluation.sh
# via the env-vars it already reads. Defaults (when neither flag nor env is
# set) come from transcribe_and_evaluation.sh itself: zipformer-{zh,en}.
if [ -n "$CHINESE_ASR_OVERRIDE" ]; then
    export CHINESE_ASR_MODEL="$CHINESE_ASR_OVERRIDE"
fi
if [ -n "$ENGLISH_ASR_OVERRIDE" ]; then
    export ENGLISH_ASR_MODEL="$ENGLISH_ASR_OVERRIDE"
fi

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
    echo "  test_set   : $TEST_SET"
    echo "  dataset    : $TEST_SET_DIR"
    echo "  cuda       : $CUDA_VISIBLE_DEVICES"
    echo "  metrics  : ${EVAL_METRICS_SUBDIR:-.}"
    echo "  ASR (zh) : ${CHINESE_ASR_MODEL:-zipformer-zh (default)}"
    echo "  ASR (en) : ${ENGLISH_ASR_MODEL:-zipformer-en (default)}"

    PRE_SCAN_CMD=(python3 "${REAL_T_ROOT}/utils/dataset_lang.py" scan "$OUTPUT_DIR")
    if [ -n "${WESPEAKER_DATASET_LANG_OVERRIDES:-}" ]; then
        PRE_SCAN_CMD+=(--overrides "$WESPEAKER_DATASET_LANG_OVERRIDES")
    fi
    if [ -n "${DNSMOS_DATASET_LANG_OVERRIDES:-}" ]; then
        PRE_SCAN_CMD+=(--overrides "$DNSMOS_DATASET_LANG_OVERRIDES")
    fi
    if [ "${DATASET_LANG_STRICT:-0}" = "1" ]; then
        PRE_SCAN_CMD+=(--strict)
    fi
    run_stage "DATASET_LANG_PRESCAN" "${PRE_SCAN_CMD[@]}"

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
