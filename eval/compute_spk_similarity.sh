#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TEST_SET_DIR="${TEST_SET_DIR:-./datasets/REAL-T/PRIMARY}"
MAPPING_CSV="${MAPPING_CSV:-./datasets/REAL-T/mapping.csv}"
WESPEAKER_LANG="${WESPEAKER_LANG:-en}"
WESPEAKER_PROVIDER="${WESPEAKER_PROVIDER:-auto}"
WESPEAKER_DATASET_LANG_OVERRIDES="${WESPEAKER_DATASET_LANG_OVERRIDES:-AISHELL-4:chs,AliMeeting:chs}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SPK_SIM_PAIR_MODE="${SPK_SIM_PAIR_MODE:-tse_enrol}"

init_eval_common "./output/PRIMARY/bsrnn_vox1"

if [ "$SPK_SIM_PAIR_MODE" != "tse_enrol" ] && [ "$SPK_SIM_PAIR_MODE" != "mixture_enrol" ]; then
    echo "Invalid SPK_SIM_PAIR_MODE=$SPK_SIM_PAIR_MODE (must be tse_enrol or mixture_enrol)."
    exit 1
fi

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (compute & generate CSV), 2 (generate TXT from CSV), or both."
    exit 1
fi

spk_output_names() {
    local base_name="$1"
    local pair_mode="$2"

    local csv_name=""
    local txt_name=""
    if [ "$pair_mode" = "tse_enrol" ]; then
        csv_name="${base_name}_spk_similarity.csv"
        txt_name="${base_name}_spk_similarity_summary.txt"
    else
        csv_name="${base_name}_spk_similarity_${pair_mode}.csv"
        txt_name="${base_name}_spk_similarity_${pair_mode}_summary.txt"
    fi

    if [ -n "${EVAL_METRICS_SUBDIR:-}" ]; then
        csv_name="${EVAL_METRICS_SUBDIR}/${csv_name}"
        txt_name="${EVAL_METRICS_SUBDIR}/${txt_name}"
    fi
    echo "${csv_name}|${txt_name}"
}

run_spk_sim_full() {
    python3 - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("wespeakerruntime") is None:
    print("Dependency missing: wespeakerruntime. Install with: pip install wespeakerruntime")
    sys.exit(1)
PY

    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        BASE_NAME="$(basename "$BASE_DIR")"
        OUTPUT_NAMES="$(spk_output_names "$BASE_NAME" "$SPK_SIM_PAIR_MODE")"
        OUTPUT_CSV_NAME="${OUTPUT_NAMES%%|*}"
        OUTPUT_TXT_NAME="${OUTPUT_NAMES##*|}"

        CMD=(
            python3 "${REAL_T_ROOT}/utils/spk_similarity_eval.py"
            --base_dir "$BASE_DIR"
            --test_set_dir "$TEST_SET_DIR"
            --mapping_csv "$MAPPING_CSV"
            --wespeaker_lang "$WESPEAKER_LANG"
            --provider "$WESPEAKER_PROVIDER"
            --dataset_lang_overrides "$WESPEAKER_DATASET_LANG_OVERRIDES"
            --num_workers "$NUM_WORKERS"
            --pair_mode "$SPK_SIM_PAIR_MODE"
            --output_csv_name "$OUTPUT_CSV_NAME"
            --output_txt_name "$OUTPUT_TXT_NAME"
            --csv_only
        )
        if [ -n "$MAX_SAMPLES" ]; then
            CMD+=(--max_samples "$MAX_SAMPLES")
        fi

        echo "Running spk-sim (mode 1: compute & CSV) for BASE_DIR=$BASE_DIR pair_mode=$SPK_SIM_PAIR_MODE"
        "${CMD[@]}"
    done
}

run_spk_sim_regen_txt() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        BASE_NAME="$(basename "$BASE_DIR")"
        OUTPUT_NAMES="$(spk_output_names "$BASE_NAME" "$SPK_SIM_PAIR_MODE")"
        OUTPUT_CSV_NAME="${OUTPUT_NAMES%%|*}"
        OUTPUT_TXT_NAME="${OUTPUT_NAMES##*|}"

        python3 "${REAL_T_ROOT}/utils/spk_similarity_eval.py" \
            --base_dir "$BASE_DIR" \
            --wespeaker_lang "$WESPEAKER_LANG" \
            --dataset_lang_overrides "$WESPEAKER_DATASET_LANG_OVERRIDES" \
            --num_workers "$NUM_WORKERS" \
            --pair_mode "$SPK_SIM_PAIR_MODE" \
            --output_csv_name "$OUTPUT_CSV_NAME" \
            --output_txt_name "$OUTPUT_TXT_NAME" \
            --regen_txt_only
    done
}

for mode in "${MODES[@]}"; do
    if [ "$mode" = "1" ]; then
        run_spk_sim_full
    elif [ "$mode" = "2" ]; then
        run_spk_sim_regen_txt
    else
        echo "Invalid mode: $mode. Please use 1 (compute & CSV), 2 (generate TXT), or both."
        exit 1
    fi
done

echo "Speaker similarity evaluation finished."
