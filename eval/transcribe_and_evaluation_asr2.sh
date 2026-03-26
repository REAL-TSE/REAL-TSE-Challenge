#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ASR2_SCRIPT="${REAL_T_ROOT}/asr/asr_inference_fireredasr2.py"
EVAL_SCRIPT="${REAL_T_ROOT}/utils/asr_evaluation.py"

CHINESE_ASR2_MODEL="${CHINESE_ASR2_MODEL:-FireRedASR2-AED}"
FIREREDASR2S_ROOT="${FIREREDASR2S_ROOT:-./FireRedASR2S}"
FIREREDASR2_MODEL_DIR="${FIREREDASR2_MODEL_DIR:-${FIREREDASR2S_ROOT}/pretrained_models/FireRedASR2-AED}"
USE_GPU="${USE_GPU:-1}"
USE_HALF="${USE_HALF:-0}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-1}"
ASR_MAX_SAMPLES="${ASR_MAX_SAMPLES:-}"

init_eval_common "./output/PRIMARY/BSRNN"

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (ASR2 for all datasets), 2 (Evaluation -> TER_ASR2_AED), or both."
    exit 1
fi

run_asr2_all_datasets() {
    if [ ! -d "$FIREREDASR2S_ROOT" ]; then
        echo "FireRedASR2S root not found: $FIREREDASR2S_ROOT"
        exit 1
    fi
    if [ ! -d "$FIREREDASR2_MODEL_DIR" ]; then
        echo "FireRedASR2-AED model directory not found: $FIREREDASR2_MODEL_DIR"
        exit 1
    fi

    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        echo "Processing base directory (FireRedASR2-AED): $BASE_DIR"
        mapfile -t dataset_dirs < <(list_dataset_dirs "$BASE_DIR")
        if [ ${#dataset_dirs[@]} -eq 0 ]; then
            echo "No datasets found under $BASE_DIR, skipping."
            continue
        fi

        for dataset_path in "${dataset_dirs[@]}"; do
            dataset="$(basename "$dataset_path")"
            if ! dataset_enabled "$dataset"; then
                echo "Dataset $dataset not in DATASETS list, skipping."
                continue
            fi

            TSE_MAPPING_CSV="${dataset_path}/${MAPPING_CSV_NAME}"
            if [ ! -f "$TSE_MAPPING_CSV" ]; then
                echo "Mapping not found: $TSE_MAPPING_CSV, skipping dataset $dataset."
                continue
            fi

            PREDICTED_DIR="${dataset_path}/${CHINESE_ASR2_MODEL}"
            mkdir -p "$PREDICTED_DIR"

            CMD=(
                python3 "$ASR2_SCRIPT"
                --audio_mapping "$TSE_MAPPING_CSV"
                --output_dir "$PREDICTED_DIR"
                --dataset_name "$dataset"
                --fireredasr2s_root "$FIREREDASR2S_ROOT"
                --model_dir "$FIREREDASR2_MODEL_DIR"
                --use_gpu "$USE_GPU"
                --use_half "$USE_HALF"
                --batch_size "$ASR_BATCH_SIZE"
            )
            if [ -n "$ASR_MAX_SAMPLES" ]; then
                CMD+=(--max_samples "$ASR_MAX_SAMPLES")
            fi

            echo "Running FireRedASR2-AED for dataset: $dataset"
            echo "  mapping: $TSE_MAPPING_CSV"
            echo "  output : $PREDICTED_DIR"
            "${CMD[@]}"
        done
    done
    echo "FireRedASR2-AED ASR completed!"
}

run_evaluation_ter_asr2() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        BASE_NAME="$(basename "$BASE_DIR")"
        RESULT_TXT="${BASE_DIR}/${BASE_NAME}_TER_ASR2_AED.txt"
        RESULT_CSV="${BASE_DIR}/${BASE_NAME}_TER_ASR2_AED.csv"

        python3 "$EVAL_SCRIPT" \
            --ground_truth_dir "$TEST_SET_DIR" \
            --save_path "$RESULT_CSV" \
            --predicted_dir "$BASE_DIR" \
            --chinese_asr_model "$CHINESE_ASR2_MODEL" \
            --english_asr_model "$CHINESE_ASR2_MODEL" \
            > "$RESULT_TXT"

        echo "Evaluation (TER_ASR2_AED) completed:"
        echo "  detail : $RESULT_CSV"
        echo "  report : $RESULT_TXT"
    done
}

for mode in "${MODES[@]}"; do
    if [ "$mode" = "1" ]; then
        run_asr2_all_datasets
    elif [ "$mode" = "2" ]; then
        run_evaluation_ter_asr2
    else
        echo "Invalid mode: $mode. Please use 1 (ASR2 for all datasets), 2 (Evaluation -> TER_ASR2_AED), or both."
        exit 1
    fi
done

echo "All tasks finished successfully!"
