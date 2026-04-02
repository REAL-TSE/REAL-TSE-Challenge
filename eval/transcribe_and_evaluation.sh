#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ASR_SCRIPT="${REAL_T_ROOT}/asr/asr_inference.py"
EVAL_SCRIPT="${REAL_T_ROOT}/utils/asr_evaluation.py"

CHINESE_ASR_MODEL="${CHINESE_ASR_MODEL:-FireRedASR-AED-L}"
ENGLISH_ASR_MODEL="${ENGLISH_ASR_MODEL:-whisper-large-v2}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
ASR_MAX_SAMPLES="${ASR_MAX_SAMPLES:-}"

init_eval_common

CHINESE_DATASETS="${CHINESE_DATASETS:-$(filter_datasets "$KNOWN_CHINESE_DATASETS" "$DATASETS")}"
ENGLISH_DATASETS="${ENGLISH_DATASETS:-$(filter_datasets "$KNOWN_ENGLISH_DATASETS" "$DATASETS")}"

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (ASR), 2 (Evaluation), or both."
    exit 1
fi

run_asr() {
    for OUTPUT_DIR in "${OUTPUT_DIR_LIST[@]}"; do
        echo "Processing output directory: $OUTPUT_DIR"
        mapfile -t dataset_dirs < <(list_dataset_dirs "$OUTPUT_DIR")
        if [ ${#dataset_dirs[@]} -eq 0 ]; then
            echo "No datasets found under $OUTPUT_DIR, skipping."
            continue
        fi

        for dataset_path in "${dataset_dirs[@]}"; do
            dataset="$(basename "$dataset_path")"
            if ! dataset_enabled "$dataset"; then
                echo "Dataset $dataset not in DATASETS list, skipping."
                continue
            fi

            ASR_MODEL_NAME=""
            if [[ " ${CHINESE_DATASETS} " == *" ${dataset} "* ]]; then
                ASR_MODEL_NAME="$CHINESE_ASR_MODEL"
            elif [[ " ${ENGLISH_DATASETS} " == *" ${dataset} "* ]]; then
                ASR_MODEL_NAME="$ENGLISH_ASR_MODEL"
            fi

            if [ -z "$ASR_MODEL_NAME" ]; then
                echo "Dataset $dataset is not in CHINESE_DATASETS or ENGLISH_DATASETS. Skipping."
                continue
            fi

            TSE_MAPPING_CSV="${dataset_path}/${MAPPING_CSV_NAME}"
            if [ ! -f "$TSE_MAPPING_CSV" ]; then
                echo "Mapping not found: $TSE_MAPPING_CSV, skipping dataset $dataset."
                continue
            fi

            PREDICTED_DIR="${dataset_path}/${ASR_MODEL_NAME}"
            mkdir -p "$PREDICTED_DIR"

            CMD=(
                python3 "$ASR_SCRIPT"
                --audio_mapping "$TSE_MAPPING_CSV"
                --model_name "$ASR_MODEL_NAME"
                --dataset_name "$dataset"
                --output_dir "$PREDICTED_DIR"
                --device "$ASR_DEVICE"
            )
            if [ -n "$ASR_MAX_SAMPLES" ]; then
                CMD+=(--max_samples "$ASR_MAX_SAMPLES")
            fi

            echo "Running ASR for dataset: $dataset"
            echo "  mapping: $TSE_MAPPING_CSV"
            echo "  output : $PREDICTED_DIR"
            "${CMD[@]}"
        done
    done
    echo "ASR processing completed!"
}

run_asr_evaluation() {
    for OUTPUT_DIR in "${OUTPUT_DIR_LIST[@]}"; do
        echo "Running TER evaluation for output directory: $OUTPUT_DIR"
        OUTPUT_NAME="$(basename "$OUTPUT_DIR")"
        METRICS_DIR="$(eval_metrics_dir "$OUTPUT_DIR")"
        mkdir -p "$METRICS_DIR"

        RESULT_TXT="${METRICS_DIR}/${OUTPUT_NAME}_TER.txt"
        RESULT_CSV="${METRICS_DIR}/${OUTPUT_NAME}_TER.csv"

        CMD=(
            env
            HF_HUB_OFFLINE=1
            TRANSFORMERS_OFFLINE=1
            python3 "$EVAL_SCRIPT"
            --ground_truth_dir "$TEST_SET_DIR"
            --save_path "$RESULT_CSV"
            --predicted_dir "$OUTPUT_DIR"
            --chinese_asr_model "$CHINESE_ASR_MODEL"
            --english_asr_model "$ENGLISH_ASR_MODEL"
        )

        "${CMD[@]}" > "$RESULT_TXT"
        echo "Evaluation completed:"
        echo "  detail : $RESULT_CSV"
        echo "  report : $RESULT_TXT"
    done
}

for mode in "${MODES[@]}"; do
    if [ "$mode" = "1" ]; then
        run_asr
    elif [ "$mode" = "2" ]; then
        run_asr_evaluation
    else
        echo "Invalid mode: $mode. Please use 1 (ASR), 2 (Evaluation), or both."
        exit 1
    fi
done

echo "All tasks finished successfully!"
