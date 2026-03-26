#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

VAD_SCRIPT="${REAL_T_ROOT}/utils/vad_inference_firered.py"
PREPARE_LABEL_SCRIPT="${REAL_T_ROOT}/utils/prepare_label_segments.py"
EVAL_SCRIPT="${REAL_T_ROOT}/utils/tse_timing_evaluation.py"
VIS_SCRIPT="${REAL_T_ROOT}/utils/plot_tse_timeline.py"

GT_JSON_BASE_DIR="${GT_JSON_BASE_DIR:-./datasets/REAL-T/json}"
METADATA_DIR="${METADATA_DIR:-./datasets/REAL-T/metadata}"
FIREREDASR2S_ROOT="${FIREREDASR2S_ROOT:-./FireRedASR2S}"
FIRERED_VAD_MODEL_DIR="${FIRERED_VAD_MODEL_DIR:-${FIREREDASR2S_ROOT}/pretrained_models/FireRedVAD/VAD}"
VAD_DIR_NAME="${VAD_DIR_NAME:-FireRedVAD}"
VAD_JSONL_NAME="${VAD_JSONL_NAME:-vad_segments.jsonl}"
SPEECH_THRESHOLD="${SPEECH_THRESHOLD:-0.5}"
FRAME_SHIFT="${FRAME_SHIFT:-0.01}"
COLLAR="${COLLAR:-0.05}"
MATCH_TOLERANCE="${MATCH_TOLERANCE:-0.02}"

init_eval_common "./output/PRIMARY/bsrnn_vox1"

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (VAD), 2 (Timing Evaluation), 3 (Visualization), or combination."
    exit 1
fi

run_vad() {
    if [ ! -d "$FIREREDASR2S_ROOT" ]; then
        echo "FireRedASR2S root not found: $FIREREDASR2S_ROOT"
        exit 1
    fi
    if [ ! -d "$FIRERED_VAD_MODEL_DIR" ]; then
        echo "FireRedVAD model directory not found: $FIRERED_VAD_MODEL_DIR"
        exit 1
    fi

    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        echo "Running VAD for base directory: $BASE_DIR"
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

            mapping_csv="${dataset_path}/${MAPPING_CSV_NAME}"
            if [ ! -f "$mapping_csv" ]; then
                echo "Mapping not found: $mapping_csv, skipping dataset $dataset."
                continue
            fi

            vad_dir="${dataset_path}/${VAD_DIR_NAME}"
            vad_jsonl="${vad_dir}/${VAD_JSONL_NAME}"
            mkdir -p "$vad_dir"

            python3 "$VAD_SCRIPT" \
                --audio_mapping "$mapping_csv" \
                --output_jsonl "$vad_jsonl" \
                --model_dir "$FIRERED_VAD_MODEL_DIR" \
                --fireredasr2s_root "$FIREREDASR2S_ROOT" \
                --speech_threshold "$SPEECH_THRESHOLD" \
                --use_gpu "$USE_GPU"
        done
    done
    echo "VAD processing completed!"
}

run_timing_evaluation() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        echo "Running timing evaluation for base directory: $BASE_DIR"
        BASE_NAME="$(basename "$BASE_DIR")"
        RESULT_CSV="${BASE_DIR}/${BASE_NAME}_TSE_TIMING.csv"
        RESULT_TXT="${BASE_DIR}/${BASE_NAME}_TSE_TIMING.txt"

        python3 "$PREPARE_LABEL_SCRIPT" \
            --ground_truth_dir "$TEST_SET_DIR" \
            --metadata_dir "$METADATA_DIR" \
            --gt_json_base_dir "$GT_JSON_BASE_DIR" \
            --predicted_dir "$BASE_DIR" \
            --datasets "$DATASETS" \
            --vad_dir_name "$VAD_DIR_NAME" \
            --collar "$COLLAR" \
            --match_tolerance "$MATCH_TOLERANCE"

        python3 "$EVAL_SCRIPT" \
            --ground_truth_dir "$TEST_SET_DIR" \
            --predicted_dir "$BASE_DIR" \
            --gt_json_base_dir "$GT_JSON_BASE_DIR" \
            --datasets "$DATASETS" \
            --vad_dir_name "$VAD_DIR_NAME" \
            --vad_jsonl_name "$VAD_JSONL_NAME" \
            --frame_shift "$FRAME_SHIFT" \
            --collar "$COLLAR" \
            --match_tolerance "$MATCH_TOLERANCE" \
            --save_path "$RESULT_CSV" \
            --report_path "$RESULT_TXT"

        echo "Timing evaluation completed:"
        echo "  detail : $RESULT_CSV"
        echo "  report : $RESULT_TXT"
    done
}

run_visualization() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        echo "Running visualization for base directory: $BASE_DIR"
        mapfile -t dataset_dirs < <(list_dataset_dirs "$BASE_DIR")

        for dataset_path in "${dataset_dirs[@]}"; do
            dataset="$(basename "$dataset_path")"
            if ! dataset_enabled "$dataset"; then
                continue
            fi

            vad_dir="${dataset_path}/${VAD_DIR_NAME}"
            label_jsonl="${vad_dir}/label_segments.jsonl"
            vad_jsonl="${vad_dir}/${VAD_JSONL_NAME}"
            output_dir="${vad_dir}/figures"
            BASE_NAME="$(basename "$BASE_DIR")"
            METRICS_CSV="${BASE_DIR}/${BASE_NAME}_TSE_TIMING.csv"

            if [ ! -f "$label_jsonl" ]; then
                echo "Skipping $dataset: missing label segments ($label_jsonl)"
                continue
            fi
            if [ ! -f "$vad_jsonl" ]; then
                echo "Skipping $dataset: missing VAD segments ($vad_jsonl)"
                continue
            fi

            mkdir -p "$output_dir"
            if [ ! -f "$METRICS_CSV" ]; then
                python3 "$VIS_SCRIPT" \
                    --label_jsonl "$label_jsonl" \
                    --vad_jsonl "$vad_jsonl" \
                    --output_dir "$output_dir"
            else
                python3 "$VIS_SCRIPT" \
                    --label_jsonl "$label_jsonl" \
                    --vad_jsonl "$vad_jsonl" \
                    --output_dir "$output_dir" \
                    --metrics_csv "$METRICS_CSV"
            fi
        done
    done
    echo "Visualization completed!"
}

for mode in "${MODES[@]}"; do
    if [ "$mode" = "1" ]; then
        run_vad
    elif [ "$mode" = "2" ]; then
        run_timing_evaluation
    elif [ "$mode" = "3" ]; then
        run_visualization
    else
        echo "Invalid mode: $mode. Please use 1 (VAD), 2 (Timing Evaluation), or 3 (Visualization)."
        exit 1
    fi
done
