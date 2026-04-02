#!/bin/bash

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_T_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

source "${REAL_T_ROOT}/env_setup.sh"

# --- Dataset language mapping (used by auto-filtering helpers) ---
KNOWN_CHINESE_DATASETS="AliMeeting AISHELL-4"
KNOWN_ENGLISH_DATASETS="AMI DipCo CHiME6"

auto_detect_datasets() {
    local dir="$1"
    local result=""
    for meta in "$dir"/*_meta.csv; do
        [ -f "$meta" ] || continue
        local name
        name="$(basename "$meta" _meta.csv)"
        result="${result:+$result }$name"
    done
    echo "$result"
}

filter_datasets() {
    local candidates="$1"
    local allowed="$2"
    local result=""
    for d in $candidates; do
        if [[ " $allowed " == *" $d "* ]]; then
            result="${result:+$result }$d"
        fi
    done
    echo "$result"
}

init_eval_common() {
    local default_output_dirs="${1:-}"

    TEST_SET_DIR="${TEST_SET_DIR:-./datasets/REAL-T-eval/EVAL}"
    MAPPING_CSV_NAME="${MAPPING_CSV_NAME:-tse_audio_mapping.csv}"
    EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR:-eval_metrics}"
    USE_GPU="${USE_GPU:-1}"

    if [ -z "${DATASETS:-}" ] && [ -d "$TEST_SET_DIR" ]; then
        DATASETS="$(auto_detect_datasets "$TEST_SET_DIR")"
        if [ -n "$DATASETS" ]; then
            echo "[auto-detect] DATASETS from ${TEST_SET_DIR}: ${DATASETS}"
        fi
    fi
    DATASETS="${DATASETS:-AliMeeting AISHELL-4 AMI DipCo CHiME6}"

    EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR#/}"
    EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR%/}"
    if [ "$EVAL_METRICS_SUBDIR" = "." ]; then
        EVAL_METRICS_SUBDIR=""
    fi

    if [ -z "${OUTPUT_DIRS:-}" ]; then
        OUTPUT_DIRS="${default_output_dirs}"
    fi

    read -r -a OUTPUT_DIR_LIST <<< "${OUTPUT_DIRS:-}"
    if [ "${#OUTPUT_DIR_LIST[@]}" -eq 0 ]; then
        echo "No OUTPUT_DIRS provided."
        exit 1
    fi

    local normalized_output_dirs=()
    local output_dir=""
    for output_dir in "${OUTPUT_DIR_LIST[@]}"; do
        if [ ! -d "$output_dir" ]; then
            echo "Output directory not found: $output_dir"
            exit 1
        fi
        normalized_output_dirs+=("$(cd "$output_dir" && pwd -P)")
    done
    OUTPUT_DIR_LIST=("${normalized_output_dirs[@]}")

    read -r -a DATASET_LIST <<< "${DATASETS}"
    if [ "${#DATASET_LIST[@]}" -eq 0 ]; then
        echo "No DATASETS provided."
        exit 1
    fi
}

dataset_enabled() {
    local dataset="$1"
    [[ " ${DATASET_LIST[*]} " == *" ${dataset} "* ]]
}

list_dataset_dirs() {
    local output_dir="$1"
    find -L "$output_dir" -maxdepth 1 -mindepth 1 -type d | sort
}

eval_metrics_dir() {
    local output_dir="$1"
    if [ -z "${EVAL_METRICS_SUBDIR:-}" ]; then
        echo "$output_dir"
        return
    fi
    echo "${output_dir}/${EVAL_METRICS_SUBDIR}"
}
