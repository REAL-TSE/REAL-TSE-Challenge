#!/bin/bash

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_T_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

source "${REAL_T_ROOT}/env_setup.sh"

# --- Dataset language mapping (used by auto-filtering helpers) ---
# EVAL2 uses language-tagged virtual datasets unseen_CN / unseen_EN.
KNOWN_CHINESE_DATASETS="AliMeeting AISHELL-4 unseen_CN"
KNOWN_ENGLISH_DATASETS="AMI DipCo CHiME6 unseen_EN"
# Shared chs bucket for WeSpeaker / DNSMOS per-language stats on all splits.
DEFAULT_CHS_DATASET_LANG_OVERRIDES="AISHELL-4:chs,AliMeeting:chs,unseen_CN:chs"

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

    # Standalone default is DEV; run_eval.sh exports TEST_SET_DIR for every split.
    TEST_SET_DIR="${TEST_SET_DIR:-./datasets/REAL-T-dev/DEV}"
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

    if [ -z "${MAPPING_CSV:-}" ]; then
        MAPPING_CSV="$(dirname "$TEST_SET_DIR")/mapping.csv"
    fi

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

require_mapping_csv() {
    if [ -z "${MAPPING_CSV:-}" ]; then
        echo "MAPPING_CSV is empty."
        exit 1
    fi
    if [ ! -f "$MAPPING_CSV" ]; then
        echo "mapping.csv not found: $MAPPING_CSV"
        echo "Generate it with: bash -i ./pre.sh"
        exit 1
    fi
}
