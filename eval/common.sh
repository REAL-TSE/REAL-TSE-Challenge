#!/bin/bash

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_T_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

source "${REAL_T_ROOT}/env_setup.sh"

init_eval_common() {
    local default_base_dirs="${1:-}"

    TEST_SET_DIR="${TEST_SET_DIR:-./datasets/REAL-T/PRIMARY}"
    INCLUDING_FISHER="${INCLUDING_FISHER:-False}"
    MAPPING_CSV_NAME="${MAPPING_CSV_NAME:-tse_audio_mapping.csv}"
    EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR:-eval_metrics}"
    USE_GPU="${USE_GPU:-1}"
    DATASETS="${DATASETS:-AliMeeting AISHELL-4 AMI DipCo CHiME6 Fisher}"

    EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR#/}"
    EVAL_METRICS_SUBDIR="${EVAL_METRICS_SUBDIR%/}"
    if [ "$EVAL_METRICS_SUBDIR" = "." ]; then
        EVAL_METRICS_SUBDIR=""
    fi

    if [ -z "${BASE_DIRS:-}" ]; then
        BASE_DIRS="${default_base_dirs}"
    fi

    read -r -a BASE_DIR_LIST <<< "${BASE_DIRS:-}"
    if [ "${#BASE_DIR_LIST[@]}" -eq 0 ]; then
        echo "No BASE_DIRS provided."
        exit 1
    fi

    # Normalize base dirs to their physical paths so symlinked output roots
    # work the same way as regular directories across all eval scripts.
    local normalized_base_dirs=()
    local base_dir=""
    for base_dir in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$base_dir" ]; then
            echo "Base directory not found: $base_dir"
            exit 1
        fi
        normalized_base_dirs+=("$(cd "$base_dir" && pwd -P)")
    done
    BASE_DIR_LIST=("${normalized_base_dirs[@]}")

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
    local base_dir="$1"
    find -L "$base_dir" -maxdepth 1 -mindepth 1 -type d | sort
}

eval_metrics_dir() {
    local base_dir="$1"
    if [ -z "${EVAL_METRICS_SUBDIR:-}" ]; then
        echo "$base_dir"
        return
    fi
    echo "${base_dir}/${EVAL_METRICS_SUBDIR}"
}
