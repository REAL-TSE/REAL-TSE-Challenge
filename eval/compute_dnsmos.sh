#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TEST_SET_DIR="${TEST_SET_DIR:-./datasets/REAL-T/PRIMARY}"
DNSMOS_MODEL_DIR="${DNSMOS_MODEL_DIR:-./DNSMOS}"
DNSMOS_PROVIDER="${DNSMOS_PROVIDER:-auto}"
DNSMOS_NO_DOWNLOAD="${DNSMOS_NO_DOWNLOAD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

init_eval_common "./output/PRIMARY/bsrnn_vox1 ./output/PRIMARY/BSRNN"

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (compute & CSV), 2 (generate TXT from CSV), or both."
    exit 1
fi

run_dnsmos_full() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        CMD=(
            python3 "${REAL_T_ROOT}/utils/dnsmos_eval.py"
            --base_dir "$BASE_DIR"
            --test_set_dir "$TEST_SET_DIR"
            --dnsmos_model_dir "$DNSMOS_MODEL_DIR"
            --provider "$DNSMOS_PROVIDER"
            --csv_only
        )
        if [ "$DNSMOS_NO_DOWNLOAD" = "1" ]; then
            CMD+=(--no_download_models)
        fi
        if [ -n "$MAX_SAMPLES" ]; then
            CMD+=(--max_samples "$MAX_SAMPLES")
        fi

        echo "Running DNSMOS (mode 1: compute & CSV) for BASE_DIR=$BASE_DIR provider=$DNSMOS_PROVIDER"
        "${CMD[@]}"
    done
}

run_dnsmos_regen_txt() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        python3 "${REAL_T_ROOT}/utils/dnsmos_eval.py" \
            --base_dir "$BASE_DIR" \
            --dnsmos_model_dir "$DNSMOS_MODEL_DIR" \
            --regen_txt_only
    done
}

for mode in "${MODES[@]}"; do
    if [ "$mode" = "1" ]; then
        run_dnsmos_full
    elif [ "$mode" = "2" ]; then
        run_dnsmos_regen_txt
    else
        echo "Invalid mode: $mode. Please use 1 (compute & CSV), 2 (generate TXT), or both."
        exit 1
    fi
done

echo "DNSMOS evaluation finished."
