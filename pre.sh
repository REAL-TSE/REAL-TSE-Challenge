#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"

require_binary_flag() {
    local name="$1"
    local value="$2"
    if [ "$value" != "0" ] && [ "$value" != "1" ]; then
        echo "${name} must be 0 or 1, got: ${value}" >&2
        exit 1
    fi
}

download_hf_snapshot_if_needed() {
    local repo_id="$1"
    local save_root="$2"
    local target_dir="${save_root}/$(basename "$repo_id")"
    mkdir -p "$target_dir"
    if [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        echo "[Skip] HF model already exists: ${target_dir}"
        return
    fi

    python3 - "$repo_id" "$target_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=target_dir)
print(f"[Saved] {repo_id} -> {target_dir}")
PY
}

download_firered_vad_if_needed() {
    local repo_id="$1"
    local save_dir="$2"
    local vad_dir="${save_dir}/VAD"
    mkdir -p "$save_dir"
    if [ -d "$vad_dir" ] && [ -n "$(ls -A "$vad_dir" 2>/dev/null)" ]; then
        echo "[Skip] FireRedVAD already exists: ${vad_dir}"
        return
    fi

    python3 - "$repo_id" "$save_dir" <<'PY'
import sys
from modelscope import snapshot_download

repo_id, save_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id, local_dir=save_dir)
print(f"[Saved] {repo_id} -> {save_dir}")
PY
}

download_dnsmos_if_needed() {
    local repo_id="$1"
    local model_dir="$2"
    mkdir -p "$model_dir"
    if [ -f "${model_dir}/sig_bak_ovr.onnx" ] && [ -f "${model_dir}/model_v8.onnx" ]; then
        echo "[Skip] DNSMOS ONNX models already exist under: ${model_dir}"
        return
    fi

    python3 - "$repo_id" "$model_dir" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id, model_dir = sys.argv[1], Path(sys.argv[2]).resolve()
model_dir.mkdir(parents=True, exist_ok=True)

for fname in ("sig_bak_ovr.onnx", "model_v8.onnx"):
    target = model_dir / fname
    if target.is_file():
        print(f"[Skip] DNSMOS file already exists: {target}")
        continue
    path = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(model_dir))
    print(f"[Saved] {path}")
PY
}

# ---- One-command preparation switches (all default to enabled) ----
REALT_PREP_PREPARE_DATASET="${REALT_PREP_PREPARE_DATASET:-${REALT_PREP_DOWNLOAD_DATASET:-1}}"
REALT_PREP_DOWNLOAD_FIRERED_ASR="${REALT_PREP_DOWNLOAD_FIRERED_ASR:-1}"
REALT_PREP_DOWNLOAD_WHISPER="${REALT_PREP_DOWNLOAD_WHISPER:-1}"
REALT_PREP_DOWNLOAD_FIRERED_VAD="${REALT_PREP_DOWNLOAD_FIRERED_VAD:-1}"
REALT_PREP_DOWNLOAD_DNSMOS="${REALT_PREP_DOWNLOAD_DNSMOS:-1}"

require_binary_flag "REALT_PREP_PREPARE_DATASET" "$REALT_PREP_PREPARE_DATASET"
require_binary_flag "REALT_PREP_DOWNLOAD_FIRERED_ASR" "$REALT_PREP_DOWNLOAD_FIRERED_ASR"
require_binary_flag "REALT_PREP_DOWNLOAD_WHISPER" "$REALT_PREP_DOWNLOAD_WHISPER"
require_binary_flag "REALT_PREP_DOWNLOAD_FIRERED_VAD" "$REALT_PREP_DOWNLOAD_FIRERED_VAD"
require_binary_flag "REALT_PREP_DOWNLOAD_DNSMOS" "$REALT_PREP_DOWNLOAD_DNSMOS"

# ---- Model download locations / repo ids ----
FIRERED_ASR_REPO_ID="${REALT_FIRERED_ASR_REPO_ID:-FireRedTeam/FireRedASR-AED-L}"
FIRERED_ASR_SAVE_ROOT="${REALT_FIRERED_ASR_SAVE_ROOT:-./FireRedASR/pretrained_models}"
WHISPER_REPO_ID="${REALT_WHISPER_REPO_ID:-openai/whisper-large-v2}"
WHISPER_SAVE_ROOT="${REALT_WHISPER_SAVE_ROOT:-./whisper/pretrained_models}"
FIRERED_VAD_REPO_ID="${REALT_FIRERED_VAD_REPO_ID:-xukaituo/FireRedVAD}"
FIRERED_VAD_SAVE_DIR="${REALT_FIRERED_VAD_SAVE_DIR:-./FireRedASR2S/pretrained_models/FireRedVAD}"
DNSMOS_HF_REPO_ID="${REALT_DNSMOS_REPO_ID:-Vyvo-Research/dnsmos}"
DNSMOS_MODEL_DIR="${REALT_DNSMOS_MODEL_DIR:-./DNSMOS}"

if [ "$REALT_PREP_DOWNLOAD_WHISPER" = "1" ]; then
    download_hf_snapshot_if_needed "$WHISPER_REPO_ID" "$WHISPER_SAVE_ROOT"
else
    echo "[Skip] Whisper download disabled by REALT_PREP_DOWNLOAD_WHISPER=0"
fi

if [ "$REALT_PREP_DOWNLOAD_FIRERED_ASR" = "1" ]; then
    download_hf_snapshot_if_needed "$FIRERED_ASR_REPO_ID" "$FIRERED_ASR_SAVE_ROOT"
else
    echo "[Skip] FireRedASR download disabled by REALT_PREP_DOWNLOAD_FIRERED_ASR=0"
fi

if [ "$REALT_PREP_DOWNLOAD_FIRERED_VAD" = "1" ]; then
    download_firered_vad_if_needed "$FIRERED_VAD_REPO_ID" "$FIRERED_VAD_SAVE_DIR"
else
    echo "[Skip] FireRedVAD download disabled by REALT_PREP_DOWNLOAD_FIRERED_VAD=0"
fi

if [ "$REALT_PREP_DOWNLOAD_DNSMOS" = "1" ]; then
    download_dnsmos_if_needed "$DNSMOS_HF_REPO_ID" "$DNSMOS_MODEL_DIR"
else
    echo "[Skip] DNSMOS download disabled by REALT_PREP_DOWNLOAD_DNSMOS=0"
fi

DATASETS_ROOT="./datasets"
DATASET_ROOT="${DATASETS_ROOT}/REAL-T"
ARCHIVE_DIR="${DATASETS_ROOT}/archives"
ARCHIVE_NAME="${REALT_DATASET_ARCHIVE_NAME:-REAL-T-dev.tar.gz}"
DEFAULT_ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}"
ARCHIVE_PATH="${REALT_DATASET_ARCHIVE_PATH:-$DEFAULT_ARCHIVE_PATH}"
REBUILD_MAPPING_MODE="${REALT_MAPPING_MODE:-absolute}"
FORCE_DATASET_PREPARE="${REALT_FORCE_DATASET_PREPARE:-${REALT_FORCE_DATASET_DOWNLOAD:-0}}"

mkdir -p "$DATASETS_ROOT"
mkdir -p "$ARCHIVE_DIR"

have_full_dataset() {
    [ -d "${DATASET_ROOT}/mixtures" ] && \
    [ -d "${DATASET_ROOT}/enrolment_speakers" ] && \
    [ -d "${DATASET_ROOT}/DEV" ] && \
    [ -d "${DATASET_ROOT}/metadata" ] && \
    [ -d "${DATASET_ROOT}/json" ]
}

print_dataset_manual_download_help() {
    cat >&2 <<EOF
REAL-T dataset not found under ${DATASET_ROOT}.

Please download the DEV dataset archive manually from:
  https://drive.google.com/file/d/1uGTcTfRjOdqPa4PJAhjrXYLzxbGVy6pY/view?usp=sharing

Then choose one of the following:
  1. Extract it so the dataset is available at ${DATASET_ROOT}
  2. Save the archive at ${DEFAULT_ARCHIVE_PATH}
  3. Save it anywhere else and rerun with:
     REALT_DATASET_ARCHIVE_PATH=/absolute/path/to/REAL-T-dev.tar.gz bash -i ./pre.sh
EOF
}

prepare_dataset_from_local_archive() {
    if [ ! -f "$ARCHIVE_PATH" ]; then
        print_dataset_manual_download_help
        exit 1
    fi

    echo "Extracting REAL-T dataset from local archive: ${ARCHIVE_PATH}"
    rm -rf "$DATASET_ROOT"
    case "$ARCHIVE_PATH" in
        *.tar.gz|*.tgz)
            tar -xzf "$ARCHIVE_PATH" -C "$DATASETS_ROOT"
            ;;
        *.zip)
            unzip -o "$ARCHIVE_PATH" -d "$DATASETS_ROOT"
            ;;
        *)
            echo "Unsupported dataset archive format: $ARCHIVE_PATH" >&2
            exit 1
            ;;
    esac
}

if [ "$REALT_PREP_PREPARE_DATASET" = "1" ]; then
    if [ "$FORCE_DATASET_PREPARE" = "1" ] || ! have_full_dataset; then
        prepare_dataset_from_local_archive
    else
        echo "Existing REAL-T dataset detected under $DATASET_ROOT. Skipping dataset extraction."
    fi
else
    echo "[Skip] Dataset preparation disabled by REALT_PREP_PREPARE_DATASET=0"
fi

if [ -d "$DATASET_ROOT" ]; then
    python3 ./utils/regen_realt_dataset_mappings.py \
      --dataset-root "$DATASET_ROOT" \
      --mapping-mode "$REBUILD_MAPPING_MODE"
    echo "mapping.csv generated at $(realpath "${DATASET_ROOT}/mapping.csv")"
else
    echo "[Skip] ${DATASET_ROOT} not found; mapping.csv regeneration skipped."
fi
