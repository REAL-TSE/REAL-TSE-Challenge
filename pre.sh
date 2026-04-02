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

# ---- Model download switches (all default to enabled) ----
REALT_PREP_DOWNLOAD_FIRERED_ASR="${REALT_PREP_DOWNLOAD_FIRERED_ASR:-1}"
REALT_PREP_DOWNLOAD_WHISPER="${REALT_PREP_DOWNLOAD_WHISPER:-1}"
REALT_PREP_DOWNLOAD_FIRERED_VAD="${REALT_PREP_DOWNLOAD_FIRERED_VAD:-1}"
REALT_PREP_DOWNLOAD_DNSMOS="${REALT_PREP_DOWNLOAD_DNSMOS:-1}"

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

# Regenerate mapping.csv for any dataset directories found under ./datasets/.
DATASETS_ROOT="./datasets"
REBUILD_MAPPING_MODE="${REALT_MAPPING_MODE:-absolute}"

for dataset_dir in "${DATASETS_ROOT}"/REAL-T-*/; do
    [ -d "$dataset_dir" ] || continue
    python3 ./utils/regen_realt_dataset_mappings.py \
      --dataset-root "$dataset_dir" \
      --mapping-mode "$REBUILD_MAPPING_MODE"
    echo "mapping.csv generated at $(realpath "${dataset_dir}/mapping.csv")"
done
