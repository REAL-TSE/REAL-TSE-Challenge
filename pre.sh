#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"

mkdir -p ./FireRedASR/pretrained_models
mkdir -p ./whisper/pretrained_models
python3 ./utils/download_asr_model.py \
  --zh_repo_id FireRedTeam/FireRedASR-AED-L \
  --zh_save_dir ./FireRedASR/pretrained_models \
  --en_repo_id openai/whisper-large-v2 \
  --en_save_dir ./whisper/pretrained_models

DATASETS_ROOT="./datasets"
DATASET_ROOT="${DATASETS_ROOT}/REAL-T"
ARCHIVE_DIR="${DATASETS_ROOT}/archives"
ARCHIVE_NAME="${REALT_DATASET_ARCHIVE_NAME:-REAL-T-dataset.tar.gz}"
ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}"
REBUILD_MAPPING_MODE="${REALT_MAPPING_MODE:-absolute}"
FORCE_DATASET_DOWNLOAD="${REALT_FORCE_DATASET_DOWNLOAD:-0}"

mkdir -p "$DATASETS_ROOT"
mkdir -p "$ARCHIVE_DIR"

have_full_dataset() {
    [ -d "${DATASET_ROOT}/mixtures" ] && \
    [ -d "${DATASET_ROOT}/enrolment_speakers" ] && \
    [ -d "${DATASET_ROOT}/BASE" ] && \
    [ -d "${DATASET_ROOT}/PRIMARY" ]
}

download_dataset_from_google_drive() {
    local archive_source_desc
    local gdrive_value

    if [ -n "${REALT_DATASET_GDRIVE_FILE_ID:-}" ]; then
        gdrive_value="${REALT_DATASET_GDRIVE_FILE_ID}"
        archive_source_desc="Google Drive file ${REALT_DATASET_GDRIVE_FILE_ID}"
    elif [ -n "${REALT_DATASET_GDRIVE_URL:-}" ]; then
        gdrive_value="${REALT_DATASET_GDRIVE_URL}"
        archive_source_desc="Google Drive URL"
    else
        echo "No Google Drive dataset source configured. Set REALT_DATASET_GDRIVE_FILE_ID or REALT_DATASET_GDRIVE_URL." >&2
        exit 1
    fi

    python3 ./utils/download_dataset_archive.py \
      --output "$ARCHIVE_PATH" \
      --gdrive-file-id "$gdrive_value"

    echo "Extracting REAL-T dataset from ${archive_source_desc}"
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

if [ "$FORCE_DATASET_DOWNLOAD" = "1" ] || ! have_full_dataset; then
    download_dataset_from_google_drive
else
    echo "Existing REAL-T dataset detected under $DATASET_ROOT. Skipping dataset download."
fi

python3 ./utils/regen_realt_dataset_mappings.py \
  --dataset-root "$DATASET_ROOT" \
  --mapping-mode "$REBUILD_MAPPING_MODE"

echo "mapping.csv generated at $(realpath "${DATASET_ROOT}/mapping.csv")"
