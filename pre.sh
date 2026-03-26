#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"

mkdir -p ./FireRedASR/pretrained_models
mkdir -p ./whisper/pretrained_models
python3 ./utils/download_asr_model.py \
  --zh_repo_id FireRedTeam/FireRedASR-AED-L \
  --zh_save_dir ./FireRedASR/pretrained_models \
  --en_repo_id openai/whisper-large-v2 \
  --en_save_dir ./whisper/pretrained_models

mkdir -p ./datasets
python3 ./utils/download_REAL-T.py \
--save_dir "./datasets/REAL-T" \
--hub_repo "SLbaba/REAL-T"


MIX_DIR="./datasets/REAL-T/mixtures"
ENROL_DIR="./datasets/REAL-T/enrolment_speakers"
OUT_CSV="./datasets/REAL-T/mapping.csv"


echo "utterance,path" > "$OUT_CSV"

find "$MIX_DIR" -type f -name "*.wav" | while read -r file; do
    utt_id=$(basename "$file" .wav)
    abs_path=$(realpath "$file")
    echo "$utt_id,$abs_path" >> "$OUT_CSV"
done

find "$ENROL_DIR" -type f -name "*.wav" | while read -r file; do
    utt_id=$(basename "$file" .wav)
    abs_path=$(realpath "$file")
    echo "$utt_id,$abs_path" >> "$OUT_CSV"
done

echo "mapping.csv generated at $(realpath $OUT_CSV)"
