#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"


MODEL_NAME=${1:-"tfmap_context_100"}    # bsrnn pretrained on Libri2mix-100
MODEL_DIR="./pretrained"          # Directory containing the model checkpoint

# Dataset names
DATASETS=("AliMeeting" "AMI" "CHiME6" "AISHELL-4" "DipCo")

# Test subset
TEST_SET="DEV"

DEVICE="cuda"

# Paths
DATASET_ROOT="./datasets/REAL-T"
OUTPUT_ROOT="./output"


TSE_SCRIPT="./tse_baseline/tse_inference.py"

OUTPUT_ROOT="${OUTPUT_ROOT}/${TEST_SET}"
mkdir -p "$OUTPUT_ROOT"
# Iterate over each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    META_CSV_PATH="${DATASET_ROOT}/${TEST_SET}/${DATASET}_meta.csv"
    UTTERANCE_MAP_CSV="${DATASET_ROOT}/mapping.csv"
    OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/"

    mkdir -p "$OUTPUT_DIR"

    echo "Starting TSE processing for $DATASET..."
    python3 "$TSE_SCRIPT" \
        --model_name "$MODEL_DIR/$MODEL_NAME" \
        --meta_csv_path "$META_CSV_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --utterance_map_csv "$UTTERANCE_MAP_CSV" \
        --device "$DEVICE"

    # Check if TSE processing succeeded
    if [ $? -ne 0 ]; then
        echo "TSE processing failed for $DATASET."
        exit 1
    fi
    echo "TSE processing completed for $DATASET."
done

echo "All datasets processed successfully!"
