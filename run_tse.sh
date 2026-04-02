#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"

usage() {
    cat <<'EOF'
Usage:
  bash ./run_tse.sh --model <name> --test-set <DEV|EVAL> [options]

Required:
  --model         TSE model name (e.g. tfmap_context_100)
  --test-set      Evaluation split: DEV or EVAL

Optional:
  --device        Inference device (default: cuda)
  --model-dir     Directory containing model checkpoints (default: ./pretrained)
  --dataset-root  Root of the REAL-T dataset (default: ./datasets/REAL-T-{dev|eval})
  --output-root   Root output directory (default: ./output)

Example:
  bash ./run_tse.sh --model tfmap_context_100 --test-set DEV
  bash ./run_tse.sh --model tfmap_context_100 --test-set EVAL --device cuda
EOF
}

MODEL_NAME=""
TEST_SET=""
DEVICE="cuda"
MODEL_DIR="./pretrained"
DATASET_ROOT=""
OUTPUT_ROOT="./output"

while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODEL_NAME="${2:-}";  shift 2 ;;
        --test-set)     TEST_SET="${2:-}";    shift 2 ;;
        --device)       DEVICE="${2:-}";      shift 2 ;;
        --model-dir)    MODEL_DIR="${2:-}";   shift 2 ;;
        --dataset-root) DATASET_ROOT="${2:-}"; shift 2 ;;
        --output-root)  OUTPUT_ROOT="${2:-}";  shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_NAME" ] || [ -z "$TEST_SET" ]; then
    echo "Error: --model and --test-set are required."
    echo
    usage
    exit 1
fi

if [ "$TEST_SET" != "EVAL" ] && [ "$TEST_SET" != "DEV" ]; then
    echo "Error: --test-set must be DEV or EVAL."
    exit 1
fi

if [ -z "$DATASET_ROOT" ]; then
    DATASET_ROOT="./datasets/REAL-T-$(echo "$TEST_SET" | tr '[:upper:]' '[:lower:]')"
fi

TEST_SET_DIR="${DATASET_ROOT}/${TEST_SET}"
if [ ! -d "$TEST_SET_DIR" ]; then
    echo "Test set directory not found: $TEST_SET_DIR"
    echo "Available splits:"
    ls -d "${DATASET_ROOT}"/*/ 2>/dev/null || echo "  (none)"
    exit 1
fi

# Auto-detect datasets from meta CSVs in the test set directory
DATASETS=()
for meta in "${TEST_SET_DIR}"/*_meta.csv; do
    [ -f "$meta" ] || continue
    name="$(basename "$meta" _meta.csv)"
    DATASETS+=("$name")
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No *_meta.csv found under ${TEST_SET_DIR}."
    exit 1
fi
echo "[auto-detect] TEST_SET=${TEST_SET}, DATASETS: ${DATASETS[*]}"

TSE_SCRIPT="./tse_baseline/tse_inference.py"

OUTPUT_ROOT="${OUTPUT_ROOT}/${TEST_SET}"
mkdir -p "$OUTPUT_ROOT"

for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    META_CSV_PATH="${TEST_SET_DIR}/${DATASET}_meta.csv"
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

    if [ $? -ne 0 ]; then
        echo "TSE processing failed for $DATASET."
        exit 1
    fi
    echo "TSE processing completed for $DATASET."
done

echo "All datasets processed successfully!"
