#!/bin/bash

# Shared REAL-T split naming:
#   DEV   -> ./datasets/REAL-T-dev/DEV
#   EVAL1 -> ./datasets/REAL-T-eval1/EVAL1
#   EVAL2 -> ./datasets/REAL-T-eval2/EVAL2

validate_test_set() {
    case "$1" in
        DEV|EVAL1|EVAL2) return 0 ;;
        *)
            echo "Error: --test-set must be DEV, EVAL1, or EVAL2 (got: $1)."
            return 1
            ;;
    esac
}

resolve_dataset_paths() {
    local test_set="$1"
    local dataset_root="${2:-}"

    if [ -z "$dataset_root" ]; then
        dataset_root="./datasets/REAL-T-$(echo "$test_set" | tr '[:upper:]' '[:lower:]')"
    fi

    DATASET_ROOT="$dataset_root"
    TEST_SET_DIR="${dataset_root}/${test_set}"
    MAPPING_CSV="${dataset_root}/mapping.csv"
}

verify_test_set_dir() {
    local test_set_dir="$1"
    local dataset_root="$2"

    if [ ! -d "$test_set_dir" ]; then
        echo "Test set directory not found: $test_set_dir"
        echo "Available splits:"
        ls -d "${dataset_root}"/*/ 2>/dev/null || echo "  (none)"
        return 1
    fi
}

verify_eval_assets() {
    local test_set_dir="$1"
    local mapping_csv="$2"

    if [ ! -f "$mapping_csv" ]; then
        echo "mapping.csv not found: $mapping_csv"
        echo "Generate it with: bash -i ./pre.sh"
        return 1
    fi
    if ! compgen -G "${test_set_dir}"/*_meta.csv > /dev/null; then
        echo "No *_meta.csv found under ${test_set_dir}."
        return 1
    fi
    if [ ! -d "${test_set_dir}/json" ]; then
        echo "Overlap JSON directory not found: ${test_set_dir}/json"
        return 1
    fi
}
