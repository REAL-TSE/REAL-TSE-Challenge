#!/bin/bash

REAL_T_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REALT_CONDA_ENV="${REALT_CONDA_ENV:-REAL-T}"

if ! declare -F conda >/dev/null 2>&1; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda not found in PATH" >&2
        return 1 2>/dev/null || exit 1
    fi
    eval "$(conda shell.bash hook)"
fi

conda activate "$REALT_CONDA_ENV"

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ONNX Runtime GPU wheels depend on NVIDIA shared libraries shipped under site-packages.
NVIDIA_LIB_ROOT="${CONDA_PREFIX:-}/lib/python3.10/site-packages/nvidia"
for nvidia_subdir in \
    "cudnn/lib" \
    "cublas/lib" \
    "cuda_runtime/lib" \
    "cufft/lib" \
    "curand/lib" \
    "cusolver/lib" \
    "cusparse/lib" \
    "nvjitlink/lib"
do
    nvidia_lib_dir="${NVIDIA_LIB_ROOT}/${nvidia_subdir}"
    if [ -d "${nvidia_lib_dir}" ]; then
        export LD_LIBRARY_PATH="${nvidia_lib_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

if [ -d "${REAL_T_ROOT}/FireRedASR/fireredasr" ]; then
    export PATH="${REAL_T_ROOT}/FireRedASR/fireredasr:${REAL_T_ROOT}/FireRedASR/fireredasr/utils${PATH:+:$PATH}"
fi

for local_repo in "${REAL_T_ROOT}/FireRedASR" "${REAL_T_ROOT}/FireRedASR2S" "${REAL_T_ROOT}/wesep"; do
    if [ -d "${local_repo}" ]; then
        export PYTHONPATH="${local_repo}${PYTHONPATH:+:$PYTHONPATH}"
    fi
done

cd "${REAL_T_ROOT}"
