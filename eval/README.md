# Evaluation Guide

## Recommended Full Eval

Run the full REAL-T evaluation pipeline from the repo root with one command:

```bash
cd REAL-TSE-Challenge

# Evaluate on DEV split (5 datasets)
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0

# Evaluate on EVAL split (4 datasets)
bash ./run_eval.sh --output-dir ./output/EVAL/BSRNN --test-set EVAL --cuda 0
```

`run_eval.sh` now supports top-level modes:

- `1`: run all evaluation sub-scripts only
- `2`: regenerate the aggregated summary from existing CSV files only
- `1 2`: run sub-scripts first, then generate the aggregated summary

The `--test-set` flag accepts `EVAL` or `DEV`. The pipeline auto-detects
which datasets are available from the `*_meta.csv` files in the test set
directory, so it works seamlessly with either 4 or 5 datasets.

Examples:

```bash
# Run all sub-scripts, then summarize (EVAL)
bash ./run_eval.sh --output-dir ./output/EVAL/BSRNN --test-set EVAL --cuda 0 1 2

# Only run all sub-scripts (DEV)
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0 1

# Only summarize existing CSVs
bash ./run_eval.sh --output-dir ./output/EVAL/BSRNN --test-set EVAL --cuda 0 2
```

This sequentially runs:

1. `TER` via `eval/transcribe_and_evaluation.sh`
2. `TSE timing` via `eval/vad_and_evaluation.sh`
3. `speaker similarity (tse_enrol)` via `eval/compute_spk_similarity.sh`
4. `speaker similarity (mixture_enrol)` via `eval/compute_spk_similarity.sh`
5. `DNSMOS` via `eval/compute_dnsmos.sh`

## Shared Conventions

- All commands below are intended to be run from the REAL-T repo root.
- `OUTPUT_DIRS` is a space-separated list of TSE output roots such as `./output/EVAL/BSRNN`.
- `TEST_SET_DIR` should point to `./datasets/REAL-T-eval/EVAL` or `./datasets/REAL-T-dev/DEV`.
- `DATASETS` is auto-detected from `*_meta.csv` in `TEST_SET_DIR` when not set explicitly.
- All eval shell scripts source `env_setup.sh` automatically.
- `run_eval.sh` sets one `CUDA_VISIBLE_DEVICES` value for the entire pipeline and forces ONNX-based stages onto CUDA with `WESPEAKER_PROVIDER=cuda` and `DNSMOS_PROVIDER=cuda`.
- `run_eval.sh` accepts both absolute and relative `--output-dir` paths.
- `EVAL_METRICS_SUBDIR` controls where detailed metric CSV/TXT files are stored under each `OUTPUT_DIR` (default: `eval_metrics`).

Expected outputs under each `OUTPUT_DIR`:

- Detailed metric files under `${EVAL_METRICS_SUBDIR}`:
  - `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TER.csv` and `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TER.txt`
  - `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TSE_TIMING.csv` and `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TSE_TIMING.txt`
  - `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity.csv` and `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity_summary.txt`
  - `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity_mixture_enrol.csv` and `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity_mixture_enrol_summary.txt`
  - `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_dnsmos.csv` and `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_dnsmos.txt`
- Aggregated report at `OUTPUT_DIR` root:
  - `{OUTPUT_NAME}_summary.txt`

`{OUTPUT_NAME}_summary.txt` is the new aggregated report. It is recomputed from CSV files and contains two mean-only tables:

- `Mean by dataset`: typically 5 rows for `AISHELL-4 / AMI / AliMeeting / CHiME6 / DipCo`
- `Mean by language`: typically 2 rows for `en / chs`

Its columns are organized as grouped headers:

- `TER`
  - `fireredasr-1/whisper`
- `SIM`
  - `enrol-mixture`
  - `enrol-tse`
- `DNSMOS`
  - `SIG`
  - `BAK`
  - `OVRL`
  - `P808`
- `RATIO`
  - `precision`
  - `recall`
  - `f1`

Current metric sources for the aggregated summary:

- `TER / fireredasr-1/whisper`: mean `wer_or_cer` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TER.csv`
- `SIM / enrol-mixture`: mean `speaker_cosine_similarity` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity_mixture_enrol.csv`
- `SIM / enrol-tse`: mean `speaker_cosine_similarity` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity.csv`
- `DNSMOS / *`: mean `SIG / BAK / OVRL / P808` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_dnsmos.csv`
- `RATIO / precision, recall, f1`: mean `precision / recall / f1` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TSE_TIMING.csv`

## Prerequisites

Evaluation requires:

- REAL-T dataset
- ASR model weights: `FireRedASR-AED-L`, `whisper-large-v2`
- Timing-VAD model weights: `FireRedVAD`
- DNSMOS ONNX model weights

### Recommended: One-command preparation via `pre.sh`

`pre.sh` prepares a manually downloaded dataset plus all 4 model groups in one command (all enabled by default):

```bash
bash -i ./pre.sh
```

Place one or both archives under `./datasets/archives/`:

- `REAL-T-dev.tar.gz` — DEV split (5 datasets) -> extracts to `./datasets/REAL-T-dev/`
- `REAL-T-eval.tar.gz` — EVAL split (4 datasets, no AISHELL-4) -> extracts to `./datasets/REAL-T-eval/`

```bash
mkdir -p ./datasets/archives
cp /path/to/REAL-T-dev.tar.gz  ./datasets/archives/
cp /path/to/REAL-T-eval.tar.gz ./datasets/archives/
```

`pre.sh` auto-detects all archives and extracts them incrementally (both
each archive extracts to its own directory). You can also pass archives
explicitly:

```bash
REALT_DATASET_ARCHIVE_PATHS="/path/to/REAL-T-dev.tar.gz /path/to/REAL-T-eval.tar.gz" bash -i ./pre.sh
```

If you already have a local dataset under `./datasets/REAL-T`, `pre.sh` will skip extraction and still prepare mappings.

Optional switches (all default to `1`):

- `REALT_PREP_PREPARE_DATASET`
- `REALT_PREP_DOWNLOAD_FIRERED_ASR`
- `REALT_PREP_DOWNLOAD_WHISPER`
- `REALT_PREP_DOWNLOAD_FIRERED_VAD`
- `REALT_PREP_DOWNLOAD_DNSMOS`

Example: only prepare the local dataset archive + FireRedVAD

```bash
REALT_DATASET_ARCHIVE_PATHS="/path/to/REAL-T-dev.tar.gz" \
REALT_PREP_DOWNLOAD_FIRERED_ASR=0 \
REALT_PREP_DOWNLOAD_WHISPER=0 \
REALT_PREP_DOWNLOAD_DNSMOS=0 \
bash -i ./pre.sh
```

### Optional: Manual downloads (separate from `pre.sh`)

Use the commands below when you want to fetch one component independently.

### FireRedASR-AED-L + Whisper

```bash
mkdir -p ./FireRedASR/pretrained_models ./whisper/pretrained_models
python3 ./utils/download_asr_model.py \
  --zh_repo_id FireRedTeam/FireRedASR-AED-L \
  --zh_save_dir ./FireRedASR/pretrained_models \
  --en_repo_id openai/whisper-large-v2 \
  --en_save_dir ./whisper/pretrained_models
```

### FireRedVAD for Timing Eval

`eval/vad_and_evaluation.sh` expects FireRedVAD weights in:

```bash
./FireRedASR2S/pretrained_models/FireRedVAD/VAD
```

Recommended download flow:

```bash
git submodule update --init --recursive FireRedASR2S
pip install modelscope
mkdir -p ./FireRedASR2S/pretrained_models/FireRedVAD
python -c "from modelscope import snapshot_download; snapshot_download('xukaituo/FireRedVAD', local_dir='./FireRedASR2S/pretrained_models/FireRedVAD')"
```

Timing evaluation also requires overlap JSON under `./datasets/REAL-T-{dev,eval}/<SPLIT>/json`.

If your dataset was prepared from the recommended manually downloaded archive via `bash -i ./pre.sh`, that directory is already included.


### DNSMOS

`eval/compute_dnsmos.sh` uses `./DNSMOS` by default. If the ONNX files are missing, mode 1 auto-downloads them unless `DNSMOS_NO_DOWNLOAD=1`.

Manual download option:

```bash
mkdir -p ./DNSMOS
python3 - <<'PY'
from huggingface_hub import hf_hub_download
for fname in ("sig_bak_ovr.onnx", "model_v8.onnx"):
    hf_hub_download(repo_id="Vyvo-Research/dnsmos", filename=fname, local_dir="./DNSMOS")
PY
```

## Script Details

### ASR TER

`eval/transcribe_and_evaluation.sh` runs transcription and TER using `FireRedASR-AED-L` for Chinese datasets and `whisper-large-v2` for English datasets.

```bash
# Only ASR
bash -i ./eval/transcribe_and_evaluation.sh 1

# Only evaluation
bash -i ./eval/transcribe_and_evaluation.sh 2

# Both
bash -i ./eval/transcribe_and_evaluation.sh 1 2
```

Important env vars:

- `OUTPUT_DIRS`
- `TEST_SET_DIR`
- `DATASETS`
- `CHINESE_DATASETS`
- `ENGLISH_DATASETS`
- `ASR_DEVICE`
- `MAPPING_CSV_NAME`

### Timing / VAD Eval

`eval/vad_and_evaluation.sh` supports:

- mode `1`: FireRedVAD inference
- mode `2`: timing evaluation
- mode `3`: visualization

```bash
# Only VAD
bash -i ./eval/vad_and_evaluation.sh 1

# Only timing evaluation
bash -i ./eval/vad_and_evaluation.sh 2

# Full timing pipeline
bash -i ./eval/vad_and_evaluation.sh 1 2

# Optional visualization after mode 2
bash -i ./eval/vad_and_evaluation.sh 3
```

Important env vars:

- `OUTPUT_DIRS`
- `TEST_SET_DIR`
- `DATASETS`
- `GT_JSON_BASE_DIR`
- `METADATA_DIR`
- `FIREREDASR2S_ROOT`
- `FIRERED_VAD_MODEL_DIR`
- `USE_GPU`
- `SPEECH_THRESHOLD`
- `FRAME_SHIFT`
- `COLLAR`
- `MATCH_TOLERANCE`

Mode `1` writes `FireRedVAD/vad_segments.jsonl` under each dataset directory. Mode `2` writes `FireRedVAD/label_segments.jsonl` plus `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TSE_TIMING.csv` and `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TSE_TIMING.txt`.

### Speaker Similarity

`eval/compute_spk_similarity.sh` supports two pair modes:

- `SPK_SIM_PAIR_MODE=tse_enrol`
- `SPK_SIM_PAIR_MODE=mixture_enrol`

```bash
# TSE vs enrol
bash -i ./eval/compute_spk_similarity.sh 1 2

# Mixture vs enrol baseline
SPK_SIM_PAIR_MODE=mixture_enrol bash -i ./eval/compute_spk_similarity.sh 1 2
```

Important env vars:

- `OUTPUT_DIRS`
- `TEST_SET_DIR`
- `MAPPING_CSV`
- `WESPEAKER_LANG`
- `WESPEAKER_PROVIDER`
- `WESPEAKER_DATASET_LANG_OVERRIDES`
- `MAX_SAMPLES`

### DNSMOS

`eval/compute_dnsmos.sh` computes `SIG`, `BAK`, `OVRL`, and `P808`.

```bash
# Compute CSV and regenerate TXT
bash -i ./eval/compute_dnsmos.sh 1 2
```

Important env vars:

- `OUTPUT_DIRS`
- `TEST_SET_DIR`
- `DNSMOS_MODEL_DIR`
- `DNSMOS_PROVIDER`
- `DNSMOS_NO_DOWNLOAD`
- `MAX_SAMPLES`

## Aggregated Summary Internals

The aggregated report is generated by:

```bash
python3 ./utils/aggregate_eval_summary.py \
  --output_dir ./output/DEV/BSRNN \
  --metrics_subdir eval_metrics
```

You usually do not need to call it directly, because `run_eval.sh ... 2` already wraps it.

The script expects the following CSV files under `OUTPUT_DIR/{metrics_subdir}` by default, and falls back to legacy flat files under `OUTPUT_DIR` if needed:

- `{OUTPUT_NAME}_TER.csv`
- `{OUTPUT_NAME}_spk_similarity.csv`
- `{OUTPUT_NAME}_spk_similarity_mixture_enrol.csv`
- `{OUTPUT_NAME}_dnsmos.csv`
- `{OUTPUT_NAME}_TSE_TIMING.csv`

If any of them is missing, summary generation will stop with an error so the missing stage is visible immediately.
