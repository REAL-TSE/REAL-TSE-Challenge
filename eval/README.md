# Evaluation Guide

## Recommended Full Eval

Run the full REAL-T evaluation pipeline from the repo root with one command:

```bash
cd REAL-TSE-Challenge

# Evaluate on DEV, EVAL1, or EVAL2 (same pipeline; auto-detects datasets)
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0
bash ./run_eval.sh --output-dir ./output/EVAL1/BSRNN --test-set EVAL1 --cuda 0
bash ./run_eval.sh --output-dir ./output/EVAL2/BSRNN --test-set EVAL2 --cuda 0
```

`run_eval.sh` supports `DEV`, `EVAL1`, and `EVAL2`. Each split uses the
same local evaluation pipeline and expects the same ground-truth layout
as DEV (meta CSVs, transcripts, and overlap JSON). Datasets are
auto-detected from `*_meta.csv` in the test set directory.

`run_eval.sh` now supports top-level modes:

- `1`: run all evaluation sub-scripts only
- `2`: regenerate the aggregated summary from existing CSV files only
- `1 2`: run sub-scripts first, then generate the aggregated summary

Examples:

```bash
# Run all sub-scripts, then summarize
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0 1 2

# Only run all sub-scripts
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0 1

# Only summarize existing CSVs
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0 2
```

This sequentially runs:

1. `TER` via `eval/transcribe_and_evaluation.sh`
2. `TSE timing` via `eval/vad_and_evaluation.sh`
3. `speaker similarity (tse_enrol)` via `eval/compute_spk_similarity.sh`
4. `speaker similarity (mixture_enrol)` via `eval/compute_spk_similarity.sh`
5. `DNSMOS` via `eval/compute_dnsmos.sh`

## Shared Conventions

- All commands below are intended to be run from the REAL-T repo root.
- `OUTPUT_DIRS` is a space-separated list of TSE output roots such as `./output/DEV/BSRNN`.
- `TEST_SET_DIR` should point at the split folder: `./datasets/REAL-T-dev/DEV`, `./datasets/REAL-T-eval1/EVAL1`, or `./datasets/REAL-T-eval2/EVAL2`.
- `DATASETS` is auto-detected from `*_meta.csv` in `TEST_SET_DIR` when not set explicitly.
- All eval shell scripts source `env_setup.sh` automatically.
- `run_eval.sh` sets one `CUDA_VISIBLE_DEVICES` value for the entire pipeline and forces ONNX-based stages onto CUDA with `WESPEAKER_PROVIDER=cuda` and `DNSMOS_PROVIDER=cuda`.
- `run_eval.sh` accepts both absolute and relative `--output-dir` paths.
- `EVAL_METRICS_SUBDIR` controls where detailed metric CSV/TXT files are stored under each `OUTPUT_DIR` (default: `eval_metrics`).

Standalone `eval/*.sh` scripts default `TEST_SET_DIR` to DEV. For EVAL1/EVAL2, export the split directory (and `OUTPUT_DIRS`) first:

```bash
export OUTPUT_DIRS=./output/EVAL2/BSRNN
export TEST_SET_DIR=./datasets/REAL-T-eval2/EVAL2
# mapping.csv is inferred as dirname(TEST_SET_DIR)/mapping.csv
bash -i ./eval/transcribe_and_evaluation.sh 1 2
bash -i ./eval/vad_and_evaluation.sh 1 2
```

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

- `Mean by dataset`: `DEV` typically has 5 rows (`AISHELL-4 / AMI / AliMeeting / CHiME6 / DipCo`); `EVAL1` has 4 (`AliMeeting / AMI / CHiME6 / DipCo`); `EVAL2` has 2 (`unseen_CN / unseen_EN`)
- `Mean by language`: typically 2 rows for `en / zh`

Its columns are organized as grouped headers:

- `TER`
  - `zipformer-zh/en`
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

- `TER / zipformer-zh/en`: mean `wer_or_cer` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TER.csv`
- `SIM / enrol-mixture`: mean `speaker_cosine_similarity` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity_mixture_enrol.csv`
- `SIM / enrol-tse`: mean `speaker_cosine_similarity` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_spk_similarity.csv`
- `DNSMOS / *`: mean `SIG / BAK / OVRL / P808` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_dnsmos.csv`
- `RATIO / precision, recall, f1`: mean `precision / recall / f1` from `${EVAL_METRICS_SUBDIR}/{OUTPUT_NAME}_TSE_TIMING.csv`

## Prerequisites

Evaluation requires:

- A released REAL-T split (`DEV`, `EVAL1`, or `EVAL2`) with ground-truth references
- ASR model weights: `zipformer-en` (sherpa-onnx-zipformer-gigaspeech-2023-12-12) and `zipformer-zh` (sherpa-onnx-zipformer-multi-zh-hans-2023-9-2)
- Timing-VAD model weights: `FireRedVAD` (provided by FireRedASR2S; not used for ASR)
- DNSMOS ONNX model weights

### Recommended: One-command preparation via `pre.sh`

Download [REAL-T](https://huggingface.co/datasets/REAL-TSE/REAL-T) into `./datasets/` (full commands, including a Hugging Face mirror, are in the Dataset subsection of the root README):

```bash
huggingface-cli download REAL-TSE/REAL-T --repo-type dataset --local-dir ./datasets
# if huggingface.co is slow: export HF_ENDPOINT=https://hf-mirror.com
```

Then run `pre.sh` to regenerate mappings and prepare the default model
groups in one command. Zipformer, FireRedVAD, and DNSMOS downloads are
enabled by default; Whisper-large-v2 and FireRedASR-AED-L are disabled
unless explicitly requested.

```bash
bash -i ./pre.sh
```

`pre.sh` regenerates `mapping.csv` for all `./datasets/REAL-T-*/`
directories it finds.

Optional switches:

- `REALT_PREP_DOWNLOAD_ZIPFORMER_EN` (default `1`)
- `REALT_PREP_DOWNLOAD_ZIPFORMER_ZH` (default `1`)
- `REALT_PREP_DOWNLOAD_FIRERED_ASR` (default `0`)
- `REALT_PREP_DOWNLOAD_WHISPER` (default `0`)
- `REALT_PREP_DOWNLOAD_FIRERED_VAD` (default `1`)
- `REALT_PREP_DOWNLOAD_DNSMOS` (default `1`)

Example: only download FireRedVAD when the other default model weights
are already present:

```bash
REALT_PREP_DOWNLOAD_ZIPFORMER_EN=0 \
REALT_PREP_DOWNLOAD_ZIPFORMER_ZH=0 \
REALT_PREP_DOWNLOAD_DNSMOS=0 \
bash -i ./pre.sh
```

### Optional: Manual downloads (separate from `pre.sh`)

Use the commands below when you want to fetch one component independently.

### Zipformer ASR Models (sherpa-onnx)

```bash
mkdir -p ./zipformer/pretrained_models
python3 ./utils/download_zipformer.py
# Or only one language:
#   python3 ./utils/download_zipformer.py --only en
#   python3 ./utils/download_zipformer.py --only zh
```

Both models are streamed from the
[k2-fsa/sherpa-onnx GitHub Releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
(`sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2`,
`sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2`, ~290 MB each)
and extracted into `./zipformer/pretrained_models/<release_name>/`.

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

Timing evaluation also requires overlap JSON under `${TEST_SET_DIR}/json`
(for example `./datasets/REAL-T-dev/DEV/json`,
`./datasets/REAL-T-eval1/EVAL1/json`, or
`./datasets/REAL-T-eval2/EVAL2/json`).

If you copied the released split folder, that directory is already
included.


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

`eval/transcribe_and_evaluation.sh` runs transcription and TER using `zipformer-zh` for Chinese datasets and `zipformer-en` for English datasets. Dataset language (`en` / `zh`) and therefore which ASR backend to run come from `utils/dataset_lang.py`. Both backends are CPU-only sherpa-onnx Zipformer transducers; the `--device` flag is kept for API parity but does not enable a CUDA EP.

```bash
# DEV (default TEST_SET_DIR)
bash -i ./eval/transcribe_and_evaluation.sh 1

# EVAL2
OUTPUT_DIRS=./output/EVAL2/BSRNN TEST_SET_DIR=./datasets/REAL-T-eval2/EVAL2 \
  bash -i ./eval/transcribe_and_evaluation.sh 1 2
```

Important env vars:

- `OUTPUT_DIRS`
- `TEST_SET_DIR`
- `DATASETS`
- `ASR_DEVICE`
- `MAPPING_CSV_NAME`

### Timing / VAD Eval

`eval/vad_and_evaluation.sh` supports:

- mode `1`: FireRedVAD inference
- mode `2`: timing evaluation
- mode `3`: visualization

```bash
# DEV (default TEST_SET_DIR)
bash -i ./eval/vad_and_evaluation.sh 1 2

# EVAL1
OUTPUT_DIRS=./output/EVAL1/BSRNN TEST_SET_DIR=./datasets/REAL-T-eval1/EVAL1 \
  bash -i ./eval/vad_and_evaluation.sh 1 2
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
# TSE vs enrol (DEV default)
bash -i ./eval/compute_spk_similarity.sh 1 2

# Mixture vs enrol baseline on EVAL2
OUTPUT_DIRS=./output/EVAL2/BSRNN TEST_SET_DIR=./datasets/REAL-T-eval2/EVAL2 \
  SPK_SIM_PAIR_MODE=mixture_enrol bash -i ./eval/compute_spk_similarity.sh 1 2
```

Important env vars:

- `OUTPUT_DIRS`
- `TEST_SET_DIR`
- `MAPPING_CSV`
- `WESPEAKER_PROVIDER`
- `WESPEAKER_DATASET_LANG_OVERRIDES` (optional; default is the shared `en`/`zh` map in `utils/dataset_lang.py`)
- `MAX_SAMPLES`

### DNSMOS

`eval/compute_dnsmos.sh` computes `SIG`, `BAK`, `OVRL`, and `P808`.

```bash
# DEV (default TEST_SET_DIR)
bash -i ./eval/compute_dnsmos.sh 1 2

# EVAL1
OUTPUT_DIRS=./output/EVAL1/BSRNN TEST_SET_DIR=./datasets/REAL-T-eval1/EVAL1 \
  bash -i ./eval/compute_dnsmos.sh 1 2
```

Important env vars:

- `OUTPUT_DIRS`
- `TEST_SET_DIR`
- `DNSMOS_MODEL_DIR`
- `DNSMOS_PROVIDER`
- `DNSMOS_NO_DOWNLOAD`
- `DNSMOS_DATASET_LANG_OVERRIDES` (optional; default is the shared `en`/`zh` map in `utils/dataset_lang.py`)
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
