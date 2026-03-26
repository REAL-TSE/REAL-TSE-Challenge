# Evaluation Guide

## Recommended Full Eval

Run the full REAL-T evaluation pipeline from the repo root with one command:

```bash
cd REAL-T
bash ./run_eval.sh --base-dir ./output/PRIMARY/BSRNN --test-set PRIMARY --cuda 0
```

`run_eval.sh` now supports top-level modes:

- `1`: run all evaluation sub-scripts only
- `2`: regenerate the aggregated summary from existing CSV files only
- `1 2`: run sub-scripts first, then generate the aggregated summary

Examples:

```bash
# Run all sub-scripts, then summarize
bash ./run_eval.sh --base-dir ./output/PRIMARY/BSRNN --test-set PRIMARY --cuda 0 1 2

# Only run all sub-scripts
bash ./run_eval.sh --base-dir ./output/PRIMARY/BSRNN --test-set PRIMARY --cuda 0 1

# Only summarize existing CSVs
bash ./run_eval.sh --base-dir ./output/PRIMARY/BSRNN --test-set PRIMARY --cuda 0 2
```

This sequentially runs:

1. `TER` via `eval/transcribe_and_evaluation.sh`
2. `TER_ASR2_AED` via `eval/transcribe_and_evaluation_asr2.sh`
3. `TSE timing` via `eval/vad_and_evaluation.sh`
4. `speaker similarity (tse_enrol)` via `eval/compute_spk_similarity.sh`
5. `speaker similarity (mixture_enrol)` via `eval/compute_spk_similarity.sh`
6. `DNSMOS` via `eval/compute_dnsmos.sh`

Use `--include-fisher` if the target output directory also contains `Fisher`.

## Shared Conventions

- All commands below are intended to be run from the REAL-T repo root.
- `BASE_DIRS` is a space-separated list of TSE output roots such as `./output/PRIMARY/BSRNN`.
- `TEST_SET_DIR` should point to `./datasets/REAL-T/PRIMARY` or `./datasets/REAL-T/BASE`.
- `DATASETS` defaults to `AliMeeting AISHELL-4 AMI DipCo CHiME6 Fisher`.
- All eval shell scripts source `env_setup.sh` automatically.
- `run_eval.sh` sets one `CUDA_VISIBLE_DEVICES` value for the entire pipeline and forces ONNX-based stages onto CUDA with `WESPEAKER_PROVIDER=cuda` and `DNSMOS_PROVIDER=cuda`.
- `run_eval.sh` accepts both absolute and relative `--base-dir` paths.

Expected summary outputs under each `BASE_DIR`:

- `{BASE_NAME}_TER.csv` and `{BASE_NAME}_TER.txt`
- `{BASE_NAME}_TER_ASR2_AED.csv` and `{BASE_NAME}_TER_ASR2_AED.txt`
- `{BASE_NAME}_TSE_TIMING.csv` and `{BASE_NAME}_TSE_TIMING.txt`
- `{BASE_NAME}_spk_similarity.csv` and `{BASE_NAME}_spk_similarity_summary.txt`
- `{BASE_NAME}_spk_similarity_mixture_enrol.csv` and `{BASE_NAME}_spk_similarity_mixture_enrol_summary.txt`
- `{BASE_NAME}_dnsmos.csv` and `{BASE_NAME}_dnsmos.txt`
- `{BASE_NAME}_summary.txt`

`{BASE_NAME}_summary.txt` is the new aggregated report. It is recomputed from CSV files and contains two mean-only tables:

- `Mean by dataset`: typically 5 rows for `AISHELL-4 / AMI / AliMeeting / CHiME6 / DipCo`
- `Mean by language`: typically 2 rows for `en / chs`

Its columns are organized as grouped headers:

- `TER`
  - `fireredasr-1/whisper`
  - `fireredasr-2`
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

- `TER / fireredasr-1/whisper`: mean `wer_or_cer` from `{BASE_NAME}_TER.csv`
- `TER / fireredasr-2`: mean `wer_or_cer` from `{BASE_NAME}_TER_ASR2_AED.csv`
- `SIM / enrol-mixture`: mean `speaker_cosine_similarity` from `{BASE_NAME}_spk_similarity_mixture_enrol.csv`
- `SIM / enrol-tse`: mean `speaker_cosine_similarity` from `{BASE_NAME}_spk_similarity.csv`
- `DNSMOS / *`: mean `SIG / BAK / OVRL / P808` from `{BASE_NAME}_dnsmos.csv`
- `RATIO / precision, recall, f1`: mean `precision / recall / f1` from `{BASE_NAME}_TSE_TIMING.csv`

## Prerequisites

Evaluation requires the `REAL-T` dataset and the ASR checkpoint `FireRedASR-AED-L` plus `whisper-large-v2`. To prepare the dataset and standard ASR models:

```bash
bash -i ./pre.sh
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

Timing evaluation also requires overlap JSON copied from `REAL-T-Ext-channel-re-seclection`:

```bash
mkdir -p ./datasets/REAL-T/json
cp -r /path/to/REAL-T-Ext-channel-re-seclection/output/REAL-T-datasets/json/* ./datasets/REAL-T/json/
```

### FireRedASR2-AED for TER_ASR2_AED

`eval/transcribe_and_evaluation_asr2.sh` expects local weights in:

```bash
./FireRedASR2S/pretrained_models/FireRedASR2-AED
```

Download with either ModelScope or Hugging Face:

```bash
pip install -U modelscope
modelscope download --model xukaituo/FireRedASR2-AED --local_dir ./FireRedASR2S/pretrained_models/FireRedASR2-AED
```

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download FireRedTeam/FireRedASR2-AED --local-dir ./FireRedASR2S/pretrained_models/FireRedASR2-AED
```

The directory must contain:

- `model.pth.tar`
- `cmvn.ark`
- `dict.txt`
- `train_bpe1000.model`

### DNSMOS

`eval/compute_dnsmos.sh` uses `./DNSMOS` by default. If the ONNX files are missing, mode 1 auto-downloads them unless `DNSMOS_NO_DOWNLOAD=1`.

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

- `BASE_DIRS`
- `TEST_SET_DIR`
- `INCLUDING_FISHER`
- `DATASETS`
- `CHINESE_DATASETS`
- `ENGLISH_DATASETS`
- `ASR_DEVICE`
- `MAPPING_CSV_NAME`

### ASR2 TER

`eval/transcribe_and_evaluation_asr2.sh` uses vendored `FireRedASR2-AED` for all datasets.

```bash
# Only ASR2 transcription
bash -i ./eval/transcribe_and_evaluation_asr2.sh 1

# Only evaluation
bash -i ./eval/transcribe_and_evaluation_asr2.sh 2

# Both
bash -i ./eval/transcribe_and_evaluation_asr2.sh 1 2
```

Important env vars:

- `BASE_DIRS`
- `TEST_SET_DIR`
- `DATASETS`
- `MAPPING_CSV_NAME`
- `FIREREDASR2S_ROOT`
- `FIREREDASR2_MODEL_DIR`
- `USE_GPU`
- `USE_HALF`
- `ASR_BATCH_SIZE`

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

- `BASE_DIRS`
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

Mode `1` writes `FireRedVAD/vad_segments.jsonl` under each dataset directory. Mode `2` writes `FireRedVAD/label_segments.jsonl` plus `{BASE_NAME}_TSE_TIMING.csv` and `{BASE_NAME}_TSE_TIMING.txt`.

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

- `BASE_DIRS`
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

- `BASE_DIRS`
- `TEST_SET_DIR`
- `DNSMOS_MODEL_DIR`
- `DNSMOS_PROVIDER`
- `DNSMOS_NO_DOWNLOAD`
- `MAX_SAMPLES`

## Aggregated Summary Internals

The aggregated report is generated by:

```bash
python3 ./utils/aggregate_eval_summary.py --base_dir ./output/PRIMARY/BSRNN
```

You usually do not need to call it directly, because `run_eval.sh ... 2` already wraps it.

The script expects the following CSV files to exist under `BASE_DIR`:

- `{BASE_NAME}_TER.csv`
- `{BASE_NAME}_TER_ASR2_AED.csv`
- `{BASE_NAME}_spk_similarity.csv`
- `{BASE_NAME}_spk_similarity_mixture_enrol.csv`
- `{BASE_NAME}_dnsmos.csv`
- `{BASE_NAME}_TSE_TIMING.csv`

If any of them is missing, summary generation will stop with an error so the missing stage is visible immediately.
