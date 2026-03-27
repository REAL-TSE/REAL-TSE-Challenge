# REAL-T: Real Conversational Mixtures for Target Speaker Extraction

<p align="center">
  <img src="./figure/logo.drawio.svg" alt="REAL-T Logo" width="200"/>
</p>

<p align="center">
  <a href="xxxxxxxx">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="Paper">
  </a>
  <a href="https://real-tse.github.io/">
    <img src="https://img.shields.io/badge/REAL--T-Page-blue" alt="REAL-T Page">
  </a>
</p>

![Pipeline](./figure/pipeline.svg)


## 1. Introduction

Target Speaker Extraction (TSE) models have demonstrated impressive performance on synthetic datasets such as **LibriMix** and **WSJMix**. However, these benchmarks lack the acoustic realism and conversational dynamics of actual human interactions — such as spontaneous speech, overlapping turns, and environmental noise — limiting their relevance to real-world scenarios.

Efforts like **REAL-M** and **LibriCSS** have attempted to bridge this gap. REAL-M collects simultaneous speech in shared environments through multi-speaker read-aloud sessions, while LibriCSS re-records isolated utterances via synchronized playback. Though valuable, these datasets still fall short of capturing the nuances of real conversations — including irregular speaker turns, sporadic utterances, and authentic background conditions.

To address these limitations, we introduce **REAL-T**, the first conversation-centric dataset specifically designed for TSE under real-world conditions. Built from speaker diarization corpora, REAL-T naturally includes overlapping speech, enrollment-ready segments, and complex conversational behaviors.

Key features of REAL-T include:

- **Multi-lingual**: English and Mandarin recordings
- **Multi-genre**: Covering diverse conversational scenarios
- **Multi-enrollment**: Multiple enrollment utterance from different parts of the conversation

REAL-T now ships a single official evaluation split:

- **DEV**: The conversation-centric development benchmark used throughout this repository

Evaluations reveal that existing TSE models suffer significant performance degradation on REAL-T, highlighting the need for more robust approaches tailored to real conversational speech.

For more details, refer to our paper: [REAL-T Paper](xxxxxxxxx)


## 2. Installation

### 2.1 Clone the repository

```bash
git clone https://github.com/REAL-TSE/REAL-T.git
cd REAL-TSE-Challenge

# install submodules (wesep + FireRedASR2S)
git submodule update --init --recursive
```

### 2.2 Create a Conda environment and install dependencies

```bash
conda create -n REAL-T python=3.10
conda activate REAL-T
pip install -r requirements.txt
# Reinstall GPU ORT last so silero-vad / wespeakerruntime do not leave CPU ORT active.
pip install --force-reinstall --no-deps onnxruntime-gpu==1.19.2
```

`requirements.txt` is the only supported Python dependency entrypoint for this repo. For RTX 5090 / `sm_120`, it resolves the `cu128` PyTorch wheels automatically. If you encounter any issues, try using `requirements2.txt` instead. `wespeaker` remains the only GitHub dependency because local `wesep` imports it directly.

All top-level scripts source `env_setup.sh` by default. That helper activates `REAL-T` and appends the local `FireRedASR` / `FireRedASR2S` / `wesep` paths automatically. If you want to use a different env name temporarily, run them with `REALT_CONDA_ENV=<your_env_name>`.

### 2.3 Set up Linux PATH and PYTHONPATH

> Please replace `$PWD` below with the absolute path to this project (REAL-T repo root).

FireRedASR (ASR transcription) and **FireRedASR2S** (e.g. FireRedVAD) are expected under the REAL-T repo root. Initialize/update the submodules first, then add the repo roots to `PYTHONPATH` so that `import fireredasr` / `import fireredasr2s` work.

```
$ export PATH=$PWD/FireRedASR/fireredasr/:$PWD/FireRedASR/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/FireRedASR/:$PYTHONPATH
$ export PYTHONPATH=$PWD/FireRedASR2S/:$PYTHONPATH
$ export PYTHONPATH=$PWD/wesep/:$PYTHONPATH
```


### 2.4 Prepare Dataset and Checkpoints

`pre.sh` is the recommended one-command preparation entrypoint. By default, it prepares the dataset and all model weights required by evaluation.

Run with Google Drive file id:

```bash
REALT_DATASET_GDRIVE_FILE_ID=<google_drive_file_id> bash -i ./pre.sh
```

Or run with a Google Drive sharing URL:

```bash
REALT_DATASET_GDRIVE_URL='https://drive.google.com/file/d/.../view?usp=sharing' bash -i ./pre.sh
```

`pre.sh` supports 5 optional switches (all default to `1`):

- `REALT_PREP_DOWNLOAD_DATASET`
- `REALT_PREP_DOWNLOAD_FIRERED_ASR`
- `REALT_PREP_DOWNLOAD_WHISPER`
- `REALT_PREP_DOWNLOAD_FIRERED_VAD`
- `REALT_PREP_DOWNLOAD_DNSMOS`

After a default run, files are prepared at:

- Dataset root: `./datasets/REAL-T`
- Dataset mapping: `./datasets/REAL-T/mapping.csv`
- FireRedASR-AED-L: `./FireRedASR/pretrained_models/FireRedASR-AED-L`
- Whisper large-v2: `./whisper/pretrained_models/whisper-large-v2`
- FireRedVAD: `./FireRedASR2S/pretrained_models/FireRedVAD/VAD`
- DNSMOS ONNX: `./DNSMOS/sig_bak_ovr.onnx` and `./DNSMOS/model_v8.onnx`

Existing files are reused when possible, so repeated runs are safe.

## 3. Inference and Evaluation

### 3.1 TSE Inference on REAL-T

The `run_tse.sh` script below demonstrates how to perform TSE inference with the [Wesep toolkit](https://github.com/wenet-e2e/wesep) using a **BSRNN model** trained on **VoxCeleb1**. You can adapt its `input/output` structure to suit your own TSE model.

```bash
cd REAL-TSE-Challenge
bash -i run_tse.sh
```

This script runs TSE inference for multiple datasets using a specified model. Each dataset will be processed individually, generating separated target speaker audio files.

| **Variable Name** | **Description** |
| :--- | :--- |
| `MODEL_NAME 🚩` | Name of the TSE model used for inference (e.g., `bsrnn_vox1`). |
| `DATASETS 🚩` | List of datasets to process (e.g., AliMeeting, AMI, CHiME6, AISHELL-4, DipCo). |
| `TEST_SET 🚩` | Test subset to use: `DEV`. |
| `DEVICE 🚩` | Device on which to run inference (`cuda` for GPU, `cpu` for CPU). |
| `DATASET_ROOT` | Root directory containing metadata CSV files for each dataset. |
| `OUTPUT_ROOT` | Directory where the separated audios will be saved. |
| `TSE_SCRIPT` | Path to the TSE inference Python script (`tse.py`). |
| `META_CSV_PATH` | Path to the CSV file containing mixture and enrolment utterance metadata. |
| `UTTERANCE_MAP_CSV` | Path to the CSV mapping enrolment utterances to mixture utterances. |
| `OUTPUT_DIR` | Directory where output audios for each dataset will be stored. |

---

### 3.2 One-Click Evaluation

The recommended evaluation entrypoint is now `run_eval.sh` at the repo root. It runs the full evaluation pipeline sequentially on one `OUTPUT_DIR`, using one CUDA device for all stages.

```bash
cd REAL-TSE-Challenge
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0
```

`run_eval.sh` supports three common usages:

```bash
# 1 2: run all evaluation sub-scripts, then summarize
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0 1 2

# 1: only run all evaluation sub-scripts
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0 1

# 2: only summarize existing CSV results
bash ./run_eval.sh --output-dir ./output/DEV/BSRNN --test-set DEV --cuda 0 2
```

If no mode is provided, the default is `1 2`.

By default, `run_eval.sh` runs:

1. `TER`
2. `RATIP (TSE timing)`
3. `speaker similarity (tse_enrol)`
4. `speaker similarity (mixture_enrol)`
5. `DNSMOS`

Expected outputs under the chosen `OUTPUT_DIR`:

- Detailed metric files under `eval_metrics/` (configurable via `EVAL_METRICS_SUBDIR`):
  - `eval_metrics/{OUTPUT_NAME}_TER.csv` and `eval_metrics/{OUTPUT_NAME}_TER.txt`
  - `eval_metrics/{OUTPUT_NAME}_TSE_TIMING.csv` and `eval_metrics/{OUTPUT_NAME}_TSE_TIMING.txt`
  - `eval_metrics/{OUTPUT_NAME}_spk_similarity.csv` and `eval_metrics/{OUTPUT_NAME}_spk_similarity_summary.txt`
  - `eval_metrics/{OUTPUT_NAME}_spk_similarity_mixture_enrol.csv` and `eval_metrics/{OUTPUT_NAME}_spk_similarity_mixture_enrol_summary.txt`
  - `eval_metrics/{OUTPUT_NAME}_dnsmos.csv` and `eval_metrics/{OUTPUT_NAME}_dnsmos.txt`
- Aggregated report at the output directory root:
  - `{OUTPUT_NAME}_summary.txt`

`{OUTPUT_NAME}_summary.txt` is a compact aggregated report recomputed from the metric CSV files above. It contains:

- `Mean by dataset`
- `Mean by language`

with grouped columns:

- `TER`: `fireredasr-1/whisper`
- `SIM`: `enrol-mixture`, `enrol-tse`
- `DNSMOS`: `SIG`, `BAK`, `OVRL`, `P808`
- `RATIO`: `precision`, `recall`, `f1`

At the moment, `RATIO` is fully sourced from `eval_metrics/{OUTPUT_NAME}_TSE_TIMING.csv`, using the mean `precision`, `recall`, and `f1`.

Detailed per-metric instructions, prerequisites, and optional visualization are now documented in [`eval/README.md`](./eval/README.md).

### 3.3 TSE Inference vs Eval

- Use `run_tse.sh` for TSE inference.
- Use `run_eval.sh` for the full evaluation pipeline.
- Use scripts under `./eval/` only when you want to run individual evaluation sub-steps manually.

## 4. Results

We evaluate four BSRNN-based TSE models with different speaker information fusion strategies (speaker embedding vs. time-frequency featuremap interaction) and causality (causal vs. non-causal), all trained on Libri2Mix-100. The table below compares their performance on the simulated dev sets.


<div align="center">

| Model        | TER (fireredasr-1/whisper) | SIM (enrol-mixture) | SIM (enrol-tse) | DNSMOS SIG | DNSMOS BAK | DNSMOS OVRL | RATIO P | RATIO R | RATIO F1 |
|--------------|---------------------------|---------------------|-----------------|------------|------------|-------------|-------------|---------|---------|
| BSRNN_EMB    |                      0.770 |               0.506 |           0.501 |       2.15 |        1.90 |        1.66 |    0.780 |   0.946 |    0.841 |
| BSRNN_EMB_CAUSAL |                     0.788 |               0.506 |           0.492 |       2.09 |       1.92 |        1.63 |   0.781 |    0.920 |    0.829 |
| BSRNN_TFMAP  |                     0.766 |               0.506 |           0.521 |        1.90 |       1.66 |         1.50 |   0.776 |   0.946 |    0.838 |
| BSRNN_TFMAP_CAUSAL |                     0.744 |               0.506 |           0.535 |       1.99 |       1.72 |        1.56 |   0.779 |   0.952 |    0.844 |

</div>


The checkpoints is availibale at [Google Drive](https://drive.google.com/uc?export=download&id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC) . The directory structure for the pretrained models in the REAL-T project is suggested to be:

```
REAL-T/
├── pretrained/
│ ├── spk_emb_100/
│ │ ├── avg_model.pt
│ │ └── config.yaml
│ ├── spk_emb_causal_100/
│ ├── tfmap_context_100/
│ └── tfmap_context_causal_100/
```

## 5. Citation

```
@inproceedings{li25da_interspeech,
  title     = {{REAL-T: Real Conversational Mixtures for Target Speaker Extraction}},
  author    = {{Shaole Li and Shuai Wang and Jiangyu Han and Ke Zhang and Wupeng Wang and Haizhou Li}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{1923--1927}},
  doi       = {{10.21437/Interspeech.2025-2662}},
  issn      = {{2958-1796}},
}
```


## 6. Contact

For any questions, please contact: `shuaiwang@nju.edu.cn`
