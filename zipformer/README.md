# Zipformer ASR Models (sherpa-onnx)

This directory hosts the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
pretrained Zipformer transducer checkpoints that REAL-TSE-Challenge uses for
TER evaluation.

## Layout

```
zipformer/
├── README.md
└── pretrained_models/
    ├── sherpa-onnx-zipformer-gigaspeech-2023-12-12/        # English (GigaSpeech 10000h)
    │   ├── encoder-epoch-30-avg-1.onnx
    │   ├── decoder-epoch-30-avg-1.onnx
    │   ├── joiner-epoch-30-avg-1.onnx
    │   ├── tokens.txt
    │   ├── bpe.model
    │   └── test_wavs/0.wav, 1.wav, 8k.wav
    └── sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/       # Chinese (WenetSpeech + AISHELL-{1,2,4} + AliMeeting + KeSpeech + MagicData-RAMC)
        ├── encoder-epoch-20-avg-1.onnx
        ├── decoder-epoch-20-avg-1.onnx
        ├── joiner-epoch-20-avg-1.onnx
        ├── tokens.txt
        └── test_wavs/0.wav, 1.wav, 2.wav
```

The `asr/backends/zipformer_sherpa.py` backend auto-discovers the encoder /
decoder / joiner / tokens.txt inside each model directory so you do NOT
need to know the exact epoch number.

## Download

### Option A — `pre.sh` (recommended, downloads the default model set)

```bash
bash -i ./pre.sh
```

`pre.sh` downloads both Zipformer models by default, along with the other
default evaluation assets. To disable one Zipformer model:

```bash
REALT_PREP_DOWNLOAD_ZIPFORMER_EN=0 bash -i ./pre.sh
REALT_PREP_DOWNLOAD_ZIPFORMER_ZH=0 bash -i ./pre.sh
```

### Option B — standalone downloader

```bash
# Both models
python utils/download_zipformer.py

# Only English / only Chinese
python utils/download_zipformer.py --only en
python utils/download_zipformer.py --only zh
```

The script streams the `.tar.bz2` tarballs from
`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/`,
strips the leading `<release_name>/` so files land directly under
`zipformer/pretrained_models/<release_name>/`, and skips already-populated
targets.

## Manual download (no Python helper)

```bash
mkdir -p zipformer/pretrained_models
cd zipformer/pretrained_models

for name in \
    sherpa-onnx-zipformer-gigaspeech-2023-12-12 \
    sherpa-onnx-zipformer-multi-zh-hans-2023-9-2; do
    wget -O "$name.tar.bz2" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$name.tar.bz2"
    tar xjf "$name.tar.bz2"
    rm "$name.tar.bz2"
done
```

## Smoke test

A tiny end-to-end check that does NOT need REAL-T data:

```bash
python tests/test_zipformer_smoke.py
```

It loads both backends through the registry and runs `transcribe()` on the
`test_wavs/0.wav` shipped inside each model directory.
