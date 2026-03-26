import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from asr_models import FireRedASR2_AED_ASRModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIREREDASR2S_ROOT = PROJECT_ROOT / "FireRedASR2S"
DEFAULT_FIREREDASR2_MODEL_DIR = (
    DEFAULT_FIREREDASR2S_ROOT / "pretrained_models" / "FireRedASR2-AED"
)
REQUIRED_MODEL_FILES = ("model.pth.tar", "cmvn.ark", "dict.txt", "train_bpe1000.model")


def validate_model_dir(model_dir: Path) -> Path:
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"FireRedASR2-AED model directory is incomplete: {model_dir}. "
            f"Missing files: {', '.join(missing)}"
        )
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="FireRedASR2-AED local inference")
    parser.add_argument(
        "--audio_mapping",
        "--audio_mapping_csv",
        dest="audio_mapping_csv",
        type=str,
        required=True,
        help="Path to the audio mapping CSV (columns: utterance, path)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save predicted.csv"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name kept for CLI compatibility; not used for model selection.",
    )
    parser.add_argument(
        "--fireredasr2s_root",
        type=str,
        default=str(DEFAULT_FIREREDASR2S_ROOT),
        help="Root path of the vendored FireRedASR2S repository inside REAL-T.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(DEFAULT_FIREREDASR2_MODEL_DIR),
        help="FireRedASR2-AED model directory under the REAL-T project.",
    )
    parser.add_argument("--use_gpu", type=int, default=1, help="Use GPU if available.")
    parser.add_argument("--use_half", type=int, default=0, help="Use fp16 on GPU.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for local ASR.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of rows to transcribe for smoke tests.")
    args = parser.parse_args()

    del args.dataset_name

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "predicted.csv"

    audio_mapping_path = Path(args.audio_mapping_csv).resolve()
    fireredasr2s_root = Path(args.fireredasr2s_root).resolve()
    model_dir = validate_model_dir(Path(args.model_dir).resolve())

    if not fireredasr2s_root.is_dir():
        raise FileNotFoundError(f"FireRedASR2S root not found: {fireredasr2s_root}")

    df = pd.read_csv(audio_mapping_path)
    if "utterance" not in df.columns or "path" not in df.columns:
        raise ValueError("CSV must have columns 'utterance' and 'path'")
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    model = FireRedASR2_AED_ASRModel(
        model_path=str(model_dir),
        use_gpu=bool(args.use_gpu),
        use_half=bool(args.use_half),
        batch_size=args.batch_size,
    )

    results = []
    batch_utterances = []
    batch_audio_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FireRedASR2-AED"):
        utterance = row["utterance"]
        audio_path = Path(row["path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        batch_utterances.append(utterance)
        batch_audio_paths.append(str(audio_path))

        if len(batch_audio_paths) < args.batch_size:
            continue

        transcripts = model.transcribe_batch(batch_utterances, batch_audio_paths)
        results.extend(
            {"utterance": utt, "transcript": transcript}
            for utt, transcript in zip(batch_utterances, transcripts)
        )
        batch_utterances = []
        batch_audio_paths = []

    if batch_audio_paths:
        transcripts = model.transcribe_batch(batch_utterances, batch_audio_paths)
        results.extend(
            {"utterance": utt, "transcript": transcript}
            for utt, transcript in zip(batch_utterances, transcripts)
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    print(f"Saved {len(results)} transcriptions to {output_file}")


if __name__ == "__main__":
    main()
