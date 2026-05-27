"""Unified ASR inference driver.

Reads an ``audio_mapping`` CSV (``utterance``/``path``), runs the chosen
backend through the central registry, and writes
``<output_dir>/predicted.csv`` with columns ``[utterance, transcript]``
so the existing ``utils/asr_evaluation.py`` keeps working.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Allow ``python3 asr/asr_inference.py`` and ``python3 asr_inference.py``
# (from inside asr/) to both work by making the package importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from asr import (  # noqa: E402  (must follow sys.path setup)
    DATASET_LANGUAGE,
    get_asr_model,
    init_cache_env,
    list_models,
    truncate_repetition,
)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402


# Configure deterministic ops the same way the original script did.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    # Some torch builds raise if a kernel doesn't have a deterministic impl.
    pass
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# Keep the legacy module-level ``language`` dict for backwards compatibility
# (some downstream scripts import it).
language = DATASET_LANGUAGE


def get_asr_model_compat(model_name: str, device: str):
    """Thin wrapper preserving the old function name for external callers."""
    return get_asr_model(model_name, device=device)


def main() -> None:
    # Short-circuit ``--list_models`` before argparse so users can run it
    # without supplying the otherwise-required arguments.
    if "--list_models" in sys.argv[1:]:
        print("Registered ASR backends:")
        for name in sorted(list_models()):
            print(f"  - {name}")
        return

    parser = argparse.ArgumentParser(description="Unified ASR Inference Script")
    parser.add_argument(
        "--audio_mapping",
        "--audio_mapping_csv",
        dest="audio_mapping_csv",
        type=str,
        required=True,
        help="Path to the audio mapping CSV file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "ASR Model to use. Built-in: zipformer-en, zipformer-zh. "
            "See asr/registry.py for aliases."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset (used to pick zh/en language)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the transcription result",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device hint, e.g. 'cuda:0' or 'cpu'. The sherpa-onnx CPU "
             "build is used regardless; the flag is kept for API parity.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of rows to transcribe for smoke tests.",
    )
    parser.add_argument(
        "--repeat_max",
        type=int,
        default=4,
        help="Trigger threshold for the repetition detector "
             "(>=N consecutive same n-grams).",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help=(
            "Enable repetition-truncation post-processing on the predicted "
            "transcript. Default OFF, matching historical behaviour where "
            "predicted.csv contains the raw model output."
        ),
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all registered ASR backends and exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Registered ASR backends:")
        for name in sorted(list_models()):
            print(f"  - {name}")
        return

    cache_root = init_cache_env()
    print(f"[cache] ASR_CACHE_DIR = {cache_root}")

    os.makedirs(args.output_dir, exist_ok=True)
    predicted_csv = os.path.join(args.output_dir, "predicted.csv")

    lang = DATASET_LANGUAGE.get(args.dataset_name)
    if lang is None:
        raise KeyError(
            f"Unknown dataset_name {args.dataset_name!r}; expected one of "
            f"{sorted(DATASET_LANGUAGE.keys())}"
        )

    print(f"[asr] loading model: {args.model_name}")
    model = get_asr_model(args.model_name, device=args.device)

    audio_mapping = pd.read_csv(args.audio_mapping_csv)
    if args.max_samples is not None:
        audio_mapping = audio_mapping.head(args.max_samples)

    print(f"Processing {len(audio_mapping)} audio files with {args.model_name} ...")

    results = []
    for _, row in tqdm(
        audio_mapping.iterrows(),
        total=len(audio_mapping),
        desc=f"ASR[{args.model_name}]",
    ):
        audio_path = row["path"]
        raw_text = model.transcribe(audio_path, language=lang)

        if args.postprocess:
            final_text, _, _ = truncate_repetition(
                raw_text, language=lang, max_repeat=args.repeat_max
            )
        else:
            final_text = raw_text

        results.append({"utterance": row["utterance"], "transcript": final_text})

    if len(results) != len(audio_mapping):
        failed_count = len(audio_mapping) - len(results)
        raise RuntimeError(
            f"Some files failed to transcribe. {failed_count} audio files "
            "were not processed successfully."
        )

    pd.DataFrame(results).to_csv(predicted_csv, index=False)
    print(f"[asr] done. predicted: {predicted_csv}")


if __name__ == "__main__":
    main()
