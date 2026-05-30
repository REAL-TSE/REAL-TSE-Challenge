import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def import_firered_vad(fireredasr2s_root: str):
    """Import FireRedVAD from fireredasr2s. Prefer PYTHONPATH=$PWD/FireRedASR2S (see README); else add fireredasr2s_root to sys.path."""
    root = Path(fireredasr2s_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"FireRedASR2S root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
    except ImportError as exc:
        raise ImportError(
            "Failed to import FireRedVAD from fireredasr2s. "
            "Please check --fireredasr2s_root path."
        ) from exc
    return FireRedVad, FireRedVadConfig


def resolve_audio_path(audio_path: str, mapping_path: Path) -> str:
    raw_path = Path(audio_path)
    if raw_path.is_absolute():
        return str(raw_path)
    candidate = (Path.cwd() / raw_path).resolve()
    if candidate.exists():
        return str(candidate)
    candidate = (mapping_path.parent / raw_path).resolve()
    return str(candidate)


def peak_normalize_audio(audio_path: str, peak: float = 0.9):
    """Read an audio file and peak-normalize it to ``peak`` in the float
    domain by uniform scaling, then convert it back to int16 PCM range so
    that it can be fed directly into FireRedVAD.

    FireRedVAD's ``AudioFeat`` reads files with ``sf.read(..., dtype="int16")``
    and the downstream kaldi-style fbank / cmvn are calibrated for that
    amplitude scale. To preserve this scale while still applying our peak
    normalization, we scale in the float ``[-1, 1]`` domain first (so that
    the new peak amplitude is exactly ``peak``) and then multiply by 32767
    and cast to int16.

    Multi-channel inputs are mixed down to mono by averaging across
    channels (FireRedVAD's fbank requires a 1-D waveform). Silent inputs
    (``max(|wav|) == 0``) are returned as zeros, skipping the scaling but
    keeping a uniform downstream flow.

    Returns
    -------
    (sample_rate, wav_int16) : tuple
        Ready to be passed as ``vad.detect((sr, wav_int16))``. Using the
        tuple form makes ``AudioFeat`` assert ``sr == 16000`` instead of
        silently assuming 16 kHz when a bare ndarray is given.
    """
    wav, sr = sf.read(audio_path, always_2d=False, dtype="float32")
    if wav.ndim > 1:  # mix-down to mono, fbank requires a 1-D waveform
        wav = wav.mean(axis=1)
    max_abs = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    if max_abs > 0:
        wav = wav * (peak / max_abs)
    wav_int16 = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
    return sr, wav_int16


def main():
    parser = argparse.ArgumentParser(description="Run FireRedVAD inference for TSE outputs.")
    parser.add_argument("--audio_mapping", type=str, required=True, help="Path to tse_audio_mapping.csv")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output jsonl path")
    parser.add_argument("--model_dir", type=str, required=True, help="FireRedVAD model directory")
    parser.add_argument(
        "--fireredasr2s_root",
        type=str,
        default="./FireRedASR2S",
        help="Root path of FireRedASR2S repository (clone under REAL-T)",
    )
    parser.add_argument("--use_gpu", type=int, default=1, help="Use GPU if 1, else CPU")
    parser.add_argument("--speech_threshold", type=float, default=0.5, help="VAD speech threshold")
    parser.add_argument("--smooth_window_size", type=int, default=5)
    parser.add_argument("--min_speech_frame", type=int, default=20)
    parser.add_argument("--max_speech_frame", type=int, default=2000)
    parser.add_argument("--min_silence_frame", type=int, default=10)
    parser.add_argument("--merge_silence_frame", type=int, default=50)
    parser.add_argument("--extend_speech_frame", type=int, default=5)
    parser.add_argument("--chunk_max_frame", type=int, default=30000)
    args = parser.parse_args()

    audio_mapping_path = Path(args.audio_mapping).resolve()
    output_jsonl_path = Path(args.output_jsonl).resolve()
    model_dir = Path(args.model_dir).resolve()

    if not audio_mapping_path.exists():
        raise FileNotFoundError(f"audio_mapping not found: {audio_mapping_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    FireRedVad, FireRedVadConfig = import_firered_vad(args.fireredasr2s_root)

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    mapping_df = pd.read_csv(audio_mapping_path)
    if "utterance" not in mapping_df.columns or "path" not in mapping_df.columns:
        raise ValueError("audio_mapping csv must include columns: utterance,path")

    vad_config = FireRedVadConfig(
        use_gpu=bool(args.use_gpu),
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        min_speech_frame=args.min_speech_frame,
        max_speech_frame=args.max_speech_frame,
        min_silence_frame=args.min_silence_frame,
        merge_silence_frame=args.merge_silence_frame,
        extend_speech_frame=args.extend_speech_frame,
        chunk_max_frame=args.chunk_max_frame,
    )
    vad = FireRedVad.from_pretrained(str(model_dir), vad_config)

    results = []
    print(f"Processing {len(mapping_df)} files with FireRedVAD...")
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="FireRedVAD"):
        utterance = row["utterance"]
        rel_audio_path = row["path"]
        audio_path = resolve_audio_path(rel_audio_path, audio_mapping_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        # Peak-normalize in the float domain to 0.9, then feed the
        # int16 PCM tuple directly into FireRedVAD. Passing a (sr, wav)
        # tuple lets AudioFeat assert sr == 16000 instead of silently
        # assuming 16 kHz when a bare ndarray is given.
        sr,sr_wav = peak_normalize_audio(audio_path, peak=0.9)
        result, _ = vad.detect(sr_wav)
        pred_segments = result.get("timestamps", [])
        results.append(
            {
                "utterance": utterance,
                "path": rel_audio_path,
                "duration": result.get("dur", None),
                "pred_segments": pred_segments,
                "vad_config": {
                    "speech_threshold": args.speech_threshold,
                    "smooth_window_size": args.smooth_window_size,
                    "min_speech_frame": args.min_speech_frame,
                    "max_speech_frame": args.max_speech_frame,
                    "min_silence_frame": args.min_silence_frame,
                    "merge_silence_frame": args.merge_silence_frame,
                    "extend_speech_frame": args.extend_speech_frame,
                    "chunk_max_frame": args.chunk_max_frame,
                    "use_gpu": bool(args.use_gpu),
                },
            }
        )

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} VAD entries to: {output_jsonl_path}")


if __name__ == "__main__":
    main()
