"""Smoke test for the new sherpa-onnx Zipformer ASR backends.

Loads ``zipformer-en`` and ``zipformer-zh`` through the registry and runs
``transcribe()`` on the ``test_wavs/0.wav`` shipped inside each model
directory. Asserts the output is a non-empty string.

This test does NOT depend on the REAL-T dataset and runs on CPU.

Usage
-----
    python tests/test_zipformer_smoke.py

Exits with code 0 on success, 1 on any failure. Skips a backend cleanly
if the model files were not yet downloaded (so it can be wired into CI
without forcing a model download).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from asr import get_asr_model  # noqa: E402


_PRETRAINED_ROOT = _REPO_ROOT / "zipformer" / "pretrained_models"

_CASES = [
    {
        "name": "zipformer-en",
        "release": "sherpa-onnx-zipformer-gigaspeech-2023-12-12",
        # english model ships LibriSpeech-style sample names; we glob for
        # any *.wav under test_wavs/ and pick the first match.
        "wav_glob": "test_wavs/*.wav",
        "language": "en",
    },
    {
        "name": "zipformer-zh",
        "release": "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2",
        "wav_glob": "test_wavs/0.wav",
        "language": "zh",
    },
]


def _resolve_wav(model_dir: Path, glob_pat: str) -> Path | None:
    hits = sorted(model_dir.glob(glob_pat))
    return hits[0] if hits else None


def _run_one(case: dict) -> tuple[bool, str]:
    """Returns (passed, message)."""
    model_dir = _PRETRAINED_ROOT / case["release"]

    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        return False, (
            f"[skip] model dir missing: {model_dir}\n"
            "        run `python utils/download_zipformer.py "
            f"--only {case['language']}` first"
        )

    wav_path = _resolve_wav(model_dir, case["wav_glob"])
    if wav_path is None or not wav_path.is_file():
        return False, (
            f"[fail] no sample wav matched {case['wav_glob']!r} under {model_dir}"
        )

    try:
        model = get_asr_model(case["name"], device="cpu")
    except Exception as exc:
        return False, f"[fail] {case['name']} failed to load: {exc!r}"

    try:
        text = model.transcribe(str(wav_path), language=case["language"])
    except Exception as exc:
        return False, f"[fail] {case['name']} transcribe raised: {exc!r}"

    if not isinstance(text, str):
        return False, f"[fail] {case['name']} returned non-str: {type(text)!r}"
    if not text.strip():
        return False, f"[fail] {case['name']} returned empty transcript"
    return True, f"[ok]   {case['name']:<14s} -> {text!r} ({wav_path.name})"


def main() -> int:
    print(f"[smoke] repo root: {_REPO_ROOT}")
    print(f"[smoke] pretrained root: {_PRETRAINED_ROOT}")
    print()

    results = []
    skipped = 0
    failed = 0

    for case in _CASES:
        ok, msg = _run_one(case)
        print(msg)
        if not ok:
            if msg.startswith("[skip]"):
                skipped += 1
            else:
                failed += 1
        results.append((case["name"], ok, msg))

    print()
    if failed:
        print(f"[smoke] FAILED: {failed} hard failure(s), {skipped} skipped")
        return 1
    if skipped == len(_CASES):
        print("[smoke] all cases skipped (no model dirs populated). "
              "Run download script first.")
        return 1
    print(f"[smoke] PASSED ({skipped} skipped, "
          f"{len(_CASES) - skipped - failed} ran cleanly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
