#!/usr/bin/env python3
"""Download sherpa-onnx Zipformer ASR checkpoints for REAL-TSE-Challenge.

This is a focused replacement for the historical
``utils/download_asr_model.py``, which used to download FireRedASR-AED-L
and Whisper-large-v2 from Hugging Face. The current ASR layer uses
sherpa-onnx Zipformer transducers instead.

Each model is a ``.tar.bz2`` published as a GitHub Release asset under
``k2-fsa/sherpa-onnx`` (tag ``asr-models``). We stream the tarball, strip
the leading ``<release_name>/`` directory, and extract the ONNX shards
plus ``tokens.txt`` directly into ``./zipformer/pretrained_models/<release_name>/``.

Usage
-----
    # Download both English (GigaSpeech) and Chinese (multi-zh-hans) models
    python utils/download_zipformer.py

    # Only one of them
    python utils/download_zipformer.py --only en
    python utils/download_zipformer.py --only zh

    # Inspect what would be downloaded without doing it
    python utils/download_zipformer.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRETRAINED_ROOT = _REPO_ROOT / "zipformer" / "pretrained_models"

_RELEASE_BASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
)


@dataclass
class Target:
    key: str
    release_name: str
    notes: str


TARGETS: Dict[str, Target] = {
    "en": Target(
        key="en",
        release_name="sherpa-onnx-zipformer-gigaspeech-2023-12-12",
        notes="Zipformer English (GigaSpeech 10000h, ~290 MB)",
    ),
    "zh": Target(
        key="zh",
        release_name="sherpa-onnx-zipformer-multi-zh-hans-2023-9-2",
        notes=(
            "Zipformer Mandarin (WenetSpeech + AISHELL-1/2/4 + AliMeeting + "
            "KeSpeech + MagicData-RAMC, ~290 MB)"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dir_populated(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def _human_size(p: Path) -> str:
    if not p.exists():
        return "missing"
    try:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while total >= 1024 and i < len(units) - 1:
        total /= 1024
        i += 1
    return f"{total:.1f} {units[i]}"


def _download_github_release(release_name: str, target_dir: Path) -> Path:
    """Download ``<release_name>.tar.bz2`` and extract it to ``target_dir``,
    stripping the leading ``<release_name>/`` so files land directly inside
    ``target_dir``.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_RELEASE_BASE_URL}/{release_name}.tar.bz2"

    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        print(f"  fetching {url}")
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)

        prefix = release_name.rstrip("/") + "/"
        with tarfile.open(tmp_path, "r:bz2") as tar:
            kept = 0
            for m in tar.getmembers():
                if m.name == release_name or m.name == prefix.rstrip("/"):
                    continue
                if m.name.startswith(prefix):
                    m.name = m.name[len(prefix):]
                if not m.name:
                    continue
                tar.extract(m, path=str(target_dir))
                kept += 1
        print(f"  extracted {kept} entries -> {target_dir}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return target_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(TARGETS.keys()),
        default=None,
        help="Restrict to these target keys (default: all of en, zh).",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(_PRETRAINED_ROOT),
        help=f"Destination root for the extracted models. "
             f"Default: {_PRETRAINED_ROOT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without doing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target directory is already populated.",
    )
    args = parser.parse_args()

    keys: List[str] = list(args.only) if args.only else list(TARGETS.keys())
    dest_root = Path(args.dest).expanduser().resolve()

    print(f"[zipformer] dest_root = {dest_root}")

    failures: List[str] = []
    for k in keys:
        t = TARGETS[k]
        local = dest_root / t.release_name
        print(f"=== {k} ({t.release_name}) ===")
        print(f"  {t.notes}")
        print(f"  target: {local}")

        if _is_dir_populated(local) and not args.force:
            print(f"  [skip] already populated ({_human_size(local)})")
            print()
            continue

        if args.dry_run:
            print("  [dry-run] would download")
            print()
            continue

        try:
            _download_github_release(t.release_name, local)
            print(f"  [done] ({_human_size(local)})")
        except Exception as exc:
            print(f"  [error] {exc!r}", file=sys.stderr)
            failures.append(k)
        print()

    if failures:
        print(f"[summary] {len(failures)} target(s) failed: {failures}",
              file=sys.stderr)
        sys.exit(1)
    print("[summary] all done.")


if __name__ == "__main__":
    main()
