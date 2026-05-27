"""I/O helpers for the unified ASR pipeline.

Cache strategy
--------------
``ASR_CACHE_DIR`` selects where intermediate writes (when needed) land.
Default is ``<repo_root>/.asr_cache``. The current zipformer pipeline does
not need any model cache (sherpa-onnx loads ONNX files directly from
``./zipformer/pretrained_models/...``), so ``init_cache_env`` is mostly a
no-op kept for API compatibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio


# Repo root = parent of this file's package dir (asr/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = str(_REPO_ROOT / ".asr_cache")


def get_cache_dir(sub: Optional[str] = None) -> Path:
    """Return the writable ASR cache directory (creates it on demand)."""
    root = Path(os.environ.get("ASR_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser()
    target = root / sub if sub else root
    target.mkdir(parents=True, exist_ok=True)
    return target


def init_cache_env() -> Path:
    """Ensure the cache root exists. Idempotent and safe to call repeatedly."""
    return get_cache_dir()


def load_audio_16k(audio_path: str) -> Tuple[np.ndarray, int]:
    """Read an audio file and return a mono float32 array at 16 kHz.

    Returns
    -------
    tuple
        ``(audio_np, sr)``. ``audio_np`` is shape ``(samples,)``, float32,
        range roughly ``[-1, 1]``. ``sr`` is always ``16000``.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        sr = 16000
    return waveform.detach().cpu().to(torch.float32).numpy(), sr
