"""sherpa-onnx Zipformer (icefall offline transducer) backend.

We expose two registered names:

* ``zipformer-en``       -> sherpa-onnx-zipformer-gigaspeech-2023-12-12
                            (GigaSpeech 10000h, English; strong on
                            conversation/meeting)
* ``zipformer-zh``       -> sherpa-onnx-zipformer-multi-zh-hans-2023-9-2
                            (WenetSpeech + AISHELL-1/2/4 + AliMeeting +
                            KeSpeech + MagicData-RAMC, Mandarin)

Why sherpa-onnx, not k2/icefall directly?
-----------------------------------------
icefall + k2 is the original training/decoding stack, but its inference path
requires building k2 against a specific torch+CUDA combo (slow and brittle in
practice). ``sherpa-onnx`` ships pure ONNX + onnxruntime artifacts of the same
zipformer transducers via :pypi:`sherpa-onnx` - no compiler, no CUDA. CPU
works out of the box; GPU just needs ``onnxruntime-gpu`` instead.

Model packaging
---------------
Each model is a ``.tar.bz2`` from sherpa-onnx GitHub Releases that extracts
to:

    <pretrained_dir>/<model_name>/
        encoder-epoch-XX-avg-Y.onnx
        decoder-epoch-XX-avg-Y.onnx
        joiner-epoch-XX-avg-Y.onnx
        encoder-epoch-XX-avg-Y.int8.onnx        (we prefer fp32 by default)
        decoder-epoch-XX-avg-Y.int8.onnx
        joiner-epoch-XX-avg-Y.int8.onnx
        tokens.txt
        bpe.model                                (optional, English BPE only)
        test_wavs/...

We auto-discover the encoder/decoder/joiner inside the directory so we do NOT
have to hard-code the epoch number.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import ASRModel
from ..io_utils import load_audio_16k
from ..registry import register_asr


# ---------------------------------------------------------------------------
# Default model directories
# ---------------------------------------------------------------------------
# REAL-TSE-Challenge convention: top-level ``zipformer/pretrained_models/``
# directory, mirroring the historical ``whisper/`` and ``FireRedASR/`` layout.

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PRETRAINED_ROOT = _REPO_ROOT / "zipformer" / "pretrained_models"

_ZIPFORMER_EN_NAME = "sherpa-onnx-zipformer-gigaspeech-2023-12-12"
_ZIPFORMER_ZH_NAME = "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_dir(model_path: Optional[str], default_name: str) -> Path:
    """Resolve a model directory either from an explicit path or the default."""
    if model_path:
        p = Path(model_path).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(
                f"zipformer model_path is not a directory: {p}"
            )
        return p
    p = _PRETRAINED_ROOT / default_name
    if not p.is_dir():
        raise FileNotFoundError(
            f"Default zipformer dir missing: {p}\n"
            f"  Did you run `python utils/download_zipformer.py` "
            f"or `bash pre.sh`?"
        )
    return p


def _pick_first(paths_glob: list, model_dir: Path, label: str) -> Path:
    """Return the first matching file under ``model_dir`` for one of the glob patterns.

    Files that also match a later (lower-priority) pattern are excluded from
    the current pattern so they get deferred to their most-specific match.
    This prevents ``encoder*.onnx`` from swallowing ``encoder*.int8.onnx``
    hits when fp32 is preferred.
    """
    for i, pat in enumerate(paths_glob):
        hits = sorted(model_dir.glob(pat))
        if hits and i + 1 < len(paths_glob):
            later_hits: set = set()
            for later_pat in paths_glob[i + 1:]:
                later_hits.update(model_dir.glob(later_pat))
            hits = [h for h in hits if h not in later_hits]
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find {label} under {model_dir} (tried {paths_glob})"
    )


def _discover_files(model_dir: Path, prefer_int8: bool = False) -> dict:
    """Auto-discover encoder/decoder/joiner/tokens within an extracted
    sherpa-onnx zipformer directory."""
    suffix_main = ".int8.onnx" if prefer_int8 else ".onnx"
    suffix_alt = ".onnx" if prefer_int8 else ".int8.onnx"
    return {
        "encoder": str(_pick_first(
            [f"encoder*{suffix_main}", f"encoder*{suffix_alt}"],
            model_dir, "encoder onnx",
        )),
        "decoder": str(_pick_first(
            [f"decoder*{suffix_main}", f"decoder*{suffix_alt}"],
            model_dir, "decoder onnx",
        )),
        "joiner": str(_pick_first(
            [f"joiner*{suffix_main}", f"joiner*{suffix_alt}"],
            model_dir, "joiner onnx",
        )),
        "tokens": str(_pick_first(["tokens.txt"], model_dir, "tokens.txt")),
    }


def _build_recognizer(model_dir: Path, num_threads: int, prefer_int8: bool):
    """Lazy import of sherpa_onnx so the dependency is only required when used."""
    import sherpa_onnx  # type: ignore

    files = _discover_files(model_dir, prefer_int8=prefer_int8)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=files["encoder"],
        decoder=files["decoder"],
        joiner=files["joiner"],
        tokens=files["tokens"],
        num_threads=num_threads,
        # 16 kHz is what we feed via load_audio_16k()
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class _ZipformerBase(ASRModel):
    """Shared inference path for both Zipformer variants."""

    default_name: str = ""

    def __init__(
        self,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        num_threads: Optional[int] = None,
        int8: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(device=device, model_path=model_path, **kwargs)

        # sherpa-onnx CPU build is enough for our use case. ``device`` is
        # kept in the signature for ASRModel API parity but does not select
        # a CUDA provider here (that would need onnxruntime-gpu plus extra
        # wiring).
        if num_threads is None:
            num_threads = max(1, min(8, (os.cpu_count() or 4) // 2))

        self.model_dir = _resolve_model_dir(model_path, self.default_name)
        self.recognizer = _build_recognizer(
            self.model_dir, num_threads, prefer_int8=int8
        )

    def transcribe(self, audio_path: str, language: str = "") -> str:
        del language  # checkpoint is already language-specific
        audio_np, _sr = load_audio_16k(audio_path)
        # sherpa-onnx wants float32 in [-1, 1]; load_audio_16k already
        # returns that.
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio_np.astype(np.float32))
        self.recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()


@register_asr(
    "zipformer-en",
    aliases=[
        "zipformer-gigaspeech",
        "sherpa-onnx-zipformer-gigaspeech-2023-12-12",
    ],
)
class ZipformerEn(_ZipformerBase):
    default_name = _ZIPFORMER_EN_NAME


@register_asr(
    "zipformer-zh",
    aliases=[
        "zipformer-multi-zh-hans",
        "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2",
    ],
)
class ZipformerZh(_ZipformerBase):
    default_name = _ZIPFORMER_ZH_NAME
