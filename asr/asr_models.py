"""Backwards-compatible facade for the historical ``asr_models`` import path.

Older callers used to do ``from asr_models import WhisperASR,
FireRedASR_AED_L_ASRModel``. The Whisper / FireRedASR backends have been
removed (replaced by sherpa-onnx Zipformer); we keep a thin re-export of
the new registry-backed API so any remaining ``from asr_models import
get_asr_model`` paths still work.

New code should prefer:

    from asr import get_asr_model
    model = get_asr_model("zipformer-en", device="cpu")
"""

from __future__ import annotations

# Trigger backend registration via the asr package import.
from asr.backends.zipformer_sherpa import ZipformerEn, ZipformerZh
from asr.registry import get_asr_model, list_models


__all__ = [
    "ZipformerEn",
    "ZipformerZh",
    "get_asr_model",
    "list_models",
]
