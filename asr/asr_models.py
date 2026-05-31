"""Backwards-compatible facade for the historical ``asr_models`` import path.

Older callers used to do::

    from asr_models import WhisperASR, FireRedASR_AED_L_ASRModel

The pipeline has since been refactored around an ABC + registry, but the
two original backends are still registered (alongside the default
sherpa-onnx Zipformer). We keep aliases on the historical class names so
external imports keep working.

New code should prefer::

    from asr import get_asr_model
    model = get_asr_model("zipformer-en", device="cpu")
"""

from __future__ import annotations

# Trigger backend registration via the asr package import.
from asr.backends.firered_aed import FireRedASR_AED_L
from asr.backends.whisper_hf import WhisperLargeV2
from asr.backends.zipformer_sherpa import ZipformerEn, ZipformerZh
from asr.registry import get_asr_model, list_models


# ---------------------------------------------------------------------------
# Legacy class-name aliases. The original ``asr/asr_models.py`` exposed
# ``WhisperASR`` and ``FireRedASR_AED_L_ASRModel``; we keep the names so
# any external script still doing ``from asr_models import WhisperASR``
# continues to work.
# ---------------------------------------------------------------------------
WhisperASR = WhisperLargeV2
FireRedASR_AED_L_ASRModel = FireRedASR_AED_L


__all__ = [
    "FireRedASR_AED_L",
    "FireRedASR_AED_L_ASRModel",
    "WhisperASR",
    "WhisperLargeV2",
    "ZipformerEn",
    "ZipformerZh",
    "get_asr_model",
    "list_models",
]
