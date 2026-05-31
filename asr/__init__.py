"""Public asr package entry point.

Importing this package triggers registration of all built-in ASR backends,
so callers only need:

    from asr import get_asr_model, list_models
    model = get_asr_model("zipformer-en", device="cpu")
    text = model.transcribe(wav_path, language="en")
"""

from __future__ import annotations

from .base import ASRModel
from .dataset_lang import DATASET_LANGUAGE, get_language
from .io_utils import get_cache_dir, init_cache_env, load_audio_16k
from .postprocess import detect_repetition, truncate_repetition
from .registry import get_asr_model, list_models, register_asr, resolve_name

# Trigger backend registration. Each backend module performs its own lazy
# heavy-dependency import inside the class __init__, so importing the package
# stays cheap.
from . import backends  # noqa: F401  (side-effect import)


__all__ = [
    "ASRModel",
    "DATASET_LANGUAGE",
    "detect_repetition",
    "get_asr_model",
    "get_cache_dir",
    "get_language",
    "init_cache_env",
    "list_models",
    "load_audio_16k",
    "register_asr",
    "resolve_name",
    "truncate_repetition",
]
