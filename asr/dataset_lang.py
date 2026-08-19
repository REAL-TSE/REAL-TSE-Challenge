"""Re-export of the canonical dataset -> language map.

The source of truth lives in ``utils/dataset_lang.py`` so eval scripts can
import it without pulling in the ASR package. This module keeps
``from asr.dataset_lang import DATASET_LANGUAGE, get_language`` working.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_UTILS_PATH = Path(__file__).resolve().parents[1] / "utils" / "dataset_lang.py"
_SPEC = importlib.util.spec_from_file_location("_real_t_dataset_lang", _UTILS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load dataset language map from {_UTILS_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

CANONICAL_LANGS = _MOD.CANONICAL_LANGS
DATASET_LANGUAGE = _MOD.DATASET_LANGUAGE
asr_model_for = _MOD.asr_model_for
chinese_datasets = _MOD.chinese_datasets
english_datasets = _MOD.english_datasets
get_language = _MOD.get_language
language_for_dataset = _MOD.language_for_dataset
normalize_language = _MOD.normalize_language
parse_dataset_lang_overrides = _MOD.parse_dataset_lang_overrides
to_wespeaker_lang = _MOD.to_wespeaker_lang

__all__ = [
    "CANONICAL_LANGS",
    "DATASET_LANGUAGE",
    "asr_model_for",
    "chinese_datasets",
    "english_datasets",
    "get_language",
    "language_for_dataset",
    "normalize_language",
    "parse_dataset_lang_overrides",
    "to_wespeaker_lang",
]
