"""ASR transcript normalization for TER evaluation.

Historically used ``transformers.WhisperTokenizer.normalize`` (which depends
on the Whisper-large-v2 checkpoint being present locally). We now use
``whisper-normalizer`` (a small pip package that ships the same English /
basic normalizer logic from OpenAI Whisper without the model weights), so
we no longer need any Whisper checkpoint just to run TER.

Public API (unchanged signatures, used by ``utils/calculate_TER.py``):

* ``normalizer_for_transcript(transcript, option, language)``
* ``normalizer_for_zh(transcript, option=None)``
* ``normalizer_for_en(transcript, option=None)``
"""

from __future__ import annotations

import re
from typing import Optional

import opencc


# ---------------------------------------------------------------------------
# Whisper-style normalizer (model-free)
# ---------------------------------------------------------------------------

def _build_normalizers():
    """Lazy-build the two normalizers from ``whisper-normalizer``.

    Returns
    -------
    (en_normalize, basic_normalize) : tuple of callables
        Each callable maps ``str -> str``.
    """
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        from whisper_normalizer.basic import BasicTextNormalizer
    except Exception as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "whisper-normalizer is required for TER evaluation. "
            "Install it via `pip install whisper-normalizer`."
        ) from exc

    en = EnglishTextNormalizer()
    basic = BasicTextNormalizer()
    return en, basic


_EN_NORMALIZER = None
_BASIC_NORMALIZER = None


def _get_normalizers():
    global _EN_NORMALIZER, _BASIC_NORMALIZER
    if _EN_NORMALIZER is None or _BASIC_NORMALIZER is None:
        _EN_NORMALIZER, _BASIC_NORMALIZER = _build_normalizers()
    return _EN_NORMALIZER, _BASIC_NORMALIZER


def whisper_normalize(transcript: str, language: str = "en") -> str:
    """Apply the Whisper-style text normalizer (no model weights needed).

    Parameters
    ----------
    transcript : str
        Raw transcript.
    language : str
        ``"en"`` -> use the English normalizer; otherwise -> basic
        normalizer (works for Mandarin, since we tokenize per-character
        afterwards via the existing space-padding step).
    """
    transcript = transcript.replace("(", "").replace(")", "")
    en, basic = _get_normalizers()
    if language == "en":
        normalized_text = en(transcript)
    else:
        normalized_text = basic(transcript)
    cleaned_text = " ".join(normalized_text.split())
    return cleaned_text.strip()


# ---------------------------------------------------------------------------
# Per-language normalizers (signatures unchanged for calculate_TER.py)
# ---------------------------------------------------------------------------

_ZH_CONVERTER = opencc.OpenCC("t2s")


def normalizer_for_zh(transcript: str, option: Optional[str] = None) -> str:
    assert option in ("Predicted", "Ground Truth"), f"Invalid option: {option}"
    
    transcript = whisper_normalize(transcript, language="zh")
    transcript = _ZH_CONVERTER.convert(transcript)
    transcript = re.sub(r"\s+", "", transcript)
    return transcript


def normalizer_for_en(transcript: str, option: Optional[str] = None) -> str:
    transcript = transcript.strip().replace("…", "")
    assert option in ("Predicted", "Ground Truth"), f"Invalid option: {option}"

    transcript = whisper_normalize(transcript, language="en")
    transcript = transcript.replace(".", "")
    transcript = re.sub(r"\s+", " ", transcript.strip())
    return transcript.strip()


def normalizer_for_transcript(
    transcript: str,
    option: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    assert language in ("zh", "en"), f"Invalid language: {language!r}"
    if language == "en":
        transcript = normalizer_for_en(transcript, option)
    else:
        transcript = normalizer_for_zh(transcript, option)
    return transcript.strip()
