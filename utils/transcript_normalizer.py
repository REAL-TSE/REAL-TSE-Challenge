"""Light-weight transcript text normalizer.

Originally backed by ``transformers.WhisperTokenizer`` (which required the
Whisper-large-v2 checkpoint to be present). It now uses the
``whisper-normalizer`` PyPI package, which ships only the textual
normalization logic from OpenAI Whisper without any model weights.

Kept separately from :mod:`asr_metrics` so that callers who only want a
stateless text normalizer do not pull in opencc and the language-specific
post-processing.
"""

from __future__ import annotations


def _load_normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except Exception as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "whisper-normalizer is required. "
            "Install it via `pip install whisper-normalizer`."
        ) from exc
    return EnglishTextNormalizer()


normalizer = _load_normalizer()


def normalize_text(transcript: str, normalizer=normalizer) -> str:
    transcript = transcript.replace("(", "").replace(")", "")
    normalized_text = normalizer(transcript)
    cleaned_text = " ".join(normalized_text.split())
    return cleaned_text
