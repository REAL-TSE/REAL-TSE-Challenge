"""Hallucination-style repetition detection / truncation.

Originally added for Whisper's looping output (``"the the the ..."`` /
``"啊啊啊啊"``), but kept here as a general defensive layer that any ASR
backend can opt into.

Conventions
-----------
* Chinese ("zh"): tokenization is character-level (each non-whitespace
  character is one token).
* English ("en"): tokenization is whitespace-split, lower-cased.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def _tokenize(text: str, language: str) -> List[str]:
    if not text:
        return []
    if language == "zh":
        compact = re.sub(r"\s+", "", text)
        return list(compact)
    return [tok for tok in re.split(r"\s+", text.strip().lower()) if tok]


def _max_consecutive_ngram_runs(
    tokens: List[str], ngram: int
) -> Tuple[int, Optional[Tuple[str, ...]], int]:
    if ngram <= 0 or len(tokens) < ngram * 2:
        return 1, None, 0

    best_run = 1
    best_ngram: Optional[Tuple[str, ...]] = None
    best_start = 0

    i = 0
    n = len(tokens)
    while i + ngram <= n:
        current = tuple(tokens[i : i + ngram])
        run = 1
        j = i + ngram
        while j + ngram <= n and tuple(tokens[j : j + ngram]) == current:
            run += 1
            j += ngram
        if run > best_run:
            best_run = run
            best_ngram = current
            best_start = i
        i = j if run > 1 else i + 1

    return best_run, best_ngram, best_start


def detect_repetition(
    text: str,
    language: str,
    ngrams: Tuple[int, ...] = (1, 2, 3, 4, 5),
    max_repeat: int = 4,
) -> Tuple[bool, Optional[str], int]:
    """Detect hallucination-style repetition."""
    tokens = _tokenize(text, language)
    if len(tokens) < max_repeat:
        return False, None, 0

    for ngram in ngrams:
        run, ng_tuple, start = _max_consecutive_ngram_runs(tokens, ngram)
        if run >= max_repeat and ng_tuple is not None:
            joiner = "" if language == "zh" else " "
            return True, joiner.join(ng_tuple), start
    return False, None, 0


def truncate_repetition(
    text: str,
    language: str,
    ngrams: Tuple[int, ...] = (1, 2, 3, 4, 5),
    max_repeat: int = 4,
    keep_repeats: int = 1,
) -> Tuple[str, bool, Optional[str]]:
    """Truncate hallucinated repetition tails from a transcript."""
    tokens = _tokenize(text, language)
    if len(tokens) < max_repeat:
        return text, False, None

    triggered = False
    repeated_str: Optional[str] = None
    cut_token_idx: Optional[int] = None

    for ngram in ngrams:
        run, ng_tuple, start = _max_consecutive_ngram_runs(tokens, ngram)
        if run >= max_repeat and ng_tuple is not None:
            triggered = True
            joiner = "" if language == "zh" else " "
            repeated_str = joiner.join(ng_tuple)
            cut_token_idx = start + ngram * max(1, keep_repeats)
            break

    if not triggered or cut_token_idx is None:
        return text, False, None

    kept = tokens[:cut_token_idx]
    if language == "zh":
        cleaned = "".join(kept)
    else:
        cleaned = " ".join(kept)
    return cleaned, True, repeated_str
