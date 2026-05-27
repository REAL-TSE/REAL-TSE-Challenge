"""Dataset -> language map shared by inference and aggregation scripts.

Kept in one place so backends, the inference driver and the multi-model
aggregator never disagree on which language a dataset belongs to.
"""

from __future__ import annotations

from typing import Dict


DATASET_LANGUAGE: Dict[str, str] = {
    "CHiME6": "en",
    "AISHELL-4": "zh",
    "AliMeeting": "zh",
    "AMI": "en",
    "DipCo": "en",
}


def get_language(dataset_name: str) -> str:
    """Return ``"zh"`` or ``"en"`` for a known dataset."""
    return DATASET_LANGUAGE[dataset_name]
