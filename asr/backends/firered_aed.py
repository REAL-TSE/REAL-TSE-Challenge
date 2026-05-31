"""FireRedASR-AED-L backend (Mandarin).

Restored alongside the Whisper backend so the historical Chinese ASR
path (``FireRedASR-AED-L``) is still available behind the unified
registry. The decode hyper-parameters mirror the original
``asr/asr_models.py`` values so existing TER numbers stay comparable.

The FireRedASR Python package is shipped under ``./FireRedASR/`` at the
repo root (NOT a PyPI dependency); we add it to ``sys.path`` lazily.

Note: FireRedASR-AED-L can also produce hallucinated repetitions on
challenging real-world conversation audio. The README documents that the
final TER metric in this repo uses Zipformer; this backend is kept
available so historical numbers remain reproducible via
``--chinese-asr FireRedASR-AED-L``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

from ..base import ASRModel
from ..registry import register_asr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIREREDASR_ROOT = PROJECT_ROOT / "FireRedASR"


def _project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _require_model_files(model_dir: str, required_files):
    model_path = Path(model_dir)
    missing = [name for name in required_files if not (model_path / name).is_file()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required model files in {model_path}: {missing_str}"
        )
    return str(model_path)


def _import_fireredasr():
    if str(FIREREDASR_ROOT) not in sys.path:
        sys.path.insert(0, str(FIREREDASR_ROOT))
    from fireredasr.models.fireredasr import FireRedAsr  # type: ignore

    return FireRedAsr


@register_asr("FireRedASR-AED-L", aliases=["firered-aed", "fireredasr-aed-l"])
class FireRedASR_AED_L(ASRModel):
    """Wrap FireRedASR-AED-L behind the unified ASRModel interface."""

    def __init__(
        self,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(device=device, model_path=model_path, **kwargs)

        resolved = self.model_path or _project_path(
            "FireRedASR", "pretrained_models", "FireRedASR-AED-L"
        )
        _require_model_files(
            resolved,
            ["model.pth.tar", "cmvn.ark", "dict.txt", "train_bpe1000.model"],
        )

        FireRedAsr = _import_fireredasr()
        # FireRedAsr.from_pretrained expects ``asr_type`` in {"aed", "llm"}.
        self.model = FireRedAsr.from_pretrained("aed", resolved)
        self.torch_device = torch.device(
            self.device if torch.cuda.is_available() else "cpu"
        )
        self.use_gpu = self.torch_device.type == "cuda"

    def transcribe(self, audio_path: str, language: str) -> str:
        del language  # FireRedASR-AED-L does not take a language hint.
        if self.use_gpu:
            torch.cuda.set_device(self.torch_device)
        with torch.no_grad():
            results = self.model.transcribe(
                ["dummy_id"],
                [audio_path],
                {
                    "use_gpu": int(self.use_gpu),
                    "beam_size": 3,
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.0,
                    "aed_length_penalty": 0.0,
                    "eos_penalty": 1.0,
                },
            )
        text = results[0]["text"]
        return text.strip()
