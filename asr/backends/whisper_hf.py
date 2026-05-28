"""HuggingFace Whisper backend (whisper-large-v2 only).

Restored from the historical implementation so that the original Whisper
(English) path remains available alongside the default Zipformer.

Implementation notes
--------------------
* Audio is read in-memory via ``load_audio_16k`` (read-only on the source
  wav).
* ``return_timestamps="word"`` and ``return_segments=True`` are kept to
  mirror the historical decode path so existing Whisper TER numbers stay
  comparable.
* The processor / model are loaded with ``local_files_only=True`` whenever
  a local checkpoint directory is present under
  ``whisper/pretrained_models/whisper-large-v2``, so no implicit
  HuggingFace download is triggered at evaluation time.

Note: Whisper has well-known long-form hallucination behaviour
(repeated n-grams, looped phrases). The README documents that the final
TER metric in this repo uses Zipformer; this backend is kept available
behind the same registry interface so the historical numbers can still be
reproduced with ``--english-asr whisper-large-v2``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from ..base import ASRModel
from ..io_utils import load_audio_16k
from ..registry import register_asr


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_local_dir(model_id: str) -> Optional[str]:
    """Return ``whisper/pretrained_models/<short>`` if it exists locally."""
    short = model_id.split("/")[-1]
    candidate = PROJECT_ROOT / "whisper" / "pretrained_models" / short
    return str(candidate) if candidate.is_dir() else None


@register_asr("whisper-large-v2", aliases=["openai/whisper-large-v2"])
class WhisperLargeV2(ASRModel):
    """OpenAI Whisper large-v2 via HuggingFace ``transformers``."""

    hf_model_id: str = "openai/whisper-large-v2"

    def __init__(
        self,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(device=device, model_path=model_path, **kwargs)

        # Heavy imports kept inside __init__ so module import stays cheap.
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch_device = torch.device(
            self.device if torch.cuda.is_available() else "cpu"
        )

        resolved_path = self.model_path or _default_local_dir(self.hf_model_id)
        if resolved_path and os.path.isdir(resolved_path):
            source = resolved_path
            local_only = True
        else:
            source = self.hf_model_id
            local_only = False

        self.processor = WhisperProcessor.from_pretrained(
            source, task="transcribe", local_files_only=local_only
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            source, local_files_only=local_only
        ).to(self.torch_device)
        self.model.eval()

    def _forced_decoder_ids(self, language: str):
        return self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

    def transcribe(self, audio_path: str, language: str) -> str:
        audio_np, sr = load_audio_16k(audio_path)
        return self._transcribe_np(audio_np, sr, language)

    def transcribe_array(self, audio, sr: int, language: str) -> str:
        if sr != 16000:
            import torchaudio

            audio_t = torch.as_tensor(audio, dtype=torch.float32)
            if audio_t.ndim == 1:
                audio_t = audio_t.unsqueeze(0)
            audio_t = torchaudio.functional.resample(
                audio_t, orig_freq=sr, new_freq=16000
            )
            audio = audio_t.squeeze(0).cpu().numpy()
            sr = 16000
        return self._transcribe_np(audio, sr, language)

    def _transcribe_np(self, audio_np, sr: int, language: str) -> str:
        with torch.no_grad():
            inputs = self.processor(
                audio_np,
                sampling_rate=sr,
                return_tensors="pt",
                truncation=False,
                padding=True,
            )
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}
            attention_mask = (
                inputs["input_features"] != self.processor.tokenizer.pad_token_id
            ).long()
            inputs["attention_mask"] = attention_mask
            forced_decoder_ids = self._forced_decoder_ids(language)

            gen_out = self.model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps="word",
                return_segments=True,
            )
            transcripts = self.processor.batch_decode(
                gen_out["sequences"],
                output_offsets=True,
                skip_special_tokens=True,
            )
        return transcripts[0]["text"].strip()
