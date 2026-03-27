import os
import sys
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIREREDASR_ROOT = PROJECT_ROOT / "FireRedASR"
if str(FIREREDASR_ROOT) not in sys.path:
    sys.path.insert(0, str(FIREREDASR_ROOT))



def _project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def _require_model_files(model_dir, required_files):
    model_path = Path(model_dir)
    missing = [name for name in required_files if not (model_path / name).is_file()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required model files in {model_path}: {missing_str}"
        )
    return str(model_path)


def _import_fireredasr():
    from fireredasr.models.fireredasr import FireRedAsr

    return FireRedAsr


class WhisperASR:
    def __init__(self, model_name="openai/whisper-large-v2", model_path=None, device="cuda"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_path = model_path or _project_path("whisper", "pretrained_models", "whisper-large-v2")
        model_source = model_path if os.path.isdir(model_path) else model_name
        self.processor = WhisperProcessor.from_pretrained(model_source, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_source).to(self.device)

    def transcribe_audio(self, audio_path, language="en"):

        audio, sr = torchaudio.load(audio_path)
        audio = audio.squeeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)

        with torch.no_grad():
            whisper_inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", truncation=False, padding=True)
            whisper_inputs = {k: v.to(self.device) for k, v in whisper_inputs.items()}
            # attention_mask is add after see the warning
            attention_mask = (whisper_inputs['input_features'] != self.processor.tokenizer.pad_token_id).long()
            whisper_inputs['attention_mask'] = attention_mask
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")

            asr_model_out = self.model.generate(
                **whisper_inputs, 
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps="word",
                return_segments=True
            )

            transcripts = self.processor.batch_decode(asr_model_out['sequences'], output_offsets=True, skip_special_tokens=True)

        return transcripts[0]['text'].strip()

class FireRedASR_AED_L_ASRModel:
    def __init__(self, model_name="aed", model_path=None, device="cuda:0"):
        model_path = model_path or _project_path(
            "FireRedASR", "pretrained_models", "FireRedASR-AED-L"
        )
        _require_model_files(
            model_path,
            ["model.pth.tar", "cmvn.ark", "dict.txt", "train_bpe1000.model"],
        )
        FireRedAsr = _import_fireredasr()
        self.model = FireRedAsr.from_pretrained(model_name, model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_gpu = self.device.type == "cuda"

    def transcribe_audio(self, audio_path, language="zh"):
        del language
        if self.use_gpu:
            torch.cuda.set_device(self.device)
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
                    "eos_penalty": 1.0
                }
            )
            transcription = results[0]['text']
            return transcription.strip()
