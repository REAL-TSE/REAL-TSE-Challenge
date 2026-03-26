from pathlib import Path

from transformers import WhisperTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_WHISPER_DIR = PROJECT_ROOT / "whisper" / "pretrained_models" / "whisper-large-v2"
MODEL_NAME = "openai/whisper-large-v2"


def _load_whisper_tokenizer() -> WhisperTokenizer:
    if LOCAL_WHISPER_DIR.is_dir():
        return WhisperTokenizer.from_pretrained(
            str(LOCAL_WHISPER_DIR),
            task="transcribe",
            local_files_only=True,
        )
    return WhisperTokenizer.from_pretrained(MODEL_NAME, task="transcribe")


normalizer = _load_whisper_tokenizer()

def normalize_text(transcript: str, normalizer = normalizer) -> str:
    transcript = transcript.replace("(", "").replace(")", "")
    normalized_text = normalizer.normalize(transcript)
    cleaned_text = " ".join(normalized_text.split())
    return cleaned_text
