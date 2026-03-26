from pathlib import Path
import re

import opencc
from transformers import WhisperTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_WHISPER_DIR = PROJECT_ROOT / "whisper" / "pretrained_models" / "whisper-large-v2"
MODEL_NAME = "openai/whisper-large-v2"


def _load_whisper_tokenizer() -> WhisperTokenizer:
    if LOCAL_WHISPER_DIR.is_dir():
        return WhisperTokenizer.from_pretrained(
            str(LOCAL_WHISPER_DIR),
            local_files_only=True,
        )
    return WhisperTokenizer.from_pretrained(MODEL_NAME)


normalizer = _load_whisper_tokenizer()

def whisper_normalize(transcript: str) -> str:
    transcript = transcript.replace("(", "").replace(")", "")
    normalized_text = normalizer.normalize(transcript)
    cleaned_text = " ".join(normalized_text.split())
    return cleaned_text.strip()


def normalizer_for_zh(transcript: str, option: str = None) -> str:

    assert option in ['Predicted', 'Ground Truth'], f"Invalid option: {option}"
    if option == 'Predicted':
        transcript = whisper_normalize(transcript)
    elif option == 'Ground Truth':
        pass

    # transcript = whisper_normalize(transcript.strip()).strip()
    converter = opencc.OpenCC('t2s')
    transcript = converter.convert(transcript)
    transcript = re.sub(r'\s+', '', transcript)
    transcript = " ".join(transcript.strip())
    return transcript

def normalizer_for_en(transcript: str, option: str = None) -> str:

    transcript = transcript.strip()
    transcript = transcript.replace("…", "")
    assert option in ['Predicted', 'Ground Truth'], f"Invalid option: {option}"
    if option == 'Predicted':
        transcript = whisper_normalize(transcript)
    elif option == 'Ground Truth':
        pass
    
    transcript = transcript.replace(".", "")
    re.sub(r'\s+', ' ', transcript.strip())
    return transcript.strip()

def normalizer_for_transcript(transcript: str, option: str = None, language: str = None) -> str:

    assert language in ['zh', 'en']

    if language == 'en':
        transcript = normalizer_for_en(transcript, option)
    elif language == 'zh':
        transcript = normalizer_for_zh(transcript, option)
    
    return transcript.strip()
