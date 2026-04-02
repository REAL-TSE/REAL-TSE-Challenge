import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
from tqdm import tqdm
from asr_models import WhisperASR, FireRedASR_AED_L_ASRModel

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

language = {
    'CHiME6': 'en',
    'AISHELL-4': 'zh',
    'AliMeeting': 'zh',
    'AMI': 'en',
    'DipCo': 'en'
}

def get_asr_model(model_name, device):
    if model_name == "whisper-large-v2":
        return WhisperASR(model_name="openai/whisper-large-v2", device=device)
    elif model_name == "FireRedASR-AED-L":
        return FireRedASR_AED_L_ASRModel(model_name="aed", device=device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="ASR Inference Script")
    parser.add_argument("--audio_mapping", "--audio_mapping_csv", dest="audio_mapping_csv", type=str, required=True, help="Path to the audio mapping CSV file")
    parser.add_argument("--model_name", type=str, required=True, help="ASR Model to use")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the transcription result")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run ASR on, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of rows to transcribe for smoke tests.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, "predicted.csv")

    model = get_asr_model(args.model_name, args.device)

    audio_mapping = pd.read_csv(args.audio_mapping_csv)
    if args.max_samples is not None:
        audio_mapping = audio_mapping.head(args.max_samples)

    results = []

    print(f"Processing {len(audio_mapping)} audio files...")

    for idx, row in tqdm(audio_mapping.iterrows(), total=len(audio_mapping), desc="Processing Audio Files"):
        audio_path = row['path']

        transcription = model.transcribe_audio(audio_path, language=language[args.dataset_name])

        print(f"Transcription for {audio_path}: {transcription}")
        results.append({"utterance": row['utterance'], "transcript": transcription})

    
    if len(results) == len(audio_mapping):
        print("All transcriptions successfully saved to CSV.")
    else:
        failed_count = len(audio_mapping) - len(results)
        raise RuntimeError(f"Some files failed to transcribe. {failed_count} audio files were not processed successfully.")

    transcription_df = pd.DataFrame(results)
    transcription_df.to_csv(output_file, index=False)
    print(f"Transcriptions saved to {output_file}")
    

if __name__ == "__main__":
    main()
