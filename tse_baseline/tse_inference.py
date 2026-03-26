import argparse
import os
import pandas as pd
import soundfile as sf
import wesep
from tqdm import tqdm
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform Target Speaker Extraction (TSE)")
    parser.add_argument("--utterance_map_csv", type=str, required=True, help="Path to the CSV file mapping utterances to WAV paths")
    parser.add_argument("--meta_csv_path", type=str, required=True, help="Path to the CSV file with metadata")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the TSE result")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (e.g., 'cuda' or 'cpu')")
    return parser.parse_args()

def process_tse_row(mixture_utterance, enrolment_utterance, utterance_map, model, output_wav_dir):

    if mixture_utterance not in utterance_map or enrolment_utterance not in utterance_map:
        raise ValueError(f"Audio paths not found for {mixture_utterance} or {enrolment_utterance}")

    # Generate the new audio file name and path
    tse_audio_filename = f"{mixture_utterance}-{enrolment_utterance}.wav"
    tse_audio_path = os.path.join(output_wav_dir, tse_audio_filename)

    # Perform TSE
    try:
        with torch.no_grad():
            audio_mixture_path = utterance_map[mixture_utterance]
            enrolment_speaker_path = utterance_map[enrolment_utterance]
            speech = model.extract_speech(audio_mixture_path, enrolment_speaker_path)

        if speech is None:
            raise RuntimeError(f"Failed to extract speech for {tse_audio_filename}")
        
        # Save extracted audio
        sf.write(tse_audio_path, speech[0], 16000)

        return {"utterance": tse_audio_filename.split(".wav")[0], "path": tse_audio_path}
    except Exception as e:
        raise RuntimeError(f"Error processing {mixture_utterance}-{enrolment_utterance}: {e}")

def main():
    args = parse_arguments()
    
    model = wesep.load_model("english")
    model.set_device(args.device)


    utterance_map = pd.read_csv(args.utterance_map_csv).set_index("utterance")["path"].to_dict()

    meta_data = pd.read_csv(args.meta_csv_path)

    dataset_name = os.path.basename(args.meta_csv_path).split("_meta.csv")[0]

    output_wav_dir = os.path.join(args.output_dir, dataset_name, "wav")
    os.makedirs(output_wav_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, dataset_name, "tse_audio_mapping.csv")

    output_data = []

    for _, row in tqdm(meta_data.iterrows(), total=len(meta_data), desc="Processing TSE"):
        mixture_utterance = row["mixture_utterance"]
        enrolment_utterance = row["enrolment_speakers_utterance"]
        try:
            result = process_tse_row(mixture_utterance, enrolment_utterance, utterance_map, model, output_wav_dir)
            if result:
                output_data.append(result)
        except Exception as e:
            raise ValueError(f"Error processing {mixture_utterance}-{enrolment_utterance}: {e}")

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)

    print(f"Audio mapping saved to {output_csv_path}")

if __name__ == "__main__":
    main()