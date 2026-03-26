import os
import csv
from datasets import load_dataset
import torch
import torchaudio
import argparse
import pandas as pd


def export_subset(repo_id, subset, out_root):
    print(f"Loading subset {subset}...")
    ds = load_dataset(repo_id, name=subset, split="test")

    subset_dir = os.path.join(out_root, subset)
    mixtures_dir = os.path.join(out_root, "mixtures")
    enrol_dir = os.path.join(out_root, "enrolment_speakers")
    os.makedirs(subset_dir, exist_ok=True)
    os.makedirs(mixtures_dir, exist_ok=True)
    os.makedirs(enrol_dir, exist_ok=True)

    csv_fields = [
        "mixture_utterance", "enrolment_speakers_utterance",
        "source", "language", "total_number_of_speaker", "speaker", "gender",
        "speaker_ratio", "mixture_ratio", "enrolment_speakers_duration",
        "mixture_overlap_duration", "mixture_duration", "ground_truth_transcript"
    ]

    rows = []

    for item in ds:
        mix_name = f"{item['mixture_utterance']}.wav"
        mix_path = os.path.join(mixtures_dir, mix_name)
        torchaudio.save(
            mix_path,
            torch.tensor(item["mixture_audio"]["array"]).unsqueeze(0),
            item["mixture_audio"]["sampling_rate"]
        )

        enrol_name = f"{item['enrolment_speakers_utterance']}.wav"
        enrol_path = os.path.join(enrol_dir, enrol_name)
        torchaudio.save(
            enrol_path,
            torch.tensor(item["enrolment_audio"]["array"]).unsqueeze(0),
            item["enrolment_audio"]["sampling_rate"]
        )

        row = {}
        for k in csv_fields:
            v = item[k]
            if isinstance(v, float):
                row[k] = round(v, 2)
            else:
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    for source, df_group in df.groupby("source"):
        save_path = os.path.join(subset_dir, f"{source}_meta.csv")
        df_group.to_csv(save_path, index=False, encoding="utf-8")

    print(f"{subset} successfully export -> {subset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub_repo", type=str, required=True,
                        help="Hugging Face repo_id, e.g., SLbaba/REAL-T",default="SLbaba/REAL-T")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Path to save REAL-T datasets")
    args = parser.parse_args()
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    for subset in ["PRIMARY", "BASE"]:
        export_subset(args.hub_repo, subset, args.save_dir)
