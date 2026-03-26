import argparse
import os
import pandas as pd
from calculate_TER import evaluate_TER
from print_stats import print_avg_wer_or_cer, compute_statistics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def construct_predicted_path_mapping(predicted_dir, chinese_asr_model, english_asr_model, include_fisher):
    datasets = ["AISHELL-4", "AliMeeting", "AMI", "CHiME6", "DipCo"]
    if include_fisher:
        datasets.append("Fisher")

    mapping = {}
    for dataset in datasets:
        if dataset in ["AISHELL-4", "AliMeeting"]:
            model_name = chinese_asr_model
        elif dataset in ["AMI", "CHiME6", "DipCo", "Fisher"]:
            model_name = english_asr_model
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        predicted_csv_path = os.path.join(predicted_dir, dataset, model_name, "predicted.csv")
        if not os.path.exists(predicted_csv_path):
            print(f"Warning: {predicted_csv_path} does not exist, skipping.")
            continue

        mapping[dataset] = predicted_csv_path
    return mapping

def main(args):
    predicted_path_mapping = construct_predicted_path_mapping(
        args.predicted_dir,
        args.chinese_asr_model,
        args.english_asr_model,
        args.include_fisher
    )

    if not predicted_path_mapping:
        print("No valid predicted.csv found. Exiting.")
        return

    utterance_to_predicted = {dataset: {} for dataset in predicted_path_mapping.keys()}

    print("Evaluating...\n")
    evaluation_df = evaluate_TER(args.ground_truth_dir, predicted_path_mapping, args.include_fisher)

    print_avg_wer_or_cer(evaluation_df)
    compute_statistics(evaluation_df)

    if args.save_path:
        evaluation_df.to_csv(args.save_path, index=False)
        print(f"\nResults saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR Predictions")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Directory of the ground truth datasets.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the evaluation results CSV.")
    parser.add_argument("--predicted_dir", type=str, required=True, help="Base directory containing predicted results.")
    parser.add_argument("--chinese_asr_model", type=str, default="FireRedASR-AED-L", help="ASR model name for Chinese datasets.")
    parser.add_argument("--english_asr_model", type=str, default="whisper-large-v2", help="ASR model name for English datasets.")
    parser.add_argument("--include_fisher", action='store_true', help="Whether to include Fisher dataset.")

    args = parser.parse_args()
    main(args)