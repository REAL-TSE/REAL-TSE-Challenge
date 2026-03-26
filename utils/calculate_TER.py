import os
import pandas as pd
import numpy as np
from typing import Dict
from meeteval.wer.wer.siso import siso_word_error_rate
from asr_metrics import normalizer_for_transcript

def evaluate_TER(
    full_dir: str,
    predicted_path_mapping: Dict[str, str],
    include_fisher: bool = False
) -> pd.DataFrame:

    utterance_to_predicted = {
        dataset: dict(zip(pd.read_csv(path)["utterance"], pd.read_csv(path)["transcript"]))
        for dataset, path in predicted_path_mapping.items()
    }

    # Load all CSV files, filter based on include_fisher
    merged_df = pd.concat(
        [
            pd.read_csv(os.path.join(full_dir, f))
            for f in os.listdir(full_dir)
            if f.endswith(".csv") and (include_fisher or not f.startswith("Fisher"))
        ],
        ignore_index=True
    )


    if "wer_or_cer" not in merged_df.columns:
        merged_df.insert(merged_df.columns.get_loc("mixture_ratio") + 1, "wer_or_cer", np.nan)
    if "transcript_length" not in merged_df.columns:
        merged_df.insert(merged_df.columns.get_loc("wer_or_cer") + 1, "transcript_length", np.nan)
    if "predicted" not in merged_df.columns:
        merged_df["predicted"] = np.nan

    # Only evaluate rows for which we have predictions (e.g. when only Chinese ASR2 is run)
    datasets_with_pred = set(utterance_to_predicted.keys())
    merged_df = merged_df[merged_df["source"].isin(datasets_with_pred)].copy()

    def fill_result(row):
        dataset_name = row["source"]
        language = row["language"]
        utt_key = f"{row['mixture_utterance']}-{row['enrolment_speakers_utterance']}"
        gt = row["ground_truth_transcript"]
        gt = normalizer_for_transcript(gt, "Ground Truth", language)

        pred = utterance_to_predicted[dataset_name].get(utt_key, "")
        if pd.isna(pred):
            pred = ""
        pred = normalizer_for_transcript(pred, "Predicted", language)

        row["ground_truth_transcript"] = gt
        row["predicted"] = utterance_to_predicted[dataset_name].get(utt_key, "")
        row["transcript_length"] = len(gt.strip().split())
        row["wer_or_cer"] = siso_word_error_rate(gt, pred).error_rate

        return row

    merged_df = merged_df.apply(fill_result, axis=1)

    return merged_df
