import pandas as pd

def compute_statistics(df: pd.DataFrame) -> None:
    """
    Analyze and print statistics of the test data based on the given DataFrame.

    The DataFrame must contain at least the following columns:
      - mixture_duration
      - mixture_overlap_duration
      - mixture_utterance
      - mixture_ratio
      - source
      - language

    The statistics include:
      1. Total number of test samples (without deduplication).
      2. Overall statistics after deduplicating based on 'mixture_utterance':
         - Total duration (in minutes)
         - Total overlap duration (in minutes)
         - Number of unique 'mixture_utterance' entries
         - Average 'mixture_ratio'
      3. Statistics grouped by dataset (source), including the original number of test samples.
      4. Statistics grouped by language, including the original number of test samples.
    """

    total_test_count = df.shape[0]

    df_combined = df.drop_duplicates(subset=["mixture_utterance"])

    total_mixture_duration = df_combined["mixture_duration"].sum()
    total_mixture_overlap_duration = df_combined["mixture_overlap_duration"].sum()
    total_mixture_utterance_count = df_combined["mixture_utterance"].nunique()
    average_mixture_ratio = df_combined["mixture_ratio"].mean()

    total_mixture_duration_minutes = round(total_mixture_duration / 60, 2)
    total_mixture_overlap_duration_minutes = round(total_mixture_overlap_duration / 60, 2)
    average_mixture_ratio = round(average_mixture_ratio, 2)

    dataset_stats = df_combined.groupby("source").agg(
        total_mixture_duration=("mixture_duration", "sum"),
        total_mixture_overlap_duration=("mixture_overlap_duration", "sum"),
        total_mixture_utterance_count=("mixture_utterance", "nunique"),
        average_mixture_ratio=("mixture_ratio", "mean")
    ).reset_index()

    test_count_stats = df.groupby("source").size().reset_index(name="test_count")
    dataset_stats = dataset_stats.merge(test_count_stats, on="source", how="left")

    dataset_stats["total_mixture_duration"] = round(dataset_stats["total_mixture_duration"] / 60, 2)
    dataset_stats["total_mixture_overlap_duration"] = round(dataset_stats["total_mixture_overlap_duration"] / 60, 2)
    dataset_stats["average_mixture_ratio"] = round(dataset_stats["average_mixture_ratio"], 2)

    language_stats = df_combined.groupby("language").agg(
        total_mixture_duration=("mixture_duration", "sum"),
        total_mixture_overlap_duration=("mixture_overlap_duration", "sum"),
        total_mixture_utterance_count=("mixture_utterance", "nunique"),
        average_mixture_ratio=("mixture_ratio", "mean")
    ).reset_index()

    language_test_stats = df.groupby("language").size().reset_index(name="test_count")
    language_stats = language_stats.merge(language_test_stats, on="language", how="left")

    language_stats["total_mixture_duration"] = round(language_stats["total_mixture_duration"] / 60, 2)
    language_stats["total_mixture_overlap_duration"] = round(language_stats["total_mixture_overlap_duration"] / 60, 2)
    language_stats["average_mixture_ratio"] = round(language_stats["average_mixture_ratio"], 2)

    print("Total statistics:")
    print(f"Total mixture duration: {total_mixture_duration_minutes} minutes")
    print(f"Total mixture overlap duration: {total_mixture_overlap_duration_minutes} minutes")
    print(f"Total unique mixture utterance count: {total_mixture_utterance_count}")
    print(f"Average mixture ratio: {average_mixture_ratio}")
    print(f"Total test count: {total_test_count}")

    print("\nDataset statistics:")
    print(dataset_stats)

    print("\nLanguage statistics:")
    print(language_stats)

def print_avg_wer_or_cer(df: pd.DataFrame) -> None:
    """
    Compute and print statistics based on the 'source', 'language', and 'wer_or_cer' columns
    in the provided DataFrame.

    Specifically:
      - Calculate the average 'wer_or_cer' for each dataset (source).
      - Calculate the average 'wer_or_cer' grouped by language (typically 'zh' or 'en').

    The results are printed to the console.
    """
    required_cols = ['source', 'language', 'wer_or_cer']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in DataFrame: {col}")
    
    dataset_stats = df.groupby('source')['wer_or_cer'].mean().reset_index()
    print("Average WER/CER by dataset (source):")
    print(dataset_stats)
    
    language_stats = df.groupby('language')['wer_or_cer'].mean().reset_index()
    print("\nAverage WER/CER by language:")
    print(language_stats)

def print_SISDR(df: pd.DataFrame) -> None:
    """
    Print overall SI-SDR and SI-SDRi statistics for the 'Fisher' dataset.

    The provided DataFrame must contain the following columns:
      - 'source'
      - 'language'
      - 'SI-SDR'
      - 'SI-SDRi'

    Only the 'Fisher' dataset will be considered.
    """
    
    # Check if all required columns exist
    required_cols = ['source', 'language', 'SI-SDR', 'SI-SDRi']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in DataFrame: {col}")

    # Only process if 'Fisher' exists
    if (df['source'] == 'Fisher').any():
        print("=== SI-SDR and SI-SDRi Analysis for Fisher ===")
        fisher_df = df[df['source'] == 'Fisher']

        fisher_sdr = fisher_df['SI-SDR'].mean()
        fisher_sdr_i = fisher_df['SI-SDRi'].mean()

        print(f"\nFisher - Average SI-SDRi: {fisher_sdr_i:.2f} dB")
        print(f"Fisher - Average SI-SDR : {fisher_sdr:.2f} dB")
    else:
        print("No 'Fisher' dataset found. Skipping SI-SDR/SI-SDRi analysis.")

