#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_DATASET_ORDER = ["AISHELL-4", "AMI", "AliMeeting", "CHiME6", "DipCo", "Fisher"]
DEFAULT_LANG_ORDER = ["en", "chs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate REAL-T evaluation CSVs into one summary report."
    )
    parser.add_argument(
        "--base_dir",
        action="append",
        required=True,
        help="Base result directory. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output_txt_name",
        default=None,
        help="Output TXT filename under each base_dir. Default: <base_name>_summary.txt.",
    )
    parser.add_argument(
        "--metrics_subdir",
        default="eval_metrics",
        help=(
            "Subdirectory under each base_dir that stores metric CSVs. "
            "Default: eval_metrics. Falls back to legacy flat paths when files are missing."
        ),
    )
    return parser.parse_args()


def format_float(value: object) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    return f"{float(value):.6f}"


def build_ascii_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    lines = [sep, header_line, sep]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    lines.append(sep)
    return lines


def build_two_level_table(
    index_name: str,
    row_names: Sequence[str],
    grouped_columns: Sequence[tuple[str, str, str]],
    values: Dict[str, Dict[str, float]],
) -> List[str]:
    top_headers = [index_name] + [group for group, _, _ in grouped_columns]
    sub_headers = [""] + [sub for _, sub, _ in grouped_columns]

    rows: List[List[str]] = []
    for row_name in row_names:
        row = [row_name]
        row_values = values.get(row_name, {})
        for _, _, metric_key in grouped_columns:
            row.append(format_float(row_values.get(metric_key)))
        rows.append(row)

    widths = [max(len(top_headers[i]), len(sub_headers[i])) for i in range(len(top_headers))]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep]
    lines.append("| " + " | ".join(top_headers[i].ljust(widths[i]) for i in range(len(top_headers))) + " |")
    lines.append("| " + " | ".join(sub_headers[i].ljust(widths[i]) for i in range(len(sub_headers))) + " |")
    lines.append(sep)
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    lines.append(sep)
    return lines


def normalize_language(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"zh", "zho", "chs", "cn", "chinese", "mandarin"}:
        return "chs"
    if text in {"en", "eng", "english"}:
        return "en"
    return text


def ordered_names(names: Iterable[str], preferred: Sequence[str]) -> List[str]:
    clean = [str(name) for name in names if str(name)]
    seen = set()
    ordered = []
    for name in preferred:
        if name in clean and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in sorted(clean):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def resolve_metric_csv(base_dir: Path, metrics_subdir: str, filename: str) -> Path:
    candidates: List[Path] = []
    subdir = metrics_subdir.strip().strip("/")
    if subdir and subdir != ".":
        candidates.append(base_dir / subdir / filename)
    candidates.append(base_dir / filename)

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "Required CSV not found. Tried: {}".format(
            ", ".join(str(path) for path in candidates)
        )
    )


def mean_by_group(
    df: pd.DataFrame,
    group_col: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, *metrics])
    return (
        df.groupby(group_col, dropna=False)[list(metrics)]
        .mean(numeric_only=True)
        .reset_index()
    )


def to_metric_map(
    df: pd.DataFrame,
    key_col: str,
    rename_map: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return result

    for row in df.to_dict(orient="records"):
        key = str(row[key_col])
        result[key] = {}
        for src_col, dst_col in rename_map.items():
            result[key][dst_col] = row.get(src_col)
    return result


def merge_metric_maps(*metric_maps: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    for metric_map in metric_maps:
        for name, metrics in metric_map.items():
            merged.setdefault(name, {}).update(metrics)
    return merged


def extract_language_maps(
    timing_df: pd.DataFrame,
    ter_df: pd.DataFrame,
) -> tuple[Dict[str, str], Dict[str, str]]:
    utterance_to_lang: Dict[str, str] = {}
    dataset_to_lang: Dict[str, str] = {}

    if not timing_df.empty:
        for row in timing_df[["utterance", "dataset", "language"]].dropna(subset=["utterance"]).itertuples(index=False):
            lang = normalize_language(row.language)
            if lang:
                utterance_to_lang[str(row.utterance)] = lang
                dataset_to_lang.setdefault(str(row.dataset), lang)

    if not ter_df.empty:
        ter_lang_df = ter_df.copy()
        ter_lang_df["language"] = ter_lang_df["language"].map(normalize_language)
        for row in ter_lang_df[["mixture_utterance", "source", "language"]].dropna(subset=["mixture_utterance"]).itertuples(index=False):
            lang = normalize_language(row.language)
            if lang:
                utterance_to_lang.setdefault(str(row.mixture_utterance), lang)
                dataset_to_lang.setdefault(str(row.source), lang)

    return utterance_to_lang, dataset_to_lang


def attach_language(
    df: pd.DataFrame,
    utterance_col: str,
    dataset_col: str,
    utterance_to_lang: Dict[str, str],
    dataset_to_lang: Dict[str, str],
) -> pd.DataFrame:
    data = df.copy()
    data["language"] = data[utterance_col].map(utterance_to_lang)
    fallback = data[dataset_col].map(dataset_to_lang)
    data["language"] = data["language"].fillna(fallback)
    data["language"] = data["language"].map(normalize_language)
    return data


def summarize_one_base_dir(
    base_dir: Path,
    output_txt_name: str | None,
    metrics_subdir: str,
) -> Path:
    base_name = base_dir.name
    output_txt = base_dir / (output_txt_name or f"{base_name}_summary.txt")

    ter_path = resolve_metric_csv(base_dir, metrics_subdir, f"{base_name}_TER.csv")
    sim_path = resolve_metric_csv(base_dir, metrics_subdir, f"{base_name}_spk_similarity.csv")
    sim_baseline_path = resolve_metric_csv(
        base_dir, metrics_subdir, f"{base_name}_spk_similarity_mixture_enrol.csv"
    )
    dnsmos_path = resolve_metric_csv(base_dir, metrics_subdir, f"{base_name}_dnsmos.csv")
    timing_path = resolve_metric_csv(base_dir, metrics_subdir, f"{base_name}_TSE_TIMING.csv")

    ter_df = pd.read_csv(ter_path)
    sim_df = pd.read_csv(sim_path)
    sim_baseline_df = pd.read_csv(sim_baseline_path)
    dnsmos_df = pd.read_csv(dnsmos_path)
    timing_df = pd.read_csv(timing_path)

    utterance_to_lang, dataset_to_lang = extract_language_maps(timing_df, ter_df)

    ter_work = ter_df.copy()
    ter_work["language"] = ter_work["language"].map(normalize_language)
    ter_work = ter_work.rename(columns={"source": "dataset"})

    sim_work = sim_df[sim_df["status"] == "ok"].copy()
    sim_work["language"] = sim_work["language"].map(normalize_language)

    sim_baseline_work = sim_baseline_df[sim_baseline_df["status"] == "ok"].copy()
    sim_baseline_work["language"] = sim_baseline_work["language"].map(normalize_language)

    timing_work = timing_df.copy()
    timing_work["language"] = timing_work["language"].map(normalize_language)

    dnsmos_work = dnsmos_df[dnsmos_df["status"] == "ok"].copy()
    dnsmos_work = attach_language(
        dnsmos_work,
        utterance_col="utterance",
        dataset_col="dataset",
        utterance_to_lang=utterance_to_lang,
        dataset_to_lang=dataset_to_lang,
    )

    ter_dataset = mean_by_group(ter_work, "dataset", ["wer_or_cer"]).rename(
        columns={"wer_or_cer": "ter_whisper"}
    )
    ter_lang = mean_by_group(ter_work, "language", ["wer_or_cer"]).rename(
        columns={"language": "lang", "wer_or_cer": "ter_whisper"}
    )

    sim_dataset = mean_by_group(sim_work, "dataset", ["speaker_cosine_similarity"]).rename(
        columns={"speaker_cosine_similarity": "sim_enrol_tse"}
    )
    sim_lang = mean_by_group(sim_work, "language", ["speaker_cosine_similarity"]).rename(
        columns={"language": "lang", "speaker_cosine_similarity": "sim_enrol_tse"}
    )

    sim_baseline_dataset = mean_by_group(
        sim_baseline_work, "dataset", ["speaker_cosine_similarity"]
    ).rename(columns={"speaker_cosine_similarity": "sim_enrol_mixture"})
    sim_baseline_lang = mean_by_group(
        sim_baseline_work, "language", ["speaker_cosine_similarity"]
    ).rename(columns={"language": "lang", "speaker_cosine_similarity": "sim_enrol_mixture"})

    dnsmos_dataset = mean_by_group(dnsmos_work, "dataset", ["SIG", "BAK", "OVRL", "P808"]).rename(
        columns={
            "SIG": "dnsmos_sig",
            "BAK": "dnsmos_bak",
            "OVRL": "dnsmos_ovrl",
            "P808": "dnsmos_p808",
        }
    )
    dnsmos_lang = mean_by_group(dnsmos_work, "language", ["SIG", "BAK", "OVRL", "P808"]).rename(
        columns={
            "language": "lang",
            "SIG": "dnsmos_sig",
            "BAK": "dnsmos_bak",
            "OVRL": "dnsmos_ovrl",
            "P808": "dnsmos_p808",
        }
    )

    timing_dataset = mean_by_group(timing_work, "dataset", ["precision", "recall", "f1"]).rename(
        columns={"precision": "ratio_precision", "recall": "ratio_recall", "f1": "ratio_f1"}
    )
    timing_lang = mean_by_group(timing_work, "language", ["precision", "recall", "f1"]).rename(
        columns={"language": "lang", "precision": "ratio_precision", "recall": "ratio_recall", "f1": "ratio_f1"}
    )

    dataset_values = merge_metric_maps(
        to_metric_map(ter_dataset, "dataset", {"ter_whisper": "ter_whisper"}),
        to_metric_map(sim_baseline_dataset, "dataset", {"sim_enrol_mixture": "sim_enrol_mixture"}),
        to_metric_map(sim_dataset, "dataset", {"sim_enrol_tse": "sim_enrol_tse"}),
        to_metric_map(
            dnsmos_dataset,
            "dataset",
            {
                "dnsmos_sig": "dnsmos_sig",
                "dnsmos_bak": "dnsmos_bak",
                "dnsmos_ovrl": "dnsmos_ovrl",
                "dnsmos_p808": "dnsmos_p808",
            },
        ),
        to_metric_map(
            timing_dataset,
            "dataset",
            {
                "ratio_precision": "ratio_precision",
                "ratio_recall": "ratio_recall",
                "ratio_f1": "ratio_f1",
            },
        ),
    )

    lang_values = merge_metric_maps(
        to_metric_map(ter_lang, "lang", {"ter_whisper": "ter_whisper"}),
        to_metric_map(sim_baseline_lang, "lang", {"sim_enrol_mixture": "sim_enrol_mixture"}),
        to_metric_map(sim_lang, "lang", {"sim_enrol_tse": "sim_enrol_tse"}),
        to_metric_map(
            dnsmos_lang,
            "lang",
            {
                "dnsmos_sig": "dnsmos_sig",
                "dnsmos_bak": "dnsmos_bak",
                "dnsmos_ovrl": "dnsmos_ovrl",
                "dnsmos_p808": "dnsmos_p808",
            },
        ),
        to_metric_map(
            timing_lang,
            "lang",
            {
                "ratio_precision": "ratio_precision",
                "ratio_recall": "ratio_recall",
                "ratio_f1": "ratio_f1",
            },
        ),
    )

    dataset_names = ordered_names(dataset_values.keys(), DEFAULT_DATASET_ORDER)
    lang_names = ordered_names(lang_values.keys(), DEFAULT_LANG_ORDER)

    grouped_columns = [
        ("TER", "fireredasr-1/whisper", "ter_whisper"),
        ("SIM", "enrol-mixture", "sim_enrol_mixture"),
        ("SIM", "enrol-tse", "sim_enrol_tse"),
        ("DNSMOS", "SIG", "dnsmos_sig"),
        ("DNSMOS", "BAK", "dnsmos_bak"),
        ("DNSMOS", "OVRL", "dnsmos_ovrl"),
        ("DNSMOS", "P808", "dnsmos_p808"),
        ("RATIO", "precision", "ratio_precision"),
        ("RATIO", "recall", "ratio_recall"),
        ("RATIO", "f1", "ratio_f1"),
    ]

    lines: List[str] = []
    lines.append("REAL-T Aggregated Evaluation Summary")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Base dir: {base_dir}")
    lines.append("")
    lines.append("Mean by dataset")
    lines.extend(build_two_level_table("dataset", dataset_names, grouped_columns, dataset_values))
    lines.append("")
    lines.append("Mean by language")
    lines.extend(build_two_level_table("lang", lang_names, grouped_columns, lang_values))
    lines.append("")
    lines.append("Input CSVs")
    lines.extend(
        build_ascii_table(
            ["metric", "file"],
            [
                ["TER", str(ter_path)],
                ["SIM enrol-mixture", str(sim_baseline_path)],
                ["SIM enrol-tse", str(sim_path)],
                ["DNSMOS", str(dnsmos_path)],
                ["RATIO precision/recall/f1", str(timing_path)],
            ],
        )
    )
    lines.append("")
    lines.append("Notes")
    lines.append("  RATIO columns are the mean of precision / recall / f1 from TSE_TIMING CSV.")
    lines.append("  All values are recomputed from CSV files instead of reusing existing summary TXT files.")

    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_txt


def main() -> None:
    args = parse_args()
    for base_dir_raw in args.base_dir:
        base_dir = Path(base_dir_raw).resolve()
        if not base_dir.is_dir():
            raise SystemExit(f"Base directory not found: {base_dir}")
        output_txt = summarize_one_base_dir(
            base_dir=base_dir,
            output_txt_name=args.output_txt_name,
            metrics_subdir=args.metrics_subdir,
        )
        print(f"[Saved] {output_txt}")


if __name__ == "__main__":
    main()
