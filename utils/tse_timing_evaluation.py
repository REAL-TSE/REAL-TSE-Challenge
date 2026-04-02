import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-8

# Dataset -> language for CN/EN statistics (same as transcribe_and_evaluation.sh)
DATASET_LANGUAGE = {
    "AISHELL-4": "zh",
    "AliMeeting": "zh",
    "AMI": "en",
    "CHiME6": "en",
    "DipCo": "en",
}


@dataclass
class EvalCounters:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    missing_meta: int = 0
    missing_vad: int = 0
    missing_json: int = 0
    missing_record: int = 0
    malformed_utterance: int = 0


def parse_mixture_utterance(mixture_utterance: str) -> Tuple[str, float, float]:
    pattern = r"^(.+)_mixture_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)$"
    match = re.match(pattern, mixture_utterance)
    if match is None:
        raise ValueError(f"Invalid mixture_utterance format: {mixture_utterance}")
    session_id = match.group(1)
    mix_start = float(match.group(2))
    mix_end = float(match.group(3))
    return session_id, mix_start, mix_end


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for cur_start, cur_end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))
    return merged


def clip_intervals(intervals: List[Tuple[float, float]], start: float, end: float) -> List[Tuple[float, float]]:
    clipped = []
    for seg_start, seg_end in intervals:
        seg_start = max(start, seg_start)
        seg_end = min(end, seg_end)
        if seg_end > seg_start:
            clipped.append((seg_start, seg_end))
    return merge_intervals(clipped)


def apply_collar_to_segments(
    intervals: List[Tuple[float, float]], collar: float, duration: float
) -> List[Tuple[float, float]]:
    expanded = [(seg_start - collar, seg_end + collar) for seg_start, seg_end in intervals]
    return clip_intervals(expanded, 0.0, duration)


def segments_to_mask(
    intervals: List[Tuple[float, float]], duration: float, frame_shift: float
) -> np.ndarray:
    if duration <= 0:
        return np.zeros(0, dtype=bool)
    n_frames = int(math.ceil(duration / frame_shift))
    mask = np.zeros(n_frames, dtype=bool)
    for seg_start, seg_end in intervals:
        seg_start = max(0.0, min(duration, seg_start))
        seg_end = max(0.0, min(duration, seg_end))
        if seg_end <= seg_start:
            continue
        start_idx = max(0, int(math.floor(seg_start / frame_shift)))
        end_idx = min(n_frames, int(math.ceil(seg_end / frame_shift)))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = True
    return mask


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= EPS:
        return 0.0
    return numerator / denominator


def compute_frame_metrics(
    pred_segments: List[Tuple[float, float]],
    gt_segments: List[Tuple[float, float]],
    duration: float,
    frame_shift: float,
) -> Dict[str, float]:
    pred_mask = segments_to_mask(pred_segments, duration, frame_shift)
    gt_mask = segments_to_mask(gt_segments, duration, frame_shift)

    tp = float(np.sum(pred_mask & gt_mask) * frame_shift)
    fp = float(np.sum(pred_mask & (~gt_mask)) * frame_shift)
    fn = float(np.sum((~pred_mask) & gt_mask) * frame_shift)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)

    return {
        "tp_dur": tp,
        "fp_dur": fp,
        "fn_dur": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_speech_dur": float(np.sum(gt_mask) * frame_shift),
        "pred_speech_dur": float(np.sum(pred_mask) * frame_shift),
    }


def load_vad_predictions(vad_jsonl_path: Path) -> Dict[str, Dict]:
    vad_map = {}
    with open(vad_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            utterance = obj["utterance"]
            vad_map[utterance] = obj
    return vad_map


def load_label_segments(label_jsonl_path: Path) -> Dict[str, Dict]:
    """Load label_segments.jsonl: utterance -> {mix_duration, label_segments (list of (s,e) for target_speaker)}.
    Supports: (1) segments_by_speaker + target_speaker -> use that speaker's segments;
              (2) legacy label_segments -> use as-is.
    """
    label_map = {}
    with open(label_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            utterance = obj["utterance"]
            mix_duration = float(obj["mix_duration"])
            if "segments_by_speaker" in obj:
                target_speaker = obj.get("target_speaker", "")
                by_speaker = obj["segments_by_speaker"]
                segs = by_speaker.get(target_speaker, [])
                label_segments = [(float(s), float(e)) for s, e in segs]
            else:
                label_segments = [(s, e) for s, e in obj.get("label_segments", [])]
            label_map[utterance] = {
                "mix_duration": mix_duration,
                "label_segments": label_segments,
            }
    return label_map


def build_meta_mapping(meta_df: pd.DataFrame) -> Dict[str, Dict]:
    mapping = {}
    for _, row in meta_df.iterrows():
        utt_key = f"{row['mixture_utterance']}-{row['enrolment_speakers_utterance']}"
        if utt_key not in mapping:
            mapping[utt_key] = row.to_dict()
    return mapping


def find_overlap_record(
    records_for_session: List[Dict], mix_start: float, mix_end: float, tol: float
) -> Optional[Dict]:
    for item in records_for_session:
        info = item.get("overlap_info", {})
        start = safe_float(info.get("overlap_start_time"))
        end = safe_float(info.get("overlap_end_time"))
        if abs(start - mix_start) <= tol and abs(end - mix_end) <= tol:
            return item
    return None


def parse_segments(segments: List) -> List[Tuple[float, float]]:
    parsed = []
    for seg in segments:
        if len(seg) != 2:
            continue
        seg_start = safe_float(seg[0])
        seg_end = safe_float(seg[1])
        if np.isnan(seg_start) or np.isnan(seg_end):
            continue
        if seg_end > seg_start:
            parsed.append((seg_start, seg_end))
    return parsed


def evaluate_dataset(
    dataset_name: str,
    dataset_dir: Path,
    ground_truth_dir: Path,
    gt_json_base_dir: Path,
    vad_dir_name: str,
    vad_jsonl_name: str,
    frame_shift: float,
    collar: float,
    match_tolerance: float,
) -> Tuple[pd.DataFrame, EvalCounters]:
    counters = EvalCounters()
    detail_rows = []

    mapping_path = dataset_dir / "tse_audio_mapping.csv"
    vad_jsonl_path = dataset_dir / vad_dir_name / vad_jsonl_name
    label_jsonl_path = dataset_dir / vad_dir_name / "label_segments.jsonl"

    if not mapping_path.exists():
        print(f"[WARN] mapping csv missing, skip dataset {dataset_name}: {mapping_path}")
        return pd.DataFrame(), counters
    if not vad_jsonl_path.exists():
        print(f"[WARN] vad jsonl missing, skip dataset {dataset_name}: {vad_jsonl_path}")
        return pd.DataFrame(), counters

    mapping_df = pd.read_csv(mapping_path)
    vad_map = load_vad_predictions(vad_jsonl_path)
    counters.total = len(mapping_df)

    # Phase-2: prefer label_segments.jsonl only (no meta/overlap read)
    if label_jsonl_path.exists():
        label_map = load_label_segments(label_jsonl_path)
        for _, row in mapping_df.iterrows():
            utterance = row["utterance"]
            rel_audio_path = row["path"]
            if utterance not in vad_map:
                counters.missing_vad += 1
                continue
            if utterance not in label_map:
                counters.skipped += 1
                continue
            lab = label_map[utterance]
            mix_duration = lab["mix_duration"]
            label_segments = lab["label_segments"]
            pred_segments_raw = vad_map[utterance].get("pred_segments", [])
            pred_segments = clip_intervals(parse_segments(pred_segments_raw), 0.0, mix_duration)
            metrics = compute_frame_metrics(pred_segments, label_segments, mix_duration, frame_shift)
            lang = DATASET_LANGUAGE.get(dataset_name, "en")
            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "language": lang,
                    "utterance": utterance,
                    "path": rel_audio_path,
                    "mixture_utterance": "",
                    "speaker": "",
                    "mix_start": float("nan"),
                    "mix_end": float("nan"),
                    "mix_duration": mix_duration,
                    "gt_segment_count": len(label_segments),
                    "pred_segment_count": len(pred_segments),
                    "gt_empty": int(len(label_segments) == 0),
                    **metrics,
                }
            )
            counters.processed += 1
        return pd.DataFrame(detail_rows), counters

    # Fallback: read meta + overlap (phase-1 style)
    meta_path = ground_truth_dir / f"{dataset_name}_meta.csv"
    overlap_json_path = gt_json_base_dir / dataset_name / "overlap_records.json"
    if not meta_path.exists():
        print(f"[WARN] meta csv missing, skip dataset {dataset_name}: {meta_path}")
        return pd.DataFrame(), counters
    if not overlap_json_path.exists():
        print(f"[WARN] overlap json missing, skip dataset {dataset_name}: {overlap_json_path}")
        return pd.DataFrame(), counters

    meta_df = pd.read_csv(meta_path)
    meta_map = build_meta_mapping(meta_df)
    with open(overlap_json_path, "r", encoding="utf-8") as f:
        overlap_data = json.load(f)

    for _, row in mapping_df.iterrows():
        utterance = row["utterance"]
        rel_audio_path = row["path"]

        if utterance not in meta_map:
            counters.skipped += 1
            counters.missing_meta += 1
            continue
        if utterance not in vad_map:
            counters.skipped += 1
            counters.missing_vad += 1
            continue

        meta_row = meta_map[utterance]
        mixture_utterance = meta_row["mixture_utterance"]
        speaker = meta_row["speaker"]

        try:
            session_id, mix_start, mix_end = parse_mixture_utterance(mixture_utterance)
        except ValueError:
            counters.skipped += 1
            counters.malformed_utterance += 1
            continue

        if session_id not in overlap_data:
            counters.skipped += 1
            counters.missing_json += 1
            continue

        mix_duration = max(0.0, mix_end - mix_start)
        records_for_session = overlap_data[session_id]
        matched_record = find_overlap_record(records_for_session, mix_start, mix_end, match_tolerance)
        if matched_record is None:
            counters.skipped += 1
            counters.missing_record += 1
            continue

        overlap_segments = matched_record.get("overlap_segments", [])
        gt_abs_segments = []
        for seg in overlap_segments:
            if seg.get("speaker") != speaker:
                continue
            seg_start = safe_float(seg.get("start_time"))
            seg_end = safe_float(seg.get("end_time"))
            if np.isnan(seg_start) or np.isnan(seg_end):
                continue
            gt_abs_segments.append((seg_start, seg_end))

        gt_rel_segments = [(seg_start - mix_start, seg_end - mix_start) for seg_start, seg_end in gt_abs_segments]
        gt_rel_segments = clip_intervals(gt_rel_segments, 0.0, mix_duration)
        gt_rel_segments_collar = apply_collar_to_segments(gt_rel_segments, collar, mix_duration)

        pred_segments_raw = vad_map[utterance].get("pred_segments", [])
        pred_segments = clip_intervals(parse_segments(pred_segments_raw), 0.0, mix_duration)

        metrics = compute_frame_metrics(pred_segments, gt_rel_segments_collar, mix_duration, frame_shift)

        lang = DATASET_LANGUAGE.get(dataset_name, "en")
        detail_rows.append(
            {
                "dataset": dataset_name,
                "language": lang,
                "utterance": utterance,
                "path": rel_audio_path,
                "mixture_utterance": mixture_utterance,
                "speaker": speaker,
                "mix_start": mix_start,
                "mix_end": mix_end,
                "mix_duration": mix_duration,
                "gt_segment_count": len(gt_rel_segments),
                "pred_segment_count": len(pred_segments),
                "gt_empty": int(len(gt_rel_segments) == 0),
                **metrics,
            }
        )
        counters.processed += 1

    return pd.DataFrame(detail_rows), counters


def summarize_micro(
    df: pd.DataFrame, group_key: str = "dataset", include_total: bool = True
) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(group_key):
        tp = group["tp_dur"].sum()
        fp = group["fp_dur"].sum()
        fn = group["fn_dur"].sum()
        rows.append(
            {
                group_key: name,
                "samples": len(group),
                "tp_dur": tp,
                "fp_dur": fp,
                "fn_dur": fn,
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
                "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
                "gt_speech_dur": group["gt_speech_dur"].sum(),
                "pred_speech_dur": group["pred_speech_dur"].sum(),
            }
        )
    if include_total:
        tp = df["tp_dur"].sum()
        fp = df["fp_dur"].sum()
        fn = df["fn_dur"].sum()
        rows.append(
            {
                group_key: "__ALL__",
                "samples": len(df),
                "tp_dur": tp,
                "fp_dur": fp,
                "fn_dur": fn,
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
                "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
                "gt_speech_dur": df["gt_speech_dur"].sum(),
                "pred_speech_dur": df["pred_speech_dur"].sum(),
            }
        )
    return pd.DataFrame(rows)


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def format_duration(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.2f}"


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


def format_kv_block(items: List[Tuple[str, str]]) -> List[str]:
    if not items:
        return []
    max_key_len = max(len(k) for k, _ in items)
    return [f"  {key.ljust(max_key_len)} : {value}" for key, value in items]


def _format_summary_table(
    df: pd.DataFrame,
    group_col: str,
    include_total_row: bool,
) -> List[str]:
    if df is None or df.empty:
        return ["  (no rows)"]

    rows = []
    for row in df.itertuples(index=False):
        name = getattr(row, group_col)
        if not include_total_row and name == "__ALL__":
            continue
        rows.append(
            [
                str(name),
                str(int(row.samples)),
                format_duration(float(row.tp_dur)),
                format_duration(float(row.fp_dur)),
                format_duration(float(row.fn_dur)),
                format_float(float(row.precision)),
                format_float(float(row.recall)),
                format_float(float(row.f1)),
                format_duration(float(row.gt_speech_dur)),
                format_duration(float(row.pred_speech_dur)),
            ]
        )

    if not rows:
        return ["  (no rows)"]

    return build_ascii_table(
        [
            group_col,
            "samples",
            "tp_dur_s",
            "fp_dur_s",
            "fn_dur_s",
            "precision",
            "recall",
            "f1",
            "gt_speech_s",
            "pred_speech_s",
        ],
        rows,
    )


def _format_counter_table(dataset_counters: Dict[str, EvalCounters]) -> List[str]:
    if not dataset_counters:
        return ["  (no rows)"]

    rows = []
    for dataset_name in sorted(dataset_counters):
        c = dataset_counters[dataset_name]
        rows.append(
            [
                dataset_name,
                str(int(c.total)),
                str(int(c.processed)),
                str(int(c.skipped)),
                str(int(c.missing_meta)),
                str(int(c.missing_vad)),
                str(int(c.missing_json)),
                str(int(c.missing_record)),
                str(int(c.malformed_utterance)),
            ]
        )

    return build_ascii_table(
        [
            "dataset",
            "total",
            "processed",
            "skipped",
            "missing_meta",
            "missing_vad",
            "missing_json",
            "missing_record",
            "bad_utt",
        ],
        rows,
    )


def write_report(
    report_path: Path,
    summary_by_dataset: Optional[pd.DataFrame],
    summary_by_language: Optional[pd.DataFrame],
    dataset_counters: Dict[str, EvalCounters],
    args: argparse.Namespace,
    save_path: Optional[Path] = None,
):
    """Write a readable timing-evaluation report with summary blocks and ASCII tables."""
    lines = []
    lines.append("TSE Timing Evaluation Summary")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Metric type: frame-level micro Precision / Recall / F1")
    lines.append("")

    if summary_by_dataset is None or summary_by_dataset.empty:
        lines.append("No valid samples evaluated.")
    else:
        all_row = summary_by_dataset[summary_by_dataset["dataset"] == "__ALL__"]
        if not all_row.empty:
            r = all_row.iloc[0]
            lines.append("Overall Statistics")
            lines.extend(
                format_kv_block(
                    [
                        ("Total samples", str(int(r["samples"]))),
                        ("TP duration (s)", format_duration(float(r["tp_dur"]))),
                        ("FP duration (s)", format_duration(float(r["fp_dur"]))),
                        ("FN duration (s)", format_duration(float(r["fn_dur"]))),
                        ("Micro precision", format_float(float(r["precision"]))),
                        ("Micro recall", format_float(float(r["recall"]))),
                        ("Micro F1", format_float(float(r["f1"]))),
                        ("GT speech duration (s)", format_duration(float(r["gt_speech_dur"]))),
                        ("Pred speech duration (s)", format_duration(float(r["pred_speech_dur"]))),
                    ]
                )
            )
            lines.append("  Note: overall metrics are computed by accumulating TP / FP / FN first.")
            lines.append("")

        lines.append("Per-dataset Statistics")
        lines.extend(_format_summary_table(summary_by_dataset, "dataset", include_total_row=False))
        lines.append("")

        if summary_by_language is not None and not summary_by_language.empty:
            lines.append("Per-language Statistics")
            lines.extend(_format_summary_table(summary_by_language, "language", include_total_row=True))
            lines.append("")

    lines.append("Processing Summary")
    lines.extend(_format_counter_table(dataset_counters))
    lines.append("")

    if save_path is not None:
        lines.append("Output Files")
        lines.extend(format_kv_block([("Detail CSV", str(save_path))]))

    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def main(args):
    predicted_dir = Path(args.predicted_dir).resolve()
    ground_truth_dir = Path(args.ground_truth_dir).resolve()
    gt_json_base_dir = Path(args.gt_json_base_dir).resolve()

    if not predicted_dir.exists():
        raise FileNotFoundError(f"predicted_dir not found: {predicted_dir}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"ground_truth_dir not found: {ground_truth_dir}")
    if not gt_json_base_dir.exists():
        raise FileNotFoundError(f"gt_json_base_dir not found: {gt_json_base_dir}")

    dataset_dirs = sorted([p for p in predicted_dir.iterdir() if p.is_dir()])
    if args.datasets:
        allowed = set(s.strip() for s in args.datasets.split() if s.strip())
        dataset_dirs = [p for p in dataset_dirs if p.name in allowed]
        if not dataset_dirs:
            print(f"No dataset directories match --datasets in: {predicted_dir}")
            return
    if not dataset_dirs:
        print(f"No dataset directories found under: {predicted_dir}")
        return

    detail_dfs = []
    dataset_counters = {}
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        detail_df, counters = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            ground_truth_dir=ground_truth_dir,
            gt_json_base_dir=gt_json_base_dir,
            vad_dir_name=args.vad_dir_name,
            vad_jsonl_name=args.vad_jsonl_name,
            frame_shift=args.frame_shift,
            collar=args.collar,
            match_tolerance=args.match_tolerance,
        )
        dataset_counters[dataset_name] = counters
        if not detail_df.empty:
            detail_dfs.append(detail_df)

    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path = Path(args.summary_csv_path).resolve() if args.summary_csv_path else None
    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not detail_dfs:
        detail_df = pd.DataFrame(
            columns=[
                "dataset",
                "utterance",
                "path",
                "mixture_utterance",
                "speaker",
                "mix_start",
                "mix_end",
                "mix_duration",
                "gt_segment_count",
                "pred_segment_count",
                "gt_empty",
                "tp_dur",
                "fp_dur",
                "fn_dur",
                "precision",
                "recall",
                "f1",
                "gt_speech_dur",
                "pred_speech_dur",
            ]
        )
        summary_df = pd.DataFrame(
            columns=[
                "dataset",
                "samples",
                "tp_dur",
                "fp_dur",
                "fn_dur",
                "precision",
                "recall",
                "f1",
                "gt_speech_dur",
                "pred_speech_dur",
            ]
        )
        detail_df.to_csv(save_path, index=False)
        if summary_csv_path is not None:
            summary_df.to_csv(summary_csv_path, index=False)
        write_report(report_path, summary_df, None, dataset_counters, args, save_path)
        print("No valid samples evaluated. Please check mapping/meta/json/vad inputs.")
        print(f"Saved empty detail csv : {save_path}")
        if summary_csv_path is not None:
            print(f"Saved empty summary csv: {summary_csv_path}")
        print(f"Saved report txt       : {report_path}")
        return

    detail_df = pd.concat(detail_dfs, ignore_index=True)
    summary_df = summarize_micro(detail_df, group_key="dataset")
    summary_lang = summarize_micro(detail_df, group_key="language", include_total=False)
    detail_df.to_csv(save_path, index=False)
    if summary_csv_path is not None:
        summary_df.to_csv(summary_csv_path, index=False)
    write_report(report_path, summary_df, summary_lang, dataset_counters, args, save_path)

    print(f"Saved detail csv   : {save_path}")
    if summary_csv_path is not None:
        print(f"Saved summary csv  : {summary_csv_path}")
    print(f"Saved report txt   : {report_path}")
    print("")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TSE timing via FireRedVAD frame-level metrics.")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Path to REAL-T EVAL meta csv dir")
    parser.add_argument("--predicted_dir", type=str, required=True, help="Path like output/EVAL/bsrnn_vox1")
    parser.add_argument(
        "--gt_json_base_dir",
        type=str,
        required=True,
        help="Path to GT overlap json (e.g. ./datasets/REAL-T/json, see README)",
    )
    parser.add_argument("--datasets", type=str, default=None,
        help="Space-separated dataset names to evaluate (default: all subdirs under predicted_dir)",
    )
    parser.add_argument("--vad_dir_name", type=str, default="FireRedVAD")
    parser.add_argument("--vad_jsonl_name", type=str, default="vad_segments.jsonl")
    parser.add_argument("--frame_shift", type=float, default=0.01)
    parser.add_argument("--collar", type=float, default=0.05)
    parser.add_argument("--match_tolerance", type=float, default=0.02)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save detail csv")
    parser.add_argument("--summary_csv_path", type=str, default=None, help="Path to save summary csv (optional, not written if omitted)")
    parser.add_argument("--report_path", type=str, required=True, help="Path to save report txt")
    main(parser.parse_args())
