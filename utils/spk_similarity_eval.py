#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


OUTPUT_COLUMNS = [
    "output_dir",
    "dataset",
    "utterance",
    "mixture_utterance",
    "enrolment_speakers_utterance",
    "speaker",
    "language",
    "pair_mode",
    "estimation_path",
    "reference_path",
    "speaker_cosine_similarity",
    "status",
    "error_message",
]

PAIR_MODE_CHOICES = ("tse_enrol", "mixture_enrol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute speaker cosine similarity for one of: "
            "TSE estimation vs enrolment (tse_enrol), "
            "or mixture vs enrolment baseline (mixture_enrol)."
        )
    )
    parser.add_argument(
        "--output_dir",
        action="append",
        required=True,
        help="TSE output directory. Repeat this argument for multiple model result roots.",
    )
    parser.add_argument(
        "--test_set_dir",
        default=None,
        help="Directory containing *_meta.csv (e.g., ./datasets/REAL-T/DEV). Required unless --regen_txt_only.",
    )
    parser.add_argument(
        "--mapping_csv",
        default=None,
        help="Path to mapping.csv (utterance -> wav path for mixture/enrolment). Required unless --regen_txt_only.",
    )
    parser.add_argument(
        "--wespeaker_lang",
        default="en",
        help="Language argument for wespeakerruntime.Speaker(lang=...).",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="ONNX Runtime provider policy. auto prefers CUDA when available.",
    )
    parser.add_argument(
        "--dataset_lang_overrides",
        default="AISHELL-4:chs,AliMeeting:chs",
        help=(
            "Comma-separated dataset->lang mapping, e.g. "
            "'AISHELL-4:chs,AliMeeting:chs'. Unlisted datasets use --wespeaker_lang."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of processed rows per output_dir.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Reserved for future use. Current implementation runs in serial only.",
    )
    parser.add_argument(
        "--output_csv_name",
        default=None,
        help=(
            "Output CSV filename under each output_dir. "
            "Default: <output_name>_spk_similarity.csv (tse_enrol) or "
            "<output_name>_spk_similarity_mixture_enrol.csv (mixture_enrol)."
        ),
    )
    parser.add_argument(
        "--output_txt_name",
        default=None,
        help=(
            "Output TXT filename under each output_dir. "
            "Default: <output_name>_spk_similarity_summary.txt (tse_enrol) or "
            "<output_name>_spk_similarity_mixture_enrol_summary.txt (mixture_enrol)."
        ),
    )
    parser.add_argument(
        "--pair_mode",
        choices=PAIR_MODE_CHOICES,
        default="tse_enrol",
        help=(
            "Similarity pair mode. "
            "tse_enrol=compute similarity between TSE output and enrolment; "
            "mixture_enrol=use mixture audio as baseline against enrolment."
        ),
    )
    parser.add_argument(
        "--regen_txt_only",
        action="store_true",
        help="Only regenerate TXT from existing CSV (skip WeSpeaker, no embedding extraction).",
    )
    parser.add_argument(
        "--csv_only",
        action="store_true",
        help="When computing similarity, only write CSV (do not write summary TXT). Use mode 2 to generate TXT later.",
    )
    return parser.parse_args()


def load_reference_mapping(mapping_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(mapping_csv, usecols=["utterance", "path"])
    df = df.dropna(subset=["utterance", "path"])
    return dict(zip(df["utterance"].astype(str), df["path"].astype(str)))


def resolve_estimation_path(
    raw_path: object,
    repo_root: Path,
    output_dir: Path,
    dataset: str,
) -> Optional[Path]:
    if raw_path is None or pd.isna(raw_path):
        return None

    path_obj = Path(str(raw_path))
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj.resolve()

    repo_candidate = (repo_root / path_obj).resolve()
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = path_obj.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    fallback = (output_dir / dataset / "wav" / path_obj.name).resolve()
    if fallback.exists():
        return fallback

    return None


def resolve_reference_path(enrol_utt: str, reference_map: Dict[str, str]) -> Optional[Path]:
    raw = reference_map.get(enrol_utt)
    if raw is None:
        return None
    path_obj = Path(raw).expanduser()
    if path_obj.exists():
        return path_obj.resolve()
    return None


def get_pair_mode_labels(pair_mode: str) -> Tuple[str, str]:
    if pair_mode == "tse_enrol":
        return "estimation vs enrolment", "Estimation audio"
    if pair_mode == "mixture_enrol":
        return "mixture vs enrolment", "Mixture audio"
    raise ValueError(f"Unsupported pair_mode: {pair_mode}")


def default_output_names(output_name: str, pair_mode: str) -> Tuple[str, str]:
    if pair_mode == "tse_enrol":
        return f"{output_name}_spk_similarity.csv", f"{output_name}_spk_similarity_summary.txt"
    return (
        f"{output_name}_spk_similarity_{pair_mode}.csv",
        f"{output_name}_spk_similarity_{pair_mode}_summary.txt",
    )


def resolve_existing_csv_path(
    output_dir: Path,
    csv_name: str,
    legacy_csv_name: str,
) -> Path:
    primary = output_dir / csv_name
    candidates = [primary]
    if "/" in csv_name or "\\" in csv_name:
        legacy = output_dir / legacy_csv_name
        if legacy not in candidates:
            candidates.append(legacy)
    for path in candidates:
        if path.exists():
            return path
    return primary


def get_embedding(
    speaker_model,
    audio_path: Path,
    embedding_cache: Dict[str, object],
    cache_namespace: str,
):
    audio_path_str = str(audio_path)
    cache_key = f"{cache_namespace}::{audio_path_str}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    embedding = speaker_model.extract_embedding(audio_path_str)
    if hasattr(embedding, "squeeze"):
        embedding = embedding.squeeze()
    embedding_cache[cache_key] = embedding
    return embedding


def build_dataset_rows(
    output_dir: Path,
    dataset: str,
    test_set_dir: Path,
    reference_map: Dict[str, str],
    speaker_model,
    embedding_cache: Dict[str, object],
    cache_namespace: str,
    repo_root: Path,
    pair_mode: str,
    max_samples: Optional[int],
    processed_so_far: int,
) -> Tuple[List[dict], int]:
    rows: List[dict] = []

    meta_csv = test_set_dir / f"{dataset}_meta.csv"
    tse_mapping_csv = output_dir / dataset / "tse_audio_mapping.csv"
    if not meta_csv.exists():
        print(f"[Skip] Missing meta csv: {meta_csv}")
        return rows, processed_so_far
    if pair_mode == "tse_enrol" and not tse_mapping_csv.exists():
        print(f"[Skip] Missing tse mapping csv: {tse_mapping_csv}")
        return rows, processed_so_far

    meta_cols = [
        "mixture_utterance",
        "enrolment_speakers_utterance",
        "speaker",
        "language",
    ]
    meta_df = pd.read_csv(meta_csv, usecols=meta_cols)
    meta_df["utterance"] = (
        meta_df["mixture_utterance"].astype(str)
        + "-"
        + meta_df["enrolment_speakers_utterance"].astype(str)
    )

    if pair_mode == "tse_enrol":
        tse_df = pd.read_csv(tse_mapping_csv, usecols=["utterance", "path"])
        merged = meta_df.merge(tse_df, on="utterance", how="left")
    elif pair_mode == "mixture_enrol":
        merged = meta_df.copy()
    else:
        raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    iterator = tqdm(
        merged.itertuples(index=False),
        total=len(merged),
        desc=f"{output_dir.name}/{dataset}",
        leave=False,
    )
    _, candidate_label = get_pair_mode_labels(pair_mode)
    for item in iterator:
        if max_samples is not None and processed_so_far >= max_samples:
            break

        mixture_utt = str(item.mixture_utterance)
        enrol_utt = str(item.enrolment_speakers_utterance)
        utt = str(item.utterance)
        speaker = str(item.speaker) if not pd.isna(item.speaker) else ""
        language = str(item.language) if not pd.isna(item.language) else ""

        if pair_mode == "tse_enrol":
            estimation = resolve_estimation_path(item.path, repo_root, output_dir, dataset)
        elif pair_mode == "mixture_enrol":
            estimation = resolve_reference_path(mixture_utt, reference_map)
        else:
            raise ValueError(f"Unsupported pair_mode: {pair_mode}")
        reference = resolve_reference_path(enrol_utt, reference_map)

        score = np.nan
        status = "ok"
        error_message = ""

        if estimation is None:
            status = "missing_est"
            error_message = f"{candidate_label} not found."
        elif reference is None:
            status = "missing_ref"
            error_message = "Reference enrolment audio not found."
        else:
            try:
                est_emb = get_embedding(speaker_model, estimation, embedding_cache, cache_namespace)
                ref_emb = get_embedding(speaker_model, reference, embedding_cache, cache_namespace)
                score = float(speaker_model.compute_cosine_score(est_emb, ref_emb))
            except Exception as exc:  # pylint: disable=broad-except
                status = "error"
                error_message = f"WeSpeaker inference failed: {exc}"

        rows.append(
            {
                "output_dir": str(output_dir),
                "dataset": dataset,
                "utterance": utt,
                "mixture_utterance": mixture_utt,
                "enrolment_speakers_utterance": enrol_utt,
                "speaker": speaker,
                "language": language,
                "pair_mode": pair_mode,
                "estimation_path": str(estimation) if estimation is not None else "",
                "reference_path": str(reference) if reference is not None else "",
                "speaker_cosine_similarity": score,
                "status": status,
                "error_message": error_message,
            }
        )
        processed_so_far += 1

    return rows, processed_so_far


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.6f}"


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


def summarize(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    total = len(df)
    valid_mask = df["status"] == "ok"
    valid = int(valid_mask.sum())
    failed = total - valid
    missing_ref = int((df["status"] == "missing_ref").sum())
    missing_est = int((df["status"] == "missing_est").sum())

    score_series = pd.to_numeric(df.loc[valid_mask, "speaker_cosine_similarity"], errors="coerce").dropna()
    overall = {
        "total": total,
        "valid": valid,
        "failed": failed,
        "missing_ref": missing_ref,
        "missing_est": missing_est,
        "mean": float(score_series.mean()) if not score_series.empty else np.nan,
        "std": float(score_series.std()) if not score_series.empty else np.nan,
        "min": float(score_series.min()) if not score_series.empty else np.nan,
        "max": float(score_series.max()) if not score_series.empty else np.nan,
    }

    per_dataset = []
    for dataset, group in df.groupby("dataset", sort=True):
        g_valid_mask = group["status"] == "ok"
        g_scores = pd.to_numeric(group.loc[g_valid_mask, "speaker_cosine_similarity"], errors="coerce").dropna()
        g_total = len(group)
        g_valid = int(g_valid_mask.sum())
        per_dataset.append(
            {
                "dataset": dataset,
                "total": g_total,
                "valid": g_valid,
                "failed": g_total - g_valid,
                "missing_ref": int((group["status"] == "missing_ref").sum()),
                "missing_est": int((group["status"] == "missing_est").sum()),
                "mean": float(g_scores.mean()) if not g_scores.empty else np.nan,
                "std": float(g_scores.std()) if not g_scores.empty else np.nan,
                "min": float(g_scores.min()) if not g_scores.empty else np.nan,
                "max": float(g_scores.max()) if not g_scores.empty else np.nan,
            }
        )

    return overall, pd.DataFrame(per_dataset)


def summarize_per_language(
    df: pd.DataFrame,
    dataset_lang_overrides: Dict[str, str],
    default_lang: str,
) -> pd.DataFrame:
    """Group by language (en/chs) and compute stats. Order: en first, then chs."""
    df = df.copy()
    df["_lang"] = df["dataset"].map(
        lambda d: dataset_lang_overrides.get(d, default_lang)
    )
    per_lang = []
    for lang in ["en", "chs"]:
        group = df[df["_lang"] == lang]
        if len(group) == 0:
            continue
        g_valid_mask = group["status"] == "ok"
        g_scores = pd.to_numeric(
            group.loc[g_valid_mask, "speaker_cosine_similarity"],
            errors="coerce",
        ).dropna()
        g_total = len(group)
        g_valid = int(g_valid_mask.sum())
        per_lang.append(
            {
                "lang": lang,
                "total": g_total,
                "valid": g_valid,
                "failed": g_total - g_valid,
                "missing_ref": int((group["status"] == "missing_ref").sum()),
                "missing_est": int((group["status"] == "missing_est").sum()),
                "mean": float(g_scores.mean()) if not g_scores.empty else np.nan,
                "std": float(g_scores.std()) if not g_scores.empty else np.nan,
                "min": float(g_scores.min()) if not g_scores.empty else np.nan,
                "max": float(g_scores.max()) if not g_scores.empty else np.nan,
            }
        )
    return pd.DataFrame(per_lang)


def write_summary_txt(
    output_txt: Path,
    overall: dict,
    per_dataset_df: pd.DataFrame,
    status_counts: Dict[str, int],
    pair_mode: str,
    per_lang_df: Optional[pd.DataFrame] = None,
) -> None:
    pair_mode_label, candidate_label = get_pair_mode_labels(pair_mode)
    lines = []
    lines.append(f"Speaker Similarity Summary ({pair_mode_label})")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Pair mode: {pair_mode}")
    lines.append("")
    lines.append("Overall Statistics")
    lines.extend(
        format_kv_block(
            [
                ("Total samples", str(overall["total"])),
                ("Valid scored", str(overall["valid"])),
                ("Failed", str(overall["failed"])),
                ("Missing reference", str(overall["missing_ref"])),
                (f"Missing candidate ({candidate_label})", str(overall["missing_est"])),
                ("Mean score", format_float(overall["mean"])),
                ("Std score", format_float(overall["std"])),
                ("Min score", format_float(overall["min"])),
                ("Max score", format_float(overall["max"])),
            ]
        )
    )
    lines.append("  Note: score statistics are computed on rows with status=ok.")
    lines.append("")
    lines.append("Per-dataset Statistics")
    if per_dataset_df.empty:
        lines.append("  (no rows)")
    else:
        headers = ["dataset", "total", "valid", "failed", "missing_ref", "missing_est", "mean", "std", "min", "max"]
        rows = []
        for row in per_dataset_df.itertuples(index=False):
            rows.append(
                [
                    str(row.dataset),
                    str(int(row.total)),
                    str(int(row.valid)),
                    str(int(row.failed)),
                    str(int(row.missing_ref)),
                    str(int(row.missing_est)),
                    format_float(float(row.mean)) if not pd.isna(row.mean) else "nan",
                    format_float(float(row.std)) if not pd.isna(row.std) else "nan",
                    format_float(float(row.min)) if not pd.isna(row.min) else "nan",
                    format_float(float(row.max)) if not pd.isna(row.max) else "nan",
                ]
            )
        lines.extend(build_ascii_table(headers, rows))
    lines.append("")
    if per_lang_df is not None and not per_lang_df.empty:
        lines.append("Per-language Statistics")
        headers = ["lang", "total", "valid", "failed", "missing_ref", "missing_est", "mean", "std", "min", "max"]
        rows = []
        for row in per_lang_df.itertuples(index=False):
            rows.append(
                [
                    str(row.lang),
                    str(int(row.total)),
                    str(int(row.valid)),
                    str(int(row.failed)),
                    str(int(row.missing_ref)),
                    str(int(row.missing_est)),
                    format_float(float(row.mean)) if not pd.isna(row.mean) else "nan",
                    format_float(float(row.std)) if not pd.isna(row.std) else "nan",
                    format_float(float(row.min)) if not pd.isna(row.min) else "nan",
                    format_float(float(row.max)) if not pd.isna(row.max) else "nan",
                ]
            )
        lines.extend(build_ascii_table(headers, rows))
        lines.append("")
    lines.append("Status Distribution")
    if not status_counts:
        lines.append("  (no rows)")
    else:
        ordered_status = ["ok", "missing_ref", "missing_est", "error"]
        seen = set()
        rows = []
        for status in ordered_status:
            if status in status_counts:
                rows.append([status, str(status_counts[status])])
                seen.add(status)
        for status, count in sorted(status_counts.items()):
            if status not in seen:
                rows.append([status, str(count)])
        lines.extend(build_ascii_table(["status", "count"], rows))

    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_output_dir(
    output_dir: Path,
    test_set_dir: Path,
    reference_map: Dict[str, str],
    speaker_model_cache: Dict[str, object],
    args,
    dataset_lang_overrides: Dict[str, str],
    repo_root: Path,
    max_samples: Optional[int],
    output_csv_name: Optional[str],
    output_txt_name: Optional[str],
    pair_mode: str,
    embedding_cache: Optional[Dict[str, object]] = None,
) -> None:
    if not output_dir.exists():
        print(f"[Skip] Output dir does not exist: {output_dir}")
        return

    dataset_dirs = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and (p / "tse_audio_mapping.csv").is_file()]
    )
    if not dataset_dirs:
        print(f"[Skip] No dataset directories found under {output_dir}")
        return

    processed = 0
    if embedding_cache is None:
        embedding_cache = {}
    all_rows: List[dict] = []

    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        dataset_lang = dataset_lang_overrides.get(dataset, args.wespeaker_lang)
        speaker_model = get_or_create_speaker_model(args, dataset_lang, speaker_model_cache)
        print(f"[WeSpeaker] dataset={dataset} lang={dataset_lang}")
        rows, processed = build_dataset_rows(
            output_dir=output_dir,
            dataset=dataset,
            test_set_dir=test_set_dir,
            reference_map=reference_map,
            speaker_model=speaker_model,
            embedding_cache=embedding_cache,
            cache_namespace=dataset_lang,
            repo_root=repo_root,
            pair_mode=pair_mode,
            max_samples=max_samples,
            processed_so_far=processed,
        )
        all_rows.extend(rows)
        if max_samples is not None and processed >= max_samples:
            break

    result_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)

    default_csv_name, default_txt_name = default_output_names(output_dir.name, pair_mode)
    csv_name = output_csv_name or default_csv_name
    txt_name = output_txt_name or default_txt_name
    output_csv = output_dir / csv_name
    output_txt = output_dir / txt_name
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_csv, index=False)
    overall, per_dataset_df = summarize(result_df)
    status_counts = {str(k): int(v) for k, v in result_df["status"].value_counts(dropna=False).to_dict().items()}

    if not args.csv_only:
        per_lang_df = summarize_per_language(
            result_df, dataset_lang_overrides, args.wespeaker_lang
        )
        write_summary_txt(
            output_txt, overall, per_dataset_df, status_counts,
            pair_mode=pair_mode, per_lang_df=per_lang_df,
        )
        print(f"[Saved] TXT: {output_txt}")

    print(f"[Saved] CSV: {output_csv}")
    print(
        "[Overall] pair_mode={pair_mode} total={total} valid={valid} failed={failed} "
        "missing_ref={missing_ref} missing_est={missing_est} mean={mean} std={std} min={min} max={max}".format(
            pair_mode=pair_mode,
            total=overall["total"],
            valid=overall["valid"],
            failed=overall["failed"],
            missing_ref=overall["missing_ref"],
            missing_est=overall["missing_est"],
            mean=format_float(overall["mean"]),
            std=format_float(overall["std"]),
            min=format_float(overall["min"]),
            max=format_float(overall["max"]),
        )
    )


def init_speaker_model(args, model_lang: str):
    import onnxruntime as ort
    import wespeakerruntime as wespeaker
    from wespeakerruntime.hub import Hub

    available_providers = ort.get_available_providers()
    provider_mode = args.provider

    if provider_mode == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise SystemExit(
                "CUDAExecutionProvider is not available in current onnxruntime. "
                "Please install/configure onnxruntime-gpu correctly."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider_mode == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    onnx_path = Hub.get_model_by_lang(model_lang)

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    speaker_model = wespeaker.Speaker(onnx_path=onnx_path, lang=model_lang)
    speaker_model.session = session

    print(f"[WeSpeaker] lang={model_lang} onnx_path={onnx_path}")
    print(f"[WeSpeaker] available_providers={available_providers}")
    print(f"[WeSpeaker] active_providers={speaker_model.session.get_providers()}")

    return speaker_model


def run_regen_txt_only(
    output_dir: Path,
    dataset_lang_overrides: Dict[str, str],
    default_lang: str,
    output_csv_name: Optional[str],
    output_txt_name: Optional[str],
    pair_mode: str,
) -> None:
    """Read existing CSV and regenerate TXT with Per-language Statistics."""
    if not output_dir.exists():
        print(f"[Skip] Output dir does not exist: {output_dir}")
        return

    default_csv_name, default_txt_name = default_output_names(output_dir.name, pair_mode)
    csv_name = output_csv_name or default_csv_name
    txt_name = output_txt_name or default_txt_name
    output_csv = resolve_existing_csv_path(output_dir, csv_name, default_csv_name)
    output_txt = output_dir / txt_name

    if not output_csv.exists():
        print(f"[Skip] CSV not found: {output_csv}")
        return
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.read_csv(output_csv)
    overall, per_dataset_df = summarize(result_df)
    per_lang_df = summarize_per_language(result_df, dataset_lang_overrides, default_lang)
    status_counts = {
        str(k): int(v)
        for k, v in result_df["status"].value_counts(dropna=False).to_dict().items()
    }
    pair_mode_label, candidate_label = get_pair_mode_labels(pair_mode)
    write_summary_txt(
        output_txt, overall, per_dataset_df, status_counts,
        pair_mode=pair_mode, per_lang_df=per_lang_df,
    )
    print(f"[Saved] TXT: {output_txt}")


def get_or_create_speaker_model(args, model_lang: str, speaker_model_cache: Dict[str, object]):
    if model_lang in speaker_model_cache:
        return speaker_model_cache[model_lang]
    speaker_model_cache[model_lang] = init_speaker_model(args, model_lang)
    return speaker_model_cache[model_lang]


def parse_dataset_lang_overrides(raw: str) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if raw is None:
        return overrides
    raw = raw.strip()
    if not raw:
        return overrides

    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                f"Invalid dataset_lang_overrides item '{part}'. Expected format: dataset:lang"
            )
        dataset, lang = part.split(":", 1)
        dataset = dataset.strip()
        lang = lang.strip()
        if lang not in {"en", "chs"}:
            raise SystemExit(
                f"Invalid language '{lang}' in dataset_lang_overrides. Supported: en, chs"
            )
        if not dataset:
            raise SystemExit("dataset_lang_overrides contains empty dataset name.")
        overrides[dataset] = lang
    return overrides


def main() -> None:
    args = parse_args()

    if args.num_workers != 1:
        print(
            f"[Info] num_workers={args.num_workers} requested, but current implementation is serial. "
            "Proceeding with num_workers=1."
        )

    dataset_lang_overrides = parse_dataset_lang_overrides(args.dataset_lang_overrides)
    if dataset_lang_overrides:
        print(f"[WeSpeaker] dataset_lang_overrides={dataset_lang_overrides}")

    if args.regen_txt_only:
        for output_dir_raw in args.output_dir:
            run_regen_txt_only(
                output_dir=Path(output_dir_raw).resolve(),
                dataset_lang_overrides=dataset_lang_overrides,
                default_lang=args.wespeaker_lang,
                output_csv_name=args.output_csv_name,
                output_txt_name=args.output_txt_name,
                pair_mode=args.pair_mode,
            )
        return

    if args.test_set_dir is None or args.mapping_csv is None:
        raise SystemExit(
            "--test_set_dir and --mapping_csv are required unless --regen_txt_only."
        )

    test_set_dir = Path(args.test_set_dir).resolve()
    mapping_csv = Path(args.mapping_csv).resolve()
    repo_root = Path.cwd().resolve()

    if not test_set_dir.exists():
        raise SystemExit(f"test_set_dir does not exist: {test_set_dir}")
    if not mapping_csv.exists():
        raise SystemExit(f"mapping_csv does not exist: {mapping_csv}")

    try:
        import wespeakerruntime  # noqa: F401
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(
            "Failed to import wespeakerruntime. Please install it first: pip install wespeakerruntime"
        ) from exc

    reference_map = load_reference_mapping(mapping_csv)
    if not reference_map:
        raise SystemExit(f"No reference entries loaded from {mapping_csv}")

    speaker_model_cache: Dict[str, object] = {}

    # Share embedding cache across multiple output dirs to avoid recomputing
    # repeated enrolment embeddings when OUTPUT_DIRS contains several models.
    global_embedding_cache: Dict[str, object] = {}

    for output_dir_raw in args.output_dir:
        run_one_output_dir(
            output_dir=Path(output_dir_raw).resolve(),
            test_set_dir=test_set_dir,
            reference_map=reference_map,
            speaker_model_cache=speaker_model_cache,
            args=args,
            dataset_lang_overrides=dataset_lang_overrides,
            repo_root=repo_root,
            max_samples=args.max_samples,
            output_csv_name=args.output_csv_name,
            output_txt_name=args.output_txt_name,
            pair_mode=args.pair_mode,
            embedding_cache=global_embedding_cache,
        )


if __name__ == "__main__":
    main()
