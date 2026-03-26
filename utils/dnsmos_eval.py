#!/usr/bin/env python3
"""
Compute DNSMOS (SIG, BAK, OVRL, P808) for TSE output audios.
Uses the same data layout as spk_similarity_eval: base_dir/dataset/tse_audio_mapping.csv + test_set_dir/{dataset}_meta.csv.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# DNSMOS constants (aligned with Microsoft DNS-Challenge dnsmos_local.py)
SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

OUTPUT_COLUMNS = [
    "base_dir",
    "dataset",
    "utterance",
    "path",
    "SIG",
    "BAK",
    "OVRL",
    "P808",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute DNSMOS (SIG, BAK, OVRL, P808) for TSE output audios."
    )
    parser.add_argument(
        "--base_dir",
        action="append",
        required=True,
        help="Base TSE output directory. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--test_set_dir",
        default=None,
        help="Directory containing *_meta.csv (e.g. ./datasets/REAL-T/PRIMARY). Required unless --regen_txt_only.",
    )
    parser.add_argument(
        "--dnsmos_model_dir",
        default=None,
        help="Directory with sig_bak_ovr.onnx and model_v8.onnx. Default: env DNSMOS_MODEL_DIR or ./DNSMOS.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="ONNX Runtime provider: auto (prefer CUDA if available), cuda, or cpu. CUDA can speed up inference.",
    )
    parser.add_argument(
        "--dataset_lang_overrides",
        default="AISHELL-4:chs,AliMeeting:chs",
        help="Comma-separated dataset->lang for per-language stats (e.g. AISHELL-4:chs,AliMeeting:chs).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of rows per base_dir.",
    )
    parser.add_argument(
        "--personalized_MOS",
        action="store_true",
        help="Use personalized MOS polynomial fitting (DNSMOS -p).",
    )
    parser.add_argument(
        "--output_csv_name",
        default=None,
        help="Output CSV filename under each base_dir. Default: <base_name>_dnsmos.csv.",
    )
    parser.add_argument(
        "--output_txt_name",
        default=None,
        help="Output TXT filename under each base_dir. Default: <base_name>_dnsmos.txt.",
    )
    parser.add_argument(
        "--regen_txt_only",
        action="store_true",
        help="Only regenerate TXT from existing CSV (no DNSMOS inference).",
    )
    parser.add_argument(
        "--csv_only",
        action="store_true",
        help="When computing, only write CSV (do not write summary TXT). Use mode 2 to generate TXT later.",
    )
    parser.add_argument(
        "--no_download_models",
        action="store_true",
        help="Do not auto-download missing ONNX models; require them to exist in DNSMOS_MODEL_DIR.",
    )
    return parser.parse_args()


# ----- Optional: auto-download ONNX models from Hugging Face -----

DNSMOS_HF_REPO = "Vyvo-Research/dnsmos"
DNSMOS_FILES = ("sig_bak_ovr.onnx", "model_v8.onnx")


def _download_dnsmos_models(model_dir: Path) -> None:
    """Download sig_bak_ovr.onnx and model_v8.onnx from Hugging Face if missing."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "Auto-download requires huggingface_hub. Install with: pip install huggingface_hub"
        ) from e
    model_dir.mkdir(parents=True, exist_ok=True)
    for fname in DNSMOS_FILES:
        dest = model_dir / fname
        if dest.exists():
            continue
        print(f"[DNSMOS] Downloading {fname} from {DNSMOS_HF_REPO} ...")
        path = hf_hub_download(
            repo_id=DNSMOS_HF_REPO,
            filename=fname,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[DNSMOS] Saved to {path}")


def _ensure_dnsmos_models(model_dir: Path, allow_download: bool) -> None:
    """Ensure both ONNX files exist; optionally download from Hugging Face if missing."""
    primary = model_dir / "sig_bak_ovr.onnx"
    p808 = model_dir / "model_v8.onnx"
    if primary.exists() and p808.exists():
        return
    if allow_download:
        _download_dnsmos_models(model_dir)
        return
    raise SystemExit(
        f"DNSMOS models not found in {model_dir}. "
        f"Need sig_bak_ovr.onnx and model_v8.onnx. "
        "Run without --no_download_models to auto-download, or see README for manual download."
    )


# ----- DNSMOS inference (aligned with Microsoft DNS-Challenge dnsmos_local.py) -----


def _audio_melspec(
    audio: np.ndarray,
    n_mels: int = 120,
    frame_size: int = 320,
    hop_length: int = 160,
    sr: int = 16000,
    to_db: bool = True,
) -> np.ndarray:
    import librosa

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
    )
    if to_db:
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def _get_polyfit_val(
    sig_raw: float, bak_raw: float, ovr_raw: float, is_personalized_MOS: bool
) -> Tuple[float, float, float]:
    if is_personalized_MOS:
        p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
        p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
        p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
    else:
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    sig_poly = float(p_sig(sig_raw))
    bak_poly = float(p_bak(bak_raw))
    ovr_poly = float(p_ovr(ovr_raw))
    return sig_poly, bak_poly, ovr_poly


def _get_onnx_providers(provider: str) -> List[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise SystemExit(
                "CUDAExecutionProvider not available. Install onnxruntime-gpu or set --provider cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    # auto: prefer CUDA if available
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class DNSMOSComputeScore:
    """DNSMOS scorer using sig_bak_ovr.onnx and model_v8.onnx (P808)."""

    def __init__(
        self,
        primary_model_path: Path,
        p808_model_path: Path,
        is_personalized_MOS: bool = False,
        provider: str = "auto",
    ) -> None:
        import onnxruntime as ort

        self._is_personalized = is_personalized_MOS
        providers = _get_onnx_providers(provider)
        self._onnx_sess = ort.InferenceSession(
            str(primary_model_path),
            providers=providers,
        )
        self._p808_sess = ort.InferenceSession(
            str(p808_model_path),
            providers=providers,
        )

    def __call__(self, audio_path: Path, sampling_rate: int = SAMPLING_RATE) -> Dict[str, float]:
        import librosa
        import soundfile as sf

        aud, input_fs = sf.read(str(audio_path))
        if input_fs != sampling_rate:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=sampling_rate)
        else:
            audio = aud
        actual_len = len(audio)
        len_samples = int(INPUT_LENGTH * sampling_rate)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / sampling_rate) - INPUT_LENGTH) + 1
        hop_len_samples = sampling_rate
        pred_sig, pred_bak, pred_ovr, pred_p808 = [], [], [], []

        for idx in range(num_hops):
            start = int(idx * hop_len_samples)
            end = int((idx + INPUT_LENGTH) * hop_len_samples)
            audio_seg = audio[start:end]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg, dtype=np.float32)[np.newaxis, :]
            p808_input = np.array(
                _audio_melspec(audio=audio_seg[:-160]), dtype=np.float32
            )[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input}

            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self._onnx_sess.run(None, oi)[0][0]
            sig, bak, ovr = _get_polyfit_val(
                float(mos_sig_raw),
                float(mos_bak_raw),
                float(mos_ovr_raw),
                self._is_personalized,
            )
            p808_mos = float(self._p808_sess.run(None, p808_oi)[0][0][0])
            pred_sig.append(sig)
            pred_bak.append(bak)
            pred_ovr.append(ovr)
            pred_p808.append(p808_mos)

        return {
            "SIG": float(np.mean(pred_sig)),
            "BAK": float(np.mean(pred_bak)),
            "OVRL": float(np.mean(pred_ovr)),
            "P808": float(np.mean(pred_p808)),
        }


def resolve_estimation_path(
    raw_path: object,
    repo_root: Path,
    base_dir: Path,
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
    fallback = (base_dir / dataset / "wav" / path_obj.name).resolve()
    if fallback.exists():
        return fallback
    return None


def default_output_names(base_name: str) -> Tuple[str, str]:
    return f"{base_name}_dnsmos.csv", f"{base_name}_dnsmos.txt"


def resolve_existing_csv_path(
    base_dir: Path,
    csv_name: str,
    legacy_csv_name: str,
) -> Path:
    primary = base_dir / csv_name
    candidates = [primary]
    if "/" in csv_name or "\\" in csv_name:
        legacy = base_dir / legacy_csv_name
        if legacy not in candidates:
            candidates.append(legacy)
    for path in candidates:
        if path.exists():
            return path
    return primary


def parse_dataset_lang_overrides(raw: str) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not raw or not raw.strip():
        return overrides
    for item in raw.strip().split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        dataset, lang = part.split(":", 1)
        dataset, lang = dataset.strip(), lang.strip()
        if lang not in {"en", "chs"}:
            continue
        if dataset:
            overrides[dataset] = lang
    return overrides


def build_dataset_rows(
    base_dir: Path,
    dataset: str,
    test_set_dir: Path,
    scorer: Optional[DNSMOSComputeScore],
    repo_root: Path,
    max_samples: Optional[int],
    processed_so_far: int,
) -> Tuple[List[dict], int]:
    rows: List[dict] = []
    meta_csv = test_set_dir / f"{dataset}_meta.csv"
    tse_mapping_csv = base_dir / dataset / "tse_audio_mapping.csv"
    if not meta_csv.exists():
        print(f"[Skip] Missing meta csv: {meta_csv}")
        return rows, processed_so_far
    if not tse_mapping_csv.exists():
        print(f"[Skip] Missing tse mapping csv: {tse_mapping_csv}")
        return rows, processed_so_far

    meta_cols = ["mixture_utterance", "enrolment_speakers_utterance"]
    meta_df = pd.read_csv(meta_csv, usecols=meta_cols)
    meta_df["utterance"] = (
        meta_df["mixture_utterance"].astype(str)
        + "-"
        + meta_df["enrolment_speakers_utterance"].astype(str)
    )
    tse_df = pd.read_csv(tse_mapping_csv, usecols=["utterance", "path"])
    merged = meta_df.merge(tse_df, on="utterance", how="left")

    for item in tqdm(
        merged.itertuples(index=False),
        total=len(merged),
        desc=f"{base_dir.name}/{dataset}",
        leave=False,
    ):
        if max_samples is not None and processed_so_far >= max_samples:
            break
        utt = str(item.utterance)
        wav_path = resolve_estimation_path(
            item.path, repo_root, base_dir, dataset
        )

        sig = bak = ovrl = p808 = np.nan
        status = "ok"
        error_message = ""

        if wav_path is None or not wav_path.exists():
            status = "missing_file"
            error_message = "TSE wav not found."
        elif scorer is not None:
            try:
                out = scorer(str(wav_path), SAMPLING_RATE)
                sig, bak, ovrl = out["SIG"], out["BAK"], out["OVRL"]
                p808 = out["P808"]
            except Exception as exc:  # pylint: disable=broad-except
                status = "error"
                error_message = str(exc)

        rows.append(
            {
                "base_dir": str(base_dir),
                "dataset": dataset,
                "utterance": utt,
                "path": str(wav_path) if wav_path is not None else "",
                "SIG": sig,
                "BAK": bak,
                "OVRL": ovrl,
                "P808": p808,
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


def _score_stats(
    df: pd.DataFrame, valid_mask: pd.Series, metrics: List[str]
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        s = pd.to_numeric(df.loc[valid_mask, m], errors="coerce").dropna()
        stats[m] = {
            "mean": float(s.mean()) if not s.empty else np.nan,
            "std": float(s.std()) if not s.empty else np.nan,
            "min": float(s.min()) if not s.empty else np.nan,
            "max": float(s.max()) if not s.empty else np.nan,
        }
    return stats


def summarize(
    df: pd.DataFrame, metrics: List[str]
) -> Tuple[dict, pd.DataFrame]:
    total = len(df)
    valid_mask = df["status"] == "ok"
    valid = int(valid_mask.sum())
    failed = total - valid
    missing_file = int((df["status"] == "missing_file").sum())
    error_count = int((df["status"] == "error").sum())

    overall_stats = _score_stats(df, valid_mask, metrics)
    overall = {
        "total": total,
        "valid": valid,
        "failed": failed,
        "missing_file": missing_file,
        "error": error_count,
        **{m: overall_stats[m] for m in metrics},
    }

    per_dataset_rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        g_valid = group["status"] == "ok"
        g_stats = _score_stats(group, g_valid, metrics)
        per_dataset_rows.append(
            {
                "dataset": dataset,
                "total": len(group),
                "valid": int(g_valid.sum()),
                "failed": len(group) - int(g_valid.sum()),
                "missing_file": int((group["status"] == "missing_file").sum()),
                "error": int((group["status"] == "error").sum()),
                **{m: g_stats[m] for m in metrics},
            }
        )
    per_dataset_df = pd.DataFrame(per_dataset_rows)

    return overall, per_dataset_df


def summarize_per_language(
    df: pd.DataFrame,
    dataset_lang_overrides: Dict[str, str],
    default_lang: str,
    metrics: List[str],
) -> pd.DataFrame:
    df = df.copy()
    df["_lang"] = df["dataset"].map(
        lambda d: dataset_lang_overrides.get(d, default_lang)
    )
    per_lang = []
    for lang in ["en", "chs"]:
        group = df[df["_lang"] == lang]
        if len(group) == 0:
            continue
        g_valid = group["status"] == "ok"
        g_stats = _score_stats(group, g_valid, metrics)
        per_lang.append(
            {
                "lang": lang,
                "total": len(group),
                "valid": int(g_valid.sum()),
                "failed": len(group) - int(g_valid.sum()),
                **{m: g_stats[m] for m in metrics},
            }
        )
    return pd.DataFrame(per_lang)


def build_ascii_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    lines = [sep, header_line, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
    lines.append(sep)
    return lines


def format_kv_block(items: List[Tuple[str, str]]) -> List[str]:
    if not items:
        return []
    max_key = max(len(k) for k, _ in items)
    return [f"  {k.ljust(max_key)} : {v}" for k, v in items]


def write_summary_txt(
    output_txt: Path,
    overall: dict,
    per_dataset_df: pd.DataFrame,
    per_lang_df: pd.DataFrame,
    status_counts: Dict[str, int],
    metrics: List[str],
) -> None:
    lines = []
    lines.append("DNSMOS Summary (SIG, BAK, OVRL, P808)")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ---- Mean-only tables (prominent, at top) ----
    if per_dataset_df is not None and not per_dataset_df.empty:
        lines.append("Mean by dataset")
        mean_headers_ds = ["dataset"] + [f"{m}_mean" for m in metrics]
        rows_mean = []
        for _, row in per_dataset_df.iterrows():
            r = [str(row["dataset"])]
            for m in metrics:
                r.append(format_float(row[m]["mean"]))
            rows_mean.append(r)
        lines.extend(build_ascii_table(mean_headers_ds, rows_mean))
        lines.append("")
    if per_lang_df is not None and not per_lang_df.empty:
        lines.append("Mean by language")
        mean_headers_lang = ["lang"] + [f"{m}_mean" for m in metrics]
        rows_mean_lang = []
        for _, row in per_lang_df.iterrows():
            r = [str(row["lang"])]
            for m in metrics:
                r.append(format_float(row[m]["mean"]))
            rows_mean_lang.append(r)
        lines.extend(build_ascii_table(mean_headers_lang, rows_mean_lang))
        lines.append("")

    # ---- Overall + full tables (unchanged, after mean tables) ----
    lines.append("Overall Statistics")
    lines.extend(
        format_kv_block(
            [
                ("Total samples", str(overall["total"])),
                ("Valid scored", str(overall["valid"])),
                ("Failed", str(overall["failed"])),
                ("Missing file", str(overall["missing_file"])),
                ("Error", str(overall["error"])),
            ]
        )
    )
    for m in metrics:
        lines.append(f"  {m}: mean={format_float(overall[m]['mean'])} std={format_float(overall[m]['std'])} min={format_float(overall[m]['min'])} max={format_float(overall[m]['max'])}")
    lines.append("  Note: score statistics are computed on rows with status=ok.")
    lines.append("")
    lines.append("Per-dataset Statistics")
    if per_dataset_df.empty:
        lines.append("  (no rows)")
    else:
        headers_flat = ["dataset", "total", "valid", "failed", "missing_file", "error"]
        for m in metrics:
            headers_flat.extend([f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max"])
        rows_flat = []
        for _, row in per_dataset_df.iterrows():
            r = [
                str(row["dataset"]),
                str(int(row["total"])),
                str(int(row["valid"])),
                str(int(row["failed"])),
                str(int(row["missing_file"])),
                str(int(row["error"])),
            ]
            for m in metrics:
                st = row[m]
                r.extend([
                    format_float(st["mean"]),
                    format_float(st["std"]),
                    format_float(st["min"]),
                    format_float(st["max"]),
                ])
            rows_flat.append(r)
        lines.extend(build_ascii_table(headers_flat, rows_flat))
    lines.append("")
    if per_lang_df is not None and not per_lang_df.empty:
        lines.append("Per-language Statistics")
        headers_flat = ["lang", "total", "valid", "failed"]
        for m in metrics:
            headers_flat.extend([f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max"])
        rows_flat = []
        for _, row in per_lang_df.iterrows():
            r = [
                str(row["lang"]),
                str(int(row["total"])),
                str(int(row["valid"])),
                str(int(row["failed"])),
            ]
            for m in metrics:
                st = row[m]
                r.extend([
                    format_float(st["mean"]),
                    format_float(st["std"]),
                    format_float(st["min"]),
                    format_float(st["max"]),
                ])
            rows_flat.append(r)
        lines.extend(build_ascii_table(headers_flat, rows_flat))
        lines.append("")
    lines.append("Status Distribution")
    if status_counts:
        ordered = ["ok", "missing_file", "error"]
        seen = set()
        rows = []
        for s in ordered:
            if s in status_counts:
                rows.append([s, str(status_counts[s])])
                seen.add(s)
        for s, c in sorted(status_counts.items()):
            if s not in seen:
                rows.append([s, str(c)])
        lines.extend(build_ascii_table(["status", "count"], rows))
    else:
        lines.append("  (no rows)")
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_base_dir(
    base_dir: Path,
    test_set_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
    dataset_lang_overrides: Dict[str, str],
    repo_root: Path,
    metrics: List[str],
) -> None:
    if not base_dir.exists():
        print(f"[Skip] Base dir does not exist: {base_dir}")
        return

    dataset_dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and (p / "tse_audio_mapping.csv").is_file()]
    )
    if not dataset_dirs:
        print(f"[Skip] No dataset directories under {base_dir}")
        return

    scorer = None
    if not args.regen_txt_only:
        _ensure_dnsmos_models(model_dir, allow_download=not args.no_download_models)
        primary_path = model_dir / "sig_bak_ovr.onnx"
        p808_path = model_dir / "model_v8.onnx"
        scorer = DNSMOSComputeScore(
            primary_path,
            p808_path,
            is_personalized_MOS=args.personalized_MOS,
            provider=args.provider,
        )

    all_rows: List[dict] = []
    processed = 0
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        rows, processed = build_dataset_rows(
            base_dir=base_dir,
            dataset=dataset,
            test_set_dir=test_set_dir,
            scorer=scorer,
            repo_root=repo_root,
            max_samples=args.max_samples,
            processed_so_far=processed,
        )
        all_rows.extend(rows)
        if args.max_samples is not None and processed >= args.max_samples:
            break

    result_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)

    default_csv_name, default_txt_name = default_output_names(base_dir.name)
    csv_name = args.output_csv_name or default_csv_name
    txt_name = args.output_txt_name or default_txt_name
    output_csv = base_dir / csv_name
    output_txt = base_dir / txt_name
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_csv, index=False)
    print(f"[Saved] CSV: {output_csv}")

    result_full = pd.DataFrame(all_rows)
    overall, per_dataset_df = summarize(result_full, metrics)
    status_counts = {
        str(k): int(v)
        for k, v in result_full["status"].value_counts(dropna=False).to_dict().items()
    }
    per_lang_df = summarize_per_language(
        result_full, dataset_lang_overrides, "en", metrics
    )

    if not args.csv_only:
        write_summary_txt(
            output_txt, overall, per_dataset_df, per_lang_df, status_counts, metrics
        )
        print(f"[Saved] TXT: {output_txt}")

    print(
        "[Overall] total={} valid={} failed={} missing_file={} error={} "
        "SIG_mean={} BAK_mean={} OVRL_mean={} P808_mean={}".format(
            overall["total"],
            overall["valid"],
            overall["failed"],
            overall["missing_file"],
            overall["error"],
            format_float(overall["SIG"]["mean"]),
            format_float(overall["BAK"]["mean"]),
            format_float(overall["OVRL"]["mean"]),
            format_float(overall["P808"]["mean"]),
        )
    )


def run_regen_txt_only(
    base_dir: Path,
    dataset_lang_overrides: Dict[str, str],
    args: argparse.Namespace,
    metrics: List[str],
) -> None:
    if not base_dir.exists():
        print(f"[Skip] Base dir does not exist: {base_dir}")
        return
    default_csv_name, default_txt_name = default_output_names(base_dir.name)
    csv_name = args.output_csv_name or default_csv_name
    txt_name = args.output_txt_name or default_txt_name
    output_csv = resolve_existing_csv_path(base_dir, csv_name, default_csv_name)
    output_txt = base_dir / txt_name
    if not output_csv.exists():
        print(f"[Skip] CSV not found: {output_csv}")
        return
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.read_csv(output_csv)
    overall, per_dataset_df = summarize(result_df, metrics)
    status_counts = {
        str(k): int(v) for k, v in result_df["status"].value_counts(dropna=False).to_dict().items()
    }
    per_lang_df = summarize_per_language(
        result_df, dataset_lang_overrides, "en", metrics
    )
    write_summary_txt(
        output_txt, overall, per_dataset_df, per_lang_df, status_counts, metrics
    )
    print(f"[Saved] TXT: {output_txt}")


def main() -> None:
    args = parse_args()
    metrics = ["SIG", "BAK", "OVRL", "P808"]

    model_dir = Path(
        args.dnsmos_model_dir or __import__("os").environ.get("DNSMOS_MODEL_DIR", "./DNSMOS")
    ).resolve()

    dataset_lang_overrides = parse_dataset_lang_overrides(args.dataset_lang_overrides)

    if args.regen_txt_only:
        for base_dir_raw in args.base_dir:
            run_regen_txt_only(
                base_dir=Path(base_dir_raw).resolve(),
                dataset_lang_overrides=dataset_lang_overrides,
                args=args,
                metrics=metrics,
            )
        return

    if args.test_set_dir is None:
        raise SystemExit("--test_set_dir is required unless --regen_txt_only.")

    test_set_dir = Path(args.test_set_dir).resolve()
    repo_root = Path.cwd().resolve()
    if not test_set_dir.exists():
        raise SystemExit(f"test_set_dir does not exist: {test_set_dir}")

    for base_dir_raw in args.base_dir:
        run_one_base_dir(
            base_dir=Path(base_dir_raw).resolve(),
            test_set_dir=test_set_dir,
            model_dir=model_dir,
            args=args,
            dataset_lang_overrides=dataset_lang_overrides,
            repo_root=repo_root,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
