"""Canonical dataset -> language map shared by eval scripts.

Stored and reported codes are ``en`` / ``zh`` only. Vendor-specific ids
(WeSpeaker ``chs``) are produced by :func:`to_wespeaker_lang` at the API
call, never written to CSVs or summary tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


CANONICAL_LANGS = ("en", "zh")

DATASET_LANGUAGE: Dict[str, str] = {
    "CHiME6": "en",
    "AISHELL-4": "zh",
    "AliMeeting": "zh",
    "AMI": "en",
    "DipCo": "en",
    # EVAL2 unseen splits (language-tagged virtual datasets)
    "unseen_CN": "zh",
    "unseen_EN": "en",
}

_ZH_ALIASES = frozenset({"zh", "zho", "chs", "cn", "chinese", "mandarin"})
_EN_ALIASES = frozenset({"en", "eng", "english"})


def get_language(dataset_name: str) -> str:
    """Return ``"zh"`` or ``"en"`` for a known dataset.

    Raises
    ------
    KeyError
        If ``dataset_name`` is not in :data:`DATASET_LANGUAGE`.
    """
    try:
        return DATASET_LANGUAGE[dataset_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset_name {dataset_name!r}; expected one of "
            f"{sorted(DATASET_LANGUAGE)}"
        ) from exc


def chinese_datasets() -> list:
    return [name for name, lang in DATASET_LANGUAGE.items() if lang == "zh"]


def english_datasets() -> list:
    return [name for name, lang in DATASET_LANGUAGE.items() if lang == "en"]


def asr_model_for(dataset: str, zh_model: str, en_model: str) -> str:
    """Pick the Chinese or English ASR backend name for ``dataset``."""
    return zh_model if get_language(dataset) == "zh" else en_model


def normalize_language(value: object) -> str:
    """Map aliases to canonical ``en`` / ``zh``. Empty / NA -> ``""``.

    Unknown non-empty tokens are returned lowercased as-is so a stray
    label still surfaces in reports instead of being silently dropped.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "<na>"}:
        return ""
    if text in _ZH_ALIASES:
        return "zh"
    if text in _EN_ALIASES:
        return "en"
    return text


def to_wespeaker_lang(lang: str) -> str:
    """Map a canonical (or aliased) language to WeSpeaker's ``en`` / ``chs``."""
    canonical = normalize_language(lang)
    if canonical == "zh":
        return "chs"
    if canonical == "en":
        return "en"
    raise ValueError(
        f"Cannot map language {lang!r} to a WeSpeaker lang; expected en/zh"
    )


def language_for_dataset(
    dataset_name: str,
    overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Canonical language for ``dataset_name``, honoring optional overrides.

    Override values are passed through :func:`normalize_language` so ``chs``
    is accepted as an alias of ``zh``.
    """
    if overrides and dataset_name in overrides:
        canonical = normalize_language(overrides[dataset_name])
        if canonical not in CANONICAL_LANGS:
            raise ValueError(
                f"Invalid language {overrides[dataset_name]!r} for dataset "
                f"{dataset_name!r}; expected en/zh"
            )
        return canonical
    return get_language(dataset_name)


def parse_dataset_lang_overrides(raw: Optional[str]) -> Dict[str, str]:
    """Parse ``dataset:lang,...`` into canonical ``en`` / ``zh`` values.

    ``chs`` and other aliases are accepted and normalized. Empty input
    yields an empty dict (callers then use :func:`get_language`).
    """
    overrides: Dict[str, str] = {}
    if raw is None:
        return overrides
    text = str(raw).strip()
    if not text:
        return overrides

    for item in text.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid dataset_lang_overrides item {part!r}. "
                "Expected format: dataset:lang"
            )
        dataset, lang = part.split(":", 1)
        dataset = dataset.strip()
        canonical = normalize_language(lang)
        if not dataset:
            raise ValueError("dataset_lang_overrides contains empty dataset name.")
        if canonical not in CANONICAL_LANGS:
            raise ValueError(
                f"Invalid language {lang!r} in dataset_lang_overrides. "
                "Supported: en, zh (chs is accepted as an alias of zh)"
            )
        overrides[dataset] = canonical
    return overrides


def scan_output_dirs(
    output_dirs: List[str],
    overrides: Optional[Dict[str, str]] = None,
) -> int:
    """Print language-mapping status for dataset dirs under each output dir.

    A dataset dir is a subdirectory containing ``tse_audio_mapping.csv``
    (the same discovery rule used by the metric scripts). Returns the
    number of dataset dirs without a language mapping.
    """
    unknown = 0
    for raw in output_dirs:
        output_dir = Path(raw)
        if not output_dir.is_dir():
            print(f"[scan] Output dir not found: {output_dir}")
            continue
        dataset_dirs = sorted(
            p
            for p in output_dir.iterdir()
            if p.is_dir() and (p / "tse_audio_mapping.csv").is_file()
        )
        if not dataset_dirs:
            print(f"[scan] No dataset dirs with tse_audio_mapping.csv under {output_dir}")
            continue
        print(f"[scan] {output_dir}")
        for dataset_dir in dataset_dirs:
            name = dataset_dir.name
            try:
                lang = language_for_dataset(name, overrides)
            except KeyError:
                unknown += 1
                print(f"[scan]   UNKNOWN  {name} (no language mapping; will be skipped)")
            else:
                print(f"[scan]   ok       {name} -> {lang}")
    return unknown


def _main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Dataset language helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_model = sub.add_parser("asr-model", help="Print zh/en ASR model for a dataset")
    p_model.add_argument("dataset")
    p_model.add_argument("zh_model")
    p_model.add_argument("en_model")

    p_lang = sub.add_parser("lang", help="Print canonical en/zh for a dataset")
    p_lang.add_argument("dataset")

    p_scan = sub.add_parser(
        "scan", help="Scan output dirs for datasets lacking a language mapping"
    )
    p_scan.add_argument("output_dirs", nargs="+")
    p_scan.add_argument(
        "--overrides",
        action="append",
        default=None,
        help="dataset:lang,... overrides; may be repeated",
    )
    p_scan.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any dataset dir lacks a language mapping",
    )

    args = parser.parse_args()
    try:
        if args.cmd == "asr-model":
            print(asr_model_for(args.dataset, args.zh_model, args.en_model))
        elif args.cmd == "lang":
            print(get_language(args.dataset))
        elif args.cmd == "scan":
            merged: Dict[str, str] = {}
            for raw in args.overrides or []:
                merged.update(parse_dataset_lang_overrides(raw))
            unknown = scan_output_dirs(args.output_dirs, merged)
            if unknown:
                print(
                    f"[scan] {unknown} dataset dir(s) have no language mapping and "
                    "will be skipped by SPK_SIM / DNSMOS / TSE_TIMING. Add them to "
                    "DATASET_LANGUAGE in utils/dataset_lang.py or pass overrides.",
                    file=sys.stderr,
                )
                if args.strict:
                    sys.exit(1)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
