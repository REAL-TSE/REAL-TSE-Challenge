import argparse
from collections import OrderedDict
from pathlib import Path
import shutil

import pandas as pd


def format_path(dataset_root: Path, audio_path: Path, mode: str) -> str:
    resolved = audio_path.resolve()
    if mode == "absolute":
        return str(resolved)
    return str(resolved.relative_to(dataset_root.resolve()))


def build_audio_inventory(dataset_root: Path, mode: str) -> OrderedDict[str, str]:
    inventory: OrderedDict[str, str] = OrderedDict()
    for subdir in ("mixtures", "enrolment_speakers"):
        for audio_path in sorted((dataset_root / subdir).glob("*.wav")):
            if audio_path.stem in inventory:
                raise SystemExit(
                    f"Duplicate utterance name found while scanning audio files: {audio_path.stem}"
                )
            inventory[audio_path.stem] = format_path(dataset_root, audio_path, mode)
    return inventory


def write_mapping_csv(path: Path, mapping: OrderedDict[str, str]) -> None:
    df = pd.DataFrame(
        [{"utterance": utterance, "path": audio_path} for utterance, audio_path in mapping.items()]
    )
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate machine-local REAL-T mapping.csv."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the REAL-T dataset directory.",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help="How to write paths into generated CSV files.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    for required in ("mixtures", "enrolment_speakers"):
        if not (dataset_root / required).is_dir():
            raise SystemExit(f"Missing required directory: {dataset_root / required}")

    inventory = build_audio_inventory(dataset_root, args.mapping_mode)
    if not inventory:
        raise SystemExit(f"No wav files found under {dataset_root}")

    mapping_csv = dataset_root / "mapping.csv"
    write_mapping_csv(mapping_csv, inventory)
    print(f"Wrote {mapping_csv}")
    mapping_dir = dataset_root / "mapping"
    if mapping_dir.exists():
        shutil.rmtree(mapping_dir)
        print(f"Removed legacy mapping directory: {mapping_dir}")


if __name__ == "__main__":
    main()
