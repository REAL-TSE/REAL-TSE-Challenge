"""
Prepare label_segments.jsonl for each dataset under a predicted_dir.
Reads meta + overlap_records.json once, writes one jsonl per dataset with
GT segments (relative to mixture, clipped + collar) per utterance.
Phase-2 eval then only needs vad_segments.jsonl + label_segments.jsonl.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow importing from same package when run as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tse_timing_evaluation import (
    apply_collar_to_segments,
    build_meta_mapping,
    clip_intervals,
    find_overlap_record,
    parse_mixture_utterance,
    safe_float,
)


def main():
    parser = argparse.ArgumentParser(
        description="Write label_segments.jsonl per dataset (GT segments per utterance, relative time, collar applied)."
    )
    parser.add_argument("--ground_truth_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, default=None, help="Path to metadata dir for gender info (optional)")
    parser.add_argument("--gt_json_base_dir", type=str, required=True)
    parser.add_argument("--predicted_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, default=None, help="Space-separated dataset names")
    parser.add_argument("--vad_dir_name", type=str, default="FireRedVAD")
    parser.add_argument("--collar", type=float, default=0.05)
    parser.add_argument("--match_tolerance", type=float, default=0.02)
    args = parser.parse_args()

    predicted_dir = Path(args.predicted_dir).resolve()
    ground_truth_dir = Path(args.ground_truth_dir).resolve()
    gt_json_base_dir = Path(args.gt_json_base_dir).resolve()
    vad_dir_name = args.vad_dir_name

    metadata_dir = Path(args.metadata_dir).resolve() if args.metadata_dir else None

    if args.datasets:
        dataset_list = [s.strip() for s in args.datasets.split() if s.strip()]
    else:
        dataset_list = [p.name for p in predicted_dir.iterdir() if p.is_dir()]

    # Build global speaker -> gender map
    # REMOVED: Gender logic is no longer needed.
    # global_spk_gender = {}
    
    # if metadata_dir and metadata_dir.exists():
    #     ...
    
    for dataset_name in dataset_list:
        dataset_dir = predicted_dir / dataset_name
        mapping_path = dataset_dir / "tse_audio_mapping.csv"
        meta_path = ground_truth_dir / f"{dataset_name}_meta.csv"
        overlap_json_path = gt_json_base_dir / dataset_name / "overlap_records.json"
        out_dir = dataset_dir / vad_dir_name
        out_path = out_dir / "label_segments.jsonl"

        if not mapping_path.exists():
            print(f"[SKIP] no mapping: {mapping_path}")
            continue
        if not meta_path.exists():
            print(f"[SKIP] no meta: {meta_path}")
            continue
        if not overlap_json_path.exists():
            print(f"[SKIP] no overlap json: {overlap_json_path}")
            continue

        mapping_df = pd.read_csv(mapping_path)
        meta_df = pd.read_csv(meta_path)
        meta_map = build_meta_mapping(meta_df)
        
        # REMOVED: Gender augmentation logic
        
        with open(overlap_json_path, "r", encoding="utf-8") as f:
            overlap_data = json.load(f)

        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in mapping_df.iterrows():
                utterance = row["utterance"]
                if utterance not in meta_map:
                    continue
                meta_row = meta_map[utterance]
                mixture_utterance = meta_row["mixture_utterance"]
                speaker = meta_row["speaker"]
                try:
                    session_id, mix_start, mix_end = parse_mixture_utterance(mixture_utterance)
                except ValueError:
                    continue
                if session_id not in overlap_data:
                    continue
                mix_duration = max(0.0, mix_end - mix_start)
                records_for_session = overlap_data[session_id]
                matched = find_overlap_record(
                    records_for_session, mix_start, mix_end, args.match_tolerance
                )
                if matched is None:
                    continue
                overlap_segments = matched.get("overlap_segments", [])
                # Group by all speakers: speaker -> list of (abs_start, abs_end)
                by_speaker_abs: Dict[str, List[Tuple[float, float]]] = {}
                for seg in overlap_segments:
                    spk = seg.get("speaker")
                    if not spk:
                        continue
                    s = safe_float(seg.get("start_time"))
                    e = safe_float(seg.get("end_time"))
                    if np.isnan(s) or np.isnan(e):
                        continue
                    if spk not in by_speaker_abs:
                        by_speaker_abs[spk] = []
                    by_speaker_abs[spk].append((s, e))

                # Per speaker: to relative, clip, collar, round
                mix_duration_rounded = round(mix_duration, 6)
                segments_by_speaker: Dict[str, List[List[float]]] = {}
                for spk, abs_list in by_speaker_abs.items():
                    gt_rel = [(s - mix_start, e - mix_start) for s, e in abs_list]
                    gt_rel = clip_intervals(gt_rel, 0.0, mix_duration)
                    with_collar = apply_collar_to_segments(gt_rel, args.collar, mix_duration)
                    segments_by_speaker[spk] = [[round(a, 6), round(b, 6)] for a, b in with_collar]

                # Lookup genders using pre-built map
                # REMOVED: speaker_genders = {}
                # for spk in segments_by_speaker.keys():
                #     speaker_genders[spk] = global_spk_gender.get(str(spk), "Unknown")

                obj = {
                    "utterance": utterance,
                    "mix_duration": mix_duration_rounded,
                    "target_speaker": speaker,
                    "segments_by_speaker": segments_by_speaker,
                    # "speaker_genders": speaker_genders,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
        print(f"[OK] {dataset_name}: wrote {written} rows to {out_path}")


if __name__ == "__main__":
    main()
