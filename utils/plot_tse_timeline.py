
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # key by utterance
            if 'utterance' in item:
                data[item['utterance']] = item
            elif 'key' in item: # vad output usually uses 'key'
                data[item['key']] = item
    return data

def load_metrics(csv_path):
    import pandas as pd
    metrics = {}
    try:
        df = pd.read_csv(csv_path)
        # Check required columns
        req_cols = ['utterance', 'precision', 'recall', 'f1']
        if not all(col in df.columns for col in req_cols):
            print(f"Warning: Metrics CSV missing one of {req_cols}")
            return metrics
            
        for _, row in df.iterrows():
            utt = row['utterance']
            metrics[utt] = {
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1']
            }
    except Exception as e:
        print(f"Error loading metrics CSV: {e}")
    return metrics

def plot_timeline(
    utterance_id, 
    label_info, 
    vad_info, 
    output_path,
    metrics=None
):
    """
    label_info: dict from label_segments.jsonl
      {
        "utterance": "...",
        "mix_duration": float,
        "target_speaker": "...",
        "segments_by_speaker": { "spk1": [[s,e],...], ... },
        "speaker_genders": { "spk1": "M", ... }
      }
    vad_info: dict from vad_segments.jsonl (optional)
      {
        "key": "...",
        "value": [[s,e], ...]
      }
    """
    
    mix_duration = label_info.get('mix_duration', 10.0)
    target_speaker = label_info.get('target_speaker', 'Unknown')
    segments_by_speaker = label_info.get('segments_by_speaker', {})
    
    # Sort speakers: maybe target speaker first? or just sorted.
    # Let's put target speaker at the bottom of the GT block (closest to TSE result)
    # or just sort alphabetically.
    speakers = sorted(segments_by_speaker.keys())
    
    # Create figure
    # Height depends on number of speakers + 1 (TSE result)
    num_rows = len(speakers) + 1
    fig_height = max(4, num_rows * 0.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Y-positions
    # Speakers at y=1, 2, ..., N
    # TSE Result at y=0
    
    # Colors
    color_map = {
        'Target': '#1f77b4', # Blue
        'Other': 'gray',
        'TSE': '#1f77b4' # Blue (same as target)
    }
    
    # Plot GT speakers
    y_ticks = []
    y_labels = []
    
    for i, spk in enumerate(speakers):
        y_pos = i + 1
        y_ticks.append(y_pos)
        y_labels.append(f"{spk}")
        
        if spk == target_speaker:
             color = color_map['Target']
        else:
             color = color_map['Other']
        
        # Plot segments
        segs = segments_by_speaker.get(spk, [])
        for start, end in segs:
            width = end - start
            if width <= 0: continue
            rect = patches.Rectangle(
                (start, y_pos - 0.4), 
                width, 
                0.8, 
                linewidth=1, 
                edgecolor='black', 
                facecolor=color,
                alpha=0.9
            )
            ax.add_patch(rect)
            
    # Plot TSE Result
    y_ticks.insert(0, 0)
    y_labels.insert(0, f"TSE Result\n(Target: {target_speaker})")
    
    if vad_info:
        # vad_segments.jsonl from vad_inference_firered.py uses "pred_segments"
        # but some other formats might use "value" or "segments"
        if "pred_segments" in vad_info:
             vad_segments = vad_info["pred_segments"]
        elif "value" in vad_info:
             vad_segments = vad_info["value"]
        else:
             vad_segments = []

        for start, end in vad_segments:
            width = end - start
            if width <= 0: continue
            rect = patches.Rectangle(
                (start, 0 - 0.4), 
                width, 
                0.8, 
                linewidth=1, 
                edgecolor='black', 
                facecolor=color_map['TSE'],
                alpha=0.9
            )
            ax.add_patch(rect)
            
    # Configure axes
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, mix_duration)
    ax.set_ylim(-0.8, len(speakers) + 0.8)
    ax.set_xlabel("Time (s)")
    
    # Set x-ticks to integers (5, 10, ...)
    # Determine step size based on duration
    if mix_duration < 10:
        step = 1
    elif mix_duration < 60:
        step = 5
    else:
        step = 10
        
    x_ticks = np.arange(0, int(mix_duration) + step, step)
    ax.set_xticks(x_ticks)
    
    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Add legend
    handles = [
        patches.Patch(color=color_map['Target'], label='Target Speaker'),
        patches.Patch(color=color_map['Other'], label='Other Speaker'),
        patches.Patch(color=color_map['TSE'], label='TSE Prediction')
    ]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1.1), ncol=3)
    
    title_text = f"Utterance: {utterance_id}"
    if metrics:
        p = metrics.get('precision', 0.0)
        r = metrics.get('recall', 0.0)
        f = metrics.get('f1', 0.0)
        # Add metrics text below the plot or in the title
        # Let's add it below the x-axis
        plt.figtext(0.5, 0.05, 
                   f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}", 
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Adjust layout to make room for text
        plt.subplots_adjust(bottom=0.25)

    plt.title(title_text, pad=20)
    # plt.tight_layout() # tight_layout might conflict with subplots_adjust if not careful
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' helps include outside text
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_jsonl", required=True)
    parser.add_argument("--vad_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics_csv", default=None, help="Path to TSE_TIMING.csv for metrics")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_map = {}
    if args.metrics_csv:
        print(f"Loading metrics from {args.metrics_csv}...")
        metrics_map = load_metrics(args.metrics_csv)
    
    print(f"Loading labels from {args.label_jsonl}...")
    labels = load_jsonl(args.label_jsonl)
    
    print(f"Loading VAD segments from {args.vad_jsonl}...")
    vads = load_jsonl(args.vad_jsonl)
    
    print(f"Generating plots to {output_dir}...")
    for utt_id, label_item in tqdm(labels.items()):
        vad_item = vads.get(utt_id)
        metric = metrics_map.get(utt_id)
        out_path = output_dir / f"{utt_id}.png"
        plot_timeline(utt_id, label_item, vad_item, out_path, metrics=metric)
        
    print(f"Done. Generated {len(labels)} figures.")

if __name__ == "__main__":
    main()
