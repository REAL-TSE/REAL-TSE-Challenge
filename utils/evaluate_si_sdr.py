import os
import argparse
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from score import cal_SISNRi
from print_stats import print_SISDR, compute_statistics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def resample_to_16k(waveform, orig_sr, target_sr=16000):
    if orig_sr != target_sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform

def main(args):
    # Load TSE prediction mapping
    tse_mapping = pd.read_csv(args.tse_mapping_path)
    tse_mapping_dict = dict(zip(tse_mapping['utterance'], tse_mapping['path']))

    # Load evaluation metadata
    eval_df = pd.read_csv(args.eval_metadata_path)

    # Prepare save dataframe, initialize with +infinity
    save_df = eval_df.copy()
    save_df['SI-SDR']  = float('inf')
    save_df['SI-SDRi'] = float('inf')

    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
        mix_utt = row['mixture_utterance']
        enrol_utt = row['enrolment_speakers_utterance']
        speaker = row['speaker']

        # Construct file paths
        ref_name  = f"clean_{speaker}_in_{mix_utt}.wav"
        ref_path  = os.path.abspath(os.path.join(args.clean_audio_dir, ref_name))
        ests_name = f"{mix_utt}-{enrol_utt}"
        ests_path = tse_mapping_dict[ests_name]
        mix_path  = os.path.abspath(os.path.join(args.mixture_audio_dir, f"{mix_utt}.wav"))

        # Load and resample to 16kHz
        ref,  ref_sr  = torchaudio.load(ref_path)
        ests, ests_sr = torchaudio.load(ests_path)
        mix,  mix_sr  = torchaudio.load(mix_path)

        ref  = resample_to_16k(ref,  ref_sr)
        ests = resample_to_16k(ests, ests_sr)
        mix  = resample_to_16k(mix,  mix_sr)

        ref = ref.numpy()
        ests = ests.numpy()
        mix = mix.numpy()

        # Align the length if necessary
        min_length = min(ref.shape[1], ests.shape[1], mix.shape[1])
        ref   = ref[:, :min_length]
        ests  = ests[:, :min_length]
        mix   = mix[:, :min_length]

        # Calculate SI-SDR and SI-SDRi
        sisdr, sisdr_i = cal_SISNRi(ests[0], ref[0], mix[0])

        save_df.at[index, 'SI-SDR']  = sisdr
        save_df.at[index, 'SI-SDRi'] = sisdr_i

    # Print and save results
    compute_statistics(save_df)
    print_SISDR(save_df)
    save_df.to_csv(args.save_path, index=False)
    print(f"Saved evaluation results to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Target Speaker Extraction results.")
    parser.add_argument("--eval_metadata_path", type=str, required=True,
                        help="Path to the evaluation metadata CSV (absolute path).")
    parser.add_argument("--tse_mapping_path", type=str, required=True,
                        help="Path to TSE result mapping CSV (absolute path).")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save evaluation result CSV.")
    parser.add_argument("--clean_audio_dir", type=str, required=True,
                        help="Directory containing clean reference audio.")
    parser.add_argument("--mixture_audio_dir", type=str, required=True,
                        help="Directory containing mixture audio.")

    args = parser.parse_args()
    main(args)
