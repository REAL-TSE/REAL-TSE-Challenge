import argparse
import os
from huggingface_hub import snapshot_download

def download_model(repo_id, save_dir):
    save_path = os.path.abspath(save_dir)
    save_path = os.path.join(save_path, repo_id.split("/")[-1])
    os.makedirs(save_path, exist_ok=True)

    snapshot_download(repo_id=repo_id, local_dir=save_path)
    print(f"Model from '{repo_id}' downloaded to '{save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ASR models from Hugging Face Hub")

    parser.add_argument("--zh_repo_id", type=str, default="FireRedTeam/FireRedASR-AED-L",
                        help="Hugging Face repo ID for Chinese ASR model")
    parser.add_argument("--zh_save_dir", type=str, required=True,
                        help="Directory to save the Chinese ASR model")

    parser.add_argument("--en_repo_id", type=str, default="openai/whisper-large-v2",
                        help="Hugging Face repo ID for English ASR model")
    parser.add_argument("--en_save_dir", type=str, required=True,
                        help="Directory to save the English ASR model")

    args = parser.parse_args()

    download_model(args.en_repo_id, args.en_save_dir)

    download_model(args.zh_repo_id, args.zh_save_dir)

