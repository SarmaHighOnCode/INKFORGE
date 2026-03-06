"""
INKFORGE — Pretrained Checkpoint Download Script

Downloads a pretrained LSTM+MDN checkpoint for inference.

Usage:
    python scripts/download_checkpoint.py --model lstm_mdn_v1
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained model checkpoint.")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm_mdn_v1",
        help="Model identifier to download.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/",
        help="Output directory for checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Inkforge — Checkpoint Download")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    import requests
    from tqdm import tqdm

    # Try downloading from HuggingFace
    base_url = "https://huggingface.co/SarmaHighOnCode/INKFORGE/resolve/main/checkpoints"
    filename = f"{args.model}.pt"
    url = f"{base_url}/{filename}"
    save_path = output_dir / filename

    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Checkpoint saved to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        print("Creating dummy checkpoint as fallback for local testing...")
        import torch
        checkpoint = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "epoch": 0,
            "loss": 0.0,
            "model_config": {},
            "vocab": {},
            "stroke_mean": torch.tensor([0.0, 0.0]),
            "stroke_std": torch.tensor([1.0, 1.0]),
        }
        torch.save(checkpoint, save_path)
        print(f"Dummy checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
