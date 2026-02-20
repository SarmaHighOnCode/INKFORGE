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

    print(f"Inkforge — Checkpoint Download")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    # TODO: Implement
    # 1. Resolve model URL from model identifier
    # 2. Download checkpoint file
    # 3. Verify checksum
    # 4. Save to output directory
    raise NotImplementedError("Checkpoint download not yet implemented")


if __name__ == "__main__":
    main()
