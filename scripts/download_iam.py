"""
INKFORGE — IAM Dataset Download Script

Downloads and extracts the IAM On-Line Handwriting Database.

Usage:
    python scripts/download_iam.py --output data/iam/

Dataset Info:
    - 13,049 transcribed handwritten texts
    - 221 unique writers
    - Source: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IAM On-Line dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/iam/",
        help="Output directory for downloaded data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inkforge — IAM Dataset Download")
    print(f"Output: {output_dir}")

    # TODO: Implement download
    # 1. Check for existing download
    # 2. Request credentials (IAM requires registration)
    # 3. Download stroke XML files
    # 4. Download ASCII transcriptions
    # 5. Verify checksums
    # 6. Extract to output directory
    raise NotImplementedError("IAM download not yet implemented")


if __name__ == "__main__":
    main()
