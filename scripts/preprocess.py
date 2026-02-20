"""
INKFORGE — Data Preprocessing Script

Preprocesses raw IAM On-Line Handwriting data into training-ready format.

Usage:
    python scripts/preprocess.py --input data/iam/ --output data/processed/

Pipeline:
    1. Parse IAM stroke XML files
    2. Convert to (Δx, Δy, p1, p2, p3) tuples
    3. Normalize stroke coordinates
    4. Writer-level train/val/test split (80/10/10)
    5. Save as .npz files
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IAM dataset.")
    parser.add_argument("--input", type=str, default="data/iam/", help="Raw IAM data dir.")
    parser.add_argument("--output", type=str, default="data/processed/", help="Output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Inkforge — Data Preprocessing")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # TODO: Implement preprocessing
    raise NotImplementedError("Preprocessing not yet implemented")


if __name__ == "__main__":
    main()
