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
    5. Save as .npz/.pt files
"""

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IAM dataset.")
    parser.add_argument("--input", type=str, default="data/iam/", help="Raw IAM data dir.")
    parser.add_argument("--output", type=str, default="data/processed/", help="Output dir.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--max-seq-len", type=int, default=700, help="Maximum stroke sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    return parser.parse_args()


def parse_stroke_xml(xml_path: Path) -> list[tuple[float, float, int]] | None:
    """
    Parse IAM On-Line stroke XML file.

    The IAM format has structure:
        <WhiteboardCapture>
            <StrokeSet>
                <Stroke>
                    <Point x="..." y="..." time="..." />
                    ...
                </Stroke>
                ...
            </StrokeSet>
        </WhiteboardCapture>

    Args:
        xml_path: Path to XML stroke file.

    Returns:
        List of (x, y, pen_down) tuples, or None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    points = []

    # Try different XML structures (IAM has variations)
    stroke_sets = root.findall(".//StrokeSet")
    if not stroke_sets:
        stroke_sets = root.findall(".//Whiteboard")
    if not stroke_sets:
        # Try root level strokes
        stroke_sets = [root]

    for stroke_set in stroke_sets:
        for stroke in stroke_set.findall(".//Stroke"):
            stroke_points = []

            for point in stroke.findall("Point"):
                x = float(point.get("x", 0))
                y = float(point.get("y", 0))
                stroke_points.append((x, y, 1))  # pen_down = 1

            if stroke_points:
                points.extend(stroke_points)
                # Mark last point of stroke as pen-up
                if points:
                    x, y, _ = points[-1]
                    points[-1] = (x, y, 0)

    return points if points else None


def absolute_to_relative(points: list[tuple[float, float, int]]) -> list[tuple[float, float, int, int, int]]:
    """
    Convert absolute coordinates to relative stroke deltas.

    Args:
        points: List of (x, y, pen_down) absolute coordinates.

    Returns:
        List of (Δx, Δy, p1, p2, p3) stroke tuples.
    """
    strokes = []
    prev_x, prev_y = 0.0, 0.0

    for i, (x, y, pen_down) in enumerate(points):
        dx = x - prev_x
        dy = y - prev_y

        # Pen states: p1=pen_down, p2=pen_up, p3=end_of_sequence
        p1 = 1 if pen_down else 0
        p2 = 0 if pen_down else 1
        p3 = 1 if i == len(points) - 1 else 0

        strokes.append((dx, dy, p1, p2, p3))
        prev_x, prev_y = x, y

    return strokes


def normalize_strokes(strokes: list[tuple]) -> tuple[list[tuple], float, float]:
    """
    Normalize stroke deltas to zero mean and unit variance.

    Args:
        strokes: List of stroke tuples.

    Returns:
        Tuple of (normalized_strokes, mean, std).
    """
    if not strokes:
        return strokes, 0.0, 1.0

    deltas = np.array([[s[0], s[1]] for s in strokes])
    mean = deltas.mean()
    std = deltas.std()

    if std < 1e-6:
        std = 1.0

    normalized = []
    for dx, dy, p1, p2, p3 in strokes:
        norm_dx = (dx - mean) / std
        norm_dy = (dy - mean) / std
        normalized.append((norm_dx, norm_dy, p1, p2, p3))

    return normalized, float(mean), float(std)


def load_transcription(ascii_path: Path, line_id: str) -> str | None:
    """
    Load text transcription for a line from ASCII files.

    Args:
        ascii_path: Path to ascii directory.
        line_id: Line identifier (e.g., "a01-000u-00").

    Returns:
        Transcription text or None if not found.
    """
    # Parse line_id to find the correct file
    # Format: writer-form-line (e.g., a01-000u-00)
    parts = line_id.split("-")
    if len(parts) < 2:
        return None

    writer_form = f"{parts[0]}-{parts[1]}"

    # Look for transcription file
    trans_file = ascii_path / f"{writer_form}.txt"
    if not trans_file.exists():
        # Try subdirectory structure
        trans_file = ascii_path / parts[0] / f"{writer_form}.txt"

    if not trans_file.exists():
        return None

    try:
        with open(trans_file, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Find the specific line in the transcription
        # IAM format varies; try to extract text
        for line in content.split("\n"):
            if line_id in line:
                # Extract text after line ID
                match = re.search(r'"([^"]+)"', line)
                if match:
                    return match.group(1)
                # Try space-separated format
                parts = line.split()
                if len(parts) > 1:
                    return " ".join(parts[1:])

        return None

    except Exception:
        return None


def get_writer_id(line_id: str) -> str:
    """Extract writer ID from line identifier."""
    # Format: writer-form-line (e.g., a01-000u-00 -> a01)
    parts = line_id.split("-")
    return parts[0] if parts else "unknown"


def split_by_writer(
    samples: dict[str, list],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split samples by writer to prevent data leakage.

    Args:
        samples: Dict mapping writer_id to list of samples.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        seed: Random seed.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    random.seed(seed)

    writers = list(samples.keys())
    random.shuffle(writers)

    n_writers = len(writers)
    n_train = int(n_writers * train_ratio)
    n_val = int(n_writers * val_ratio)

    train_writers = writers[:n_train]
    val_writers = writers[n_train:n_train + n_val]
    test_writers = writers[n_train + n_val:]

    train_samples = []
    val_samples = []
    test_samples = []

    for writer in train_writers:
        train_samples.extend(samples[writer])
    for writer in val_writers:
        val_samples.extend(samples[writer])
    for writer in test_writers:
        test_samples.extend(samples[writer])

    return train_samples, val_samples, test_samples


def save_samples(samples: list[dict], output_dir: Path, split_name: str) -> None:
    """
    Save preprocessed samples to disk.

    Args:
        samples: List of sample dicts.
        output_dir: Output directory.
        split_name: Split name ("train", "val", "test").
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save individual samples
    for i, sample in enumerate(tqdm(samples, desc=f"Saving {split_name}")):
        sample_path = split_dir / f"sample_{i:06d}.pt"
        torch.save(sample, sample_path)

    # Also save as JSON index
    index_path = output_dir / f"{split_name}.json"

    # Convert numpy scalars to native Python types for JSON serialization
    def _to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = []
    for sample in samples:
        serializable.append({
            k: _to_native(v) if not isinstance(v, (list, dict, str)) else v
            for k, v in sample.items()
        })

    with open(index_path, "w") as f:
        json.dump(serializable, f)

    print(f"Saved {len(samples)} {split_name} samples to {split_dir}")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INKFORGE — Data Preprocessing")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find stroke XML files
    stroke_dir = input_dir / "lineStrokes"
    if not stroke_dir.exists():
        stroke_dir = input_dir / "lineStrokes-all"
    if not stroke_dir.exists():
        # Try flat structure
        stroke_files = list(input_dir.glob("**/*.xml"))
    else:
        stroke_files = list(stroke_dir.glob("**/*.xml"))

    if not stroke_files:
        print("ERROR: No stroke XML files found!")
        print(f"Expected stroke files in: {input_dir}/lineStrokes/")
        print("\nPlease download the IAM dataset first:")
        print("  python scripts/download_iam.py --output data/iam/")
        return

    print(f"Found {len(stroke_files)} stroke files")

    # Find ASCII transcription directory
    ascii_dir = input_dir / "ascii"
    if not ascii_dir.exists():
        ascii_dir = input_dir / "ascii-all"

    # Process samples grouped by writer
    samples_by_writer: dict[str, list] = defaultdict(list)
    stats = {"total": 0, "success": 0, "skipped": 0, "too_long": 0, "too_short": 0}

    print("\nProcessing stroke files...")
    for xml_path in tqdm(stroke_files):
        stats["total"] += 1

        # Parse stroke XML
        points = parse_stroke_xml(xml_path)
        if points is None or len(points) < 10:
            stats["skipped"] += 1
            continue

        # Convert to relative strokes
        strokes = absolute_to_relative(points)

        # Check sequence length
        if len(strokes) > args.max_seq_len:
            stats["too_long"] += 1
            # Truncate instead of skipping
            strokes = strokes[:args.max_seq_len]

        if len(strokes) < 10:
            stats["too_short"] += 1
            continue

        # Get line ID and writer
        line_id = xml_path.stem
        writer_id = get_writer_id(line_id)

        # Load transcription
        text = None
        if ascii_dir.exists():
            text = load_transcription(ascii_dir, line_id)

        if text is None:
            # Use placeholder text
            text = f"[line_{line_id}]"

        # Create sample
        sample = {
            "strokes": strokes,
            "text": text,
            "writer_id": writer_id,
            "line_id": line_id,
        }

        samples_by_writer[writer_id].append(sample)
        stats["success"] += 1

    print(f"\nProcessing stats:")
    print(f"  Total files:    {stats['total']}")
    print(f"  Successful:     {stats['success']}")
    print(f"  Skipped:        {stats['skipped']}")
    print(f"  Too long:       {stats['too_long']}")
    print(f"  Too short:      {stats['too_short']}")
    print(f"  Unique writers: {len(samples_by_writer)}")

    if stats["success"] == 0:
        print("\nERROR: No valid samples extracted!")
        return

    # Split by writer
    print(f"\nSplitting data (train={args.train_ratio}, val={args.val_ratio})...")
    train_samples, val_samples, test_samples = split_by_writer(
        samples_by_writer,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    # Save samples
    print("\nSaving preprocessed data...")
    save_samples(train_samples, output_dir, "train")
    save_samples(val_samples, output_dir, "val")
    save_samples(test_samples, output_dir, "test")

    # Compute and save normalization statistics
    print("\nComputing normalization statistics...")
    all_deltas = []
    for sample in train_samples[:1000]:  # Use subset for efficiency
        deltas = [(s[0], s[1]) for s in sample["strokes"]]
        all_deltas.extend(deltas)

    deltas_array = np.array(all_deltas)
    mean = float(deltas_array.mean())
    std = float(deltas_array.std())

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "mean": mean,
            "std": std,
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "num_test": len(test_samples),
            "num_writers": len(samples_by_writer),
        }, f, indent=2)

    print(f"\nNormalization stats: mean={mean:.4f}, std={std:.4f}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output saved to: {output_dir}")
    print("\nNext step: Start training")
    print("  python train.py --config configs/lstm_mdn_base.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
