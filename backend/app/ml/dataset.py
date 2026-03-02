"""
INKFORGE — IAM Dataset Loader

Handles loading, preprocessing, and batching of the IAM On-Line
Handwriting Database for training the LSTM+MDN model.

Dataset: IAM On-Line Handwriting Database
    - 13,049 transcribed handwritten texts
    - 221 unique writers
    - Split: 80/10/10 train/val/test (writer-level)

Stroke Format: (Δx, Δy, p1, p2, p3) per timestep
"""

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import build_vocab, tokenize


class IAMStrokeDataset(Dataset):
    """
    PyTorch Dataset for IAM On-Line handwriting strokes.

    Loads stroke sequences and their corresponding text transcriptions.
    Applies data augmentation during training.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = 700,
        max_text_len: int = 100,
        augment: bool = True,
        vocab: dict[str, int] | None = None,
        stroke_mean: torch.Tensor | None = None,
        stroke_std: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the IAM dataset.

        Args:
            data_dir: Path to preprocessed IAM data.
            split: Dataset split — "train", "val", or "test".
            max_seq_len: Maximum stroke sequence length.
            max_text_len: Maximum text length.
            augment: Whether to apply data augmentation (train only).
            vocab: Character vocabulary. If None, builds default vocab.
            stroke_mean: Pre-computed stroke mean (for val/test splits).
            stroke_std: Pre-computed stroke std (for val/test splits).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.augment = augment and (split == "train")
        self.vocab = vocab if vocab is not None else build_vocab()

        # Load preprocessed data
        self.samples = self._load_split(split)

        # Compute normalization stats from training data
        if split == "train" and len(self.samples) > 0:
            self.stroke_mean, self.stroke_std = self._compute_stats()
        elif stroke_mean is not None and stroke_std is not None:
            # Use training stats provided by caller for val/test
            self.stroke_mean = stroke_mean
            self.stroke_std = stroke_std
        else:
            # Fallback defaults when no stats available
            self.stroke_mean = torch.tensor([0.0, 0.0])
            self.stroke_std = torch.tensor([1.0, 1.0])

    def _load_split(self, split: str) -> list[dict]:
        """
        Load preprocessed data for the given split.

        Args:
            split: "train", "val", or "test".

        Returns:
            List of sample dicts with "strokes", "text", "writer_id".
        """
        split_file = self.data_dir / f"{split}.json"

        if not split_file.exists():
            # Try loading from stroke files directly
            return self._load_from_stroke_files(split)

        with open(split_file) as f:
            samples = json.load(f)

        return samples

    def _load_from_stroke_files(self, split: str) -> list[dict]:
        """
        Load data directly from stroke files if preprocessed JSON not available.

        Args:
            split: Dataset split.

        Returns:
            List of sample dicts.
        """
        samples = []
        split_dir = self.data_dir / split

        if not split_dir.exists():
            # Return empty list if no data available
            return []

        for stroke_file in split_dir.glob("*.npz"):
            try:
                data = np.load(stroke_file, allow_pickle=False)

                # Check membership safely using 'in'
                writer_id = int(data["writer_id"]) if "writer_id" in data else 0

                # Carefully extract and decode text to avoid b'...' strings
                text_val = data["text"]
                if hasattr(text_val, "item"):
                    text_val = text_val.item()
                if isinstance(text_val, bytes):
                    text_str = text_val.decode("utf-8")
                else:
                    text_str = str(text_val)

                samples.append(
                    {
                        "strokes": data["strokes"].tolist(),
                        "text": text_str,
                        "writer_id": writer_id,
                    }
                )
            except Exception:
                # Skip files that cannot be loaded without pickle or have other errors
                continue

        for stroke_file in split_dir.glob("*.pt"):
            data = torch.load(stroke_file, weights_only=True)
            samples.append(data)

        return samples

    def _compute_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std of stroke deltas for normalization.

        Returns:
            Tuple of (mean, std) tensors.
        """
        all_deltas = []

        for sample in self.samples[:1000]:  # Use subset for efficiency
            strokes = sample["strokes"]
            if len(strokes) > 0:
                deltas = torch.tensor([[s[0], s[1]] for s in strokes])
                all_deltas.append(deltas)

        if len(all_deltas) == 0:
            return torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])

        all_deltas = torch.cat(all_deltas, dim=0)
        mean = all_deltas.mean(dim=0)
        std = all_deltas.std(dim=0).clamp(min=1e-6)

        return mean, std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with keys:
                - "strokes": [seq_len, 5] stroke tensor
                - "text_indices": [text_len] character indices
                - "stroke_lengths": actual stroke sequence length
                - "text_lengths": actual text length
                - "pen_states": [seq_len] pen state indices (0, 1, or 2)
        """
        sample = self.samples[idx]

        # Parse strokes
        strokes = sample["strokes"]
        if isinstance(strokes, list):
            strokes = torch.tensor(strokes, dtype=torch.float32)
        else:
            strokes = strokes.clone()

        # Normalize stroke deltas
        strokes[:, :2] = (strokes[:, :2] - self.stroke_mean) / self.stroke_std

        # Apply augmentation
        if self.augment:
            strokes = self._augment(strokes)

        # Truncate to max length
        stroke_len = min(len(strokes), self.max_seq_len)
        strokes = strokes[:stroke_len]

        # Pad to max_seq_len
        if len(strokes) < self.max_seq_len:
            pad_size = self.max_seq_len - len(strokes)
            padding = torch.zeros(pad_size, 5)
            padding[:, 4] = 1  # Set p3 (end) for padding
            strokes = torch.cat([strokes, padding], dim=0)

        # Extract pen states as class indices
        # p1=pen_down -> 0, p2=pen_up -> 1, p3=end -> 2
        pen_states = strokes[:, 2:5].argmax(dim=-1)

        # Tokenize text
        text = sample.get("text", "")
        text_indices = tokenize(text, self.vocab)
        text_len = min(len(text_indices), self.max_text_len)
        text_indices = text_indices[:text_len]

        # Pad text
        if len(text_indices) < self.max_text_len:
            text_indices += [0] * (self.max_text_len - len(text_indices))

        text_indices = torch.tensor(text_indices, dtype=torch.long)

        return {
            "strokes": strokes,
            "text_indices": text_indices,
            "stroke_lengths": torch.tensor(stroke_len, dtype=torch.long),
            "text_lengths": torch.tensor(text_len, dtype=torch.long),
            "pen_states": pen_states,
        }

    def _augment(self, strokes: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to stroke sequence.

        Augmentation pipeline (from PRD):
            - Random affine transforms (scale, rotation)
            - Per-stroke velocity jitter
            - Synthetic baseline drift

        Args:
            strokes: Normalized stroke tensor [seq_len, 5].

        Returns:
            Augmented stroke tensor [seq_len, 5].
        """
        strokes = strokes.clone()
        seq_len = len(strokes)

        # 1. Random scale (0.9 - 1.1)
        if random.random() < 0.5:
            scale = 0.9 + random.random() * 0.2
            strokes[:, :2] *= scale

        # 2. Random rotation (-5 to +5 degrees)
        if random.random() < 0.5:
            angle = (random.random() - 0.5) * 10 * (3.14159 / 180)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            strokes[:, :2] = strokes[:, :2] @ rotation.T

        # 3. Velocity jitter (add small noise to deltas)
        if random.random() < 0.5:
            noise = torch.randn(seq_len, 2) * 0.05
            strokes[:, :2] += noise

        # 4. Baseline drift (sinusoidal offset on y)
        if random.random() < 0.5:
            t = torch.linspace(0, 2 * np.pi, seq_len)
            amplitude = random.random() * 0.1
            phase = random.random() * 2 * np.pi
            drift = amplitude * torch.sin(t + phase)
            strokes[:, 1] += drift

        return strokes


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of sample dicts from __getitem__.

    Returns:
        Batched dict with stacked tensors.
    """
    return {
        "strokes": torch.stack([b["strokes"] for b in batch]),
        "text_indices": torch.stack([b["text_indices"] for b in batch]),
        "stroke_lengths": torch.stack([b["stroke_lengths"] for b in batch]),
        "text_lengths": torch.stack([b["text_lengths"] for b in batch]),
        "pen_states": torch.stack([b["pen_states"] for b in batch]),
    }


def parse_iam_xml(xml_path: Path) -> list[tuple[float, float, int, int, int]]:
    """
    Parse IAM On-Line XML stroke file.

    The IAM format stores strokes as sequences of (x, y, time) points
    with pen-up indicators. This function converts absolute coordinates
    to relative deltas and one-hot pen states.

    Args:
        xml_path: Path to XML file.

    Returns:
        List of (dx, dy, p1, p2, p3) tuples.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    abs_points = []

    for stroke_set in root.findall(".//StrokeSet"):
        for stroke in stroke_set.findall("Stroke"):
            stroke_points = []
            for point in stroke.findall("Point"):
                x = float(point.get("x", 0))
                y = float(point.get("y", 0))
                stroke_points.append((x, y, 1))  # pen_down = 1

            if stroke_points:
                # Mark last point of each stroke as pen-up
                abs_points.extend(stroke_points)
                abs_points[-1] = (abs_points[-1][0], abs_points[-1][1], 0)

    if not abs_points:
        return []

    # Convert absolute (x, y, pen_down) to relative (dx, dy, p1, p2, p3)
    strokes = []
    prev_x, prev_y = 0.0, 0.0

    for i, (x, y, pen_down) in enumerate(abs_points):
        dx = x - prev_x
        dy = y - prev_y
        is_last = i == len(abs_points) - 1

        if is_last:
            p1, p2, p3 = 0, 0, 1  # end-of-sequence
        elif pen_down == 1:
            p1, p2, p3 = 1, 0, 0  # pen down
        else:
            p1, p2, p3 = 0, 1, 0  # pen up

        strokes.append((dx, dy, p1, p2, p3))
        prev_x, prev_y = x, y

    return strokes
