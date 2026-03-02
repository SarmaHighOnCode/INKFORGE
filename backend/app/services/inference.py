"""
INKFORGE — Inference Service

Handles model loading, tokenization, and autoregressive stroke generation
using the LSTM+MDN model.
"""

import json
from pathlib import Path

import torch

from ..ml.model import HandwritingLSTM
from ..ml.utils import build_vocab, tokenize

# Precomputed style embeddings for MVP presets (z ∈ ℝ¹²⁸)
# These would be clustered from IAM writer embeddings in production
STYLE_PRESETS = {
    "neat_cursive": 0,
    "casual_print": 1,
    "rushed_notes": 2,
    "doctors_scrawl": 3,
    "elegant_formal": 4,
}


class InferenceService:
    """
    Service for running LSTM+MDN inference.

    Manages model lifecycle (load, warmup, generate) and
    converts text + style + params into stroke sequences.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        vocab_path: str | None = None,
    ) -> None:
        """
        Initialize the inference service.

        Args:
            checkpoint_path: Path to the trained model checkpoint (.pt).
            device: PyTorch device string ("cpu" or "cuda").
            vocab_path: Optional path to vocabulary JSON file.
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device)
        self.model: HandwritingLSTM | None = None
        self.is_loaded = False

        # Build vocabulary
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = build_vocab()

        self.vocab_size = len(self.vocab)

        # Style embeddings (initialized with random, would be learned/clustered)
        self.style_embeddings = torch.randn(len(STYLE_PRESETS), 128)

        # Normalization stats (should match training data)
        self.stroke_mean = torch.tensor([0.0, 0.0])
        self.stroke_std = torch.tensor([1.0, 1.0])

    def load_model(self) -> None:
        """
        Load the LSTM+MDN model from checkpoint.

        Loads either TorchScript-exported model or regular checkpoint.
        Sets model to eval mode and moves to target device.
        """
        if self.checkpoint_path is None:
            raise ValueError("No checkpoint path provided")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Try loading as TorchScript first
        if self.checkpoint_path.suffix == ".pts":
            self.model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        else:
            # Load regular checkpoint
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )

            # Initialize model
            model_config = checkpoint.get("model_config", {})
            self.model = HandwritingLSTM(
                vocab_size=model_config.get("vocab_size", self.vocab_size),
                char_embed_dim=model_config.get("char_embed_dim", 256),
                style_dim=model_config.get("style_dim", 128),
                hidden_dim=model_config.get("hidden_dim", 512),
                num_layers=model_config.get("num_layers", 3),
                dropout=0.0,  # No dropout during inference
                num_mixtures=model_config.get("num_mixtures", 20),
            )

            # Load state dict
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load normalization stats if available
            if "stroke_mean" in checkpoint:
                self.stroke_mean = checkpoint["stroke_mean"]
            if "stroke_std" in checkpoint:
                self.stroke_std = checkpoint["stroke_std"]

            # Load style embeddings if available
            if "style_embeddings" in checkpoint:
                self.style_embeddings = checkpoint["style_embeddings"]

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def get_style_embedding(self, style_id: str | int) -> torch.Tensor:
        """
        Get style embedding for a preset or custom style.

        Args:
            style_id: Style preset name or index.

        Returns:
            Style embedding tensor [128].
        """
        if isinstance(style_id, str):
            idx = STYLE_PRESETS.get(style_id, 0)
        else:
            idx = min(style_id, len(STYLE_PRESETS) - 1)

        return self.style_embeddings[idx].to(self.device)

    def generate(
        self,
        text: str,
        style_embedding: torch.Tensor | None = None,
        style_id: str | int = "neat_cursive",
        params: dict | None = None,
        temperature: float = 0.4,
        max_strokes: int = 2000,
        bias: float = 0.0,
    ) -> list[tuple[float, float, int, int, int]]:
        """
        Generate a stroke sequence for the given text.

        Args:
            text: Input text string.
            style_embedding: Style latent vector z ∈ ℝ¹²⁸. If None, uses style_id.
            style_id: Style preset name or index.
            params: Humanization parameter dict (not used in base implementation).
            temperature: Sampling temperature τ (controls randomness).
            max_strokes: Maximum number of strokes to generate per character.
            bias: Writing bias (higher = more consistent, lower = more varied).

        Returns:
            List of (Δx, Δy, p1, p2, p3) stroke tuples.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Normalize style embedding dimensions
        if style_embedding is None:
            style_embedding = self.get_style_embedding(style_id)

        if style_embedding.dim() == 1:
            style_z = style_embedding.unsqueeze(0).to(self.device)  # [1, 128]
        elif style_embedding.dim() == 2:
            if style_embedding.size(0) != 1:
                raise ValueError(
                    f"Expected style_embedding batch size 1, got {style_embedding.size(0)}"
                )
            style_z = style_embedding.to(self.device)
        else:
            raise ValueError(f"style_embedding must be 1D or 2D, got {style_embedding.dim()}D")

        # Tokenize text
        char_indices = self._tokenize(text)
        char_seq = torch.tensor(
            [char_indices], dtype=torch.long, device=self.device
        )  # [1, text_len]

        # Generate strokes autoregressively
        strokes = []
        hidden = self.model.get_initial_hidden(batch_size=1)

        # Initial stroke (zeros)
        prev_stroke = torch.zeros(1, 1, 5, device=self.device)

        with torch.no_grad():
            for char_idx in range(len(char_indices)):
                # Get current character
                current_char = char_seq[:, char_idx : char_idx + 1]  # [1, 1]

                # Generate strokes for this character
                char_strokes = 0
                while char_strokes < max_strokes:
                    # Forward pass
                    mdn_params, pen_logits, hidden = self.model(
                        current_char, prev_stroke, style_z, hidden
                    )

                    # Sample stroke
                    stroke = self.model.sample(
                        mdn_params[0, 0],  # [M*6]
                        pen_logits[0, 0],  # [3]
                        temperature=temperature,
                    )

                    dx, dy, p1, p2, p3 = stroke

                    # Denormalize
                    dx = dx * self.stroke_std[0].item() + self.stroke_mean[0].item()
                    dy = dy * self.stroke_std[1].item() + self.stroke_mean[1].item()

                    strokes.append((dx, dy, p1, p2, p3))

                    # Update previous stroke
                    prev_stroke = torch.tensor(
                        [[[dx, dy, p1, p2, p3]]],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    char_strokes += 1

                    # Check for pen-up (end of character)
                    if p2 == 1:
                        break

                    # Check for end-of-sequence
                    if p3 == 1:
                        return strokes

        return strokes

    def generate_word(
        self,
        word: str,
        style_embedding: torch.Tensor | None = None,
        style_id: str | int = "neat_cursive",
        temperature: float = 0.4,
    ) -> tuple[list[tuple[float, float, int, int, int]], None]:
        """
        Generate strokes for a single word.

        Note: LSTM state passing for cross-word consistency is not
        yet exposed. Hidden state is managed internally by generate().
        Use this method for independent single-word generation.

        Args:
            word: Input word string.
            style_embedding: Style latent vector.
            style_id: Style preset name.
            temperature: Sampling temperature.

        Returns:
            Tuple of (strokes, None). Hidden state is not exposed.
        """
        strokes = self.generate(
            text=word,
            style_embedding=style_embedding,
            style_id=style_id,
            temperature=temperature,
        )
        return strokes, None  # Hidden state management handled internally

    def _tokenize(self, text: str) -> list[int]:
        """
        Convert text string to character token indices.

        Args:
            text: Input text string.

        Returns:
            List of integer token indices.
        """
        return tokenize(text, self.vocab)

    def warmup(self) -> None:
        """
        Warmup the model with a dummy inference.

        This pre-compiles any lazy operations for faster subsequent inference.
        """
        if not self.is_loaded:
            return

        with torch.no_grad():
            dummy_text = "Hello"
            _ = self.generate(dummy_text, temperature=1.0, max_strokes=10)


class DocumentGenerator:
    """
    Document-level handwriting generation with layout management.

    Handles full-page composition with:
        - Auto line wrapping
        - Paragraph indentation
        - Writing fatigue simulation
        - Margin awareness
    """

    def __init__(
        self,
        inference_service: InferenceService,
        page_width: float = 210.0,  # A4 width in mm
        page_height: float = 297.0,  # A4 height in mm
        margin_left: float = 20.0,
        margin_right: float = 20.0,
        margin_top: float = 25.0,
        margin_bottom: float = 25.0,
        line_height: float = 8.0,
        char_width: float = 3.0,  # Average character width
    ) -> None:
        """
        Initialize document generator.

        Args:
            inference_service: InferenceService for stroke generation.
            page_width: Page width in mm.
            page_height: Page height in mm.
            margin_left: Left margin in mm.
            margin_right: Right margin in mm.
            margin_top: Top margin in mm.
            margin_bottom: Bottom margin in mm.
            line_height: Line height in mm.
            char_width: Average character width in mm.
        """
        self.inference = inference_service
        self.page_width = page_width
        self.page_height = page_height
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.line_height = line_height
        self.char_width = char_width

        self.content_width = page_width - margin_left - margin_right
        self.content_height = page_height - margin_top - margin_bottom

    def generate_document(
        self,
        text: str,
        style_id: str = "neat_cursive",
        temperature: float = 0.4,
        fatigue: float = 0.3,
        baseline_drift: float = 0.3,
    ) -> list[dict]:
        """
        Generate a full document with page layout.

        Args:
            text: Full document text.
            style_id: Style preset.
            temperature: Sampling temperature.
            fatigue: Fatigue simulation intensity (0-1).
            baseline_drift: Baseline drift intensity (0-1).

        Returns:
            List of stroke data dicts with position information.
        """
        import math

        # Split text into paragraphs
        paragraphs = text.split("\n\n")

        all_strokes = []
        cursor_x = self.margin_left
        cursor_y = self.margin_top
        line_number = 0

        for para_idx, paragraph in enumerate(paragraphs):
            # Add paragraph indent
            if para_idx > 0:
                cursor_y += self.line_height * 1.2  # Paragraph spacing
                cursor_x = self.margin_left

            cursor_x += 15.0  # Paragraph indent (1.5cm)

            # Split into words
            words = paragraph.split()

            for word_idx, word in enumerate(words):
                # Estimate word width
                word_width = len(word) * self.char_width

                # Check if word fits on current line
                if cursor_x + word_width > self.page_width - self.margin_right:
                    # Line wrap
                    cursor_x = self.margin_left + (torch.randn(1).item() * 2)  # Slight variation
                    cursor_y += self.line_height + (torch.randn(1).item() * 0.5 * baseline_drift)
                    line_number += 1

                # Apply fatigue: increase temperature slightly over document
                progress = (para_idx * 100 + word_idx) / max(1, len(text.split()) / 10)
                current_temp = temperature + (fatigue * 0.2 * progress)
                current_temp = min(current_temp, 1.0)

                # Generate strokes for word
                strokes = self.inference.generate(
                    word,
                    style_id=style_id,
                    temperature=current_temp,
                )

                # Apply baseline drift (sine wave)
                y_offset = baseline_drift * 2 * math.sin(line_number * 0.5 + para_idx)

                all_strokes.append(
                    {
                        "word": word,
                        "strokes": strokes,
                        "position": (cursor_x, cursor_y + y_offset),
                        "line_number": line_number,
                    }
                )

                # Advance cursor
                cursor_x += word_width + 4.0 + (torch.randn(1).item() * 1.5)  # Word spacing

                # Check for page break
                if cursor_y > self.page_height - self.margin_bottom:
                    # Stop all further generation to prevent overflow
                    break
            else:
                # Inner loop completed normally — continue to next paragraph
                continue
            # Inner loop broke (page full) — stop outer loop too
            break

        return all_strokes
