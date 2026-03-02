"""
INKFORGE — ML Utilities

Helper functions for stroke processing, MDN computations,
and visualization utilities.
"""

import math

import torch
import torch.nn.functional as functional


def compute_mdn_loss(
    mdn_params: torch.Tensor,
    pen_logits: torch.Tensor,
    target_strokes: torch.Tensor,
    target_pen: torch.Tensor,
    num_mixtures: int = 20,
) -> torch.Tensor:
    """
    Compute the MDN loss (negative log-likelihood) for stroke prediction.

    Args:
        mdn_params: Predicted MDN parameters [batch, seq, M*6].
        pen_logits: Predicted pen state logits [batch, seq, 3].
        target_strokes: Ground truth (Δx, Δy) [batch, seq, 2].
        target_pen: Ground truth pen states [batch, seq] (class indices).
        num_mixtures: Number of Gaussian mixture components (M=20).

    Returns:
        Scalar loss tensor.
    """
    batch_size, seq_len, _ = mdn_params.shape
    num_m = num_mixtures  # Renamed num_m to M for consistency with patch

    # Reshape MDN params: [batch, seq, M*6] -> [batch, seq, M, 6]
    params = mdn_params.view(batch_size, seq_len, num_m, 6)

    # Extract parameters
    pi_logits = params[:, :, :, 0]  # [batch, seq, M]
    mu_x = params[:, :, :, 1]  # [batch, seq, M]
    mu_y = params[:, :, :, 2]  # [batch, seq, M]
    sigma_x_raw = params[:, :, :, 3]  # [batch, seq, M]
    sigma_y_raw = params[:, :, :, 4]  # [batch, seq, M]
    rho_raw = params[:, :, :, 5]  # [batch, seq, M]

    # Apply activations to ensure valid parameter ranges
    sigma_x = torch.exp(sigma_x_raw)
    sigma_y = torch.exp(sigma_y_raw)
    rho = torch.tanh(rho_raw)

    # Get mixture log-probabilities
    log_pi = functional.log_softmax(pi_logits, dim=-1)  # [batch, seq, M]

    # Target strokes: [batch, seq, 2] -> expand for M mixtures
    target_x = target_strokes[:, :, 0:1].expand(-1, -1, num_m)  # [batch, seq, M]
    target_y = target_strokes[:, :, 1:2].expand(-1, -1, num_m)  # [batch, seq, M]

    # Compute bivariate Gaussian log-likelihood
    # log N(x, y | μ, Σ) = -log(2π) - log(σx) - log(σy) - 0.5*log(1-ρ²) - Z/(2*(1-ρ²))
    # where Z = ((x-μx)/σx)² + ((y-μy)/σy)² - 2ρ((x-μx)/σx)((y-μy)/σy)

    dx = (target_x - mu_x) / sigma_x
    dy = (target_y - mu_y) / sigma_y
    rho_sq = rho**2

    # Avoid division by zero
    one_minus_rho_sq = (1 - rho_sq).clamp(min=1e-6)

    z_val = dx**2 + dy**2 - 2 * rho * dx * dy  # Corrected Z variable name from patch
    log_gaussian = (
        -math.log(2 * math.pi)
        - torch.log(sigma_x)  # Changed from log_sigma_x
        - torch.log(sigma_y)  # Changed from log_sigma_y
        - 0.5 * torch.log(one_minus_rho_sq)
        - z_val / (2 * one_minus_rho_sq)  # Changed from Z to z_val
    )  # [batch, seq, M]

    # Weighted sum using log-sum-exp: log(Σ π_k * N_k) = logsumexp(log π_k + log N_k)
    log_likelihood = torch.logsumexp(log_pi + log_gaussian, dim=-1)  # [batch, seq]

    # Stroke loss: negative log-likelihood
    stroke_loss = -log_likelihood.mean()

    # Pen state loss: cross-entropy (target must be long/int64)
    pen_logits_flat = pen_logits.view(-1, 3)  # [batch*seq, 3]
    target_pen_flat = target_pen.long().view(-1)  # [batch*seq]
    pen_loss = functional.cross_entropy(
        pen_logits_flat, target_pen_flat
    )  # Corrected syntax from patch

    # Total loss
    total_loss = stroke_loss + pen_loss

    return total_loss


def normalize_strokes(strokes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize stroke deltas to zero mean, unit variance.

    Args:
        strokes: Raw stroke tensor [seq_len, 5] or [batch, seq_len, 5].

    Returns:
        Tuple of (normalized_strokes, mean, std).
    """
    # Only normalize the position deltas (Δx, Δy), not pen states
    if strokes.dim() == 2:
        # Single sequence: [seq_len, 5]
        deltas = strokes[:, :2]
        mean = deltas.mean(dim=0, keepdim=True)
        std = deltas.std(dim=0, keepdim=True).clamp(min=1e-6)
        normalized_deltas = (deltas - mean) / std
        normalized = torch.cat([normalized_deltas, strokes[:, 2:]], dim=-1)
    else:
        # Batch: [batch, seq_len, 5]
        deltas = strokes[:, :, :2]
        mean = deltas.mean(dim=(0, 1), keepdim=True)
        std = deltas.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        normalized_deltas = (deltas - mean) / std
        normalized = torch.cat([normalized_deltas, strokes[:, :, 2:]], dim=-1)

    return normalized, mean.squeeze(), std.squeeze()


def denormalize_strokes(
    strokes: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Denormalize stroke deltas.

    Args:
        strokes: Normalized stroke tensor.
        mean: Original mean values.
        std: Original std values.

    Returns:
        Denormalized stroke tensor.
    """
    if strokes.dim() == 2:
        deltas = strokes[:, :2]
        denorm_deltas = deltas * std + mean
        return torch.cat([denorm_deltas, strokes[:, 2:]], dim=-1)
    else:
        deltas = strokes[:, :, :2]
        denorm_deltas = deltas * std + mean
        return torch.cat([denorm_deltas, strokes[:, :, 2:]], dim=-1)


def strokes_to_absolute(strokes: list[tuple[float, ...]]) -> list[tuple[float, float, int]]:
    """
    Convert relative stroke deltas to absolute coordinates.

    Args:
        strokes: List of (Δx, Δy, p1, p2, p3) tuples.

    Returns:
        List of (x, y, pen_down) where pen_down is 1 if drawing.
    """
    absolute = []
    x, y = 0.0, 0.0

    for stroke in strokes:
        dx, dy, p1, p2, p3 = stroke
        x += dx
        y += dy
        pen_down = int(p1)  # p1 = pen is down (drawing)
        absolute.append((x, y, pen_down))

        # Check for end-of-sequence
        if p3 == 1:
            break

    return absolute


def absolute_to_strokes(
    points: list[tuple[float, float, int]],
) -> list[tuple[float, float, int, int, int]]:
    """
    Convert absolute coordinates to relative stroke deltas.

    Args:
        points: List of (x, y, pen_down) tuples.

    Returns:
        List of (Δx, Δy, p1, p2, p3) tuples.
    """
    strokes = []
    prev_x, prev_y = 0.0, 0.0

    for i, (x, y, pen_down) in enumerate(points):
        dx = x - prev_x
        dy = y - prev_y
        p1 = 1 if pen_down else 0
        p2 = 0 if pen_down else 1
        p3 = 1 if i == len(points) - 1 else 0

        strokes.append((dx, dy, p1, p2, p3))
        prev_x, prev_y = x, y

    return strokes


def build_vocab(chars: str | None = None) -> dict[str, int]:
    """
    Build character vocabulary for tokenization.

    Args:
        chars: Optional string of characters. Defaults to printable ASCII.

    Returns:
        Dict mapping characters to indices.
    """
    if chars is None:
        # Printable ASCII (32-126) plus special tokens
        chars = "".join(chr(i) for i in range(32, 127))

    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for _i, char in enumerate(chars):
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """
    Convert text to token indices.

    Args:
        text: Input text string.
        vocab: Character vocabulary dict.

    Returns:
        List of token indices.
    """
    unk_idx = vocab.get("<unk>", 3)
    return [vocab.get(char, unk_idx) for char in text]


def detokenize(indices: list[int], vocab: dict[str, int]) -> str:
    """
    Convert token indices back to text.

    Args:
        indices: List of token indices.
        vocab: Character vocabulary dict.

    Returns:
        Decoded text string.
    """
    idx_to_char = {v: k for k, v in vocab.items()}
    chars = [idx_to_char.get(idx, "") for idx in indices]
    return "".join(chars)
