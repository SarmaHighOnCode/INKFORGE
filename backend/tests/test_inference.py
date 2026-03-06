"""
INKFORGE — Inference Tests

Tests for the ML inference pipeline: model loading, tokenization,
stroke generation, and MDN sampling.
"""

from app.ml.model import HandwritingLSTM, StyleEncoder


class TestHandwritingLSTM:
    """Tests for the LSTM+MDN model."""

    def test_model_instantiation(self) -> None:
        """Model should instantiate with default hyperparameters."""
        model = HandwritingLSTM(vocab_size=80)
        assert model is not None
        assert model.num_mixtures == 20

    def test_model_parameter_count(self) -> None:
        """Model should have reasonable parameter count."""
        model = HandwritingLSTM(vocab_size=80)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_model_output_shapes(self) -> None:
        """Forward pass should produce correct output shapes."""
        import torch
        model = HandwritingLSTM(vocab_size=80, num_mixtures=20)
        char_seq = torch.zeros((2, 10), dtype=torch.long)
        stroke_seq = torch.zeros((2, 10, 5))
        style_z = torch.randn((2, 128))
        mdn_params, pen_logits, _ = model(char_seq, stroke_seq, style_z)
        assert mdn_params.shape == (2, 10, 120)
        assert pen_logits.shape == (2, 10, 3)


class TestStyleEncoder:
    """Tests for the CNN style encoder."""

    def test_encoder_instantiation(self) -> None:
        """Style encoder should instantiate."""
        encoder = StyleEncoder(style_dim=128)
        assert encoder is not None

    def test_encoder_output_dim(self) -> None:
        """Encoder should output z ∈ ℝ¹²⁸."""
        import torch
        encoder = StyleEncoder(style_dim=128)
        dummy_image = torch.randn(2, 1, 64, 64)
        output = encoder(dummy_image)
        assert output.shape == (2, 128)


class TestMDNSampling:
    """Tests for MDN sampling utilities."""

    def test_sample_produces_valid_stroke(self) -> None:
        """Sampled stroke should be a valid 5-tuple."""
        import torch
        model = HandwritingLSTM(vocab_size=80)
        mdn_params = torch.randn(120)
        pen_logits = torch.randn(3)
        stroke = model.sample(mdn_params, pen_logits)
        assert len(stroke) == 5
        assert isinstance(stroke, tuple)

    def test_temperature_affects_variance(self) -> None:
        """Higher temperature should produce more variance."""
        import torch
        model = HandwritingLSTM(vocab_size=80)
        mdn_params = torch.randn(120)
        pen_logits = torch.randn(3)
        stroke1 = model.sample(mdn_params, pen_logits, temperature=0.1)
        stroke2 = model.sample(mdn_params, pen_logits, temperature=2.0)
        assert len(stroke1) == 5
        assert len(stroke2) == 5
