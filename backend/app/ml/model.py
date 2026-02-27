"""
INKFORGE — LSTM + Mixture Density Network Model

Core generative model for handwriting synthesis.
Architecture based on Graves (2013) — "Generating Sequences with
Recurrent Neural Networks" (arXiv:1308.0850).

Model Architecture:
    Input Embedding:  char one-hot → d=256
    Style Conditioning: concat latent z ∈ ℝ¹²⁸
    Encoder:          3-layer LSTM, hidden=512, dropout=0.2
    Output Head:      MDN with M=20 Gaussian mixtures
    Pen State:        Bernoulli (sigmoid × 3)

Stroke Representation:
    (Δx, Δy, p1, p2, p3)
    Δx, Δy = relative pen displacements
    p1 = pen-down (drawing)
    p2 = pen-up (moving)
    p3 = end-of-sequence
"""

import torch
import torch.nn as nn


class HandwritingLSTM(nn.Module):
    """
    LSTM + Mixture Density Network for handwriting stroke generation.

    Generates sequences of pen strokes conditioned on character input
    and style embedding. Output is a mixture of bivariate Gaussians
    for (Δx, Δy) prediction plus Bernoulli pen state.
    """

    # Model hyperparameters (from PRD Section 4.2.2)
    CHAR_EMBED_DIM = 256
    STYLE_DIM = 128
    HIDDEN_DIM = 512
    NUM_LAYERS = 3
    DROPOUT = 0.2
    NUM_MIXTURES = 20  # M=20 Gaussian components

    def __init__(
        self,
        vocab_size: int,
        char_embed_dim: int = CHAR_EMBED_DIM,
        style_dim: int = STYLE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_mixtures: int = NUM_MIXTURES,
    ) -> None:
        """
        Initialize the LSTM+MDN model.

        Args:
            vocab_size: Number of unique characters in vocabulary.
            char_embed_dim: Character embedding dimension (d=256).
            style_dim: Style latent vector dimension (z ∈ ℝ¹²⁸).
            hidden_dim: LSTM hidden state dimension (512).
            num_layers: Number of LSTM layers (3).
            dropout: Dropout probability (0.2).
            num_mixtures: Number of MDN Gaussian components (M=20).
        """
        super().__init__()

        self.num_mixtures = num_mixtures

        # Character embedding: one-hot → d=256
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim)

        # Input projection: char_embed + style_dim + stroke_dim → hidden_dim
        # stroke_dim = 5 (Δx, Δy, p1, p2, p3) from previous timestep
        input_dim = char_embed_dim + style_dim + 5
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 3-layer LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # MDN output head
        # Per mixture: π (weight), μx, μy, σx, σy, ρ (correlation)
        # Total per mixture: 6 parameters
        # Plus 3 pen state logits
        mdn_output_dim = num_mixtures * 6  # π, μx, μy, σx, σy, ρ
        pen_state_dim = 3                  # p1, p2, p3

        self.mdn_head = nn.Linear(hidden_dim, mdn_output_dim)
        self.pen_head = nn.Linear(hidden_dim, pen_state_dim)

    def forward(
        self,
        char_seq: torch.Tensor,
        stroke_seq: torch.Tensor,
        style_z: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            char_seq: Character indices [batch, seq_len].
            stroke_seq: Previous stroke tuples [batch, seq_len, 5].
            style_z: Style embedding [batch, style_dim].
            hidden: Optional LSTM hidden state tuple (h, c).

        Returns:
            Tuple of (mdn_params, pen_logits, hidden_state).
            - mdn_params: [batch, seq_len, M*6] — mixture parameters.
            - pen_logits: [batch, seq_len, 3] — pen state logits.
            - hidden_state: Updated LSTM hidden state.
        """
        batch_size, seq_len = char_seq.shape

        # 1. Embed characters: [batch, seq_len] -> [batch, seq_len, char_embed_dim]
        char_embed = self.char_embedding(char_seq)

        # 2. Expand style_z to match sequence length: [batch, style_dim] -> [batch, seq_len, style_dim]
        style_expanded = style_z.unsqueeze(1).expand(-1, seq_len, -1)

        # 3. Concatenate: char_embed + style + previous stroke
        # [batch, seq_len, char_embed_dim + style_dim + 5]
        combined = torch.cat([char_embed, style_expanded, stroke_seq], dim=-1)

        # 4. Project to hidden dimension
        projected = self.input_projection(combined)

        # 5. Pass through LSTM
        lstm_out, hidden = self.lstm(projected, hidden)

        # 6. Compute MDN parameters and pen state logits
        mdn_params = self.mdn_head(lstm_out)
        pen_logits = self.pen_head(lstm_out)

        return mdn_params, pen_logits, hidden

    def sample(
        self,
        mdn_params: torch.Tensor,
        pen_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[float, float, int, int, int]:
        """
        Sample a single stroke tuple from the MDN output.

        Args:
            mdn_params: MDN parameters for current timestep [M*6].
            pen_logits: Pen state logits [3].
            temperature: Sampling temperature τ.

        Returns:
            Single stroke tuple (Δx, Δy, p1, p2, p3).
        """
        num_m = self.num_mixtures

        # 1. Extract MDN parameters: π, μx, μy, σx, σy, ρ
        # Each component has 6 params, total = M*6
        params = mdn_params.view(num_m, 6)

        pi_logits = params[:, 0]          # Mixture weights (logits)
        mu_x = params[:, 1]               # Mean x
        mu_y = params[:, 2]               # Mean y
        sigma_x = torch.exp(params[:, 3]) # Std x (exp to ensure positive)
        sigma_y = torch.exp(params[:, 4]) # Std y
        rho = torch.tanh(params[:, 5])    # Correlation (tanh to bound [-1, 1])

        # 2. Apply temperature
        # Scale mixture logits by 1/τ, scale sigmas by √τ
        pi = torch.softmax(pi_logits / temperature, dim=0)
        sigma_x = sigma_x * (temperature ** 0.5)
        sigma_y = sigma_y * (temperature ** 0.5)

        # 3. Sample mixture component from categorical distribution
        mixture_idx = torch.multinomial(pi, 1).item()

        # 4. Sample (Δx, Δy) from bivariate Gaussian
        mu_x_k = mu_x[mixture_idx].item()
        mu_y_k = mu_y[mixture_idx].item()
        sigma_x_k = sigma_x[mixture_idx].item()
        sigma_y_k = sigma_y[mixture_idx].item()
        rho_k = rho[mixture_idx].item()

        # Bivariate Gaussian sampling using conditional method
        # x ~ N(μx, σx²)
        # y | x ~ N(μy + ρ*σy/σx*(x - μx), σy²*(1 - ρ²))
        z1 = torch.randn(1).item()
        z2 = torch.randn(1).item()

        dx = mu_x_k + sigma_x_k * z1
        dy = mu_y_k + sigma_y_k * (rho_k * z1 + (1 - rho_k ** 2) ** 0.5 * z2)

        # 5. Sample pen state from Bernoulli
        pen_probs = torch.softmax(pen_logits / temperature, dim=0)
        pen_state = torch.multinomial(pen_probs, 1).item()

        # Convert to one-hot: p1=pen_down, p2=pen_up, p3=end
        p1 = 1 if pen_state == 0 else 0
        p2 = 1 if pen_state == 1 else 0
        p3 = 1 if pen_state == 2 else 0

        return (dx, dy, p1, p2, p3)

    def get_initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get zero-initialized hidden state for LSTM.

        Args:
            batch_size: Batch size.

        Returns:
            Tuple of (h0, c0) tensors.
        """
        device = next(self.parameters()).device
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        return (h0, c0)


class StyleEncoder(nn.Module):
    """
    CNN-based style encoder.

    Processes a handwriting sample image and outputs a fixed-length
    style embedding z ∈ ℝ¹²⁸ for conditioning the LSTM+MDN generator.

    Architecture:
        - 4 convolutional blocks with batch norm and max pooling
        - Global average pooling
        - Fully connected layer to style_dim
    """

    STYLE_DIM = 128

    def __init__(self, style_dim: int = STYLE_DIM, input_channels: int = 1) -> None:
        """
        Initialize the style encoder CNN.

        Args:
            style_dim: Output embedding dimension (128).
            input_channels: Number of input channels (1 for grayscale).
        """
        super().__init__()

        self.style_dim = style_dim

        # Convolutional blocks: progressively extract features
        self.conv_blocks = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        # Fully connected layer to style embedding
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, style_dim),
            nn.Tanh(),  # Bound output to [-1, 1]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode a handwriting sample image to style embedding.

        Args:
            image: Input image tensor [batch, 1, H, W].
                   Expected input: grayscale image, any size (will be pooled).

        Returns:
            Style embedding z [batch, style_dim].
        """
        features = self.conv_blocks(image)
        embedding = self.fc(features)
        return embedding

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> "StyleEncoder":
        """
        Load a pretrained style encoder.

        Args:
            checkpoint_path: Path to checkpoint file.
            device: Target device.

        Returns:
            Loaded StyleEncoder model.
        """
        model = cls()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["style_encoder_state_dict"])
        model.to(torch.device(device))
        model.eval()
        return model
