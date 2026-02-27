"""
INKFORGE — Gradio Web Interface

Hugging Face Spaces deployment for handwriting synthesis.
Generates realistic handwritten text using LSTM+MDN model.

Usage:
    python app.py
    # Opens at http://localhost:7860
"""

import io
import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from backend.app.ml.model import HandwritingLSTM
from backend.app.ml.utils import build_vocab, strokes_to_absolute
from backend.app.services.renderer import Renderer


# Global model instance
MODEL = None
VOCAB = None
RENDERER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Style presets with display names
STYLE_PRESETS = {
    "Neat Cursive": 0,
    "Casual Print": 1,
    "Rushed Notes": 2,
    "Doctor's Scrawl": 3,
    "Elegant Formal": 4,
}

# Ink colors
INK_COLORS = ["black", "blue", "dark_blue", "sepia"]

# Paper styles
PAPER_STYLES = ["white", "cream", "aged", "lined"]


def load_model():
    """Load the handwriting model."""
    global MODEL, VOCAB, RENDERER

    VOCAB = build_vocab()
    vocab_size = len(VOCAB)

    # Initialize model
    MODEL = HandwritingLSTM(
        vocab_size=vocab_size,
        char_embed_dim=256,
        style_dim=128,
        hidden_dim=512,
        num_layers=3,
        dropout=0.0,
        num_mixtures=20,
    )
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    # Try to load checkpoint if available
    checkpoint_path = Path("checkpoints/lstm_mdn_v1_best.pt")
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            MODEL.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Using randomly initialized model (demo mode)")
    else:
        print("No checkpoint found. Using randomly initialized model (demo mode)")
        print("Train a model first: python train.py --config configs/lstm_mdn_base.yaml")

    RENDERER = Renderer()
    print(f"Model loaded on {DEVICE}")


def tokenize(text: str) -> list[int]:
    """Convert text to token indices."""
    if VOCAB is None:
        raise RuntimeError("Model not loaded — call load_model() before tokenize()")
    unk_idx = VOCAB.get("<unk>", 3)
    return [VOCAB.get(char, unk_idx) for char in text]


def generate_strokes(
    text: str,
    style_idx: int,
    temperature: float,
    max_strokes_per_char: int = 50,
) -> list[tuple]:
    """Generate stroke sequence for text."""
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    # Create deterministic style embedding per style (thread-safe)
    gen = torch.Generator(device=DEVICE).manual_seed(style_idx * 42)
    style_z = torch.randn(1, 128, device=DEVICE, generator=gen)

    # Tokenize text
    char_indices = tokenize(text)
    char_seq = torch.tensor([char_indices], dtype=torch.long, device=DEVICE)

    # Generate strokes
    strokes = []
    hidden = MODEL.get_initial_hidden(batch_size=1)
    prev_stroke = torch.zeros(1, 1, 5, device=DEVICE)

    with torch.no_grad():
        for char_idx in range(len(char_indices)):
            current_char = char_seq[:, char_idx:char_idx + 1]
            char_strokes = 0

            while char_strokes < max_strokes_per_char:
                mdn_params, pen_logits, hidden = MODEL(
                    current_char, prev_stroke, style_z, hidden
                )

                stroke = MODEL.sample(
                    mdn_params[0, 0],
                    pen_logits[0, 0],
                    temperature=temperature,
                )

                dx, dy, p1, p2, p3 = stroke
                strokes.append((dx, dy, p1, p2, p3))

                prev_stroke = torch.tensor(
                    [[[dx, dy, p1, p2, p3]]],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                char_strokes += 1

                if p2 == 1:  # Pen up
                    break
                if p3 == 1:  # End of sequence
                    return strokes

    return strokes


def generate_handwriting(
    text: str,
    style: str,
    temperature: float,
    ink_color: str,
    paper_style: str,
    stroke_width: float,
) -> Image.Image:
    """
    Generate handwritten image from text.

    Args:
        text: Input text to render.
        style: Writing style preset.
        temperature: Randomness (0.1=consistent, 1.0=varied).
        ink_color: Ink color name.
        paper_style: Paper background style.
        stroke_width: Stroke thickness.

    Returns:
        PIL Image of rendered handwriting.
    """
    if not text.strip():
        # Return blank image
        return Image.new("RGB", (800, 200), "white")

    # Limit text length for demo
    text = text[:200]

    # Get style index
    style_idx = STYLE_PRESETS.get(style, 0)

    # Generate strokes
    try:
        strokes = generate_strokes(text, style_idx, temperature)
    except Exception as e:
        print(f"Generation error: {e}")
        # Return error image
        img = Image.new("RGB", (800, 200), "white")
        return img

    if not strokes:
        return Image.new("RGB", (800, 200), "white")

    # Render to image
    add_lines = paper_style == "lined"
    paper_color = "white" if paper_style == "lined" else paper_style

    image = RENDERER.render_to_image(
        strokes,
        width=800,
        height=200,
        ink_color=ink_color,
        background_color=paper_color,
        stroke_width=stroke_width,
        scale=3.0,
        padding=30.0,
    )

    return image


def generate_document(
    text: str,
    style: str,
    temperature: float,
    ink_color: str,
    paper_style: str,
    stroke_width: float,
    fatigue: float,
) -> Image.Image:
    """
    Generate full document with paragraph layout.

    Args:
        text: Multi-paragraph input text.
        style: Writing style.
        temperature: Base temperature.
        ink_color: Ink color.
        paper_style: Paper style.
        stroke_width: Stroke width.
        fatigue: Fatigue simulation (0-1).

    Returns:
        PIL Image of full page.
    """
    if not text.strip():
        return Image.new("RGB", (800, 1000), "white")

    # Split into words and generate each
    words = text.split()[:50]  # Limit for demo

    style_idx = STYLE_PRESETS.get(style, 0)
    all_word_data = []

    cursor_x, cursor_y = 30.0, 40.0
    line_height = 25.0
    char_width = 8.0

    for i, word in enumerate(words):
        # Apply fatigue: increase temperature over document
        progress = i / max(len(words), 1)
        current_temp = temperature + (fatigue * 0.3 * progress)
        current_temp = min(current_temp, 1.0)

        try:
            strokes = generate_strokes(word, style_idx, current_temp)
        except Exception:
            continue

        if not strokes:
            continue

        word_width = len(word) * char_width

        # Line wrap
        if cursor_x + word_width > 770:
            cursor_x = 30.0
            cursor_y += line_height

        all_word_data.append({
            "word": word,
            "strokes": strokes,
            "position": (cursor_x, cursor_y),
        })

        cursor_x += word_width + 15.0

    if not all_word_data:
        return Image.new("RGB", (800, 1000), "white")

    # Render document
    add_lines = paper_style == "lined"
    svg_content = RENDERER.strokes_to_svg_with_positions(
        all_word_data,
        ink_color=ink_color,
        stroke_width_base=stroke_width,
        paper_size="a4",
        paper_color="white" if paper_style == "lined" else paper_style,
        add_lines=add_lines,
        scale=2.0,
    )

    # Convert to image
    png_bytes = RENDERER.export_png(svg_content, output_path=None)
    return Image.open(io.BytesIO(png_bytes))


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="INKFORGE - Handwriting Synthesis",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # INKFORGE - Handwriting Synthesis Engine

        Generate realistic handwritten text using deep learning.
        Type your text below and customize the writing style.

        **Note:** This demo uses a randomly initialized model.
        For realistic output, train the model on the IAM dataset first.
        """)

        with gr.Tabs():
            # Tab 1: Single Line
            with gr.TabItem("Single Line"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Text to write",
                            placeholder="Enter text here...",
                            value="Hello, World!",
                            max_lines=1,
                        )

                    with gr.Column(scale=1):
                        style_dropdown = gr.Dropdown(
                            label="Writing Style",
                            choices=list(STYLE_PRESETS.keys()),
                            value="Neat Cursive",
                        )

                with gr.Row():
                    temperature_slider = gr.Slider(
                        label="Variation (Temperature)",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.4,
                        step=0.1,
                        info="Lower = more consistent, Higher = more varied",
                    )
                    stroke_width_slider = gr.Slider(
                        label="Stroke Width",
                        minimum=0.5,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                    )

                with gr.Row():
                    ink_dropdown = gr.Dropdown(
                        label="Ink Color",
                        choices=INK_COLORS,
                        value="black",
                    )
                    paper_dropdown = gr.Dropdown(
                        label="Paper Style",
                        choices=PAPER_STYLES,
                        value="white",
                    )

                generate_btn = gr.Button("Generate Handwriting", variant="primary")

                output_image = gr.Image(
                    label="Generated Handwriting",
                    type="pil",
                    height=250,
                )

                generate_btn.click(
                    fn=generate_handwriting,
                    inputs=[
                        text_input,
                        style_dropdown,
                        temperature_slider,
                        ink_dropdown,
                        paper_dropdown,
                        stroke_width_slider,
                    ],
                    outputs=output_image,
                )

            # Tab 2: Document
            with gr.TabItem("Full Document"):
                doc_text_input = gr.Textbox(
                    label="Document Text",
                    placeholder="Enter your document text here...",
                    value="This is a sample document. It demonstrates the full page generation capability of INKFORGE.",
                    lines=5,
                )

                with gr.Row():
                    doc_style = gr.Dropdown(
                        label="Style",
                        choices=list(STYLE_PRESETS.keys()),
                        value="Neat Cursive",
                    )
                    doc_temp = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.4,
                    )

                with gr.Row():
                    doc_ink = gr.Dropdown(
                        label="Ink",
                        choices=INK_COLORS,
                        value="blue",
                    )
                    doc_paper = gr.Dropdown(
                        label="Paper",
                        choices=PAPER_STYLES,
                        value="lined",
                    )

                with gr.Row():
                    doc_stroke = gr.Slider(
                        label="Stroke Width",
                        minimum=0.5,
                        maximum=3.0,
                        value=1.2,
                    )
                    doc_fatigue = gr.Slider(
                        label="Writing Fatigue",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        info="Simulates tiredness over long text",
                    )

                doc_generate_btn = gr.Button("Generate Document", variant="primary")

                doc_output = gr.Image(
                    label="Generated Document",
                    type="pil",
                    height=600,
                )

                doc_generate_btn.click(
                    fn=generate_document,
                    inputs=[
                        doc_text_input,
                        doc_style,
                        doc_temp,
                        doc_ink,
                        doc_paper,
                        doc_stroke,
                        doc_fatigue,
                    ],
                    outputs=doc_output,
                )

        gr.Markdown("""
        ---
        **INKFORGE** - Long-form handwriting synthesis engine

        Built with PyTorch, based on [Graves 2013](https://arxiv.org/abs/1308.0850)

        [GitHub](https://github.com/SarmaHighOnCode/INKFORGE) |
        Train your own model with the IAM On-Line Handwriting Database
        """)

    return demo


if __name__ == "__main__":
    print("Loading INKFORGE model...")
    load_model()

    print("Starting Gradio interface...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
