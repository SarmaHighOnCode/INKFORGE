---
title: INKFORGE - Handwriting Synthesis
emoji: ✍️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
tags:
  - handwriting
  - synthesis
  - lstm
  - generative
  - pytorch
---

# INKFORGE - Handwriting Synthesis Engine

Generate realistic handwritten text using deep learning.

## Features

- **Multiple Writing Styles**: Neat cursive, casual print, rushed notes, and more
- **Customizable Parameters**: Control variation, stroke width, ink color
- **Full Document Generation**: Generate multi-paragraph handwritten documents
- **Writing Fatigue Simulation**: Text degrades naturally over long passages

## How It Works

INKFORGE uses an LSTM + Mixture Density Network (MDN) architecture based on
[Graves 2013](https://arxiv.org/abs/1308.0850) to generate handwriting stroke-by-stroke.

## Usage

1. Enter your text in the input box
2. Select a writing style preset
3. Adjust temperature (variation) and other parameters
4. Click "Generate Handwriting"

## Note

This demo uses a randomly initialized model for demonstration.
For realistic handwriting, train the model on the IAM On-Line Handwriting Database.

## Links

- [GitHub Repository](https://github.com/SarmaHighOnCode/INKFORGE)
- [IAM Dataset](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)

## License

MIT License
