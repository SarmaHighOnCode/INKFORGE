#!/usr/bin/env python3
"""
INKFORGE — Hugging Face Space Setup Script

Prepares the repository for deployment to Hugging Face Spaces.

Usage:
    python setup_hf.py
"""

import shutil
from pathlib import Path


def setup_huggingface_space():
    """Prepare files for Hugging Face Spaces deployment."""

    print("INKFORGE — Hugging Face Space Setup")
    print("=" * 50)

    # 1. Copy SPACE_README.md to README.md for HF (backup original)
    readme_path = Path("README.md")
    space_readme = Path("SPACE_README.md")

    if space_readme.exists():
        print("\nTo deploy to Hugging Face Spaces:")
        print("1. Create a new Space at https://huggingface.co/spaces")
        print("2. Choose 'Gradio' as the SDK")
        print("3. Clone your Space repository")
        print("4. Copy these files to your Space:")
        print("   - app.py")
        print("   - backend/ (entire directory)")
        print("   - requirements-hf.txt -> requirements.txt")
        print("   - SPACE_README.md -> README.md")
        print("   - checkpoints/ (if you have trained models)")
        print("5. Push to Hugging Face")

    # 2. Create minimal __init__.py files if missing
    init_files = [
        "backend/__init__.py",
        "backend/app/__init__.py",
        "backend/app/ml/__init__.py",
        "backend/app/services/__init__.py",
    ]

    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.write_text('"""INKFORGE package."""\n')
            print(f"Created: {init_file}")

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nQuick start locally:")
    print("  pip install -r backend/requirements.txt")
    print("  python app.py")
    print("\nThen open: http://localhost:7860")


if __name__ == "__main__":
    setup_huggingface_space()
