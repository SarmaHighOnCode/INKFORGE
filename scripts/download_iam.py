"""
INKFORGE — IAM Dataset Download Script

Downloads and extracts the IAM On-Line Handwriting Database.

Usage:
    python scripts/download_iam.py --output data/iam/

Dataset Info:
    - 13,049 transcribed handwritten texts
    - 221 unique writers
    - Source: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database

Note: IAM requires free registration at https://fki.tic.heia-fr.ch/register
"""

import argparse
import getpass
import hashlib
import os
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm


# IAM On-Line Database URLs and checksums
IAM_BASE_URL = "https://fki.tic.heia-fr.ch/DBs/iamOnDB/"
IAM_FILES = {
    "lineStrokes-all.tar.gz": {
        "url": "lineStrokes-all.tar.gz",
        "md5": None,  # Checksum verification optional
        "description": "Stroke XML files",
    },
    "ascii-all.tar.gz": {
        "url": "ascii-all.tar.gz",
        "md5": None,
        "description": "ASCII transcriptions",
    },
    "original-xml-all.tar.gz": {
        "url": "original-xml-all.tar.gz",
        "md5": None,
        "description": "Original XML metadata",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IAM On-Line dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/iam/",
        help="Output directory for downloaded data.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="IAM database username (or set IAM_USERNAME env var).",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="IAM database password (or set IAM_PASSWORD env var).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction after download.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(IAM_FILES.keys()),
        help="Specific files to download.",
    )
    return parser.parse_args()


def get_credentials(args) -> tuple[str, str]:
    """Get IAM credentials from args, env, or prompt."""
    username = args.username or os.environ.get("IAM_USERNAME")
    password = args.password or os.environ.get("IAM_PASSWORD")

    if not username:
        print("\nIAM On-Line Database requires registration.")
        print("Register at: https://fki.tic.heia-fr.ch/register")
        print()
        username = input("Username: ").strip()

    if not password:
        password = getpass.getpass("Password: ")

    return username, password


def download_file(
    url: str,
    output_path: Path,
    auth: tuple[str, str],
    chunk_size: int = 8192,
) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, auth=auth, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"Authentication failed. Please check your credentials.")
        else:
            print(f"HTTP error: {e}")
        return False

    except requests.exceptions.RequestException as e:
        print(f"Download error: {e}")
        return False


def verify_checksum(file_path: Path, expected_md5: str | None) -> bool:
    """Verify file MD5 checksum."""
    if expected_md5 is None:
        return True

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    actual_md5 = md5.hexdigest()
    if actual_md5 != expected_md5:
        print(f"Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return False

    return True


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract tar.gz or zip archive with path traversal protection."""
    try:
        resolved_out = output_dir.resolve()

        if archive_path.suffix == ".gz" or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                # Validate members against path traversal
                safe_members = []
                for member in tar.getmembers():
                    member_path = (resolved_out / member.name).resolve()
                    if not member_path.is_relative_to(resolved_out):
                        print(f"Skipping suspicious member: {member.name}")
                        continue
                    safe_members.append(member)
                tar.extractall(output_dir, members=safe_members, filter="data")
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                for info in zip_ref.infolist():
                    member_path = (resolved_out / info.filename).resolve()
                    if not member_path.is_relative_to(resolved_out):
                        print(f"Skipping suspicious member: {info.filename}")
                        continue
                    zip_ref.extract(info, output_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False

        print(f"Extracted: {archive_path.name}")
        return True

    except Exception as e:
        print(f"Extraction error: {e}")
        return False


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INKFORGE — IAM On-Line Database Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Check for existing files
    existing_files = []
    for filename in args.files:
        file_path = output_dir / filename
        if file_path.exists():
            existing_files.append(filename)

    if existing_files:
        print(f"Found existing files: {', '.join(existing_files)}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != "y":
            args.files = [f for f in args.files if f not in existing_files]

    if not args.files:
        print("All files already downloaded.")
        if not args.skip_extract:
            print("\nExtracting archives...")
            for filename in IAM_FILES.keys():
                archive_path = output_dir / filename
                if archive_path.exists():
                    extract_archive(archive_path, output_dir)
        return

    # Get credentials
    username, password = get_credentials(args)
    auth = (username, password)

    # Download files
    print("\nDownloading files...")
    success_count = 0

    for filename in args.files:
        if filename not in IAM_FILES:
            print(f"Unknown file: {filename}")
            continue

        file_info = IAM_FILES[filename]
        url = urljoin(IAM_BASE_URL, file_info["url"])
        output_path = output_dir / filename

        print(f"\n{file_info['description']}: {filename}")

        if download_file(url, output_path, auth):
            if verify_checksum(output_path, file_info["md5"]):
                success_count += 1

                # Extract if not skipped
                if not args.skip_extract:
                    extract_archive(output_path, output_dir)

    print()
    print("=" * 60)
    print(f"Downloaded {success_count}/{len(args.files)} files")

    if success_count == len(args.files):
        print("\nNext steps:")
        print("  1. Preprocess the data:")
        print("     python scripts/preprocess.py --input data/iam/ --output data/processed/")
        print("  2. Start training:")
        print("     python train.py --config configs/lstm_mdn_base.yaml")
    else:
        print("\nSome downloads failed. Please check your credentials and try again.")

    print("=" * 60)


if __name__ == "__main__":
    main()
