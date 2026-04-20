"""
Auto-downloads the BirdLLM dataset from Kaggle if not already present.

Requirements:
  - kaggle Python package  (pip install kaggle)
  - KAGGLE_API_TOKEN environment variable set to your Kaggle API token

Set these environment variables before running:
  set KAGGLE_API_TOKEN=your-token
  set KAGGLE_DATASET=your-username/knoxville-ebird-data
"""

import os
import zipfile
from pathlib import Path

DATASET_DIR = Path(__file__).parent
DATASET_FILE = DATASET_DIR / "BirdLLM_dataset.csv"
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "mollymckenzie2/knoxville-ebird-data")


def _configure_kaggle_auth():
    """Write KAGGLE_API_TOKEN to the kaggle.json file the API client expects."""
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        return

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    # Only write if not already present to avoid overwriting existing credentials
    if not kaggle_json.exists():
        import json
        # KAGGLE_API_TOKEN format doesn't include username,
        # but the kaggle package can use it directly via the env var
        kaggle_json.write_text(
            json.dumps({"key": token, "username": ""}),
            encoding="utf-8",
        )
        kaggle_json.chmod(0o600)


def dataset_exists() -> bool:
    return DATASET_FILE.exists()


def download():
    if not KAGGLE_DATASET:
        raise RuntimeError(
            "KAGGLE_DATASET environment variable is not set.\n"
            "Set it to your Kaggle dataset slug, e.g.:\n"
            "  set KAGGLE_DATASET=your-username/knoxville-ebird-data\n\n"
            "Find your Kaggle username at: https://www.kaggle.com/account"
        )

    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError(
            "KAGGLE_API_TOKEN environment variable is not set.\n"
            "Set it to your Kaggle API token:\n"
            "  set KAGGLE_API_TOKEN=your-token"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle")

    _configure_kaggle_auth()

    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
    api = KaggleApiExtended()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATASET_DIR), unzip=False)

    # Extract the downloaded zip
    zip_path = DATASET_DIR / f"{KAGGLE_DATASET.split('/')[-1]}.zip"
    if zip_path.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATASET_DIR)
        zip_path.unlink()

    if not DATASET_FILE.exists():
        # Rename if the extracted file has a different name
        csvs = list(DATASET_DIR.glob("*.csv"))
        if csvs:
            csvs[0].rename(DATASET_FILE)

    if DATASET_FILE.exists():
        print(f"Dataset ready: {DATASET_FILE}")
    else:
        raise RuntimeError("Download succeeded but BirdLLM_dataset.csv was not found.")


def ensure_dataset():
    """Call this at app startup — downloads only if the file is missing."""
    if not dataset_exists():
        download()
