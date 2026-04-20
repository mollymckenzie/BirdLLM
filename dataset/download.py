"""
Auto-downloads the BirdLLM dataset from Kaggle if not already present.

Reads KAGGLE_API_TOKEN from the .env file (loaded automatically by app.py).
"""

import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DATASET_DIR = Path(__file__).parent
DATASET_FILE = DATASET_DIR / "BirdLLM_dataset.csv"
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "mollymckenzie2/knoxville-ebird-data")


def dataset_exists() -> bool:
    return DATASET_FILE.exists()


def download():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError(
            "KAGGLE_API_TOKEN is not set. Add it to your .env file:\n"
            "  KAGGLE_API_TOKEN=your-token"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle")

    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
    api = KaggleApi()
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
        # Rename if Kaggle saved the file under a different name
        csvs = list(DATASET_DIR.glob("*.csv"))
        if csvs:
            csvs[0].rename(DATASET_FILE)

    if DATASET_FILE.exists():
        print(f"Dataset ready: {DATASET_FILE}")
    else:
        raise RuntimeError("Download succeeded but BirdLLM_dataset.csv was not found.")


def ensure_dataset():
    """Call at app startup — skips download if the file already exists."""
    if not dataset_exists():
        download()
