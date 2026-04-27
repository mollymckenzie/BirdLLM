from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset.download import ensure_dataset, DATASET_FILE  # adjust import path as needed

def load_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_col: str = "eventDate",
    random_state: int = 42
):
    """
    Ensures the dataset is downloaded, then returns train/val/test splits.
    """
    # This will download only if not already present
    ensure_dataset()

    df = pd.read_csv(DATASET_FILE)

    stratify = df[stratify_col] if stratify_col else None

    # First split off the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Then split train into train + validation
    val_split = val_size / (1 - test_size)  # adjust val proportion
    stratify_val = train_val_df[stratify_col] if stratify_col else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split,
        random_state=random_state,
        stratify=stratify_val
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df