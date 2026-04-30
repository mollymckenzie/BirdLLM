from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset.download import ensure_dataset, DATASET_FILE  # adjust import path as needed

def load_splits(
    test_years: list = None,
    val_years: list = None,
    val_size: float = .1,
    random_state: int = 42
):
    """
    Ensures the dataset is downloaded, then returns train/val/test splits.
    """
    # This will download only if not already present
    ensure_dataset()

    df = pd.read_csv(DATASET_FILE)

    df["year"] = pd.to_datetime(df["eventDate"], errors="coerce").dt.year

        # Drop rows where year couldn't be parsed
    unparseable = df["year"].isna().sum()
    if unparseable > 0:
        print(f"Warning: dropping {unparseable} rows with unparseable eventDate")
        df = df.dropna(subset=["year"])

    df["year"] = df["year"].astype(int)

    # Show available years so you can make an informed choice
    year_counts = df.groupby("year").size().sort_index()
    print("Observations per year:")
    print(year_counts.to_string())
    print()

    # Default: hold out the most recent year as test
    if test_years is None:
        test_years = [int(df["year"].max())]
        print(f"No test_years specified — defaulting to most recent year: {test_years}")

    # Carve out test set by year
    test_df    = df[df["year"].isin(test_years)]
    remaining  = df[~df["year"].isin(test_years)]

    # Carve out val set by year (if specified)
    if val_years is not None:
        val_df  = remaining[remaining["year"].isin(val_years)]
        train_df = remaining[~remaining["year"].isin(val_years)]
    else:
        # Fall back to random split within the remaining years
        train_df, val_df = train_test_split(
            remaining,
            test_size=val_size / (1 - len(test_df) / len(df)),
            random_state=random_state
        )

    print(f"Train years: {sorted(train_df['year'].unique())}")
    print(f"Val years:   {sorted(val_df['year'].unique())}")
    print(f"Test years:  {sorted(test_df['year'].unique())}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df