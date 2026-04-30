"""
test_pipeline.py
----------------
Runs an end-to-end evaluation of the bird observation pipeline using a
year-based train/test split to avoid temporal leakage.

Usage:
    python test_pipeline.py
    python test_pipeline.py --test-years 2023 --val-years 2022
    python test_pipeline.py --species "robin" --location "Knoxville"
    python test_pipeline.py --species "hummingbird" --test-years 2023 2022
"""

import argparse
import json
import sys
import time
from collections import defaultdict

import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
# Adjust these if your module paths differ
from dataset.download import ensure_dataset, DATASET_FILE
from preprocessing.preprocess import run_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(
    test_years: list[int] | None = None,
    val_years:  list[int] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the eBird CSV and carve out train / val / test by calendar year so
    that no future observations can leak into the training window.
    """
    ensure_dataset()
    df = pd.read_csv(DATASET_FILE, low_memory=False)

    # ── parse year ─────────────────────────────────────────────────────────────
    df["year"] = pd.to_datetime(df["eventDate"], errors="coerce").dt.year
    bad = df["year"].isna().sum()
    if bad:
        print(f"  [warn] dropping {bad} rows with unparseable eventDate")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # ── also extract week if not already present ───────────────────────────────
    if "week" not in df.columns:
        df["week"] = pd.to_datetime(df["eventDate"], errors="coerce").dt.isocalendar().week.astype(int)

    year_counts = df.groupby("year").size().sort_index()
    print("\nObservations per year in full dataset:")
    for yr, cnt in year_counts.items():
        print(f"  {yr}: {cnt:,}")
    print()

    # ── default: hold out the single most-recent year ─────────────────────────
    if test_years is None:
        test_years = [int(df["year"].max())]
        print(f"  [info] no --test-years supplied; defaulting to {test_years}\n")

    test_df   = df[df["year"].isin(test_years)].copy()
    remaining = df[~df["year"].isin(test_years)].copy()

    if val_years is not None:
        val_df    = remaining[remaining["year"].isin(val_years)].copy()
        train_df  = remaining[~remaining["year"].isin(val_years)].copy()
    else:
        # No explicit val years → 10 % random slice of remaining
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            remaining, test_size=0.1, random_state=random_state
        )

    print(f"  Train years : {sorted(train_df['year'].unique())}")
    print(f"  Val years   : {sorted(val_df['year'].unique())}")
    print(f"  Test years  : {sorted(test_df['year'].unique())}")
    print(f"  Sizes       : train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}\n")

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def peak_precision(test_df: pd.DataFrame, peak_week_nums: set[int]) -> float:
    """
    What fraction of held-out observations fall inside the predicted peak weeks?
    A random baseline would score len(peak_weeks)/52.
    """
    if test_df.empty or not peak_week_nums:
        return 0.0
    hits = test_df["week"].isin(peak_week_nums).sum()
    return hits / len(test_df)


def random_baseline(n_peak_weeks: int, n_weeks: int = 52) -> float:
    """Expected precision if peak weeks were chosen at random."""
    return n_peak_weeks / n_weeks


def lift(precision: float, baseline: float) -> float:
    """How many times better than random?"""
    return precision / baseline if baseline > 0 else float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SINGLE-SPECIES TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_species(
    species: str,
    location: str | None,
    lat: float | None,
    lon: float | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Run run_pipeline on the TRAINING split, then score predictions against
    both the validation and test splits.

    run_pipeline is expected to accept a `data` kwarg (a pre-filtered
    DataFrame) so we can pass the train split instead of re-reading the file.
    If your version doesn't support that yet, see the note in the comments.
    """
    if verbose:
        print(f"  Testing species='{species}' location='{location or 'all'}'")

    t0 = time.perf_counter()

    # ── run pipeline on training data only ────────────────────────────────────
    # Pass `data=train_df` so the pipeline doesn't re-read the full CSV.
    # If run_pipeline doesn't accept a `data` kwarg yet, add the following
    # two lines to your preprocess.py run_pipeline() signature:
    #
    #   def run_pipeline(species_query, location_query=None,
    #                    lat=None, lon=None, data=None):
    #       df = data if data is not None else pd.read_csv(DATASET_FILE)
    #
    result = run_pipeline(
        species_query=species,
        location_query=location,
        lat=lat,
        lon=lon,
        data=train_df,          # ← remove this line if not yet supported
    )
    elapsed = time.perf_counter() - t0

    if "error" in result:
        print(f"    [skip] pipeline error: {result['error']}")
        return {"species": species, "skipped": True, "reason": result["error"]}

    peak_week_nums = {p["week"] for p in result.get("peak_weeks", [])}
    n_peak = len(peak_week_nums)

    # ── filter val/test to the same species the pipeline matched ──────────────
    matched_species = result.get("species_found", [])
    # eBird common name column is typically "commonName"
    name_col = next(
        (c for c in test_df.columns if c.lower() in ("commonname", "common_name")),
        None,
    )

    def filter_to_species(frame: pd.DataFrame) -> pd.DataFrame:
        if name_col and matched_species:
            mask = frame[name_col].str.lower().isin(
                [s.lower() for s in matched_species]
            )
            return frame[mask]
        return frame   # fallback: use everything

    val_species  = filter_to_species(val_df)
    test_species = filter_to_species(test_df)

    # ── compute metrics ───────────────────────────────────────────────────────
    val_prec   = peak_precision(val_species,  peak_week_nums)
    test_prec  = peak_precision(test_species, peak_week_nums)
    baseline   = random_baseline(n_peak)
    val_lift   = lift(val_prec,  baseline)
    test_lift  = lift(test_prec, baseline)

    record = {
        "species":          species,
        "location":         location or "all",
        "matched_species":  matched_species,
        "train_records":    result.get("total_records", 0),
        "val_records":      len(val_species),
        "test_records":     len(test_species),
        "peak_weeks":       sorted(peak_week_nums),
        "n_peak_weeks":     n_peak,
        "random_baseline":  round(baseline, 4),
        "val_precision":    round(val_prec,  4),
        "test_precision":   round(test_prec, 4),
        "val_lift":         round(val_lift,  2),
        "test_lift":        round(test_lift, 2),
        "pipeline_ms":      round(elapsed * 1000, 1),
        "skipped":          False,
    }

    if verbose:
        status = "✓ PASS" if test_lift >= 1.5 else "✗ WEAK"
        print(f"    {status}  peak_weeks={sorted(peak_week_nums)}")
        print(f"           val  precision={val_prec:.1%}  lift={val_lift:.2f}x")
        print(f"           test precision={test_prec:.1%}  lift={test_lift:.2f}x  "
              f"(baseline={baseline:.1%})  [{elapsed*1000:.0f} ms]")

    return record


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

# Default species/location combos to evaluate when none are supplied via CLI
DEFAULT_TEST_CASES = [
    {"species": "robin",           "location": "Knoxville", "lat": 35.96, "lon": -83.92},
    {"species": "hummingbird",     "location": "Knoxville", "lat": 35.96, "lon": -83.92},
    {"species": "sandhill crane",  "location": None,        "lat": None,  "lon": None},
    {"species": "warbler",         "location": "Tennessee", "lat": 35.5,  "lon": -86.5},
    {"species": "bald eagle",      "location": None,        "lat": None,  "lon": None},
]


def run_suite(
    test_cases: list[dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict]:
    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] ─────────────────────────────────────────")
        rec = test_species(
            species=tc["species"],
            location=tc.get("location"),
            lat=tc.get("lat"),
            lon=tc.get("lon"),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )
        results.append(rec)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict]) -> None:
    ran     = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]

    if not ran:
        print("\n[!] All test cases were skipped — nothing to report.")
        return

    avg_test_prec = sum(r["test_precision"] for r in ran) / len(ran)
    avg_test_lift = sum(r["test_lift"]      for r in ran) / len(ran)
    passed        = sum(1 for r in ran if r["test_lift"] >= 1.5)

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  Cases run     : {len(ran)}")
    print(f"  Cases skipped : {len(skipped)}")
    print(f"  Passed (lift ≥ 1.5x) : {passed}/{len(ran)}")
    print(f"  Avg test precision   : {avg_test_prec:.1%}")
    print(f"  Avg test lift        : {avg_test_lift:.2f}x")
    print()

    # Per-case table
    header = f"{'Species':<20} {'Location':<14} {'TestPrec':>9} {'Lift':>6} {'PeakWks':>8} {'Status'}"
    print(header)
    print("-" * len(header))
    for r in ran:
        status = "PASS" if r["test_lift"] >= 1.5 else "WEAK"
        print(
            f"  {r['species']:<18} {r['location']:<14} "
            f"{r['test_precision']:>8.1%} {r['test_lift']:>5.2f}x "
            f"{str(r['peak_weeks'][:3]):<20} {status}"
        )

    if skipped:
        print("\nSkipped:")
        for r in skipped:
            print(f"  {r['species']}: {r.get('reason', '?')}")

    print("═" * 60)

    # Overall pass/fail exit code
    if passed < len(ran):
        print(f"\n[!] {len(ran) - passed} case(s) scored below 1.5x lift threshold.\n")
    else:
        print("\n[✓] All cases passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test the bird pipeline with a year holdout.")
    p.add_argument("--test-years", nargs="+", type=int,
                   help="Year(s) to hold out as test set (default: most recent year)")
    p.add_argument("--val-years",  nargs="+", type=int,
                   help="Year(s) to hold out as validation set (default: random 10%%)")
    p.add_argument("--species",   type=str,
                   help="Single species to test instead of the default suite")
    p.add_argument("--location",  type=str, default=None)
    p.add_argument("--lat",       type=float, default=None)
    p.add_argument("--lon",       type=float, default=None)
    p.add_argument("--output-json", type=str, default=None,
                   help="Optional path to write full results as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Bird Pipeline — Test Evaluation")
    print("=" * 60)

    # ── build splits ──────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits(
        test_years=args.test_years,
        val_years=args.val_years,
    )

    # ── choose test cases ─────────────────────────────────────────────────────
    if args.species:
        test_cases = [
            {
                "species":  args.species,
                "location": args.location,
                "lat":      args.lat,
                "lon":      args.lon,
            }
        ]
    else:
        test_cases = DEFAULT_TEST_CASES

    # ── run ───────────────────────────────────────────────────────────────────
    results = run_suite(test_cases, train_df, val_df, test_df)

    # ── report ────────────────────────────────────────────────────────────────
    print_summary(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results written to {args.output_json}")

    # Non-zero exit if any case failed, useful for CI
    failed = sum(
        1 for r in results
        if not r.get("skipped") and r["test_lift"] < 1.5
    )
    sys.exit(failed)


if __name__ == "__main__":
    main()