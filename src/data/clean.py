"""Cleaning pipeline for application_train.csv.

Pipeline (in order):
  1. Copy (never mutate caller data).
  2. Replace DAYS_EMPLOYED == 365243 sentinel with NaN.
  3. Drop columns with >60% missing values.
  4. Impute numeric NaNs with column median, categorical NaNs with 'MISSING'.
  5. Encode categoricals:
       - low-cardinality  (nunique <= HIGH_CARDINALITY_THRESHOLD) -> one-hot
       - high-cardinality (> threshold) -> frequency encoding
  6. Stratified 70/15/15 split on TARGET, random_state=42.
  7. Standardize numeric features with StandardScaler fit on the 70% train slice only.
  8. Add a SPLIT column and persist to data/processed/train_clean.parquet.

The human-facing pipeline is `notebooks/00_data_cleaning.ipynb`. This module
keeps the same steps as importable functions so `tests/test_clean.py` can run
them on synthetic frames. `run_cli()` is optional; prefer the notebook.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Threshold above which a column is dropped entirely.
DROP_MISSING_THRESHOLD: float = 0.60

# Categorical-cardinality cutoff: anything with more unique values is treated
# as "high-cardinality" and frequency-encoded rather than one-hot.
HIGH_CARDINALITY_THRESHOLD: int = 10

# Sentinel value used in DAYS_EMPLOYED for applicants with no/unknown employment.
DAYS_EMPLOYED_SENTINEL: int = 365_243

# Split fractions (train / val / test) and the seed that pins them.
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
RANDOM_STATE: int = 42

# Output paths.
PARQUET_PATH = Path("data") / "processed" / "train_clean.parquet"
INDICES_PATH = Path("data") / "processed" / "split_indices.npz"
CLEANING_SUMMARY_PATH = Path("reports") / "cleaning_summary.md"

# Columns that should never be standardized (IDs and the target).
NON_FEATURE_COLS: tuple[str, ...] = ("SK_ID_CURR", "TARGET", "SPLIT")


def _fix_days_employed_sentiment(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace DAYS_EMPLOYED == 365243 with NaN. Returns (df, count_replaced)."""
    if "DAYS_EMPLOYED" not in df.columns:
        return df, 0
    mask = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    n = int(mask.sum())
    if n:
        df = df.copy()
        df.loc[mask, "DAYS_EMPLOYED"] = np.nan
    return df, n


def _drop_high_missing(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[dict]]:
    """Drop columns with missing fraction above `threshold`. Returns (df, dropped_log)."""
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    log = [
        {"column": col, "missing_pct": float(missing_frac[col])}
        for col in to_drop
    ]
    return df.drop(columns=to_drop), log


def _impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Median for numeric NaNs, 'MISSING' for object NaNs. Returns (df, fill_counts)."""
    fill_counts: dict[str, int] = {}
    df = df.copy()
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("MISSING")
        fill_counts[col] = n_missing
    return df, fill_counts


def _encode_categoricals(df: pd.DataFrame, threshold: int = HIGH_CARDINALITY_THRESHOLD) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One-hot low-cardinality columns, frequency-encode high-cardinality columns.

    Returns (df, one_hot_columns, freq_encoded_columns).
    """
    cat_cols = [c for c in df.select_dtypes(include="object").columns]
    if not cat_cols:
        return df, [], []

    # Identify high-cardinality columns up front so they don't get one-hot'd.
    high_card = [c for c in cat_cols if df[c].nunique(dropna=False) > threshold]
    low_card = [c for c in cat_cols if c not in high_card]

    # Frequency encoding: replace each category with its occurrence count.
    for col in high_card:
        counts = df[col].value_counts(dropna=False)
        df[col] = df[col].map(counts).astype(float)

    # One-hot encoding: drop_first to avoid the dummy-variable trap.
    df = pd.get_dummies(df, columns=low_card, drop_first=True)

    # After one-hot, ensure the new dummy columns are float (not bool) so the
    # downstream StandardScaler accepts them cleanly.
    bool_cols = df.select_dtypes(include="bool").columns
    for col in bool_cols:
        df[col] = df[col].astype(float)

    return df, low_card, high_card


def _stratified_split(
    n_rows: int, target: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks (train_mask, val_mask, test_mask) stratified on target."""
    # First split: train vs (val + test). The (val + test) pool gets 30%.
    pool_frac = VAL_FRAC + TEST_FRAC
    train_idx, pool_idx, _, _ = train_test_split(
        np.arange(n_rows),
        target.values,
        train_size=TRAIN_FRAC,
        test_size=pool_frac,
        stratify=target.values,
        random_state=RANDOM_STATE,
    )
    # Second split: carve val vs test from the pool, preserving stratification.
    val_frac_of_pool = VAL_FRAC / pool_frac
    val_idx, test_idx, _, _ = train_test_split(
        pool_idx,
        target.values[pool_idx],
        train_size=val_frac_of_pool,
        test_size=1 - val_frac_of_pool,
        stratify=target.values[pool_idx],
        random_state=RANDOM_STATE,
    )
    train_mask = np.zeros(n_rows, dtype=bool)
    val_mask = np.zeros(n_rows, dtype=bool)
    test_mask = np.zeros(n_rows, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    # Sanity: every row must belong to exactly one slice.
    assert int(train_mask.sum() + val_mask.sum() + test_mask.sum()) == n_rows
    return train_mask, val_mask, test_mask


def clean(
    df: pd.DataFrame,
    *,
    drop_threshold: float = DROP_MISSING_THRESHOLD,
    high_card_threshold: int = HIGH_CARDINALITY_THRESHOLD,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline on ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw application_train dataframe (must already include TARGET).
    drop_threshold : float
        Columns whose missing fraction exceeds this are dropped.
    high_card_threshold : int
        Categorical columns with more unique values than this are frequency-encoded.
    random_state : int
        Seed for the stratified train/val/test split.

    Returns
    -------
    (cleaned_df, info)
        ``cleaned_df`` includes the original SK_ID_CURR and TARGET, a new SPLIT
        column (``'train'`` / ``'val'`` / ``'test'``), and all features
        standardized on train statistics. ``info`` is a dict of metadata useful
        for the cleaning summary report.
    """
    info: dict = {
        "n_rows_in": int(df.shape[0]),
        "n_cols_in": int(df.shape[1]),
        "sentinel_replaced": 0,
        "dropped_columns": [],
        "fill_counts": {},
        "one_hot_columns": [],
        "frequency_encoded_columns": [],
        "split_sizes": {},
        "n_cols_out": 0,
        "n_rows_out": 0,
    }

    # 1. Sentinel fix
    df, n_replaced = _fix_days_employed_sentiment(df)
    info["sentinel_replaced"] = n_replaced

    # 2. Drop high-missing
    df, dropped = _drop_high_missing(df, drop_threshold)
    info["dropped_columns"] = dropped

    # 3. Impute
    df, fill_counts = _impute(df)
    info["fill_counts"] = fill_counts

    # 4. Encode categoricals (BEFORE split — frequency encoding uses full-population
    #    counts, which is acceptable because counts are target-independent).
    df, one_hot, freq_enc = _encode_categoricals(df, threshold=high_card_threshold)
    info["one_hot_columns"] = one_hot
    info["frequency_encoded_columns"] = freq_enc

    # 5. Stratified split (BEFORE scaling so the scaler is fit on train only).
    assert "TARGET" in df.columns, "clean() expects the input df to include TARGET."
    train_mask, val_mask, test_mask = _stratified_split(len(df), df["TARGET"])
    info["split_sizes"] = {
        "train": int(train_mask.sum()),
        "val": int(val_mask.sum()),
        "test": int(test_mask.sum()),
    }

    # 6. Standardize numeric features fit on train only.
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Up-cast int columns to float before standardization to avoid the
    # pandas FutureWarning about incompatible dtypes when we assign the
    # scaled float values back into int64 columns.
    for col in feature_cols:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols])
    scaled = scaler.transform(df[feature_cols])
    # Use a fresh DataFrame slice to avoid any SettingWithCopy warnings.
    df = df.copy()
    df.loc[:, feature_cols] = scaled

    # 7. Add SPLIT column.
    split_col = np.full(len(df), "", dtype=object)
    split_col[train_mask] = "train"
    split_col[val_mask] = "val"
    split_col[test_mask] = "test"
    df["SPLIT"] = split_col

    info["n_rows_out"] = int(df.shape[0])
    info["n_cols_out"] = int(df.shape[1])
    info["feature_count"] = len(feature_cols)
    info["scaler_mean_norm"] = float(np.linalg.norm(scaler.mean_))  # diagnostic only
    return df, info


# ---------------------------------------------------------------------------
# CLI runner — full end-to-end against the real CSV.
# ---------------------------------------------------------------------------

def _render_cleaning_summary(info: dict) -> str:
    lines: list[str] = []
    lines.append("# Cleaning Summary — application_train.csv")
    lines.append("")
    lines.append("Generated by `src.data.clean` (Phase 1).")
    lines.append("")
    lines.append("## Shape")
    lines.append("")
    lines.append(f"- Rows in: **{info['n_rows_in']:,}**, columns in: **{info['n_cols_in']}**")
    lines.append(f"- Rows out: **{info['n_rows_out']:,}**, columns out: **{info['n_cols_out']}**")
    lines.append(f"- Numeric feature columns (post-encoding): **{info.get('feature_count', 0)}**")
    lines.append("")

    lines.append("## DAYS_EMPLOYED sentinel replaced")
    lines.append("")
    lines.append(f"- Rows where `DAYS_EMPLOYED == {DAYS_EMPLOYED_SENTINEL}` were set to NaN: **{info['sentinel_replaced']:,}**")
    lines.append("")

    lines.append(f"## Columns dropped (> {DROP_MISSING_THRESHOLD:.0%} missing)")
    lines.append("")
    if info["dropped_columns"]:
        lines.append(f"**{len(info['dropped_columns'])} column(s) dropped:**")
        lines.append("")
        lines.append("| Column | Missing % |")
        lines.append("|---|---|")
        for entry in info["dropped_columns"]:
            lines.append(f"| `{entry['column']}` | {entry['missing_pct']:.2%} |")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## Encoding")
    lines.append("")
    lines.append(f"- One-hot encoded ({len(info['one_hot_columns'])} columns): {', '.join(f'`{c}`' for c in info['one_hot_columns'])}")
    lines.append(f"- Frequency encoded ({len(info['frequency_encoded_columns'])} columns): {', '.join(f'`{c}`' for c in info['frequency_encoded_columns'])}")
    lines.append("")

    lines.append("## Stratified split")
    lines.append("")
    lines.append(f"- Train: **{info['split_sizes'].get('train', 0):,}**")
    lines.append(f"- Val:   **{info['split_sizes'].get('val', 0):,}**")
    lines.append(f"- Test:  **{info['split_sizes'].get('test', 0):,}**")
    lines.append("")

    lines.append("## Imputation")
    lines.append("")
    lines.append(f"- Columns with NaNs filled: **{len(info['fill_counts'])}**")
    lines.append("- Numeric: column median. Categorical: literal `'MISSING'`.")
    lines.append("")

    return "\n".join(lines)


def run_cli() -> None:
    """Load -> clean -> persist parquet + indices + summary report."""
    from src.data.load import load_application_train

    df = load_application_train()
    cleaned, info = clean(df)

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(PARQUET_PATH, index=False)

    # Recompute masks for the saved indices file (matches what was used during clean).
    target = cleaned["TARGET"].values
    split_col = cleaned["SPLIT"].values
    np.savez(
        INDICES_PATH,
        train_mask=(split_col == "train"),
        val_mask=(split_col == "val"),
        test_mask=(split_col == "test"),
        target=target,
    )

    CLEANING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_SUMMARY_PATH.write_text(_render_cleaning_summary(info), encoding="utf-8")

    print(f"Wrote cleaned parquet to {PARQUET_PATH}")
    print(f"Wrote split indices to {INDICES_PATH}")
    print(f"Wrote cleaning summary to {CLEANING_SUMMARY_PATH}")
    print(
        f"Shape: {info['n_rows_in']:,}x{info['n_cols_in']} -> "
        f"{info['n_rows_out']:,}x{info['n_cols_out']}; "
        f"dropped {len(info['dropped_columns'])} columns; "
        f"DAYS_EMPLOYED sentinel fixed in {info['sentinel_replaced']:,} rows; "
        f"split={info['split_sizes']}"
    )


if __name__ == "__main__":
    run_cli()
