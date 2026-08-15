"""Cached data loaders for the Phase 5 dashboard.

All loaders use ``@st.cache_data`` so the underlying parquet / CSV is
read once per process, then handed to every rerun as a frozen DataFrame.

Three things are loaded:
  - ``load_clean()``  — the Phase 1 cleaned parquet (TARGET + SPLIT columns
    intact; numeric columns already StandardScaler'd; categoricals
    one-hot/frequency-encoded). Used for KPI cards in the Overview tab.
  - ``load_raw()``    — the raw ``application_train.csv`` (307,511 x 122).
    Used for human-readable segment filters in the Segment Analysis tab,
    where we need the original categorical strings (NAME_EDUCATION_TYPE,
    OCCUPATION_TYPE, etc.) and the un-scaled AMT_INCOME_TOTAL etc.
  - ``load_importance_table()`` — parses ``reports/feature_importance.md``
    into a DataFrame with columns
    [rank, feature, mean_rank, combined_score, source].
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

# Repo root resolves to the parent of the dashboard/ directory at runtime.
REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = REPO_ROOT / "data" / "processed" / "train_clean.parquet"
TOP30_PARQUET = REPO_ROOT / "data" / "processed" / "train_top30.parquet"
RAW_CSV = REPO_ROOT / ".home-credit-default-risk" / "application_train.csv"
IMPORTANCE_MD = REPO_ROOT / "reports" / "feature_importance.md"
PHASE4_MD = REPO_ROOT / "reports" / "phase4_model_comparison.md"


@st.cache_data(show_spinner="Loading portfolio data…")
def load_clean() -> pd.DataFrame:
    """Load the Phase 1 cleaned parquet (TARGET + SPLIT intact).

    Returns the full 307,511 x 144 DataFrame. Numeric columns are
    StandardScaler'd (z-scored); categoricals are one-hot/freq-encoded.
    """
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"{CLEAN_PARQUET} not found. Run Phase 1 notebook to regenerate."
        )
    return pd.read_parquet(CLEAN_PARQUET)


@st.cache_data(show_spinner="Loading top-30 features…")
def load_top30() -> pd.DataFrame:
    """Load the Phase 3 top-30 parquet (raw scale, no scaling applied).

    Used by ``_form_to_features.py`` to fill in median values for the
    "background" columns the user doesn't enter explicitly.
    """
    if not TOP30_PARQUET.exists():
        raise FileNotFoundError(
            f"{TOP30_PARQUET} not found. Run Phase 3 notebook to regenerate."
        )
    return pd.read_parquet(TOP30_PARQUET)


@st.cache_data(show_spinner="Loading application_train.csv…")
def load_raw() -> pd.DataFrame:
    """Load the raw application_train.csv for human-readable segment filters.

    Returns the unmodified 307,511 x 122 DataFrame (TARGET column + raw
    categoricals + raw AMT_* and DAYS_* columns).
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Re-extract the dataset or pass path=..."
        )
    return pd.read_csv(RAW_CSV)


@st.cache_data
def load_importance_table() -> pd.DataFrame:
    """Parse ``reports/feature_importance.md`` into a DataFrame.

    The file uses GitHub-flavored markdown tables with 7 columns:
        Rank | Feature | XGB rank | RF rank | SHAP rank | mean_rank | combined_score
    We extract rows whose first column is an integer rank.

    Returns
    -------
    pd.DataFrame
        Columns: [rank, feature, xgb_rank, rf_rank, shap_rank, mean_rank, combined_score].
        Sorted ascending by rank.
    """
    if not IMPORTANCE_MD.exists():
        raise FileNotFoundError(f"{IMPORTANCE_MD} not found.")
    text = IMPORTANCE_MD.read_text(encoding="utf-8")

    rows: list[tuple[int, str, int, int, int, float, float]] = []
    # Match markdown table rows: leading "|" + 7 pipe-separated cells.
    # Feature name may be wrapped in backticks (e.g. `EXT_SOURCE_MEAN`).
    pattern = re.compile(
        r"^\s*\|\s*(\d+)\s*\|\s*`?([A-Za-z0-9_ /]+)`?\s*\|\s*"
        r"(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        rank = int(match.group(1))
        feature = match.group(2).strip()
        xgb_rank = int(match.group(3))
        rf_rank = int(match.group(4))
        shap_rank = int(match.group(5))
        mean_rank = float(match.group(6))
        combined = float(match.group(7))
        rows.append((rank, feature, xgb_rank, rf_rank, shap_rank, mean_rank, combined))

    if not rows:
        raise ValueError(
            f"No importance rows parsed from {IMPORTANCE_MD}. "
            "Format may have changed; check the markdown table."
        )

    return pd.DataFrame(
        rows,
        columns=["rank", "feature", "xgb_rank", "rf_rank", "shap_rank", "mean_rank", "combined_score"],
    )


@st.cache_data
def load_phase4_summary() -> str:
    """Return the raw markdown of reports/phase4_model_comparison.md.

    The Overview tab embeds this in an expander; the sidebar reads the
    model stats directly from a parsed constant in ``_theme.py``.
    """
    if not PHASE4_MD.exists():
        return "_Phase 4 report not found._"
    return PHASE4_MD.read_text(encoding="utf-8")


__all__ = [
    "load_clean",
    "load_top30",
    "load_raw",
    "load_importance_table",
    "load_phase4_summary",
    "REPO_ROOT",
]
