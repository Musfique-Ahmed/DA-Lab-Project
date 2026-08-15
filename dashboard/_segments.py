"""Segment analysis helpers for the Phase 5 dashboard.

Mirrors the binning + default-rate logic from Phase 2's EDA notebook
(notebooks/01_eda.ipynb) so the dashboard and the notebook report the
same numbers. The default-rate-by-segment aggregation includes a
binomial-normal 95% CI so the chart's error bars are honest.

Public API
----------
- ``income_band(s: pd.Series) -> pd.Categorical``
- ``employment_bucket(s: pd.Series) -> pd.Categorical``
- ``default_rate_by_segment(df, col, *, min_n: int = 100) -> pd.DataFrame``
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Income decile edges match Phase 2 (computed via pd.qcut on AMT_INCOME_TOTAL).
# Hard-coded so the dashboard doesn't depend on a recomputed cut; the
# underlying values are from the raw train slice (median income ~ 147,150).
INCOME_DECILE_EDGES: list[float] = [
    25_650, 81_000, 99_000, 112_500, 135_000,
    147_150, 162_000, 180_000, 225_000, 270_000, 117_000_000,
]
INCOME_BAND_LABELS: list[str] = [
    "Q1 (<81k)", "Q2", "Q3", "Q4", "Q5",
    "Q6", "Q7", "Q8", "Q9", "Q10 (270k+)",
]

# Employment-length buckets (years), mirroring Phase 2's cut.
EMPLOYMENT_BINS: list[float] = [-1, 1, 3, 5, 10, 100]
EMPLOYMENT_LABELS: list[str] = ["<1 yr", "1–3 yrs", "3–5 yrs", "5–10 yrs", "10+ yrs"]

# Credit-to-income deciles — kept as a helper but not used by the dashboard's
# default segment selector (income band covers similar ground more clearly).
CTI_DECILE_EDGES: list[float] | None = None


def income_band(s: pd.Series) -> pd.Categorical:
    """Bucket AMT_INCOME_TOTAL into the 10 Phase 2 decile bands.

    Out-of-range values land in the nearest band (``right=True`` is too
    restrictive given the 117M upper edge; we clip instead).
    """
    clipped = s.clip(lower=INCOME_DECILE_EDGES[0], upper=INCOME_DECILE_EDGES[-1])
    return pd.cut(
        clipped,
        bins=INCOME_DECILE_EDGES,
        labels=INCOME_BAND_LABELS,
        include_lowest=True,
    )


def employment_bucket(s: pd.Series) -> pd.Categorical:
    """Bucket employment-length years into the 5 Phase 2 bins.

    Accepts NaN — NaN rows land in their own NaN category (excluded by
    ``default_rate_by_segment`` via the ``min_n`` filter).
    """
    return pd.cut(
        s,
        bins=EMPLOYMENT_BINS,
        labels=EMPLOYMENT_LABELS,
        include_lowest=True,
    )


def default_rate_by_segment(
    df: pd.DataFrame,
    col: str,
    *,
    min_n: int = 100,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Compute default rate + 95% CI per segment.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a binary ``TARGET`` column.
    col : str
        Column whose unique values define the segments (string or
        categorical; numeric values are converted via ``astype(str)``).
    min_n : int
        Segments with fewer than ``min_n`` rows are dropped. Default
        100 keeps the binomial-normal CI sane.
    segment_col : str | None
        If given, use this as the segment label column instead of the
        raw ``col`` (e.g. when the caller has already bucketed into
        income bands via ``income_band``).

    Returns
    -------
    pd.DataFrame
        Columns: [segment, n, default_rate, ci_lo, ci_hi].
        Sorted by descending default_rate.
    """
    label_col = segment_col if segment_col is not None else col
    work = df[[col, "TARGET"]].copy()
    work[label_col] = work[col].astype(str)
    agg = (
        work.groupby(label_col, dropna=False)
            .agg(n=("TARGET", "size"), defaults=("TARGET", "sum"))
            .reset_index()
    )
    agg = agg[agg["n"] >= min_n].copy()
    agg["default_rate"] = agg["defaults"] / agg["n"]
    # 95% binomial-normal CI: 1.96 * sqrt(p * (1-p) / n)
    z = 1.96
    se = np.sqrt(agg["default_rate"] * (1 - agg["default_rate"]) / agg["n"])
    agg["ci_lo"] = (agg["default_rate"] - z * se).clip(lower=0)
    agg["ci_hi"] = (agg["default_rate"] + z * se).clip(upper=1)
    agg = agg.sort_values("default_rate", ascending=False).reset_index(drop=True)
    return agg[[label_col, "n", "default_rate", "ci_lo", "ci_hi"]]


# Mapping from segment selector value to (raw column, label column).
# Used by the Segment Analysis tab to map a multiselect choice into the
# columns the dashboard actually wants to group by.
SEGMENT_OPTIONS: dict[str, dict[str, str]] = {
    "Income band (deciles)": {"raw": "AMT_INCOME_TOTAL", "label": "INCOME_BAND"},
    "Employment length":    {"raw": "DAYS_EMPLOYED", "label": "EMPLOYED_BUCKET"},
    "Education":            {"raw": "NAME_EDUCATION_TYPE", "label": "NAME_EDUCATION_TYPE"},
    "Income type":          {"raw": "NAME_INCOME_TYPE", "label": "NAME_INCOME_TYPE"},
    "Contract type":        {"raw": "NAME_CONTRACT_TYPE", "label": "NAME_CONTRACT_TYPE"},
    "Gender":               {"raw": "CODE_GENDER", "label": "CODE_GENDER"},
    "Owns a car":           {"raw": "FLAG_OWN_CAR", "label": "FLAG_OWN_CAR"},
    "Owns realty":          {"raw": "FLAG_OWN_REALTY", "label": "FLAG_OWN_REALTY"},
    "Region rating":        {"raw": "REGION_RATING_CLIENT", "label": "REGION_RATING_CLIENT"},
}


def build_segment_column(df: pd.DataFrame, choice: str) -> tuple[pd.Series, str]:
    """Add the segment column to a copy of ``df`` and return (series, name).

    For choices that require bucketing (income band, employment bucket),
    this returns the derived column. For raw-categorical choices, returns
    the original column unchanged.
    """
    if choice == "Income band (deciles)":
        return income_band(df["AMT_INCOME_TOTAL"]), "INCOME_BAND"
    if choice == "Employment length":
        # Raw CSV stores DAYS_EMPLOYED (negative int). Convert to years.
        # Also map the 365243 sentinel to NaN so it doesn't pollute the
        # "<1 yr" bucket.
        days = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        years = (-days) / 365.25
        return employment_bucket(years), "EMPLOYED_BUCKET"
    # All others are raw columns.
    info = SEGMENT_OPTIONS[choice]
    return df[info["raw"]], info["label"]


__all__ = [
    "INCOME_DECILE_EDGES",
    "INCOME_BAND_LABELS",
    "EMPLOYMENT_BINS",
    "EMPLOYMENT_LABELS",
    "income_band",
    "employment_bucket",
    "default_rate_by_segment",
    "SEGMENT_OPTIONS",
    "build_segment_column",
]
