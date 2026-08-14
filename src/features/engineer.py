"""Derived-feature engineering for the cleaned application_train dataset.

These are pure functions: they take a DataFrame and return a new DataFrame
with engineered columns appended. No I/O, no side effects on the input.

The 7 engineered features (Phase 3):

  AGE_YEARS               = -DAYS_BIRTH / 365.25
  EMPLOYED_YEARS          = -DAYS_EMPLOYED / 365.25        (NaN for sentinels)
  CREDIT_INCOME_RATIO     = AMT_CREDIT / AMT_INCOME_TOTAL  (NaN if income = 0)
  ANNUITY_INCOME_RATIO    = AMT_ANNUITY / AMT_INCOME_TOTAL (NaN if income = 0)
  CREDIT_GOODS_RATIO      = AMT_CREDIT / AMT_GOODS_PRICE   (NaN if goods = 0)
  INCOME_PER_FAM_MEMBER   = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS (NaN if <= 0)
  EXT_SOURCE_MEAN         = mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
                            ignoring NaNs

Why these?
  - AGE_YEARS / EMPLOYED_YEARS: directly human-readable versions of the
    DAYS_ columns (already used in Phase 2 EDA).
  - The three ratios capture leverage / debt-service / down-payment signals
    that aren't visible in the raw amounts.
  - INCOME_PER_FAM_MEMBER proxies household-level affordability.
  - EXT_SOURCE_MEAN is the single most predictive aggregate (Phase 2 H1);
    adding it explicitly makes it survive any downstream feature selection.

Why we don't impute NaNs here:
  - Tree-based models (XGBoost, RF) handle NaN natively.
  - Linear models can impute downstream without leaking (LR does median
    imputation on the train slice only).
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

#: Names of the engineered columns this module adds.
ENGINEERED_COLUMNS: Final[tuple[str, ...]] = (
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "INCOME_PER_FAM_MEMBER",
    "EXT_SOURCE_MEAN",
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise numerator / denominator with NaN where denominator <= 0 or NaN.

    Avoids RuntimeWarning and produces NaN (consistent with the rest of
    the cleaning pipeline) when the denominator is non-positive.
    """
    num = numerator.astype(float)
    den = denominator.astype(float)
    return np.where(den > 0, num / den, np.nan)


def _partial_mean(*series: pd.Series) -> pd.Series:
    """Row-wise mean across series, ignoring NaN values.

    A row with NaN in one column but valid values in the others still gets
    a mean of the valid values. A row with all-NaN gets NaN.
    """
    stacked = pd.concat(series, axis=1)
    return stacked.mean(axis=1, skipna=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 7 engineered columns to a copy of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned application_train dataframe (must already include
        ``DAYS_BIRTH``, ``DAYS_EMPLOYED``, ``AMT_CREDIT``,
        ``AMT_INCOME_TOTAL``, ``AMT_ANNUITY``, ``AMT_GOODS_PRICE``,
        ``CNT_FAM_MEMBERS``, and the three ``EXT_SOURCE_*`` columns).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original columns plus the 7 engineered
        columns appended at the end. The input dataframe is not mutated.
    """
    out = df.copy()

    # 1 & 2. Time-derived (DAYS_ are stored as negative numbers; convert).
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"] / 365.25

    # 3, 4, 5. Ratios involving amounts.
    out["CREDIT_INCOME_RATIO"] = _safe_divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["ANNUITY_INCOME_RATIO"] = _safe_divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_GOODS_RATIO"] = _safe_divide(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])

    # 6. Per-capita income.
    out["INCOME_PER_FAM_MEMBER"] = _safe_divide(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])

    # 7. EXT_SOURCE mean, partial-NaN-safe.
    out["EXT_SOURCE_MEAN"] = _partial_mean(
        out["EXT_SOURCE_1"], out["EXT_SOURCE_2"], out["EXT_SOURCE_3"]
    )

    return out


__all__ = ["engineer_features", "ENGINEERED_COLUMNS"]
