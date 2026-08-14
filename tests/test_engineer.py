"""Tests for src.features.engineer.

These tests verify the math of the 7 derived features. The cleaning pipeline
already guarantees the input columns exist; we only need to check that
engineer_features() applies the right formulas and handles edge cases
(NaN, division by zero) gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    ENGINEERED_COLUMNS,
    engineer_features,
)


def _minimal_df(**overrides) -> pd.DataFrame:
    """Build a single-row DataFrame with all the columns engineer_features needs."""
    row = {
        "DAYS_BIRTH":         -10_000.0,   # ~27.38 years
        "DAYS_EMPLOYED":      -2_000.0,    # ~5.48 years
        "AMT_INCOME_TOTAL":   120_000.0,
        "AMT_CREDIT":         240_000.0,   # CTI = 2.0
        "AMT_ANNUITY":        24_000.0,    # AIR = 0.20
        "AMT_GOODS_PRICE":    200_000.0,   # CGR = 1.20
        "CNT_FAM_MEMBERS":    4.0,         # per-cap = 30,000
        "EXT_SOURCE_1":       0.5,
        "EXT_SOURCE_2":       0.3,
        "EXT_SOURCE_3":       0.7,         # mean = 0.5
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_age_years_correct() -> None:
    df = engineer_features(_minimal_df())
    assert "AGE_YEARS" in df.columns
    assert df["AGE_YEARS"].iloc[0] == pytest.approx(27.38, abs=0.01)


def test_employed_years_handles_nan() -> None:
    """DAYS_EMPLOYED == NaN (post-cleaning sentinel rows) must yield NaN."""
    df = engineer_features(_minimal_df(DAYS_EMPLOYED=np.nan))
    assert np.isnan(df["EMPLOYED_YEARS"].iloc[0])


def test_credit_income_ratio_safe() -> None:
    """Division by zero (income = 0) produces NaN, not inf."""
    df = engineer_features(_minimal_df(AMT_INCOME_TOTAL=0.0))
    val = df["CREDIT_INCOME_RATIO"].iloc[0]
    assert np.isnan(val)
    assert not np.isinf(val)


def test_annuity_income_ratio_correct() -> None:
    df = engineer_features(_minimal_df())
    assert df["ANNUITY_INCOME_RATIO"].iloc[0] == pytest.approx(0.20, abs=1e-9)


def test_ext_source_mean_handles_partial_nan() -> None:
    """Mean of (0.5, NaN, 0.7) should be 0.6, not NaN or 0.4."""
    df = engineer_features(_minimal_df(EXT_SOURCE_2=np.nan))
    val = df["EXT_SOURCE_MEAN"].iloc[0]
    assert val == pytest.approx(0.6, abs=1e-9)


def test_engineer_features_returns_new_df() -> None:
    """Original df must not be mutated; new columns must be appended."""
    original = _minimal_df()
    n_cols_before = original.shape[1]
    out = engineer_features(original)
    # Original is intact.
    assert original.shape[1] == n_cols_before
    # Output has all 7 engineered columns appended.
    for col in ENGINEERED_COLUMNS:
        assert col in out.columns
    assert out.shape[1] == n_cols_before + len(ENGINEERED_COLUMNS)
    # Original df didn't gain engineered columns.
    for col in ENGINEERED_COLUMNS:
        assert col not in original.columns
