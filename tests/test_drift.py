"""Tests for the drift module (src.drift).

These tests do NOT need the engineered parquet or the trained model.
They construct small synthetic samples that exercise PSI directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    DEFAULT_N_BINS,
    PSI_MODERATE_DRIFT,
    PSI_NO_DRIFT,
    _classify,
    _psi_for_feature,
    compute_psi_table,
)


# ---------------------------------------------------------------------------
# Unit tests on the PSI math itself.
# ---------------------------------------------------------------------------

def test_psi_is_zero_for_identical_samples():
    """Identical ref/cmp distributions should produce PSI ~ 0 (no drift)."""
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(20_000)
    psi, _ = _psi_for_feature(sample, sample, n_bins=DEFAULT_N_BINS)
    assert psi < 1e-9, f"PSI should be ~0 for identical samples, got {psi}"


def test_psi_is_small_for_random_split_of_same_distribution():
    """A random subsample from the same N(0,1) should show low PSI."""
    rng = np.random.default_rng(42)
    pop = rng.standard_normal(100_000)
    ref = pop[:80_000]
    cmp = pop[80_000:]
    psi, _ = _psi_for_feature(ref, cmp, n_bins=DEFAULT_N_BINS)
    assert psi < PSI_NO_DRIFT, (
        f"PSI for two random slices of the same distribution should "
        f"be < {PSI_NO_DRIFT:.2f}, got {psi:.4f}"
    )


def test_psi_detects_mean_shift():
    """A real mean shift should produce PSI > 0.25 (major drift)."""
    rng = np.random.default_rng(7)
    ref = rng.standard_normal(50_000)
    # Shift the comparison distribution by +1.5 standard deviations.
    cmp = ref + 1.5
    psi, _ = _psi_for_feature(ref, cmp, n_bins=DEFAULT_N_BINS)
    assert psi > PSI_MODERATE_DRIFT, (
        f"PSI for shifted distribution should be > {PSI_MODERATE_DRIFT:.2f}, "
        f"got {psi:.4f}"
    )


def test_psi_detects_scale_shift():
    """Scale-only shifts (mean unchanged) should also produce meaningful PSI."""
    rng = np.random.default_rng(11)
    ref = rng.standard_normal(50_000)
    cmp = ref * 2.0  # same mean, 2x stdev
    psi, _ = _psi_for_feature(ref, cmp, n_bins=DEFAULT_N_BINS)
    assert psi > PSI_NO_DRIFT, (
        f"PSI for scale-shifted distribution should exceed "
        f"{PSI_NO_DRIFT:.2f}, got {psi:.4f}"
    )


def test_psi_handles_nan_values():
    """PSI must drop NaNs and still compute a finite value."""
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(10_000)
    cmp = rng.standard_normal(10_000)
    # Inject NaN into ~10% of rows.
    ref_with_nan = ref.copy()
    cmp_with_nan = cmp.copy()
    ref_with_nan[rng.choice(10_000, 1_000, replace=False)] = np.nan
    cmp_with_nan[rng.choice(10_000, 800, replace=False)] = np.nan
    psi, _ = _psi_for_feature(ref_with_nan, cmp_with_nan, n_bins=DEFAULT_N_BINS)
    assert np.isfinite(psi), f"PSI should be finite, got {psi}"
    # Same distribution (only NaNs differ by row, not value), so PSI should be small.
    assert psi < PSI_NO_DRIFT


def test_psi_handles_near_constant_feature():
    """A near-constant feature should not crash; PSI returns 0.0."""
    ref = np.full(5_000, 1.0)
    cmp = np.full(5_000, 1.0)
    psi, _ = _psi_for_feature(ref, cmp, n_bins=DEFAULT_N_BINS)
    assert psi == 0.0


def test_psi_rejects_empty_input():
    """Empty inputs must raise ValueError (loud failure, not silent NaN)."""
    with pytest.raises(ValueError):
        _psi_for_feature(np.array([]), np.array([1.0, 2.0]))


def test_classify_thresholds():
    assert _classify(0.05) == "no_drift"
    assert _classify(0.15) == "moderate_drift"
    assert _classify(0.30) == "major_drift"
    # Boundary at PSI_NO_DRIFT.
    assert _classify(PSI_NO_DRIFT - 1e-6) == "no_drift"
    assert _classify(PSI_NO_DRIFT) == "moderate_drift"


# ---------------------------------------------------------------------------
# Integration: compute_psi_table on a synthetic SPLIT-tagged frame.
# ---------------------------------------------------------------------------

def _make_split_frame(n: int = 5_000, seed: int = 0) -> pd.DataFrame:
    """Build a small DataFrame with a SPLIT column for testing."""
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    # Feature A: stable across splits (same distribution).
    a_train = rng.standard_normal(n_train)
    a_val = rng.standard_normal(n_val)
    a_test = rng.standard_normal(n_test)

    # Feature B: drifts between train and test (mean shift).
    b_train = rng.standard_normal(n_train)
    b_val = rng.standard_normal(n_val) + 0.8
    b_test = rng.standard_normal(n_test) + 1.0

    split = np.concatenate([
        np.full(n_train, "train"),
        np.full(n_val, "val"),
        np.full(n_test, "test"),
    ])
    return pd.DataFrame({
        "SPLIT": split,
        "stable_feature": np.concatenate([a_train, a_val, a_test]),
        "drifting_feature": np.concatenate([b_train, b_val, b_test]),
    })


def test_compute_psi_table_returns_one_row_per_feature():
    df = _make_split_frame()
    out = compute_psi_table(df, ["stable_feature", "drifting_feature"])
    assert len(out) == 2
    assert set(out.columns) >= {
        "feature", "psi_train_vs_val", "psi_train_vs_test",
        "max_psi", "drift_verdict", "n_bins_used",
    }


def test_compute_psi_table_ranking_puts_drifter_first():
    """The feature with a real shift should rank higher (max_psi) than stable."""
    df = _make_split_frame()
    out = compute_psi_table(df, ["stable_feature", "drifting_feature"])
    # Sorted descending by max_psi, so the index-0 row is the worst.
    worst = out.iloc[0]["feature"]
    assert worst == "drifting_feature"


def test_compute_psi_table_classifies_real_drift():
    df = _make_split_frame(n=20_000, seed=0)
    out = compute_psi_table(df, ["stable_feature", "drifting_feature"])
    stable = out.loc[out["feature"] == "stable_feature"].iloc[0]
    drifter = out.loc[out["feature"] == "drifting_feature"].iloc[0]
    assert stable["drift_verdict"] == "no_drift"
    # mean shift by +1 stddev on 15k rows should beat the 0.10 threshold
    assert drifter["max_psi"] > PSI_NO_DRIFT


def test_compute_psi_table_raises_on_missing_split_column():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        compute_psi_table(df, ["x"])


def test_compute_psi_table_raises_on_missing_feature():
    df = _make_split_frame()
    with pytest.raises(ValueError):
        compute_psi_table(df, ["nonexistent_feature"])
