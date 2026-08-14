"""Tests for `src.models.score` (Phase 4).

These tests don't train a real model — they exercise the validation and
recommendation logic in `score.py` plus the underlying `_recommend`
helper. A full end-to-end score test requires a trained model artifact,
which lives in `models/` and is gitignored; that path is covered by the
Phase 4 notebook.

What we test:
  1. The 3-way recommendation policy at known thresholds.
  2. Input validation raises ValueError with the missing key name.
  3. End-to-end `score_application` on a model trained inside the test
     that exercises every code path without needing a pre-saved artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.score import (
    DEFAULT_INPUT_COLUMNS,
    THRESHOLD_APPROVE_MAX,
    THRESHOLD_REJECT_MIN,
    _recommend,
    score_application,
)


def test_recommend_below_threshold_is_approve() -> None:
    """p < 0.20 -> 'Approve'."""
    assert _recommend(0.05) == "Approve"
    assert _recommend(0.0) == "Approve"
    assert _recommend(THRESHOLD_APPROVE_MAX - 1e-9) == "Approve"


def test_recommend_above_threshold_is_reject() -> None:
    """p >= 0.50 -> 'Reject'."""
    assert _recommend(0.55) == "Reject"
    assert _recommend(0.99) == "Reject"
    assert _recommend(THRESHOLD_REJECT_MIN) == "Reject"


def test_recommend_in_between_is_manual_review() -> None:
    """0.20 <= p < 0.50 -> 'Manual Review'."""
    assert _recommend(0.20) == "Manual Review"
    assert _recommend(0.35) == "Manual Review"
    assert _recommend(0.4999) == "Manual Review"


def test_default_input_columns_count() -> None:
    """Phase 3's top-30 list has exactly 30 feature columns."""
    assert len(DEFAULT_INPUT_COLUMNS) == 30
    # Sorted for stability — duplicates must not silently appear.
    assert len(set(DEFAULT_INPUT_COLUMNS)) == 30


def _fitted_low_risk_model() -> None:
    """Train a tiny LR on fake data so `score_application` has an artifact.

    Mirrors what the Phase 4 notebook writes to `models/best_model.pkl`
    but doesn't touch disk unless invoked.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n = 500
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, len(DEFAULT_INPUT_COLUMNS)))
    # A simple rule: high EXT_SOURCE_MEAN (index 0) => low y.
    y = (X[:, 0] < -0.5).astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression()),
    ])
    pipe.fit(X, y)
    import src.models.score as score_mod
    score_mod._ARTIFACT = pipe
    score_mod._KIND = "sklearn"
    score_mod._INPUT_COLUMNS = list(DEFAULT_INPUT_COLUMNS)


def test_score_application_returns_valid_shape() -> None:
    """A valid input returns {probability, recommendation} with proper bounds."""
    _fitted_low_risk_model()
    applicant = {c: 0.0 for c in DEFAULT_INPUT_COLUMNS}
    out = score_application(applicant)
    assert set(out.keys()) == {"probability_of_default", "recommendation"}
    assert 0.0 <= out["probability_of_default"] <= 1.0
    assert out["recommendation"] in {"Approve", "Manual Review", "Reject"}


def test_score_application_rejects_missing_keys() -> None:
    """An input missing a required key raises ValueError with the key name."""
    _fitted_low_risk_model()
    applicant = {c: 0.0 for c in DEFAULT_INPUT_COLUMNS}
    # Drop the first key.
    dropped_key = DEFAULT_INPUT_COLUMNS[0]
    del applicant[dropped_key]
    with pytest.raises(ValueError, match=dropped_key):
        score_application(applicant)


def test_score_application_typed_input() -> None:
    """A non-dict input raises TypeError."""
    _fitted_low_risk_model()
    with pytest.raises(TypeError):
        score_application(["not", "a", "dict"])  # type: ignore[arg-type]


def test_recommendation_three_way_disjoint() -> None:
    """The three recommendation buckets partition the [0, 1] interval."""
    # Sample boundaries below/above.
    thresholds = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
    recs = {_recommend(t) for t in thresholds}
    assert recs == {"Approve", "Manual Review", "Reject"}
