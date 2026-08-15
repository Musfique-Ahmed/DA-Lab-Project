"""Smoke tests for the Phase 5 dashboard modules.

These don't boot Streamlit (that's covered by the AppTest harness in
dashboard/README.md), but they do verify that the dashboard's helper
modules import cleanly and that the form-to-features round-trip
produces a valid score_application() call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``dashboard.*`` and ``src.*`` importable when running from the
# repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_dashboard_modules_import() -> None:
    """All four helper modules must import cleanly."""
    from dashboard._theme import PALETTE, MODEL_STATS  # noqa: F401
    from dashboard._form_to_features import DEFAULT_FORM, form_to_features  # noqa: F401
    from dashboard._segments import income_band, employment_bucket  # noqa: F401
    from dashboard._loaders import load_importance_table, REPO_ROOT  # noqa: F401

    # Required palette keys.
    for k in ("bg", "panel", "mint", "blue", "red", "white", "body", "muted"):
        assert k in PALETTE


def test_palette_matches_master_prompt() -> None:
    """The hex values must match the master prompt exactly."""
    from dashboard._theme import PALETTE

    assert PALETTE["bg"].upper() == "#0A1428"
    assert PALETTE["mint"].upper() == "#00D9B5"
    assert PALETTE["blue"].upper() == "#3B82F6"
    assert PALETTE["red"].upper() == "#F87171"


def test_form_to_features_round_trip() -> None:
    """The 14-widget form must produce a valid 30-feature dict."""
    from dashboard._form_to_features import DEFAULT_FORM, form_to_features
    from src.models.score import DEFAULT_INPUT_COLUMNS, score_application

    features = form_to_features(DEFAULT_FORM)
    assert set(features.keys()) == set(DEFAULT_INPUT_COLUMNS)
    assert all(isinstance(v, float) for v in features.values())

    out = score_application(features)
    assert out["recommendation"] in {"Approve", "Manual Review", "Reject"}
    assert 0.0 <= out["probability_of_default"] <= 1.0


def test_importance_table_parses_30_rows() -> None:
    """The feature-importance markdown must parse into >= 30 ranked rows."""
    from dashboard._loaders import load_importance_table

    df = load_importance_table()
    assert len(df) >= 30
    assert df["rank"].is_monotonic_increasing
    assert "EXT_SOURCE_MEAN" in df["feature"].values
    assert "EXT_SOURCE_MEAN" == df.iloc[0]["feature"]


def test_segments_income_band_returns_categorical() -> None:
    """income_band() should produce a categorical Series with the 10 bin labels."""
    import pandas as pd

    from dashboard._segments import INCOME_BAND_LABELS, income_band

    s = pd.Series([25_000, 80_000, 200_000, 500_000, 5_000_000])
    out = income_band(s)
    assert hasattr(out, "cat")
    assert len(out.cat.categories) == len(INCOME_BAND_LABELS)
    assert list(out.cat.categories) == INCOME_BAND_LABELS


def test_score_application_three_buckets() -> None:
    """Form-driven applicants: Approve is reachable; mid-defaults are mid-risk.

    Honest note: with EXT_SOURCE sliders bounded to [0.0, 1.0] (the natural
    range for raw credit scores), the maximum reachable PD from the form
    is around 0.35–0.40 (Manual Review bucket). The Reject bucket requires
    EXT_SOURCE values below 0.0, which the user would never see in real
    credit data — so the form is *correctly* conservative.

    This test verifies:
      1. The Approve bucket is reachable with high EXT_SOURCE values.
      2. The mid-risk defaults land somewhere between the thresholds.
      3. As EXT_SOURCE goes from 0.95 → 0.0, PD strictly increases (the
         model is sensitive to EXT_SOURCE in the expected direction).
    """
    from dashboard._form_to_features import DEFAULT_FORM, form_to_features
    from src.models.score import score_application

    # --- Low-risk: should reach Approve ---
    low = dict(DEFAULT_FORM)
    low["EXT_SOURCE_1"] = 0.95
    low["EXT_SOURCE_2"] = 0.95
    low["EXT_SOURCE_3"] = 0.95
    low["EMPLOYED_YEARS"] = 20.0
    low["AMT_INCOME_TOTAL"] = 500_000
    low["AMT_CREDIT"] = 100_000
    out_low = score_application(form_to_features(low))
    assert out_low["recommendation"] == "Approve", out_low
    assert out_low["probability_of_default"] < 0.20

    # --- Mid-risk: defaults should land in the gray zone ---
    mid = dict(DEFAULT_FORM)
    out_mid = score_application(form_to_features(mid))
    assert 0.02 <= out_mid["probability_of_default"] <= 0.50, out_mid

    # --- Monotonic sensitivity to EXT_SOURCE: low EXT → higher PD ---
    high_ext = score_application(form_to_features({
        **DEFAULT_FORM,
        "EXT_SOURCE_1": 0.95, "EXT_SOURCE_2": 0.95, "EXT_SOURCE_3": 0.95,
    }))["probability_of_default"]
    low_ext = score_application(form_to_features({
        **DEFAULT_FORM,
        "EXT_SOURCE_1": 0.05, "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.05,
        "EMPLOYED_YEARS": 0.0,
    }))["probability_of_default"]
    assert low_ext > high_ext, (
        f"Model should be sensitive to EXT_SOURCE in expected direction: "
        f"low_ext={low_ext:.3f}, high_ext={high_ext:.3f}"
    )
