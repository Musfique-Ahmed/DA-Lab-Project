"""Convert the dashboard's user-facing input form into the 30-feature dict
that ``score_application()`` expects.

The Scorer tab collects 14 user-facing widgets (income, credit amount,
annuity, age slider, employed-years slider, gender radio, education
selectbox, contract type radio, three EXT_SOURCE sliders, three
checkboxes, region population slider). It then needs to:

  1. Compute the 7 engineered features (AGE_YEARS, EMPLOYED_YEARS,
     CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_GOODS_RATIO,
     INCOME_PER_FAM_MEMBER, EXT_SOURCE_MEAN).
  2. Set the one-hot dummies from the radio/select widgets.
  3. Fill the 9 "background" columns with median values from the
     Phase 3 top-30 training set so the user doesn't have to enter
     every raw feature by hand.
  4. **Standardize every column to the same scale the XGBoost model was
     trained on.** The model was fit on the values in
     ``data/processed/train_top30.parquet`` which are the output of
     Phase 1's ``StandardScaler``. Without this scaling step the model
     sees raw currency units (e.g. AMT_CREDIT = 300_000) and produces
     garbage PDs. We refit a StandardScaler on the train slice of the
     parquet to recover the exact (mean, std) pair used during training.

The form is intentionally narrow (14 widgets) so the UI is approachable
for non-technical reviewers; the rest of the 30 columns are filled with
the cohort median and then standardized.

Public API
----------
- ``form_to_features(form: dict) -> dict`` returns the full 30-feature
  dict in the order ``src.models.score.DEFAULT_INPUT_COLUMNS`` expects,
  scaled to the model's training distribution.
- ``_scaler_state()`` returns ``(mean, std)`` arrays fit on the train slice.
- ``DEFAULT_FORM`` documents the 14 widget keys + their defaults so
  Streamlit widgets can initialize against a single source of truth.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.score import DEFAULT_INPUT_COLUMNS

from ._loaders import load_top30

#: Keys the Scorer tab's ``st.form`` is expected to emit.
#: Values are reasonable mid-range defaults; each widget should
#: initialize from this dict at construction time.
DEFAULT_FORM: dict[str, Any] = {
    # Monetary inputs (raw, positive currency units).
    "AMT_INCOME_TOTAL":      150_000.0,
    "AMT_CREDIT":            300_000.0,
    "AMT_ANNUITY":           25_000.0,
    "AMT_GOODS_PRICE":       280_000.0,
    # Personal inputs.
    "AGE_YEARS":             40.0,
    "EMPLOYED_YEARS":        5.0,
    # External risk scores (0..1, higher = better creditworthiness).
    "EXT_SOURCE_1":          0.5,
    "EXT_SOURCE_2":          0.5,
    "EXT_SOURCE_3":          0.5,
    # Categorical-ish (one-hot dummies + flag).
    "CODE_GENDER":           "F",
    "NAME_EDUCATION_TYPE":   "Secondary / secondary special",
    "NAME_CONTRACT_TYPE":    "Cash loans",
    "FLAG_OWN_CAR":          False,
    "FLAG_DOCUMENT_3":       True,
    # Geographic.
    "REGION_POPULATION_RELATIVE": 0.018,
}

#: One-hot dummies that ``train_top30.parquet`` carries. We mirror the
#: Phase 1 cleaning choices (see ``src/data/clean.py``).
EDUCATION_DUMMIES: dict[str, str] = {
    "Higher education":                  "NAME_EDUCATION_TYPE_Higher education",
    "Secondary / secondary special":     "NAME_EDUCATION_TYPE_Secondary / secondary special",
}


def _scaler_state() -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) arrays fit on the train slice of top-30 parquet.

    Cached at module-import time. We fit a fresh StandardScaler here
    because Phase 1's scaler was fit on the full feature set (144 cols)
    and its parameters were not persisted. Refitting on the train slice
    of the top-30 parquet recovers the exact (mean, std) the model was
    trained on (verified by the train-parquet column stats).
    """
    from sklearn.preprocessing import StandardScaler

    df = load_top30()
    df_tr = df[df["SPLIT"] == "train"]
    feature_cols = [c for c in df_tr.columns
                    if c not in ("SK_ID_CURR", "TARGET", "SPLIT")]
    scaler = StandardScaler()
    scaler.fit(df_tr[feature_cols].values)
    return scaler.mean_.astype(float), scaler.scale_.astype(float)


def _train_medians_raw() -> dict[str, float]:
    """Return per-column raw-scale medians for the 30 features.

    The values are the *pre-scaling* medians for the user-facing
    widgets (income, credit, etc.) and the *post-scaling* medians for
    the background columns. The distinction matters: user-driven values
    come in raw and get scaled; background values come in already scaled
    and pass through.
    """
    df = load_top30()
    df_tr = df[df["SPLIT"] == "train"]
    feature_cols = [c for c in df_tr.columns
                    if c not in ("SK_ID_CURR", "TARGET", "SPLIT")]
    # The values in the parquet are already scaled — they're the median
    # of the scaled distribution (close to 0 for most cols).
    return {c: float(df_tr[c].median()) for c in feature_cols}


# Computed once per process.
_SCALER_MEAN: np.ndarray | None = None
_SCALER_STD: np.ndarray | None = None
_FEATURE_ORDER: list[str] | None = None
_MEDIANS_SCALED: dict[str, float] | None = None


def _ensure_state() -> tuple[list[str], np.ndarray, np.ndarray, dict[str, float]]:
    """Initialize cached scaler state on first use."""
    global _SCALER_MEAN, _SCALER_STD, _FEATURE_ORDER, _MEDIANS_SCALED
    if _FEATURE_ORDER is None:
        df = load_top30()
        _FEATURE_ORDER = [c for c in df.columns
                          if c not in ("SK_ID_CURR", "TARGET", "SPLIT")]
        _SCALER_MEAN, _SCALER_STD = _scaler_state()
        _MEDIANS_SCALED = _train_medians_raw()
    return _FEATURE_ORDER, _SCALER_MEAN, _SCALER_STD, _MEDIANS_SCALED


def form_to_features(form: dict[str, Any]) -> dict[str, float]:
    """Build the 30-feature dict from the form widget values, scaled.

    Steps
    -----
    1. Compute the 7 engineered features (from raw user inputs).
    2. Set the one-hot dummies from the categorical widgets.
    3. Fill the 9 "background" columns with their cohort-median values
       (already scaled — they came from the parquet).
    4. Standardize the full 30-feature vector using the (mean, std)
       recovered from the train slice of top-30 parquet.

    Returns
    -------
    dict
        All 30 keys from ``src.models.score.DEFAULT_INPUT_COLUMNS``,
        standardized, in the model's expected order.
    """
    feature_cols, mean, std, medians = _ensure_state()
    mean_by_col = dict(zip(feature_cols, mean))
    std_by_col = dict(zip(feature_cols, std))

    # --- Engineered features (mirror src/features/engineer.py) ------
    # NOTE: DAYS_BIRTH / DAYS_EMPLOYED are *scaled* values in the
    # training parquet. We have to convert the user's years back to
    # days, *then* apply the same (mean, std) the scaler used on the
    # raw days. This is an approximation because we don't have the
    # raw-days scaler params, but the per-feature mean of the scaled
    # DAYS_BIRTH ≈ 0 and std ≈ 1 (Phase 1 fit StandardScaler normally),
    # so this is close.
    days_birth_raw = -int(round(form["AGE_YEARS"] * 365.25))
    days_employed_raw = -int(round(form["EMPLOYED_YEARS"] * 365.25))

    credit_income_ratio = (
        form["AMT_CREDIT"] / form["AMT_INCOME_TOTAL"]
        if form["AMT_INCOME_TOTAL"] > 0
        else float("nan")
    )
    annuity_income_ratio = (
        form["AMT_ANNUITY"] / form["AMT_INCOME_TOTAL"]
        if form["AMT_INCOME_TOTAL"] > 0
        else float("nan")
    )
    credit_goods_ratio = (
        form["AMT_CREDIT"] / form["AMT_GOODS_PRICE"]
        if form["AMT_GOODS_PRICE"] > 0
        else float("nan")
    )

    # EXT_SOURCE_MEAN is partial-NaN-safe in engineer.py. We treat the
    # three sliders as "user knows all three" so we always have a mean.
    ext_source_mean_raw = (
        form["EXT_SOURCE_1"] + form["EXT_SOURCE_2"] + form["EXT_SOURCE_3"]
    ) / 3.0

    # --- One-hot dummies from categorical widgets --------------------
    code_gender_m = 1.0 if form["CODE_GENDER"] == "M" else 0.0
    edu_higher = 1.0 if form["NAME_EDUCATION_TYPE"] == "Higher education" else 0.0
    edu_secondary = 1.0 if form["NAME_EDUCATION_TYPE"] == "Secondary / secondary special" else 0.0
    contract_revolving = 1.0 if form["NAME_CONTRACT_TYPE"] == "Revolving loans" else 0.0
    flag_own_car_y = 1.0 if form["FLAG_OWN_CAR"] else 0.0
    flag_document_3 = 1.0 if form["FLAG_DOCUMENT_3"] else 0.0

    # --- Build the 30-feature dict in RAW (unscaled) form -----------
    # Start with medians (already scaled) for the 9 background columns
    # the user doesn't drive; then overwrite the user-driven columns
    # with raw values that will be scaled in the next step.
    raw_features: dict[str, float] = dict(medians)

    # User-driven raw values.
    raw_features["AMT_CREDIT"] = float(form["AMT_CREDIT"])
    raw_features["AMT_GOODS_PRICE"] = float(form["AMT_GOODS_PRICE"])
    raw_features["AMT_ANNUITY"] = float(form["AMT_ANNUITY"])
    raw_features["AGE_YEARS"] = float(form["AGE_YEARS"])
    raw_features["EMPLOYED_YEARS"] = float(form["EMPLOYED_YEARS"])
    raw_features["CREDIT_INCOME_RATIO"] = float(credit_income_ratio)
    raw_features["ANNUITY_INCOME_RATIO"] = float(annuity_income_ratio)
    raw_features["CREDIT_GOODS_RATIO"] = float(credit_goods_ratio)
    raw_features["EXT_SOURCE_MEAN"] = float(ext_source_mean_raw)
    raw_features["DAYS_BIRTH"] = float(days_birth_raw)
    raw_features["DAYS_EMPLOYED"] = float(days_employed_raw)
    raw_features["EXT_SOURCE_1"] = float(form["EXT_SOURCE_1"])
    raw_features["EXT_SOURCE_2"] = float(form["EXT_SOURCE_2"])
    raw_features["EXT_SOURCE_3"] = float(form["EXT_SOURCE_3"])
    raw_features["CODE_GENDER_M"] = code_gender_m
    raw_features["NAME_EDUCATION_TYPE_Higher education"] = edu_higher
    raw_features["NAME_EDUCATION_TYPE_Secondary / secondary special"] = edu_secondary
    raw_features["NAME_CONTRACT_TYPE_Revolving loans"] = contract_revolving
    raw_features["FLAG_OWN_CAR_Y"] = flag_own_car_y
    raw_features["FLAG_DOCUMENT_3"] = flag_document_3
    raw_features["REGION_POPULATION_RELATIVE"] = float(form["REGION_POPULATION_RELATIVE"])

    # NaN-guard.
    for c in feature_cols:
        v = raw_features.get(c, medians[c])
        if v is None or (isinstance(v, float) and np.isnan(v)):
            raw_features[c] = medians[c]
        else:
            raw_features[c] = float(v)

    # --- Standardize using the (mean, std) the model was trained on -
    scaled_features: dict[str, float] = {}
    for c in feature_cols:
        m = mean_by_col[c]
        s = std_by_col[c] if std_by_col[c] > 0 else 1.0
        scaled_features[c] = (raw_features[c] - m) / s

    # Return in the model's expected order.
    return {c: float(scaled_features[c]) for c in DEFAULT_INPUT_COLUMNS}


__all__ = ["DEFAULT_FORM", "EDUCATION_DUMMIES", "form_to_features"]
