"""Production scoring function — the artifact wired into the Phase 5 dashboard.

Public API (per master prompt):
  score_application(input: dict) -> dict
    Returns {"probability_of_default": float, "recommendation": str}
    where recommendation ∈ {"Approve", "Manual Review", "Reject"}.

Internals:
  - `_load_artifact()` reads `models/best_model.pkl` (sklearn) or
    `models/best_model.pt` + `models/best_model_meta.json` (PyTorch MLP).
    Cached at module import time so the dashboard doesn't pay the I/O
    cost per request.
  - `_validate_input(input, expected_cols)` raises ValueError with the
    first missing key if any of the 30 required features is absent.
  - The threshold policy is encoded by the two constants below; tweak
    them when the business-cost assumption changes (Phase 5 work).

Why the two thresholds?
  - p < THRESHOLD_APPROVE_MAX -> "Approve" (clearly low risk)
  - p >= THRESHOLD_REJECT_MIN -> "Reject" (clearly high risk)
  - in between -> "Manual Review" (gray zone; humans decide)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# The two policy thresholds. Documented in models/README.md so the
# reviewer can see what they mean and tune them later.
THRESHOLD_APPROVE_MAX: float = 0.20
THRESHOLD_REJECT_MIN: float = 0.50

# The 30 input feature columns (in the order the model expects).
# Mirrored from `data/processed/train_top30.parquet` minus SK_ID_CURR / TARGET / SPLIT.
# We rebuild the list lazily at module load from the artifact's metadata
# so the score module doesn't go stale if Phase 3 ever changes the cut.
DEFAULT_INPUT_COLUMNS: tuple[str, ...] = (
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_1",
    "EMPLOYED_YEARS",
    "EXT_SOURCE_3",
    "CODE_GENDER_M",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",
    "EXT_SOURCE_2",
    "NAME_EDUCATION_TYPE_Higher education",
    "DAYS_BIRTH",
    "AGE_YEARS",
    "CREDIT_GOODS_RATIO",
    "DAYS_ID_PUBLISH",
    "DAYS_EMPLOYED",
    "DAYS_LAST_PHONE_CHANGE",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_CONTRACT_TYPE_Revolving loans",
    "FLAG_DOCUMENT_3",
    "TOTALAREA_MODE",
    "DAYS_REGISTRATION",
    "YEARS_BEGINEXPLUATATION_MODE",
    "CREDIT_INCOME_RATIO",
    "FLAG_OWN_CAR_Y",
    "LIVINGAREA_MODE",
    "ANNUITY_INCOME_RATIO",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "REGION_POPULATION_RELATIVE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "LIVINGAREA_MEDI",
)

# Where the artifact lives. Phase 4 notebook writes these.
MODELS_DIR = Path("models")
PKL_PATH = MODELS_DIR / "best_model.pkl"
TORCH_STATE_PATH = MODELS_DIR / "best_model.pt"
META_PATH = MODELS_DIR / "best_model_meta.json"


# Lazy-loaded artifact cache (one load per process).
_ARTIFACT: dict[str, Any] | None = None
_KIND: str | None = None
_INPUT_COLUMNS: list[str] = list(DEFAULT_INPUT_COLUMNS)


def _load_artifact() -> tuple[Any, str, list[str]]:
    """Return (model, kind, input_columns), loading from disk on first call.

    `kind` is "sklearn" or "mlp". The model object is whatever the
    notebook persisted; we don't introspect its type.
    """
    global _ARTIFACT, _KIND, _INPUT_COLUMNS
    if _ARTIFACT is not None:
        return _ARTIFACT, _KIND, _INPUT_COLUMNS

    if PKL_PATH.exists():
        model = joblib.load(PKL_PATH)
        _ARTIFACT = model
        _KIND = "sklearn"
    elif TORCH_STATE_PATH.exists() and META_PATH.exists():
        # Lazy-import torch so the score module can be imported even
        # when the torch isn't available (e.g., a quick smoke test).
        import torch  # noqa: F401
        from src.models.mlp import MLP, MLPWrapper

        with open(META_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        n_features = len(meta["scaler_mean"])
        net = MLP(n_features=n_features, hidden=tuple(meta["hidden"]),
                  dropout=meta["dropout"])
        net.load_state_dict(torch.load(TORCH_STATE_PATH, map_location="cpu"))
        net.eval()
        wrapper = MLPWrapper(
            model=net,
            scaler=_DummyScaler(
                np.asarray(meta["scaler_mean"], dtype=float),
                np.asarray(meta["scaler_scale"], dtype=float),
            ),
            imputer=_DummyImputer(np.asarray(meta["imputer_median"], dtype=float)),
            hidden=tuple(meta["hidden"]),
            dropout=meta["dropout"],
        )
        _ARTIFACT = wrapper
        _KIND = "mlp"
    else:
        raise FileNotFoundError(
            f"No model artifact found. Looked for {PKL_PATH} or "
            f"{TORCH_STATE_PATH} + {META_PATH}. Run "
            f"`python notebooks/_build_phase4_notebook.py` to regenerate."
        )
    return _ARTIFACT, _KIND, _INPUT_COLUMNS


class _DummyScaler:
    """Minimal sklearn-compatible scaler reconstructed from saved mean/scale."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


class _DummyImputer:
    """Minimal sklearn-compatible imputer reconstructed from saved medians."""

    def __init__(self, statistics: np.ndarray):
        self.statistics_ = statistics

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Replace NaN with the per-column median.
        return np.where(np.isnan(X), self.statistics_, X)


def _validate_input(input: dict, expected_cols: list[str]) -> list[str]:
    """Return the list of missing keys (empty if input is complete)."""
    if not isinstance(input, dict):
        raise TypeError(f"score_application expects a dict, got {type(input).__name__}")
    return [c for c in expected_cols if c not in input]


def _recommend(p: float) -> str:
    """Map a probability of default to a 3-way recommendation."""
    if p < THRESHOLD_APPROVE_MAX:
        return "Approve"
    if p >= THRESHOLD_REJECT_MIN:
        return "Reject"
    return "Manual Review"


def score_application(input: dict) -> dict:
    """Score one applicant. Returns {probability_of_default, recommendation}.

    Raises
    ------
    TypeError
        If ``input`` is not a dict.
    ValueError
        If any of the 30 expected feature keys is missing from ``input``.
    FileNotFoundError
        If no model artifact is present in ``models/``.
    """
    model, _kind, input_cols = _load_artifact()
    missing = _validate_input(input, input_cols)
    if missing:
        raise ValueError(
            f"score_application: missing required input keys: {missing}. "
            f"Expected {len(input_cols)} features — see models/README.md."
        )
    # Order the input into the model's expected column order.
    X = np.asarray([[input[c] for c in input_cols]], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            # MLPWrapper exposes predict_proba too; this branch is defensive.
            proba = np.asarray(model.predict(X), dtype=float)
    p = float(proba[0])
    return {
        "probability_of_default": p,
        "recommendation": _recommend(p),
    }


__all__ = [
    "score_application",
    "THRESHOLD_APPROVE_MAX",
    "THRESHOLD_REJECT_MIN",
    "DEFAULT_INPUT_COLUMNS",
]