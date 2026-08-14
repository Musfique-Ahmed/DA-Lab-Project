"""Light grid search for the 4 model families.

Each tune_* function:
  1. Trains a small grid (≤4 fits).
  2. Records per-trial metrics on val.
  3. Returns (best_model, best_metrics, trials_list).

Best is chosen by val AUC; ties broken by val F1 at the chosen threshold.

Why "light":
  - Per Phase 4 plan: ≤16 fits total across all 4 models.
  - Bigger grids (Optuna etc.) would dominate the budget and add infra.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.models.evaluate import compute_metrics
from src.models.train import (
    train_logreg,
    train_mlp,
    train_random_forest,
    train_xgboost,
)
from src.models.train import choose_threshold as _choose_threshold

# Tiny helper to apply the model's threshold + re-compute metrics.
def _apply_threshold(model: Any, X_val: np.ndarray, y_val: np.ndarray, threshold: float) -> dict[str, Any]:
    proba = model.predict_proba(X_val)[:, 1]
    return compute_metrics(y_val, proba, threshold=threshold)


def _pick_best(trials: list[dict], models: list[Any]) -> tuple[Any, dict[str, Any]]:
    """Pick the trial with highest val AUC; ties broken by F1."""
    best_idx = max(
        range(len(trials)),
        key=lambda i: (trials[i]["val_metrics"]["auc_roc"],
                       trials[i]["val_metrics"]["f1"]),
    )
    return models[best_idx], trials[best_idx]["val_metrics"]


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
def tune_logreg(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "balanced",
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Grid: C ∈ {0.1, 1.0, 10.0} × the given imbalance choice."""
    trials: list[dict[str, Any]] = []
    models: list[Any] = []
    for C in (0.1, 1.0, 10.0):
        m, metrics = train_logreg(
            X_tr, y_tr, X_val, y_val,
            imbalance=imbalance, C=C, threshold=0.5,
        )
        # Pick a better threshold based on this trial's val proba.
        proba = m.predict_proba(X_val)[:, 1]
        t, _ = _choose_threshold(y_val, proba)
        m2, metrics2 = train_logreg(
            X_tr, y_tr, X_val, y_val,
            imbalance=imbalance, C=C, threshold=t,
        )
        # Use m2 for the trial artifact since it has the chosen threshold.
        trials.append({
            "config": {"C": C, "imbalance": imbalance, "threshold": t},
            "val_metrics": metrics2,
        })
        models.append(m2)
    best_model, best_metrics = _pick_best(trials, models)
    return best_model, best_metrics, trials


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def tune_random_forest(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "balanced",
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Grid: max_depth ∈ {8, 12, None} (n_estimators fixed at 300)."""
    trials: list[dict[str, Any]] = []
    models: list[Any] = []
    for depth in (8, 12, None):
        m, metrics = train_random_forest(
            X_tr, y_tr, X_val, y_val,
            imbalance=imbalance, n_estimators=300, max_depth=depth, threshold=0.5,
        )
        proba = m.predict_proba(X_val)[:, 1]
        t, _ = _choose_threshold(y_val, proba)
        m2, metrics2 = train_random_forest(
            X_tr, y_tr, X_val, y_val,
            imbalance=imbalance, n_estimators=300, max_depth=depth, threshold=t,
        )
        trials.append({
            "config": {"n_estimators": 300, "max_depth": depth,
                       "imbalance": imbalance, "threshold": t},
            "val_metrics": metrics2,
        })
        models.append(m2)
    best_model, best_metrics = _pick_best(trials, models)
    return best_model, best_metrics, trials


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def tune_xgboost(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "balanced",
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Grid: max_depth ∈ {4, 6}, lr ∈ {0.05, 0.1} (n_estimators fixed)."""
    trials: list[dict[str, Any]] = []
    models: list[Any] = []
    for depth in (4, 6):
        for lr_val in (0.05, 0.1):
            m, metrics = train_xgboost(
                X_tr, y_tr, X_val, y_val,
                imbalance=imbalance,
                n_estimators=300, max_depth=depth, learning_rate=lr_val,
                threshold=0.5,
            )
            proba = m.predict_proba(X_val)[:, 1]
            t, _ = _choose_threshold(y_val, proba)
            m2, metrics2 = train_xgboost(
                X_tr, y_tr, X_val, y_val,
                imbalance=imbalance,
                n_estimators=300, max_depth=depth, learning_rate=lr_val,
                threshold=t,
            )
            trials.append({
                "config": {"n_estimators": 300, "max_depth": depth,
                           "learning_rate": lr_val, "imbalance": imbalance,
                           "threshold": t},
                "val_metrics": metrics2,
            })
            models.append(m2)
    best_model, best_metrics = _pick_best(trials, models)
    return best_model, best_metrics, trials


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def tune_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "pos_weight",
    epochs: int = 12,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Grid: 2 hidden tuples — (64, 32) and (128, 64)."""
    trials: list[dict[str, Any]] = []
    models: list[Any] = []
    for hidden in ((64, 32), (128, 64)):
        m, metrics = train_mlp(
            X_tr, y_tr, X_val, y_val,
            imbalance=imbalance, hidden=hidden, dropout=0.3,
            epochs=epochs, lr=1e-3, batch_size=2048, patience=3,
            threshold=0.5,
        )
        proba = m.predict_proba(X_val)[:, 1]
        t, _ = _choose_threshold(y_val, proba)
        # Refit at chosen threshold (MLP doesn't care about the metric
        # threshold; we just recompute metrics for the table).
        m2, metrics2 = train_mlp(
            X_tr, y_tr, X_val, y_val,
            imbalance=imbalance, hidden=hidden, dropout=0.3,
            epochs=epochs, lr=1e-3, batch_size=2048, patience=3,
            threshold=t,
        )
        trials.append({
            "config": {"hidden": list(hidden), "imbalance": imbalance,
                       "threshold": t, "best_epoch": m.best_epoch_,
                       "best_val_auc": m.best_val_auc_},
            "val_metrics": metrics2,
        })
        models.append(m2)
    best_model, best_metrics = _pick_best(trials, models)
    return best_model, best_metrics, trials


__all__ = [
    "tune_logreg",
    "tune_random_forest",
    "tune_xgboost",
    "tune_mlp",
]