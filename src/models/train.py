"""Train each of the 4 candidate models on (X_tr, y_tr) -> eval on (X_val, y_val).

Every train_* function returns (model_or_wrapper, metrics_dict) so the
caller can pick the best by val AUC. `metrics_dict` matches the shape
produced by `evaluate.compute_metrics` so it slots straight into the
comparison table.

Imbalance strategies (per Phase 4 plan):
  - "balanced"  : class_weight='balanced' (LR, RF) or scale_pos_weight=neg/pos (XGB)
  - "smote"     : SMOTE(k=5, random_state=42) wrapped inside an imblearn Pipeline
  - "pos_weight": pass `pos_weight=neg/pos` to MLPWrapper.fit (BCEWithLogitsLoss)
  - "none"      : no re-weighting (baseline sanity check)

Each function also reports the `pos_count` / `neg_count` / `scale_pos_weight`
it used so the notebook can show the exact numbers in the imbalance table.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.evaluate import compute_metrics, pick_threshold
from src.models.mlp import DEFAULT_RANDOM_STATE, MLPWrapper

# Default seed for all sklearn/xgb estimators (MLPWrapper has its own).
DEFAULT_SEED: int = DEFAULT_RANDOM_STATE


# ---------------------------------------------------------------------------
# Helper: wrap an estimator in a Pipeline([SMOTE, estimator]) for the
# `imbalance='smote'` case. We never call SMOTE outside a pipeline.
# ---------------------------------------------------------------------------
def _smote_pipeline(estimator, *, random_state: int = DEFAULT_SEED) -> ImbPipeline:
    """Return an imblearn Pipeline: SMOTE(k=5) -> estimator."""
    return ImbPipeline([
        ("smote", SMOTE(k_neighbors=5, random_state=random_state)),
        ("est", estimator),
    ])


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """Return neg/pos for use as XGBoost scale_pos_weight."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return n_neg / max(n_pos, 1)


def _eval(model, X_val: np.ndarray, y_val: np.ndarray, threshold: float) -> dict[str, Any]:
    """Run predict_proba + compute_metrics in the standard shape."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val)[:, 1]
    else:  # XGB sometimes exposes only `predict` with `output_margin=False`
        proba = np.asarray(model.predict(X_val), dtype=float)
    m = compute_metrics(y_val, proba, threshold=threshold)
    return m


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
def train_logreg(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "balanced",
    C: float = 1.0,
    threshold: float = 0.5,
    random_state: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Fit an LR on (X_tr, y_tr) and evaluate on (X_val, y_val).

    Note: the top-30 engineered features include 3 ratio columns with NaN
    (CREDIT_GOODS_RATIO, CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO) where
    the denominator was zero. We prepend a `SimpleImputer(strategy="median")`
    to the pipeline so LR (which doesn't accept NaN) can still fit.
    Tree models (RF, XGB) handle NaN natively and ignore this.
    """
    from sklearn.impute import SimpleImputer

    base = LogisticRegression(
        max_iter=200, C=C, solver="lbfgs",
        n_jobs=-1, random_state=random_state,
    )
    if imbalance == "balanced":
        est = base.set_params(class_weight="balanced")
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", est),
        ])
    elif imbalance == "smote":
        est = base
        # SMOTE cannot synthesize NaN rows, so impute first.
        pipe = _smote_pipeline(est)
        # Prepend an impute step by replacing the standard pipeline.
        pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(k_neighbors=5, random_state=random_state)),
            ("est", est),
        ])
    elif imbalance == "none":
        est = base
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", est),
        ])
    else:
        raise ValueError(f"Unknown imbalance: {imbalance!r}")
    pipe.fit(X_tr, y_tr)
    metrics = _eval(pipe, X_val, y_val, threshold)
    metrics["config"] = {"imbalance": imbalance, "C": C, "threshold": threshold}
    return pipe, metrics


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def train_random_forest(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "balanced",
    n_estimators: int = 300,
    max_depth: int | None = 12,
    threshold: float = 0.5,
    random_state: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Fit a RandomForest. Class imbalance is handled by class_weight."""
    if imbalance == "balanced":
        cw = "balanced_subsample"
    elif imbalance == "none":
        cw = None
    else:
        raise ValueError(
            f"RF imbalance must be 'balanced' or 'none'; got {imbalance!r}. "
            "SMOTE on a tree forest is dominated by class-weighting — keep it simple."
        )
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=20, class_weight=cw,
        n_jobs=-1, random_state=random_state,
    )
    rf.fit(X_tr, y_tr)
    metrics = _eval(rf, X_val, y_val, threshold)
    metrics["config"] = {
        "imbalance": imbalance, "n_estimators": n_estimators,
        "max_depth": max_depth, "threshold": threshold,
    }
    return rf, metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "balanced",
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    threshold: float = 0.5,
    random_state: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Fit an XGBoost classifier.

    For "balanced" we use `scale_pos_weight = neg/pos`. For "smote" we
    wrap SMOTE+kNN inside an imblearn Pipeline so the synthetic samples
    never see validation.
    """
    spw = _compute_scale_pos_weight(y_tr) if imbalance == "balanced" else 1.0
    if imbalance == "smote":
        from sklearn.impute import SimpleImputer
        est = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss", tree_method="hist",
            random_state=random_state, n_jobs=-1,
            scale_pos_weight=1.0,  # balanced by SMOTE already
        )
        # Impute BEFORE SMOTE; SMOTE cannot synthesize from NaN rows.
        model = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(k_neighbors=5, random_state=random_state)),
            ("est", est),
        ])
        model.fit(X_tr, y_tr)
    elif imbalance in ("balanced", "none"):
        xgb = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, scale_pos_weight=spw,
            eval_metric="logloss", tree_method="hist",
            random_state=random_state, n_jobs=-1,
        )
        xgb.fit(X_tr, y_tr)
        model = xgb
    else:
        raise ValueError(f"Unknown imbalance: {imbalance!r}")
    metrics = _eval(model, X_val, y_val, threshold)
    metrics["config"] = {
        "imbalance": imbalance, "n_estimators": n_estimators,
        "max_depth": max_depth, "learning_rate": learning_rate,
        "threshold": threshold,
        "scale_pos_weight": spw if imbalance == "balanced" else 1.0,
    }
    return model, metrics


# ---------------------------------------------------------------------------
# MLP (PyTorch)
# ---------------------------------------------------------------------------
def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    imbalance: str = "pos_weight",
    hidden: tuple[int, ...] = (128, 64),
    dropout: float = 0.3,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 2048,
    patience: int = 3,
    threshold: float = 0.5,
    random_state: int = DEFAULT_SEED,
) -> tuple[MLPWrapper, dict[str, Any]]:
    """Fit an MLPWrapper.

    For "pos_weight" we pass `pos_weight = neg/pos` to BCEWithLogitsLoss.
    For "none" we use unweighted loss.
    """
    pos_w: float | None
    if imbalance == "pos_weight":
        pos_w = _compute_scale_pos_weight(y_tr)
    elif imbalance == "none":
        pos_w = None
    else:
        raise ValueError(f"MLP imbalance must be 'pos_weight' or 'none'; got {imbalance!r}")

    wrapper = MLPWrapper.fit(
        X_tr, y_tr, X_val, y_val,
        hidden=hidden, dropout=dropout, epochs=epochs, lr=lr,
        batch_size=batch_size, patience=patience,
        random_state=random_state, pos_weight=pos_w,
    )
    metrics = _eval(wrapper, X_val, y_val, threshold)
    metrics["config"] = {
        "imbalance": imbalance, "hidden": list(hidden), "dropout": dropout,
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "patience": patience,
        "threshold": threshold,
        "pos_weight": pos_w,
        "best_epoch": wrapper.best_epoch_,
        "best_val_auc": wrapper.best_val_auc_,
    }
    return wrapper, metrics


# ---------------------------------------------------------------------------
# Threshold picker shared by all 4 (used by `tune.py` after each fit).
# ---------------------------------------------------------------------------
def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Pick Youden's J if it's >= J(0.5) + 0.20; otherwise return 0.5.

    Returns (threshold, youden_J_at_that_threshold).
    """
    t_default, j_default = pick_threshold(y_true, y_proba, strategy="fixed_0_5")
    t_youden, j_youden = pick_threshold(y_true, y_proba, strategy="youden")
    if j_youden - j_default >= 0.20:
        return t_youden, j_youden
    return t_default, j_default


__all__ = [
    "train_logreg",
    "train_random_forest",
    "train_xgboost",
    "train_mlp",
    "choose_threshold",
]