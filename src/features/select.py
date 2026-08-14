"""Feature-importance extraction and top-N selection.

This module is responsible for Phase 3's selection logic. It is *pure
analysis* — no I/O, no splitting. The caller passes in the train slice
and gets back DataFrames/matrices.

Three helpers:

  train_xgb_and_rf_importances(X, y, *, random_state=42)
      Train an XGBoost and a RandomForest on (X, y), return a DataFrame
      with feature, xgb_importance, rf_importance, mean_rank, combined_score.

  shap_values_xgb(X, *, sample_size=5000, random_state=42)
      Train an XGBoost on a `sample_size`-row subsample and return
      (shap_matrix, feature_names). TreeSHAP is fast on boosted trees,
      so 5k rows is enough for stable rankings.

  select_top_n(importance_df, shap_matrix, feature_names, *, n=30)
      Combine the two importance ranks with the SHAP rank, return the
      top-n features with their scores.

All three are deterministic (random_state pinned) so the notebook can
re-run them safely.
"""
from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Global default for the quick XGBoost used in importance extraction.
# Phase 4 will tune these properly.
XGB_N_ESTIMATORS: Final[int] = 300
XGB_MAX_DEPTH: Final[int] = 6
XGB_LEARNING_RATE: Final[float] = 0.1
# Inverse of the 92/8 class balance — passed to scale_pos_weight so the
# minority class isn't drowned out in importance calc.
XGB_SCALE_POS_WEIGHT: Final[float] = 92.0 / 8.0

RF_N_ESTIMATORS: Final[int] = 200
RF_MAX_DEPTH: Final[int] = 12
RF_MIN_SAMPLES_LEAF: Final[int] = 20

# Shared parallelism — both models default to all cores.
N_JOBS: Final[int] = -1


def _make_xgb(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
        n_jobs=N_JOBS,
        random_state=random_state,
        eval_metric="logloss",
        tree_method="hist",
    )


def _make_rf(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=random_state,
        class_weight="balanced",
    )


def train_xgb_and_rf_importances(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train XGBoost + RandomForest on the full feature set, return importances.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (one column per feature).
    y : pd.Series
        Binary target.
    random_state : int
        Seed for both models.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``xgb_importance``, ``rf_importance``,
        ``xgb_rank``, ``rf_rank``, ``mean_rank``, ``combined_score``.
        Rows are sorted by ``mean_rank`` ascending (rank 1 = most important).
    """
    feature_names = list(X.columns)

    xgb = _make_xgb(random_state)
    xgb.fit(X.values, y.values)
    xgb_imp = np.asarray(xgb.feature_importances_, dtype=float)

    rf = _make_rf(random_state)
    rf.fit(X.values, y.values)
    rf_imp = np.asarray(rf.feature_importances_, dtype=float)

    df = pd.DataFrame({
        "feature": feature_names,
        "xgb_importance": xgb_imp,
        "rf_importance": rf_imp,
    })
    # Rank (0 = most important). Use method='min' so ties get the same rank.
    df["xgb_rank"] = df["xgb_importance"].rank(ascending=False, method="min").astype(int)
    df["rf_rank"] = df["rf_importance"].rank(ascending=False, method="min").astype(int)
    df["mean_rank"] = (df["xgb_rank"] + df["rf_rank"]) / 2.0

    # Combined score: normalized mean of the inverse-rank, so higher = better.
    # This is monotone with mean_rank and lives in (0, 1].
    max_rank = len(df)
    df["combined_score"] = (
        ((max_rank - df["xgb_rank"]) / (max_rank - 1)
         + (max_rank - df["rf_rank"]) / (max_rank - 1)) / 2.0
    )

    return df.sort_values("mean_rank").reset_index(drop=True)


def shap_values_xgb(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    *,
    sample_size: int = 5_000,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """TreeSHAP values for an XGBoost model trained on a subsample of X.

    TreeSHAP is exact on tree ensembles and runs in O(TLD^2) per row, so
    5,000 rows on a 300-tree xgb with depth 6 finishes in well under a
    minute on modern hardware.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (will be subsampled internally).
    y : pd.Series | None
        Labels for the rows of ``X``. If ``None``, the X column named
        ``TARGET`` (or the last column) is used. If neither exists, a
        ``ValueError`` is raised — SHAP needs a real model, which needs
        real labels.
    sample_size : int
        Number of rows to subsample for SHAP.
    random_state : int
        Seed for the subsample.

    Returns
    -------
    (shap_values, feature_names)
        ``shap_values`` has shape (sample_size, n_features). Class index 1
        is used (the positive class).
    """
    import shap

    if y is None:
        if "TARGET" in X.columns:
            y = X["TARGET"]
            X = X.drop(columns=["TARGET"])
        elif isinstance(X, pd.DataFrame) and X.shape[1] >= 2:
            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
        else:
            raise ValueError(
                "shap_values_xgb needs real labels; pass y= or include a "
                "TARGET column in X."
            )

    feature_names = list(X.columns)
    if sample_size and sample_size < len(X):
        sub_idx = X.sample(n=sample_size, random_state=random_state).index
        sub_X = X.loc[sub_idx]
        sub_y = y.loc[sub_idx]
    else:
        sub_X = X
        sub_y = y

    xgb = _make_xgb(random_state)
    xgb.fit(sub_X.values, sub_y.values)
    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(sub_X.values)
    # For binary classification, shap returns either a (n, f) array or a
    # (n, f, 2) array. Normalize to (n, f) using the positive class.
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return np.asarray(sv), feature_names


def select_top_n(
    importance_df: pd.DataFrame,
    shap_matrix: np.ndarray,
    feature_names: list[str],
    *,
    n: int = 30,
) -> pd.DataFrame:
    """Combine importance ranks + SHAP rank, return the top-n features.

    The combined ranking is the mean of:
      - XGB rank (from the importance_df)
      - RF rank (from the importance_df)
      - SHAP rank (mean |SHAP| per feature over the sample)

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of ``train_xgb_and_rf_importances``.
    shap_matrix : np.ndarray
        Output of ``shap_values_xgb`` (shape n_samples x n_features).
    feature_names : list[str]
        Column names corresponding to ``shap_matrix`` columns.
    n : int
        Number of features to return.

    Returns
    -------
    pd.DataFrame
        Top-n features with columns: ``feature``, ``xgb_importance``,
        ``rf_importance``, ``shap_mean_abs``, ``mean_rank``,
        ``combined_score``. Sorted by ``mean_rank``.
    """
    if shap_matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_matrix has {shap_matrix.shape[1]} columns but "
            f"feature_names has {len(feature_names)} entries."
        )

    shap_mean_abs = np.abs(shap_matrix).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_mean_abs": shap_mean_abs,
    })
    shap_df["shap_rank"] = (
        shap_df["shap_mean_abs"].rank(ascending=False, method="min").astype(int)
    )

    merged = importance_df.merge(shap_df[["feature", "shap_mean_abs", "shap_rank"]], on="feature")
    merged["mean_rank"] = (merged["xgb_rank"] + merged["rf_rank"] + merged["shap_rank"]) / 3.0

    # Recompute combined_score over 3 ranks instead of 2.
    max_rank = len(merged)
    merged["combined_score"] = (
        (max_rank - merged["xgb_rank"]) / (max_rank - 1)
        + (max_rank - merged["rf_rank"]) / (max_rank - 1)
        + (max_rank - merged["shap_rank"]) / (max_rank - 1)
    ) / 3.0

    merged = merged.sort_values("mean_rank").reset_index(drop=True)
    return merged.head(n).copy()


__all__ = [
    "train_xgb_and_rf_importances",
    "shap_values_xgb",
    "select_top_n",
]
