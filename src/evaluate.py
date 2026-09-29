"""Evaluate stage — MLSD pipeline.

Loads the trained models and the engineered parquet, computes metrics
on the validation AND test slices, writes `reports/metrics.json` (the
DVC-tracked metric), and saves evaluation plots to
`reports/evaluation/`.

This is the `evaluate` stage of `dvc.yaml`.

Strict rule
-----------
This stage NEVER retrains. It only loads artifacts produced by the
`train` stage and computes metrics. The test set is touched only here
— the training stage has never seen it.

Outputs
-------
- reports/metrics.json                  (DVC metrics:)
- reports/evaluation/roc_curves.png
- reports/evaluation/pr_curves.png
- reports/evaluation/confusion_matrices.png

Metrics
-------
For both models (XGBoost + LogReg baseline), on both val and test:
  - roc_auc (primary)
  - pr_auc  (precision-recall AUC)
  - precision, recall, f1, accuracy at `params.yaml::evaluation.threshold`
  - confusion_matrix [TN, FP, FN, TP]

Usage
-----
    python src/evaluate.py
        [--params params.yaml]
        [--engineered data/processed/train_engineered.parquet]
        [--xgboost models/xgboost_model.pkl]
        [--baseline models/baseline_logistic_regression.pkl]
        [--feature-columns models/feature_columns.json]
        [--metrics-out reports/metrics.json]
        [--plots-dir reports/evaluation]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import auc as sk_auc

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_params(params_path: Path) -> dict:
    """Load the `evaluation` section of params.yaml with sensible fallbacks."""
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path}. "
            "Run `dvc repro` from the repo root, or pass --params."
        )
    with open(params_path, "r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}
    ev = params.get("evaluation", {})
    return {
        "primary_metric": str(ev.get("primary_metric", "roc_auc")),
        "threshold": float(ev.get("threshold", 0.5)),
        "random_state": int(params.get("data", {}).get("random_state", 42)),
    }


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X.values)[:, 1]
    return np.asarray(model.predict(X.values), dtype=float)


def compute_metrics_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict:
    """Compute the full metric set for one (model, slice) pair."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # PR AUC — trapezoidal rule over the precision-recall curve.
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = float(sk_auc(rec, prec))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def _plot_roc_curves(
    spec: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    """`spec` is a list of (label, y_true, y_proba)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y_true, y_proba in spec:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_pr_curves(
    spec: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    """Precision-recall curves — more informative for imbalanced data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y_true, y_proba in spec:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = float(sk_auc(rec, prec))
        ax.plot(rec, prec, label=f"{label} (PR-AUC={pr_auc:.4f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_confusion_matrices(
    matrices: dict[str, dict[str, int]],
    out_path: Path,
    title: str,
) -> None:
    """One confusion matrix per model, 1xN panel."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, cm) in zip(axes, matrices.items()):
        grid = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        im = ax.imshow(grid, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{grid[i, j]:,}",
                    ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black",
                    fontsize=14,
                )
        ax.set_title(
            f"{label}\nTN={cm['TN']:,}  FP={cm['FP']:,}\nFN={cm['FN']:,}  TP={cm['TP']:,}",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(
    *,
    engineered_parquet: Path,
    xgboost_path: Path,
    baseline_path: Path,
    feature_columns_path: Path,
    metrics_out: Path,
    plots_dir: Path,
    params: dict,
) -> dict:
    """Run the evaluate stage. Returns a small info dict for logging."""
    print(f"[evaluate] Loading engineered parquet from {engineered_parquet} ...")
    engineered = pd.read_parquet(engineered_parquet)

    feat = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    feature_cols: list[str] = feat["features"]
    print(f"[evaluate] Using {len(feature_cols)} features from {feature_columns_path.name}.")

    # Slice by SPLIT.
    val_df = engineered.loc[engineered["SPLIT"].values == "val"]
    test_df = engineered.loc[engineered["SPLIT"].values == "test"]
    if val_df.empty or test_df.empty:
        raise ValueError("Empty val or test slice — split is broken.")

    X_val = val_df[feature_cols]
    y_val = val_df["TARGET"].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df["TARGET"].astype(int).values
    print(f"[evaluate] Val size: {len(X_val):,}; Test size: {len(X_test):,}")

    # Load both models.
    print(f"[evaluate] Loading XGBoost from {xgboost_path} ...")
    xgb = joblib.load(xgboost_path)
    print(f"[evaluate] Loading LogReg baseline from {baseline_path} ...")
    lr = joblib.load(baseline_path)

    # Compute metrics for both models on both slices.
    threshold = params["threshold"]
    metrics: dict = {
        "primary_metric": params["primary_metric"],
        "threshold": threshold,
        "models": {
            "xgboost": {
                "validation": compute_metrics_at_threshold(y_val, _predict_proba(xgb, X_val), threshold),
                "test":       compute_metrics_at_threshold(y_test, _predict_proba(xgb, X_test), threshold),
            },
            "logistic_regression_baseline": {
                "validation": compute_metrics_at_threshold(y_val, _predict_proba(lr, X_val), threshold),
                "test":       compute_metrics_at_threshold(y_test, _predict_proba(lr, X_test), threshold),
            },
        },
    }

    # Save metrics.json (DVC-tracked).
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[evaluate] Wrote metrics -> {metrics_out}")

    # Save plots.
    print("[evaluate] Saving evaluation plots ...")
    _plot_roc_curves(
        [
            ("XGBoost (val)",         y_val,  _predict_proba(xgb, X_val)),
            ("XGBoost (test)",        y_test, _predict_proba(xgb, X_test)),
            ("LogReg baseline (val)", y_val,  _predict_proba(lr, X_val)),
            ("LogReg baseline (test)",y_test, _predict_proba(lr, X_test)),
        ],
        plots_dir / "roc_curves.png",
        title="ROC curves — XGBoost vs LogReg baseline",
    )
    _plot_pr_curves(
        [
            ("XGBoost (val)",         y_val,  _predict_proba(xgb, X_val)),
            ("XGBoost (test)",        y_test, _predict_proba(xgb, X_test)),
            ("LogReg baseline (val)", y_val,  _predict_proba(lr, X_val)),
            ("LogReg baseline (test)",y_test, _predict_proba(lr, X_test)),
        ],
        plots_dir / "pr_curves.png",
        title="Precision-Recall curves — XGBoost vs LogReg baseline",
    )
    _plot_confusion_matrices(
        {
            "XGBoost (test)":         metrics["models"]["xgboost"]["test"]["confusion_matrix"],
            "LogReg baseline (test)": metrics["models"]["logistic_regression_baseline"]["test"]["confusion_matrix"],
        },
        plots_dir / "confusion_matrices.png",
        title=f"Confusion matrices (test, threshold={threshold:.3f})",
    )

    # Friendly summary on stdout.
    print("\n=== Evaluation summary ===")
    print(
        f"XGBoost           | val ROC-AUC: {metrics['models']['xgboost']['validation']['roc_auc']:.4f}"
        f"  test ROC-AUC: {metrics['models']['xgboost']['test']['roc_auc']:.4f}"
    )
    print(
        f"LogReg baseline   | val ROC-AUC: {metrics['models']['logistic_regression_baseline']['validation']['roc_auc']:.4f}"
        f"  test ROC-AUC: {metrics['models']['logistic_regression_baseline']['test']['roc_auc']:.4f}"
    )
    return {"metrics_path": str(metrics_out), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "params.yaml")
    parser.add_argument(
        "--engineered",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_engineered.parquet",
    )
    parser.add_argument("--xgboost", type=Path, default=REPO_ROOT / "models" / "xgboost_model.pkl")
    parser.add_argument(
        "--baseline", type=Path, default=REPO_ROOT / "models" / "baseline_logistic_regression.pkl"
    )
    parser.add_argument(
        "--feature-columns", type=Path, default=REPO_ROOT / "models" / "feature_columns.json"
    )
    parser.add_argument("--metrics-out", type=Path, default=REPO_ROOT / "reports" / "metrics.json")
    parser.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "reports" / "evaluation")
    args = parser.parse_args()

    params = load_params(args.params)
    run(
        engineered_parquet=args.engineered,
        xgboost_path=args.xgboost,
        baseline_path=args.baseline,
        feature_columns_path=args.feature_columns,
        metrics_out=args.metrics_out,
        plots_dir=args.plots_dir,
        params=params,
    )


if __name__ == "__main__":
    main()