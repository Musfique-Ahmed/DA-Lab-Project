"""Shared evaluation helpers for Phase 4.

Pure functions — no I/O except for the plot_* helpers, which save figures.
Called by `train.py`, `tune.py`, and the notebook cells.

Why this module exists:
  - All 4 models (LR, RF, XGB, MLP) need the same metric set + threshold
    policy + ROC plot helpers. Centralizing avoids drift between models.
  - `compute_metrics` is the single source of truth for "what does this
    model look like on a given dataset".
  - `pick_threshold` implements the documented Youden's J vs 0.5 rule.

Conventions:
  - y_true / y_proba are always 1D numpy arrays.
  - The "positive class" is always index 1 (default = default = 1).
  - `compute_metrics` returns a dict so the same shape is used for the
    comparison table regardless of model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Compute AUC-ROC + precision/recall/F1 + confusion matrix at one threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0/1).
    y_proba : np.ndarray
        Predicted probability of the positive class (0..1).
    threshold : float
        Decision threshold for the precision/recall/F1/confusion-matrix numbers.
        AUC is threshold-free and does not depend on this value.

    Returns
    -------
    dict with keys:
        auc_roc          : float
        precision        : float
        recall           : float
        f1               : float
        threshold        : float (echoed back)
        confusion_matrix : list[int] of length 4 in sklearn order [TN, FP, FN, TP]
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
    }


def pick_threshold(y_true: np.ndarray, y_proba: np.ndarray, strategy: str = "youden") -> tuple[float, float]:
    """Pick an operating threshold on (y_true, y_proba).

    Parameters
    ----------
    y_true, y_proba : np.ndarray
    strategy : {"youden", "fixed_0_5"}
        - "youden"   : argmax(sensitivity + specificity - 1) over the ROC curve.
        - "fixed_0_5" : always return 0.50.

    Returns
    -------
    (threshold, youden_J_at_that_threshold)
        The J statistic at the returned threshold; callers use the delta
        vs J(0.50) to decide between strategies.
    """
    if strategy == "fixed_0_5":
        t = 0.5
    elif strategy == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        j = tpr - fpr
        # roc_curve's last threshold is > max(y_proba); restrict to a sane range.
        valid = thr <= 1.0
        j_v = j[valid]
        thr_v = thr[valid]
        idx = int(np.argmax(j_v))
        t = float(thr_v[idx])
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    # Recompute J at the chosen threshold.
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return t, float(sens + spec - 1)


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, label: str, ax=None, *, color: str | None = None) -> None:
    """Draw a single ROC curve on the given axis (or a fresh one)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    line = ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", linewidth=2)
    if color is not None:
        line[0].set_color(color)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {label}")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)


def plot_roc_overlay(specs: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    """Save a single ROC overlay (all models on one axis)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y_true, y_proba in specs:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — 4-model comparison")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrices(matrices: dict[str, list[int]], path: Path) -> None:
    """Save a 2x2 panel of confusion matrices (one per model).

    Parameters
    ----------
    matrices : dict
        Keys are model labels; values are [TN, FP, FN, TP] lists.
    path : Path
        Output PNG path.
    """
    n = len(matrices)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes_flat = axes.flat if rows > 1 else [axes] if cols == 1 else axes
    for ax, (label, cm) in zip(axes_flat, matrices.items()):
        tn, fp, fn, tp = cm
        grid = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(grid, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                        color="white" if grid[i, j] > grid.max() / 2 else "black",
                        fontsize=14)
        ax.set_title(f"{label}\nTN={tn}  FP={fp}\nFN={fn}  TP={tp}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    # Hide any unused subplots.
    for ax in axes_flat[len(matrices):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (validation set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def youden_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Convenience wrapper for the most common case — Youden's J."""
    return pick_threshold(y_true, y_proba, strategy="youden")


# Re-export for callers that want precision/recall curve directly.
def pr_curve_arrays(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for the positive class."""
    return precision_recall_curve(y_true, y_proba)


__all__ = [
    "compute_metrics",
    "pick_threshold",
    "youden_threshold",
    "plot_roc",
    "plot_roc_overlay",
    "plot_confusion_matrices",
    "pr_curve_arrays",
]
