"""Model training and scoring — populated in Phase 4.

Public API:
  - score_application(input) -> dict   (master-prompt artifact)
  - train_logreg / train_random_forest / train_xgboost / train_mlp
  - tune_logreg / tune_random_forest / tune_xgboost / tune_mlp
  - compute_metrics / pick_threshold / plot_roc / plot_roc_overlay

MLP / torch imports are lazy — the `MLP`, `MLPWrapper`, and
`DEFAULT_RANDOM_STATE` symbols are only resolved when explicitly
accessed. This lets the dashboard and the score module import
without torch installed (e.g., on a fresh Python install before
the optional torch dependency is pulled in).
"""
from src.models.evaluate import (
    compute_metrics,
    pick_threshold,
    plot_confusion_matrices,
    plot_roc,
    plot_roc_overlay,
    youden_threshold,
)
from src.models.score import (
    DEFAULT_INPUT_COLUMNS,
    THRESHOLD_APPROVE_MAX,
    THRESHOLD_REJECT_MIN,
    score_application,
)
from src.models.train import (
    train_logreg,
    train_mlp,
    train_random_forest,
    train_xgboost,
)
from src.models.tune import (
    tune_logreg,
    tune_mlp,
    tune_random_forest,
    tune_xgboost,
)


def __getattr__(name):
    """Lazy-load torch-dependent symbols on first access."""
    if name in ("MLP", "MLPWrapper", "DEFAULT_RANDOM_STATE"):
        from src.models import mlp as _mlp
        return getattr(_mlp, name)
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")


__all__ = [
    # scoring (Phase 5 dashboard-facing)
    "score_application",
    "THRESHOLD_APPROVE_MAX",
    "THRESHOLD_REJECT_MIN",
    "DEFAULT_INPUT_COLUMNS",
    # train / tune
    "train_logreg", "train_random_forest", "train_xgboost", "train_mlp",
    "tune_logreg", "tune_random_forest", "tune_xgboost", "tune_mlp",
    # MLP pieces (lazy)
    "MLP", "MLPWrapper", "DEFAULT_RANDOM_STATE",
    # evaluation helpers
    "compute_metrics", "pick_threshold", "youden_threshold",
    "plot_roc", "plot_roc_overlay", "plot_confusion_matrices",
]