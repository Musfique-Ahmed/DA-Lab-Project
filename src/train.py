"""Train stage — MLSD pipeline.

Loads the engineered parquet + selected feature list, trains XGBoost
(primary model) and Logistic Regression (baseline) on the train slice,
and persists both artifacts.

This is the `train` stage of `dvc.yaml`.

Outputs
-------
- models/xgboost_model.pkl              (primary)
- models/baseline_logistic_regression.pkl (baseline)
- models/feature_columns.json           (the exact X columns used)

Leakage discipline
------------------
- XGBoost sees only the train slice. Validation/test rows are reserved
  for the `evaluate` stage.
- Logistic Regression is wrapped in a sklearn Pipeline(median-impute,
  StandardScaler, LR). The imputer and scaler are fit on the train slice
  only — so LR obeys the same "fit on train" discipline as the XGBoost.
- `scale_pos_weight = neg/pos` is computed from the train labels
  (a single scalar; no information from val/test leaks).

Reproducibility
---------------
- `random_state=42` is applied to both models via params.yaml.
- XGBoost's `tree_method='hist'` is deterministic given the same data
  and seed.

Usage
-----
    python src/train.py
        [--params params.yaml]
        [--engineered data/processed/train_engineered.parquet]
        [--selected-features reports/selected_features.json]
        [--models-dir models]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_params(params_path: Path) -> dict:
    """Load the `model` and `baseline` sections of params.yaml with fallbacks."""
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path}. "
            "Run `dvc repro` from the repo root, or pass --params."
        )
    with open(params_path, "r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}

    model_cfg = params.get("model", {})
    baseline_cfg = params.get("baseline", {})
    data_cfg = params.get("data", {})

    return {
        "random_state": int(data_cfg.get("random_state", 42)),
        "model": {
            "type": str(model_cfg.get("type", "xgboost")),
            "n_estimators": int(model_cfg.get("n_estimators", 300)),
            "max_depth": int(model_cfg.get("max_depth", 4)),
            "learning_rate": float(model_cfg.get("learning_rate", 0.05)),
            "eval_metric": str(model_cfg.get("eval_metric", "logloss")),
            "tree_method": str(model_cfg.get("tree_method", "hist")),
            "random_state": int(model_cfg.get("random_state", 42)),
        },
        "baseline": {
            "type": str(baseline_cfg.get("type", "logistic_regression")),
            "C": float(baseline_cfg.get("C", 1.0)),
            "max_iter": int(baseline_cfg.get("max_iter", 200)),
            "class_weight": str(baseline_cfg.get("class_weight", "balanced")),
            "random_state": int(baseline_cfg.get("random_state", 42)),
        },
    }


def _split_slices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'train': df, 'val': df, 'test': df} sliced by SPLIT column."""
    if "SPLIT" not in df.columns:
        raise ValueError("Engineered parquet must have a SPLIT column.")
    return {k: df.loc[df["SPLIT"].values == k].copy() for k in ("train", "val", "test")}


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """neg/pos, with a floor of 1 to avoid divide-by-zero."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return n_neg / max(n_pos, 1)


def _train_xgboost(
    X_tr: pd.DataFrame, y_tr: np.ndarray, *, params: dict, random_state: int
) -> XGBClassifier:
    cfg = params["model"]
    spw = _compute_scale_pos_weight(y_tr)
    print(
        f"[train] XGBoost config: n_estimators={cfg['n_estimators']}, "
        f"max_depth={cfg['max_depth']}, lr={cfg['learning_rate']}, "
        f"scale_pos_weight={spw:.4f}, random_state={random_state}"
    )
    model = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        scale_pos_weight=spw,
        eval_metric=cfg["eval_metric"],
        tree_method=cfg["tree_method"],
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr.values, y_tr)
    return model


def _train_logreg(
    X_tr: pd.DataFrame, y_tr: np.ndarray, *, params: dict, random_state: int
) -> Pipeline:
    """Pipeline: median-impute (fit on train) -> StandardScaler (fit on train) -> LR."""
    cfg = params["baseline"]
    print(
        f"[train] LogReg config: C={cfg['C']}, max_iter={cfg['max_iter']}, "
        f"class_weight={cfg['class_weight']}, random_state={random_state}"
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=cfg["C"],
            max_iter=cfg["max_iter"],
            class_weight=cfg["class_weight"],
            random_state=random_state,
            solver="lbfgs",
        )),
    ])
    pipe.fit(X_tr.values, y_tr)
    return pipe


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probabilities regardless of model type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X.values)[:, 1]
    # Pipeline with LR always has predict_proba; this is defensive.
    return np.asarray(model.predict(X.values), dtype=float)


def run(
    *,
    engineered_parquet: Path,
    selected_features_path: Path,
    models_dir: Path,
    params: dict,
) -> dict:
    """Run the train stage. Returns a small info dict for logging."""
    print(f"[train] Loading engineered parquet from {engineered_parquet} ...")
    engineered = pd.read_parquet(engineered_parquet)

    print(f"[train] Loading selected features from {selected_features_path} ...")
    sel = json.loads(selected_features_path.read_text(encoding="utf-8"))
    feature_cols: list[str] = sel["features"]
    print(f"[train] Using {len(feature_cols)} features.")

    # Validate that every selected feature is in the engineered parquet.
    missing = [c for c in feature_cols if c not in engineered.columns]
    if missing:
        raise ValueError(
            f"Selected features missing from engineered parquet: {missing[:5]} ... "
            f"({len(missing)} total)"
        )

    # Slice by SPLIT.
    slices = _split_slices(engineered)
    n_train = len(slices["train"])
    n_val = len(slices["val"])
    n_test = len(slices["test"])
    print(f"[train] Split sizes: train={n_train:,}, val={n_val:,}, test={n_test:,}")

    X_tr = slices["train"][feature_cols]
    y_tr = slices["train"]["TARGET"].astype(int).values
    X_val = slices["val"][feature_cols]
    y_val = slices["val"]["TARGET"].astype(int).values

    # ---- Primary: XGBoost ----
    xgb = _train_xgboost(X_tr, y_tr, params=params, random_state=params["random_state"])
    xgb_val_auc = float(roc_auc_score(y_val, _predict_proba(xgb, X_val)))
    print(f"[train] XGBoost val AUC: {xgb_val_auc:.4f}")

    # ---- Baseline: Logistic Regression ----
    lr = _train_logreg(X_tr, y_tr, params=params, random_state=params["random_state"])
    lr_val_auc = float(roc_auc_score(y_val, _predict_proba(lr, X_val)))
    print(f"[train] LogReg   val AUC: {lr_val_auc:.4f}")

    # ---- Persist ----
    models_dir.mkdir(parents=True, exist_ok=True)
    xgb_path = models_dir / "xgboost_model.pkl"
    lr_path = models_dir / "baseline_logistic_regression.pkl"
    feat_path = models_dir / "feature_columns.json"

    joblib.dump(xgb, xgb_path)
    print(f"[train] Wrote XGBoost artifact -> {xgb_path}")
    joblib.dump(lr, lr_path)
    print(f"[train] Wrote LogReg artifact  -> {lr_path}")

    feat_path.write_text(
        json.dumps(
            {
                "n_features": len(feature_cols),
                "features": feature_cols,
                "random_state": params["random_state"],
                "xgb_val_auc": xgb_val_auc,
                "lr_val_auc": lr_val_auc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train] Wrote feature columns JSON -> {feat_path}")

    return {
        "xgb_path": str(xgb_path),
        "lr_path": str(lr_path),
        "feat_path": str(feat_path),
        "xgb_val_auc": xgb_val_auc,
        "lr_val_auc": lr_val_auc,
        "n_features": len(feature_cols),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "params.yaml")
    parser.add_argument(
        "--engineered",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_engineered.parquet",
    )
    parser.add_argument(
        "--selected-features",
        type=Path,
        default=REPO_ROOT / "reports" / "selected_features.json",
    )
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    args = parser.parse_args()

    params = load_params(args.params)
    run(
        engineered_parquet=args.engineered,
        selected_features_path=args.selected_features,
        models_dir=args.models_dir,
        params=params,
    )


if __name__ == "__main__":
    main()