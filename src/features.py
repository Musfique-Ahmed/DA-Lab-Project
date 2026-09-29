"""Features stage — MLSD pipeline.

Loads the cleaned parquet from `prepare`, applies the 7 engineered
features (from `src.features.engineer`), then selects the top-N
features using XGB + RF + SHAP importance ranks — **fit on the train
slice only**.

This is the `features` stage of `dvc.yaml`.

Outputs
-------
- `data/processed/train_engineered.parquet` — full engineered matrix
  (train + val + test, scaled) so downstream stages can slice by SPLIT.
- `reports/selected_features.json` — the top-N feature names in the
  order they should appear in the model input.

Leakage discipline
------------------
The DA project's Phase 3 notebook ran the importance extractor on the
full feature matrix (train + val + test). The MLSD pipeline restricts
it to the train slice:

    importance_df = select.train_xgb_and_rf_importances(
        X.loc[train_mask, ...], y.loc[train_mask]
    )

Validation and test rows are NEVER seen by the importance extractor
or by SHAP. This is the leakage fix from Phase 0 of the audit.

Usage
-----
    python src/features.py
        [--params params.yaml]
        [--clean-parquet data/processed/train_clean.parquet]
        [--out-parquet data/processed/train_engineered.parquet]
        [--out-features reports/selected_features.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import engineer as engineer_mod  # noqa: E402
from src.features import select as select_mod  # noqa: E402

DEFAULT_TOP_N: int = 30


def load_params(params_path: Path) -> dict:
    """Read params.yaml; return the `features` sub-section with sensible fallbacks."""
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path}. "
            "Run `dvc repro` from the repo root, or pass --params."
        )
    with open(params_path, "r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}
    feat = params.get("features", {})
    data = params.get("data", {})
    return {
        "top_n": int(feat.get("top_n", DEFAULT_TOP_N)),
        "random_state": int(data.get("random_state", 42)),
    }


def _train_only(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X_train, y_train) sliced to SPLIT == 'train'."""
    if "SPLIT" not in df.columns:
        raise ValueError("Input parquet must have a SPLIT column (from the prepare stage).")
    if "TARGET" not in df.columns:
        raise ValueError("Input parquet must have a TARGET column.")
    train_mask = df["SPLIT"].values == "train"
    if not train_mask.any():
        raise ValueError("No rows with SPLIT == 'train' — split is broken.")
    df_train = df.loc[train_mask].copy()
    y_train = df_train["TARGET"].astype(int)
    # Drop non-feature columns.
    drop = [c for c in ("SK_ID_CURR", "TARGET", "SPLIT") if c in df_train.columns]
    X_train = df_train.drop(columns=drop)
    return X_train, y_train


def run(
    *,
    clean_parquet: Path,
    out_parquet: Path,
    out_features: Path,
    params: dict,
) -> dict:
    """Run the full features stage. Returns a small info dict for logging."""
    print(f"[features] Loading cleaned parquet from {clean_parquet} ...")
    cleaned = pd.read_parquet(clean_parquet)
    print(f"[features] Cleaned shape: {cleaned.shape}")

    # 1. Engineer 7 derived columns on the full cleaned matrix.
    engineered = engineer_mod.engineer_features(cleaned)
    engineered_cols = [c for c in engineer_mod.ENGINEERED_COLUMNS if c in engineered.columns]
    print(f"[features] Added {len(engineered_cols)} engineered columns.")

    # 2. Restrict to the TRAIN slice for selection.
    X_train, y_train = _train_only(engineered)
    print(f"[features] Train slice shape (for selection): {X_train.shape}")

    # 3. XGB + RF importances on train only.
    print("[features] Computing XGB + RF importances on train slice ...")
    importance_df = select_mod.train_xgb_and_rf_importances(
        X_train, y_train, random_state=params["random_state"]
    )

    # 4. SHAP on a 5,000-row subsample of the train slice.
    print("[features] Computing SHAP on a 5,000-row train subsample ...")
    shap_matrix, shap_feature_names = select_mod.shap_values_xgb(
        X_train, y_train, sample_size=5_000, random_state=params["random_state"]
    )

    # 5. Combined top-N selection.
    top = select_mod.select_top_n(
        importance_df, shap_matrix, shap_feature_names, n=params["top_n"]
    )
    selected_features = top["feature"].tolist()
    print(f"[features] Selected {len(selected_features)} features (top-N={params['top_n']}).")
    print(f"[features] Top 5: {selected_features[:5]}")

    # 6. Persist the full engineered matrix for downstream stages.
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(out_parquet, index=False)
    print(f"[features] Wrote engineered parquet -> {out_parquet}")

    # 7. Persist the selected feature list (used by `train` and `evaluate`).
    out_features.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "top_n": params["top_n"],
        "random_state": params["random_state"],
        "n_features": len(selected_features),
        "features": selected_features,
        "ranking": [
            {
                "rank": int(row["mean_rank"]) if False else idx + 1,  # 1-based by position
                "feature": row["feature"],
                "xgb_rank": int(row["xgb_rank"]),
                "rf_rank": int(row["rf_rank"]),
                "shap_rank": int(row["shap_rank"]) if "shap_rank" in top.columns else None,
                "mean_rank": float(row["mean_rank"]),
                "combined_score": float(row["combined_score"]),
            }
            for idx, (_, row) in enumerate(top.iterrows())
        ],
    }
    out_features.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[features] Wrote selected features JSON -> {out_features}")

    return {
        "n_features_selected": len(selected_features),
        "selected_features": selected_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "params.yaml")
    parser.add_argument(
        "--clean-parquet",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_clean.parquet",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_engineered.parquet",
    )
    parser.add_argument(
        "--out-features",
        type=Path,
        default=REPO_ROOT / "reports" / "selected_features.json",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    run(
        clean_parquet=args.clean_parquet,
        out_parquet=args.out_parquet,
        out_features=args.out_features,
        params=params,
    )


if __name__ == "__main__":
    main()