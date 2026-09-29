"""Deploy the trained ML model(s) to MLflow.

Logs the trained XGBoost (primary) and Logistic Regression baseline
artifacts produced by the MLSD `train` stage into an MLflow tracking
server (default: a local `mlruns/` directory + sqlite backend registry).

For each model, this script:

  1. Starts (or attaches to) a configured MLflow experiment.
  2. Logs parameters from `params.yaml` (`model.*`, `baseline.*`,
     `data.random_state`, etc.).
  3. Logs metrics from `reports/metrics.json` (val + test ROC-AUC,
     PR-AUC, precision, recall, F1, accuracy).
  4. Logs the feature column list and the engineered parquet for
     reproducibility.
  5. Logs the model itself with `mlflow.<flavor>.log_model`, registering
     it in the MLflow Model Registry under a chosen name.
  6. Transitions the new run's model version to the chosen stage
     (`Staging` / `Production` / `Archived`).

Usage
-----
    python src/deploy_mlflow.py
        [--tracking-uri ./mlruns]      # local file backend (default)
        [--experiment CreditRisk]      # experiment name
        [--registered-name credit_risk_xgboost]
        [--stage Staging]              # None | Staging | Production | Archived
        [--no-promote]                 # skip auto transition to a stage
        [--models-dir models]
        [--metrics reports/metrics.json]
        [--engineered data/processed/train_engineered.parquet]
        [--feature-columns models/feature_columns.json]
        [--params params.yaml]

The MLflow UI can be started with:
    mlflow ui --backend-store-uri ./mlruns --port 5000
or, if using a sqlite backend via `mlflow server`:
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from mlflow.models import infer_signature

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helper loaders
# ---------------------------------------------------------------------------
def load_params(params_path: Path) -> dict:
    """Load params.yaml into a plain dict."""
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path}. "
            "Run `dvc repro` first, or pass --params."
        )
    with open(params_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.json not found at {metrics_path}. "
            "Run `dvc repro` first to produce it."
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_feature_columns(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Deployment logic
# ---------------------------------------------------------------------------
def _flatten_metrics(model_metrics: dict) -> dict[str, float]:
    """Flatten the nested metrics.json structure for MLflow.

    Example key: ``val_roc_auc``, ``test_pr_auc``, ``test_accuracy``.
    """
    flat: dict[str, float] = {}
    for split_name in ("validation", "test"):
        if split_name not in model_metrics:
            continue
        split = model_metrics[split_name]
        prefix = "val" if split_name == "validation" else split_name
        for key, value in split.items():
            if key in ("confusion_matrix", "threshold"):
                continue
            flat[f"{prefix}_{key}"] = float(value)
    return flat


def _log_model(
    *,
    model_name: str,
    registered_name: str | None,
    model_obj,
    feature_cols: list[str],
    X_sample: pd.DataFrame,
    y_sample: pd.Series | None,
) -> str:
    """Log a single model artifact using its native MLflow flavor.

    Returns the model URI.
    """
    # Build a Model signature from a small inference example. This is how
    # downstream services know the schema for inputs/outputs.
    signature = infer_signature(
        X_sample, model_obj.predict_proba(X_sample.values)[:, 1]
    )

    artifact_path = model_name.replace(" ", "_").lower()

    if "xgb" in type(model_obj).__module__.lower() or hasattr(model_obj, "get_booster"):
        # Native XGBoost flavor (better fidelity, faster load).
        # MLflow 3.x -> xgboost>=2.1 quirk: the native flavor calls
        # `model._get_type()` which now requires `_estimator_type` to be
        # set. Newer xgboost dropped the implicit sklearn-mixin setter.
        # We tag it manually so the native flavor can log the booster.
        if not hasattr(model_obj, "_estimator_type"):
            try:
                model_obj._estimator_type = "classifier"  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            model_info = mlflow.xgboost.log_model(
                xgb_model=model_obj,
                artifact_path=artifact_path,
                signature=signature,
                input_example=X_sample.iloc[: min(5, len(X_sample))],
                registered_model_name=registered_name,
            )
        except Exception as exc:  # pragma: no cover — last-resort fallback
            print(
                f"[deploy_mlflow] Native xgboost flavor failed ({exc!r}); "
                "falling back to sklearn flavor."
            )
            from mlflow.sklearn import log_model as _sk_log  # local import
            model_info = _sk_log(
                sk_model=model_obj,
                artifact_path=artifact_path,
                signature=signature,
                input_example=X_sample.iloc[: min(5, len(X_sample))],
                registered_model_name=registered_name,
            )
    else:
        # Anything else (sklearn Pipeline, etc.) uses the sklearn flavor.
        # Newer sklearn versions (>=1.4) pickle with `skops` which fails
        # the safety audit on common types like numpy.dtype. We declare
        # those trusted because the model was authored here.
        model_info = mlflow.sklearn.log_model(
            sk_model=model_obj,
            artifact_path=artifact_path,
            signature=signature,
            input_example=X_sample.iloc[: min(5, len(X_sample))],
            registered_model_name=registered_name,
            skops_trusted_types=[
                "numpy.dtype",
                "numpy.core.multiarray.scalar",
                "numpy.ndarray",
                "sklearn.pipeline.Pipeline",
                "sklearn.impute.SimpleImputer",
                "sklearn.preprocessing.StandardScaler",
                "sklearn.linear_model.LogisticRegression",
            ],
        )

    # Tag the artifact with the exact feature columns so the model is
    # self-describing.
    mlflow.set_tag("model_class", type(model_obj).__name__)
    mlflow.set_tag("n_features", str(len(feature_cols)))
    mlflow.set_tag("feature_columns_json", json.dumps(feature_cols))
    return model_info.model_uri


def deploy(
    *,
    tracking_uri: Path,
    experiment_name: str,
    registered_name_xgb: str,
    registered_name_lr: str,
    stage: str | None,
    models_dir: Path,
    metrics_path: Path,
    feature_columns_path: Path,
    params_path: Path,
    engineered_parquet: Path | None,
) -> dict:
    """Run the MLflow deployment end-to-end. Returns a small summary dict."""

    tracking_uri_path = tracking_uri.expanduser().resolve()
    tracking_uri_path.mkdir(parents=True, exist_ok=True)

    # MLflow 3.x requires a SQL backend for the tracking store. We use a
    # local sqlite file (`mlflow.db`) co-located with the artifact root.
    sqlite_db = tracking_uri_path / "mlflow.db"
    artifact_root = tracking_uri_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{sqlite_db.as_posix()}"
    print(f"[deploy] MLflow tracking URI: {tracking_uri}")
    print(f"[deploy] Artifact root: {artifact_root}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    params = load_params(params_path)
    metrics = load_metrics(metrics_path)
    feature_meta = load_feature_columns(feature_columns_path)
    feature_cols: list[str] = feature_meta["features"]

    # Load a small slice of the engineered parquet (val) to infer
    # signatures & build input examples. This is also what gets logged
    # alongside the model for easy reproducibility.
    input_example_df: pd.DataFrame | None = None
    if engineered_parquet and engineered_parquet.exists():
        df = pd.read_parquet(engineered_parquet, columns=feature_cols)
        val_mask = None
        # If SPLIT is present we only use val rows for the input example;
        # if not (synthetic), we just take the first 100 rows.
        parquet_full = pd.read_parquet(engineered_parquet, columns=["SPLIT"] + feature_cols)
        if "SPLIT" in parquet_full.columns:
            val_mask = parquet_full["SPLIT"].values == "val"
        if val_mask is not None and val_mask.any():
            input_example_df = df.loc[val_mask].head(100).reset_index(drop=True)
        else:
            input_example_df = df.head(100).reset_index(drop=True)
    else:
        print("[deploy] WARNING: engineered parquet not available — "
              "using a synthetic zero-row sample for input_example.")

    # ---------------------------------------------------------------------------
    # Run 1: XGBoost (primary model)
    # ---------------------------------------------------------------------------
    xgb_path = models_dir / "xgboost_model.pkl"
    lr_path = models_dir / "baseline_logistic_regression.pkl"

    summary: dict = {"runs": []}

    with mlflow.start_run(run_name="xgboost_primary") as xgb_run:
        run = mlflow.active_run()
        print(f"[deploy] MLflow run id (XGBoost): {run.info.run_id}")

        # --- parameters ---
        mlflow.log_params({
            "model_type":           "xgboost",
            "n_estimators":         params["model"]["n_estimators"],
            "max_depth":            params["model"]["max_depth"],
            "learning_rate":        params["model"]["learning_rate"],
            "eval_metric":          params["model"]["eval_metric"],
            "tree_method":          params["model"]["tree_method"],
            "model_random_state":   params["model"]["random_state"],
            "data_random_state":    params["data"]["random_state"],
            "train_size":           params["data"]["train_size"],
            "validation_size":      params["data"]["validation_size"],
            "test_size":            params["data"]["test_size"],
            "n_features_selected":  feature_meta["n_features"],
            "evaluation_threshold": params["evaluation"]["threshold"],
            "evaluation_primary":   params["evaluation"]["primary_metric"],
        })

        # --- tags ---
        mlflow.set_tag("model_kind", "primary")
        mlflow.set_tag("dataset", "home_credit_default_risk")
        mlflow.set_tag("stage_hint", stage or "None")

        # --- metrics ---
        flat = _flatten_metrics(metrics["models"]["xgboost"])
        mlflow.log_metrics(flat)
        for k, v in flat.items():
            print(f"[deploy]   metric {k} = {v:.4f}")

        # --- artifacts: input example + feature columns ---
        feature_columns_path_local = Path("feature_columns.json")
        feature_columns_path_local.write_text(
            json.dumps(feature_meta, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(feature_columns_path_local))
        feature_columns_path_local.unlink(missing_ok=True)

        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path))

        # --- log the model ---
        xgb_model = joblib.load(xgb_path)
        if input_example_df is None:
            input_example_df = pd.DataFrame(
                [[0.0] * len(feature_cols)], columns=feature_cols
            )

        xgb_uri = _log_model(
            model_name="xgboost_credit_default",
            registered_name=registered_name_xgb,
            model_obj=xgb_model,
            feature_cols=feature_cols,
            X_sample=input_example_df,
            y_sample=None,
        )
        print(f"[deploy] XGBoost model URI: {xgb_uri}")

        summary["runs"].append({
            "name": "xgboost_primary",
            "run_id": run.info.run_id,
            "model_uri": xgb_uri,
            "registered_name": registered_name_xgb,
            "metrics": flat,
        })

    # ---------------------------------------------------------------------------
    # Run 2: Logistic Regression (baseline)
    # ---------------------------------------------------------------------------
    with mlflow.start_run(run_name="logreg_baseline") as lr_run:
        run = mlflow.active_run()
        print(f"[deploy] MLflow run id (LogReg): {run.info.run_id}")

        mlflow.log_params({
            "model_type":          "logistic_regression",
            "C":                   params["baseline"]["C"],
            "max_iter":            params["baseline"]["max_iter"],
            "class_weight":        params["baseline"]["class_weight"],
            "baseline_random_state": params["baseline"]["random_state"],
            "data_random_state":   params["data"]["random_state"],
            "n_features_selected": feature_meta["n_features"],
            "evaluation_threshold": params["evaluation"]["threshold"],
        })
        mlflow.set_tag("model_kind", "baseline")
        mlflow.set_tag("dataset", "home_credit_default_risk")
        mlflow.set_tag("stage_hint", stage or "None")

        flat = _flatten_metrics(metrics["models"]["logistic_regression_baseline"])
        mlflow.log_metrics(flat)
        for k, v in flat.items():
            print(f"[deploy]   metric {k} = {v:.4f}")

        lr_model = joblib.load(lr_path)
        lr_uri = _log_model(
            model_name="logreg_credit_baseline",
            registered_name=registered_name_lr,
            model_obj=lr_model,
            feature_cols=feature_cols,
            X_sample=input_example_df,
            y_sample=None,
        )
        print(f"[deploy] LogReg model URI: {lr_uri}")

        summary["runs"].append({
            "name": "logreg_baseline",
            "run_id": run.info.run_id,
            "model_uri": lr_uri,
            "registered_name": registered_name_lr,
            "metrics": flat,
        })

    # ---------------------------------------------------------------------------
    # Optional: transition to a registry stage
    # ---------------------------------------------------------------------------
    if stage:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=tracking_uri)
            for reg_name in (registered_name_xgb, registered_name_lr):
                try:
                    versions = client.get_latest_versions(reg_name, stages=["None"])
                except Exception:
                    versions = []
                for v in versions:
                    print(f"[deploy] Transitioning {reg_name} v{v.version} -> {stage}")
                    client.transition_model_version_stage(
                        name=reg_name,
                        version=v.version,
                        stage=stage,
                        archive_existing_versions=False,
                    )
        except Exception as exc:  # pragma: no cover — best-effort promotion
            print(f"[deploy] WARNING: could not transition stages: {exc}")

    print("[deploy] Done.")
    summary["tracking_uri"] = tracking_uri
    summary["artifact_root"] = str(artifact_root)
    summary["experiment"] = experiment_name
    summary["registered_xgb"] = registered_name_xgb
    summary["registered_lr"] = registered_name_lr
    summary["stage"] = stage
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tracking-uri", type=Path, default=REPO_ROOT / "mlruns",
        help="Path to the local mlruns directory (file backend).",
    )
    parser.add_argument("--experiment", type=str, default="CreditRisk")
    parser.add_argument(
        "--registered-name", type=str, default="credit_risk_xgboost",
        help="Model Registry name for the primary XGBoost model.",
    )
    parser.add_argument(
        "--registered-name-lr", type=str, default="credit_risk_logreg_baseline",
        help="Model Registry name for the LogReg baseline.",
    )
    parser.add_argument(
        "--stage", type=str, default="Staging",
        choices=["None", "Staging", "Production", "Archived"],
        help="Registry stage to transition the new model version to. "
             "Pass 'None' to skip promotion.",
    )
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--metrics", type=Path, default=REPO_ROOT / "reports" / "metrics.json")
    parser.add_argument(
        "--feature-columns", type=Path, default=REPO_ROOT / "models" / "feature_columns.json"
    )
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "params.yaml")
    parser.add_argument(
        "--engineered", type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_engineered.parquet",
    )
    parser.add_argument(
        "--no-promote", action="store_true",
        help="Skip transitioning the registered model to a stage.",
    )
    args = parser.parse_args()

    summary = deploy(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        registered_name_xgb=args.registered_name,
        registered_name_lr=args.registered_name_lr,
        stage=None if args.no_promote else args.stage,
        models_dir=args.models_dir,
        metrics_path=args.metrics,
        feature_columns_path=args.feature_columns,
        params_path=args.params,
        engineered_parquet=args.engineered,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
