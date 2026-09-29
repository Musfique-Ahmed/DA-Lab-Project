# MLflow Deployment — MLSD Phase 5

This document describes how the trained credit-risk models are deployed
to MLflow as the project's **model registry** and **experiment
tracking** layer.

The MLSD pipeline already produces:
- `models/xgboost_model.pkl` — primary classifier (val AUC 0.7545, test AUC 0.7592)
- `models/baseline_logistic_regression.pkl` — interpretable baseline (val AUC 0.7416)
- `models/feature_columns.json` — exact 30-feature list used at training time
- `reports/metrics.json` — DVC-tracked metric bundle

This phase adds an **MLflow deployment layer** on top: the same artifacts
become a registered model with tracked parameters, metrics, signatures,
and registry stages.

---

## 1. What got deployed

| Component | Where |
|---|---|
| XGBoost model | `credit_risk_xgboost` v1 in the Model Registry, stage `Staging` |
| LogReg baseline | `credit_risk_logreg_baseline` v1 in the Model Registry, stage `Staging` |
| Tracking store | `mlruns/mlflow.db` (SQLite) |
| Artifact store | `mlruns/artifacts/` (local files) |
| Experiment | `CreditRisk` |
| Source script | `src/deploy_mlflow.py` |
| Verification script | `scripts/verify_mlflow.py` |
| Smoke-test inference | `scripts/load_mlflow_model.py` |

The MLflow runs log the following for each model:

- **Parameters** — every hyperparameter from `params.yaml` that was
  used to train the model (`n_estimators`, `max_depth`,
  `learning_rate`, `C`, `class_weight`, split fractions, random seed,
  evaluation threshold, etc.).
- **Metrics** — `val_*` and `test_*` versions of `roc_auc`, `pr_auc`,
  `precision`, `recall`, `f1`, `accuracy` derived from
  `reports/metrics.json`.
- **Tags** — `model_kind` (`primary`/`baseline`), `dataset`,
  `stage_hint`, `n_features`, and a `feature_columns_json` tag with
  the exact training feature list (so the run is self-describing).
- **Model artifact** — the actual XGBoost / sklearn model, serialized
  in MLflow's native flavor with a **model signature** (column schema)
  + **input example**. The model is also **registered** under a stable
  name so it can be referenced by version (`/1`, `/2`, ...) or by
  stage (`/Staging`, `/Production`).
- **Auxiliary files** — `feature_columns.json` and `metrics.json`
  attached as run artifacts.

---

## 2. How to deploy

Prereq: the MLSD `train` and `evaluate` stages must have produced
artifacts at least once (i.e. `dvc repro` has been run).

```bash
# 1. Make sure MLflow is installed
pip install -r requirements.txt   # adds mlflow>=2.10,<4

# 2. Run the deployment
python src/deploy_mlflow.py

# Optional flags:
#   --stage Production   # promote the new version to Production
#   --no-promote         # log only, do not transition to any stage
#   --tracking-uri ./mlruns   # change the location
```

The script:

1. Sets the tracking URI to `sqlite:///<tracking-uri>/mlflow.db` and
   the artifact root to `<tracking-uri>/artifacts/`.
2. Creates (or attaches to) the `CreditRisk` experiment.
3. Starts **two runs**: `xgboost_primary` and `logreg_baseline`,
   logging the parameters, metrics, tags, and feature columns.
4. Logs the model itself using `mlflow.xgboost.log_model` for the
   XGBoost model and `mlflow.sklearn.log_model` for the LogReg
   pipeline (with `skops_trusted_types` configured for safety).
5. Registers each model under a name:
   - XGBoost → `credit_risk_xgboost`
   - LogReg → `credit_risk_logreg_baseline`
6. Transitions the freshly created version to the chosen registry
   stage (`Staging` by default).

---

## 3. Inspecting the deployment

### 3a. Verify registry state

```bash
python scripts/verify_mlflow.py
```

Shows:

- All registered models with versions + stages.
- Every run in the `CreditRisk` experiment with the key metrics.
- A live load of `models:/credit_risk_xgboost/1` to prove the
  artifact is round-trippable.

### 3b. MLflow UI

```bash
mlflow server \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --default-artifact-root ./mlruns/artifacts \
    --host 127.0.0.1 \
    --port 5000
```

Then open <http://localhost:5000> for the full UI:
`Experiments → CreditRisk` shows the two runs; `Models` shows the
registry with v1 of both models. Stage transitions and model
comparison happen here.

### 3c. Smoke-test inference

```bash
python scripts/load_mlflow_model.py
    [--model-name credit_risk_xgboost]
    [--stage Staging]
    [--engineered data/processed/train_engineered.parquet]
    [--out reports/mlflow_predictions.csv]
```

Loads the registered XGBoost (or LogReg) model from the registry,
scores the engineered parquet (307,511 applicants), and writes a CSV
with `pred_proba_default` + the project's 3-way recommendation
(`Approve` / `Manual Review` / `Reject`). The output also recomputes
val + test ROC-AUC so you can sanity-check that the registered model
matches the original bytestream:

```text
val ROC-AUC  = 0.7545   (n=46,127)
test ROC-AUC = 0.7592   (n=46,127)
```

These match `reports/metrics.json` exactly.

---

## 4. Loading the model from a fresh machine

Any service can load the model with two values: a **tracking URI** and
a **model URI**:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///<path>/mlflow.db")  # or http://<server>
model = mlflow.xgboost.load_model("models:/credit_risk_xgboost/Production")
```

The native XGBoost flavor is used so `predict_proba` returns the
positive-class probability directly. For LogReg the same idea works
with `mlflow.sklearn.load_model`.

When the **stage** changes (e.g. `Staging` → `Production`), the model
URI above automatically resolves to the new version — no application
code changes are needed.

---

## 5. File map

| Path | Purpose |
|---|---|
| `src/deploy_mlflow.py` | Idempotent deployment script. Logs + registers both trained models. |
| `scripts/verify_mlflow.py` | Inspects the registry and confirms a model round-trip. |
| `scripts/load_mlflow_model.py` | End-to-end smoke test: load from registry, score, write CSV. |
| `mlruns/mlflow.db` | SQLite tracking + registry backend (created on first run). |
| `mlruns/artifacts/` | MLflow artifact storage (model binaries, plots, files). |
| `mlruns/` | Whole folder is gitignored — only `requirements.txt` references `mlflow`. |

---

## 6. Why a SQLite backend here?

MLflow 3 removed `mlflow migrate-filestore` support but still allows
file-only tracking for legacy deployments. We use the **recommended**
backend — a local SQLite database — because:

- It persists runs without external infrastructure.
- The Model Registry UI works correctly out of the box.
- It can be swapped for Postgres / MySQL without code changes
  (`mlflow.set_tracking_uri("postgresql://...")`).
- We retain the same artifact layout (`mlruns/artifacts/`).

For a remote / production deployment, replace `sqlite:///` with a
network backend URI and point `--default-artifact-root` at S3 / GCS /
Azure Blob. The `src/deploy_mlflow.py` script accepts `--tracking-uri`
for exactly this purpose.

---

## 7. Limitations & next steps

- **Local backend.** This is a single-machine deployment. A real
  production setup would point MLflow at a Postgres backend (e.g. AWS
  RDS / Supabase) and S3 for artifacts.
- **No CI hook.** Deployment is currently on-demand (`python
  src/deploy_mlflow.py`). Hooking `dvc repro` to auto-deploy on
  metric improvement is straightforward via MLflow's autolog
  integration.
- **No model serving container.** This is "model-registry" only;
  serving is via batch scoring (`scripts/load_mlflow_model.py`) or
  the existing Streamlit dashboard, which can be pointed at the
  MLflow registry by switching `dashboard/app.py` to use
  `mlflow.xgboost.load_model(...)`.
