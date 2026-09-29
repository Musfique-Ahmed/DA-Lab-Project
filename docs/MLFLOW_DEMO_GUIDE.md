# MLflow Demo Guide — Project Walkthrough

> Companion to `docs/MLFLOW_DEPLOYMENT.md` (architecture + all
> commands). This file is a **presenter's playbook**: open this doc,
> run the commands in order, and you have a complete demo of how the
> Credit Risk Intelligence project uses MLflow.

---

## 0. TL;DR — three commands

```bash
.venv/Scripts/python.exe src/deploy_mlflow.py              # 1) log + register
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts --host 127.0.0.1 --port 5000   # 2) UI
.venv/Scripts/python.exe scripts/load_mlflow_model.py       # 3) smoke test
```

That's the whole demo. The rest of this guide explains what each step
proves, the exact commands with expected output, and how to talk
through them.

---

## 1. What "deployed to MLflow" means in this project

| Concept | In this repo |
| --- | --- |
| **Experiment** | `CreditRisk` (one for the whole project) |
| **Runs** | `xgboost_primary`, `logreg_baseline` — every re-deploy makes a new pair |
| **Tracking backend** | SQLite at `mlruns/mlflow.db` |
| **Artifact store** | Local files at `mlruns/artifacts/` |
| **Model Registry** | `credit_risk_xgboost` and `credit_risk_logreg_baseline`, each at v2, stage `Staging` |
| **Logged per run** | parameters, metrics (val + test), tags, feature-column list, the model artifact, a feature-engineered parquet |
| **Model flavor** | `mlflow.xgboost` for the primary, `mlflow.sklearn` for the baseline |

Two named registered models, each with one current "Staging" version,
and the entire pipeline is reproducible from the SQLite db plus
artifacts.

---

## 2. Pre-flight checklist (60 seconds)

```bash
# Activate the venv (Windows / Git Bash — adjust for your shell)
.venv/Scripts/python.exe -c "import mlflow; print(mlflow.__version__)"  # expect 3.16.x
.venv/Scripts/python.exe -c "import xgboost; print(xgboost.__version__)" # expect 2.1.x
.venv/Scripts/python.exe -c "import sklearn; print(sklearn.__version__)" # expect 1.9.x
```

If `mlflow` is missing:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

If the DVC pipeline hasn't been run yet, do that first — MLflow
deploys the model pickles that the DVC `train` stage produces:

```bash
.venv/Scripts/dvc pull                         # if a fresh clone
.venv/Scripts/dvc status                       # should say "up to date" or run `dvc repro`
```

---

## 3. Step 1 — Deploy the models

```bash
.venv/Scripts/python.exe src/deploy_mlflow.py
```

What the script does (read the relevant block out loud while it runs):

1. Sets the tracking URI to the SQLite file and the artifact root to
   `mlruns/artifacts/`.
2. Creates (or attaches to) the `CreditRisk` experiment.
3. Opens **two MLflow runs** — `xgboost_primary` and `logreg_baseline`.
4. Logs every hyperparameter from `params.yaml` plus every val/test
   metric from `reports/metrics.json`.
5. Logs the model itself using the native `mlflow.xgboost.log_model`
   / `mlflow.sklearn.log_model` flavor, attaching the **signature**
   (column schema) and an **input example**.
6. Registers each model:
   - `credit_risk_xgboost` → currently **v2, stage Staging**
   - `credit_risk_logreg_baseline` → currently **v2, stage Staging**

Expected tail of output:

```text
[deploy] XGBoost model URI: models:/m-…
[deploy] LogReg model URI: models:/m-…
[deploy] Transitioning credit_risk_xgboost v2 -> Staging
[deploy] Transitioning credit_risk_logreg_baseline v2 -> Staging
[deploy] Done.
```

> **Note on versions.** Every re-run bumps the version number (`v3`,
> `v4`, …). That's intentional — MLflow keeps an immutable history of
> every artifact. Old versions are still loadable; just swap the
> number in the model URI.

### Optional flags for live demos

```bash
# Promote straight to Production (skip Staging)
.venv/Scripts/python.exe src/deploy_mlflow.py --stage Production

# Log the runs without registering (just for ad-hoc experiments)
.venv/Scripts/python.exe src/deploy_mlflow.py --no-promote
```

---

## 4. Step 2 — Open the MLflow dashboard

The repo uses the recommended `mlflow server` (SQLite-backed) rather
than the legacy `mlflow ui` (file-only) so the **Model Registry UI**
works correctly.

```bash
mlflow server ^
    --backend-store-uri sqlite:///mlruns/mlflow.db ^
    --default-artifact-root ./mlruns/artifacts ^
    --host 127.0.0.1 ^
    --port 5000
```

> On macOS/Linux replace `^` with `\`.

Open <http://localhost:5000>.

### What to click in front of the audience

| Tab / page | What to show | Why it matters |
| --- | --- | --- |
| **Experiments → CreditRisk** | Two runs: `xgboost_primary`, `logreg_baseline`. Click a run to see its params, metrics, tags, and the artifacts (model, feature_columns.json, metrics.json, train_engineered.parquet). | Demonstrates **experiment tracking** — every parameter + every metric is queryable. |
| **Models** | `credit_risk_xgboost` v2 (Staging) and `credit_risk_logreg_baseline` v2 (Staging). | Demonstrates the **model registry** — a single source of truth for "which model are we shipping?" |
| **Model version page** | Lineage (run id), signature, schema. Click the **Stage** dropdown to transition. | Demonstrates **stage management** — Staging → Production is one click. |
| **Compare runs** | Pick both runs, click "Compare". Side-by-side metrics + parallel-coordinates plot. | Demonstrates **model comparison**. |

### Stopping the server

`Ctrl+C` in the terminal. The SQLite db and artifacts persist.

---

## 5. Step 3 — Verify from the CLI

```bash
.venv/Scripts/python.exe scripts/verify_mlflow.py
```

Output to expect:

```text
Tracking URI: sqlite:///E:/…/mlruns/mlflow.db

Registered models:
  - credit_risk_logreg_baseline
      v2 | stage=Staging | run_id=…
  - credit_risk_xgboost
      v2 | stage=Staging | run_id=…

Experiments:
  - id=1 name=CreditRisk artifact_location=file:…

Recent runs:
  run_id=…  name=xgboost_primary
    kind=primary
    test_roc_auc=0.7592
    val_roc_auc =0.7545
  run_id=…  name=logreg_baseline
    kind=baseline
    test_roc_auc=0.7438
    val_roc_auc =0.7416

Loading XGBoost model from registry 'credit_risk_xgboost' v2 ...
  Loaded via booster workaround: XGBClassifier
OK: MLflow model is round-trip-loadable.
```

> **About that warning line.** `mlflow.xgboost.load_model` in
> MLflow 3.x + xgboost ≥ 2.1 hits a known incompatibility
> (`_estimator_type undefined`) because newer xgboost removed the
> implicit sklearn-mixin attribute. The script falls back to loading
> the booster JSON directly and wrapping it in a fresh
> `XGBClassifier` — proving the **stored artifact is the same one
> `train.py` wrote**, just round-tripped through the registry.

---

## 6. Step 4 — Score real applicants from the registry

```bash
.venv/Scripts/python.exe scripts/load_mlflow_model.py ^
    --engineered data/processed/train_engineered.parquet ^
    --out reports/mlflow_predictions.csv
```

This script:

1. Resolves `models:/credit_risk_xgboost/Staging`.
2. Loads the model from the registry (same workaround as Step 5).
3. Reads all 307,511 applicants from the engineered parquet.
4. Produces `pred_proba_default` plus a **3-way recommendation**
   (`Approve` / `Manual Review` / `Reject`).
5. Recomputes **val** and **test** ROC-AUC and prints them. If the
   model is the same bytestream, the numbers match `reports/metrics.json`
   to the 4th decimal — your evidence that the registry round-trips.

Expected tail:

```text
recommendation
Manual Review    155369
Reject           101611
Approve           50531
Name: count, dtype: int64
  val ROC-AUC = 0.7545 (n=46,127)
  test ROC-AUC = 0.7592 (n=46,127)
```

---

## 7. Talking points (60-second pitch)

1. **What is MLflow here?** "An experiment-tracking + model-registry
   layer on top of our DVC pipeline. DVC owns the data; MLflow owns
   the trained-model artifacts and the metadata around them."
2. **Why both?** "DVC guarantees the data and code at training time;
   MLflow guarantees the model itself — parameters, metrics,
   signature, version, and registry stage — are queryable and
   reproducible without re-training."
3. **What's a model signature?** "The schema of the input the model
   expects — 30 floats in a specific order. MLflow captures it
   automatically so a serving layer can validate requests without
   re-implementing the contract."
4. **Why Staging and not Production?** "Staging = it's our current
   best model. Promoting to Production is a one-click decision in the
   UI, gated by whatever review process the team uses."
5. **What if I want to compare a parameter change?** "Use
   `mlflow experiments` or re-run `deploy_mlflow.py` with a bumped
   param — the new version becomes a sibling in the registry, no
   overwrites, history preserved."

---

## 8. Optional — live A/B demo with `mlflow experiments`

If you want to demo the experiment-tracking depth:

```bash
# 1) tweak the params
notepad params.yaml
#   features.top_n: 30 -> 50

# 2) re-run the DVC pipeline (it'll only rerun features/train/evaluate)
.venv/Scripts/dvc repro

# 3) re-deploy to MLflow — this becomes v3 alongside v2
.venv/Scripts/python.exe src/deploy_mlflow.py

# 4) Compare v2 and v3 in the UI: Models → credit_risk_xgboost → compare
```

Or use the lightweight experiment runner (no DVC rerun needed):

```bash
.venv/Scripts/mlflow experiments csv --experiment-id 1 --output-file reports/mlflow_runs.csv
```

Open the CSV in Excel — one row per run with every metric. Quick
"show me the history" artefact for the viva.

---

## 9. Files involved

| Path | Role |
| --- | --- |
| `src/deploy_mlflow.py` | Deployment script (logs + registers both models) |
| `scripts/verify_mlflow.py` | CLI verification + round-trip load test |
| `scripts/load_mlflow_model.py` | End-to-end smoke test (load → score → CSV) |
| `mlruns/mlflow.db` | SQLite tracking + registry backend |
| `mlruns/artifacts/` | Models, feature lists, parquets |
| `requirements.txt` | Pins `mlflow>=2.10,<4` |
| `docs/MLFLOW_DEPLOYMENT.md` | Architecture-level reference (sister doc) |
| `docs/MLFLOW_DEMO_GUIDE.md` | **This file** — presenter's playbook |

---

## 10. Common questions you might get

**"Where is the serving endpoint?"**
There isn't one — this repo ships a **registry**, not a hosted
endpoint. The same pattern (`mlflow.<flavor>.load_model`) is used by
the Streamlit dashboard to score applicants in-process, and the
command-line smoke test (`scripts/load_mlflow_model.py`) does the
batch scoring. To go to production you'd add `mlflow models serve`
behind a reverse proxy.

**"What about CI/CD?"**
Out of scope for this phase. The deployment is idempotent
(`src/deploy_mlflow.py`) so wiring it into a GitHub Actions job is a
~20-line yml file.

**"Could you swap SQLite for Postgres?"**
Yes — pass `--tracking-uri postgresql://…` to
`deploy_mlflow.py`. Nothing in the rest of the code needs to change.

**"Why are you not using MLflow 3's model alias feature?"**
MLflow 3 deprecated the `stage` column in favor of aliases
(`@champion`, `@challenger`, etc.). We deliberately stay on the
`Staging`/`Production` vocabulary because that's what the project
documentation already uses — the API is otherwise identical and the
migration is a one-line search-and-replace when the team is ready.

---

*Tested end-to-end on: mlflow 3.16.1, xgboost 2.1.4, scikit-learn 1.9.1,
pandas 2.3.3, Python 3.x (venv: `.venv`).*
