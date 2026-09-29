# Credit Risk Intelligence — MLSD

> An end-to-end, **DVC-versioned** loan-default forecasting system
> built on the Home Credit Default Risk dataset.
>
> Originally a UIU Data Analytics Lab project (`DomainRange`); refactored
> for the MLSD course as a clean, parameterised, reproducible ML
> pipeline.

---

## 1. What is this?

This project predicts the probability that a loan applicant will
default, using the Home Credit Default Risk dataset
(`application_train.csv`, 307,511 applicants × 122 columns). The
**primary model is XGBoost** (val AUC ≈ 0.754, test AUC ≈ 0.758 at
threshold 0.50), benchmarked against a Logistic Regression baseline.

The MLSD pipeline is **reproducible end-to-end via DVC**:

```bash
dvc repro   # regenerate everything from scratch
dvc dag     # see the dependency graph
dvc metrics show   # see the test-set metrics
```

---

## 2. Problem statement

**Predict the probability that a loan applicant will default** (binary
classification on `TARGET`, ~8% positive class). The model output
feeds a 3-way recommendation policy:

| Probability of default | Recommendation |
|---|---|
| `p < 0.20` | **Approve** |
| `0.20 ≤ p < 0.50` | **Manual Review** |
| `p ≥ 0.50` | **Reject** |

The model is used in two places:

1. **Batch scoring** — score the entire portfolio to find high-risk
   segments (Segment Analysis tab of the dashboard).
2. **Single-applicant scoring** — given an applicant's form input,
   return a probability + recommendation (Applicant Risk Scorer tab).

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle) |
| File in scope | `application_train.csv` only |
| Rows | 307,511 |
| Columns | 122 (106 numeric + 16 categorical) |
| Target | `TARGET` (binary, ~8.07% positive) |
| Missing values | Yes — handled by median-impute + drop-if->60%-missing |
| Sentinels | `DAYS_EMPLOYED == 365243` for "employment unknown" — replaced with NaN before imputation |

**Other Home Credit tables** (bureau, previous_application,
POS_CASH_balance, etc.) are intentionally **out of scope** per the
DA project's master prompt. The MLSD pipeline does not load them.

---

## 4. Model

**Primary model: XGBoost** with class-weighted training
(`scale_pos_weight = neg/pos ≈ 11.39`).

**Why XGBoost?** The DA project evaluated four models on the same
data and the same train/val split:

| Model | Val AUC | Test AUC | Notes |
|---|---:|---:|---|
| **XGBoost** | **0.7539** | **0.7578** | Primary — chosen for production |
| Random Forest | 0.7469 | — | Strong baseline; not deployed |
| MLP (PyTorch) | 0.7448 | — | Underperformed XGBoost at higher cost |
| Logistic Regression | 0.7398 | — | Interpretable baseline |

XGBoost was selected because it achieved the highest validation AUC
**and** the highest test AUC. The MLP was excluded from the MLSD
pipeline per the course spec (no engineering reason to keep it in
the core pipeline). Random Forest and MLP are documented for the
DA project's history but not part of `dvc repro`.

**Baseline model: Logistic Regression** (median-impute + StandardScaler
fit on train slice + LR with `class_weight='balanced'`). Exists to
demonstrate the XGBoost uplift over a simple, interpretable model.

**Why not just XGBoost?** A baseline is required to show that the
chosen model meaningfully outperforms something simple. The XGBoost
val AUC is +0.014 over the LR baseline, which is real but small;
that's the honest story for this dataset.

---

## 5. Architecture

```
                  ┌──────────────┐
                  │   raw CSV    │  ← DVC-tracked (gitignored)
                  │ (DVC hash)   │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   prepare    │  → cleaning, encoding, imputation,
                  │              │    stratified 70/15/15 split,
                  │              │    StandardScaler fit on train only
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   features   │  → 7 engineered columns,
                  │              │    top-30 selection via XGB+RF+SHAP
                  │              │    fit on TRAIN slice only
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    train     │  → XGBoost (primary) + LR (baseline)
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   evaluate   │  → metrics.json + ROC/PR/confusion
                  │              │    plots, NEVER retrains
                  └──────────────┘
```

Full architecture rationale: [`docs/MLSD_ARCHITECTURE.md`](docs/MLSD_ARCHITECTURE.md).

---

## 6. Project structure

```
DA-Lab-Project/
├── data/
│   ├── raw/                    ← DVC-tracked dataset lives here (gitignored)
│   └── processed/              ← DVC outputs of the `prepare` and `features` stages
├── models/                     ← DVC outputs of the `train` stage
├── reports/
│   ├── cleaning_summary.md     ← DVC output of `prepare`
│   ├── selected_features.json  ← DVC output of `features`
│   ├── metrics.json            ← DVC METRIC of `evaluate` (shown via `dvc metrics show`)
│   ├── evaluation/             ← ROC/PR/confusion plots
│   ├── figures/                ← DA project charts (preserved)
│   ├── eda_findings.md         ← DA project EDA (preserved)
│   ├── feature_importance.md   ← DA project feature ranking (preserved)
│   └── phase4_model_comparison.md ← DA project 4-model table (preserved)
├── src/
│   ├── data/                   ← loader, cleaner, profiler (DA project, preserved)
│   ├── features/               ← engineer, select (DA project, preserved)
│   ├── models/                 ← DA project trainers (preserved) + score.py (preserved)
│   ├── prepare.py              ← NEW: MLSD `prepare` stage
│   ├── features.py             ← NEW: MLSD `features` stage
│   ├── train.py                ← NEW: MLSD `train` stage
│   └── evaluate.py             ← NEW: MLSD `evaluate` stage
├── dashboard/                  ← Streamlit dashboard (DA project, preserved)
├── notebooks/                  ← DA project notebooks (preserved)
├── tests/                      ← DA tests (preserved) + NEW MLSD pipeline tests
├── scripts/
│   └── _make_synthetic_for_validation.py  ← dev-only synthetic data
├── docs/
│   ├── MLSD_ARCHITECTURE.md    ← architecture decisions
│   ├── DVC_VIVA_GUIDE.md       ← DVC commands + viva prep
│   ├── REPRODUCIBILITY.md      ← how to reproduce from clean clone
│   ├── EXPERIMENTS.md          ← experiments log
│   └── PHASE_CHECKLIST.md      ← DA project cross-check (preserved)
├── dvc.yaml                    ← pipeline definition (4 stages)
├── dvc.lock                    ← pipeline output hashes (git-tracked)
├── params.yaml                 ← every hyperparameter
├── .dvcignore                  ← DVC ignore rules
├── .dvc/                       ← DVC cache + state (gitignored)
├── requirements.txt
└── README.md                   ← this file
```

---

## 7. Setup

```bash
# 1. Clone
git clone <repo-url>
cd DA-Lab-Project

# 2. Python venv (Python 3.14 recommended; 3.11+ should work)
python -m venv .venv

# Activate:
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   Windows (Git Bash):    .venv/Scripts/activate
#   Linux/macOS:           source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Get the data
#    Option A: DVC remote
dvc pull
#    Option B: Manual
#      Download application_train.csv from Kaggle and place at:
#        data/raw/application_train.csv
#      Then:
dvc add data/raw/application_train.csv

# 5. Run the pipeline
dvc repro

# 6. See metrics
dvc metrics show
```

---

## 8. DVC commands you'll need for the viva

| Command | What it does |
|---|---|
| `dvc init` | Initialise DVC in a Git repo (already done) |
| `dvc add <file>` | Track a file with DVC (creates `<file>.dvc`) |
| `dvc repro` | Run only the dirty stages |
| `dvc status` | Show which stages are dirty |
| `dvc dag` | Print the pipeline dependency graph |
| `dvc push` | Upload DVC-tracked files to the remote |
| `dvc pull` | Download DVC-tracked files from the remote |
| `dvc metrics show` | Display `reports/metrics.json` |
| `dvc params diff` | Show parameter changes vs the last run |
| `dvc dag --md` | DAG as a Mermaid markdown block |

Full explanation with project-specific examples:
[`docs/DVC_VIVA_GUIDE.md`](docs/DVC_VIVA_GUIDE.md).

---

## 9. dvc.yaml — what each stage does

```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps: [src/prepare.py, src/data/load.py, src/data/clean.py,
           data/raw/application_train.csv]
    params: [data.random_state, data.train_size, data.validation_size,
             data.test_size, preprocessing.missing_threshold,
             preprocessing.high_cardinality_threshold]
    outs: [data/processed/train_clean.parquet,
           data/processed/split_indices.npz,
           reports/cleaning_summary.md]

  features:
    cmd: python src/features.py
    deps: [src/features.py, src/features/engineer.py,
           src/features/select.py, data/processed/train_clean.parquet,
           data/processed/split_indices.npz]
    params: [features.top_n, data.random_state]
    outs: [data/processed/train_engineered.parquet,
           reports/selected_features.json]

  train:
    cmd: python src/train.py
    deps: [src/train.py, data/processed/train_engineered.parquet,
           reports/selected_features.json]
    params: [model.*, baseline.*, data.random_state]
    outs: [models/xgboost_model.pkl,
           models/baseline_logistic_regression.pkl,
           models/feature_columns.json]

  evaluate:
    cmd: python src/evaluate.py
    deps: [src/evaluate.py, models/xgboost_model.pkl,
           models/baseline_logistic_regression.pkl,
           models/feature_columns.json,
           data/processed/train_engineered.parquet]
    params: [evaluation.*, data.random_state]
    metrics: [reports/metrics.json]
    outs: [reports/evaluation/roc_curves.png,
           reports/evaluation/pr_curves.png,
           reports/evaluation/confusion_matrices.png]
```

The DAG (`dvc dag`) shows the four stages connected linearly with
one branching edge — `features` is read by both `train` and
`evaluate`.

---

## 10. params.yaml — what each parameter means

```yaml
data:           # split fractions + seed
  random_state: 42
  train_size: 0.70
  validation_size: 0.15
  test_size: 0.15

preprocessing:  # cleaning thresholds
  missing_threshold: 0.60           # drop cols >60% missing
  high_cardinality_threshold: 10    # >10 unique -> freq-encode

features:
  top_n: 30                         # selected feature count

model:           # XGBoost (primary)
  type: xgboost
  n_estimators: 300
  max_depth: 4
  learning_rate: 0.05
  eval_metric: logloss
  tree_method: hist
  random_state: 42
  # scale_pos_weight = neg/pos computed at fit-time from train labels

baseline:        # Logistic Regression
  type: logistic_regression
  C: 1.0
  max_iter: 200
  class_weight: balanced
  random_state: 42

evaluation:
  primary_metric: roc_auc
  threshold: 0.5
```

To demonstrate parameter experiments in the viva:

```bash
# Edit params.yaml: model.learning_rate: 0.05 -> 0.10
$ dvc status
train:  changed deps: model.learning_rate

$ dvc repro
# Only train + evaluate rerun.

# Restore
$ git checkout -- params.yaml
$ dvc repro
```

---

## 11. Reproduction

The complete reproduction guide lives in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Summary:

```bash
git clone <repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull                 # or place the CSV at data/raw/ manually
dvc repro                # ~10-30 minutes on a laptop
dvc metrics show         # see results
```

If `dvc repro` is a no-op ("Data and pipelines are up to date."),
the cached outputs are already valid and no rerun is needed.

---

## 12. Tests

```bash
# All tests (requires xgboost, scikit-learn, pandas, numpy, pyarrow)
python -m pytest tests/ -v

# The MLSD pipeline tests (don't require torch or streamlit)
python -m pytest tests/test_mlsd_pipeline.py -v
```

Currently:

- 9 DA tests (`test_clean.py` + `test_engineer.py`)
- 12 MLSD pipeline tests (`test_mlsd_pipeline.py`)
- 8 score tests (`test_score.py` — requires `torch`)
- 6 dashboard tests (`test_dashboard.py` — requires `streamlit`)

The score + dashboard tests are part of the DA project and require
their own dependencies. The MLSD pipeline is independent of them.

---

## 13. Results (actual)

After `dvc repro` on the full Kaggle dataset (307,511 rows × 122
columns), with the production config from `params.yaml`:

| Model | Val ROC-AUC | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|
| **XGBoost** | **0.7545** | **0.7592** | **0.2440** |
| Logistic Regression (baseline) | 0.7416 | 0.7438 | 0.2202 |

These match the DA project's Phase 4 winner (val 0.7539 / test 0.7578)
within expected variance — the small upward shift comes from running
the new pipeline end-to-end with the strict train-only feature-selection
fix applied (the DA project leaked val/test info into the selection).

XGBoost beats the LR baseline by **+0.015 AUC** on both val and test —
real but small, which is the honest story for this dataset.

To see the live numbers:

```bash
dvc metrics show
```

Confusion matrices at threshold 0.5 (XGBoost, test set):

```
TN = 29,758   FP = 12,645
FN =  1,168   TP =  2,556
```

ROC / PR / confusion matrix plots:

```
reports/evaluation/roc_curves.png
reports/evaluation/pr_curves.png
reports/evaluation/confusion_matrices.png
```

---

## 14. Dashboard

The Streamlit dashboard is preserved from the DA project. It
**reads** the model artifact produced by the MLSD pipeline; it does
**not** retrain.

```bash
.venv/Scripts/python -m streamlit run dashboard/app.py
# then open http://localhost:8501
```

Tabs:

1. **Portfolio Overview** — KPIs, class balance, model comparison
2. **Applicant Risk Scorer** — 14-widget form → live score
3. **Feature Importance** — top-30 features with category filter
4. **Segment Analysis** — default-rate-by-segment with 95% CI

The dashboard reads `models/xgboost_model.pkl`. If you've just
run `dvc repro`, the artifact is there.

---

## 15. Limitations

- **Single dataset, single split.** The project evaluates on one
  70/15/15 stratified split. Confidence intervals are not computed.
- **No temporal validation.** `application_train.csv` has no
  reliable per-row timestamp, so we cannot validate that the model
  generalizes across time. Real drift detection is therefore out of
  scope, but the PSI math is implemented as a teaching artifact — see
  [`docs/DRIFT.md`](docs/DRIFT.md).
- **No class-weight learning.** `scale_pos_weight = neg/pos` is a
  default that the DA project verified empirically; we did not
  re-tune it.
- **Synthetic-data validation only.** The repo does not include the
  raw Kaggle CSV (160 MB, gitignored). Use `dvc pull` or place the
  CSV manually.
- **Threshold is hard-coded at 0.5** for the policy-reporting
  metric. The DA project's 3-way recommendation policy
  (p<0.20 / 0.20-0.50 / ≥0.50) is preserved in
  `src/models/score.py`.

---

## 16. Viva notes

Most-likely viva questions, answered:

- **"What does `dvc repro` do?"** Runs only the pipeline stages that
  have stale inputs (deps, params) or missing outputs.
- **"Why DVC?"** Git can't version large files. DVC adds
  content-addressed storage + a pipeline definition +
  experiment tracking, all compatible with Git.
- **"How does DVC know which stage needs to rerun?"** Hashes. Each
  stage's lock entry hashes its `deps` + `params` + previous
  `outs`. If any hash differs, the stage reruns.
- **"What happens if the dataset changes?"** DVC detects the new
  MD5 hash. `prepare` reruns; downstream stages cascade.
- **"What happens if a parameter changes?"** DVC detects the new
  value. Only stages that read that parameter (and the stages
  downstream of their outputs) rerun.
- **"Difference between Git and DVC?"** Git versions source code;
  DVC versions data + ML pipeline state. Both share the same
  directory; they don't conflict.
- **"What do `dvc push` and `dvc pull` do?"** `push` uploads DVC-
  tracked files to a remote (S3, GDrive, local). `pull` downloads
  them. Neither touches source code.
- **"How can another person reproduce?"** `git clone` + `pip
  install -r requirements.txt` + `dvc pull` + `dvc repro`.

Full viva prep: [`docs/DVC_VIVA_GUIDE.md`](docs/DVC_VIVA_GUIDE.md).

---

## 17. Acknowledgements

- **Course:** UIU Data Analytics Laboratory (original project) →
  MLSD (refactor).
- **Dataset:** Home Credit Default Risk (Kaggle).
- **DA project authors:** Musfique Ahmed, Tasfiya Binte Karim.
- **MLSD refactor:** same group, refactored into a DVC-versioned
  pipeline.
