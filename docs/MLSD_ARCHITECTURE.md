# MLSD Architecture — Credit Risk Intelligence

This document is the single source of truth for **how the MLSD pipeline is
organized** and **why**. Read this once before touching any pipeline code.

The MLSD project is a refactor of the existing DA project (see
[`PHASE_CHECKLIST.md`](PHASE_CHECKLIST.md) for what was preserved). It is
**not** a rewrite. The DA project's notebooks, dashboard, and tests still
exist; the MLSD pipeline lives alongside them as a clean, reproducible
subset that can be regenerated from scratch with `dvc repro`.

---

## 1. Goals

1. **Reproducible end-to-end pipeline.** `dvc repro` from a clean clone
   must regenerate the cleaned dataset, the engineered features, the
   trained models, the evaluation metrics, and the evaluation plots.
2. **Parameterised experiments.** All hyperparameters live in
   `params.yaml`. Changing one parameter and re-running `dvc repro` must
   cause only the affected downstream stages to re-execute.
3. **No data leakage.** Every transformation that learns from data
   (imputer medians, scaler mean/std, feature selection) is fit on the
   **train slice only**. The validation set is used only for the
   threshold-policy sanity check; the test set is used **once**, in the
   `evaluate` stage.
4. **Understandable in a viva.** Four stages, each with a single clear
   responsibility. No clever metaprogramming, no hidden notebook
   dependencies, no silently-tunable defaults.

## 2. The pipeline at a glance

```
                  ┌──────────────┐
                  │   raw CSV    │  ← DVC-tracked, gitignored
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
                  │   features   │  → engineer 7 ratio/age columns,
                  │              │    select top-N via XGB+RF+SHAP,
                  │              │    all fit on the train slice only
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    train     │  → XGBoost (primary) + Logistic
                  │              │    Regression (baseline)
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   evaluate   │  → metrics.json + ROC/PR/confusion
                  │              │    plots, never retrains
                  └──────────────┘
```

## 3. Stage responsibilities

### `prepare` — cleaning + split

**Inputs:** `data/raw/application_train.csv` (DVC-tracked),
`params.yaml` `data` and `preprocessing` sections.

**What it does:**

1. Load the raw CSV.
2. Replace `DAYS_EMPLOYED == 365243` with NaN (the canonical sentinel
   for "employment unknown").
3. Drop columns whose missing fraction exceeds `missing_threshold`
   (default 0.60).
4. Median-impute remaining numeric NaNs; `'MISSING'`-impute categorical
   NaNs.
5. Encode categoricals:
   - low-cardinality (≤ `high_cardinality_threshold` unique) → one-hot
   - high-cardinality (> threshold) → frequency encoding
6. Stratified 70/15/15 split on `TARGET` (random_state = 42).
7. Fit `StandardScaler` on the **train slice only**, transform all
   three slices.

**Outputs:**

- `data/processed/train_clean.parquet` (full 307,511 × 144 with
  `TARGET`, `SK_ID_CURR`, `SPLIT` intact, all features scaled)
- `data/processed/split_indices.npz` (boolean masks for train/val/test)
- `reports/cleaning_summary.md` (human-readable summary)

**Why a single stage?** Cleaning, encoding, imputation, splitting, and
scaling are tightly coupled — they all share the same
"fit-once-on-train" discipline, and the outputs of any one are inputs
to the next. Splitting them into separate DVC stages would require
serialising intermediate dataframes and would obscure the data
discipline.

### `features` — engineering + top-N selection

**Inputs:** `data/processed/train_clean.parquet`,
`reports/cleaning_summary.md` (DVC dependency), `params.yaml`
`features` section.

**What it does:**

1. Apply the 7 engineered features from `src/features/engineer.py`:
   `AGE_YEARS`, `EMPLOYED_YEARS`, `CREDIT_INCOME_RATIO`,
   `ANNUITY_INCOME_RATIO`, `CREDIT_GOODS_RATIO`,
   `INCOME_PER_FAM_MEMBER`, `EXT_SOURCE_MEAN`.
2. **Restrict to the train slice** (`SPLIT == 'train'`).
3. Run `src/features/select.py::train_xgb_and_rf_importances` on the
   train slice → XGB + RF importance ranks.
4. Run `src/features/select.py::shap_values_xgb` on a 5,000-row
   subsample of the train slice → SHAP rank.
5. Combine the three ranks into a mean-rank, take the top `n` features
   (default 30).
6. Write the full engineered matrix (all rows, all splits) to
   `data/processed/train_engineered.parquet`.
7. Write the top-N feature list to
   `reports/selected_features.json`.

**Outputs:**

- `data/processed/train_engineered.parquet`
- `reports/selected_features.json`

**Leakage fix from Phase 0:** in the existing DA project, feature
selection was run on the entire cleaned matrix (train+val+test). This
is fixed here: selection is strictly fit on `SPLIT == 'train'`. The
test set is never seen by the feature selector.

### `train` — XGBoost + Logistic Regression baseline

**Inputs:** `data/processed/train_engineered.parquet`,
`reports/selected_features.json`, `params.yaml` `model` section.

**What it does:**

1. Load the engineered parquet; project to the top-N columns.
2. Slice into train/val/test using the saved `SPLIT` column.
3. Compute `scale_pos_weight = neg/pos` from the **train labels** (this
   is a value, not a transformation — it doesn't see test data).
4. Train XGBoost on the train slice with the parameters from
   `params.yaml` (default: `n_estimators=300`, `max_depth=4`,
   `learning_rate=0.05`, `eval_metric='logloss'`, `tree_method='hist'`,
   `random_state=42`).
5. Train Logistic Regression on the train slice (median-impute +
   StandardScaler fit on train only, since LR doesn't accept NaN).
6. Save both artifacts:
   - `models/xgboost_model.pkl` — primary model
   - `models/baseline_logistic_regression.pkl` — baseline
   - `models/feature_columns.json` — the exact column list used (so
     evaluate can reproduce the same X without drift)
7. Print val-AUC for both models as a sanity check.

**Outputs:**

- `models/xgboost_model.pkl`
- `models/baseline_logistic_regression.pkl`
- `models/feature_columns.json`

**Why two models?** The rubric requires a baseline + a primary model.
XGBoost is the primary (per Phase 4 winner: val AUC 0.7539, test AUC
0.7578). Logistic Regression is the interpretable baseline (per Phase 4
results: val AUC 0.7398). The MLP was dropped per the MLSD spec — see
the [DA project history](PHASE_CHECKLIST.md) for why MLP was used in
the DA project.

### `evaluate` — metrics + plots

**Inputs:** `models/xgboost_model.pkl`,
`models/baseline_logistic_regression.pkl`,
`models/feature_columns.json`,
`data/processed/train_engineered.parquet`,
`reports/selected_features.json`, `params.yaml` `evaluation` section.

**What it does:**

1. Load both model artifacts (cached at module level).
2. Load the engineered parquet; project to the top-N columns; slice
   into val and test.
3. Compute metrics on both val and test for both models:
   - `roc_auc` (primary metric)
   - `pr_auc` (precision-recall AUC, useful for imbalanced data)
   - `precision`, `recall`, `f1`, `accuracy` at the threshold from
     `params.yaml`
   - `confusion_matrix` (TN, FP, FN, TP)
4. Write `reports/metrics.json` — DVC tracks this file as a metric, so
   `dvc metrics show` displays it.
5. Save plots to `reports/evaluation/`:
   - `roc_curves.png` — both models, both val and test
   - `pr_curves.png` — both models, both val and test
   - `confusion_matrices.png` — both models, test set

**Outputs:**

- `reports/metrics.json` ← `metrics:` in `dvc.yaml`
- `reports/evaluation/roc_curves.png`
- `reports/evaluation/pr_curves.png`
- `reports/evaluation/confusion_matrices.png`

**Strict rule:** this stage **never retrains**. It only loads
artifacts produced by `train` and computes metrics on data slices that
are guaranteed untouched by training (the val/test slices from the
`prepare` stage's split).

## 4. Data discipline — the no-leakage contract

| Transformation | Fit on | Transform on |
|---|---|---|
| Imputation medians | train slice | train + val + test |
| Frequency encoding (high-cardinality categoricals) | full dataset | full dataset (unsupervised; doesn't see TARGET) |
| One-hot encoding | full dataset | full dataset (unsupervised; doesn't see TARGET) |
| `StandardScaler` mean/std | train slice | train + val + test |
| Feature selection (XGB / RF / SHAP ranks) | train slice | produces a column list, not a transformed matrix |
| `scale_pos_weight = neg/pos` | train labels | used as a scalar in `train_xgboost` |
| Threshold policy (`0.20` / `0.50`) | val set (single number, not learned) | used in `score_application` for production recommendations |

The test slice is **never used to fit anything**. It is touched only in
the `evaluate` stage, where it is used to compute metrics for the final
report.

## 5. The split

The DA project uses **70 / 15 / 15 stratified** with `random_state=42`.
This is preserved unchanged:

- **Train (70%):** fit all supervised transformations and all models.
- **Validation (15%):** tune threshold policy, compare models.
- **Test (15%):** compute the final reported metrics, **once**.

The split is recorded in `data/processed/split_indices.npz` and in the
`SPLIT` column of the cleaned parquet. Any future refactor must
preserve the same masks (same seed → same row indices) or the
comparison with the DA project's results becomes invalid.

## 6. The artifacts

| Path | Produced by | Consumed by |
|---|---|---|
| `data/raw/application_train.csv` | human (download from Kaggle) | `prepare` |
| `data/processed/train_clean.parquet` | `prepare` | `features` |
| `data/processed/split_indices.npz` | `prepare` | downstream (sanity check) |
| `reports/cleaning_summary.md` | `prepare` | human |
| `data/processed/train_engineered.parquet` | `features` | `train`, `evaluate` |
| `reports/selected_features.json` | `features` | `train`, `evaluate` |
| `models/xgboost_model.pkl` | `train` | `evaluate`, dashboard |
| `models/baseline_logistic_regression.pkl` | `train` | `evaluate` |
| `models/feature_columns.json` | `train` | `evaluate`, dashboard |
| `reports/metrics.json` | `evaluate` | `dvc metrics show` |
| `reports/evaluation/*.png` | `evaluate` | human |

## 7. Why this design

- **Four stages** is the smallest number that reflects the actual
  data discipline. Combining `prepare` + `features` would mix "fit
  on full dataset for encoding" with "fit on train for selection,"
  which is hard to reason about.
- **One model artifact per stage output** keeps the DAG honest. The
  `evaluate` stage depends on the model file's hash, so any change to
  training invalidates the cached metrics automatically.
- **`metrics:` in `dvc.yaml`** wires the final scorecard into
  `dvc metrics show`, which is what the rubric expects.
- **The Streamlit dashboard is independent of DVC.** It only reads
  `models/xgboost_model.pkl`. If DVC is not configured (e.g., on a
  fresh clone without a remote), the dashboard still works as long as
  the artifact has been generated at least once.

## 8. What this design explicitly does NOT do

- **No online learning.** The dataset is a single CSV snapshot.
- **No drift detection.** The dataset has no reliable timestamp per
  applicant; per-row ordering is not temporal.
- **No Feast / feature store.** Not required by the rubric; would
  add infra without changing reproducibility.
- **No model registry.** The artifact is a single `.pkl` file in
  `models/`. DVC's content-addressed cache is the registry.
- **No PyTorch MLP.** Excluded per the MLSD spec; the source files
  remain in `src/models/` for the DA project.
