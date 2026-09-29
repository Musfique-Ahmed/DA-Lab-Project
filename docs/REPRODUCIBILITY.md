# Reproducibility — Credit Risk Intelligence

This document explains, in detail, **how another student can reproduce
the exact MLSD pipeline from a clean clone**. Read this end-to-end
before the viva; it covers both the happy path and the failure modes.

---

## 1. The four sources of reproducibility

1. **Git** — source code, `dvc.yaml`, `params.yaml`, documentation.
2. **DVC** — the raw dataset, intermediate parquets, model artifacts,
   metrics.
3. **`params.yaml`** — every hyperparameter and data-prep threshold
   used by the pipeline.
4. **`dvc.lock`** — the cryptographic record of what the pipeline
   produced last.

Together, these four constitute the "experiment" in a way that can
be re-run bit-for-bit on another machine with the same Python
version.

---

## 2. The clean-clone happy path

```bash
# Step 1: Get the code
git clone <repository-url> credit-risk-intelligence
cd credit-risk-intelligence

# Step 2: Set up Python (Python 3.14 is the project's target; older
# 3.11+ should also work)
python -m venv .venv

# Activate:
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   Windows (Git Bash):    .venv/Scripts/activate
#   Linux/macOS:           source .venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Get the data
# Option A: from a configured DVC remote
dvc pull
# Option B: from Kaggle (manual)
#   1. Download application_train.csv from
#      https://www.kaggle.com/c/home-credit-default-risk/data
#   2. Place it at data/raw/application_train.csv
#   3. Run: dvc add data/raw/application_train.csv

# Step 5: Reproduce the pipeline
dvc repro

# Step 6: Inspect metrics
dvc metrics show
```

When `dvc repro` finishes, you should have:

- `data/processed/train_clean.parquet` (cleaned, scaled)
- `data/processed/train_engineered.parquet` (with 7 engineered columns)
- `data/processed/split_indices.npz` (boolean masks)
- `reports/cleaning_summary.md` (human-readable summary)
- `reports/selected_features.json` (the 30 chosen features)
- `models/xgboost_model.pkl` (the trained XGBoost model)
- `models/baseline_logistic_regression.pkl` (the baseline)
- `models/feature_columns.json` (the exact X columns used)
- `reports/metrics.json` (DVC-tracked metrics)
- `reports/evaluation/roc_curves.png`
- `reports/evaluation/pr_curves.png`
- `reports/evaluation/confusion_matrices.png`

---

## 3. What's deterministic, what's not

### Deterministic

- The stratified train/val/test split (random_state=42 in
  `params.yaml:data.random_state`).
- The cleaning pipeline (no stochastic operations).
- The XGBoost training (`random_state=42`, `tree_method='hist'`).
- The Logistic Regression training (`random_state=42`).
- The feature selection (uses deterministic XGB + RF, both with
  `random_state=42`).

### Stochastic

- SHAP's TreeExplainer internally uses random sampling for some
  operations. In practice the variance is negligible on a 5,000-row
  subsample.
- The synthetic-data validation helper
  (`scripts/_make_synthetic_for_validation.py`) uses a fixed seed so
  its output is reproducible.

The pipeline can be run twice in a row with no changes and will
produce **bit-identical `dvc.lock`**.

---

## 4. Verifying reproducibility on a fresh clone

```bash
$ dvc repro
# ... runs all four stages ...

$ md5sum models/xgboost_model.pkl  # or `certutil -hashfile` on Windows
$ cat dvc.lock
```

The hash in `dvc.lock` for each output is the contract. If two clones
produce different hashes for the same output, something has drifted.

---

## 5. Running a parameter experiment

```bash
# Step 1: Edit params.yaml
#   model.learning_rate: 0.05  →  model.learning_rate: 0.10

# Step 2: Check what will change
$ dvc status
train:
	changed deps:
		params.yaml:
			modified:           model.learning_rate

# Step 3: Re-run
$ dvc repro
# Only the train and evaluate stages rerun. prepare and features
# are skipped because their deps + params didn't change.

# Step 4: Compare metrics
$ dvc metrics show

# Step 5: Revert and re-run
$ git checkout -- params.yaml
$ dvc repro
# train and evaluate rerun again, restoring the original metrics.
```

---

## 6. Running a dataset experiment

```bash
# Step 1: Replace the tracked dataset
$ dvc add data/raw/application_train.csv
# (or copy a new CSV into data/raw/ and re-add it)

# Step 2: Check what will change
$ dvc status
prepare:
	changed deps:
		modified:           data\raw\application_train.csv

# Step 3: Re-run — every stage cascades
$ dvc repro
```

---

## 7. Failure modes & recovery

### "I changed `src/prepare.py` and now everything reruns."

That's expected. `src/prepare.py` is in the `prepare` stage's
`deps:`, so any edit invalidates that stage's lock. Downstream
stages cascade.

### "I edited `reports/cleaning_summary.md` by hand and DVC complains."

That output is owned by DVC. Either:

- Delete the file and re-run `dvc repro` (DVC will regenerate it).
- Or use `dvc unprotect` if you want to edit it in place.

Never edit DVC outputs by hand; the next `dvc repro` will overwrite
them anyway.

### "`dvc push` / `dvc pull` fail."

Check:

- The remote URL in `.dvc/config`.
- Network access / credentials.

If no remote is configured, you can still use `dvc repro` locally —
DVC falls back to the local cache. To configure a remote:

```bash
# Local file system
dvc remote add -d mlsd_local /absolute/path/to/remote

# S3 (uses env vars; no credentials in the repo)
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify s3remote secret_access_key $AWS_SECRET_ACCESS_KEY

# GDrive (uses OAuth, no credentials in the repo)
dvc remote add -d gdrive gdrive://folder-id
dvc push  # triggers OAuth on first run
```

### "`dvc repro` says 'output is already tracked by SCM'."

You accidentally committed a DVC output to Git. Untrack it:

```bash
git rm --cached <path>
git commit -m "stop tracking <path>"
dvc repro
```

---

## 8. What is NOT committed to this repo

For safety, the following never appear in Git or DVC:

- API keys
- AWS credentials
- Cloud tokens
- Passwords
- `.env` files

If you need cloud credentials, set them as environment variables in
your shell. Never paste them into `params.yaml`, `.dvc/config`, or
any tracked file.

---

## 9. The reproducibility contract

If a reviewer runs:

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
dvc repro
```

…and gets:

- A working `models/xgboost_model.pkl`
- A working `reports/metrics.json`
- A clean `dvc status`

…then the project is reproducible. Anything less is a bug.

---

## 10. Reproducibility checklist

- [x] `random_state=42` pinned in `params.yaml`
- [x] Stratified split uses `random_state=42` via `train_test_split`
- [x] `StandardScaler` fit on train slice only
- [x] Feature selection fit on train slice only
- [x] `scale_pos_weight = neg/pos` computed at fit-time, not
      hardcoded
- [x] Test set never used to fit anything
- [x] `dvc.lock` committed to Git (cryptographic record)
- [x] `params.yaml` committed to Git
- [x] `dvc.yaml` committed to Git
- [x] All DVC-tracked files in `.gitignore`
- [x] No hardcoded paths in production code (CLI args + defaults)
- [x] No credentials in any tracked file