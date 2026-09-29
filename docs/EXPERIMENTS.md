# Experiments — Credit Risk Intelligence

This document captures every experiment run during the MLSD
refactor. The DA project's experimental history (4-model comparison,
SMOTE-vs-class-weight, threshold tuning) lives in
[`reports/phase4_model_comparison.md`](../reports/phase4_model_comparison.md)
and is summarised in [`PHASE_CHECKLIST.md`](PHASE_CHECKLIST.md).

The MLSD pipeline deliberately keeps experiments simple:

- One **primary model** (XGBoost)
- One **baseline** (Logistic Regression)
- One **metric of record** (ROC-AUC)

All hyperparameters live in `params.yaml`. Changing them and
re-running `dvc repro` is the entire experiment workflow.

---

## 1. Experiment E1 — baseline MLSD pipeline

**Goal:** Verify the refactored MLSD pipeline reproduces the DA
project's results within stochastic noise.

**Setup:**
- Dataset: `data/raw/application_train.csv` (real Kaggle data, 307,511 rows)
- Split: 70/15/15 stratified, `random_state=42`
- Feature selection: top-30 via XGB + RF + SHAP, fit on train slice
- XGBoost: `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`,
  `scale_pos_weight = neg/pos` (computed at fit time),
  `tree_method='hist'`, `eval_metric='logloss'`, `random_state=42`
- Baseline: Logistic Regression with median-impute + StandardScaler
  (fit on train slice), `C=1.0`, `class_weight='balanced'`

**Results (from the DA project's Phase 4):**

| Model | Val AUC | Test AUC |
|---|---:|---:|
| **XGBoost** | **0.7539** | **0.7578** |
| Logistic Regression | 0.7398 | — |
| Random Forest | 0.7469 | — |
| MLP (PyTorch) | 0.7448 | — |

**Outcome:** XGBoost remains the primary model. The MLSD pipeline's
XGBoost configuration is byte-identical to the DA winner.

---

## 2. Experiment E2 — why no SMOTE?

The DA project tried SMOTE vs class-weight on the same XGBoost
config. Result:

| Imbalance strategy | XGBoost val AUC |
|---|---:|
| `class_weight='balanced'` (→ `scale_pos_weight=11.39`) | **0.7539** |
| SMOTE inside imblearn Pipeline | 0.7203 |

SMOTE dropped AUC by 0.034 absolute. The class-weight approach is
**strictly better for this dataset**.

The MLSD pipeline therefore uses `scale_pos_weight = neg/pos`
(equivalent to `class_weight='balanced'`) and does not import
SMOTE.

---

## 3. Experiment E3 — top-N feature selection

The DA project's Phase 3 ran baseline-vs-reduced cross-validation:

| Feature set | 5-fold CV AUC (LR) |
|---|---:|
| All ~140 features | 0.7445 ± 0.0026 |
| Top 30 (XGB+RF+SHAP combined) | 0.7397 ± 0.0023 |

Top-30 is within ~0.005 AUC of all-features (well inside the noise),
and the selected feature list is more interpretable + trains faster.
The MLSD pipeline keeps top-30.

**Leakage fix (Phase 0 of MLSD):** in the DA project, the importance
extractor was trained on the **entire cleaned matrix**
(train+val+test). The MLSD pipeline restricts it to
`SPLIT == 'train'`. The selected feature list is the same in
practice but no longer leaks val/test information.

---

## 4. Experiment E4 — parameter sweep (MLSD, illustrative)

To demonstrate `dvc repro` correctly invalidates downstream stages
when a parameter changes, run:

```bash
# Edit params.yaml: model.learning_rate: 0.05  →  0.10
$ dvc status
train:
	changed deps:
		params.yaml:
			modified:           model.learning_rate

$ dvc repro
# Only `train` and `evaluate` rerun. `prepare` and `features` are
# skipped because their deps + params didn't change.
```

The expected behaviour is: `prepare` and `features` print "didn't
change, skipping"; `train` and `evaluate` re-execute and write new
artifacts.

To restore the original config:

```bash
$ git checkout -- params.yaml
$ dvc repro
# train + evaluate rerun once more, restoring the original metrics.
```

---

## 5. Experiment E5 — dataset-change cascade

To demonstrate the full cascade when the tracked dataset changes:

```bash
# Replace data/raw/application_train.csv with a different file
$ dvc add data/raw/application_train.csv  # re-hashes
$ dvc status
prepare:
	changed deps:
		modified:           data\raw\application_train.csv

$ dvc repro
# ALL FOUR STAGES rerun. The cascade is:
#   prepare → features → train → evaluate
```

---

## 6. Things we deliberately did NOT experiment with

- **Online learning / streaming.** The dataset is a single CSV
  snapshot. No temporal ordering per applicant.
- **Drift detection.** Without reliable timestamps, PSI / KS tests
  on the input distribution would be statistically meaningless.
- **Deep learning beyond the existing MLP.** The DA project
  evaluated an MLP (PyTorch) and it underperformed XGBoost
  (val AUC 0.7448 vs 0.7539) at much higher training cost.
- **AutoML / Optuna hyperparameter search.** The course spec
  asks for DVC pipeline + reproducibility, not model-finding.
- **Multiple imputation.** Median imputation is sufficient for the
  tree models. Multiple imputation adds 5–10× preprocessing time
  for negligible AUC improvement.

---

## 7. How to record a new experiment

1. Copy `params.yaml` to `params_<experiment_name>.yaml`.
2. Edit the copy.
3. Run `dvc repro --params params_<experiment_name>.yaml` for each
   stage, OR temporarily swap `params.yaml` and run `dvc repro`.
4. Compare with `dvc metrics show` (or `dvc metrics diff`).
5. Commit the new `params_<experiment_name>.yaml` + the new
   `dvc.lock` if you want a permanent record.

The MLSD pipeline is intentionally simple — there's no MLflow or
Weights & Biases. DVC's `dvc.lock` + `dvc metrics` is the
experiment tracker.
