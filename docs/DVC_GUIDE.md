# DVC Guide — Credit Risk Intelligence

A complete walkthrough of how **Data Version Control (DVC)** is used in
this project: what it does, how the pipeline is wired, every command
you'll need, and how to fix the most common issues.

---

## 1. What problem DVC solves

Git is great for code but bad for **large binaries**: 166 MB CSVs,
multi-megaparquet files, trained `.pkl` models, ROC plots. Committing
those to Git bloats the repo and breaks cloning.

**DVC** is "Git for data + ML pipelines." It does two things:

1. **Tracks data files by content hash** — a 90-byte `.dvc` pointer
   file in Git replaces the 166 MB CSV. The actual bytes live in a
   "remote" (a folder, S3, GDrive, Azure, etc.) and can be pulled on
   demand with `dvc pull`.
2. **Defines the ML pipeline as a graph** — `dvc.yaml` declares
   stages, each with `cmd` / `deps` / `params` / `outs`. DVC figures
   out what's stale and only re-runs what changed.

In this project: **Git owns code, `params.yaml`, `.dvc` pointers,
markdown reports.** **DVC owns** the 166 MB raw CSV, the four parquets,
the trained models, and the figures.

---

## 2. Repository layout (DVC-relevant parts)

```
DA-Lab-Project/
├── dvc.yaml              # pipeline definition (4 stages + drift)
├── dvc.lock              # frozen md5 hashes of every dep / out / param
├── params.yaml           # tunable parameters (read by stages)
├── .dvc/
│   ├── config            # remote URL + local cache config
│   ├── cache/            # content-addressed local file store
│   └── tmp/              # transient working files
├── .dvc_remote/
│   └── files/...         # the "remote" for this repo (local folder)
├── data/
│   ├── raw/
│   │   ├── application_train.csv          # the 166 MB CSV (gitignored)
│   │   └── application_train.csv.dvc      # pointer to it (in Git)
│   └── processed/        # generated parquets (gitignored, tracked by DVC)
├── models/               # generated .pkl pickles (gitignored)
├── reports/              # generated metrics + plots
├── src/                  # the pipeline scripts
└── tests/                # pytest suite for the scripts
```

The `.dvc_remote/files/...` is a *local* remote — fine for a single
developer. In a team setting you'd point DVC at S3/GDrive/Azure via
`.dvc/config` and `dvc remote modify`.

---

## 3. The pipeline (`dvc.yaml`)

### 3.1 Topology

```
        data/raw/application_train.csv.dvc
                        │
                        ▼
                  ┌──────────┐
                  │ prepare  │   src/prepare.py
                  └──────────┘
                        │
                        ▼
                  ┌──────────┐
                  │ features │   src/features.py
                  └──────────┘
                    │       │
            ┌───────┘       └───────┐
            ▼                       ▼
       ┌────────┐              ┌────────┐
       │ train  │              │ drift  │   (parallel; optional)
       └────────┘              └────────┘
            │
            ▼
       ┌──────────┐
       │ evaluate │   src/evaluate.py
       └──────────┘
```

`dvc dag` prints this as ASCII.

### 3.2 Stage anatomy — example

```yaml
features:
  desc: >-
    Apply the 7 engineered features, then select top-N using XGB + RF +
    SHAP importance ranks, fit on TRAIN only.
  cmd: python src/features.py
  deps:
    - src/features.py
    - src/features/engineer.py
    - src/features/select.py
    - data/processed/train_clean.parquet
    - data/processed/split_indices.npz
  params:
    - features.top_n
    - data.random_state
  outs:
    - data/processed/train_engineered.parquet
    - reports/selected_features.json
```

Each field has a precise role:

| Field     | Purpose                                                                                  |
| --------- | ---------------------------------------------------------------------------------------- |
| `cmd`     | The exact shell command DVC runs.                                                        |
| `deps`    | Files DVC hashes (md5). If any hash changes (or a file disappears), the stage is dirty.   |
| `params`  | Keys from `params.yaml` whose values feed this stage. Value changes also trigger reruns.  |
| `outs`    | Files DVC tracks **after** the stage succeeds. Hashes go into `dvc.lock`.                |
| `metrics` | A special kind of `out` whose contents DVC knows are numeric. Enables `dvc metrics show`. |

### 3.3 The four core stages (+ drift)

| # | Stage        | Script                | Inputs                                              | Outputs                                                                                                  |
| - | ------------ | --------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1 | `prepare`    | `src/prepare.py`      | `data/raw/application_train.csv` + cleaning params  | `data/processed/train_clean.parquet`, `split_indices.npz`, `reports/cleaning_summary.md`                |
| 2 | `features`   | `src/features.py`     | cleaned parquet + `features.top_n`                  | `data/processed/train_engineered.parquet`, `reports/selected_features.json`                             |
| 3 | `train`      | `src/train.py`        | engineered parquet + `model.*` + `baseline.*` params| `models/xgboost_model.pkl`, `models/baseline_logistic_regression.pkl`, `models/feature_columns.json`     |
| 4 | `evaluate`   | `src/evaluate.py`     | trained models + engineered parquet + threshold    | `reports/metrics.json` (DVC metric), `reports/evaluation/{roc_curves,pr_curves,confusion_matrices}.png` |
| – | `drift` *(opt)* | `src/drift.py`     | engineered parquet + selected features              | `reports/drift/{psi_table.csv, psi_top_features.png, drift_report.md, summary.json}`                     |

`drift` is marked optional in the file comments — the dataset has no
per-row timestamp so real temporal drift can't be measured; it compares
train vs val vs test slices of the same snapshot as a teaching artifact.
Run it explicitly with `dvc repro drift`.

---

## 4. The parameters contract (`params.yaml`)

DVC reads only the keys listed in each stage's `params:` block. The
canonical values in this repo:

```yaml
data:
  random_state: 42
  train_size: 0.7
  validation_size: 0.15
  test_size: 0.15
preprocessing:
  missing_threshold: 0.6
  high_cardinality_threshold: 10
features:
  top_n: 30           # ← the Phase 3 selection size
model:
  type: xgboost
  n_estimators: 300
  max_depth: 4
  learning_rate: 0.05
  eval_metric: logloss
  tree_method: hist
  random_state: 42
baseline:
  type: logistic_regression
  C: 1.0
  max_iter: 200
  class_weight: balanced
  random_state: 42
evaluation:
  primary_metric: roc_auc
  threshold: 0.5
  random_state: 42
```

**Tip for demos:** bumping `features.top_n` from 30 → 50 (and committing
the new `params.yaml`) will rerun `features` → `train` → `evaluate`
automatically. `dvc metrics diff` then shows whether the change helped.

---

## 5. `dvc.lock` — the reproducibility file

Every successful `dvc repro` rewrites `dvc.lock`. It pins the md5 hash
of every dep, every output, and every param value at the last green
run. That's how a teammate can `git pull` the code + lock file, then
`dvc checkout` to restore the exact bytes DVC had on its cache.

Schema:

```yaml
schema: '2.0'
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - path: data/raw/application_train.csv
        md5: 793a017f41fbac1dc28176b26dbab30e
        size: 166133370
    ...
```

If you accidentally edit a tracked output by hand, `dvc status` will
complain that the on-disk file's hash no longer matches the lock.

---

## 6. Every DVC command you'll actually use

> All examples assume you're in the repo root and the venv is
> `.venv`. On Windows replace `.venv/Scripts/python` with
> `.venv\Scripts\python`. On macOS/Linux use `.venv/bin/python`.

### 6.1 Inspect the pipeline

```bash
# ASCII graph of stages + their relationships
.venv/Scripts/dvc dag

# Which stages are out of date and why
.venv/Scripts/dvc status

# What changed in params.yaml vs the last run
.venv/Scripts/dvc params diff

# What tracked metrics look like right now
.venv/Scripts/dvc metrics show

# Compare metrics between two branches / commits
.venv/Scripts/dvc metrics diff main..feature-branch

# List every output that DVC knows about
.venv/Scripts/dvc list-targets outs
```

### 6.2 Run the pipeline

```bash
# Run everything that's dirty, topologically
.venv/Scripts/dvc repro

# Run specific stages (and their dependencies)
.venv/Scripts/dvc repro prepare features
.venv/Scripts/dvc repro train evaluate
.venv/Scripts/dvc repro drift                # the optional stage

# Force a re-run even if DVC thinks outputs are fresh
.venv/Scripts/dvc repro --force train

# Single-process (debug; print stack traces)
.venv/Scripts/dvc repro --no-singlproc -v train
```

### 6.3 Manage data

```bash
# After a fresh clone, fetch the dataset + every other DVC-tracked output
.venv/Scripts/dvc pull

# Push your local DVC cache back to the configured remote
.venv/Scripts/dvc push

# Track a new artifact (creates <file>.dvc + adds path to .gitignore)
.venv/Scripts/dvc add path/to/big_file.csv

# Track a whole directory
.venv/Scripts/dvc add data/processed/

# Remove tracking (deletes the .dvc file; doesn't touch data)
.venv/Scripts/dvc remove path/to/file.dvc --outs

# Restore exact bytes from dvc.lock after a fresh checkout
.venv/Scripts/dvc checkout

# Garbage-collect unused cache entries
.venv/Scripts/dvc gc --workspace -c  # DRY-RUN; drop -c to actually delete
```

### 6.4 Manage remotes

```bash
.venv/Scripts/dvc remote list                              # show configured remotes
.venv/Scripts/dvc remote add myremote s3://bucket/path     # add an S3 remote
.venv/Scripts/dvc remote modify myremote endpointurl ...    # tweak an existing one
.venv/Scripts/dvc remote default myremote                  # set as default
```

In this repo, `.dvc/config` points `storage` at the **local**
`.dvc_remote/files/...` folder, so `pull`/`push` are local file copies.

### 6.5 Experiments (optional)

```bash
.venv/Scripts/dvc exp run --set-param features.top_n=50            # run with a one-off override
.venv/Scripts/dvc exp run --name exp-xgb-tuned --set-param model.learning_rate=0.1 \
                                              --set-param model.max_depth=6
.venv/Scripts/dvc exp list                                            # list recorded experiments
.venv/Scripts/dvc exp show                                            # table view (params + metrics)
.venv/Scripts/dvc exp diff main@{run} exp-xgb-tuned                   # compare two experiments
.venv/Scripts/dvc exp apply exp-xgb-tuned                             # restore files + params
```

This is the proper way to A/B test parameter changes without polluting
the working tree.

---

## 7. Reproducibility workflow (what a teammate does)

```bash
git clone <repo-url>
cd DA-Lab-Project
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# Pull every DVC artifact (166 MB CSV + parquets + pickles) from the remote
.venv/Scripts/dvc pull

# Verify everything is up to date
.venv/Scripts/dvc status
.venv/Scripts/dvc metrics show

# (Optional) re-run to confirm end-to-end determinism
.venv/Scripts/dvc repro
```

After the run:
- File hashes match `dvc.lock`
- Metrics match `reports/metrics.json`
- The dashboard boots:
  ```bash
  .venv/Scripts/python -m streamlit run dashboard/app.py
  ```

That's the full reproducibility contract.

---

## 8. Common gotchas

### 8.1 "data/raw/application_train.csv not found" after `git clone`

You forgot `dvc pull`. Fix:

```bash
.venv/Scripts/dvc pull data/raw/application_train.csv.dvc
```

### 8.2 `dvc repro` says "broken link" for some output

Someone edited a DVC-tracked file outside the pipeline. Reset it:

```bash
.venv/Scripts/dvc checkout
```

### 8.3 A stage runs every time, even though nothing changed

Usually a dep that DVC *can't* hash (e.g. an environment variable, a
network call) is missing from the `deps:` block. Add the missing input
to `deps` or `params`.

### 8.4 `train_top30.parquet` is missing but `dvc status` says "up to date"

`train_top30.parquet` is produced by the **Phase 3 notebook builder**
(`notebooks/_build_phase3_notebook.py`), not the MLSD pipeline.
`dvc.yaml` only owns `prepare → features → train → evaluate (+drift)`.
Fix:

```bash
.venv/Scripts/python notebooks/_build_phase3_notebook.py
```

(Or move the top-30 selection into the `features` stage.)

### 8.5 `dvc status` reports everything as "modified dep"

This means an `out` of an upstream stage changed (its md5 no longer
matches the lock). Replay the downstream chain:

```bash
.venv/Scripts/dvc repro
```

It will only re-execute stages whose inputs actually changed.

---

## 9. Reference numbers (current state of the repo)

From `dvc metrics show` on the latest run:

| Model               | Slice       | ROC-AUC | PR-AUC | F1     |
| ------------------- | ----------- | ------- | ------ | ------ |
| XGBoost             | validation  | 0.7545  | 0.2438 | 0.2650 |
| XGBoost             | test        | 0.7593  | 0.2440 | 0.2701 |
| Logistic Regression | validation  | 0.7438  | 0.2202 | 0.2522 |
| Logistic Regression | test        | 0.7438  | 0.2202 | 0.2587 |

Primary metric configured as **`roc_auc`**, decision threshold
**`0.5`**. To switch to F1 as the primary comparator, edit
`params.yaml` (`evaluation.primary_metric: f1`) and rerun
`dvc repro evaluate`.

---

## 10. Where to read more

- `docs/DVC_VIVA_GUIDE.md` — concise Q&A you can use for an oral exam.
- `docs/REPRODUCIBILITY.md` — the project's broader reproducibility
  statement (covers MLflow, tests, and this DVC pipeline together).
- `docs/MLSD_ARCHITECTURE.md` — how the stages fit into the MLSD phases.
- `dashboard/README.md` — what consumes the artifacts the pipeline
  produces.
- Official DVC docs: <https://dvc.org/doc>

---

*Generated from the actual pipeline as committed in `dvc.yaml` /
`dvc.lock`. Run `dvc dag` to regenerate the ASCII graph if you edit
the pipeline.*
