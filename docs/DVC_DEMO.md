# DVC Demo — Credit Risk Intelligence

> A hands-on walkthrough of the Data Version Control layer in this
> project. Read this if you want to **see DVC working**, not just read
> about it. For Q&A-style prep, see `docs/DVC_VIVA_GUIDE.md`.

The MLSD Credit Risk Intelligence project uses **DVC 3.x** to track
the 160-MB raw dataset, every intermediate parquet, the trained
models, the metrics, and the pipeline definition itself. Git owns the
text (code, configs, docs); DVC owns the bytes.

---

## 1. What DVC is doing in this project (at a glance)

| Artifact                                  | Owned by | How                           |
| ----------------------------------------- | -------- | ----------------------------- |
| `src/*.py`, `dvc.yaml`, `params.yaml`     | Git      | Plain text, diff-able         |
| `data/raw/application_train.csv` (160 MB) | DVC      | `dvc add` → pointer + cache   |
| `data/processed/*.parquet`, `*.npz`       | DVC      | Pipeline `outs:`              |
| `models/*.pkl`, `models/feature_columns.json` | DVC  | Pipeline `outs:`              |
| `reports/metrics.json`, `reports/evaluation/*.png` | DVC | Pipeline `metrics:` + `outs:` |
| `.dvc/cache/`                             | DVC      | Content-addressed, gitignored |
| `.dvc_remote/` (local remote on disk)     | DVC      | `dvc push` / `dvc pull` target |

---

## 2. One-time setup (already done in this repo)

DVC is initialised, the remote is configured, and the raw dataset is
tracked. You only need the venv + dependencies.

```bash
# from repo root
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

# macOS / Linux
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Verify DVC is installed and the remote is wired up:

```bash
.venv\Scripts\dvc version
.venv\Scripts\dvc remote list
# mlsd_local  E:/poridhi/DA-Lab-Project/.dvc_remote
```

---

## 3. The 5-minute demo script

Run each block from the repo root and read the output. Every command
below is a real command run against this project.

### Step 1 — Confirm everything is already up to date

```bash
.venv\Scripts\dvc status
```

Expected output:

```text
Data and pipelines are up to date.
```

This means every stage in `dvc.yaml` has hashes in `dvc.lock` that
match the current files on disk. Nothing to do.

### Step 2 — Look at the pipeline DAG

```bash
.venv\Scripts\dvc dag
```

Expected output (ASCII art of the dependency graph):

```text
         +---------+
         | prepare |
         +---------+
              *
              *
              *
        +----------+
        | features |
        +----------+
         **        **
       **            *
      *               **
+-------+               *
| train |             **
+-------+            *
         **        **
           **    **
             *  *
        +----------+
        | evaluate |
        +----------+
```

The `prepare → features → {train, drift} → evaluate` cascade is the
entire ML lineage, drawn as a picture.

### Step 3 — Run the pipeline once (idempotent — no-op)

```bash
.venv\Scripts\dvc repro
```

Expected output:

```text
Stage 'prepare' didn't change, skipping
Stage 'features' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Stage 'drift' didn't change, skipping
Data and pipelines are up to date.
```

DVC compared every `deps` / `params` hash against `dvc.lock` and
found no change. It didn't invoke Python. **This is the killer
feature** — re-runs are free when nothing changed.

### Step 4 — Show the metrics DVC is tracking

```bash
.venv\Scripts\dvc metrics show
```

Expected output (numbers from `reports/metrics.json`):

```text
reports/metrics.json:
    xgboost.test.roc_auc                       0.75925
    xgboost.test.pr_auc                        0.24403
    xgboost.test.f1                            0.27012
    xgboost.validation.roc_auc                 0.75452
    logistic_regression_baseline.test.roc_auc  0.74380
    logistic_regression_baseline.test.pr_auc   0.22018
    ...
```

These are the live values from `reports/metrics.json`. Because the
`evaluate` stage declares this file as a DVC metric, `dvc metrics
show` reads it without you having to open the JSON.

### Step 5 — Show the current parameters

```bash
.venv\Scripts\dvc params diff
```

Expected output: nothing (no diff against the last committed run).

```bash
.venv\Scripts\dvc params show
```

Expected output: every parameter declared in `params.yaml`, grouped
by stage.

---

## 4. The "watch DVC react" demo (5 more minutes)

This is the most impressive part — change one thing and watch DVC
invalidate only the affected stages.

### Step 4.1 — Change a hyperparameter

Open `params.yaml` and bump the learning rate:

```yaml
model:
  learning_rate: 0.05   # was 0.05
  ↓
  learning_rate: 0.10
```

### Step 4.2 — See the dirty-stage report

```bash
.venv\Scripts\dvc status
```

Expected output:

```text
train:
	changed deps:
		params.yaml:
			modified:           model.learning_rate
evaluate:
	changed deps:
		modified:           models\xgboost_model.pkl
```

DVC noticed that `train` reads `model.learning_rate`, and `evaluate`
depends on the model artifact that `train` will rebuild. Nothing
else is dirty — `prepare` and `features` are still up to date because
they don't read that parameter.

### Step 4.3 — Rerun only the dirty stages

```bash
.venv\Scripts\dvc repro
```

Expected output:

```text
Running stage 'train':
> python src/train.py
...
Running stage 'evaluate':
> python src/features.py
...
Stage 'prepare' didn't change, skipping
Stage 'features' didn't change, skipping
Stage 'drift' didn't change, skipping
```

`prepare` and `features` are skipped. Only `train` and its
downstream `evaluate` re-ran. That's the DAG in action.

### Step 4.4 — Diff the metrics after the change

```bash
.venv\Scripts\dvc metrics diff
```

Expected output: a side-by-side comparison of every tracked metric
before vs after the `learning_rate` change. This is how DVC replaces
ad-hoc spreadsheets for experiment tracking.

### Step 4.5 — Revert and verify

```bash
git checkout -- params.yaml
.venv\Scripts\dvc status
```

Expected output:

```text
Data and pipelines are up to date.
```

`dvc repro` will skip everything again. The lock file is back in
sync.

### Step 4.6 — Alternative demo: change source code

Edit `src/features.py` (e.g. add a comment or rename a variable) and
run:

```bash
.venv\Scripts\dvc status
.venv\Scripts\dvc repro
```

DVC detects that `features.py` is in `features`'s `deps`, marks the
stage dirty, and cascades to `train` and `evaluate`. `prepare`
remains clean. This is the same mechanism, triggered by source code
instead of params.

---

## 5. The remote-storage demo (2 minutes)

The repo is wired to a **local** remote at `.dvc_remote/`. The
"remote" is just a directory; in production you'd swap it for S3 /
GCS / Azure / GDrive / SSH without changing any other command.

```bash
# what files does DVC own remotely?
.venv\Scripts\dvc list .dvc_remote
```

```bash
# push everything not already on the remote (no-op if already there)
.venv\Scripts\dvc push
```

```bash
# simulate a fresh checkout in another directory
mkdir ..\mlsd-fresh-clone
cd ..\mlsd-fresh-clone
git clone ..\DA-Lab-Project .
.venv\Scripts\dvc pull         # downloads the raw CSV + cached outputs
.venv\Scripts\dvc repro        # no-op — everything is already cached
```

After `git clone + dvc pull`, a brand-new collaborator has the
160-MB dataset, every processed parquet, every trained model, every
metric — without anyone emailing binaries around.

---

## 6. Where to look in the repo

```
DA-Lab-Project/
├── dvc.yaml              # pipeline definition (4 core stages + drift)
├── dvc.lock              # hash ledger; updated only when stages re-run
├── params.yaml           # all experiment hyperparameters
├── .dvc/
│   ├── config            # remote URL, cache settings
│   └── cache/            # content-addressed store (gitignored)
├── .dvcignore            # files DVC should never touch
├── .dvc_remote/          # local DVC remote (acts like S3)
└── data/raw/
    └── application_train.csv.dvc   # pointer file (git-tracked)
```

| File                       | Tracks                                                |
| -------------------------- | ----------------------------------------------------- |
| `dvc.yaml`                 | Pipeline DAG: `cmd`, `deps`, `outs`, `params`, `metrics` |
| `dvc.lock`                 | Frozen hashes from the last successful run           |
| `params.yaml`              | Experiment hyperparameters, one file, one place       |
| `*.dvc` (e.g. raw CSV)     | Pointer to a hash in `.dvc/cache/`                   |
| `.dvc/config`              | Remote URL + cache settings                           |

---

## 7. Cheat-sheet of the commands used above

| Command                       | What it does                                           |
| ----------------------------- | ------------------------------------------------------ |
| `dvc init`                    | Initialise DVC inside a Git repo (already done)        |
| `dvc add <file>`              | Track a file with DVC → pointer + cache entry          |
| `dvc repro`                   | Run only stale stages; update `dvc.lock`              |
| `dvc status`                  | Show which stages are dirty                            |
| `dvc dag`                     | Print the pipeline dependency graph                    |
| `dvc dag --md`                | Print the DAG as a Mermaid block                       |
| `dvc metrics show`            | Print every tracked metric                             |
| `dvc metrics diff`            | Compare metrics against the last commit                |
| `dvc params show`             | Print every tracked parameter                          |
| `dvc params diff`             | Compare parameters against the last commit             |
| `dvc push`                    | Upload DVC-tracked files to the remote                 |
| `dvc pull`                    | Download DVC-tracked files from the remote             |
| `dvc remote list`             | Show configured remotes                                |
| `dvc list <remote>`           | List files on a remote                                 |
| `dvc version`                 | Print DVC version                                      |

---

## 8. What this demo proves

1. **DVC is a layer on top of Git, not a replacement.** Git still
   owns `dvc.yaml`, `params.yaml`, every `.py` file, every `.md` file,
   and the tiny `.dvc` pointer files. DVC only owns the bytes.
2. **`dvc repro` is incremental.** A no-op re-run is instant; only
   the dirty stages re-execute. The DAG (from `dvc.yaml`) decides
   the cascade.
3. **Hashes are the contract.** Every `deps`, every `params` value,
   and every `outs` file is content-hashed. Change one byte → the
   hash changes → the stage goes dirty.
4. **Metrics + params are first-class.** `dvc metrics show` and
   `dvc params show` are how DVC replaces ad-hoc experiment
   spreadsheets. `dvc metrics diff` is how you compare two runs.
5. **The remote is just a directory.** Replace `.dvc_remote/` with S3
   / GCS / Azure / SSH and every other command stays the same.

---

## 9. Troubleshooting

- **`dvc: command not found`** → activate the venv (`.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`), or invoke via the absolute path as shown above.
- **`dvc status` says a stage is dirty but I didn't change anything** → run `git status` to see if a file was modified, then `git checkout -- <file>` to revert.
- **`dvc pull` is a no-op but `dvc repro` wants to rebuild** → the local cache is missing the upstream version; run `dvc fetch` then `dvc checkout` to materialise outputs.
- **Pipeline fails mid-way** → re-run `dvc repro` after fixing the script; DVC will skip the clean stages and only retry the broken one.
- **Want to nuke the local cache and rebuild from scratch** → delete `.dvc/cache/` and `.dvc_remote/files/`, then `dvc pull && dvc repro`.

---

See `docs/DVC_VIVA_GUIDE.md` for the Q&A-style companion to this
walkthrough, and `docs/REPRODUCIBILITY.md` for the end-to-end
"fresh-clone" instructions.
