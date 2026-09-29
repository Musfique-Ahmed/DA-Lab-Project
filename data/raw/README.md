# Raw dataset location

This directory holds the raw Kaggle CSV(s) used by the MLSD pipeline.

## Required file (in scope)

**`application_train.csv`** is the **only** file the MLSD pipeline
consumes. It is **DVC-tracked** (see `application_train.csv.dvc`) and
**gitignored** — the bytes live in `.dvc/cache/` and the working tree
has only the `.dvc` pointer.

| Property | Value |
|---|---|
| Rows | 307,511 |
| Columns | 122 |
| Target | `TARGET` (binary, ~8.07% positive) |
| Source | [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) |

## Optional file (out of scope)

**`application_test.csv`** is the unlabeled holdout from the Kaggle
competition. The MLSD pipeline **does not use it** — only
`application_train.csv` is in scope per the project's master prompt.
The file may live in this directory for reference but is intentionally
not tracked by DVC.

## How the file was tracked

```bash
# 1. Place the CSV here
cp /path/to/download/application_train.csv data/raw/

# 2. Track it with DVC (computes MD5, moves bytes to .dvc/cache/)
dvc add data/raw/application_train.csv

# 3. Commit the pointer file
git add data/raw/application_train.csv.dvc data/raw/.gitignore
git commit -m "data: track application_train.csv with DVC"
```

## Push to / pull from a remote

```bash
dvc push          # uploads the dataset to the configured remote
dvc pull          # downloads it on a fresh clone
```

If no remote is configured (the default for local development is the
local `.dvc_remote/` directory), DVC falls back to the local cache —
`dvc repro` still works on the same machine.
