# Credit Risk Intelligence

An end-to-end loan-default forecasting system. Trains a 4-model comparison on the
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset
(`application_train.csv`, 307,511 loans × 122 columns) and ships a Streamlit dashboard
with a 3-way Approve / Manual Review / Reject scoring policy.

- **Dataset:** Home Credit Default Risk — `application_train.csv` only (other
  Home Credit tables are intentionally out of scope per the project brief).
- **Model:** XGBoost — **val AUC 0.7539**, **test AUC 0.7578** at threshold 0.500.
- **Dashboard:** 4 tabs (Portfolio Overview, Applicant Risk Scorer, Feature Importance,
  Segment Analysis) wired live to `src/models/score.py::score_application(...)`.

This is a UIU Data Analytics Laboratory course project — **Group: DomainRange**
(Musfique Ahmed, Tasfya Binte Karim). The full project specification lives at
`puku_master_prompt.md` in this repo.

---

## Table of contents

| Phase | What | Where |
|---|---|---|
| 1 | Environment + cleaning pipeline | `src/data/clean.py`, `data/processed/train_clean.parquet` |
| 2 | EDA + 3 hypothesis tests + 12 figures | `notebooks/01_eda.ipynb`, `reports/figures/01..12_*.png` |
| 3 | Feature engineering + top-30 selection | `src/features/`, `notebooks/02_feature_selection.ipynb` |
| 4 | 4-model comparison + scoring function | `src/models/`, `notebooks/03_model_building.ipynb` |
| 5 | Streamlit dashboard | `dashboard/app.py`, `dashboard/README.md` |
| 6 | Docs + cross-check | `README.md` (this file), `docs/PHASE_CHECKLIST.md` |

Per-phase detail:
- Dashboard tab-by-tab → [`dashboard/README.md`](dashboard/README.md)
- Model artifact + regen → [`models/README.md`](models/README.md)
- Master-prompt checklist → [`docs/PHASE_CHECKLIST.md`](docs/PHASE_CHECKLIST.md)
- Findings per phase → `reports/*.md` (see [Reports & figures](#reports--figures))

---

## Repo structure

```
.
├── data/
│   └── processed/                 # parquet artifacts (gitignored) + .gitkeep
├── dashboard/
│   ├── app.py                     # 4-tab Streamlit app
│   ├── _theme.py                  # palette + CSS injection
│   ├── _loaders.py                # cached parquet/CSV/md loaders
│   ├── _form_to_features.py       # form -> 30-feature dict
│   ├── _segments.py               # income/employment buckets + CI aggregation
│   └── README.md
├── docs/
│   └── PHASE_CHECKLIST.md         # master-prompt cross-check
├── models/
│   ├── best_model.pkl             # XGBoost artifact (gitignored)
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb               # generated from _build_eda_notebook.py
│   ├── 02_feature_selection.ipynb # generated from _build_phase3_notebook.py
│   ├── 03_model_building.ipynb    # generated from _build_phase4_notebook.py
│   └── _build_*.py                # canonical notebook builders
├── reports/
│   ├── cleaning_summary.md
│   ├── data_quality_report.md
│   ├── eda_findings.md
│   ├── feature_importance.md      # top-30 ranked
│   ├── phase4_model_comparison.md
│   └── figures/                   # 23 PNGs
├── src/
│   ├── data/                      # load + clean + profile
│   ├── features/                  # engineer + select
│   └── models/                    # train + tune + evaluate + score
├── tests/                         # 23 pytest tests
├── puku_master_prompt.md          # project specification
├── requirements.txt
└── README.md                      # this file
```

---

## Setup

```bash
# Clone and enter the repo
git clone <repo-url> "DA Lab Project"
cd "DA Lab Project"

# Create the venv (Python 3.14+)
python -m venv .venv

# Activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (Git Bash):   .venv/Scripts/activate
# Linux/macOS:          source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull the dataset (one-time, NOT in repo)
# Place application_train.csv at .home-credit-default-risk/application_train.csv
```

> **Python version:** this repo is pinned to versions compatible with **Python 3.14**
> (`requirements.txt` header notes the cap). Streamlit is intentionally pinned to
> `>=1.38,<1.50` because the 1.50+ ASGI stack has a WebSocket-after-handshake
> regression on Python 3.14 that leaves the browser stuck on skeleton placeholders.

---

## Reproduce the pipeline

Each phase has a notebook builder that emits the `.ipynb` and (re)computes the
artifacts. Run them in order:

```bash
# Phase 1 — cleaning pipeline
# Implemented in src/data/clean.py; run its tests to verify:
.venv/Scripts/python -m pytest tests/test_clean.py -v

# Phase 2 — EDA notebook (also re-saves reports/figures/ and reports/eda_findings.md)
.venv/Scripts/python notebooks/_build_eda_notebook.py

# Phase 3 — feature engineering + top-30 selection
.venv/Scripts/python notebooks/_build_phase3_notebook.py

# Phase 4 — 4-model comparison + writes models/best_model.pkl
.venv/Scripts/python notebooks/_build_phase4_notebook.py

# Phase 5 — dashboard is `dashboard/app.py` (no notebook; see "Run the dashboard")
```

To verify everything:

```bash
.venv/Scripts/python -m pytest tests/ -v
# expected: 23 passed
```

---

## Run the dashboard

```bash
.venv/Scripts/python -m streamlit run dashboard/app.py
```

Then open http://localhost:8501 in your browser. The dashboard reads from
`models/best_model.pkl` and the parquets in `data/processed/`, so make sure
those exist (run the Phase 4 builder first if you don't have them).

What you'll see:

| Tab | Purpose |
|---|---|
| **Portfolio Overview** | KPI cards (215,257 apps, 8.07% default rate, 17,377 default loans), class balance, 4-model AUC chart, 3-way policy panel |
| **Applicant Risk Scorer** | 14-widget form → live XGBoost score → hero recommendation card (Approve / Manual Review / Reject) + gauge + sensitivity panel |
| **Feature Importance** | Top-30 features from `reports/feature_importance.md` with category filter |
| **Segment Analysis** | Default rate by income decile / employment bucket / region / occupation with 95% CI error bars |

The full tab-by-tab walkthrough is in [`dashboard/README.md`](dashboard/README.md).

---

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

23 tests across 4 files:

| File | Tests | Phase |
|---|---|---|
| `tests/test_clean.py` | 3 | Phase 1 |
| `tests/test_engineer.py` | 6 | Phase 3 |
| `tests/test_score.py` | 8 | Phase 4 |
| `tests/test_dashboard.py` | 6 | Phase 5 |

The AppTest harness in `tests/test_dashboard.py` also smoke-tests the Streamlit
app by running it in-process and asserting that no exceptions are raised.

---

## Results at a glance

### 4-model comparison (validation AUC-ROC)

| Model | Val AUC | Test AUC | Notes |
|---|---:|---:|---|
| **XGBoost** | **0.7539** | **0.7578** | Winner — chosen for production |
| Random Forest | 0.7469 | — | Strong baseline, lower recall |
| MLP (PyTorch) | 0.7448 | — | Tuned hidden=[128, 64], dropout=0.3 |
| Logistic Regression | 0.7398 | — | Interpretable baseline |

The full table with precision / recall / F1 / confusion matrices is in
`reports/phase4_model_comparison.md`.

### Most actionable EDA findings

- **H1** (lower EXT_SOURCE → higher default): **SUPPORTED.** Correlation
  r ≈ −0.16 to −0.18 across all three EXT_SOURCE columns, p ≈ 0. EXT_SOURCE_3 is
  the single most actionable signal.
- **H2** (higher credit-to-income → higher default): **NOT supported as stated.**
  The naive CTI relationship is **inverted-U** (peak 9.2% at decile 6, 7.1% at top).
  Adding EXT_SOURCE_3 to the regression flips the CTI coefficient sign — Simpson's
  paradox from the EXT_SOURCE confound.
- **H3** (occupation/education segments have elevated risk): **SUPPORTED.**
  Low-skill Laborers default at 17.2% vs Accountants at 4.8% (3.5×); education
  shows a clean monotonic gradient (10.9% → 1.8%). Chi-square tests are highly
  significant (p ≪ 1e-200) but effect sizes are small (Cramér's V ≈ 0.03–0.08).

Full narrative is in `reports/eda_findings.md`.

---

## Reports & figures

| File | Purpose |
|---|---|
| `reports/data_quality_report.md` | 122-column dtype + missingness profile (Phase 1) |
| `reports/cleaning_summary.md` | What cleaning dropped / imputed / encoded (Phase 1) |
| `reports/eda_findings.md` | 3 hypothesis tests + business recommendations (Phase 2) |
| `reports/feature_importance.md` | Top-30 ranked table (XGB / RF / SHAP) (Phase 3) |
| `reports/phase4_model_comparison.md` | 4-model AUC + config + recommendation policy (Phase 4) |
| `reports/figures/01..12_*.png` | EDA charts (Phase 2) |
| `reports/figures/13..16_*.png` | Feature engineering + importance (Phase 3) |
| `reports/figures/17..23_*.png` | 4-model ROC + confusion matrices (Phase 4) |

---

## Authors & acknowledgements

- **Course:** UIU Data Analytics Laboratory
- **Group:** DomainRange
- **Members:** Musfique Ahmed, Tasfiya Binte Karim
- **Dataset:** Home Credit Default Risk (Kaggle) — application data only
- **Project brief:** `puku_master_prompt.md` in this repo

The full plan + per-phase deliverables are tracked in `docs/PHASE_CHECKLIST.md`.
