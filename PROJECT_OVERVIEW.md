# Credit Risk Intelligence — Full Project Documentation

> **Group:** DomainRange · **Course:** UIU Data Analytics Laboratory
> **Members:** Musfique Ahmed & Tasfiya Binte Karim
> **Repository:** `DA Lab Project/`

---

## 1. What This Project Is

**Credit Risk Intelligence** is an end-to-end loan-default forecasting and decision-support system built around the publicly available **Home Credit Default Risk** dataset (Kaggle). It takes a single tabular CSV — `application_train.csv` (307,511 rows × 122 columns, ~8.07% default rate) — and produces:

1. A **clean, model-ready parquet** of features (307,511 × 144 columns).
2. A **curated top-30 feature set** ranked by combined XGBoost + RandomForest + SHAP importance.
3. **Four trained and tuned ML/DL models** compared head-to-head on validation AUC.
4. A **deployable scoring function** `score_application(input: dict) -> {probability_of_default, recommendation}` that maps a probability to a 3-way decision: **Approve / Manual Review / Reject**.
5. A **custom-themed Streamlit dashboard** with four tabs — Portfolio Overview, Applicant Risk Scorer, Feature Importance Explorer, and Segment Analysis — all wired live to the production model.

The project is a graded UIU coursework deliverable (Group: DomainRange) and follows a six-phase rubric codified in `puku_master_prompt.md`.

---

## 2. Business Motivation

Home Credit is a real-world lending institution whose task is to decide whether to extend a loan to a given applicant. The company has historic data on prior applicants including demographics, employment, asset information, and whether they eventually defaulted. The goal of the project is to convert this historic record into a predictive model that:

- Quantifies default risk for a new applicant as a probability of default (PD).
- Translates that probability into an operational policy (Approve / Manual Review / Reject).
- Surfaces the *reasons* behind each decision (which features drove the PD up or down).

---

## 3. Tech Stack

| Layer | Library | Why |
|---|---|---|
| Data handling | `pandas ≥ 2.2`, `numpy ≥ 2.0` | Standard tabular stack. |
| Visualization (EDA + dashboard) | `matplotlib`, `seaborn`, `plotly` | Static PNGs for the report deck, interactive for the dashboard. |
| Classical ML | `scikit-learn ≥ 1.5` | Logistic Regression, Random Forest, preprocessing, metrics. |
| Gradient boosting | `xgboost ≥ 2.0` | Chosen over LightGBM for parity with scikit-learn APIs. |
| Class-imbalance | `imbalanced-learn ≥ 0.12` | SMOTE was tried in Phase 4 and rejected in favor of class weighting. |
| Deep learning | `torch ≥ 2.11` | Chosen over TensorFlow because TF has no Python 3.14 wheels. |
| Interpretability | `shap ≥ 0.46` | TreeSHAP for the XGBoost feature ranking. |
| Dashboard | `streamlit ≥ 1.38, < 1.50` | 1.50+ ASGI migration has a WebSocket regression on Python 3.14. |
| Tests | `pytest ≥ 8.3` | 23 tests across 4 files. |
| Notebook pipeline | `jupyter`, `nbformat`, `nbconvert`, `ipykernel` | Phases 2-4 are generated + executed by `.py` builders. |
| Persistence | `pyarrow ≥ 17.0`, `joblib ≥ 1.4` | Parquet for cleaned data; joblib for the XGBoost artifact. |

Pinned to **Python 3.14-compatible majors**.

---

## 4. Repository Layout

```
DA Lab Project/
├── .home-credit-default-risk/        # dataset (gitignored; not committed)
│   └── application_train.csv
├── data/
│   └── processed/                    # parquet artifacts (gitignored) + .gitkeep
│       ├── train_clean.parquet       # 307,511 × 144 (Phase 1 output)
│       ├── train_engineered.parquet  # +7 engineered features (Phase 3)
│       ├── train_top30.parquet       # top-30 feature subset (Phase 3)
│       └── split_indices.npz        # train/val/test boolean masks
├── dashboard/
│   ├── app.py                        # 4-tab Streamlit app
│   ├── _theme.py                     # palette + CSS injection (impeccable-designed)
│   ├── _loaders.py                   # cached parquet/CSV/md loaders
│   ├── _form_to_features.py          # 14-widget form → 30-feature dict
│   ├── _segments.py                  # income/employment buckets + CI aggregation
│   └── README.md
├── docs/
│   ├── PHASE_CHECKLIST.md            # rubric cross-check
│   ├── PROJECT_DOCUMENTATION.md      # human-readable project write-up
│   ├── PROGRESS_UPDATE.md
│   ├── SPEAKING_SCRIPT.md
│   └── *.pdf                         # presentation PDFs
├── models/
│   ├── best_model.pkl                # XGBoost artifact (gitignored)
│   └── README.md
├── notebooks/
│   ├── 00_data_cleaning.ipynb        # Phase 1 — run this to clean + write parquet
│   ├── 01_eda.ipynb                  # generated from _build_eda_notebook.py
│   ├── 02_feature_selection.ipynb    # generated from _build_phase3_notebook.py
│   ├── 03_model_building.ipynb       # generated from _build_phase4_notebook.py
│   └── _build_*.py                   # canonical notebook builders
├── reports/
│   ├── cleaning_summary.md
│   ├── data_quality_report.md
│   ├── eda_findings.md
│   ├── feature_importance.md         # top-30 ranked
│   ├── phase4_model_comparison.md
│   └── figures/                      # 23 PNGs (figures 00..23)
├── src/
│   ├── data/
│   │   ├── load.py                   # loads application_train.csv with shape assertion
│   │   ├── clean.py                  # 8-step cleaning pipeline
│   │   └── profile.py                # data-quality markdown
│   ├── features/
│   │   ├── engineer.py               # 7 engineered columns
│   │   └── select.py                 # XGB+RF+SHAP ranking → top-N
│   └── models/
│       ├── train.py                  # train one of {LR, RF, XGB, MLP}
│       ├── tune.py                   # light grid search per family
│       ├── evaluate.py               # metrics, threshold picking, ROC plotting
│       ├── mlp.py                    # PyTorch MLP + sklearn-style wrapper
│       └── score.py                  # score_application() — the production scorer
├── tests/
│   ├── test_clean.py                 # 3 tests
│   ├── test_engineer.py              # 6 tests
│   ├── test_score.py                 # 8 tests
│   └── test_dashboard.py             # 6 tests
├── .venv/                            # Python virtualenv (gitignored)
├── requirements.txt
├── puku_master_prompt.md             # project specification (rubric)
├── WAYFINDER_GUIDE.md                # skill/agent discovery guide
└── README.md
```

---

## 5. How It Was Built — The Six Phases

The project was executed strictly one phase at a time. Each phase has a stop-point summary, a clear "what was created / how to verify" output, and an explicit go-ahead gate before the next phase starts. This discipline comes straight from `puku_master_prompt.md`.

### Phase 1 — Environment + Data Cleaning

**Goal:** a clean, model-ready dataset and a reproducible environment.

**What was built:**

- `requirements.txt` — pinned compatible-with-Python-3.14 major versions.
- `src/data/load.py::load_application_train()` — reads `.home-credit-default-risk/application_train.csv` and **asserts the shape is exactly `(307_511, 122)`** — fails loudly otherwise.
- `src/data/clean.py::clean(df)` — the 8-step pipeline:
  1. Copy (never mutate caller data).
  2. Replace `DAYS_EMPLOYED == 365243` sentinel with NaN (**55,374 rows** affected).
  3. Drop columns with **> 60% missing values** (**17 columns dropped** — housing-related fields like `COMMONAREA_*`, `LIVINGAPARTMENTS_*`, `OWN_CAR_AGE`, etc.).
  4. Impute remaining numeric NaNs with column **median**; categorical NaNs with literal `'MISSING'` (**51 columns imputed**).
  5. Encode categoricals: 13 low-cardinality (≤ 10 unique values) → one-hot; 2 high-cardinality (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`) → frequency encoding.
  6. Stratified **70 / 15 / 15 split** on `TARGET` (random_state=42) → train=215,257 / val=46,127 / test=46,127.
  7. `StandardScaler` **fit on the train slice only** (avoids leakage).
  8. Add a `SPLIT` column and persist as parquet.
- `src/data/profile.py::profile_dataframe(df)` — emits `reports/data_quality_report.md`.
- `notebooks/00_data_cleaning.ipynb` — the human-facing pipeline (run-all reproducible).
- `tests/test_clean.py` — **3 tests** covering the sentinel replacement, the missing-value threshold logic, and the **"scaler fit only on train"** invariant.

**Output:** `data/processed/train_clean.parquet` (22.9 MB, 307,511 × 144).

---

### Phase 2 — Exploratory Data Analysis

**Goal:** a notebook section with genuine business insight, not generic charts.

**What was built:**

- `notebooks/01_eda.ipynb` (10 sections, 34 markdown cells, 21 code cells) — programmatically generated by `notebooks/_build_eda_notebook.py`, executed in-process via `nbclient`.
- 23 PNG figures saved to `reports/figures/`.
- `reports/eda_findings.md` — narrative findings + three hypothesis tests.

**Three formal hypothesis tests** (with statistical support, not just charts):

- **H1: Lower EXT_SOURCE scores correlate with higher default probability.** **SUPPORTED.** Correlation r ≈ −0.16 to −0.18 across all three `EXT_SOURCE_*` columns, p ≈ 0. `EXT_SOURCE_3` is the single most actionable signal.
- **H2: Higher credit-to-income ratio increases default likelihood.** **NOT supported as stated.** The naive CTI relationship is **inverted-U** — default rate peaks at 9.2% in decile 6 and falls to 7.1% at the top decile. Adding `EXT_SOURCE_3` to the regression **flips the sign** of the CTI coefficient — Simpson's paradox from the EXT_SOURCE confound.
- **H3: Certain occupation/education segments show elevated risk.** **SUPPORTED.** Low-skill Laborers default at **17.2%** vs Accountants at **4.8%** (3.5× spread). Education shows a clean monotonic gradient (10.9% → 1.8%). Chi-square tests highly significant (p ≪ 1e-200) but effect sizes small (Cramér's V ≈ 0.03–0.08).

**Business recommendations** captured in `reports/eda_findings.md`:

1. `EXT_SOURCE_3` should be a required field on every credit application.
2. Avoid a simple CTI threshold — high-CTI applicants are also high-income and high-EXT-SOURCE.
3. Segment-level underwriting is defensible but should sit alongside individual features.
4. Income is the dominant axis of risk.

---

### Phase 3 — Feature Engineering & Selection

**Goal:** a justified, reduced feature set (top-30).

**What was built:**

- `src/features/engineer.py::engineer_features(df)` — appends **7 derived columns**:
  - `AGE_YEARS` = `-DAYS_BIRTH / 365.25`
  - `EMPLOYED_YEARS` = `-DAYS_EMPLOYED / 365.25`
  - `CREDIT_INCOME_RATIO` = `AMT_CREDIT / AMT_INCOME_TOTAL`
  - `ANNUITY_INCOME_RATIO` = `AMT_ANNUITY / AMT_INCOME_TOTAL`
  - `CREDIT_GOODS_RATIO` = `AMT_CREDIT / AMT_GOODS_PRICE`
  - `INCOME_PER_FAM_MEMBER` = `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`
  - `EXT_SOURCE_MEAN` = mean of (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) — partial-NaN-safe.
- `src/features/select.py`:
  - `train_xgb_and_rf_importances(X, y)` — train quick XGB + RandomForest on full feature set, return combined rank.
  - `shap_values_xgb(X, y, sample_size=5000)` — TreeSHAP on a 5,000-row subsample.
  - `select_top_n(importance_df, shap_matrix, n=30)` — combine XGB rank + RF rank + SHAP rank into a single mean-rank.
- `notebooks/02_feature_selection.ipynb` (built from `_build_phase3_notebook.py`).
- `reports/feature_importance.md` — full ranking of all 148 features + top-30 selection.
- `tests/test_engineer.py` — **6 tests** (correct formulas, NaN handling, division-by-zero safety, EXT_SOURCE partial NaN, immutability).

**Baseline-vs-reduced comparison** (5-fold CV on Logistic Regression):

- All-features mean AUC: **0.7445 ± 0.0026**
- Top-30 mean AUC: **0.7397 ± 0.0023**
- Delta: **−0.0048** (top-30 within 0.01 AUC of all-features)

**Decision:** carry the **top-30** forward to Phase 4 — the AUC delta is small and the top-30 set is faster to train and more interpretable.

**Top-5 features (rank 1–5):** `EXT_SOURCE_MEAN`, `EXT_SOURCE_1`, `EMPLOYED_YEARS`, `EXT_SOURCE_3`, `CODE_GENDER_M`.

---

### Phase 4 — Model Building, Tuning & Selection

**Goal:** four trained, compared models and a working scoring function.

**What was built:**

- `src/models/evaluate.py` — shared metrics (`compute_metrics`), threshold picker (`pick_threshold` with Youden's J strategy), ROC plotting, confusion matrix plotting.
- `src/models/train.py` — `train_logreg`, `train_random_forest`, `train_xgboost`, `train_mlp` — each returns `(model, metrics_dict)` so the comparison table is uniform. Each supports three imbalance strategies (`balanced`, `smote`, `none`/`pos_weight` for MLP).
- `src/models/tune.py` — light grid search per family (≤4 fits each, 16 fits total):
  - LR: `C ∈ {0.1, 1.0, 10.0}`
  - RF: `max_depth ∈ {8, 12, None}`, n_estimators=300
  - XGB: `max_depth ∈ {4, 6}`, `lr ∈ {0.05, 0.1}`, n_estimators=300
  - MLP: hidden ∈ `{(64, 32), (128, 64)}`, dropout=0.3
- `src/models/mlp.py` — PyTorch MLP + sklearn-style `MLPWrapper` (carries scaler + imputer so it matches `predict_proba` / `predict` API).
- `src/models/score.py::score_application(input: dict) -> dict` — **the production scorer**, exposes the 3-way recommendation policy:
  - `p < 0.20` → **Approve**
  - `0.20 ≤ p < 0.50` → **Manual Review**
  - `p ≥ 0.50` → **Reject**
- `notebooks/03_model_building.ipynb` (built from `_build_phase4_notebook.py`).
- `reports/phase4_model_comparison.md` — comparison table + winner.
- `models/best_model.pkl` — the XGBoost artifact (joblib-pickled).
- `models/README.md` — artifact + input schema documentation.
- `tests/test_score.py` — **8 tests** covering recommendation thresholds, input validation, end-to-end scoring.

**Imbalance strategy comparison (val AUC):**

| Family | class_weight | SMOTE |
|---|---:|---:|
| Logistic Regression | 0.7398 | 0.7394 |
| XGBoost | **0.7539** | 0.7203 |

SMOTE **dropped** XGBoost val AUC from 0.7539 to 0.7203 — rejected. Class weighting chosen.

**Final 4-model comparison (validation):**

| Model | Val AUC | Test AUC | Threshold | Val confusion (TN/FP/FN/TP) |
|---|---:|---:|---:|---|
| **XGBoost** (winner) | **0.7539** | **0.7578** | 0.500 | 29,615 / 12,788 / 1,212 / 2,512 |
| Random Forest | 0.7469 | — | 0.500 | 37,669 / 4,734 / 2,255 / 1,469 |
| MLP (PyTorch) | 0.7448 | — | 0.500 | 28,179 / 14,224 / 1,162 / 2,562 |
| Logistic Regression | 0.7398 | — | 0.500 | 29,089 / 13,314 / 1,267 / 2,457 |

**Winner: XGBoost** — `n_estimators=300, max_depth=4, lr=0.05, scale_pos_weight=11.39`. Saved as `models/best_model.pkl`.

`score_application()` demo on three synthetic applicants:

| Profile | PD | Recommendation |
|---|---:|---|
| Low-risk | 0.0073 | Approve |
| Mid-risk | 0.3865 | Manual Review |
| High-risk | 0.9548 | Reject |

---

### Phase 5 — Streamlit Dashboard

**Goal:** a custom-themed 4-tab dashboard wired to the live Phase 4 scorer.

**Files built:**

- `dashboard/app.py` — main entry, 4 tabs.
- `dashboard/_theme.py` — palette + single-block CSS injection (dark navy + mint accents).
- `dashboard/_loaders.py` — `@st.cache_data` loaders for the parquet, raw CSV, importance markdown, and Phase 4 summary.
- `dashboard/_form_to_features.py` — converts a 14-widget user form into the 30-feature dict `score_application()` expects, with a **recovered `StandardScaler`** fit on the train slice of the top-30 parquet (since Phase 1's scaler was fit on 144 cols and not persisted).
- `dashboard/_segments.py` — income-band and employment-length bucketing, default-rate aggregation with **binomial-normal 95% CI error bars**.
- `dashboard/README.md` — operator-facing docs.

**The 4 tabs:**

1. **Portfolio Overview** — 4 KPI cards (total applications, default rate, avg PD, default-loan count), class-balance bar, 4-model AUC comparison bar, 3-way policy explainer.
2. **Applicant Risk Scorer** — 14-widget form (income, credit, annuity, goods price, age, employed years, 3 EXT_SOURCE sliders, gender, education, contract, FLAG_DOCUMENT_3, FLAG_OWN_CAR, region population). On submit: builds the 30-feature dict, calls `score_application()`, renders a hero recommendation card with semantic color (mint / blue / red), a horizontal segmented gauge showing where `p` falls between the two thresholds, and a sensitivity panel that bumps each numeric input ±10% and re-scores.
3. **Feature Importance Explorer** — parses `reports/feature_importance.md` into a DataFrame, renders a Plotly horizontal bar with a category multiselect filter (EXT_SOURCE / Time-derived / One-hot / Ratio / Amount / Area).
4. **Segment Analysis** — loads the raw `application_train.csv`; region-rating filter + segment-dimension selector (income band, employment length, education, income type, contract, gender, owns a car, owns realty, region rating). Default rate per segment plotted as a bar with 95% binomial-normal CI error bars. Underlying table downloadable as CSV.

**Design system** (from `dashboard/_theme.py` — pinned by `puku_master_prompt.md`):

| Token | Hex | Use |
|---|---|---|
| Background | `#0A1428` | App background (deep navy). |
| Card panels | `#10203D` / `#162A4D` / `#0B1830` | Panel + alt panel + deep panel (gradient layers). |
| Primary accent | `#00D9B5` | Approve + CTAs + metric values. |
| Supporting accent | `#3B82F6` | Manual Review + test AUC. |
| Risk accent | `#F87171` | Reject + default indicators only. |
| Text | `#FFFFFF` headers, `#CBD5E1` body, `#94A3B8` muted | — |
| Line | `rgba(255,255,255,0.06)` | Borders. |

CSS is a single block in `_theme.py` and was refined via the project's `/impeccable` design pass. The visual language: dark, premium fintech aesthetic; soft multi-layer shadows for depth (no colored stripes); JetBrains Mono for numerics; browser surfaces (selection, scrollbar) themed to match.

---

### Phase 6 — Documentation & Cross-Check

**What was built:**

- `README.md` — repo overview, setup, reproduce commands, run-the-dashboard section, results-at-a-glance, authors.
- `docs/PHASE_CHECKLIST.md` — rubric cross-check (every master-prompt line marked ✓ with a file reference).
- `docs/PROJECT_DOCUMENTATION.md` — narrative write-up of every phase.
- `docs/PROGRESS_UPDATE.md` / `docs/PROGRESS_PRESENTATION.md` / `docs/SPEAKING_SCRIPT.md` — class progress deliverables.
- Notebooks re-audited; structure intact (25 sections across 3 notebooks, all with intro markdown before code).

**Test summary:** 23/23 tests passing in ~3s.

---

## 6. How It Works — End-to-End Walk-Through

### 6.1 Reproducing the pipeline

```bash
git clone <repo-url> "DA Lab Project"
cd "DA Lab Project"
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Place dataset at .home-credit-default-risk/application_train.csv
# (one-time, NOT in repo)

# Phase 1 — cleaning notebook
.venv/Scripts/python -m jupyter nbconvert --to notebook --execute \
    notebooks/00_data_cleaning.ipynb --inplace
.venv/Scripts/python -m pytest tests/test_clean.py -v

# Phase 2 — EDA
.venv/Scripts/python notebooks/_build_eda_notebook.py

# Phase 3 — feature engineering + top-30
.venv/Scripts/python notebooks/_build_phase3_notebook.py

# Phase 4 — 4-model comparison + scoring function
.venv/Scripts/python notebooks/_build_phase4_notebook.py

# Verify
.venv/Scripts/python -m pytest tests/ -v
```

### 6.2 Running the dashboard

```bash
.venv/Scripts/python -m streamlit run dashboard/app.py
# open http://localhost:8501
```

### 6.3 Calling the scorer programmatically

```python
from src.models.score import score_application

result = score_application({
    "EXT_SOURCE_MEAN": 0.42,
    "EXT_SOURCE_1": 0.30,
    "EMPLOYED_YEARS": 4.0,
    "EXT_SOURCE_3": 0.55,
    "CODE_GENDER_M": 1.0,
    "AMT_CREDIT": 250_000.0,
    "AMT_GOODS_PRICE": 220_000.0,
    "AMT_ANNUITY": 22_000.0,
    "EXT_SOURCE_2": 0.50,
    "NAME_EDUCATION_TYPE_Higher education": 0.0,
    "DAYS_BIRTH": -16_000,
    "AGE_YEARS": 44.0,
    "CREDIT_GOODS_RATIO": 1.136,
    "DAYS_ID_PUBLISH": -3_000,
    "DAYS_EMPLOYED": -1_500,
    "DAYS_LAST_PHONE_CHANGE": -800,
    "NAME_EDUCATION_TYPE_Secondary / secondary special": 1.0,
    "NAME_CONTRACT_TYPE_Revolving loans": 0.0,
    "FLAG_DOCUMENT_3": 1.0,
    "TOTALAREA_MODE": 0.10,
    "DAYS_REGISTRATION": -3_000,
    "YEARS_BEGINEXPLUATATION_MODE": 0.95,
    "CREDIT_INCOME_RATIO": 1.7,
    "FLAG_OWN_CAR_Y": 0.0,
    "LIVINGAREA_MODE": 0.07,
    "ANNUITY_INCOME_RATIO": 0.15,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    "REGION_POPULATION_RELATIVE": 0.018,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
    "LIVINGAREA_MEDI": 0.07,
})

print(result)
# {'probability_of_default': 0.087, 'recommendation': 'Approve'}
```

The function:
1. Loads `models/best_model.pkl` (joblib-pickled XGBoost) on first call; cached thereafter.
2. Validates that all 30 expected keys are present.
3. Arranges them into a single row in the model's expected order.
4. Calls `predict_proba(X)[:, 1]` to get the PD.
5. Maps PD → recommendation via the policy in `_recommend()`.

---

## 7. The 3-Way Recommendation Policy

Encoded in `src/models/score.py`:

| PD band | Recommendation | Color | Operational meaning |
|---|---|---|---|
| `p < 0.20` | **Approve** | mint `#00D9B5` | Clearly low risk. Auto-approve. |
| `0.20 ≤ p < 0.50` | **Manual Review** | blue `#3B82F6` | Borderline — human underwriter decides. |
| `p ≥ 0.50` | **Reject** | red `#F87171` | Clearly high risk. Auto-reject. |

These thresholds are **policy** parameters, not model parameters. The model itself outputs a calibrated probability; the dashboard exposes these two thresholds as the only knobs a business operator would ever need to turn.

---

## 8. Key Findings & Business Value

### 8.1 What the data tells us

1. **External credit-bureau data (`EXT_SOURCE_*`) is the strongest single signal.** `EXT_SOURCE_MEAN` is rank #1 across all three importance rankings (XGB, RF, SHAP), with combined score 1.000.
2. **Naive credit-to-income ratio is misleading.** The default rate peaks in the middle of the CTI distribution (inverted-U) because high-CTI applicants are also high-income — the relationship is confounded by EXT_SOURCE.
3. **Segment-level risk is real but moderate.** Low-skill Laborers default at 17.2% vs Accountants at 4.8% (3.5×), and education shows a clean monotonic gradient. Use segments as auxiliary signals, not as standalone decision rules.
4. **Income is the dominant axis of risk** among employment + income features.

### 8.2 Model performance

The winning **XGBoost** model achieves **val AUC 0.7539, test AUC 0.7578** — a 2.1% lift over Logistic Regression and 0.7% over the tuned MLP. The gap between val and test AUC is small (0.0014), suggesting the model generalizes well and isn't over-fit to the validation slice.

At the 0.5 threshold, the model catches **67.5% of true defaults (recall)** with **16.4% precision** — the precision is low because the base rate is 8%, so every alert is a 6:1 false-positive ratio. The Manual Review band (0.20 ≤ p < 0.50) is the operational buffer that catches the borderline cases before they hit auto-decision.

### 8.3 What the dashboard enables

For an underwriter:
- **Portfolio Overview** lets a manager see the portfolio shape (215,257 train loans, 8.07% default rate, 17,377 default loans) and compare the 4 models at a glance.
- **Applicant Risk Scorer** is the day-to-day tool — enter the applicant, get the PD and recommendation in milliseconds.
- **Feature Importance Explorer** explains *why* the model says what it says — which inputs drove the PD up.
- **Segment Analysis** shows whether an applicant matches a known high-risk cohort.

---

## 9. Test Suite

23 tests across 4 files, run via:

```bash
.venv/Scripts/python -m pytest tests/ -v
```

| File | Tests | Phase |
|---|---:|---|
| `tests/test_clean.py` | 3 | Phase 1 |
| `tests/test_engineer.py` | 6 | Phase 3 |
| `tests/test_score.py` | 8 | Phase 4 |
| `tests/test_dashboard.py` | 6 | Phase 5 |
| **Total** | **23** | — |

Highlights:
- `test_scaler_fit_only_on_train` — guards against the most common cleaning refactor mistake (fitting the scaler on the full dataset).
- `test_ext_source_mean_handles_partial_NaN` — verifies the mean is partial-NaN-safe (NaN values are skipped, not zeroed).
- `test_score_application_returns_valid_shape` — full round-trip through the production scorer.
- `test_dashboard_modules_import` — verifies all four dashboard helper modules import cleanly.
- `test_palette_matches_master_prompt` — asserts the dashboard hex colors match the master-prompt palette exactly.

---

## 10. Honest Flags & Caveats

From `docs/PHASE_CHECKLIST.md`:

1. **No "AI agent" beyond the scoring function.** The master prompt's Phase 4 line reads "4+ ML/DL models + AI agent". This project ships the four models plus a deployable `score_application()` function (the "agent" in the operational sense — it consumes an applicant dict and returns a recommendation), but does not include a chat-style / tool-using agent. Per master-prompt §"A Note on Scope," this was intentionally out of scope.
2. **`score_application()` expects StandardScaler'd inputs.** Documented via the model artifact (XGBoost trained on StandardScaler'd top-30 parquet). The dashboard's `_form_to_features.py` refits the same scaler on the train slice and applies `(x − mean) / std` before calling `score_application()`.
3. **SMOTE was tried and rejected.** It dropped XGBoost val AUC from 0.7539 to 0.7203 in side-by-side runs. Class weighting chosen instead.
4. **Streamlit pinned to < 1.50** because the 1.50+ ASGI migration has a WebSocket-after-handshake regression on Python 3.14.

---

## 11. Authors & Acknowledgements

- **Course:** UIU Data Analytics Laboratory
- **Group:** DomainRange
- **Members:** Musfique Ahmed, Tasfiya Binte Karim
- **Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- **Project brief:** `puku_master_prompt.md` in this repo
- **Design refinement:** `/impeccable` design pass (Phase 5) for the dashboard theming

The full per-phase deliverable checklist lives in `docs/PHASE_CHECKLIST.md`. The full project narrative lives in `docs/PROJECT_DOCUMENTATION.md`. The class-presentation materials live in `docs/PROGRESS_PRESENTATION.md` and `docs/SPEAKING_SCRIPT.md`.
