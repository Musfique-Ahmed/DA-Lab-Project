# Master Prompt — Credit Risk Intelligence App (for puku-cli `/plan`)

Paste everything below into puku-cli as the argument to `/plan`.

---

## Project Goal

Build **Credit Risk Intelligence** — an AI-driven loan default forecasting and decisioning system — inside this repository, end to end: data cleaning, EDA, feature selection, four ML/DL models, a Credit Risk Scoring Agent, and a heavily custom-themed Streamlit dashboard. This supports a UIU Data Analytics Laboratory course project (Group: DomainRange; Musfique Ahmed, Tasfiya Binte Karim).

## Current Repository State — Do Not Re-download Anything

The project root already looks like this:

```
DA LAB PROJECT/
├── .puku/
├── home-credit-default-risk/
│   ├── application_test.csv
│   ├── application_train.csv
│   ├── bureau_balance.csv
│   ├── bureau.csv
│   ├── credit_card_balance.csv
│   ├── HomeCredit_columns_description.csv
│   ├── installments_payments.csv
│   ├── POS_CASH_balance.csv
│   ├── previous_application.csv
│   └── sample_submission.csv
├── .gitattributes
├── .gitignore
└── README.md
```

**Only `home-credit-default-risk/application_train.csv` is in scope for this project** (307,511 rows, 122 columns — 106 numerical, 16 categorical — target column `TARGET`, ~92%/8% class balance). Do not load, join, or reference `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, or `installments_payments.csv` anywhere in the pipeline — that's out of scope by design. `application_test.csv` and `sample_submission.csv` are also unused (this is a coursework project, not a competition submission). `HomeCredit_columns_description.csv` may be read for reference when writing feature documentation, but nothing else.

## Execution Rules — Read This Before Doing Anything

1. **This is a 6-phase project. Work through exactly one phase per run.** At the end of each phase: summarize what you did, list every file created or modified, give me the exact command(s) to run to verify it myself, then **stop and wait for my explicit go-ahead** before starting the next phase. Do not chain phases together in one sitting even if you have budget left.
2. **Use `/plan` at the start of each phase** to lay out the concrete steps before touching files, so I can see the plan before you execute it.
3. **Check for and use relevant skills and agents before improvising.** Run `/skills` and `/agents` to see what's available in this project and personal scope before starting a phase. Specifically:
   - For **any frontend/dashboard styling work (Phase 5)**, invoke **`/impecable`** for the visual/UI design — don't hand-roll the Streamlit theming without it.
   - If a heavy, long-running analysis step (e.g., full EDA generation, SHAP computation across all 122 features) risks consuming a lot of the conversation's context, prefer running it as a **forked** skill/sub-agent (`context: fork`) so the parent session stays lean.
   - If no existing skill fits a repeated task you find yourself doing more than once in this project (e.g., "run the notebook and report pass/fail," "regenerate the EDA charts"), consider creating a small project skill under `.puku-cli/skills/` for it rather than repeating the same manual steps each phase.
   - Use the built-in `/verify` skill after each phase to confirm the code actually runs before declaring the phase done.
4. **Git workflow:** run `git remote -v` first. Commit locally after each phase with a clear conventional-commit message (`feat:`, `fix:`, `docs:`, etc.) regardless of remote status. **Do not `git push` under any circumstances unless I explicitly ask you to** — even if a remote is configured.
5. **Don't fabricate results.** If a model's performance is worse than expected, or a step fails, report it honestly in the phase summary rather than smoothing it over. This is graded coursework — the notebook needs to reflect what actually happened.
6. **Comment and narrate the code** (docstrings, markdown cells in the notebook, section headers) as if a course instructor will read it — this isn't just an internal tool, it's part of the submission.

## Tech Stack (fixed — don't substitute alternatives without asking)

- Python 3.11, managed via a `venv` at the project root (`python -m venv .venv`)
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn`, `plotly` — EDA visualization
- `scikit-learn` — Logistic Regression, Random Forest, preprocessing, metrics
- `xgboost` or `lightgbm` (your choice, pick one and justify it briefly) — gradient boosting model
- `tensorflow`/`keras` or `pytorch` (your choice) — the MLP neural network
- `shap` — feature importance/interpretability
- `imbalanced-learn` — SMOTE, for handling the ~92/8 class imbalance
- `streamlit` — dashboard, with custom CSS/HTML injected for the visual theme (this is where `/impecable` comes in)
- `pytest` — a thin test layer for the data-cleaning and scoring functions (not a full test suite — just enough to catch regressions)
- Track all dependencies in a `requirements.txt`, pinned to major versions

## Design System for the Dashboard (Phase 5) — hand this to `/impecable`

Match the color palette already used in the project's slide deck and script, so the dashboard feels like part of the same product:

- Background: `#0A1428` (deep navy)
- Card panels: `#10203D` / `#162A4D`
- Primary accent: `#00D9B5` (mint/teal)
- Supporting accent: `#3B82F6` (blue)
- Risk/warning accent: `#F87171` (soft red) — reserved for default-risk indicators only
- Text: `#FFFFFF` (headers), `#CBD5E1` (body), `#94A3B8` (muted)
- Dark, premium, fintech aesthetic throughout — no default Streamlit light theme left visible anywhere

---

## Phase 1 — Environment Setup & Data Cleaning

**Goal:** a clean, model-ready dataset and a reproducible environment.

1. Create `.venv`, install the pinned dependency set, write `requirements.txt`
2. Scaffold the repo: `src/`, `src/data/`, `src/features/`, `src/models/`, `notebooks/`, `dashboard/`, `reports/`, `tests/`
3. Write `src/data/load.py` to load `application_train.csv` and assert its shape is exactly `(307511, 122)` — fail loudly if not
4. Write a data-quality profiling step (missing-value %, dtype breakdown) and save it as `reports/data_quality_report.md`
5. Implement cleaning in `src/data/clean.py`:
   - Replace `DAYS_EMPLOYED == 365243` with `NaN`
   - Drop columns with >60% missing values (log which ones)
   - Impute remaining numeric gaps with median, categorical gaps with mode
   - One-hot encode low-cardinality categoricals; target-encode (or frequency-encode) high-cardinality ones
   - Standardize numeric features with `StandardScaler`, fit only on the training split (avoid leakage — split before scaling)
6. Save the cleaned dataset to `data/processed/train_clean.parquet` (ensure `data/` is gitignored appropriately for large files — check `.gitignore` and add rules if missing)
7. Write `tests/test_clean.py` covering the `DAYS_EMPLOYED` fix and the missing-value threshold logic
8. **Stop here.** Summarize row/column counts before and after cleaning, which columns were dropped and why, and how to run the tests.

## Phase 2 — Exploratory Data Analysis

**Goal:** a notebook section with genuine business insight, not generic charts.

1. In `notebooks/01_eda.ipynb`, build:
   - Univariate distributions: income, credit amount, age, employment length, target class balance
   - Default rate vs. credit-to-income ratio
   - `EXT_SOURCE_1/2/3` vs. default outcome (correlation + visual)
   - Employment length & income stability vs. risk
   - Default rate across education, occupation, contract type
   - A correlation heatmap across numeric features to flag multicollinearity ahead of modeling
2. Explicitly test these three hypotheses with statistics (not just charts) and report the result of each in a markdown cell:
   - H1: lower `EXT_SOURCE` scores correlate with higher default probability
   - H2: higher credit-to-income ratio increases default likelihood
   - H3: certain occupation/education segments show measurably elevated risk
3. Write a "Key Insights" markdown section at the end of the notebook summarizing findings in plain business language
4. Export the key charts as static images to `reports/figures/` for reuse in the dashboard/deck later
5. **Stop here.** Summarize which hypotheses were supported, contradicted, or inconclusive, and point me to the notebook section.

## Phase 3 — Feature Engineering & Selection

**Goal:** a justified, reduced feature set.

1. Engineer a small set of derived features in `src/features/engineer.py`: credit-to-income ratio, annuity-to-income ratio, age in years (from `DAYS_BIRTH`), and any other clearly justified derived feature
2. Train a quick Random Forest and the chosen gradient-boosting model on the full feature set purely to extract importances
3. Compute SHAP values on a representative sample; cross-check against the importance ranking
4. Select the top ~30 features by combined ranking; document the full ranked list in `reports/feature_importance.md`
5. Train identical baseline Logistic Regression models on (a) all features and (b) the reduced top-30 set; compare validation AUC and report which one wins and by how much
6. **Stop here.** Show me the top-30 list, the baseline-vs-reduced AUC comparison, and your recommendation on which feature set to carry forward.

## Phase 4 — Model Building & Tuning

**Goal:** four trained, compared models and a working scoring function.

1. Stratified train/validation/test split (70/15/15) on the reduced feature set from Phase 3
2. Address class imbalance — try both class-weighting and SMOTE, compare on validation, pick the better one and justify it
3. Train and tune all four models:
   - Logistic Regression (interpretable baseline)
   - Random Forest
   - XGBoost/LightGBM (whichever you chose)
   - Neural Network (MLP)
4. Evaluate consistently across all four: AUC-ROC (primary), precision, recall, F1, confusion matrix at a chosen threshold. Present results in one comparison table.
5. Pick the best model; wrap it in `src/models/score.py` as a `score_application(input: dict) -> dict` function returning `{probability_of_default, recommendation}` where recommendation is one of `Approve` / `Manual Review` / `Reject`
6. Save the final model artifact to `models/` (gitignore large binaries if needed, document how to regenerate them instead of committing if they're large)
7. **Stop here.** Show me the 4-model comparison table and the chosen model with justification.

## Phase 5 — Dashboard (use `/impecable` for the design work)

**Goal:** a Streamlit app matching the design system above, wired to the real scoring function from Phase 4.

1. `dashboard/app.py` with four sections/tabs:
   - **Portfolio Overview** — KPI cards for approval rate, default rate, application volume, computed from `train_clean.parquet`
   - **Applicant Risk Scorer** — input form (income, credit amount, employment length, `EXT_SOURCE` values, etc.) wired live to `score_application()` from Phase 4, showing the probability and recommendation
   - **Feature Importance Explorer** — interactive chart of the SHAP/importance ranking from Phase 3
   - **Segment Analysis** — filterable charts by income band, education, region (reuse EDA logic from Phase 2)
2. Invoke `/impecable` specifically for the visual design/theming pass — custom CSS/HTML injected into Streamlit, matching the hex palette above exactly, dark throughout, no leftover default Streamlit styling visible
3. Test the app runs locally end to end (`streamlit run dashboard/app.py`) with no broken model-loading or path issues
4. **Stop here.** Give me the run command and a short note on what `/impecable` changed.

## Phase 6 — Notebook Cleanup, Docs & Final Cross-Check

**Goal:** a submission-ready repo.

1. Clean up all notebooks: remove dead/debug cells, add markdown narrative between sections so each notebook reads as a coherent story
2. Update `README.md` with: project overview, repo structure, setup instructions, how to run the notebooks, how to run the dashboard
3. Cross-check the final repo against this checklist and report status on each line:
   - [ ] 25+ features, 2,000+ rows (Phase 1)
   - [ ] Cleaning & preprocessing (Phase 1)
   - [ ] EDA with business insight (Phase 2)
   - [ ] Feature selection + baseline-vs-reduced comparison (Phase 3)
   - [ ] Summarized key insights (Phase 2 & 3)
   - [ ] 4+ ML/DL models + AI agent (Phase 4)
   - [ ] Interactive dashboard (Phase 5)
   - [ ] Notebook + docs ready for submission (Phase 6)
4. **Stop here.** Give me the final checklist status and flag anything that's incomplete or weaker than it should be — don't mark something done if it isn't.

---

## A Note on Scope

Do not add authentication, deployment/hosting, CI/CD pipelines, or additional datasets unless I explicitly ask for them in a later phase. Keep the scope exactly to what's listed above — this is a course project with a defined rubric, not a production system.
