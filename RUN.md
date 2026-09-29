# RUN — Dashboard Demo Cheat-Sheet

> Quick-start so the Phase 5 Streamlit dashboard can be demoed in front
> of an instructor without going through the full DVC pipeline.

## 1. One-time setup

Requires **Python 3.14** (matches the cp314 wheels pinned in
`requirements.txt`).

```bash
# from repo root (DA-Lab-Project/)
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

# macOS / Linux
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Required artifacts

These must exist before `streamlit run` will render all four tabs:

| Path                                        | Purpose                                |
| ------------------------------------------- | -------------------------------------- |
| `models/best_model.pkl`                     | XGBoost model loaded by the Scorer tab |
| `data/processed/train_clean.parquet`        | KPI cards (TARGET + SPLIT columns)     |
| `data/processed/train_top30.parquet`        | Cohort medians for the Scorer form     |
| `data/raw/application_train.csv`            | Segment Analysis (raw categoricals)    |
| `reports/feature_importance.md`             | Feature Importance tab                 |
| `reports/phase4_model_comparison.md`        | Sidebar model stats + policy explainer |

All of these are already produced by prior phases. If
`data/processed/train_top30.parquet` is ever missing (it isn't tracked
by `dvc.yaml`), regenerate it with:

```bash
.venv\Scripts\python notebooks/_build_phase3_notebook.py
```

For everything else, the canonical regen command is:

```bash
.venv\Scripts\dvc repro
```

## 3. Run the dashboard

```bash
# Windows
.venv\Scripts\python -m streamlit run dashboard/app.py

# macOS / Linux
.venv/bin/python -m streamlit run dashboard/app.py
```

A browser tab opens automatically at **http://localhost:8501**.

For headless / remote demos (no browser auto-open):

```bash
.venv\Scripts\python -m streamlit run dashboard/app.py \
    --server.headless=true --server.port=8501 --browser.gatherUsageStats=false
```

Then connect to the printed `Network URL` instead.

## 4. Demo script (4 tabs)

1. **Portfolio Overview**
   - 4 KPI cards: total applications, default rate, avg PD, default loan count
   - Class-balance bar chart
   - 4-model AUC comparison (XGBoost / MLP / RF / LR)
   - 3-way policy explainer: Approve (PD < 0.20) | Manual Review (0.20–0.50) | Reject (≥ 0.50)
2. **Applicant Risk Scorer**
   - 14-widget form: income, credit, annuity, age, employed years, three
     `EXT_SOURCE_*` sliders, gender, education, contract type,
     `FLAG_DOCUMENT_3`, `FLAG_OWN_CAR`, region population.
   - Press **Score** → live probability of default + recommendation card +
     gauge + sensitivity panel.
3. **Feature Importance**
   - Top-30 horizontal bar chart parsed from `reports/feature_importance.md`,
     with a category-source filter.
4. **Segment Analysis**
   - Default-rate-by-segment plots with binomial-normal 95% CI error bars
     across gender, education, occupation, contract type, etc.

## 5. Stop the server

`Ctrl+C` in the terminal that launched Streamlit.

## 6. Troubleshooting

- **Tab fails to load** → every loader raises a `FileNotFoundError` naming
  the exact path it expected. Re-run the prior phase (1–4) notebook
  builder, or `dvc repro`, to regenerate it.
- **Stale data after edits** → hamburger menu (top right) → **Clear cache**
  → rerun.
- **Port 8501 already in use** → pass `--server.port=8502` (or any free port).

---

See `dashboard/README.md` for the full Phase 5 design notes.
