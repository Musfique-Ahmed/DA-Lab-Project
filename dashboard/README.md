# Credit Risk Intelligence — Dashboard

Streamlit app wired to the production scoring function and EDA outputs
from Phases 1–4.

## Run

```bash
# from repo root
.venv/Scripts/python -m streamlit run dashboard/app.py
```

The app boots on http://localhost:8501 and exposes four tabs.

## Required artifacts

These files must exist before the dashboard can render:

| Path | Purpose | Source phase |
|---|---|---|
| `models/best_model.pkl`           | XGBoost artifact loaded by `score_application()` | Phase 4 |
| `data/processed/train_top30.parquet` | Cohort medians for the 9 "background" form fields | Phase 3 |
| `data/processed/train_clean.parquet`  | KPI cards in Portfolio Overview | Phase 1 |
| `.home-credit-default-risk/application_train.csv` | Segment Analysis (raw categoricals) | unchanged |
| `reports/feature_importance.md` | Feature Importance Explorer | Phase 3 |
| `reports/phase4_model_comparison.md` | Sidebar model stats + Overview policy explainer | Phase 4 |

If any of these are missing, the relevant tab will fail loudly with the
path it expected. Re-run the prior phase's notebook builder to regenerate.

## Tabs

### 1. Portfolio Overview

Four KPI cards (total applications, default rate, average PD, default
loan count) computed from the training slice of `train_clean.parquet`.
Plus a class-balance bar chart, a 4-model AUC comparison bar chart, and
the 3-way recommendation policy explainer.

### 2. Applicant Risk Scorer

A 14-widget input form (income, credit, annuity, age, employed years,
3 EXT_SOURCE sliders, gender radio, education selectbox, contract type
radio, FLAG_DOCUMENT_3 + FLAG_OWN_CAR checkboxes, region population
slider). On submit:

- The 30-feature dict is built by `dashboard/_form_to_features.py`
  (computes the 7 engineered features, sets the one-hot dummies, fills
  the 9 "background" columns with the cohort median).
- `score_application()` returns `{probability_of_default, recommendation}`.
- A hero recommendation card shows the PD with semantic color (mint =
  Approve, blue = Manual Review, red = Reject).
- A horizontal segmented gauge marks where `p` falls between the two
  thresholds.
- An "sensitivity" expander bumps each numeric input ±10% and re-scores,
  showing which inputs drive the most movement.

### 3. Feature Importance Explorer

Parses `reports/feature_importance.md` into a DataFrame with 7 columns
(rank, feature, xgb_rank, rf_rank, shap_rank, mean_rank, combined_score).
A Plotly horizontal bar chart shows all 30 with a category multiselect
(EXT_SOURCE / Time-derived / One-hot-or-flag / Ratio / Amount / Area).

### 4. Segment Analysis

Loads the raw `application_train.csv`. Two filters:

1. **Region rating** — restrict to one or more region ratings.
2. **Segment dimension** — Income band (deciles), Employment length
   (<1 yr / 1–3 / 3–5 / 5–10 / 10+), Education, Income type, Contract
   type, Gender, Owns a car, Owns realty, Region rating.

Default rate per segment is plotted as a Plotly bar chart with
binomial-normal 95% CI error bars. Underlying table is downloadable
as CSV.

## Design

The visual world is pinned by the master prompt and refined by
`/impeccable`:

- Background: `#0A1428` deep navy with subtle mint/blue radial gradient glows.
- Card panels: `#10203D` over `#0B1830` with `0 8px 24px rgba(0,0,0,0.35)` depth shadows.
- Mint `#00D9B5` accent reserved for Approve + primary CTAs.
- Soft red `#F87171` reserved for Reject + risk indicators.
- Blue `#3B82F6` for Manual Review + test AUC.
- Type: Inter for body, JetBrains Mono for numerics (KPI values, PD %).
- Browser surfaces themed: caret color, text selection, scrollbar,
  focus ring all on the palette.

CSS lives in `dashboard/_theme.py` as one block — a single source of
truth so a future design pass can swap the entire visual world without
touching the app code.

## Files

```
dashboard/
├── app.py                  # main app + 4 tabs
├── _theme.py               # palette + CSS injection
├── _loaders.py             # cached loaders (parquet, CSV, md)
├── _form_to_features.py    # 14-widget form → 30-feature dict
├── _segments.py            # income / employment bucketing, default-rate aggregation
└── README.md               # this file
```

## Smoke test

```bash
# AppTest harness — boots the app in-process and asserts no exceptions.
.venv/Scripts/python.exe -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('dashboard/app.py', default_timeout=30)
at.run()
assert not at.exception, at.exception
assert not at.error, at.error
print('OK')
"
```

Expected: `OK` (no exceptions, no errors, 0 warnings after the
`use_container_width` → `width='stretch'` migration).
