# Production Model — Credit Risk Intelligence (Phase 4)

## Artifact

- `models/best_model.pkl` — scikit-learn XGBoost (joblib; gitignored)

## Winner

- Model: **XGBoost**
- Val AUC: **0.7539**
- Test AUC: **0.7578**
- Threshold: **0.500**
- Imbalance: `balanced`
- Config: `{'imbalance': 'balanced', 'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'threshold': 0.5, 'scale_pos_weight': 11.38746619094205}`

## Recommendation policy

- `p < 0.2` → **Approve**
- `0.2 <= p < 0.5` → **Manual Review**
- `p >= 0.5` → **Reject**

## Input schema (30 features)

`score_application(input: dict) -> dict` requires all of:

- `EXT_SOURCE_MEAN`
- `EXT_SOURCE_1`
- `EMPLOYED_YEARS`
- `EXT_SOURCE_3`
- `CODE_GENDER_M`
- `AMT_CREDIT`
- `AMT_GOODS_PRICE`
- `AMT_ANNUITY`
- `EXT_SOURCE_2`
- `NAME_EDUCATION_TYPE_Higher education`
- `DAYS_BIRTH`
- `AGE_YEARS`
- `CREDIT_GOODS_RATIO`
- `DAYS_ID_PUBLISH`
- `DAYS_EMPLOYED`
- `DAYS_LAST_PHONE_CHANGE`
- `NAME_EDUCATION_TYPE_Secondary / secondary special`
- `NAME_CONTRACT_TYPE_Revolving loans`
- `FLAG_DOCUMENT_3`
- `TOTALAREA_MODE`
- `DAYS_REGISTRATION`
- `YEARS_BEGINEXPLUATATION_MODE`
- `CREDIT_INCOME_RATIO`
- `FLAG_OWN_CAR_Y`
- `LIVINGAREA_MODE`
- `ANNUITY_INCOME_RATIO`
- `AMT_REQ_CREDIT_BUREAU_YEAR`
- `REGION_POPULATION_RELATIVE`
- `DEF_60_CNT_SOCIAL_CIRCLE`
- `LIVINGAREA_MEDI`

## Regeneration

```
.venv/Scripts/python notebooks/_build_phase4_notebook.py
```
