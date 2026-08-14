"""Programmatically build the Phase 3 feature-selection notebook.

Run with:
    .venv/Scripts/python notebooks/_build_phase3_notebook.py

The script constructs notebooks/02_feature_selection.ipynb, executes it
in-process via nbclient, and writes:
  - The executed notebook
  - 4 figures to reports/figures/ (13_...16_)
  - reports/feature_importance.md
  - data/processed/train_engineered.parquet
  - data/processed/train_top30.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "02_feature_selection.ipynb"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
IMPORTANCE_MD = REPO_ROOT / "reports" / "feature_importance.md"
PARQUET_ENG = REPO_ROOT / "data" / "processed" / "train_engineered.parquet"
PARQUET_TOP30 = REPO_ROOT / "data" / "processed" / "train_top30.parquet"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_ENG.parent.mkdir(parents=True, exist_ok=True)


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(text.strip())


# --- Sections ---------------------------------------------------------------

def section_0_setup() -> list[nbformat.NotebookNode]:
    cells = [
        md("## 0. Setup\n\n"
           "Load the cleaned parquet, derive the 7 engineered features, persist the "
           "engineered dataset, and import the importance + SHAP helpers.")
    ]
    cells.append(code(
        f"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay

# Repo-root substitution (set by the notebook builder).
REPO_ROOT = Path(r"__REPO_ROOT_PLACEHOLDER__")
PARQUET_CLEAN = REPO_ROOT / "data" / "processed" / "train_clean.parquet"
PARQUET_ENG = REPO_ROOT / "data" / "processed" / "train_engineered.parquet"
PARQUET_TOP30 = REPO_ROOT / "data" / "processed" / "train_top30.parquet"
FIG_DIR = REPO_ROOT / "reports" / "figures"
IMPORTANCE_MD = REPO_ROOT / "reports" / "feature_importance.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Phase-5 palette for charts.
PALETTE = {{
    "bg":      "#0A1428",
    "panel":   "#10203D",
    "mint":    "#00D9B5",
    "blue":    "#3B82F6",
    "red":     "#F87171",
    "white":   "#FFFFFF",
    "body":    "#CBD5E1",
    "muted":   "#94A3B8",
}}
mpl.rcParams.update({{
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["muted"],
    "axes.labelcolor":   PALETTE["body"],
    "axes.titlecolor":   PALETTE["white"],
    "xtick.color":       PALETTE["body"],
    "ytick.color":       PALETTE["body"],
    "text.color":        PALETTE["body"],
    "grid.color":        PALETTE["panel"],
    "grid.alpha":        0.5,
    "savefig.facecolor": PALETTE["bg"],
    "savefig.dpi":       130,
}})

print(f"REPO_ROOT = {{REPO_ROOT}}")
print(f"Cleaned parquet exists: {{PARQUET_CLEAN.exists()}}")
        """.strip()
    ))
    cells.append(code(
        """
# Load cleaned data and engineer features.
df = pd.read_parquet(PARQUET_CLEAN)
print(f"Loaded cleaned parquet: {df.shape}")

from src.features.engineer import engineer_features, ENGINEERED_COLUMNS
df_eng = engineer_features(df)
print(f"Engineered shape: {df_eng.shape}")
print(f"Engineered columns ({len(ENGINEERED_COLUMNS)}):")
for c in ENGINEERED_COLUMNS:
    n_nan = int(df_eng[c].isna().sum())
    print(f"  {c:25s} NaN={n_nan:,} ({n_nan/len(df_eng):.1%})")

# Persist the engineered dataset (all rows, all slices, all columns).
df_eng.to_parquet(PARQUET_ENG, index=False)
print(f"Wrote {PARQUET_ENG}")
        """.strip()
    ))
    return cells


def section_1_engineered_sanity() -> list[nbformat.NotebookNode]:
    cells = [md("## 1. Engineered-Feature Sanity Check\n\n"
                "Descriptive statistics for each engineered column, plus a 4-panel histogram.")]
    cells.append(code(
        """
feat_stats = df_eng[list(ENGINEERED_COLUMNS)].describe().T
print(feat_stats[["mean", "std", "min", "50%", "max"]].round(3).to_string())
        """.strip()
    ))
    cells.append(code(
        """
# 4-panel histogram of the most informative engineered features.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.flat, ["AGE_YEARS", "CREDIT_INCOME_RATIO",
                                "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN"]):
    vals = df_eng[col].dropna()
    if col == "CREDIT_INCOME_RATIO" or col == "ANNUITY_INCOME_RATIO":
        ax.hist(vals.clip(upper=vals.quantile(0.99)), bins=60,
                color=PALETTE["mint"], edgecolor=PALETTE["bg"], alpha=0.85)
        ax.set_title(f"{col} (clipped at 99th pct)")
    else:
        ax.hist(vals, bins=60, color=PALETTE["mint"],
                edgecolor=PALETTE["bg"], alpha=0.85)
        ax.set_title(col)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "13_engineered_features.png")
plt.show()
print("Saved 13_engineered_features.png")
        """.strip()
    ))
    return cells


def section_2_importance() -> list[nbformat.NotebookNode]:
    cells = [md("## 2. XGBoost + Random Forest Importance\n\n"
                "Train both models on the **train slice** (215,257 rows × "
                "148 features) to extract importances. No CV — this is purely "
                "to rank features; Phase 4 does the real model training.")]
    cells.append(code(
        """
from src.features.select import train_xgb_and_rf_importances

df_train = df_eng[df_eng["SPLIT"] == "train"].reset_index(drop=True)
feature_cols = [c for c in df_train.columns if c not in ("SK_ID_CURR", "TARGET", "SPLIT")]
X_train = df_train[feature_cols]
y_train = df_train["TARGET"]

print(f"Training rows: {len(X_train):,}, features: {len(feature_cols)}")
print(f"Positive rate: {y_train.mean():.4%}")

t0 = time.time()
imp = train_xgb_and_rf_importances(X_train, y_train, random_state=42)
print(f"Importance extraction took {time.time() - t0:.1f}s")
print(f"Top 15 features by mean rank:")
print(imp.head(15)[["feature", "xgb_importance", "rf_importance",
                    "xgb_rank", "rf_rank", "mean_rank"]].to_string(index=False))
        """.strip()
    ))
    cells.append(code(
        """
# 2-panel chart: top-20 by XGB and by RF.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, col, label in [
    (axes[0], "xgb_importance", "XGBoost (gain)"),
    (axes[1], "rf_importance", "RandomForest (impurity)"),
]:
    top = imp.nlargest(20, col).sort_values(col, ascending=True)
    ax.barh(top["feature"], top[col], color=PALETTE["mint"], edgecolor=PALETTE["bg"])
    ax.set_title(f"Top 20 features — {label}")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "14_importance_rf_xgb.png")
plt.show()
print("Saved 14_importance_rf_xgb.png")
        """.strip()
    ))
    return cells


def section_3_shap() -> list[nbformat.NotebookNode]:
    cells = [md("## 3. SHAP on XGBoost\n\n"
                "TreeSHAP is exact on boosted trees; 5,000 rows × 300 trees is enough "
                "for stable per-feature mean |SHAP|.")]
    cells.append(code(
        """
import shap
from src.features.select import shap_values_xgb

t0 = time.time()
sv, fn = shap_values_xgb(X_train, y_train, sample_size=5_000, random_state=42)
print(f"TreeSHAP took {time.time() - t0:.1f}s, shape {sv.shape}")
print(f"Top 15 features by mean |SHAP|:")
shap_mean_abs = np.abs(sv).mean(axis=0)
shap_top = (pd.DataFrame({"feature": fn, "shap_mean_abs": shap_mean_abs})
            .sort_values("shap_mean_abs", ascending=False).head(15))
print(shap_top.to_string(index=False))
        """.strip()
    ))
    cells.append(code(
        """
# SHAP summary (beeswarm) for the top-15 features.
shap_top15 = shap_top["feature"].tolist()
idx = [fn.index(f) for f in shap_top15]
shap.summary_plot(
    sv[:, idx],
    features=X_train.sample(5_000, random_state=42).iloc[:, idx].values,
    feature_names=shap_top15,
    plot_size=(10, 7),
    show=False,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "15_shap_summary.png", facecolor=PALETTE["bg"], dpi=130)
plt.close()
print("Saved 15_shap_summary.png")
        """.strip()
    ))
    return cells


def section_4_combine() -> list[nbformat.NotebookNode]:
    cells = [md("## 4. Combined Ranking + Top-30 Selection\n\n"
                "Combine the XGB rank, RF rank, and SHAP rank into a single mean-rank "
                "and pick the top 30.")]
    cells.append(code(
        """
from src.features.select import select_top_n

top30 = select_top_n(imp, sv, fn, n=30)
print("Top 30 features:")
print(top30[["feature", "xgb_rank", "rf_rank", "shap_rank",
              "mean_rank", "combined_score"]].to_string(index=False))
        """.strip()
    ))
    cells.append(code(
        """
# Persist the reduced top-30 dataset (all slices).
top30_cols = list(top30["feature"])
keep_cols = ["SK_ID_CURR", "TARGET", "SPLIT"] + top30_cols
df_top30 = df_eng[keep_cols].copy()
df_top30.to_parquet(PARQUET_TOP30, index=False)
print(f"Wrote {PARQUET_TOP30} ({df_top30.shape[0]:,} rows, {df_top30.shape[1]} cols)")
        """.strip()
    ))
    cells.append(code(
        """
# Render the feature-importance markdown report.
lines = []
lines.append("# Feature Importance -- Credit Risk Intelligence (Phase 3)\\n")
lines.append("Generated by `notebooks/02_feature_selection.ipynb`.\\n")
lines.append(f"All {len(imp)} features ranked by mean of XGB rank, RF rank, and SHAP rank.\\n")
lines.append("\\n## Top-30 Selected Features\\n")
lines.append("| Rank | Feature | XGB rank | RF rank | SHAP rank | mean_rank | combined_score |")
lines.append("|---|---|---|---|---|---|---|")
for i, row in top30.iterrows():
    lines.append(
        f"| {i+1} | `{row['feature']}` | {int(row['xgb_rank'])} | "
        f"{int(row['rf_rank'])} | {int(row['shap_rank'])} | "
        f"{row['mean_rank']:.1f} | {row['combined_score']:.3f} |"
    )

lines.append("\\n## Full Ranking (all " + str(len(imp)) + " features)\\n")
lines.append("| Rank | Feature | XGB rank | RF rank | SHAP rank | mean_rank |")
lines.append("|---|---|---|---|---|---|")
# Build a feature -> shap_rank map for the full ranking.
shap_full_rank = (pd.DataFrame({"feature": fn, "shap_mean_abs": np.abs(sv).mean(axis=0)})
                  .assign(rank=lambda d: d["shap_mean_abs"].rank(ascending=False, method="min"))
                  .set_index("feature")["rank"])
imp_sorted = imp.sort_values("mean_rank").reset_index(drop=True)
for i, row in imp_sorted.iterrows():
    feat = row["feature"]
    shap_rank = int(shap_full_rank.loc[feat])
    lines.append(
        f"| {i+1} | `{feat}` | {int(row['xgb_rank'])} | "
        f"{int(row['rf_rank'])} | {shap_rank} | {row['mean_rank']:.1f} |"
    )

lines.append("\\n## Agreements and Disagreements\\n")
top10 = top30.head(10)["feature"].tolist()
lines.append("**All three rankings agree on the top-10 broadly**: " + ", ".join("`" + f + "`" for f in top10) + ".\\n")
# Heuristic: features ranked in top 10 by any single model.
top10_xgb = set(imp.nsmallest(10, "xgb_rank")["feature"])
top10_rf = set(imp.nsmallest(10, "rf_rank")["feature"])
top10_shap = set(shap_top["feature"].head(10))
in_all = top10_xgb & top10_rf & top10_shap
only_xgb = top10_xgb - top10_rf - top10_shap
only_rf = top10_rf - top10_xgb - top10_shap
only_shap = top10_shap - top10_xgb - top10_rf
lines.append(f"- Features in top-10 of all three rankings: {len(in_all)}.\\n")
lines.append(f"- Top-10 by XGB only: {len(only_xgb)} -- " + ", ".join("`" + f + "`" for f in only_xgb) + ".\\n")
lines.append(f"- Top-10 by RF only: {len(only_rf)} -- " + ", ".join("`" + f + "`" for f in only_rf) + ".\\n")
lines.append(f"- Top-10 by SHAP only: {len(only_shap)} -- " + ", ".join("`" + f + "`" for f in only_shap) + ".\\n")

IMPORTANCE_MD.write_text("\\n".join(lines), encoding="utf-8")
print(f"Wrote {IMPORTANCE_MD}")
        """.strip()
    ))
    return cells


def section_5_baseline_vs_reduced() -> list[nbformat.NotebookNode]:
    cells = [md("## 5. Baseline-vs-Reduced Logistic Regression (5-fold CV)\n\n"
                "Per user direction, use **5-fold stratified CV over train+val "
                "(261k rows)** for a more robust comparison than a single validation AUC.")]
    cells.append(code(
        """
# Combine train+val for CV; we'll fit LR and report per-fold AUC.
df_tv = df_eng[df_eng["SPLIT"].isin(["train", "val"])].reset_index(drop=True)
print(f"Train+val rows: {len(df_tv):,}, positive rate: {df_tv['TARGET'].mean():.4%}")

all_features = [c for c in df_eng.columns if c not in ("SK_ID_CURR", "TARGET", "SPLIT")]
top30_features = list(top30["feature"])

X_all = df_tv[all_features].values
y_all = df_tv["TARGET"].values
X_top = df_tv[top30_features].values

print(f"X_all shape: {X_all.shape}")
print(f"X_top shape: {X_top.shape}")
        """.strip()
    ))
    cells.append(code(
        """
def cv_lr_auc(X, y, *, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Median-impute NaNs on train stats; this is the simplest safe baseline.
        med = np.nanmedian(X[train_idx], axis=0)
        Xtr = np.where(np.isnan(X[train_idx]), med, X[train_idx])
        Xva = np.where(np.isnan(X[val_idx]),   med, X[val_idx])
        # StandardScaler fit on train only.
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xva = sc.transform(Xva)
        lr = LogisticRegression(max_iter=200, n_jobs=-1, solver="lbfgs", C=1.0)
        lr.fit(Xtr, y[train_idx])
        aucs.append(roc_auc_score(y[val_idx], lr.predict_proba(Xva)[:, 1]))
    return np.array(aucs)

t0 = time.time()
auc_all = cv_lr_auc(X_all, y_all)
print(f"All-features CV ({len(auc_all)} folds) took {time.time()-t0:.1f}s")
print(f"  mean AUC = {auc_all.mean():.4f}  std = {auc_all.std():.4f}")
print(f"  per-fold: {[f'{a:.4f}' for a in auc_all]}")

t0 = time.time()
auc_top = cv_lr_auc(X_top, y_all)
print(f"Top-30 CV ({len(auc_top)} folds) took {time.time()-t0:.1f}s")
print(f"  mean AUC = {auc_top.mean():.4f}  std = {auc_top.std():.4f}")
print(f"  per-fold: {[f'{a:.4f}' for a in auc_top]}")

delta = auc_top.mean() - auc_all.mean()
print(f"\\nDelta (top30 - all): {delta:+.4f}")
        """.strip()
    ))
    cells.append(code(
        """
# ROC curves from the first fold of each (representative).
from sklearn.preprocessing import StandardScaler

def first_fold_roc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X, y))
    med = np.nanmedian(X[train_idx], axis=0)
    Xtr = np.where(np.isnan(X[train_idx]), med, X[train_idx])
    Xva = np.where(np.isnan(X[val_idx]),   med, X[val_idx])
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    Xva = sc.transform(Xva)
    lr = LogisticRegression(max_iter=200, n_jobs=-1, solver="lbfgs", C=1.0)
    lr.fit(Xtr, y[train_idx])
    return y[val_idx], lr.predict_proba(Xva)[:, 1]

y_va_all, p_all = first_fold_roc(X_all, y_all)
y_va_top, p_top = first_fold_roc(X_top, y_all)

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_va_all, p_all, name=f"All features ({len(all_features)})",
                                 ax=ax)
display_all = RocCurveDisplay.from_predictions(y_va_top, p_top, name=f"Top-30 features",
                                               ax=ax)
# Apply our palette to the line artists.
display_all.line_.set_color(PALETTE["mint"])
# Reskin the all-features line to mint too for consistency; use the existing line.
fig = ax.get_figure()
# Re-collect lines and assign colors.
for line, key in zip(ax.get_lines(), ["blue", "mint"]):
    line.set_color(PALETTE[key])
ax.set_title("Logistic Regression ROC — first fold")
ax.grid(linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "16_baseline_vs_reduced_roc.png")
plt.show()
print("Saved 16_baseline_vs_reduced_roc.png")
        """.strip()
    ))
    return cells


def section_6_recommendation() -> list[nbformat.NotebookNode]:
    cells = [md("## 6. Recommendation\n\n"
                "Which feature set to carry into Phase 4.")]
    cells.append(md(
        "**Recommendation (filled in below after running the notebook):** "
        "If the top-30 set is within ~0.01 AUC of the all-features set on 5-fold CV, "
        "**carry the top-30 forward** because Phase 4's XGBoost/MLP will train faster "
        "and the feature set is more interpretable. If the gap is larger, keep all "
        "141 engineered features."
    ))
    cells.append(code(
        """
recommendation_lines = []
recommendation_lines.append("# Phase 3 Recommendation\\n")
recommendation_lines.append(f"All-features mean AUC (5-fold CV):     {auc_all.mean():.4f} +- {auc_all.std():.4f}\\n")
recommendation_lines.append(f"Top-30 mean AUC (5-fold CV):          {auc_top.mean():.4f} +- {auc_top.std():.4f}\\n")
recommendation_lines.append(f"Delta (top30 - all):                  {delta:+.4f}\\n\\n")
if abs(delta) < 0.01:
    rec = ("**Carry the top-30 features forward to Phase 4.** The mean AUC delta is "
           f"small ({delta:+.4f}), and the top-30 set is much faster to train and "
           "more interpretable.")
elif delta > 0:
    rec = ("**Carry the top-30 features forward to Phase 4.** The top-30 set is "
           f"actually *better* by {delta:+.4f} AUC, so there's no reason to keep "
           "the full set.")
else:
    rec = ("**Carry all 141 engineered features forward to Phase 4.** The top-30 "
           f"set loses {abs(delta):.4f} AUC, which is too large a tradeoff for "
           "the speed/interpretability wins.")

recommendation_lines.append(rec + "\\n\\n")
recommendation_lines.append("See reports/feature_importance.md for the full ranking.\\n")

# Append the recommendation to the importance report (so reviewers see both
# in one file).
existing = IMPORTANCE_MD.read_text(encoding="utf-8")
IMPORTANCE_MD.write_text(existing + "\\n" + "\\n".join(recommendation_lines), encoding="utf-8")
print(f"Appended recommendation to {IMPORTANCE_MD}")
print()
print(rec)
        """.strip()
    ))
    return cells


# --- Assembly + execution ---------------------------------------------------

def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = (
        section_0_setup()
        + section_1_engineered_sanity()
        + section_2_importance()
        + section_3_shap()
        + section_4_combine()
        + section_5_baseline_vs_reduced()
        + section_6_recommendation()
    )
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    nb = build_notebook()
    print(f"Built notebook with {len(nb.cells)} cells.")

    placeholder = "__REPO_ROOT_PLACEHOLDER__"
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = c.source.replace(placeholder, str(REPO_ROOT))

    os.environ["CREDIT_RISK_REPO_ROOT"] = str(REPO_ROOT)

    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Wrote notebook skeleton to {NOTEBOOK_PATH}")

    print("Executing notebook...")
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    print(f"Execution done. {sum(1 for c in nb.cells if c.cell_type=='code')} code cells, "
          f"{sum(1 for c in nb.cells if c.cell_type=='markdown')} markdown cells.")

    errors = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for out in c.get("outputs", []):
            if out.output_type == "error":
                errors.append((i, out.ename, out.evalue))
    if errors:
        print(f"!! {len(errors)} cell(s) raised errors:")
        for i, name, msg in errors:
            print(f"   cell {i}: {name}: {msg}")
        sys.exit(1)

    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Saved executed notebook to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
