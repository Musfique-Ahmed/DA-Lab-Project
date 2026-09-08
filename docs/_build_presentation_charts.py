"""Compact light-theme charts for the progress-update PDF.

Numbers are taken from reports/ and notebook stdout (not re-estimated).
Run: py docs/_build_presentation_charts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

OUT = Path(__file__).resolve().parent / "presentation_figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0A1428"
BLUE = "#2563EB"
MINT = "#0F766E"
RED = "#DC2626"
SLATE = "#475569"
MUTED = "#94A3B8"
PANEL = "#F8FAFC"
GRID = "#E2E8F0"
WIN = "#0F766E"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": NAVY,
    "axes.titlecolor": NAVY,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "text.color": NAVY,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, format="png", pil_kwargs={"optimize": True})
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"  {name:40s} {kb:6.1f} KB")


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.0, color=NAVY, fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center", va="bottom", color=color, fontsize=fontsize,
        )


# ---------------------------------------------------------------------------
# 1. Class imbalance + missingness (cleaning)
# ---------------------------------------------------------------------------
def fig_cleaning():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    vals = [91.93, 8.07]
    bars = ax.bar(["Repaid", "Default"], vals, color=[BLUE, RED], width=0.62)
    ax.set_ylabel("Share of loans (%)")
    ax.set_title("Class imbalance")
    ax.set_ylim(0, 110)
    ax.axhline(0, color=GRID)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v:.2f}%", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.22, "n = 307,511   ·   24,825 defaults",
            transform=ax.transAxes, ha="center", color=SLATE, fontsize=7.5)

    ax = axes[1]
    labels = [
        "COMMONAREA_*", "NONLIVINGAPARTMENTS_*", "FONDKAPREMONT_MODE",
        "LIVINGAPARTMENTS_*", "FLOORSMIN_*", "YEARS_BUILD_*",
        "OWN_CAR_AGE", "LANDAREA_*",
    ]
    pct = [69.87, 69.43, 68.39, 68.35, 67.85, 66.50, 65.99, 59.38]
    dropped = [p > 60 for p in pct]
    colors = [RED if d else BLUE for d in dropped]
    y = np.arange(len(labels))
    ax.barh(y, pct, color=colors, height=0.72)
    ax.axvline(60, color=NAVY, linestyle="--", linewidth=1, label="60% drop line")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Missing (%)")
    ax.set_title("Worst missingness — drop vs keep")
    ax.set_xlim(0, 85)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=RED, label="Dropped (>60%)"),
            Patch(facecolor=BLUE, label="Kept (LANDAREA 59%)"),
            plt.Line2D([0], [0], color=NAVY, linestyle="--", label="60% threshold"),
        ],
        fontsize=6.5, loc="lower right", frameon=False,
    )

    fig.tight_layout()
    save(fig, "01_imbalance_missingness.png")


# ---------------------------------------------------------------------------
# 2. CTI inverted-U (H2)
# ---------------------------------------------------------------------------
def fig_cti():
    rate = np.array([
        0.069085, 0.077615, 0.081126, 0.090212, 0.086542,
        0.091962, 0.086168, 0.079510, 0.074412, 0.070656,
    ])
    lo = np.array([
        0.066251, 0.074626, 0.078073, 0.087011, 0.083398,
        0.088732, 0.083031, 0.076486, 0.071482, 0.067789,
    ])
    hi = np.array([
        0.071920, 0.080604, 0.084179, 0.093413, 0.089686,
        0.095192, 0.089304, 0.082533, 0.077342, 0.073523,
    ])
    x = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.fill_between(x, lo * 100, hi * 100, color=BLUE, alpha=0.15, label="95% CI")
    ax.plot(x, rate * 100, "-o", color=MINT, linewidth=2, markersize=6, label="Default rate")
    ax.axhline(8.07, color=MUTED, linestyle=":", linewidth=1, label="Portfolio rate 8.07%")
    ax.scatter([6], [9.1962], s=70, color=RED, zorder=5)
    ax.annotate("peak 9.2%\ndecile 6", xy=(6, 9.20), xytext=(7.15, 9.55),
                fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.annotate("7.1% at the top", xy=(10, 7.07), xytext=(8.15, 6.45),
                fontsize=8, color=SLATE,
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8))
    ax.set_xticks(x)
    ax.set_xlabel("Credit-to-income decile (low → high leverage)")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("H2 rejected — default vs credit-to-income is an inverted U")
    ax.set_ylim(6.2, 10.2)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "02_cti_inverted_u.png")


# ---------------------------------------------------------------------------
# 3. H1 EXT_SOURCE + employment (supported signals)
# ---------------------------------------------------------------------------
def fig_h1_employment():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    ax = axes[0]
    names = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    r = np.array([-0.1553, -0.1605, -0.1789])
    bars = ax.barh(names, r, color=BLUE, height=0.55)
    ax.axvline(0, color=NAVY, linewidth=0.8)
    ax.set_xlabel("Pearson r with default")
    ax.set_title("H1 supported — bureau scores")
    ax.set_xlim(-0.25, 0.02)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, r):
        ax.text(v - 0.008, bar.get_y() + bar.get_height() / 2,
                f"r = {v:.3f}", va="center", ha="right", fontsize=8, color=NAVY)
    ax.text(0.02, -0.18, "all p ≪ 10⁻²⁰⁰", transform=ax.transAxes,
            fontsize=7.5, color=SLATE)

    ax = axes[1]
    labels = ["<1 yr", "1–3 yrs", "3–5 yrs", "5–10 yrs", "10+ yrs"]
    rate = np.array([10.9713, 11.0733, 9.6712, 7.3716, 5.1856])
    bars = ax.bar(labels, rate, color=MINT, width=0.65)
    ax.axhline(8.07, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Longer tenure, lower default")
    ax.set_ylim(0, 14)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    bar_labels(ax, bars, fmt="{:.1f}%", dy=0.25, fontsize=8)

    fig.tight_layout()
    save(fig, "03_ext_source_employment.png")


# ---------------------------------------------------------------------------
# 4. H3 education + occupation
# ---------------------------------------------------------------------------
def fig_h3():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    edu = [
        ("Lower secondary", 10.93),
        ("Secondary / spec.", 8.94),
        ("Incomplete higher", 8.48),
        ("Higher education", 5.36),
        ("Academic degree", 1.83),
    ]
    labels, vals = zip(*edu)
    colors = [RED] + [BLUE] * 3 + [MINT]
    bars = ax.barh(labels[::-1], list(vals)[::-1], color=colors[::-1], height=0.62)
    ax.set_xlabel("Default rate (%)")
    ax.set_title("Education is monotonic")
    ax.set_xlim(0, 14)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, list(vals)[::-1]):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8)

    ax = axes[1]
    occ = [
        ("Low-skill Laborers", 17.2),
        ("Drivers", 11.3),
        ("Waiters/barmen", 11.3),
        ("Security staff", 10.7),
        ("Laborers", 10.6),
        ("Cooking staff", 10.4),
        ("Sales staff", 9.6),
        ("Medicine staff", 6.7),
        ("Managers", 6.2),
        ("Accountants", 4.8),
    ]
    labels, vals = zip(*occ)
    y = np.arange(len(labels))
    cols = [RED if v >= 15 else (MINT if v <= 5 else BLUE) for v in vals]
    ax.barh(y, vals, color=cols, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Default rate (%)")
    ax.set_title("Occupation — 3.5× gap")
    ax.set_xlim(0, 22)
    ax.axvline(8.07, color=MUTED, linestyle=":", linewidth=1)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(17.4, 0, "17.2%", va="center", fontsize=7.5, color=RED, fontweight="bold")
    ax.text(5.1, 9, "4.8%", va="center", fontsize=7.5, color=MINT, fontweight="bold")

    fig.tight_layout()
    save(fig, "04_education_occupation.png")


# ---------------------------------------------------------------------------
# 5. Top-10 features
# ---------------------------------------------------------------------------
def fig_features():
    names = [
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_1",
        "EMPLOYED_YEARS",
        "EXT_SOURCE_3",
        "CODE_GENDER_M",
        "AMT_CREDIT",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
        "EXT_SOURCE_2",
        "Higher education",
    ]
    score = [1.000, 0.957, 0.943, 0.934, 0.932, 0.918, 0.916, 0.909, 0.905, 0.902]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    y = np.arange(len(names))
    colors = [MINT] + [BLUE] * 9
    ax.barh(y, score, color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.88, 1.02)
    ax.set_xlabel("Combined importance score (mean of XGB / RF / SHAP ranks)")
    ax.set_title("Top 10 of the top-30 feature set")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for i, s in enumerate(score):
        ax.text(s + 0.002, i, f"{s:.3f}", va="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "05_top10_features.png")


# ---------------------------------------------------------------------------
# 6. Model AUC + SMOTE A/B
# ---------------------------------------------------------------------------
def fig_models_auc():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.45))

    ax = axes[0]
    models = ["XGBoost", "Random\nForest", "MLP", "LogReg"]
    auc = [0.7539, 0.7469, 0.7448, 0.7398]
    colors = [WIN, BLUE, BLUE, SLATE]
    bars = ax.bar(models, auc, color=colors, width=0.62)
    ax.set_ylabel("Validation AUC-ROC")
    ax.set_title("Four-model bake-off")
    ax.set_ylim(0.730, 0.762)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, auc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0008,
                f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(0, 0.755, "winner", ha="center", fontsize=7, color=WIN)

    ax = axes[1]
    fam = ["LogReg", "XGBoost"]
    w = [0.7398, 0.7539]
    s = [0.7394, 0.7203]
    x = np.arange(len(fam))
    wbar = ax.bar(x - 0.18, w, 0.36, color=MINT, label="class_weight / scale_pos_weight")
    sbar = ax.bar(x + 0.18, s, 0.36, color=RED, label="SMOTE (k=5)")
    ax.set_xticks(x)
    ax.set_xticklabels(fam)
    ax.set_ylabel("Validation AUC")
    ax.set_title("SMOTE lost — we ran it, not just cited it")
    ax.set_ylim(0.70, 0.77)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    for bar, v in zip(wbar, w):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=7.5)
    for bar, v in zip(sbar, s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=7.5, color=RED)

    fig.tight_layout()
    save(fig, "06_auc_and_smote.png")


# ---------------------------------------------------------------------------
# 7. Precision/recall + XGB confusion
# ---------------------------------------------------------------------------
def fig_operating_point():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    models = ["XGBoost", "RF", "MLP", "LogReg"]
    prec = [0.164, 0.237, 0.153, 0.156]
    rec = [0.675, 0.394, 0.688, 0.660]
    x = np.arange(len(models))
    ax.bar(x - 0.18, prec, 0.36, color=SLATE, label="Precision")
    ax.bar(x + 0.18, rec, 0.36, color=MINT, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score at threshold 0.50")
    ax.set_title("Why not ship RF? It misses 60% of defaults")
    ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    ax = axes[1]
    # TN FP / FN TP
    cm = np.array([[29615, 12788], [1212, 2512]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred repaid", "Pred default"])
    ax.set_yticks([0, 1], ["True repaid", "True default"])
    ax.set_title("XGBoost confusion (val, t=0.50)")
    labels = [["TN 29,615", "FP 12,788"], ["FN 1,212", "TP 2,512"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 18000 else NAVY
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")
    ax.set_xlabel("Recall 0.675  ·  test AUC 0.7578")

    fig.tight_layout()
    save(fig, "07_recall_confusion.png")


# ---------------------------------------------------------------------------
# 8. Three-way policy (compact)
# ---------------------------------------------------------------------------
def fig_policy():
    fig, ax = plt.subplots(figsize=(7.6, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Shipped policy — score_application()", loc="left", pad=8)

    bands = [
        (0.00, 0.20, MINT, "Approve", "p < 0.20"),
        (0.20, 0.50, "#D97706", "Manual Review", "0.20 ≤ p < 0.50"),
        (0.50, 1.00, RED, "Reject", "p ≥ 0.50"),
    ]
    for x0, x1, color, name, rule in bands:
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.01, 0.38), x1 - x0 - 0.02, 0.38,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color, edgecolor="none", alpha=0.9,
        ))
        ax.text((x0 + x1) / 2, 0.57, name, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text((x0 + x1) / 2, 0.22, rule, ha="center", fontsize=8, color=SLATE)

    demos = [(0.0073, "low-risk 0.007"), (0.3865, "mid 0.39"), (0.9548, "high 0.95")]
    for p, lab in demos:
        ax.plot([p], [0.85], "v", color=NAVY, markersize=8)
        ax.text(p, 0.96, lab, ha="center", fontsize=7, color=NAVY)

    fig.tight_layout()
    save(fig, "08_policy.png")


def main():
    print("Writing charts to", OUT)
    fig_cleaning()
    fig_cti()
    fig_h1_employment()
    fig_h3()
    fig_features()
    fig_models_auc()
    fig_operating_point()
    fig_policy()
    total = sum(p.stat().st_size for p in OUT.glob("*.png")) / 1024
    print(f"Total PNG size: {total:.0f} KB")


if __name__ == "__main__":
    main()
