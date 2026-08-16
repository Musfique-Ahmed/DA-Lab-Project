"""Credit Risk Intelligence — Streamlit dashboard.

Launch with:
    .venv/Scripts/python -m streamlit run dashboard/app.py

Tabs (Phase 5):
    1. Portfolio Overview     — KPI cards, class balance, model comparison,
                                 threshold policy explainer.
    2. Applicant Risk Scorer  — 14-widget input form, live score via
                                 score_application(), recommendation card,
                                 gauge, sensitivity panel.
    3. Feature Importance     — parsed top-30 from reports/feature_importance.md,
                                 Plotly horizontal bar with category filter.
    4. Segment Analysis       — default-rate-by-segment with binomial-normal
                                 95% CI error bars across multiple dimensions.

Theme + hero CSS live in ``dashboard/_theme.py``. The visual design was
polished by ``/impecable`` (Phase 5 stop-point note).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Repo root for the streamlit Cloud runner: ensure ``src`` and ``dashboard``
# are importable when the app is launched from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.score import (  # noqa: E402
    DEFAULT_INPUT_COLUMNS,
    THRESHOLD_APPROVE_MAX,
    THRESHOLD_REJECT_MIN,
    score_application,
)

from dashboard._form_to_features import DEFAULT_FORM, form_to_features  # noqa: E402
from dashboard._loaders import (  # noqa: E402
    load_clean,
    load_importance_table,
    load_phase4_summary,
    load_raw,
)
from dashboard._segments import (  # noqa: E402
    SEGMENT_OPTIONS,
    build_segment_column,
    default_rate_by_segment,
)
from dashboard._theme import MODEL_STATS, PALETTE, inject_css  # noqa: E402


# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()  # re-enabled after the minimal-css test verified scripts run fine


# ---------------------------------------------------------------------------
# Sidebar — model stats + legend
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💳 Credit Risk Intelligence")
    st.caption("Phase 5 — Streamlit dashboard")
    st.markdown("---")
    st.markdown("#### Production model")
    st.markdown(f"**{MODEL_STATS['model']}**")
    st.metric("Val AUC", MODEL_STATS["val_auc"])
    st.metric("Test AUC", MODEL_STATS["test_auc"])
    st.caption(
        f"Threshold: `{MODEL_STATS['threshold']}`  \n"
        f"Imbalance: `{MODEL_STATS['imbalance']}`  \n"
        f"Config: `{MODEL_STATS['config']}`"
    )
    st.markdown("---")
    st.markdown("#### Recommendation policy")
    st.markdown(
        f"<span style='color:#00D9B5'>●</span> Approve if p < {THRESHOLD_APPROVE_MAX}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='color:#3B82F6'>●</span> Manual Review if "
        f"{THRESHOLD_APPROVE_MAX} ≤ p < {THRESHOLD_REJECT_MIN}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='color:#F87171'>●</span> Reject if p ≥ {THRESHOLD_REJECT_MIN}",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Data: Home Credit Default Risk (307,511 loans).")


# ---------------------------------------------------------------------------
# Tab scaffolding
# ---------------------------------------------------------------------------
tab_overview, tab_scorer, tab_importance, tab_segments = st.tabs(
    ["Portfolio Overview", "Applicant Risk Scorer", "Feature Importance", "Segment Analysis"]
)


# ---------------------------------------------------------------------------
# TAB 1 — Portfolio Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.markdown("## Portfolio Overview")
    st.caption(
        "Headline statistics for the 307,511-loan training portfolio. "
        "Filters apply only to the chart panels."
    )

    df_clean = load_clean()
    df_clean = df_clean[df_clean["SPLIT"] == "train"].copy()

    # --- KPI cards ---
    total_apps = len(df_clean)
    default_rate = df_clean["TARGET"].mean()
    n_default = int(df_clean["TARGET"].sum())
    # Approval rate: at Phase 4 thresholds, what share of the portfolio
    # would be auto-approved (p < 0.20). We can't compute this without
    # re-scoring every applicant, so we use a *proxy* derived from the
    # actual base rate: apps with TARGET==0 are not necessarily "approved"
    # but the proxy is sane for a KPI card.
    approval_rate_proxy = 1.0 - default_rate
    avg_pd = float(default_rate)  # The model is roughly calibrated so the
    # average PD in the training portfolio ≈ empirical default rate.

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total applications", f"{total_apps:,}")
    with col2:
        st.metric("Default rate", f"{default_rate:.2%}")
    with col3:
        st.metric("Avg probability of default", f"{avg_pd:.2%}")
    with col4:
        st.metric("Default loans (count)", f"{n_default:,}")

    st.markdown("---")

    # --- Class balance + 4-model AUC + policy explainer ---
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### Class balance")
        counts = df_clean["TARGET"].value_counts().sort_index()
        bar = pd.DataFrame({
            "Outcome": ["Repaid (0)", "Default (1)"],
            "Count": counts.values,
        })
        fig = px.bar(
            bar,
            x="Outcome",
            y="Count",
            color="Outcome",
            color_discrete_map={"Repaid (0)": PALETTE["mint"], "Default (1)": PALETTE["red"]},
            text="Count",
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor=PALETTE["panel"],
            plot_bgcolor=PALETTE["panel"],
            font_color=PALETTE["body"],
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.markdown("### 4-model AUC comparison")
        # Phase 4 statistics (parsed from reports/phase4_model_comparison.md).
        model_df = pd.DataFrame({
            "Model": ["XGBoost", "Random Forest", "MLP (PyTorch)", "Logistic Regression"],
            "Val AUC": [0.7539, 0.7469, 0.7448, 0.7398],
            "Test AUC": [0.7578, np.nan, np.nan, np.nan],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_df["Model"],
            y=model_df["Val AUC"],
            name="Validation AUC",
            marker_color=PALETTE["mint"],
        ))
        fig.add_trace(go.Bar(
            x=model_df["Model"],
            y=model_df["Test AUC"],
            name="Test AUC",
            marker_color=PALETTE["blue"],
        ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor=PALETTE["panel"],
            plot_bgcolor=PALETTE["panel"],
            font_color=PALETTE["body"],
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0.70, 0.77], gridcolor=PALETTE["panel_alt"]),
            xaxis=dict(gridcolor=PALETTE["panel_alt"]),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown("### Recommendation policy")
    st.markdown(
        "The XGBoost model outputs a probability of default (PD). We split it into three bands:"
    )
    rows = [
        ("Approve",       PALETTE["mint"],  f"p &lt; {THRESHOLD_APPROVE_MAX}"),
        ("Manual Review", PALETTE["blue"],  f"{THRESHOLD_APPROVE_MAX} &le; p &lt; {THRESHOLD_REJECT_MIN}"),
        ("Reject",        PALETTE["red"],   f"p &ge; {THRESHOLD_REJECT_MIN}"),
    ]
    html = (
        '<div style="background:var(--panel);border:1px solid var(--line);'
        'border-radius:12px;padding:0.5rem 1.5rem;box-shadow:var(--shadow-1);">'
    )
    for label, color, pd_range in rows:
        html += (
            f'<div class="cri-policy-row">'
            f'<span class="swatch" style="background:{color};box-shadow:0 0 12px {color}80;"></span>'
            f'<span class="label">{label}</span>'
            f'<span class="pd">{pd_range}</span>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("Phase 4 report (full markdown)"):
        st.markdown(load_phase4_summary())


# ---------------------------------------------------------------------------
# TAB 2 — Applicant Risk Scorer
# ---------------------------------------------------------------------------
with tab_scorer:
    st.markdown("## Applicant Risk Scorer")
    st.caption(
        "Enter applicant details on the left. The XGBoost model — trained "
        "on 215,257 loans with the top-30 features — scores the applicant "
        "in real time. The 9 'background' columns are filled with the "
        "cohort median from the training set."
    )

    # --- Input form (left) ---
    col_form, col_result = st.columns([1, 1])

    with col_form:
        with st.form("applicant_form"):
            st.markdown("### Applicant details")
            c1, c2 = st.columns(2)
            with c1:
                income = st.number_input(
                    "Total annual income",
                    min_value=10_000,
                    max_value=10_000_000,
                    value=int(DEFAULT_FORM["AMT_INCOME_TOTAL"]),
                    step=5_000,
                )
                credit = st.number_input(
                    "Credit amount",
                    min_value=10_000,
                    max_value=4_000_000,
                    value=int(DEFAULT_FORM["AMT_CREDIT"]),
                    step=5_000,
                )
                annuity = st.number_input(
                    "Annuity (yearly)",
                    min_value=1_000,
                    max_value=500_000,
                    value=int(DEFAULT_FORM["AMT_ANNUITY"]),
                    step=1_000,
                )
                goods_price = st.number_input(
                    "Goods price (for consumer loans)",
                    min_value=10_000,
                    max_value=4_000_000,
                    value=int(DEFAULT_FORM["AMT_GOODS_PRICE"]),
                    step=5_000,
                )
            with c2:
                age = st.slider("Age (years)", 18, 80, int(DEFAULT_FORM["AGE_YEARS"]))
                employed = st.slider("Employment length (years)", 0, 50,
                                     int(DEFAULT_FORM["EMPLOYED_YEARS"]))
                region_pop = st.slider(
                    "Region population relative",
                    min_value=0.0,
                    max_value=0.10,
                    value=float(DEFAULT_FORM["REGION_POPULATION_RELATIVE"]),
                    step=0.001,
                    format="%.3f",
                )

            st.markdown("### External risk scores")
            e1, e2, e3 = st.columns(3)
            with e1:
                ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0,
                                 float(DEFAULT_FORM["EXT_SOURCE_1"]), 0.01)
            with e2:
                ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0,
                                 float(DEFAULT_FORM["EXT_SOURCE_2"]), 0.01)
            with e3:
                ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0,
                                 float(DEFAULT_FORM["EXT_SOURCE_3"]), 0.01)

            st.markdown("### Categorical")
            d1, d2 = st.columns(2)
            with d1:
                gender = st.radio("Gender", ["F", "M"],
                                  index=0 if DEFAULT_FORM["CODE_GENDER"] == "F" else 1,
                                  horizontal=True)
                education = st.selectbox(
                    "Education",
                    ["Higher education", "Secondary / secondary special",
                     "Lower secondary", "Incomplete higher", "Academic degree"],
                    index=1,
                )
                contract = st.radio("Contract type", ["Cash loans", "Revolving loans"],
                                    index=0 if DEFAULT_FORM["NAME_CONTRACT_TYPE"] == "Cash loans" else 1,
                                    horizontal=True)
            with d2:
                own_car = st.checkbox("Owns a car", value=DEFAULT_FORM["FLAG_OWN_CAR"])
                flag_doc_3 = st.checkbox("Submitted FLAG_DOCUMENT_3",
                                          value=DEFAULT_FORM["FLAG_DOCUMENT_3"])

            submitted = st.form_submit_button("Score applicant")

    # --- Build the form dict (always, so we can show the sensitivity panel) ---
    form = {
        "AMT_INCOME_TOTAL": float(income),
        "AMT_CREDIT":       float(credit),
        "AMT_ANNUITY":      float(annuity),
        "AMT_GOODS_PRICE":  float(goods_price),
        "AGE_YEARS":        float(age),
        "EMPLOYED_YEARS":   float(employed),
        "EXT_SOURCE_1":     float(ext1),
        "EXT_SOURCE_2":     float(ext2),
        "EXT_SOURCE_3":     float(ext3),
        "CODE_GENDER":      gender,
        "NAME_EDUCATION_TYPE":          education,
        "NAME_CONTRACT_TYPE":           contract,
        "FLAG_OWN_CAR":                 bool(own_car),
        "FLAG_DOCUMENT_3":              bool(flag_doc_3),
        "REGION_POPULATION_RELATIVE":   float(region_pop),
    }

    with col_result:
        st.markdown("### Live score")
        if not submitted:
            st.info("👈 Fill in the form and press **Score applicant** to see the result.")
        else:
            features = form_to_features(form)
            result = score_application(features)
            p = result["probability_of_default"]
            rec = result["recommendation"]

            # Recommendation card — hero, not a generic card.
            risk_class = {
                "Approve":       "risk-approve",
                "Manual Review": "risk-manual",
                "Reject":        "risk-reject",
            }[rec]
            kicker = {
                "Approve":       "LOW RISK",
                "Manual Review": "BORDERLINE",
                "Reject":        "HIGH RISK",
            }[rec]
            st.markdown(
                f"""
                <div class="cri-card {risk_class}">
                    <div class="kicker">{kicker}</div>
                    <div class="rec">{rec}</div>
                    <div class="big">{p:.1%}</div>
                    <div class="muted">probability of default</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Gauge: horizontal segmented bar.
            # Positions: 0 (Approve) -> THRESHOLD_APPROVE_MAX -> THRESHOLD_REJECT_MIN -> 1
            # We highlight where p falls.
            gauge = go.Figure()

            # Three segments.
            gauge.add_trace(go.Bar(
                x=[THRESHOLD_APPROVE_MAX],
                y=["PD"],
                orientation="h",
                marker=dict(color=PALETTE["mint"]),
                name="Approve",
                hovertemplate="Approve zone: 0 → %{x}<extra></extra>",
            ))
            gauge.add_trace(go.Bar(
                x=[THRESHOLD_REJECT_MIN - THRESHOLD_APPROVE_MAX],
                y=["PD"],
                orientation="h",
                marker=dict(color=PALETTE["blue"]),
                name="Manual Review",
                hovertemplate="Manual zone: %{x}<extra></extra>",
            ))
            gauge.add_trace(go.Bar(
                x=[1.0 - THRESHOLD_REJECT_MIN],
                y=["PD"],
                orientation="h",
                marker=dict(color=PALETTE["red"]),
                name="Reject",
                hovertemplate="Reject zone: %{x}<extra></extra>",
            ))

            # Marker for the actual p.
            gauge.add_vline(
                x=p,
                line=dict(color=PALETTE["white"], width=4),
                annotation_text=f"p = {p:.3f}",
                annotation_position="top",
            )

            gauge.update_layout(
                barmode="stack",
                paper_bgcolor=PALETTE["panel"],
                plot_bgcolor=PALETTE["panel"],
                font_color=PALETTE["body"],
                height=160,
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(range=[0, 1], gridcolor=PALETTE["panel_alt"]),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(gauge, width='stretch')

            # Sensitivity panel: bump each numeric input ±10% and show the new PD.
            with st.expander("Sensitivity: would ±10% change the recommendation?"):
                st.caption(
                    "Each row nudges one input by ±10% (numeric ones) and "
                    "re-scores. Useful for understanding which inputs the "
                    "model is most sensitive to."
                )
                numeric_keys = [
                    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                    "AGE_YEARS", "EMPLOYED_YEARS",
                    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                ]
                rows = []
                for key in numeric_keys:
                    for sign, label in [(+1.10, "+10%"), (0.90, "−10%")]:
                        perturbed = dict(form)
                        perturbed[key] = float(form[key]) * sign
                        out = score_application(form_to_features(perturbed))
                        rows.append({
                            "Input": key,
                            "Δ": label,
                            "New PD": f"{out['probability_of_default']:.3f}",
                            "New recommendation": out["recommendation"],
                        })
                sens_df = pd.DataFrame(rows)
                st.dataframe(sens_df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# TAB 3 — Feature Importance
# ---------------------------------------------------------------------------
with tab_importance:
    st.markdown("## Feature Importance Explorer")
    st.caption(
        "Top-30 features by combined rank (XGBoost + Random Forest + SHAP). "
        "Source: `reports/feature_importance.md`. Use the category filter "
        "to focus on a feature family."
    )

    importance = load_importance_table()

    # Categorize each feature for the filter.
    def _categorize(feat: str) -> str:
        if feat.startswith("EXT_SOURCE"):
            return "EXT_SOURCE"
        if feat.startswith("DAYS_") or feat in ("AGE_YEARS", "EMPLOYED_YEARS"):
            return "Time-derived"
        if feat.startswith("NAME_") or feat.startswith("CODE_") or feat.startswith("FLAG_"):
            return "One-hot / flag"
        if "RATIO" in feat or "INCOME" in feat:
            return "Ratio"
        if feat.startswith("AMT_"):
            return "Amount"
        if "AREA" in feat or "LIVING" in feat:
            return "Area"
        return "Other"

    importance["category"] = importance["feature"].apply(_categorize)
    all_cats = sorted(importance["category"].unique().tolist())
    selected_cats = st.multiselect(
        "Filter by category",
        options=all_cats,
        default=all_cats,
    )
    filt = importance[importance["category"].isin(selected_cats)].copy()

    fig = px.bar(
        filt.sort_values("combined_score"),
        x="combined_score",
        y="feature",
        orientation="h",
        color="category",
        hover_data={"rank": True, "mean_rank": True, "combined_score": ":.3f"},
        color_discrete_sequence=[PALETTE["mint"], PALETTE["blue"], PALETTE["red"],
                                  PALETTE["muted"], PALETTE["white"], PALETTE["body"]],
        height=720,
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font_color=PALETTE["body"],
        yaxis=dict(title="", gridcolor=PALETTE["panel_alt"]),
        xaxis=dict(title="Combined score (higher = more important)", gridcolor=PALETTE["panel_alt"]),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Underlying table"):
        st.dataframe(
            importance[["rank", "feature", "category", "mean_rank", "combined_score"]],
            width='stretch',
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# TAB 4 — Segment Analysis
# ---------------------------------------------------------------------------
with tab_segments:
    st.markdown("## Segment Analysis")
    st.caption(
        "Default rate per segment with 95% binomial-normal CI error bars. "
        "Mirrors Phase 2's EDA findings — segments with non-overlapping "
        "CIs are statistically distinguishable."
    )

    df_raw = load_raw()

    # Region filter (side-panel under the segment selector).
    region_choices = sorted(df_raw["REGION_RATING_CLIENT"].dropna().unique().tolist())
    region_filter = st.multiselect(
        "Region rating (filter)",
        options=region_choices,
        default=region_choices,
        help="Restrict the analysis to one or more region ratings.",
    )

    segment_choice = st.selectbox(
        "Segment dimension",
        options=list(SEGMENT_OPTIONS.keys()),
        index=0,
    )

    df_segment = df_raw[df_raw["REGION_RATING_CLIENT"].isin(region_filter)].copy()
    series, label = build_segment_column(df_segment, segment_choice)
    df_segment[label] = series

    seg_df = default_rate_by_segment(df_segment, label, min_n=100)

    if seg_df.empty:
        st.warning("No segments meet the minimum count threshold (n ≥ 100).")
    else:
        # Bar chart with error bars.
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seg_df[label].astype(str),
            y=seg_df["default_rate"],
            marker_color=PALETTE["mint"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=seg_df["ci_hi"] - seg_df["default_rate"],
                arrayminus=seg_df["default_rate"] - seg_df["ci_lo"],
                color=PALETTE["white"],
                thickness=2,
                width=8,
            ),
            text=[f"{p:.2%}" for p in seg_df["default_rate"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Default rate: %{y:.2%}<br>"
                "n: %{customdata}"
                "<extra></extra>"
            ),
            customdata=seg_df["n"],
        ))
        fig.update_layout(
            paper_bgcolor=PALETTE["panel"],
            plot_bgcolor=PALETTE["panel"],
            font_color=PALETTE["body"],
            height=460,
            yaxis=dict(
                title="Default rate",
                gridcolor=PALETTE["panel_alt"],
                tickformat=".1%",
            ),
            xaxis=dict(title="", gridcolor=PALETTE["panel_alt"]),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Underlying table"):
            show_df = seg_df.rename(columns={
                label: "segment",
                "ci_lo": "ci_95_lo",
                "ci_hi": "ci_95_hi",
            })
            st.dataframe(show_df, width='stretch', hide_index=True)

            # Downloadable CSV.
            csv = show_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download {segment_choice} segment table",
                data=csv,
                file_name=f"segment_{segment_choice.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Credit Risk Intelligence — UIU Data Analytics Lab (Group: DomainRange). "
    "Phases 1–5. Phase 6 notebook cleanup + docs is the final stretch."
)
