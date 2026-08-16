"""Dashboard theme — palette + CSS injection.

Single source of truth for the hex colors used across the Streamlit app.
The CSS overrides Streamlit's default light theme so the visible chrome
matches the dark navy / mint palette used in the project's slide deck.

The palette is exactly the one defined in the master prompt:

  - Background:      #0A1428 (deep navy)
  - Card panels:     #10203D / #162A4D
  - Primary accent:  #00D9B5 (mint/teal)
  - Supporting:      #3B82F6 (blue)
  - Risk accent:     #F87171 (soft red) — reserved for default-risk
  - Text:            #FFFFFF (headers), #CBD5E1 (body), #94A3B8 (muted)

Design world (refinement, not redesign — preserves the pinned brief):
  - Dark, premium fintech aesthetic.
  - Soft, multi-layer shadows carry the depth — not colored stripes.
  - Type carries hierarchy; color carries state (Approve / Manual / Reject).
  - Browser surfaces (selection, focus, caret, scrollbar) themed to match.

`inject_css()` is idempotent — calling it more than once is safe.
"""
from __future__ import annotations

import streamlit as st

# Master prompt palette. Do not edit hex values without re-running /impeccable.
PALETTE: dict[str, str] = {
    "bg":          "#0A1428",
    "panel":       "#10203D",
    "panel_alt":   "#162A4D",
    "panel_deep":  "#0B1830",
    "mint":        "#00D9B5",
    "mint_soft":   "#3DE6C9",
    "blue":        "#3B82F6",
    "red":         "#F87171",
    "white":       "#FFFFFF",
    "body":        "#CBD5E1",
    "muted":       "#94A3B8",
    "line":        "rgba(255, 255, 255, 0.06)",
    "line_strong": "rgba(255, 255, 255, 0.12)",
}

#: Phase 4 model stats surfaced in the sidebar (parsed from phase4_model_comparison.md).
MODEL_STATS: dict[str, str] = {
    "model":      "XGBoost",
    "val_auc":    "0.7539",
    "test_auc":   "0.7578",
    "threshold":  "0.500",
    "imbalance":  "balanced (scale_pos_weight)",
    "config":     "n_estimators=300, max_depth=4, lr=0.05",
}


# Single CSS block. Targets Streamlit's data-testid attributes that drive
# the visible chrome. Kept in one place so a /impeccable pass can tweak it
# end-to-end without hunting across modules.
_CSS = """
<style>
/* @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');  -- removed: blocked CDNs broke hydration on 1.49 */

/* ===== Tokens ===== */
:root {
    --bg:        #0A1428;
    --panel:     #10203D;
    --panel-alt: #162A4D;
    --panel-deep:#0B1830;
    --mint:      #00D9B5;
    --mint-soft: #3DE6C9;
    --blue:      #3B82F6;
    --red:       #F87171;
    --white:     #FFFFFF;
    --body:      #CBD5E1;
    --muted:     #94A3B8;
    --line:      rgba(255, 255, 255, 0.06);
    --line-2:    rgba(255, 255, 255, 0.12);
    --shadow-1:  0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 8px 24px rgba(0, 0, 0, 0.35);
    --shadow-2:  0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 18px 48px rgba(0, 0, 0, 0.55);
}

/* ===== Base ===== */
.stApp {
    background:
        radial-gradient(1200px 600px at 85% -10%, rgba(0, 217, 181, 0.07), transparent 60%),
        radial-gradient(900px 500px at -10% 110%, rgba(59, 130, 246, 0.06), transparent 60%),
        var(--bg);
    color: var(--body);
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    font-feature-settings: 'cv02', 'cv11';
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}
header[data-testid="stHeader"] {
    background: rgba(10, 20, 40, 0.7);
    backdrop-filter: blur(10px);
}
body, .stMarkdown, .stText, p, label, span, div {
    color: var(--body);
}
h1, h2, h3, h4 {
    color: var(--white);
    letter-spacing: -0.02em;
    font-weight: 700;
}
h1 { font-size: 2rem; line-height: 1.15; }
h2 { font-size: 1.5rem; line-height: 1.2; }
h3 { font-size: 1.125rem; line-height: 1.25; }

/* ===== Browser surfaces ===== */
::selection { background: rgba(0, 217, 181, 0.30); color: var(--white); }
/* * { caret-color: var(--mint); }   -- removed: universal selector broke hydration on Streamlit 1.49 */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--panel-deep); }
::-webkit-scrollbar-thumb { background: var(--panel-alt); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #1F345F; }
/*
*:focus-visible {
    outline: 2px solid var(--mint);
    outline-offset: 2px;
    border-radius: 4px;
}
*/

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-deep) 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--body); }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown strong {
    color: var(--white);
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--mint);
    font-family: 'JetBrains Mono', monospace;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0;
    border-bottom: 1px solid var(--line-2);
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.85rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: -1px;
    transition: color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--body); }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--mint);
    border-bottom: 2px solid var(--mint);
}

/* ===== Metric / KPI cards ===== */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-deep) 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.1rem 1.35rem;
    box-shadow: var(--shadow-1);
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--mint), transparent);
    opacity: 0.35;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: var(--white) !important;
    font-weight: 700;
    font-size: 1.75rem !important;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
    padding-top: 0.2rem;
}
[data-testid="stMetricDelta"] {
    color: var(--muted) !important;
}

/* ===== Expanders ===== */
div[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    box-shadow: var(--shadow-1);
}
div[data-testid="stExpander"] summary {
    color: var(--body) !important;
    font-weight: 600;
}

/* ===== Buttons ===== */
.stButton > button {
    background: var(--mint);
    color: var(--panel-deep);
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.6rem;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 14px rgba(0, 217, 181, 0.20);
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stButton > button:hover {
    background: var(--mint-soft);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0, 217, 181, 0.28);
}
.stButton > button:active { transform: translateY(0); }
.stDownloadButton > button {
    background: transparent;
    color: var(--mint);
    border: 1px solid var(--mint);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-weight: 600;
}

/* ===== Inputs ===== */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--panel-deep);
    color: var(--white);
    border: 1px solid var(--line-2);
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--mint);
    box-shadow: 0 0 0 3px rgba(0, 217, 181, 0.18);
    outline: none;
}
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: var(--panel-deep);
    border: 1px solid var(--line-2);
    border-radius: 8px;
    transition: border-color 0.15s ease;
}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stMultiSelect [data-baseweb="select"] > div:focus-within {
    border-color: var(--mint);
    box-shadow: 0 0 0 3px rgba(0, 217, 181, 0.18);
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--mint);
    box-shadow: 0 0 0 4px rgba(0, 217, 181, 0.18);
    border: 2px solid var(--panel-deep);
}
.stRadio label, .stCheckbox label { color: var(--body); }

/* ===== Tables / dataframes ===== */
.stDataFrame {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    box-shadow: var(--shadow-1);
    overflow: hidden;
}

/* ===== Recommendation hero card ===== */
.cri-card {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-deep) 100%);
    border: 1px solid var(--line-2);
    border-radius: 18px;
    padding: 2rem 2rem 1.75rem 2rem;
    margin: 0.5rem 0 1.25rem 0;
    box-shadow: var(--shadow-2);
    text-align: center;
    position: relative;
    overflow: hidden;
}
.cri-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    opacity: 0.85;
}
.cri-card.risk-approve::before { background: linear-gradient(90deg, transparent, var(--mint), transparent); }
.cri-card.risk-manual::before  { background: linear-gradient(90deg, transparent, var(--blue), transparent); }
.cri-card.risk-reject::before  { background: linear-gradient(90deg, transparent, var(--red), transparent); }
.cri-card .kicker {
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.cri-card .rec {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 1.25rem;
    color: var(--white);
}
.cri-card.risk-approve .rec { color: var(--mint); }
.cri-card.risk-manual  .rec { color: var(--blue); }
.cri-card.risk-reject  .rec { color: var(--red); }
.cri-card .big {
    color: var(--white);
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.04em;
    margin: 0.25rem 0 0.5rem 0;
}
.cri-card.risk-approve .big { color: var(--mint); }
.cri-card.risk-reject  .big { color: var(--red); }
.cri-card.risk-manual  .big { color: var(--blue); }
.cri-card .muted { color: var(--muted); font-size: 0.95rem; }

/* ===== Policy explainer ===== */
.cri-policy-row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--line);
}
.cri-policy-row:last-child { border-bottom: none; }
.cri-policy-row .swatch {
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0;
}
.cri-policy-row .label {
    color: var(--white);
    font-weight: 600;
    width: 9rem;
    flex-shrink: 0;
}
.cri-policy-row .pd {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
}

/* ===== Misc ===== */
hr { border-color: var(--line); margin: 1.5rem 0; }
.stAlert { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
.stCaption, .stMarkdown small { color: var(--muted) !important; }
</style>
"""


def inject_css() -> None:
    """Inject the dashboard CSS into the current Streamlit page.

    Idempotent — Streamlit de-duplicates identical <style> blocks per rerun.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


__all__ = ["PALETTE", "MODEL_STATS", "inject_css"]
