"""
OpenFinGuard — Streamlit Dashboard
Bold fintech aesthetic: obsidian black + electric cyan + vivid signal accents
All 5 pages: Credit Assessment · Scenario Explorer · Model Dashboard · Fairness · About
"""

import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# BEFORE (hardcoded — breaks inside Docker)
API_URL = "http://localhost:8000"

# AFTER (reads from environment, falls back to localhost for local dev)
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Design Tokens ─────────────────────────────────────────────────────────────
CYAN   = "#00E5FF"
AMBER  = "#FFB300"
RED    = "#FF3D57"
GREEN  = "#00E676"
NAVY   = "#060B18"
CARD   = "#0D1424"
CARD2  = "#111B2E"
BORDER = "#1C2C44"
TEXT   = "#CBD5E1"
MUTED  = "#4A6080"
WHITE  = "#F1F5F9"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif;
    background-color: {NAVY};
    color: {TEXT};
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: #04080F !important;
    border-right: 1px solid {BORDER};
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 1rem;
}}

/* ── Global ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem !important; padding-bottom: 2rem !important; }}

/* ── Page Header ── */
.page-header {{
    background: linear-gradient(135deg, {CARD} 0%, #0A1628 100%);
    border: 1px solid {BORDER};
    border-left: 4px solid {CYAN};
    border-radius: 10px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
.page-header::before {{
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,229,255,0.04) 0%, transparent 70%);
    pointer-events: none;
}}
.page-title {{
    font-size: 1.6rem;
    font-weight: 800;
    color: {WHITE};
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}}
.page-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {MUTED};
    margin: 0.3rem 0 0 0;
    letter-spacing: 0.5px;
}}

/* ── Section Labels ── */
.sec-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {CYAN};
    margin: 1.2rem 0 0.6rem 0;
    display: block;
}}

/* ── Decision Badges ── */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 1.2rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 1.5px;
    font-family: 'IBM Plex Mono', monospace;
}}
.badge-APPROVE {{ background: rgba(0,230,118,0.15); border: 1px solid rgba(0,230,118,0.4); color: {GREEN}; }}
.badge-DECLINE {{ background: rgba(255,61,87,0.15); border: 1px solid rgba(255,61,87,0.4); color: {RED}; }}
.badge-REVIEW  {{ background: rgba(255,179,0,0.12); border: 1px solid rgba(255,179,0,0.35); color: {AMBER}; }}

/* ── Cards ── */
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}}
.kpi-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: {CYAN};
    display: block;
}}
.kpi-lbl {{
    font-size: 0.65rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 1.5px;
}}

/* ── Factor Pills ── */
.factor-risk {{
    background: rgba(255,61,87,0.08);
    border: 1px solid rgba(255,61,87,0.25);
    border-left: 3px solid {RED};
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    margin: 0.25rem 0;
    font-size: 0.82rem;
    color: #FCA5A5;
    font-family: 'IBM Plex Mono', monospace;
}}
.factor-protect {{
    background: rgba(0,230,118,0.06);
    border: 1px solid rgba(0,230,118,0.2);
    border-left: 3px solid {GREEN};
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    margin: 0.25rem 0;
    font-size: 0.82rem;
    color: #86EFAC;
    font-family: 'IBM Plex Mono', monospace;
}}

/* ── Inputs ── */
[data-testid="metric-container"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
}}
[data-testid="metric-container"] label {{
    color: {MUTED} !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'IBM Plex Mono', monospace !important;
}}
[data-testid="stMetricValue"] {{
    color: {CYAN} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}}

/* ── Buttons ── */
.stButton > button, .stFormSubmitButton > button {{
    background: linear-gradient(135deg, #007A8A 0%, {CYAN} 100%) !important;
    color: #001820 !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    border-radius: 6px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    font-size: 0.95rem !important;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,229,255,0.25) !important;
}}

/* ── Nav radio ── */
div[data-testid="stRadio"] > label {{ display: none; }}
div[data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 0.2rem;
    display: flex;
    flex-direction: column;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{
    background: transparent !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.8rem !important;
    transition: background 0.15s !important;
    font-size: 0.88rem !important;
    color: {MUTED} !important;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
    background: rgba(0,229,255,0.08) !important;
    color: {CYAN} !important;
}}

/* ── Expander ── */
details {{ background: {CARD2} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; }}
summary {{ color: {TEXT} !important; font-size: 0.88rem !important; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {CYAN} !important; }}

/* ── Table ── */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; margin: 1rem 0 !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {NAVY}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_post(endpoint: str, payload: dict, timeout: int = 10):
    try:
        r = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot reach API. Start it with: `cd api && python main.py`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_get(endpoint: str, timeout: int = 5):
    try:
        r = requests.get(f"{API_URL}/{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def decision_color(d: str) -> str:
    return {
        "APPROVE": GREEN,
        "DECLINE": RED,
        "REVIEW":  AMBER,
    }.get(d, MUTED)


def gauge(prob: float, height: int = 240, show_title: bool = True) -> go.Figure:
    color = GREEN if prob < 0.15 else AMBER if prob < 0.40 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color, "family": "IBM Plex Mono"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "ticksuffix": "%",
                "tickcolor": MUTED,
                "tickfont": {"color": MUTED, "size": 9},
                "tickwidth": 1,
            },
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  15],  "color": "rgba(0,230,118,0.10)"},
                {"range": [15, 40],  "color": "rgba(255,179,0,0.10)"},
                {"range": [40, 100], "color": "rgba(255,61,87,0.10)"},
            ],
            "threshold": {
                "line": {"color": "rgba(255,255,255,0.6)", "width": 2},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
        title={
            "text": "DEFAULT PROBABILITY" if show_title else "",
            "font": {"size": 9, "color": MUTED, "family": "IBM Plex Mono"},
        },
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=45, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT,
    )
    return fig


def mini_gauge(prob: float, label: str) -> go.Figure:
    color = GREEN if prob < 0.15 else AMBER if prob < 0.40 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 26, "color": color, "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 15],   "color": "rgba(0,230,118,0.08)"},
                {"range": [15, 40],  "color": "rgba(255,179,0,0.08)"},
                {"range": [40, 100], "color": "rgba(255,61,87,0.08)"},
            ],
        },
        title={"text": label, "font": {"size": 9, "color": MUTED, "family": "IBM Plex Mono"}},
    ))
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_bar(risk_factors: list, protect_factors: list) -> go.Figure | None:
    all_f = risk_factors + protect_factors
    if not all_f:
        return None
    names   = [f["description"] for f in all_f]
    impacts = [f["shap_impact"] for f in all_f]
    colors  = [RED if i > 0 else GREEN for i in impacts]
    pairs   = sorted(zip(impacts, names, colors), key=lambda x: x[0])
    impacts, names, colors = zip(*pairs)

    fig = go.Figure(go.Bar(
        x=list(impacts), y=list(names), orientation="h",
        marker=dict(color=list(colors), opacity=0.8,
                    line=dict(color="rgba(255,255,255,0.08)", width=0.5)),
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(
        title=dict(text="SHAP FEATURE IMPACT", font=dict(size=9, color=MUTED, family="IBM Plex Mono")),
        xaxis=dict(title="Impact on Default Probability",
                   color=MUTED, gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(color=TEXT, gridcolor="rgba(0,0,0,0)"),
        height=270,
        margin=dict(l=10, r=10, t=35, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT,
    )
    return fig


def plotly_defaults() -> dict:
    return {
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font_color":    TEXT,
        "margin":        dict(l=10, r=10, t=35, b=10),
    }


# ── API Status ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def get_health():
    return api_get("health", timeout=2)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1.2rem 0;">
        <div style="font-size:1.5rem; font-weight:800; color:{WHITE}; letter-spacing:-1px; line-height:1;">
            Open<span style="color:{CYAN};">Fin</span>Guard
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:{MUTED}; margin-top:3px;">
            AI Credit Intelligence · v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("nav", [
        "🎯  Credit Assessment",
        "🔬  Scenario Explorer",
        "📊  Model Dashboard",
        "⚖️  Fairness Audit",
        "🗄️  Predictions DB",
        "📖  About",
    ],key="main_nav")

    st.divider()

    health = get_health()
    if health:
        db_ok   = health.get("database_online", False)
        shap_ok = health.get("shap_available", False)
        st.markdown(f"""
        <div style="background:rgba(0,230,118,0.07); border:1px solid rgba(0,230,118,0.25);
                    border-radius:8px; padding:0.7rem 0.9rem; font-family:'IBM Plex Mono',monospace;">
            <div style="color:{GREEN}; font-size:0.78rem; font-weight:600; margin-bottom:4px;">● API ONLINE</div>
            <div style="color:{MUTED}; font-size:0.65rem; line-height:1.6;">
                DB: <span style="color:{'#00E676' if db_ok else RED};">{'✓' if db_ok else '✗'}</span> &nbsp;
                SHAP: <span style="color:{'#00E676' if shap_ok else AMBER};">{'✓' if shap_ok else '~'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(255,61,87,0.07); border:1px solid rgba(255,61,87,0.25);
                    border-radius:8px; padding:0.7rem 0.9rem; font-family:'IBM Plex Mono',monospace;">
            <div style="color:{RED}; font-size:0.78rem; font-weight:600;">● API OFFLINE</div>
            <div style="color:{MUTED}; font-size:0.65rem; margin-top:3px;">python api/main.py</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Credit Assessment
# ══════════════════════════════════════════════════════════════════════════════
if "Credit Assessment" in page:

    st.markdown(f"""
    <div class="page-header">
        <p class="page-title">🎯 Credit Risk Assessment</p>
        <p class="page-sub">Real-time AI scoring · SHAP explainability · ECOA compliant · PostgreSQL audit trail</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_out = st.columns([1, 1.25], gap="large")

    with col_form:
        with st.form("credit_form"):
            st.markdown('<span class="sec-label">💰 Financial Profile</span>', unsafe_allow_html=True)
            monthly_income = st.number_input(
                "Monthly Income ($)", min_value=0.0, max_value=500000.0,
                value=5000.0, step=500.0, help="Gross monthly income before taxes"
            )
            debt_ratio = st.slider(
                "Debt-to-Income Ratio", 0.0, 2.0, 0.35, 0.01,
                help="Total monthly debt / gross income. Above 0.43 is high risk (FHA threshold)"
            )
            revolving_util = st.slider(
                "Credit Card Utilization %", 0, 100, 35, 1,
                help="% of revolving credit in use. Under 30% is ideal."
            )

            st.markdown('<span class="sec-label">👤 Personal</span>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            age        = c1.number_input("Age", 18, 100, 45)
            dependents = c2.number_input("Dependents", 0, 20, 1)

            st.markdown('<span class="sec-label">🏦 Credit Lines</span>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            open_lines = c3.number_input("Open Credit Lines", 0, 50, 8)
            re_loans   = c4.number_input("Real Estate Loans", 0, 20, 1)

            st.markdown('<span class="sec-label">⚠️ Payment History (2 years)</span>', unsafe_allow_html=True)
            c5, c6, c7 = st.columns(3)
            p30 = c5.number_input("30–59d late", 0, 20, 0)
            p60 = c6.number_input("60–89d late", 0, 20, 0)
            p90 = c7.number_input("90d+ late",   0, 20, 0)

            explain   = st.checkbox("Include SHAP explanations", value=True)
            submitted = st.form_submit_button("🔍  Assess Credit Risk", use_container_width=True)

    with col_out:
        st.markdown('<span class="sec-label">📊 Decision Output</span>', unsafe_allow_html=True)

        if submitted:
            payload = {
                "revolving_utilization": revolving_util / 100,
                "age":                   int(age),
                "past_due_30_59_days":   int(p30),
                "debt_ratio":            float(debt_ratio),
                "monthly_income":        float(monthly_income),
                "open_credit_lines":     int(open_lines),
                "past_due_90_days":      int(p90),
                "real_estate_loans":     int(re_loans),
                "past_due_60_89_days":   int(p60),
                "dependents":            int(dependents),
                "explain":               explain,
            }

            with st.spinner("Scoring credit profile..."):
                result = api_post("predict", payload)

            if result:
                prob     = result["default_probability"]
                decision = result["decision"]
                risk_band = result["risk_band"]
                dc       = decision_color(decision)
                icon     = {"APPROVE": "✅", "DECLINE": "❌", "REVIEW": "⚠️"}.get(decision, "ℹ️")

                # Decision badge + band
                st.markdown(
                    f'<div style="margin-bottom:0.6rem; display:flex; align-items:center; gap:0.8rem;">'
                    f'<span class="badge badge-{decision}">{icon} {decision}</span>'
                    f'<span style="color:{MUTED}; font-size:0.85rem;">{risk_band}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"→ {result['recommended_action']}")

                # Gauge
                st.plotly_chart(gauge(prob), use_container_width=True, config={"displayModeBar": False})

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Default Prob",  f"{prob:.1%}")
                k2.metric("Confidence",    result["confidence"])
                k3.metric("Latency",       f"{result['inference_time_ms']:.1f}ms")
                k4.metric("Persisted",     "✓ DB" if result.get("persisted") else "✗")

                # SHAP factors
                rf = result.get("primary_risk_factors", [])
                pf = result.get("protective_factors", [])

                if rf:
                    st.markdown('<span class="sec-label">⬆ Risk Factors</span>', unsafe_allow_html=True)
                    for f in rf:
                        st.markdown(f'<div class="factor-risk">🔴 {f["human_readable"]}</div>',
                                    unsafe_allow_html=True)
                if pf:
                    st.markdown('<span class="sec-label">⬇ Protective Factors</span>', unsafe_allow_html=True)
                    for f in pf:
                        st.markdown(f'<div class="factor-protect">🟢 {f["human_readable"]}</div>',
                                    unsafe_allow_html=True)

                fig = shap_bar(rf, pf)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Audit trail
                st.markdown(f"""
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:{MUTED};
                            background:{CARD}; border:1px solid {BORDER}; border-radius:6px;
                            padding:0.5rem 0.8rem; margin-top:0.8rem; line-height:1.8;">
                    <span style="color:{BORDER};">REQUEST:</span> {result['request_id']}<br>
                    <span style="color:{BORDER};">MODEL:</span> {result['model_name']} v{result['model_version']}
                    &nbsp;|&nbsp; THRESHOLD: {result['threshold_used']:.3f}
                </div>
                """, unsafe_allow_html=True)

        else:
            # Placeholder
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
                        padding:2rem; margin-top:0.5rem;">
                <p style="color:{MUTED}; font-size:0.7rem; text-transform:uppercase;
                           letter-spacing:2px; font-family:'IBM Plex Mono',monospace;">How it works</p>
                <p style="color:{TEXT}; font-size:0.9rem; line-height:1.7;">
                    Fill in the application form and click
                    <strong style="color:{CYAN};">Assess Credit Risk</strong>.
                    The model will return an instant decision with SHAP explanations
                    and persist the result to PostgreSQL.
                </p>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:1.2rem;">
                    {"".join([
                        f'<div style="background:{CARD2}; border:1px solid {BORDER}; border-radius:6px; padding:0.7rem 0.9rem;">'
                        f'<div style="color:{CYAN}; font-size:0.65rem; text-transform:uppercase; letter-spacing:1px; font-family:IBM Plex Mono,monospace;">{k}</div>'
                        f'<div style="color:{TEXT}; font-size:0.85rem; margin-top:2px;">{v}</div></div>'
                        for k, v in [
                            ("Model", "GradientBoosting"),
                            ("AUC-ROC", "~0.86+"),
                            ("Training Data", "150,000 borrowers"),
                            ("Explainability", "SHAP · ECOA"),
                        ]
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Scenario Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif "Scenario Explorer" in page:

    st.markdown(f"""
    <div class="page-header">
        <p class="page-title">🔬 Scenario Explorer</p>
        <p class="page-sub">Live risk sensitivity analysis · Compare 3 profiles side-by-side · Feature sweep</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sec-label">⚙️ Base Profile</span>', unsafe_allow_html=True)
    bc1, bc2, bc3 = st.columns(3)
    base_income = bc1.slider("Monthly Income ($)", 500, 50000, 5000, 500)
    base_age    = bc2.slider("Age", 18, 85, 40, 1)
    base_util   = bc3.slider("Credit Utilization %", 0, 100, 30, 1)

    bc4, bc5, bc6 = st.columns(3)
    base_debt   = bc4.slider("Debt Ratio", 0.0, 2.0, 0.35, 0.01)
    base_30     = bc5.slider("30-59d Late", 0, 10, 0)
    base_90     = bc6.slider("90d+ Late", 0, 10, 0)

    base_payload = {
        "revolving_utilization": base_util / 100,
        "age":                   base_age,
        "past_due_30_59_days":   base_30,
        "debt_ratio":            base_debt,
        "monthly_income":        base_income,
        "open_credit_lines":     8,
        "past_due_90_days":      base_90,
        "real_estate_loans":     1,
        "past_due_60_89_days":   0,
        "dependents":            1,
        "explain":               False,
    }

    st.divider()
    st.markdown('<span class="sec-label">📊 Three-Scenario Comparison</span>', unsafe_allow_html=True)

    scenarios = [
        {
            "label": "Base Profile",
            "desc":  "Your current settings",
            "delta": {},
        },
        {
            "label": "High Utilization",
            "desc":  f"Utilization raised to 90% (from {base_util}%)",
            "delta": {"revolving_utilization": 0.90},
        },
        {
            "label": "Payment Problems",
            "desc":  "3× 30-59d late, 2× 90d+, debt ratio +0.3",
            "delta": {
                "past_due_30_59_days": 3,
                "past_due_90_days":    2,
                "debt_ratio":          min(base_debt + 0.3, 2.0),
            },
        },
    ]

    cols = st.columns(3)
    for col, sc in zip(cols, scenarios):
        p = {**base_payload, **sc["delta"]}
        with col:
            try:
                r = api_post("predict", p, timeout=5)
                if r:
                    prob = r["default_probability"]
                    dec  = r["decision"]
                    col.plotly_chart(
                        mini_gauge(prob, sc["label"]),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )
                    col.markdown(
                        f'<div style="text-align:center; margin-top:-0.8rem;">'
                        f'<span class="badge badge-{dec}">{dec}</span><br>'
                        f'<span style="color:{MUTED}; font-size:0.72rem; margin-top:4px; display:block;">{r["risk_band"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with col.expander("What changed"):
                        st.caption(sc["desc"])
            except Exception:
                col.error("API offline")

    # ── Feature Sensitivity Sweep ──
    st.divider()
    st.markdown('<span class="sec-label">📈 Feature Sensitivity — What If?</span>', unsafe_allow_html=True)
    st.caption("Watch how the risk score shifts as one variable moves across its full range")

    feat_choice = st.selectbox("Feature to vary:", [
        "Credit Utilization %",
        "Monthly Income ($)",
        "Debt Ratio",
        "Age",
        "30-59d Late Payments",
        "90d+ Late Payments",
    ])

    feat_cfg = {
        "Credit Utilization %":   (0, 100, 5,    "revolving_utilization", lambda x: x / 100),
        "Monthly Income ($)":      (500, 30000, 1000, "monthly_income",        lambda x: float(x)),
        "Debt Ratio":              (0.0, 2.0, 0.1,  "debt_ratio",            lambda x: float(x)),
        "Age":                     (18, 85, 5,    "age",                    lambda x: int(x)),
        "30-59d Late Payments":    (0, 10, 1,     "past_due_30_59_days",    lambda x: int(x)),
        "90d+ Late Payments":      (0, 10, 1,     "past_due_90_days",       lambda x: int(x)),
    }
    rmin, rmax, rstep, api_key, xform = feat_cfg[feat_choice]

    if isinstance(rstep, int):
        vals = list(range(int(rmin), int(rmax) + 1, int(rstep)))
    else:
        n    = int(round((rmax - rmin) / rstep)) + 1
        vals = [round(rmin + i * rstep, 3) for i in range(n)]

    sample_vals = vals[::max(1, len(vals) // 20)]

    probs_out = []
    with st.spinner(f"Computing sensitivity for {feat_choice}..."):
        for v in sample_vals:
            pp = dict(base_payload)
            pp[api_key] = xform(v)
            try:
                r = api_post("predict", pp, timeout=3)
                probs_out.append(r["default_probability"] * 100 if r else None)
            except Exception:
                probs_out.append(None)

    valid = [p for p in probs_out if p is not None]
    if valid:
        mcolors = [GREEN if p < 15 else AMBER if p < 40 else RED for p in probs_out]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_vals, y=probs_out,
            mode="lines+markers",
            line=dict(color=CYAN, width=2.5),
            marker=dict(color=mcolors, size=9, line=dict(color=NAVY, width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.04)",
        ))
        fig.add_hline(y=15, line_dash="dash", line_color=GREEN,
                      annotation_text="Low Risk (15%)",  annotation_font_color=GREEN)
        fig.add_hline(y=40, line_dash="dash", line_color=AMBER,
                      annotation_text="High Risk (40%)", annotation_font_color=AMBER)
        fig.update_layout(
            xaxis_title=feat_choice,
            yaxis_title="Default Probability (%)",
            yaxis=dict(range=[0, 100], gridcolor=BORDER, color=MUTED),
            xaxis=dict(gridcolor=BORDER, color=MUTED),
            height=320,
            **plotly_defaults(),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif "Model Dashboard" in page:

    st.markdown(f"""
    <div class="page-header">
        <p class="page-title">📊 Model Performance Dashboard</p>
        <p class="page-sub">Champion model metrics · Feature importance · Multi-model comparison</p>
    </div>
    """, unsafe_allow_html=True)

    info = api_get("model/info")
    if info:
        st.markdown('<span class="sec-label">🏆 Champion Model</span>', unsafe_allow_html=True)
        metrics = info.get("test_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model",     info.get("model_name", "—"))
        c2.metric("AUC-ROC",   f"{metrics.get('test_auc_roc', metrics.get('val_auc', 0)):.4f}")
        c3.metric("KS Stat",   f"{metrics.get('test_ks',  metrics.get('val_ks',  0)):.4f}")
        c4.metric("PR-AUC",    f"{metrics.get('test_pr_auc', metrics.get('val_prauc', 0)):.4f}")
        c5.metric("F1",        f"{metrics.get('test_f1',  metrics.get('val_f1',  0)):.4f}")

        # Feature importance bar
        top_feats = info.get("top_features_by_shap", {})
        if top_feats:
            st.markdown('<span class="sec-label">🔍 Global Feature Importance (SHAP)</span>', unsafe_allow_html=True)
            feat_df = pd.DataFrame(list(top_feats.items()), columns=["Feature", "Importance"])
            feat_df = feat_df.sort_values("Importance", ascending=True)

            fig = go.Figure(go.Bar(
                x=feat_df["Importance"],
                y=feat_df["Feature"],
                orientation="h",
                marker=dict(
                    color=feat_df["Importance"],
                    colorscale=[[0, BORDER], [0.4, "#0097A7"], [1, CYAN]],
                    showscale=False,
                    line=dict(color="rgba(255,255,255,0.05)", width=0.5),
                ),
            ))
            fig.update_layout(
                height=350,
                xaxis=dict(gridcolor=BORDER, color=MUTED, title="Mean |SHAP Value|"),
                yaxis=dict(color=TEXT),
                **plotly_defaults(),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("Could not reach API — showing static reports only.")

    # Static reports from disk
    st.markdown('<span class="sec-label">📈 Training Reports</span>', unsafe_allow_html=True)
    report_root = Path("reports/figures")

    img_pairs = [
        ("model_comparison.png",   "Model Comparison Dashboard"),
        ("roc_pr_curves.png",      "ROC & Precision-Recall Curves"),
        ("calibration_curve.png",  "Calibration Curve"),
        ("confusion_matrix.png",   "Confusion Matrix"),
        ("feature_importance.png", "Feature Importances"),
        ("shap_global.png",        "Global SHAP Analysis"),
        ("shap_interactions.png",  "SHAP Interaction Plots"),
    ]

    # Show 2 per row
    found = [(n, t) for n, t in img_pairs if (report_root / n).exists()]
    for i in range(0, len(found), 2):
        cols = st.columns(2)
        for col, (name, title) in zip(cols, found[i:i+2]):
            col.markdown(f'<span class="sec-label">{title}</span>', unsafe_allow_html=True)
            col.image(str(report_root / name), use_container_width=True)

    if not found:
        st.info("Run `python run_pipeline.py` to generate model comparison reports.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Fairness Audit
# ══════════════════════════════════════════════════════════════════════════════
elif "Fairness" in page:

    st.markdown(f"""
    <div class="page-header">
        <p class="page-title">⚖️ Fairness & Bias Audit</p>
        <p class="page-sub">ECOA compliance · 80% rule · Demographic parity · Equal opportunity analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,179,0,0.07); border:1px solid rgba(255,179,0,0.22);
                border-left:3px solid {AMBER}; border-radius:8px;
                padding:0.9rem 1.1rem; margin-bottom:1.2rem;">
        <strong style="color:{AMBER}; font-size:0.85rem;">Legal Context — ECOA 80% Rule</strong><br>
        <span style="color:{MUTED}; font-size:0.82rem; line-height:1.7;">
            The Equal Credit Opportunity Act prohibits disparate impact. The EEOC four-fifths rule
            requires that no protected group receives approvals at less than 80% the rate of the
            most-favored group. Violations below this threshold are legally actionable.
        </span>
    </div>
    """, unsafe_allow_html=True)

    fairness_path = Path("reports/figures/fairness_report.json")
    if fairness_path.exists():
        with open(fairness_path) as f:
            fdata = json.load(f)

        for key, report in fdata.items():
            attr       = report.get("attribute", key)
            violations = report.get("violation_count", 0)
            sc         = RED if violations > 0 else GREEN
            st_txt     = f"⚠️ {violations} VIOLATION(S)" if violations > 0 else "✅ PASSES"

            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER};
                        border-left:4px solid {sc}; border-radius:8px;
                        padding:0.9rem 1.1rem; margin-bottom:0.7rem;
                        display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:{WHITE}; font-size:0.95rem;">{attr}</span>
                <span style="color:{sc}; font-size:0.75rem; font-weight:700;
                             font-family:'IBM Plex Mono',monospace;">{st_txt}</span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"View details — {attr}"):
                if violations > 0:
                    st.warning(f"Violating groups: {report.get('violating_groups', [])}")
                else:
                    st.success("All groups pass the 80% fairness threshold")

                gm = pd.DataFrame(report.get("group_metrics", []))
                if not gm.empty:
                    cols_show = [c for c in ["group","n_samples","approval_rate","tpr_recall","auc_roc"] if c in gm.columns]
                    st.dataframe(
                        gm[cols_show].rename(columns={
                            "group": "Group", "n_samples": "N",
                            "approval_rate": "Approval Rate",
                            "tpr_recall": "TPR", "auc_roc": "AUC",
                        }).round(4),
                        use_container_width=True,
                    )

                    # Approval rate bar chart
                    fig = go.Figure(go.Bar(
                        x=gm["group"],
                        y=gm["approval_rate"],
                        marker=dict(
                            color=gm["approval_rate"],
                            colorscale=[[0, RED], [0.8, AMBER], [1, GREEN]],
                            showscale=False,
                        ),
                    ))
                    ref  = gm["approval_rate"].max()
                    threshold_line = ref * 0.80
                    fig.add_hline(y=threshold_line, line_dash="dash", line_color=RED,
                                  annotation_text="80% rule threshold",
                                  annotation_font_color=RED)
                    fig.update_layout(
                        title=dict(text="Approval Rate by Group", font=dict(size=11, color=TEXT)),
                        xaxis=dict(color=TEXT, gridcolor=BORDER),
                        yaxis=dict(color=MUTED, gridcolor=BORDER, range=[0, 1]),
                        height=220,
                        **plotly_defaults(),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Fairness image
                for candidate in [
                    f"fairness_{attr.lower().replace(' ', '_')}.png",
                    f"fairness_{key}.png",
                ]:
                    p = Path(f"reports/figures/{candidate}")
                    if p.exists():
                        st.image(str(p), use_container_width=True)
                        break

    else:
        st.info("Run `python src/fairness/fairness_metrics.py` to generate the fairness audit.")

    shap_fair = Path("reports/figures/shap_fairness_by_age.png")
    if shap_fair.exists():
        st.markdown('<span class="sec-label">🔍 SHAP Fairness by Age Group</span>', unsafe_allow_html=True)
        st.image(str(shap_fair), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Predictions DB
# ══════════════════════════════════════════════════════════════════════════════
elif "Predictions DB" in page:

    st.markdown(f"""
    <div class="page-header">
        <p class="page-title">🗄️ Predictions Database</p>
        <p class="page-sub">Live PostgreSQL audit trail · Decision distribution · Compliance log</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    stats = api_get("predictions/stats")
    if stats and stats.get("total_predictions", 0) > 0:
        st.markdown('<span class="sec-label">📊 Live Statistics</span>', unsafe_allow_html=True)
        breakdown = stats.get("decision_breakdown", {})
        total     = stats["total_predictions"]

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Predictions",  total)
        s2.metric("Approved",  breakdown.get("APPROVE", 0))
        s3.metric("Reviewed",  breakdown.get("REVIEW", 0))
        s4.metric("Declined",  breakdown.get("DECLINE", 0))
        s5.metric("Avg Latency", f"{stats.get('avg_inference_ms', 0):.1f}ms")

        # Decision distribution donut
        if breakdown:
            d_colors = {"APPROVE": GREEN, "REVIEW": AMBER, "DECLINE": RED}
            fig = go.Figure(go.Pie(
                labels=list(breakdown.keys()),
                values=list(breakdown.values()),
                marker=dict(colors=[d_colors.get(k, MUTED) for k in breakdown.keys()]),
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(color=TEXT, size=11),
            ))
            fig.update_layout(
                height=250,
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{total}</b><br><span style='font-size:10px'>total</span>",
                    font=dict(size=14, color=WHITE, family="IBM Plex Mono"),
                    showarrow=False,
                )],
                **plotly_defaults(),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    elif stats:
        st.info("No predictions yet. Use the Credit Assessment page to generate some.")
    else:
        st.warning("Database offline or API not running.")

    # History table
    st.markdown('<span class="sec-label">📋 Prediction Audit Log</span>', unsafe_allow_html=True)
    dcol1, dcol2 = st.columns([2, 1])
    limit    = dcol1.slider("Records to show", 10, 200, 50, 10)
    dec_filter = dcol2.selectbox("Filter by decision", ["All", "APPROVE", "REVIEW", "DECLINE"])

    hist_url = f"predictions/history?limit={limit}"
    if dec_filter != "All":
        hist_url += f"&decision={dec_filter}"

    hist = api_get(hist_url)
    if hist and hist.get("predictions"):
        df = pd.DataFrame(hist["predictions"])
        df["default_probability"] = df["default_probability"].apply(lambda x: f"{x:.1%}")
        df["created_at"]          = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        def color_decision(val):
            c = {
                "APPROVE": f"color:{GREEN}; font-weight:700",
                "DECLINE": f"color:{RED}; font-weight:700",
                "REVIEW":  f"color:{AMBER}; font-weight:700",
            }.get(val, "")
            return c

        st.dataframe(
            df[[c for c in ["request_id","decision","risk_band","default_probability",
                            "model_name","inference_ms","created_at"] if c in df.columns]],
            use_container_width=True,
            height=400,
        )
        st.caption(f"Showing {len(df)} records | Source: PostgreSQL openfingraud.predictions")
    elif hist:
        st.info("No records match the filter.")
    else:
        st.warning("Could not fetch history. Is the API running and DB connected?")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — About
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown(f"""
    <div class="page-header">
        <p class="page-title">📖 About OpenFinGuard</p>
        <p class="page-sub">Production-patterned fintech ML · Responsible AI · Explainable by design</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="sec-label">🏗️ Architecture</span>', unsafe_allow_html=True)
        arch = [
            ("Data",           "150K borrowers · Kaggle Give Me Some Credit · SMOTE balanced"),
            ("Models",         "4-model comparison → GradientBoosting champion"),
            ("Explainability", "TreeSHAP · per-prediction adverse action reasons"),
            ("Fairness",       "3-attribute 80% rule audit (ECOA) + SHAP disparity"),
            ("API",            "FastAPI · Pydantic validation · sub-10ms inference"),
            ("Database",       "PostgreSQL · full prediction audit trail · drift metrics"),
            ("Frontend",       "Streamlit · Plotly · real-time scenario explorer"),
            ("CI/CD",          "GitHub Actions · lint → test → Docker build → deploy"),
        ]
        for k, v in arch:
            st.markdown(f"""
            <div style="display:flex; gap:0.8rem; padding:0.5rem 0;
                        border-bottom:1px solid {BORDER}; align-items:flex-start;">
                <span style="color:{CYAN}; font-size:0.65rem; text-transform:uppercase;
                             letter-spacing:1px; font-family:'IBM Plex Mono',monospace;
                             min-width:105px; padding-top:2px;">{k}</span>
                <span style="color:{TEXT}; font-size:0.84rem; line-height:1.4;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<span class="sec-label">📏 Champion Selection Criteria</span>', unsafe_allow_html=True)
        criteria = [
            ("AUC-ROC",         40, "Core discrimination ability"),
            ("KS Statistic",    30, "Credit industry standard (>0.4 = production-ready)"),
            ("PR-AUC",          20, "Best for imbalanced class evaluation"),
            ("Inference Speed", 10, "Production latency constraint"),
        ]
        for name, weight, why in criteria:
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
                        padding:0.7rem 1rem; margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="color:{WHITE}; font-weight:700; font-size:0.9rem;">{name}</span>
                    <span style="color:{CYAN}; font-family:'IBM Plex Mono',monospace;
                                 font-weight:600; font-size:0.9rem;">{weight}%</span>
                </div>
                <div style="color:{MUTED}; font-size:0.78rem;">{why}</div>
                <div style="background:{BORDER}; border-radius:4px; height:3px; margin-top:6px;">
                    <div style="background:{CYAN}; width:{weight}%; height:100%; border-radius:4px; opacity:0.7;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<span class="sec-label">🔧 Tech Stack</span>', unsafe_allow_html=True)
        stack = [
            "Python 3.13", "scikit-learn", "SHAP", "FastAPI",
            "PostgreSQL", "SQLAlchemy", "Streamlit", "Plotly",
            "Docker", "GitHub Actions",
        ]
        badges = " ".join([
            f'<span style="background:rgba(0,229,255,0.08); border:1px solid rgba(0,229,255,0.25); '
            f'color:{CYAN}; padding:0.28rem 0.65rem; border-radius:4px; font-size:0.78rem; '
            f'margin:0.2rem; display:inline-block; font-family:\'IBM Plex Mono\',monospace;">{s}</span>'
            for s in stack
        ])
        st.markdown(f'<div style="line-height:2.2;">{badges}</div>', unsafe_allow_html=True)

    # Pipeline flow
    st.divider()
    st.markdown('<span class="sec-label">🔄 Pipeline Flow</span>', unsafe_allow_html=True)
    steps = [
        ("1", "Data Pipeline",     "run_pipeline.py",         "Clean · impute · engineer 5 features · SMOTE balance"),
        ("2", "Model Training",    "src/models/train.py",     "6-model comparison → champion selection → tune"),
        ("3", "SHAP Analysis",     "src/explainability/",     "Global importance · local explanations · fairness SHAP"),
        ("4", "Fairness Audit",    "src/fairness/",           "Age · income · dependents → 80% rule check"),
        ("5", "API Serving",       "api/main.py",             "FastAPI · PostgreSQL persistence · ECOA compliance"),
        ("6", "Dashboard",         "frontend/app.py",         "Streamlit · live scoring · scenario explorer"),
    ]
    scols = st.columns(6)
    for col, (n, title, file, desc) in zip(scols, steps):
        col.markdown(f"""
        <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
                    padding:0.8rem; text-align:center; height:100%;">
            <div style="color:{CYAN}; font-size:1.2rem; font-weight:800; font-family:'IBM Plex Mono',monospace;">{n}</div>
            <div style="color:{WHITE}; font-weight:700; font-size:0.82rem; margin:4px 0;">{title}</div>
            <div style="color:{CYAN}; font-size:0.65rem; font-family:'IBM Plex Mono',monospace; margin-bottom:6px;">{file}</div>
            <div style="color:{MUTED}; font-size:0.72rem; line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)