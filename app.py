import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import os

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="Farmer Mistake Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────
# CSS — New project's blue dark theme
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:         #0d1117;
    --bg-card:    #161b26;
    --bg-input:   #1a1e2e;
    --border:     #2a2f45;
    --border-hi:  #3d4563;

    --blue:       #4cc9f0;
    --blue-dim:   #2a8fad;
    --teal:       #48bfe3;
    --mint:       #06d6a0;
    --yellow:     #ffd166;
    --red:        #ef476f;
    --red-dim:    #b03252;
    --coral:      #f77f6e;
    --purple:     #9b72cf;
    --green:      #56cfe1;

    --text:       #c8d3f5;
    --text-dim:   #9ba3bf;
    --text-label: #6b7494;

    --font:       'Space Grotesk', sans-serif;
    --mono:       'JetBrains Mono', monospace;
}

/* ── BASE ── */
html, body, [class*="css"] { font-family: var(--font); color: var(--text); }
.stApp { background-color: var(--bg); }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label {
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: var(--mono) !important;
}

/* ── HERO BANNER ── */
.hero {
    background: linear-gradient(135deg, #161b26 0%, #0d1117 60%, #131929 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--blue);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 90% 50%, rgba(76,201,240,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--blue);
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 0.78rem;
    margin-top: 0.5rem;
    letter-spacing: 0.06em;
}
.hero-badge {
    display: inline-block;
    background: rgba(76,201,240,0.1);
    border: 1px solid rgba(76,201,240,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-family: var(--mono);
    color: var(--blue);
    margin-top: 0.7rem;
    letter-spacing: 0.05em;
}

/* ── SECTION HEADERS ── */
.sec-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
    font-family: var(--mono);
}

/* ── METRIC CARDS ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover {
    border-color: var(--border-hi);
    transform: translateY(-3px);
}
.metric-card::before {
    content: "";
    display: block;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--blue-dim), transparent);
    margin: -1.2rem -1.4rem 1rem -1.4rem;
    border-radius: 12px 12px 0 0;
}
.metric-icon  { font-size: 1.4rem; margin-bottom: 0.4rem; display: block; }
.metric-label { font-size: 0.65rem; color: var(--text-label); text-transform: uppercase; letter-spacing: 0.1em; font-family: var(--mono); margin-bottom: 0.4rem; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: var(--blue); line-height: 1; }
.metric-value.danger  { color: var(--red);    text-shadow: 0 0 18px rgba(239,71,111,0.35); }
.metric-value.warning { color: var(--yellow); text-shadow: 0 0 18px rgba(255,209,102,0.3); }
.metric-value.safe    { color: var(--mint);   text-shadow: 0 0 18px rgba(6,214,160,0.3); }

/* ── RISK CARDS ── */
.risk-card {
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin: 1.2rem 0;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-weight: 600;
    font-size: 1rem;
}
.risk-card .risk-icon { font-size: 1.8rem; flex-shrink: 0; }
.risk-high {
    background: rgba(239,71,111,0.08);
    border-color: var(--red);
    color: #ff8fa3;
}
.risk-safe {
    background: rgba(6,214,160,0.08);
    border-color: var(--mint);
    color: var(--mint);
}

/* ── INSIGHT BOX ── */
.insight-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue-dim);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: var(--text-dim);
    font-family: var(--mono);
    display: flex;
    align-items: center;
    gap: 0.6rem;
    transition: border-left-color 0.2s;
}
.insight-box:hover { border-left-color: var(--blue); }
.insight-icon { font-size: 1.1rem; flex-shrink: 0; }

/* ── SUGGESTION CARD ── */
.suggest-card {
    background: var(--bg-card);
    border: 1.5px solid var(--blue);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin: 0.8rem 0;
    box-shadow: 0 0 24px rgba(76,201,240,0.1);
    transition: transform 0.2s;
}
.suggest-card:hover { transform: translateY(-2px); }
.suggest-label { font-size: 0.65rem; color: var(--blue); text-transform: uppercase; letter-spacing: 0.12em; font-family: var(--mono); }
.suggest-crop  { font-size: 1.8rem; font-weight: 700; color: var(--teal); margin-top: 0.25rem; }
.suggest-meta  { color: var(--text-label); font-family: var(--mono); font-size: 0.8rem; margin-top: 0.4rem; }

/* ── CHART NOTE ── */
.chart-note {
    background: var(--bg-card);
    border-left: 3px solid var(--blue);
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    color: var(--text-dim);
    font-family: var(--mono);
    margin: 0.4rem 0 1.2rem 0;
    line-height: 1.6;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #2a8fad, #1a6680) !important;
    color: #e8f4fa !important;
    border: 1px solid var(--blue) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(76,201,240,0.2) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3aabcf, #2a8fad) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(76,201,240,0.35) !important;
}

/* ── INPUTS ── */
.stSelectbox label, .stNumberInput label {
    color: var(--blue) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(76,201,240,0.15) !important;
}

/* ── PROGRESS ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--mint), var(--yellow), var(--red)) !important;
    border-radius: 8px !important;
}
.stProgress > div { background: var(--bg-input) !important; border-radius: 8px !important; }

/* ── DATAFRAME ── */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── FOOTER ── */
.custom-footer {
    text-align: center;
    color: var(--text-label);
    font-family: var(--mono);
    font-size: 0.72rem;
    padding: 1.5rem 0 0.5rem 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    border-radius: 7px !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(76,201,240,0.12) !important;
    color: var(--blue) !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
small, [data-testid="stCaptionContainer"] {
    color: var(--text-label) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────
BG      = "#161b26"
BG_DARK = "#0d1117"
TICK    = "#9ba3bf"
GRID    = "#2a2f45"
SPINE   = "#2a2f45"
TITLE   = "#c8d3f5"
LABEL   = "#9ba3bf"

C_BLUE   = "#4cc9f0"
C_TEAL   = "#48bfe3"
C_MINT   = "#06d6a0"
C_YELLOW = "#ffd166"
C_RED    = "#ef476f"
C_CORAL  = "#f77f6e"
C_PURPLE = "#9b72cf"
C_GREEN  = "#56cfe1"


def set_style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(TITLE)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)


def chart_note(text):
    st.markdown(f'<div class="chart-note">📌 <b>What you are seeing:</b> {text}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model       = joblib.load("models/risk_model.pkl")
    le_state    = joblib.load("models/state_encoder.pkl")
    le_crop     = joblib.load("models/crop_encoder.pkl")
    data_df     = pd.read_csv("data/final_merged_dataset.csv")
    return model, le_state, le_crop, data_df


model, le_state, le_crop, data_df = load_artifacts()

if "Rainfall_Bucket" not in data_df.columns:
    data_df["Rainfall_Bucket"] = pd.cut(
        data_df["ANNUAL_RAINFALL"],
        bins=[0, 400, 700, 1000, 1300, 9999],
        labels=["<400 mm\n(Very Dry)", "400–700 mm\n(Dry)", "700–1000 mm\n(Moderate)",
                "1000–1300 mm\n(Good)", ">1300 mm\n(Heavy)"]
    ).astype(str)


# ─────────────────────────────────────
# SHAP EXPLAINER (cached)
# ─────────────────────────────────────
@st.cache_resource
def load_explainer(_m):
    return shap.TreeExplainer(_m)


explainer = load_explainer(model)


# ─────────────────────────────────────
# SMART ALTERNATIVE CROPS
# ─────────────────────────────────────
def suggest_alternatives(state, rainfall, current_crop, top_n=5):
    candidates = data_df[
        (data_df["State_Name"] == state) &
        (data_df["ANNUAL_RAINFALL"] >= rainfall * 0.8) &
        (data_df["ANNUAL_RAINFALL"] <= rainfall * 1.2)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    stats = candidates.groupby("Crop").agg(
        Avg_Yield=("Yield", "mean"),
        Risk_Rate=("Risk_Label", "mean"),
        Data_Points=("Yield", "count")
    ).reset_index()
    stats = stats[stats["Data_Points"] >= 3]
    if stats.empty:
        return pd.DataFrame()
    ymin, ymax = stats["Avg_Yield"].min(), stats["Avg_Yield"].max()
    stats["Norm_Yield"] = (stats["Avg_Yield"] - ymin) / (ymax - ymin) if ymax > ymin else 1.0
    stats["Score"] = 0.7 * stats["Norm_Yield"] + 0.3 * (1 - stats["Risk_Rate"])
    stats = stats.sort_values("Score", ascending=False)
    stats = stats[stats["Crop"] != current_crop]
    return stats.head(top_n)[["Crop", "Avg_Yield", "Risk_Rate", "Score"]]


# ─────────────────────────────────────
# CROP COMPARISON TABLE
# ─────────────────────────────────────
def get_crop_comparison(state, rainfall):
    candidates = data_df[
        (data_df["State_Name"] == state) &
        (data_df["ANNUAL_RAINFALL"] >= rainfall * 0.8) &
        (data_df["ANNUAL_RAINFALL"] <= rainfall * 1.2)
    ]
    if candidates.empty:
        return pd.DataFrame()
    table = candidates.groupby("Crop").agg(
        Avg_Yield=("Yield", "mean"),
        Max_Yield=("Yield", "max"),
        Risk_Rate=("Risk_Label", "mean"),
        Years=("Crop_Year", "nunique")
    ).reset_index()
    table = table[table["Years"] >= 3]
    table["Risk_%"]    = (table["Risk_Rate"] * 100).round(1)
    table["Avg_Yield"] = table["Avg_Yield"].round(2)
    table["Max_Yield"] = table["Max_Yield"].round(2)
    return table.sort_values("Avg_Yield", ascending=False)[["Crop", "Avg_Yield", "Max_Yield", "Risk_%", "Years"]]


# ═══════════════════════════════════════
# SIDEBAR FILTERS
# ═══════════════════════════════════════
st.sidebar.title("🌾 Filters")
st.sidebar.markdown("---")

state    = st.sidebar.selectbox("📍 State",    sorted(data_df["State_Name"].unique()))
district_opts = sorted(data_df[data_df["State_Name"] == state]["District_Name"].unique())
district = st.sidebar.selectbox("🏘️ District", district_opts)

filtered_df = data_df[
    (data_df["State_Name"]    == state) &
    (data_df["District_Name"] == district)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered records:** `{len(filtered_df)}`")
st.sidebar.markdown(f"**State:** `{state}`")
st.sidebar.markdown(f"**District:** `{district}`")


# ═══════════════════════════════════════
# HERO
# ═══════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-title">🌾 Farmer Mistake Detector</div>
    <div class="hero-sub">AI-POWERED AGRICULTURAL RISK ANALYSIS · INDIA 1901–2015</div>
    <div class="hero-badge">🤖 Machine Learning · 📊 SHAP Explainability · 📈 Historical Data · 🔍 District-Level Analysis</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TABS
# ═══════════════════════════════════════
tab_predict, tab_eda, tab_insights, tab_ai = st.tabs([
    "🔍 AI Prediction",
    "📊 EDA — Data Overview",
    "📈 Trends & Comparisons",
    "🧠 AI Insights"
])


# ══════════════════════════════════════════════════════════
# TAB 1 ── AI PREDICTION (full old project, new UI)
# ══════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="sec-header">⚙️ Farming Parameters</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_state    = st.selectbox("🗺️ State",                le_state.classes_,  key="p_state")
    with c2:
        sel_crop     = st.selectbox("🌱 Crop",                  le_crop.classes_,   key="p_crop")
    with c3:
        sel_rainfall = st.number_input("🌧️ Annual Rainfall (mm)", min_value=0.0, value=800.0, step=50.0)
    with c4:
        sel_area     = st.number_input("📐 Area (hectares)",       min_value=0.0, value=100.0, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 2, 2])
    with btn_col:
        analyze = st.button("🔍 Analyze Decision")

    # ── RESULTS ──────────────────────────────────────────
    if analyze:
        state_enc = le_state.transform([sel_state])[0]
        crop_enc  = le_crop.transform([sel_crop])[0]

        input_df = pd.DataFrame([[sel_rainfall, sel_area, state_enc, crop_enc]],
                                columns=["ANNUAL_RAINFALL", "Area", "State_Encoded", "Crop_Encoded"])

        proba       = model.predict_proba(input_df)[0]
        risk_idx    = list(model.classes_).index(1)
        probability = proba[risk_idx]
        THRESHOLD   = 0.35
        prediction  = 1 if probability > THRESHOLD else 0

        # ── KPI Cards ──
        st.markdown('<div class="sec-header">📊 Risk Summary</div>', unsafe_allow_html=True)

        hist = data_df[
            (data_df["State_Name"] == sel_state) &
            (data_df["Crop"]       == sel_crop)
        ]
        hist_risk  = hist["Risk_Label"].mean() * 100 if not hist.empty else None
        hist_yield = hist["Yield"].mean()             if not hist.empty else None

        risk_cls = "danger" if probability > 0.6 else ("warning" if probability > 0.35 else "safe")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">🎯</span>
                <div class="metric-label">Risk Probability</div>
                <div class="metric-value {risk_cls}">{round(probability*100,1)}%</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            verdict = "HIGH RISK" if prediction == 1 else "SAFE"
            icon    = "🚨" if prediction == 1 else "✅"
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-label">Verdict</div>
                <div class="metric-value {risk_cls}">{verdict}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            hr = f"{hist_risk:.1f}%" if hist_risk is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">📅</span>
                <div class="metric-label">Historical Risk Rate</div>
                <div class="metric-value">{hr}</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            hy = f"{hist_yield:.2f}" if hist_yield is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">🌾</span>
                <div class="metric-label">Avg Historical Yield</div>
                <div class="metric-value">{hy}</div>
            </div>""", unsafe_allow_html=True)

        # ── Risk Card ──
        if prediction == 1:
            st.markdown("""
            <div class="risk-card risk-high">
                <span class="risk-icon">🚨</span>
                <div><strong>High Risk Decision Detected</strong><br>
                This crop-state-rainfall combination has a strong historical pattern of below-average yields.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-card risk-safe">
                <span class="risk-icon">✅</span>
                <div><strong>This Farming Decision Appears Safe</strong><br>
                Historical data supports a reasonable chance of good yields.</div>
            </div>""", unsafe_allow_html=True)

        st.progress(float(probability))

        # ── Contextual Insights ──
        st.markdown('<div class="sec-header">💡 Data-Driven Insights</div>', unsafe_allow_html=True)

        med_rain = data_df["ANNUAL_RAINFALL"].median()
        med_area = data_df["Area"].median()

        insights_list = []
        if sel_rainfall < med_rain:
            insights_list.append(("💧", f"Rainfall ({sel_rainfall:.0f}mm) is below national median ({med_rain:.0f}mm) — crops may face water stress."))
        else:
            insights_list.append(("🌊", f"Rainfall ({sel_rainfall:.0f}mm) is above national median ({med_rain:.0f}mm) — water availability looks favorable."))

        if sel_area > med_area:
            insights_list.append(("📏", f"Cultivation area ({sel_area:.0f} ha) is larger than typical — operational risk scales with acreage."))
        else:
            insights_list.append(("📐", f"Cultivation area ({sel_area:.0f} ha) is within typical range."))

        if hist_risk is not None:
            if hist_risk > 40:
                insights_list.append(("📉", f"This crop in {sel_state} has historically failed {hist_risk:.1f}% of years — notably high."))
            elif hist_risk > 20:
                insights_list.append(("📊", f"Moderate historical risk: this crop failed in {hist_risk:.1f}% of recorded years."))
            else:
                insights_list.append(("📈", f"Low historical risk: {hist_risk:.1f}% failure rate for this crop in {sel_state}."))

        for icon, text in insights_list:
            st.markdown(f"""
            <div class="insight-box">
                <span class="insight-icon">{icon}</span>
                <span>{text}</span>
            </div>""", unsafe_allow_html=True)

        # ── SHAP ──────────────────────────────────────────
        st.markdown('<div class="sec-header">🔬 SHAP Explainability — Why This Prediction?</div>',
                    unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            shap_values  = explainer.shap_values(input_df)
            feature_names = ["Annual Rainfall", "Area", "State", "Crop"]

            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = np.array(shap_values)
                sv = sv[0, :, 1] if sv.ndim == 3 else (sv[0] if sv.ndim == 2 else sv.flatten())

            sv = np.array(sv).flatten()
            if len(sv) != len(feature_names):
                sv = sv[:len(feature_names)]

            fig_s, ax_s = plt.subplots(figsize=(8, 3.2))
            colors = [C_RED if v > 0 else C_MINT for v in sv]
            bars   = ax_s.barh(feature_names, sv, color=colors, edgecolor="none", height=0.5)
            ax_s.axvline(0, color=C_YELLOW, linewidth=1, linestyle="--", alpha=0.6)
            ax_s.set_xlabel("SHAP Value (impact on risk probability)", fontsize=9)
            ax_s.set_title("Feature Contributions to Risk Prediction", fontsize=11)
            for bar, val in zip(bars, sv):
                ax_s.text(
                    val + (0.0005 if val >= 0 else -0.0005),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center",
                    ha="left" if val >= 0 else "right",
                    color="#c8d3f5", fontsize=8
                )
            red_p   = mpatches.Patch(color=C_RED,  label="🔴 Increases Risk")
            green_p = mpatches.Patch(color=C_MINT, label="🟢 Reduces Risk")
            ax_s.legend(handles=[red_p, green_p], fontsize=8,
                        facecolor=BG, edgecolor=SPINE, labelcolor=TITLE)
            set_style(ax_s, fig_s)
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close()

        chart_note("Red bars = features pushing the risk UP. Green bars = features pulling risk DOWN. "
                   "Longer bar = stronger influence on this prediction.")

        # ── Alternative Crops (only if High Risk) ──
        if prediction == 1:
            st.markdown('<div class="sec-header">🌱 Smarter Alternative Crops</div>',
                        unsafe_allow_html=True)
            st.caption("🏅 Ranked by composite score: 70% normalized yield + 30% low-risk rate")

            alternatives = suggest_alternatives(sel_state, sel_rainfall, sel_crop)

            if not alternatives.empty:
                best = alternatives.iloc[0]
                st.markdown(f"""
                <div class="suggest-card">
                    <div class="suggest-label">🏆 Top Recommended Crop</div>
                    <div class="suggest-crop">🌿 {best['Crop']}</div>
                    <div class="suggest-meta">
                        📊 Avg Yield: {best['Avg_Yield']:.2f} &nbsp;·&nbsp;
                        🛡️ Risk Rate: {best['Risk_Rate']*100:.1f}% &nbsp;·&nbsp;
                        ⭐ Score: {best['Score']:.3f}
                    </div>
                </div>""", unsafe_allow_html=True)

                if len(alternatives) > 1:
                    st.markdown("**🌱 Other alternatives:**")
                    disp = alternatives.copy()
                    disp["Risk_%"]    = (disp["Risk_Rate"] * 100).round(1)
                    disp["Score"]     = disp["Score"].round(3)
                    disp["Avg_Yield"] = disp["Avg_Yield"].round(2)
                    st.dataframe(disp[["Crop", "Avg_Yield", "Risk_%", "Score"]].iloc[1:],
                                 use_container_width=True, hide_index=True)
            else:
                st.info("No suitable alternative crops found for this state/rainfall combination.")

        # ── Crop Comparison Table ──
        st.markdown(f'<div class="sec-header">📋 All Crops in {sel_state} — Comparison Table</div>',
                    unsafe_allow_html=True)
        st.caption(f"All crops in {sel_state} with ±20% of your rainfall ({sel_rainfall:.0f}mm), sorted by avg yield")

        comparison = get_crop_comparison(sel_state, sel_rainfall)
        if not comparison.empty:
            def _highlight(row):
                if row["Crop"] == sel_crop:
                    return ["background-color: rgba(76,201,240,0.1); font-weight: bold"] * len(row)
                return [""] * len(row)

            styled = (comparison.style
                      .apply(_highlight, axis=1)
                      .format({"Avg_Yield": "{:.2f}", "Max_Yield": "{:.2f}", "Risk_%": "{:.1f}%"})
                      .background_gradient(subset=["Risk_%"], cmap="YlOrRd", vmin=0, vmax=60))
            st.dataframe(styled, use_container_width=True, hide_index=True, height=340)
        else:
            st.warning("Not enough data to build comparison table.")

        # ── Historical Yield Trend ──
        st.markdown(f'<div class="sec-header">📈 Historical Yield Trend — {sel_crop} in {sel_state}</div>',
                    unsafe_allow_html=True)

        hist_data = data_df[
            (data_df["State_Name"] == sel_state) &
            (data_df["Crop"]       == sel_crop)
        ].sort_values("Crop_Year")

        if not hist_data.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(hist_data["Crop_Year"], hist_data["Yield"],
                     color=C_BLUE, linewidth=2.5, label="Yield", zorder=3)
            risky  = hist_data[hist_data["Risk_Label"] == 1]
            safe_y = hist_data[hist_data["Risk_Label"] == 0]
            ax1.scatter(risky["Crop_Year"],  risky["Yield"],  color=C_RED,  s=70,
                        label="🚨 Risk Year", zorder=5, edgecolors=C_CORAL, linewidths=0.8)
            ax1.scatter(safe_y["Crop_Year"], safe_y["Yield"], color=C_MINT, s=35,
                        label="✅ Safe Year", zorder=4, alpha=0.7)
            if len(hist_data) >= 5:
                rolling = hist_data["Yield"].rolling(5, center=True).mean()
                ax1.plot(hist_data["Crop_Year"], rolling,
                         color=C_YELLOW, linewidth=1.5, linestyle="--",
                         label="5-yr Rolling Avg", alpha=0.9)
            ax1.set_xlabel("Year", fontsize=10)
            ax1.set_ylabel("Yield (tonnes/ha)", fontsize=10)
            ax1.set_title(f"🌾 {sel_crop} Yield History in {sel_state}", fontsize=12)
            ax1.legend(facecolor=BG, edgecolor=SPINE, labelcolor=TITLE, fontsize=9)
            set_style(ax1, fig1)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()
            chart_note("Red dots = years when yield fell significantly below normal (risk years). "
                       "Yellow dashed line = 5-year rolling average to show the long-term trend.")
        else:
            st.warning("No historical data available for this state-crop combination.")

        # ── Rainfall Trend ──
        st.markdown(f'<div class="sec-header">🌧️ Rainfall Trend — {sel_state}</div>',
                    unsafe_allow_html=True)

        rain_data = data_df[data_df["State_Name"] == sel_state].sort_values("Crop_Year")
        if not rain_data.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 3.5))
            ax2.fill_between(rain_data["Crop_Year"], rain_data["ANNUAL_RAINFALL"],
                             alpha=0.18, color=C_BLUE)
            ax2.plot(rain_data["Crop_Year"], rain_data["ANNUAL_RAINFALL"],
                     color=C_BLUE, linewidth=2, label="Annual Rainfall")
            ax2.axhline(sel_rainfall, color=C_YELLOW, linewidth=1.5, linestyle="--",
                        label=f"Your Input ({sel_rainfall:.0f}mm)", alpha=0.85)
            ax2.set_xlabel("Year", fontsize=10)
            ax2.set_ylabel("Rainfall (mm)", fontsize=10)
            ax2.set_title(f"🌧️ Rainfall History in {sel_state}", fontsize=12)
            ax2.legend(facecolor=BG, edgecolor=SPINE, labelcolor=TITLE, fontsize=9)
            set_style(ax2, fig2)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            chart_note("Yellow dashed line = your rainfall input. See how your input compares to historical rainfall in this state.")
        else:
            st.warning("No rainfall data for selected state.")


# ══════════════════════════════════════════════════════════
# TAB 2 ── EDA
# ══════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("📊 Exploratory Data Analysis")
    st.caption("Basic data distribution and structure — no predictions, just understanding the data.")

    if len(filtered_df) == 0:
        st.warning("No data available for the selected State/District.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Yield Distribution (kg/ha)**")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            set_style(ax, fig)
            sns.histplot(filtered_df["Yield"], bins=30, kde=True, ax=ax,
                         color=C_BLUE, edgecolor=BG_DARK)
            mean_y = filtered_df["Yield"].mean()
            ax.axvline(mean_y, color=C_YELLOW, linestyle="--", linewidth=1.8,
                       label=f"Average: {mean_y:.0f}")
            ax.set_xlabel("Yield (kg/ha)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title("How is crop yield distributed?", fontsize=10, color=TITLE)
            ax.legend(fontsize=8, labelcolor=TITLE, facecolor=BG, edgecolor=SPINE)
            st.pyplot(fig); plt.close()
            chart_note("Yellow line = average yield. Right-skewed? Some farms are highly productive — study them!")

        with col2:
            st.markdown("**🌧️ Rainfall Distribution (mm)**")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            set_style(ax, fig)
            sns.histplot(filtered_df["ANNUAL_RAINFALL"], bins=30, kde=True, ax=ax,
                         color=C_MINT, edgecolor=BG_DARK)
            mean_r = filtered_df["ANNUAL_RAINFALL"].mean()
            ax.axvline(mean_r, color=C_YELLOW, linestyle="--", linewidth=1.8,
                       label=f"Average: {mean_r:.0f} mm")
            ax.set_xlabel("Annual Rainfall (mm)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title("How is rainfall distributed?", fontsize=10, color=TITLE)
            ax.legend(fontsize=8, labelcolor=TITLE, facecolor=BG, edgecolor=SPINE)
            st.pyplot(fig); plt.close()
            chart_note("Narrow peak = consistent rainfall (predictable farming). Wide = high variability.")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**⚠️ Safe vs At-Risk Farms**")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            set_style(ax, fig)
            cnts = filtered_df["Risk_Label"].value_counts()
            s, r = cnts.get(0, 0), cnts.get(1, 0)
            total = s + r
            bars = ax.bar(["✅ Safe", "⚠️ At Risk"], [s, r],
                          color=[C_MINT, C_RED], edgecolor=BG_DARK, width=0.5)
            for bar, val in zip(bars, [s, r]):
                pct = (val / total * 100) if total > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f"{val}\n({pct:.1f}%)", ha="center", fontsize=9, color=TITLE)
            ax.set_ylabel("Number of farms", fontsize=10)
            ax.set_title("Safe vs. At-Risk distribution", fontsize=10, color=TITLE)
            st.pyplot(fig); plt.close()
            chart_note("If red bar > 30%, this district needs policy attention.")

        with col4:
            st.markdown("**🔗 Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            set_style(ax, fig)
            cdf = filtered_df[["Yield", "ANNUAL_RAINFALL", "Area"]].copy()
            cdf.columns = ["Yield\n(kg/ha)", "Rainfall\n(mm)", "Area\n(ha)"]
            sns.heatmap(cdf.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax, linewidths=1, linecolor=BG_DARK,
                        annot_kws={"size": 12, "weight": "bold"}, vmin=-1, vmax=1)
            ax.set_title("Variable relationships", fontsize=10, color=TITLE)
            st.pyplot(fig); plt.close()
            chart_note("+1 = both rise together. -1 = one up, other down. 0 = no relationship.")

        # Boxplot
        st.markdown("**📦 Yield Outlier Detection — Boxplot**")
        fig, ax = plt.subplots(figsize=(9, 2.8))
        set_style(ax, fig)
        ax.boxplot(filtered_df["Yield"].dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor=C_BLUE+"33", color=C_BLUE),
                   medianprops=dict(color=C_YELLOW, linewidth=2),
                   whiskerprops=dict(color=C_BLUE), capprops=dict(color=C_BLUE),
                   flierprops=dict(marker="o", color=C_RED, alpha=0.5, markersize=4))
        med = filtered_df["Yield"].median()
        ax.axvline(med, color=C_YELLOW, linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Yield (kg/ha)", fontsize=10)
        ax.set_title("Outlier detection — unusually high or low performers", fontsize=10, color=TITLE)
        q1 = filtered_df["Yield"].quantile(0.25)
        q3 = filtered_df["Yield"].quantile(0.75)
        yl = ax.get_ylim()
        ax.text(q1,  yl[1]*0.85, f"Q1: {q1:.0f}",  ha="center", fontsize=8, color=C_MINT)
        ax.text(med, yl[1]*0.9,  f"Med: {med:.0f}", ha="center", fontsize=8, color=C_YELLOW)
        ax.text(q3,  yl[1]*0.85, f"Q3: {q3:.0f}",  ha="center", fontsize=8, color=C_MINT)
        st.pyplot(fig); plt.close()
        chart_note("Red dots = extreme outlier farms. Study high-yield outliers for best practices.")

        # Scatter: Rainfall vs Yield
        st.markdown("**🔵 Rainfall vs Yield (colored by risk)**")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        set_style(ax, fig)
        s_df = filtered_df[filtered_df["Risk_Label"] == 0]
        r_df = filtered_df[filtered_df["Risk_Label"] == 1]
        ax.scatter(s_df["ANNUAL_RAINFALL"], s_df["Yield"], c=C_MINT, alpha=0.5, s=18, label="✅ Safe")
        ax.scatter(r_df["ANNUAL_RAINFALL"], r_df["Yield"], c=C_RED,  alpha=0.5, s=18, label="⚠️ At-risk")
        if len(filtered_df) > 5:
            z  = np.polyfit(filtered_df["ANNUAL_RAINFALL"].dropna(), filtered_df["Yield"].dropna(), 1)
            xl = np.linspace(filtered_df["ANNUAL_RAINFALL"].min(), filtered_df["ANNUAL_RAINFALL"].max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), color=C_YELLOW, linewidth=1.8, linestyle="--", label="Trend")
        ax.set_xlabel("Annual Rainfall (mm)", fontsize=10)
        ax.set_ylabel("Yield (kg/ha)", fontsize=10)
        ax.set_title("Does higher rainfall lead to higher yield?", fontsize=11, color=TITLE)
        ax.legend(fontsize=9, labelcolor=TITLE, facecolor=BG, edgecolor=SPINE)
        st.pyplot(fig); plt.close()
        chart_note("Yellow line = trend. Watch where red dots cluster — that's the drought threshold for this district.")

        with st.expander("🔍 View raw data (first 50 rows)"):
            st.dataframe(filtered_df.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 ── TRENDS & COMPARISONS
# ══════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("📈 Trends & Comparisons")
    st.caption("Time-based trends, crop comparisons, and state-level patterns.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Yield Trend Over Years**")
        if len(filtered_df) > 0 and "Crop_Year" in filtered_df.columns:
            trend = filtered_df.groupby("Crop_Year")["Yield"].mean()
            slope = np.polyfit(range(len(trend)), trend.values, 1)[0] if len(trend) > 1 else 0
            fig, ax = plt.subplots(figsize=(5, 3.5))
            set_style(ax, fig)
            ax.plot(trend.index, trend.values, marker="o", color=C_BLUE, linewidth=2, markersize=5)
            ax.fill_between(trend.index, trend.values, alpha=0.12, color=C_BLUE)
            if len(trend) > 1:
                z = np.polyfit(trend.index, trend.values, 1)
                ax.plot(trend.index, np.poly1d(z)(trend.index),
                        color=C_YELLOW, linestyle="--", linewidth=1.8,
                        label=f"Trend (slope: {slope:+.1f})")
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("Avg Yield (kg/ha)", fontsize=10)
            ax.set_title("Annual yield trend", fontsize=10, color=TITLE)
            ax.legend(fontsize=8, labelcolor=TITLE, facecolor=BG, edgecolor=SPINE)
            set_style(ax, fig)
            st.pyplot(fig); plt.close()
            direction = "📈 Improving" if slope > 0 else "📉 Declining"
            chart_note(f"Slope = {slope:+.1f} kg/ha per year ({direction}). Falling line = soil/pest/rainfall issues.")

    with col2:
        st.markdown("**⚠️ Crop Risk % Trend (Full Dataset)**")
        yr = data_df.groupby("Crop_Year")["Risk_Label"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        set_style(ax, fig)
        ax.plot(yr.index, yr.values, marker="o", color=C_RED, linewidth=2, markersize=5)
        ax.fill_between(yr.index, yr.values, alpha=0.12, color=C_RED)
        ax.axhline(30, color=C_YELLOW, linestyle="--", linewidth=1.2, alpha=0.8, label="30% danger line")
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("At-Risk Farms (%)", fontsize=10)
        ax.set_title("Is overall crop risk increasing?", fontsize=10, color=TITLE)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(fontsize=8, labelcolor=TITLE, facecolor=BG, edgecolor=SPINE)
        st.pyplot(fig); plt.close()
        chart_note("Spikes often match historical drought years. Yellow line = 30% alert threshold.")

    # Top 10 Crops
    st.markdown("**🌱 Top 10 Crops by Average Yield (this district)**")
    if len(filtered_df) > 0:
        top_crops = filtered_df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        set_style(ax, fig)
        palette = [C_BLUE, C_TEAL, C_MINT, C_GREEN, C_PURPLE, C_CORAL, C_YELLOW, C_RED, "#a29bfe", "#fd79a8"]
        top_crops.sort_values().plot(kind="barh", ax=ax,
                                     color=palette[:len(top_crops)][::-1], edgecolor=BG_DARK)
        for i, (val, lbl) in enumerate(zip(top_crops.sort_values(), top_crops.sort_values().index)):
            ax.text(val + 30, i, f"{val:.0f}", va="center", fontsize=9, color=TITLE)
        ax.set_xlabel("Average Yield (kg/ha)", fontsize=10)
        ax.set_title(f"Best crop: {top_crops.index[0]} ({top_crops.values[0]:.0f} kg/ha)", fontsize=11, color=TITLE)
        st.pyplot(fig); plt.close()
        chart_note(f"Longest bar = best performer. If farmers avoid it, investigate market/water/subsidy barriers.")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🗺️ State-wise Avg Yield (Top 10)**")
        sv = data_df.groupby("State_Name")["Yield"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        set_style(ax, fig)
        sv.sort_values().plot(kind="barh", ax=ax, color=C_TEAL, edgecolor=BG_DARK)
        if state in sv.index:
            idx = list(sv.sort_values().index).index(state)
            ax.patches[idx].set_facecolor(C_YELLOW)
            ax.text(2, idx, "← Your state", va="center", fontsize=8, color=C_YELLOW)
        ax.set_xlabel("Avg Yield (kg/ha)", fontsize=10)
        ax.set_title("Which state is most productive?", fontsize=10, color=TITLE)
        st.pyplot(fig); plt.close()
        chart_note("Yellow bar = your state. Study techniques from top-performing states.")

    with col4:
        st.markdown("**🌦️ Rainfall Range vs Avg Yield**")
        bkt = data_df.groupby("Rainfall_Bucket")["Yield"].mean()
        valid = bkt[bkt.index != "nan"]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        set_style(ax, fig)
        bcolors = [C_RED, C_CORAL, C_BLUE, C_MINT, C_PURPLE]
        ax.bar(range(len(valid)), valid.values,
               color=bcolors[:len(valid)], edgecolor=BG_DARK, width=0.6)
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels(valid.index, fontsize=8, color=TICK)
        ax.set_ylabel("Avg Yield (kg/ha)", fontsize=10)
        ax.set_title("Optimal rainfall range for yield?", fontsize=10, color=TITLE)
        for i, (v, lbl) in enumerate(zip(valid.values, valid.index)):
            ax.text(i, v + 30, f"{v:.0f}", ha="center", fontsize=8, color=TITLE)
        best_bkt = valid.idxmax()
        st.pyplot(fig); plt.close()
        chart_note(f"Best rainfall range = {best_bkt}. If outside this range, plan irrigation accordingly.")

    # Crop Risk vs Yield Scatter
    st.markdown("**🎯 Crop-wise: Risk Rate vs Average Yield**")
    cs = data_df.groupby("Crop").agg(Yield=("Yield", "mean"), Risk=("Risk_Label", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    set_style(ax, fig)
    sc = ax.scatter(cs["Risk"], cs["Yield"], alpha=0.75,
                    c=cs["Risk"], cmap="RdYlGn_r", s=45)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Risk Rate", color=TICK, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TICK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TICK)
    ax.set_xlabel("Risk Rate (0=Safe → 1=High Risk)", fontsize=10)
    ax.set_ylabel("Average Yield (kg/ha)", fontsize=10)
    ax.set_title("Ideal crops = Low Risk + High Yield (top-left corner)", fontsize=11, color=TITLE)
    ax.axvline(0.3, color=C_YELLOW, linestyle="--", linewidth=1.2, alpha=0.8, label="30% threshold")
    ax.legend(fontsize=8, labelcolor=TITLE, facecolor=BG, edgecolor=SPINE)
    st.pyplot(fig); plt.close()
    chart_note("Top-left = best crops (low risk + high yield). Bottom-right = avoid these.")

    # KMeans Cluster Plot
    if "Cluster" in data_df.columns:
        st.markdown("**🔵 KMeans Cluster Analysis — Rainfall vs Yield**")
        sample_c = data_df.sample(min(1500, len(data_df)), random_state=42)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        set_style(ax, fig)
        palette = {0: C_MINT, 1: C_BLUE, 2: C_CORAL}
        cnames  = {
            0: "Cluster 0: Low Rain, Low Yield",
            1: "Cluster 1: Medium Rain, Medium Yield",
            2: "Cluster 2: High Rain, High Yield"
        }
        for cid, grp in sample_c.groupby("Cluster"):
            if cid < 0: continue
            ax.scatter(grp["ANNUAL_RAINFALL"], grp["Yield"],
                       label=cnames.get(int(cid), f"Cluster {cid}"),
                       color=palette.get(int(cid), "#aaa"), alpha=0.5, s=15)
        ax.set_xlabel("Annual Rainfall (mm)", fontsize=10)
        ax.set_ylabel("Yield (kg/ha)", fontsize=10)
        ax.set_title("ML grouped farms into 3 clusters", fontsize=11, color=TITLE)
        ax.legend(fontsize=8, labelcolor=TITLE, facecolor=BG, edgecolor=SPINE)
        st.pyplot(fig); plt.close()
        chart_note("ML auto-divided farms by rainfall+yield pattern. Each cluster = different farm type. "
                   "Useful for targeted government schemes — one strategy per cluster.")
    else:
        st.info("Cluster column not found. Re-run train_model.py.")


# ══════════════════════════════════════════════════════════
# TAB 4 ── AI INSIGHTS (rule-based, district-level)
# ══════════════════════════════════════════════════════════
with tab_ai:
    st.subheader("🧠 AI Insights — Automated Analysis")
    st.caption("Rule-based AI automatically analyzes your district's data and gives actionable recommendations.")

    if len(filtered_df) == 0:
        st.warning("No data for selected region.")
    else:
        insights = []

        avg_y   = filtered_df["Yield"].mean()
        st_avg  = data_df[data_df["State_Name"] == state]["Yield"].mean()
        nat_avg = data_df["Yield"].mean()

        if avg_y < st_avg:
            insights.append({"type": "warning", "icon": "📉",
                "msg": f"Yield in **{district}** ({avg_y:.1f} kg/ha) is **below** the state average "
                       f"({st_avg:.1f} kg/ha). Gap = {st_avg - avg_y:.1f} kg/ha. "
                       f"Study high-performing districts in {state}."})
        else:
            insights.append({"type": "success", "icon": "📈",
                "msg": f"Yield in **{district}** ({avg_y:.1f} kg/ha) is **above** the state average "
                       f"({st_avg:.1f} kg/ha) and "
                       f"{'also above' if avg_y > nat_avg else 'below'} national average ({nat_avg:.1f} kg/ha)."})

        rr = filtered_df["Risk_Label"].mean()
        if rr > 0.4:
            insights.append({"type": "error", "icon": "🚨",
                "msg": f"**High risk area** — {rr*100:.1f}% crop failure rate. "
                       "Crop insurance, irrigation investment, and contingency planning are strongly recommended."})
        elif rr > 0.2:
            insights.append({"type": "warning", "icon": "⚠️",
                "msg": f"**Moderate risk** — {rr*100:.1f}% risk rate. "
                       "Try drought-resistant varieties, seasonal planning, and water conservation."})
        else:
            insights.append({"type": "success", "icon": "✅",
                "msg": f"**Low risk area** — only {rr*100:.1f}% risk rate. "
                       "Stable conditions. Focus on yield optimization and market linkages."})

        rain_avg = filtered_df["ANNUAL_RAINFALL"].mean()
        if rain_avg < 600:
            insights.append({"type": "warning", "icon": "🌵",
                "msg": f"Low rainfall ({rain_avg:.0f} mm/yr). "
                       "Consider drip irrigation, rainwater harvesting, millet/sorghum."})
        elif rain_avg > 1200:
            insights.append({"type": "info", "icon": "🌧️",
                "msg": f"High rainfall ({rain_avg:.0f} mm/yr). "
                       "Good for rice, sugarcane, banana. Plan waterlogging prevention."})
        else:
            insights.append({"type": "info", "icon": "🌦️",
                "msg": f"Moderate rainfall ({rain_avg:.0f} mm/yr). "
                       "Diverse crop options — wheat, maize, soybean. Supplemental irrigation helps."})

        bc = filtered_df.groupby("Crop")["Yield"].mean()
        insights.append({"type": "success", "icon": "🌱",
            "msg": f"Best crop: **{bc.idxmax()}** ({bc.max():.0f} kg/ha). "
                   f"Lowest: **{bc.idxmin()}** ({bc.min():.0f} kg/ha). Reallocate resources accordingly."})

        if "Crop_Year" in filtered_df.columns and filtered_df["Crop_Year"].nunique() > 1:
            tv   = filtered_df.groupby("Crop_Year")["Yield"].mean()
            slp  = np.polyfit(range(len(tv)), tv.values, 1)[0]
            if slp > 0:
                insights.append({"type": "success", "icon": "📊",
                    "msg": f"Yield trend **improving** (+{slp:.2f} kg/ha/yr). "
                           "Driven by better technology, seeds, or irrigation."})
            else:
                insights.append({"type": "warning", "icon": "📊",
                    "msg": f"Yield trend **declining** ({slp:.2f} kg/ha/yr). "
                           "Soil health test recommended. Try crop rotation."})

        if len(filtered_df) > 5:
            corr = filtered_df["ANNUAL_RAINFALL"].corr(filtered_df["Yield"])
            insights.append({"type": "info", "icon": "🔗",
                "msg": f"Rainfall-Yield correlation: **{corr:.2f}** "
                       f"({'strong' if corr > 0.5 else 'moderate' if corr > 0.2 else 'weak'}). "
                       + ("Invest in irrigation — rainfall is the key driver here." if corr > 0.5
                          else "Soil quality, seeds, and technique also matter here.")})

        if len(filtered_df) > 5:
            crs = filtered_df.groupby("Crop").agg(
                avg_y=("Yield", "mean"), risk=("Risk_Label", "mean"), cnt=("Yield", "count")
            ).reset_index()
            safe_hi = crs[(crs["risk"] < 0.2) & (crs["cnt"] >= 5)].sort_values("avg_y", ascending=False)
            if len(safe_hi) > 0:
                top = safe_hi.iloc[0]
                insights.append({"type": "success", "icon": "🏆",
                    "msg": f"**Best safe recommendation: {top['Crop']}** — "
                           f"{top['avg_y']:.0f} kg/ha with only {top['risk']*100:.1f}% risk. "
                           "Low risk + high reward — ideal combination!"})

        for ins in insights:
            if ins["type"] == "success":
                st.success(f"{ins['icon']}  {ins['msg']}")
            elif ins["type"] == "warning":
                st.warning(f"{ins['icon']}  {ins['msg']}")
            elif ins["type"] == "error":
                st.error(f"{ins['icon']}  {ins['msg']}")
            else:
                st.info(f"{ins['icon']}  {ins['msg']}")

        st.markdown("---")
        st.markdown("**📋 Summary Statistics**")
        summary = filtered_df[["Yield", "ANNUAL_RAINFALL", "Area", "Risk_Label"]].describe().round(2)
        summary.rename(columns={
            "Yield": "Yield (kg/ha)", "ANNUAL_RAINFALL": "Rainfall (mm)",
            "Area": "Area (ha)", "Risk_Label": "Risk Rate (0–1)"
        }, inplace=True)
        st.dataframe(summary, use_container_width=True)

        st.markdown("**🌾 All Crops Performance (this district)**")
        csm = filtered_df.groupby("Crop").agg(
            Avg_Yield=("Yield", "mean"),
            Risk_Rate=("Risk_Label", "mean"),
            Records=("Yield", "count")
        ).round(2).sort_values("Avg_Yield", ascending=False).reset_index()
        csm["Avg_Yield"]  = csm["Avg_Yield"].apply(lambda x: f"{x:.0f} kg/ha")
        csm["Risk_Rate"]  = csm["Risk_Rate"].apply(lambda x: f"{x*100:.1f}%")
        csm.columns       = ["Crop", "Avg Yield", "Risk Rate", "Records"]
        st.dataframe(csm, use_container_width=True)
        st.caption("Low Risk Rate + High Yield = ideal crop choice.")

# ─────────────────────────────────────
# FOOTER
# ─────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="custom-footer">'
    '⚡ Predictions based on historical agricultural patterns &nbsp;·&nbsp; India 1901–2015 &nbsp;·&nbsp; '
    '🌾 Farmer Mistake Detector'
    '</div>',
    unsafe_allow_html=True
)