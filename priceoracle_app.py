"""
╔══════════════════════════════════════════════════════════╗
║   🔮 PriceOracle — AI Dynamic Pricing Intelligence       ║
║   Predict Demand → Simulate Prices → Optimize Profit     ║
║   → Explain with AI (Groq)                               ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run priceoracle_app.py
Deps: pip install streamlit pandas numpy scikit-learn joblib
      plotly requests python-dotenv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
import requests
from datetime import datetime, date
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()
load_dotenv(dotenv_path=os.path.join("PriceOptimizer", ".env"), override=False)

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PriceOracle — AI Pricing Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS  (dark premium theme)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Dark background */
  .stApp { background: #0d1117; color: #e6edf3; }
  .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
  .metric-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
  .metric-card .value { font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .metric-card .delta { font-size: 0.8rem; margin-top: 0.2rem; }
  .green  { color: #3fb950; }
  .red    { color: #f85149; }
  .blue   { color: #58a6ff; }
  .gold   { color: #d29922; }
  .white  { color: #e6edf3; }

  /* ── Section headers ── */
  .section-header {
    font-size: 1rem; font-weight: 600;
    color: #58a6ff; letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem; margin-bottom: 1rem;
  }

  /* ── Optimal price banner ── */
  .optimal-banner {
    background: linear-gradient(135deg, #0d4a2f 0%, #0e3a29 100%);
    border: 2px solid #3fb950;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .optimal-banner .opt-label { font-size: 0.8rem; color: #7ee787; text-transform: uppercase; letter-spacing: 0.1em; }
  .optimal-banner .opt-price { font-size: 3rem; font-weight: 800; color: #3fb950; font-family: 'JetBrains Mono', monospace; }
  .optimal-banner .opt-sub   { font-size: 0.9rem; color: #8b949e; margin-top: 0.4rem; }

  /* ── AI Recommendation box ── */
  .ai-box {
    background: linear-gradient(135deg, #161b22 0%, #1a1f2e 100%);
    border-left: 4px solid #bc8cff;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
  }
  .ai-box .ai-label { font-size: 0.7rem; color: #bc8cff; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
  .ai-box .ai-text  { font-size: 0.92rem; line-height: 1.6; color: #c9d1d9; }

  /* ── Insight pill ── */
  .insight-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    margin: 0.2rem;
    color: #c9d1d9;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] .stMarkdown { color: #8b949e; }

  /* Input widgets */
  .stNumberInput input, .stSelectbox select,
  .stSlider .stSlider { background: #161b22 !important; color: #e6edf3 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
  .stTabs [data-baseweb="tab"]      { color: #8b949e; }
  .stTabs [aria-selected="true"]    { color: #58a6ff !important; }

  /* Divider */
  hr { border-color: #21262d; }

  /* Header */
  .main-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 1rem 0 1.5rem;
    margin-bottom: 1.5rem;
  }
  .app-title {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 50%, #3fb950 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .app-subtitle { font-size: 0.9rem; color: #8b949e; margin-top: -0.3rem; }

  /* Warning / success banners */
  .warning-box {
    background: #2d1f0e; border: 1px solid #d29922;
    border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #e3b341;
  }
  .success-box {
    background: #0d2d1a; border: 1px solid #3fb950;
    border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #7ee787;
  }

    /* Keep Streamlit header visible so rerun/deploy controls are available */
    #MainMenu, footer { visibility: visible; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS & ENCODER DATA
# ─────────────────────────────────────────────────────────
FEATURES = [
    'price', 'cost_price', 'competitor_price', 'price_diff', 'price_ratio',
    'discount_pct', 'is_discount',
    'category_enc', 'size_enc', 'price_tier_enc',
    'day_of_week', 'month', 'week_of_year', 'quarter',
    'is_weekend', 'is_festive_season',
    'region_tier', 'profit_margin_pct',
]

METRO_REGIONS  = ['MAHARASHTRA','DELHI','KARNATAKA','TAMIL NADU','TELANGANA','WEST BENGAL']
TIER1_REGIONS  = ['UTTAR PRADESH','GUJARAT','RAJASTHAN','ANDHRA PRADESH','HARYANA',
                   'KERALA','MADHYA PRADESH','PUNJAB']

CATEGORIES = ['Kurta','Set','Top','Western Dress','Skirt','Ethnic Dress','Blouse']
SIZES      = ['XS','S','M','L','XL','XXL','XXXL']
REGIONS    = sorted([
    'MAHARASHTRA','KARNATAKA','TELANGANA','TAMIL NADU','UTTAR PRADESH','DELHI',
    'RAJASTHAN','GUJARAT','WEST BENGAL','ANDHRA PRADESH','HARYANA','KERALA',
    'MADHYA PRADESH','PUNJAB','BIHAR','ODISHA','ASSAM','JHARKHAND',
    'UTTARAKHAND','HIMACHAL PRADESH','GOA','CHANDIGARH','CHHATTISGARH',
    'TRIPURA','MANIPUR','MEGHALAYA','NAGALAND','MIZORAM',
    'ARUNACHAL PRADESH','SIKKIM','JAMMU AND KASHMIR','LADAKH',
    'PUDUCHERRY','LAKSHADWEEP','ANDAMAN AND NICOBAR',
    'DADRA AND NAGAR HAVELI','DAMAN AND DIU'
])

# Hard-coded encodings matching training
CAT_ENC  = {'Blouse':0,'Ethnic Dress':1,'Kurta':2,'Set':3,'Skirt':4,'Top':5,'Western Dress':6}
SIZE_ENC = {'L':0,'M':1,'S':2,'XL':3,'XS':4,'XXXL':5,'XXL':6}
TIER_ENC = {'Budget':0,'Mid-range':1,'Premium':2}

# ─────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained demand model. Returns model or None."""
    def _patch_tree_compat(loaded_model):
        """Backfill attrs required by newer sklearn for older pickled tree models."""
        try:
            estimators = getattr(loaded_model, "estimators_", None)
            if estimators is None:
                return loaded_model

            for est in estimators:
                if not hasattr(est, "monotonic_cst"):
                    est.monotonic_cst = None
            return loaded_model
        except Exception:
            # Keep app usable even if compatibility patching fails unexpectedly.
            return loaded_model

    paths = [
        'model_artifacts/demand_model.pkl',
        'demand_model.pkl',
        './model_artifacts/random_forest.pkl',
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                loaded = joblib.load(p)
                loaded = _patch_tree_compat(loaded)
                return loaded, p
            except Exception:
                pass
    return None, None

@st.cache_data(show_spinner=False)
def load_metadata():
    paths = ['model_artifacts/metadata.json', 'metadata.json']
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

model, model_path = load_model()
metadata = load_metadata()

# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def encode_features(category, size, price, cost_price, competitor_price,
                    discount_pct, is_discount, region, sel_date):
    """Build the feature dict required by the model."""
    dt = pd.Timestamp(sel_date)
    price_tier = 'Budget' if price < 400 else ('Mid-range' if price < 800 else 'Premium')

    region_tier = (3 if region in METRO_REGIONS else
                   2 if region in TIER1_REGIONS else 1)

    return {
        'price':             price,
        'cost_price':        cost_price,
        'competitor_price':  competitor_price,
        'price_diff':        price - competitor_price,
        'price_ratio':       price / max(competitor_price, 1),
        'discount_pct':      discount_pct,
        'is_discount':       is_discount,
        'category_enc':      CAT_ENC.get(category, 2),
        'size_enc':          SIZE_ENC.get(size, 1),
        'price_tier_enc':    TIER_ENC.get(price_tier, 1),
        'day_of_week':       dt.dayofweek + 1,
        'month':             dt.month,
        'week_of_year':      int(dt.isocalendar()[1]),
        'quarter':           dt.quarter,
        'is_weekend':        int(dt.dayofweek >= 5),
        'is_festive_season': int(dt.month in [9, 10, 11]),
        'region_tier':       region_tier,
        'profit_margin_pct': (price - cost_price) / price * 100 if price > 0 else 0,
    }

def predict_demand(features_dict):
    """Predict demand using the ML model."""
    return float(predict_demand_batch(pd.DataFrame([features_dict]))[0])

def predict_demand_batch(features_input):
    """Vectorized demand prediction for one or many feature rows."""
    if isinstance(features_input, pd.DataFrame):
        X = features_input.copy()
    else:
        X = pd.DataFrame(features_input)

    X = X[FEATURES]

    if model is None:
        p = X['price'].to_numpy(dtype=float)
        cp = X['competitor_price'].to_numpy(dtype=float)
        disc = X['discount_pct'].to_numpy(dtype=float) / 100.0
        region_tier = X['region_tier'].to_numpy(dtype=float)
        category_enc = X['category_enc'].to_numpy(dtype=float)
        month = X['month'].to_numpy(dtype=int)
        festive = X['is_festive_season'].to_numpy(dtype=int)

        ref = 599.0
        elas = 1.3
        base = 7.0
        fest = np.where(festive > 0, 1.15, 1.0)
        discount_boost = 1.0 + np.clip(disc, 0, 0.50) * 0.7
        comp_factor = (np.maximum(cp, 1.0) / np.maximum(p, 1.0)) ** 0.35
        region_factor = 1.0 + (region_tier - 1.0) * 0.04
        category_factor = 1.0 + (category_enc - 3.0) * 0.015
        season_factor = np.where(np.isin(month, [10, 11, 12]), 1.06,
                                 np.where(np.isin(month, [4, 5, 6]), 0.98, 1.0))

        q = base * (ref / np.maximum(p, 1.0)) ** elas
        q = q * fest * discount_boost * comp_factor * region_factor * category_factor * season_factor
        return np.clip(q, 1, 20)

    preds = model.predict(X)
    return np.clip(preds, 1, 20)

def run_optimizer(base_features, cost_price, current_price,
                  floor_pct=0.70, ceil_pct=1.35, n=60):
    """Simulate price range, return full optimization result."""
    floor = max(cost_price * 1.05, current_price * floor_pct)
    ceil  = current_price * ceil_pct
    prices = np.linspace(floor, ceil, n)

    sim_df = pd.DataFrame([base_features] * len(prices))
    sim_df['price'] = prices
    sim_df['price_diff'] = sim_df['price'] - sim_df['competitor_price']
    sim_df['price_ratio'] = sim_df['price'] / np.maximum(sim_df['competitor_price'], 1)
    sim_df['profit_margin_pct'] = (sim_df['price'] - cost_price) / np.maximum(sim_df['price'], 1) * 100
    sim_df['price_tier_enc'] = np.where(
        sim_df['price'] < 400,
        TIER_ENC['Budget'],
        np.where(sim_df['price'] < 800, TIER_ENC['Mid-range'], TIER_ENC['Premium'])
    )

    demands_arr = predict_demand_batch(sim_df)
    profits_arr = (prices - cost_price) * demands_arr
    best_idx    = int(np.argmax(profits_arr))

    # Current metrics
    curr_d = predict_demand(base_features)
    curr_p = (current_price - cost_price) * curr_d

    improvement = (profits_arr[best_idx] - curr_p) / max(abs(curr_p), 1) * 100

    return {
        'optimal_price':   round(float(prices[best_idx]), 2),
        'max_profit':      round(float(profits_arr[best_idx]), 2),
        'current_demand':  round(curr_d, 1),
        'optimal_demand':  round(float(demands_arr[best_idx]), 1),
        'current_profit':  round(float(curr_p), 2),
        'improvement_pct': round(float(improvement), 1),
        'price_range':     prices.tolist(),
        'profit_curve':    profits_arr.tolist(),
        'demand_curve':    demands_arr.tolist(),
    }

# ─────────────────────────────────────────────────────────
# AI RECOMMENDATION (Groq)
# ─────────────────────────────────────────────────────────
def get_ai_recommendation(inputs: dict, result: dict, groq_api_key: str) -> str:
    """Call Groq LLM to generate business recommendation."""
    if not groq_api_key:
        return _fallback_recommendation(inputs, result)

    prompt = f"""You are PriceOracle, an expert AI pricing strategist for Amazon India fashion retail.

PRODUCT CONTEXT:
- Category: {inputs['category']} | Size: {inputs['size']} | Region: {inputs['region']}
- Current Price: ₹{inputs['current_price']} | Cost Price: ₹{inputs['cost_price']}
- Competitor Price: ₹{inputs['competitor_price']} | Discount: {inputs['discount_pct']}%
- Date: {inputs['sel_date']} | Festive Season: {'Yes 🎉' if inputs['is_festive'] else 'No'}

ML OPTIMIZATION RESULT:
- Predicted Demand (current): {result['current_demand']} units
- Predicted Demand (optimal): {result['optimal_demand']} units
- Current Profit: ₹{result['current_profit']}
- Optimal Price: ₹{result['optimal_price']}
- Max Profit: ₹{result['max_profit']}
- Profit Improvement: +{result['improvement_pct']}%

Write a concise 3-4 sentence business recommendation. Cover:
1. Whether to raise or lower the price, and by how much
2. Why (demand sensitivity, competition, seasonality)
3. One actionable tactic (discount strategy, bundling, regional pricing, etc.)

Tone: confident, data-driven, professional. Use ₹ for currency. No bullet points — flowing prose only."""

    try:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.6
        }
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=15
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return _fallback_recommendation(inputs, result)
    except Exception:
        return _fallback_recommendation(inputs, result)

def _fallback_recommendation(inputs, result):
    """Rule-based fallback when Groq is unavailable."""
    direction = "increase" if result['optimal_price'] > inputs['current_price'] else "decrease"
    change_pct = abs(result['optimal_price'] - inputs['current_price']) / inputs['current_price'] * 100

    price_pos = "below" if inputs['current_price'] < inputs['competitor_price'] else "above"
    comp_gap  = abs(inputs['current_price'] - inputs['competitor_price'])

    season_note = (
        "With the festive season active, demand elasticity is lower — consumers are more willing to pay."
        if inputs.get('is_festive') else
        "Outside the festive window, price sensitivity is elevated — competitive positioning matters more."
    )

    discount_note = (
        f"Your current {inputs['discount_pct']}% discount signals value; "
        "consider testing the optimised price alongside a reduced discount to protect margin."
        if inputs['discount_pct'] > 10 else
        "Adding a targeted discount (10–15%) at the optimised price point could further stimulate volume."
    )

    return (
        f"PriceOracle recommends a {change_pct:.1f}% {direction} to ₹{result['optimal_price']:.0f}, "
        f"which is projected to improve profit by {result['improvement_pct']:.1f}%. "
        f"Your price is currently ₹{comp_gap:.0f} {price_pos} competitor, "
        f"giving you room to {'capture more margin' if direction == 'increase' else 'gain volume at competitive parity'}. "
        f"{season_note} {discount_note}"
    )

# ─────────────────────────────────────────────────────────
# PLOTLY CHART HELPERS
# ─────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,27,34,0.8)',
    font=dict(family='Space Grotesk', color='#c9d1d9', size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
    yaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
)

def profit_curve_chart(result, current_price):
    prices  = result['price_range']
    profits = result['profit_curve']
    opt_p   = result['optimal_price']
    opt_pr  = result['max_profit']
    curr_pr = result['current_profit']

    fig = go.Figure()

    # Gradient fill
    fig.add_trace(go.Scatter(
        x=prices, y=profits,
        fill='tozeroy', fillcolor='rgba(63,185,80,0.1)',
        line=dict(color='#3fb950', width=2.5),
        name='Profit Curve', hovertemplate='Price: ₹%{x:.0f}<br>Profit: ₹%{y:.2f}<extra></extra>'
    ))

    # Optimal marker
    fig.add_vline(x=opt_p, line_dash='dash', line_color='#3fb950', line_width=1.5)
    fig.add_trace(go.Scatter(
        x=[opt_p], y=[opt_pr],
        mode='markers+text',
        marker=dict(size=14, color='#3fb950', symbol='star'),
        text=[f'  ₹{opt_p:.0f}'], textfont=dict(color='#3fb950', size=11),
        name='Optimal', hovertemplate=f'Optimal: ₹{opt_p:.0f}<br>Profit: ₹{opt_pr:.0f}<extra></extra>'
    ))

    # Current marker
    fig.add_vline(x=current_price, line_dash='dot', line_color='#d29922', line_width=1.5)
    fig.add_trace(go.Scatter(
        x=[current_price], y=[curr_pr],
        mode='markers+text',
        marker=dict(size=12, color='#d29922', symbol='circle'),
        text=[f'  ₹{current_price:.0f}'], textfont=dict(color='#d29922', size=11),
        name='Current', hovertemplate=f'Current: ₹{current_price:.0f}<br>Profit: ₹{curr_pr:.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text='Price → Profit Curve', font=dict(size=13, color='#e6edf3')),
        xaxis_title='Price (₹)',
        yaxis_title='Expected Profit (₹)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)', bordercolor='#30363d'),
        **PLOT_LAYOUT
    )
    return fig

def demand_curve_chart(result, current_price):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result['price_range'], y=result['demand_curve'],
        fill='tozeroy', fillcolor='rgba(88,166,255,0.1)',
        line=dict(color='#58a6ff', width=2.5),
        name='Demand', hovertemplate='Price: ₹%{x:.0f}<br>Demand: %{y:.1f} units<extra></extra>'
    ))
    fig.add_vline(x=result['optimal_price'], line_dash='dash', line_color='#3fb950', line_width=1.5)
    fig.add_vline(x=current_price, line_dash='dot', line_color='#d29922', line_width=1.5)
    fig.update_layout(
        title=dict(text='Price → Demand Curve', font=dict(size=13, color='#e6edf3')),
        xaxis_title='Price (₹)',
        yaxis_title='Predicted Demand (units)',
        **PLOT_LAYOUT
    )
    return fig

def comparison_bar_chart(result, current_price):
    labels = ['Current Price', 'Optimal Price']
    profits = [result['current_profit'], result['max_profit']]
    demands = [result['current_demand'], result['optimal_demand']]
    prices  = [current_price, result['optimal_price']]
    colors  = ['#d29922', '#3fb950']

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['Price (₹)', 'Predicted Demand', 'Profit (₹)'])

    for col, (vals, fmt) in enumerate(zip([prices, demands, profits], ['₹{:.0f}', '{:.1f}', '₹{:.0f}']), 1):
        fig.add_trace(go.Bar(
            x=labels, y=vals,
            marker_color=colors, text=[fmt.format(v) for v in vals],
            textposition='outside', textfont=dict(color='#e6edf3', size=12),
            showlegend=False,
            hovertemplate='%{x}: %{text}<extra></extra>'
        ), row=1, col=col)

    fig.update_layout(
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(22,27,34,0.8)',
        font=dict(family='Space Grotesk', color='#c9d1d9', size=11),
        margin=dict(l=30, r=20, t=50, b=30),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor='#21262d', row=1, col=i)
        fig.update_yaxes(gridcolor='#21262d', zerolinecolor='#21262d', row=1, col=i)

    return fig

def scenario_heatmap(base_features, cost_price, current_price):
    """Profit heatmap across competitor_price × discount_pct grid."""
    comp_prices = np.linspace(
        current_price * 0.75, current_price * 1.30, 12
    )
    discounts = np.linspace(0, 40, 10)

    z_matrix = []
    for disc in discounts:
        row_profits = []
        for cp in comp_prices:
            f = base_features.copy()
            f['competitor_price'] = cp
            f['discount_pct']     = disc
            f['is_discount']      = int(disc > 0)
            f['price_diff']       = current_price - cp
            f['price_ratio']      = current_price / max(cp, 1)
            d = predict_demand(f)
            row_profits.append(round((current_price - cost_price) * d, 1))
        z_matrix.append(row_profits)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=[f'₹{p:.0f}' for p in comp_prices],
        y=[f'{d:.0f}%' for d in discounts],
        colorscale='RdYlGn',
        hovertemplate='Competitor: %{x}<br>Discount: %{y}<br>Profit: ₹%{z}<extra></extra>',
        colorbar=dict(title='Profit (₹)', tickfont=dict(color='#c9d1d9'))
    ))
    fig.update_layout(
        title=dict(text='Profit Sensitivity: Competitor Price × Discount', font=dict(size=12, color='#e6edf3')),
        xaxis_title='Competitor Price',
        yaxis_title='Your Discount %',
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(22,27,34,0.8)',
        font=dict(family='Space Grotesk', color='#c9d1d9', size=10),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-size:2.5rem;'>🔮</div>
      <div style='font-size:1.2rem; font-weight:700; color:#58a6ff; letter-spacing:-0.02em;'>PriceOracle</div>
      <div style='font-size:0.7rem; color:#8b949e; letter-spacing:0.08em;'>AI PRICING INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # App status (without exposing backend details)
    if model:
        st.markdown("""
        <div class='success-box'>
          ✅ <strong>Pricing Engine Ready</strong><br>
          <span style='font-size:0.75rem; opacity:0.8;'>Live optimisation is enabled.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box'>
          ⚠️ <strong>Simulation Mode</strong><br>
          <span style='font-size:0.75rem; opacity:0.8;'>You can still test pricing flows and scenarios.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;'>🤖 AI Insights</div>", unsafe_allow_html=True)
    env_groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if "groq_connected" not in st.session_state:
        st.session_state.groq_connected = bool(env_groq_key)
    selected_key = env_groq_key

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Connect AI", use_container_width=True):
            if selected_key:
                st.session_state.groq_connected = True
            else:
                st.session_state.groq_connected = False
                st.warning("No key found. Add GROQ_API_KEY to your .env file.")
    with c2:
        if st.button("Disconnect", use_container_width=True):
            st.session_state.groq_connected = False

    groq_key = selected_key if st.session_state.groq_connected and selected_key else ""
    status_text = "Connected" if groq_key else "Not connected"
    status_color = "#3fb950" if groq_key else "#f85149"
    st.markdown(
        f"<div style='font-size:0.8rem; color:{status_color}; margin-top:0.35rem;'>AI status: <strong>{status_text}</strong></div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;'>⚙️ Optimizer Settings</div>", unsafe_allow_html=True)
    price_floor = st.slider("Price Floor (% of current)", 50, 90, 70,
                            help="Minimum price as % of current price")
    price_ceil  = st.slider("Price Ceiling (% of current)", 110, 200, 135,
                            help="Maximum price as % of current price")
    n_simulations = st.slider("Simulation Steps", 20, 100, 60,
                              help="More steps = smoother curve but slower")

    if st.button("Reset All App State", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("App state reset. Reloading...")
        st.rerun()

    st.markdown("---")

# ─────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='app-title'>PriceOracle</div>
    <div class='app-subtitle'>Dynamic Pricing and Profit Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab_optimize, tab_ai, tab_scenario, tab_batch, tab_about = st.tabs([
    "Price Optimizer", "AI Insights", "Scenario Lab", "Batch Analysis", "About"
])

# ══════════════════════════════════════════════════════════
# TAB 1: PRICE OPTIMIZER
# ══════════════════════════════════════════════════════════
with tab_optimize:

    col_input, col_results = st.columns([1.1, 1.9], gap="large")

    # ── INPUT PANEL ──────────────────────────────────────
    with col_input:
        st.markdown("<div class='section-header'>Product Inputs</div>", unsafe_allow_html=True)

        category = st.selectbox("Category", CATEGORIES, index=0)
        size     = st.selectbox("Size", SIZES, index=3)
        region   = st.selectbox("Region", REGIONS, index=0)
        sel_date = st.date_input("Date", value=date(2022, 10, 15))

        st.markdown("<div class='section-header' style='margin-top:1rem;'>Pricing Inputs</div>", unsafe_allow_html=True)

        current_price     = st.number_input("Your Price (₹)", min_value=100.0, max_value=5000.0,
                                            value=799.0, step=50.0)
        cost_price_input  = st.number_input("Cost Price (₹)", min_value=50.0, max_value=4000.0,
                                            value=480.0, step=10.0)
        competitor_price  = st.number_input("Competitor Price (₹)", min_value=100.0, max_value=5000.0,
                                            value=849.0, step=50.0)
        discount_pct      = st.slider("Discount (%)", 0.0, 60.0, 15.0, 0.5)
        is_discount       = int(discount_pct > 0)

        # Quick validation
        margin = (current_price - cost_price_input) / current_price * 100
        if cost_price_input >= current_price:
            st.markdown("<div class='warning-box'>⚠️ Cost price ≥ selling price — you're selling at a loss!</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='success-box'>✅ Gross margin: {margin:.1f}%</div>",
                        unsafe_allow_html=True)

        run_btn = st.button("Optimise Price", use_container_width=True, type="primary")

    # ── RESULTS ──────────────────────────────────────────
    with col_results:
        is_festive = int(pd.Timestamp(sel_date).month in [9, 10, 11])
        features = encode_features(
            category, size, current_price, cost_price_input,
            competitor_price, discount_pct, is_discount, region, sel_date
        )

        opt_signature = (
            category, size, region, str(sel_date),
            float(current_price), float(cost_price_input), float(competitor_price),
            float(discount_pct), int(is_discount), int(price_floor), int(price_ceil), int(n_simulations)
        )

        if run_btn or 'optimizer_state' not in st.session_state:
            result = run_optimizer(
                features, cost_price_input, current_price,
                price_floor/100, price_ceil/100, n_simulations
            )

            ai_inputs = {
                'category': category, 'size': size, 'region': region,
                'current_price': current_price, 'cost_price': cost_price_input,
                'competitor_price': competitor_price, 'discount_pct': discount_pct,
                'sel_date': sel_date, 'is_festive': bool(is_festive)
            }

            st.session_state['optimizer_state'] = {
                'signature': opt_signature,
                'result': result,
                'ai_inputs': ai_inputs
            }

        optimizer_state = st.session_state.get('optimizer_state')
        if optimizer_state:
            result = optimizer_state['result']

            if optimizer_state.get('signature') != opt_signature:
                st.info("Inputs changed. Click 'Optimise Price' to refresh results.")

            # ── Section 3: Baseline Metrics ──────────────
            st.markdown("<div class='section-header'>Baseline Metrics</div>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)

            with mc1:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='label'>Predicted Demand</div>
                  <div class='value blue'>{result['current_demand']:.1f}</div>
                  <div class='delta white'>units</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='label'>Current Profit</div>
                  <div class='value gold'>₹{result['current_profit']:,.0f}</div>
                  <div class='delta white'>per cycle</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                comp_diff = current_price - competitor_price
                pos_color = "red" if comp_diff > 0 else "green"
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='label'>vs Competitor</div>
                  <div class='value {pos_color}'>{"+" if comp_diff >= 0 else ""}₹{comp_diff:.0f}</div>
                  <div class='delta white'>{'pricier' if comp_diff > 0 else 'cheaper'}</div>
                </div>""", unsafe_allow_html=True)
            with mc4:
                fest_label = "🎉 YES" if is_festive else "❌ NO"
                fest_color = "green" if is_festive else "red"
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='label'>Festive Season</div>
                  <div class='value {fest_color}'>{fest_label}</div>
                  <div class='delta white'>{pd.Timestamp(sel_date).strftime("%b %Y")}</div>
                </div>""", unsafe_allow_html=True)

            # ── Section 4: Optimal Price Banner ──────────
            st.markdown("<div class='section-header' style='margin-top:1.2rem;'>Optimisation Result</div>",
                        unsafe_allow_html=True)

            imp_color = "green" if result['improvement_pct'] >= 0 else "red"
            imp_sign  = "+" if result['improvement_pct'] >= 0 else ""
            price_change = result['optimal_price'] - current_price
            pc_sign      = "▲" if price_change >= 0 else "▼"

            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.markdown(f"""
                <div class='optimal-banner'>
                  <div class='opt-label'>Optimal Price</div>
                  <div class='opt-price'>₹{result['optimal_price']:,.0f}</div>
                  <div class='opt-sub'>{pc_sign} ₹{abs(price_change):.0f} from current</div>
                </div>""", unsafe_allow_html=True)
            with oc2:
                st.markdown(f"""
                <div class='metric-card' style='height:100%; display:flex; flex-direction:column; justify-content:center;'>
                  <div class='label'>Max Profit</div>
                  <div class='value green'>₹{result['max_profit']:,.0f}</div>
                  <div class='delta white'>Demand: {result['optimal_demand']:.1f} units</div>
                </div>""", unsafe_allow_html=True)
            with oc3:
                st.markdown(f"""
                <div class='metric-card' style='height:100%; display:flex; flex-direction:column; justify-content:center;'>
                  <div class='label'>Profit Improvement</div>
                  <div class='value {imp_color}'>{imp_sign}{result['improvement_pct']:.1f}%</div>
                  <div class='delta white'>vs current strategy</div>
                </div>""", unsafe_allow_html=True)

            # ── Section 5: Visualisations ─────────────────
            st.markdown(
                "<div class='section-header' style='margin-top:1.2rem;'>Optimization Curves</div>",
                unsafe_allow_html=True
            )

            vc1, vc2 = st.columns(2)
            with vc1:
                st.plotly_chart(
                    profit_curve_chart(result, current_price),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )
            with vc2:
                st.plotly_chart(
                    demand_curve_chart(result, current_price),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )

            st.plotly_chart(
                comparison_bar_chart(result, current_price),
                use_container_width=True,
                config={'displayModeBar': False}
            )

            # ── Section 6: Business Insights ─────────────
            st.markdown(
                "<div class='section-header' style='margin-top:1rem;'>Business Insights</div>",
                unsafe_allow_html=True
            )

            elasticity_est = abs(
                (result['optimal_demand'] - result['current_demand'])
                / max(result['current_demand'], 0.1)
                / ((result['optimal_price'] - current_price) / max(current_price, 1) + 1e-9)
            )
            price_position = (
                "💚 Below competitor — opportunity to raise price"
                if current_price < competitor_price * 0.95
                else "🔴 Above competitor — risk of demand loss"
                if current_price > competitor_price * 1.05
                else "⚖️ At parity with competitor"
            )
            disc_note = (
                f"📉 Your {discount_pct:.0f}% discount is aggressive — margin is squeezed"
                if discount_pct > 30
                else f"✅ Discount of {discount_pct:.0f}% is balanced — good for conversion"
                if discount_pct > 0
                else "🏷️ No discount applied — room to add tactical discount"
            )

            pills = [
                f"📊 Price elasticity ≈ {elasticity_est:.2f}",
                price_position,
                disc_note,
                (
                    "🗓️ Festive boost active +15% demand"
                    if is_festive
                    else "🗓️ Off-peak — monitor competitors closely"
                ),
                (
                    f"🗺️ Region tier: {'Metro' if region in METRO_REGIONS else 'Tier 1' if region in TIER1_REGIONS else 'Tier 2/3'}"
                ),
                (
                    f"💰 Gross margin at optimal: {((result['optimal_price'] - cost_price_input) / result['optimal_price'] * 100):.1f}%"
                ),
            ]
            for pill in pills:
                st.markdown(f"<span class='insight-pill'>{pill}</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2: AI INSIGHTS
# ══════════════════════════════════════════════════════════
with tab_ai:
    st.subheader("AI Recommendation")
    st.caption("Generate recommendation from the latest optimizer analysis.")

    optimizer_state = st.session_state.get('optimizer_state')
    if not optimizer_state:
        st.info("Run Price Optimizer first to prepare analysis.")
    else:
        signature = optimizer_state['signature']
        saved_ai_state = st.session_state.get('ai_state', {})

        if saved_ai_state.get('signature') != signature:
            st.info("New analysis available. Click Generate Recommendation.")

        if st.button("Generate Recommendation", type="primary"):
            with st.spinner("Generating recommendation..."):
                ai_text = get_ai_recommendation(
                    optimizer_state['ai_inputs'], optimizer_state['result'], groq_key
                )
            st.session_state['ai_state'] = {
                'signature': signature,
                'ai_text': ai_text
            }

        current_ai_state = st.session_state.get('ai_state', {})
        if current_ai_state.get('ai_text'):
            st.markdown(f"""
            <div class='ai-box'>
              <div class='ai-label'>PriceOracle Recommendation</div>
              <div class='ai-text'>{current_ai_state['ai_text']}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 3: SCENARIO LAB
# ══════════════════════════════════════════════════════════
with tab_scenario:
    st.markdown("<div class='section-header'>Scenario Simulation — What If Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.85rem; color:#8b949e; margin-bottom:1rem;'>"
        "Adjust competitor price and discount to see how optimal profit changes.</div>",
        unsafe_allow_html=True
    )

    sc1, sc2 = st.columns([1, 2])

    with sc1:
        st.markdown("<div class='section-header'>Scenario Inputs</div>", unsafe_allow_html=True)
        sc_price      = st.number_input("Your Price (₹)", 100.0, 5000.0, 799.0, 50.0, key='sc_price')
        sc_cost       = st.number_input("Cost Price (₹)", 50.0, 4000.0, 480.0, 10.0, key='sc_cost')
        sc_cat        = st.selectbox("Category", CATEGORIES, key='sc_cat')
        sc_size       = st.selectbox("Size", SIZES, index=3, key='sc_size')
        sc_region     = st.selectbox("Region", REGIONS, key='sc_region')
        sc_date       = st.date_input("Date", value=date(2022, 10, 15), key='sc_date')

        st.markdown("<div class='section-header' style='margin-top:0.8rem;'>Scenario Variables</div>", unsafe_allow_html=True)
        sc_comp_range = st.slider("Competitor Price Range (₹)", 300, 3000, (600, 1200), 50)
        sc_disc_range = st.slider("Discount Range (%)", 0, 60, (0, 40), 5)

        sc_run = st.button("Run Scenario Analysis", use_container_width=True, type="primary")

    with sc2:
        sc_signature = (
            float(sc_price), float(sc_cost), sc_cat, sc_size, sc_region, str(sc_date),
            tuple(sc_comp_range), tuple(sc_disc_range)
        )

        if sc_run or 'scenario_state' not in st.session_state:
            sc_features = encode_features(
                sc_cat, sc_size, sc_price, sc_cost,
                (sc_comp_range[0] + sc_comp_range[1]) / 2,
                (sc_disc_range[0] + sc_disc_range[1]) / 2,
                1, sc_region, sc_date
            )

            # Heatmap
            comp_prices = np.linspace(sc_comp_range[0], sc_comp_range[1], 12)
            discounts   = np.linspace(sc_disc_range[0], sc_disc_range[1], 10)
            z_matrix = []

            for disc in discounts:
                row_profits = []
                for cp in comp_prices:
                    f = sc_features.copy()
                    f['competitor_price'] = cp
                    f['discount_pct']     = disc
                    f['is_discount']      = int(disc > 0)
                    f['price_diff']       = sc_price - cp
                    f['price_ratio']      = sc_price / max(cp, 1)
                    d = predict_demand(f)
                    row_profits.append(round((sc_price - sc_cost) * d, 1))
                z_matrix.append(row_profits)

            # Profit vs competitor price line
            mid_disc = (sc_disc_range[0] + sc_disc_range[1]) / 2
            profits_line = []
            for cp in comp_prices:
                f = sc_features.copy()
                f['competitor_price'] = cp
                f['discount_pct'] = mid_disc
                f['is_discount'] = int(mid_disc > 0)
                f['price_diff'] = sc_price - cp
                f['price_ratio'] = sc_price / max(cp, 1)
                d = predict_demand(f)
                profits_line.append((sc_price - sc_cost) * d)

            st.session_state['scenario_state'] = {
                'signature': sc_signature,
                'comp_prices': comp_prices.tolist(),
                'discounts': discounts.tolist(),
                'z_matrix': z_matrix,
                'profits_line': profits_line,
                'mid_disc': float(mid_disc)
            }

        scenario_state = st.session_state.get('scenario_state')
        if scenario_state:
            if scenario_state.get('signature') != sc_signature:
                st.info("Scenario inputs changed. Click 'Run Scenario Analysis' to refresh charts.")

            comp_prices = np.array(scenario_state['comp_prices'])
            discounts = np.array(scenario_state['discounts'])
            z_matrix = scenario_state['z_matrix']
            profits_line = scenario_state['profits_line']
            mid_disc = scenario_state['mid_disc']

            heat_fig = go.Figure(data=go.Heatmap(
                z=z_matrix,
                x=[f'₹{p:.0f}' for p in comp_prices],
                y=[f'{d:.0f}%' for d in discounts],
                colorscale='RdYlGn',
                hovertemplate='Competitor: %{x}<br>Discount: %{y}<br>Profit: ₹%{z}<extra></extra>',
                colorbar=dict(title='Profit (₹)', tickfont=dict(color='#c9d1d9'))
            ))
            heat_fig.update_layout(
                title=dict(text='Profit Heatmap: Competitor Price × Your Discount', font=dict(size=13, color='#e6edf3')),
                xaxis_title='Competitor Price (₹)',
                yaxis_title='Your Discount (%)',
                height=380,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(22,27,34,0.8)',
                font=dict(family='Space Grotesk', color='#c9d1d9', size=10),
                margin=dict(l=50, r=20, t=50, b=50),
            )
            st.plotly_chart(heat_fig, use_container_width=True, config={'displayModeBar': False})

            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(
                x=comp_prices, y=profits_line,
                mode='lines+markers', line=dict(color='#bc8cff', width=2.5),
                fill='tozeroy', fillcolor='rgba(188,140,255,0.1)',
                hovertemplate='Competitor: ₹%{x:.0f}<br>Profit: ₹%{y:.0f}<extra></extra>'
            ))
            line_fig.update_layout(
                title=dict(text=f'Profit vs Competitor Price (at {mid_disc:.0f}% discount)', font=dict(size=12, color='#e6edf3')),
                xaxis_title='Competitor Price (₹)',
                yaxis_title='Your Profit (₹)',
                height=280,
                **PLOT_LAYOUT
            )
            st.plotly_chart(line_fig, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════
# TAB 4: BATCH ANALYSIS
# ══════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Product Analysis")
    st.caption("Generate a product list to find optimal prices across your catalog.")

    bt1, bt2 = st.columns([1, 2])
    with bt1:
        st.markdown("<div class='section-header'>Generate Sample Products</div>", unsafe_allow_html=True)
        n_products = st.slider("Number of Products", 5, 50, 15)
        batch_cat  = st.multiselect("Categories", CATEGORIES, default=['Kurta', 'Set', 'Top'])
        batch_date = st.date_input("Analysis Date", value=date(2022, 10, 20), key='batch_date')
        gen_btn    = st.button("Generate and Optimise", use_container_width=True, type="primary")

    with bt2:
        if gen_btn or 'batch_results_df' in st.session_state:
            if gen_btn or 'batch_results_df' not in st.session_state:
                cats_to_use = batch_cat if batch_cat else CATEGORIES
                batch_rows = []

                for _ in range(n_products):
                    cat   = np.random.choice(cats_to_use)
                    size  = np.random.choice(SIZES)
                    price = float(np.random.choice([299, 399, 499, 599, 799, 999, 1199, 1499]))
                    cost  = round(price * np.random.uniform(0.55, 0.70), 2)
                    comp  = round(price * np.random.uniform(0.80, 1.25), 0) - 1
                    disc  = round(np.random.uniform(0, 40), 1)
                    reg   = np.random.choice(REGIONS[:10])

                    f = encode_features(cat, size, price, cost, comp, disc, int(disc > 0), reg, batch_date)
                    res = run_optimizer(f, cost, price, 0.70, 1.35, 40)

                    batch_rows.append({
                        'Category':       cat,
                        'Size':           size,
                        'Region':         reg,
                        'Current Price':  price,
                        'Cost Price':     cost,
                        'Competitor ₹':   comp,
                        'Discount %':     disc,
                        'Optimal Price':  res['optimal_price'],
                        'Price Change ₹': round(res['optimal_price'] - price, 2),
                        'Change %':       round((res['optimal_price'] - price) / price * 100, 1),
                        'Curr Profit':    res['current_profit'],
                        'Opt Profit':     res['max_profit'],
                        'Improvement %':  res['improvement_pct'],
                    })

                st.session_state['batch_results_df'] = pd.DataFrame(batch_rows)

            bdf = st.session_state['batch_results_df']

            # Summary KPIs
            kc1, kc2, kc3, kc4 = st.columns(4)
            avg_imp = bdf['Improvement %'].mean()
            up_cnt  = (bdf['Price Change ₹'] > 0).sum()
            dn_cnt  = (bdf['Price Change ₹'] < 0).sum()
            total_profit_gain = bdf['Opt Profit'].sum() - bdf['Curr Profit'].sum()

            with kc1:
                st.markdown(f"<div class='metric-card'><div class='label'>Avg Improvement</div><div class='value green'>+{avg_imp:.1f}%</div></div>",
                            unsafe_allow_html=True)
            with kc2:
                st.markdown(f"<div class='metric-card'><div class='label'>Price Up ↑</div><div class='value blue'>{up_cnt}</div><div class='delta white'>products</div></div>",
                            unsafe_allow_html=True)
            with kc3:
                st.markdown(f"<div class='metric-card'><div class='label'>Price Down ↓</div><div class='value gold'>{dn_cnt}</div><div class='delta white'>products</div></div>",
                            unsafe_allow_html=True)
            with kc4:
                st.markdown(f"<div class='metric-card'><div class='label'>Total Profit Gain</div><div class='value green'>₹{total_profit_gain:,.0f}</div></div>",
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Improvement chart
            bdf_sorted = bdf.sort_values('Improvement %', ascending=False).reset_index(drop=True)
            colors_bar = ['#3fb950' if x >= 0 else '#f85149' for x in bdf_sorted['Improvement %']]

            imp_fig = go.Figure(go.Bar(
                x=bdf_sorted.index,
                y=bdf_sorted['Improvement %'],
                marker_color=colors_bar,
                hovertemplate=(
                    'Category: %{customdata[0]}<br>'
                    'Current: ₹%{customdata[1]}<br>'
                    'Optimal: ₹%{customdata[2]}<br>'
                    'Improvement: +%{y:.1f}%<extra></extra>'
                ),
                customdata=bdf_sorted[['Category', 'Current Price', 'Optimal Price']].values,
            ))
            imp_fig.update_layout(
                title=dict(text='Profit Improvement % per Product', font=dict(size=12, color='#e6edf3')),
                xaxis_title='Product Index',
                yaxis_title='Improvement %',
                height=250, **PLOT_LAYOUT
            )
            st.plotly_chart(imp_fig, use_container_width=True, config={'displayModeBar': False})

            # Table
            st.dataframe(
                bdf_sorted.style.background_gradient(subset=['Improvement %'], cmap='RdYlGn')
                          .format({'Current Price': '₹{:.0f}', 'Cost Price': '₹{:.0f}',
                                   'Competitor ₹': '₹{:.0f}', 'Optimal Price': '₹{:.2f}',
                                   'Price Change ₹': '₹{:.2f}', 'Change %': '{:.1f}%',
                                   'Curr Profit': '₹{:.0f}', 'Opt Profit': '₹{:.0f}',
                                   'Improvement %': '{:.1f}%'}),
                use_container_width=True, height=300
            )

            csv = bdf.to_csv(index=False)
            st.download_button("Download Results CSV", csv, "priceoracle_batch.csv", "text/csv")
        else:
            st.info("Click 'Generate & Optimise' to run batch analysis and show charts/table.")
            st.markdown(
                "This section will display:\n"
                "- Summary metrics\n"
                "- Improvement chart\n"
                "- Full product recommendation table"
            )


# ══════════════════════════════════════════════════════════
# TAB 5: ABOUT
# ══════════════════════════════════════════════════════════
with tab_about:
        st.subheader("ℹ️ About PriceOracle")
        st.write(
                "PriceOracle helps you test and optimize product prices by balancing demand and profitability. "
                "Use it for single product decisions, scenario testing, and batch pricing recommendations."
        )

        c1, c2 = st.columns(2)
        with c1:
                st.markdown("**What You Can Do**")
                st.markdown("- Optimize one product price")
                st.markdown("- Run competitor and discount scenarios")
                st.markdown("- Compare current vs optimized profit")
        with c2:
                st.markdown("**Batch Analysis**")
                st.markdown("- Generate recommendations for many products")
                st.markdown("- Export results to CSV")
                st.markdown("- Prioritize high-uplift products")

        st.markdown("**Usage**")
        st.markdown("Use the tabs to optimize a product, run scenarios, and generate batch recommendations.")
