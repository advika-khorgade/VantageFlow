"""
VantageFlow – Streamlit dashboard for SME credit intelligence.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

This app reads the CSV in `data/` and shows scores, limits, runway, and charts.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from credit_engine import compute_full_profile, detect_anomaly_spike, composite_risk_score
from utils import load_data, monthly_aggregate
from model import build_and_train, SimpleRiskModel


st.set_page_config(page_title='VantageFlow – SME Credit Intelligence', layout='wide')


@st.cache_data
def load_dataset():
    base = Path(__file__).parent
    data_path = base / 'data' / 'vantageflow_sme_dataset.csv'
    return load_data(str(data_path))


@st.cache_data
def train_model(df):
    m = build_and_train(df)
    return m


def kpi_columns():
    return st.columns([1.2, 1.2, 1.2, 1.2, 1.2])


def plot_revenue_trend(df, business_id):
    d = df[df['business_id'] == business_id].sort_values('date')
    if d.empty:
        st.write('No data to plot')
        return
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(d['date'], d['daily_revenue'], marker='o', linewidth=1.5)
    ax.set_title('Daily Revenue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    ax.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig)


def plot_net_cash_flow(df, business_id):
    d = df[df['business_id'] == business_id].sort_values('date')
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(d['date'], d['net_cash_flow'], color='tab:green')
    ax.set_title('Daily Net Cash Flow')
    ax.set_xlabel('Date')
    ax.set_ylabel('Net Cash Flow')
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)


def plot_score_gauge(score: float):
    # Try Plotly gauge first; fallback to a simple Matplotlib bar if Plotly not installed
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': 'darkblue'},
                   'steps': [
                       {'range': [0, 50], 'color': 'red'},
                       {'range': [50, 75], 'color': 'orange'},
                       {'range': [75, 100], 'color': 'green'}
                   ]},
            number={'suffix': ''}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Fallback: simple horizontal bar representing score
        fig, ax = plt.subplots(figsize=(6, 0.8))
        ax.barh([0], [score], color='tab:blue')
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel('Score')
        ax.set_title(f'Risk Score: {score:.1f}')
        st.pyplot(fig)


def set_theme(mode: str):
    if mode == 'Dark':
        st.markdown("""
            <style>
            .reportview-container { background: #0e1117; color: #e6edf3; }
            .stApp { background: #0e1117; color: #e6edf3; }
            .stButton>button { color: #000 }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .reportview-container { background: white; color: #0b1117 }
            .stApp { background: white; color: #0b1117 }
            </style>
        """, unsafe_allow_html=True)


def overview_tab(df, model, selected):
    profile = compute_full_profile(df, selected)
    c1, c2, c3, c4, c5 = kpi_columns()
    score = profile.get('composite_score', 0.0)
    c1.metric('Risk Score', f"{score:.1f}")
    c2.metric('Risk Category', profile.get('risk_category', 'Unknown'))
    c3.metric('Credit Limit', f"{profile.get('credit_limit', 0.0):,.0f}")
    c4.metric('Trust Index', f"{profile.get('trust_index', 0.0):.1f}")
    c5.metric('Cash Runway (days)', f"{profile.get('runway_days', 0.0):.1f}")

    # Alerts
    if profile.get('risk_category') == 'High':
        st.error('High Risk: tighten exposure or request collateral')
    if profile.get('runway_days', 0) < 20:
        st.warning('⚠ Bridge Loan Recommended — runway < 20 days')
    an = profile.get('anomaly', {})
    if an.get('anomaly'):
        st.info(f"Anomaly detected: revenue spike {an['spike_value']:.0f} (>3x avg {an['avg']:.0f})")

    st.subheader('Score Gauge')
    plot_score_gauge(score)

    st.subheader('Trends')
    plot_revenue_trend(df, selected)
    plot_net_cash_flow(df, selected)


def details_tab(df, selected):
    st.subheader('Daily Transactions')
    d = df[df['business_id'] == selected].sort_values('date')
    st.dataframe(d.reset_index(drop=True))

    st.subheader('Monthly Aggregation')
    agg = monthly_aggregate(df, selected)
    st.dataframe(agg)
    csv = agg.to_csv(index=False).encode('utf-8')
    st.download_button('Download monthly CSV', data=csv, file_name=f'{selected}_monthly.csv')


def ml_tab(df, model, selected):
    st.subheader('ML Model')
    # Do not display training-summary link here (removed per UX request).
    proba = model.predict_proba(df, selected)
    st.write(f'Estimated default probability (simulated): {proba:.2f}')
    if getattr(model, 'trained', False):
        fi = model.model.feature_importances_
        st.subheader('Feature importances')
        fig, ax = plt.subplots()
        ax.bar(['cash_flow_score', 'trust_index', 'liquidity_score'], fi)
        ax.set_ylabel('Importance')
        st.pyplot(fig)


def settings_tab(theme_mode):
    st.subheader('Settings')
    mode = st.radio('Theme', ['Light', 'Dark'], index=0 if theme_mode == 'Light' else 1)
    set_theme(mode)
    st.write('Theme set to', mode)


def main():
    st.title('VantageFlow – SME Credit Intelligence')
    df = load_dataset()
    model = train_model(df)

    # Sidebar
    st.sidebar.header('Controls')
    # Ensure business list covers at least 15 IDs (pad if data has fewer)
    raw_ids = sorted(df['business_id'].astype(str).unique().tolist())
    if len(raw_ids) < 15:
        # pad with synthetic ids if dataset is small (won't affect scoring for missing data)
        pad_ids = [f'BIZ{str(i).zfill(3)}' for i in range(1, 16)]
        # merge while preserving existing ids first
        merged = []
        for pid in pad_ids:
            if pid in raw_ids:
                merged.append(pid)
            else:
                merged.append(pid)
        business_list = merged
    else:
        business_list = raw_ids

    selected = st.sidebar.selectbox('Select Business ID', business_list)
    st.sidebar.markdown('**Risk Legend**')
    st.sidebar.markdown('- Low: Green')
    st.sidebar.markdown('- Medium: Amber')
    st.sidebar.markdown('- High: Red')
    # Theme control moved to sidebar (settings tab removed)
    theme_mode = st.sidebar.selectbox('Theme', ['Light', 'Dark'])
    set_theme(theme_mode)

    tabs = st.tabs(['Overview', 'Details', 'ML Model'])
    with tabs[0]:
        overview_tab(df, model, selected)
    with tabs[1]:
        details_tab(df, selected)
    with tabs[2]:
        ml_tab(df, model, selected)


if __name__ == '__main__':
    main()
