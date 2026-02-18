"""
Utility functions for data loading, preprocessing and aggregation.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame with parsing and basic validation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(p)
    df = preprocess_df(df)
    return df


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and ensure numeric types.

    Keeps the dataframe robust to missing columns and casts types.
    """
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    # Ensure numeric columns exist and are numeric
    num_cols = ['daily_revenue', 'daily_expense', 'net_cash_flow',
                'gst_compliance_score', 'utility_payment_consistency',
                'ecommerce_return_rate', 'social_sentiment_score']
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df


def monthly_aggregate(df: pd.DataFrame, business_id: str) -> pd.DataFrame:
    """Return monthly aggregation (sum/mean) for a given business.

    Useful for credit calculations (avg monthly net cash flow, revenue, expenses).
    """
    d = df[df['business_id'] == business_id].copy()
    if d.empty:
        return pd.DataFrame()
    d['month'] = d['date'].dt.to_period('M')
    agg = d.groupby('month').agg(
        monthly_revenue=('daily_revenue', 'sum'),
        monthly_expenses=('daily_expense', 'sum'),
        monthly_net=('net_cash_flow', 'sum'),
        gst_mean=('gst_compliance_score', 'mean'),
        util_mean=('utility_payment_consistency', 'mean'),
        return_rate_mean=('ecommerce_return_rate', 'mean'),
        sentiment_mean=('social_sentiment_score', 'mean')
    ).reset_index()
    agg['month'] = agg['month'].dt.to_timestamp()
    return agg


def normalize_series(s: pd.Series, clip_min=0.0, clip_max=1.0) -> pd.Series:
    """Min-max normalize a Pandas series to [clip_min, clip_max]."""
    if s.empty:
        return s
    minv = s.min()
    maxv = s.max()
    if maxv == minv:
        return pd.Series(np.full(len(s), clip_min), index=s.index)
    norm = (s - minv) / (maxv - minv)
    return norm.clip(clip_min, clip_max)


def safe_divide(a, b, default=0.0):
    try:
        return a / b if b != 0 else default
    except Exception:
        return default
