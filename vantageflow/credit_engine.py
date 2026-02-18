"""
Credit engine: scoring logic, liquidity, runway and anomaly detection.
"""
from typing import Dict
import numpy as np
import pandas as pd
from utils import monthly_aggregate, normalize_series, safe_divide


def cash_flow_stability_score(df: pd.DataFrame, business_id: str) -> float:
    """Compute Cash Flow Stability (0-100).

    Uses mean monthly revenue and volatility and net cash flow consistency.
    Higher mean and lower std -> higher score.
    """
    agg = monthly_aggregate(df, business_id)
    if agg.empty:
        return 0.0
    mean_rev = agg['monthly_revenue'].mean()
    std_rev = agg['monthly_revenue'].std(ddof=0)
    # Net cash flow consistency: fraction of months with non-negative net
    consistency = (agg['monthly_net'] >= 0).mean()

    # Build a base metric combining mean and volatility
    rev_metric = safe_divide(mean_rev, mean_rev + std_rev, default=0.0)
    raw = 0.6 * rev_metric + 0.4 * consistency
    return float(np.clip(raw * 100.0, 0, 100))


def trust_index(df: pd.DataFrame, business_id: str) -> float:
    """Compute Trust Index (0-100) using weighted alternative signals.

    Weights:
       - GST compliance: 30%
       - Utility consistency: 25%
       - (1 - return rate): 20%
       - Normalized social sentiment: 25%
    """
    agg = monthly_aggregate(df, business_id)
    if agg.empty:
        return 0.0
    # take means
    gst = agg['gst_mean'].mean()
    util = agg['util_mean'].mean()
    ret_rate = agg['return_rate_mean'].mean()
    sentiment = agg['sentiment_mean'].mean()
    # normalize sentiment (-1..1 -> 0..1)
    sentiment_norm = (sentiment + 1.0) / 2.0

    score = (0.30 * gst) + (0.25 * util) + (0.20 * (1 - ret_rate)) + (0.25 * sentiment_norm)
    return float(np.clip(score * 100.0, 0, 100))


def liquidity_ratio_score(df: pd.DataFrame, business_id: str) -> float:
    """Compute Liquidity Ratio Score (0-100).

    Liquidity ratio = avg net cash flow / avg expense.
    We map ratio to a 0-100 scale with an upper cap for stability.
    """
    d = df[df['business_id'] == business_id]
    if d.empty:
        return 0.0
    avg_net = d['net_cash_flow'].mean()
    avg_exp = d['daily_expense'].mean()
    ratio = safe_divide(avg_net, avg_exp, default=0.0)
    # Map ratio to 0-100: ratio 0 -> 0, ratio 1 -> 60, ratio >=3 -> 100
    if ratio <= 0:
        mapped = 0.0
    elif ratio >= 3:
        mapped = 100.0
    else:
        mapped = 60.0 * ratio  # linear from 0..3 -> 0..180 then cap
        mapped = min(mapped, 100.0)
    return float(mapped)


def composite_risk_score(df: pd.DataFrame, business_id: str) -> float:
    """Combine the three scores into a normalized 0-100 risk score.

    Weights: 40% Cash Flow, 30% Trust, 30% Liquidity.
    Higher composite -> lower risk (we present as score where higher is better).
    """
    cf = cash_flow_stability_score(df, business_id)
    ti = trust_index(df, business_id)
    liq = liquidity_ratio_score(df, business_id)
    composite = 0.40 * cf + 0.30 * ti + 0.30 * liq
    return float(np.clip(composite, 0, 100))


def risk_category(score: float) -> str:
    if score > 75:
        return 'Low'
    if score >= 50:
        return 'Medium'
    return 'High'


def compute_credit_limit(df: pd.DataFrame, business_id: str) -> float:
    """Dynamic credit limit based on risk category and avg monthly net cash flow."""
    agg = monthly_aggregate(df, business_id)
    if agg.empty:
        return 0.0
    avg_monthly_net = agg['monthly_net'].mean()
    score = composite_risk_score(df, business_id)
    cat = risk_category(score)
    if cat == 'Low':
        credit = 3.0 * avg_monthly_net
    elif cat == 'Medium':
        credit = 1.5 * avg_monthly_net
    else:
        credit = 0.5 * avg_monthly_net
    return float(max(0.0, credit))


def cash_runway_days(df: pd.DataFrame, business_id: str) -> float:
    """Estimate runway days: latest cash balance / avg daily expense.

    Latest cash balance is cumulative net_cash_flow over available period.
    """
    d = df[df['business_id'] == business_id].sort_values('date')
    if d.empty:
        return 0.0
    latest_balance = d['net_cash_flow'].cumsum().iloc[-1]
    avg_daily_exp = d['daily_expense'].mean()
    days = safe_divide(latest_balance, avg_daily_exp, default=0.0)
    return float(max(0.0, days))


def detect_anomaly_spike(df: pd.DataFrame, business_id: str) -> Dict:
    """Detect if any revenue spike > 3x average daily revenue."""
    d = df[df['business_id'] == business_id]
    if d.empty:
        return {'anomaly': False, 'spike_value': 0.0}
    avg = d['daily_revenue'].mean()
    spike = d['daily_revenue'].max()
    is_anom = spike > 3 * avg
    return {'anomaly': bool(is_anom), 'spike_value': float(spike), 'avg': float(avg)}


def compute_full_profile(df: pd.DataFrame, business_id: str) -> Dict:
    """Return a complete profile with all computed metrics for the business."""
    try:
        cf_score = cash_flow_stability_score(df, business_id)
        ti_score = trust_index(df, business_id)
        liq_score = liquidity_ratio_score(df, business_id)
        composite = composite_risk_score(df, business_id)
        category = risk_category(composite)
        credit = compute_credit_limit(df, business_id)
        runway = cash_runway_days(df, business_id)
        anomaly = detect_anomaly_spike(df, business_id)
        return {
            'cash_flow_score': cf_score,
            'trust_index': ti_score,
            'liquidity_score': liq_score,
            'composite_score': composite,
            'risk_category': category,
            'credit_limit': credit,
            'runway_days': runway,
            'anomaly': anomaly
        }
    except Exception as e:
        return {'error': str(e)}
