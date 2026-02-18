"""
Simple ML model to predict default risk using RandomForestClassifier.
This trains on synthetic labels derived from avg monthly net cash flow.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import monthly_aggregate
from credit_engine import cash_flow_stability_score, trust_index, liquidity_ratio_score


class SimpleRiskModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

    def featurize(self, df: pd.DataFrame, business_id: str) -> dict:
        # Build a feature vector using the credit engine metrics
        return {
            'cash_flow_score': cash_flow_stability_score(df, business_id),
            'trust_index': trust_index(df, business_id),
            'liquidity_score': liquidity_ratio_score(df, business_id)
        }

    def train_on_dataset(self, df: pd.DataFrame):
        # Aggregate by business and create synthetic labels: low avg monthly net -> default
        business_ids = df['business_id'].unique()
        rows = []
        for b in business_ids:
            agg = monthly_aggregate(df, b)
            if agg.empty:
                continue
            avg_monthly_net = agg['monthly_net'].mean()
            features = self.featurize(df, b)
            # Synthetic label: default (1) if avg_monthly_net < 0.5 * median
            rows.append({**features, 'avg_monthly_net': avg_monthly_net, 'business_id': b})
        Xdf = pd.DataFrame(rows)
        if Xdf.empty:
            return None
        med = Xdf['avg_monthly_net'].median() if len(Xdf) > 0 else 0
        Xdf['label'] = (Xdf['avg_monthly_net'] < 0.5 * med).astype(int)
        features = ['cash_flow_score', 'trust_index', 'liquidity_score']
        X = Xdf[features].fillna(0)
        y = Xdf['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds) if len(y_test) > 0 else 0.0
        self.trained = True
        return {'accuracy': float(acc), 'trained_on': int(len(X))}

    def predict_proba(self, df: pd.DataFrame, business_id: str) -> float:
        if not self.trained:
            return 0.0
        f = self.featurize(df, business_id)
        X = np.array([[f['cash_flow_score'], f['trust_index'], f['liquidity_score']]])
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)


def build_and_train(df: pd.DataFrame) -> SimpleRiskModel:
    m = SimpleRiskModel()
    m.train_on_dataset(df)
    return m
