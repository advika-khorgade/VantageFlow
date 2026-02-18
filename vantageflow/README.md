# VantageFlow – Adaptive SME Credit Intelligence Engine

VantageFlow is a Streamlit-based prototype that computes cash-flow based credit assessments for thin-file SMEs using alternative data signals.

## Architecture (text diagram)

vantageflow/
- app.py (Streamlit UI)
- credit_engine.py (scoring logic)
- model.py (simple RandomForest simulation)
- utils.py (data preprocessing & aggregation)
- data/vantageflow_sme_dataset.csv (sample daily dataset)
- requirements.txt

## Features

- Cash Flow Stability Score
- Trust Index from alternative data (GST, utilities, returns, sentiment)
- Liquidity ratio and Composite Risk Score
- Dynamic credit limit based on risk category
- Cash runway estimation and alerting
- Simple ML model that simulates default probability
- Revenue and Net Cash Flow charts (matplotlib)

## How to run

1. Create a venv and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

## Data

The app expects `data/vantageflow_sme_dataset.csv` with the following columns:

- business_id, date, daily_revenue, daily_expense, net_cash_flow,
  gst_compliance_score, utility_payment_consistency,
  ecommerce_return_rate, social_sentiment_score

## Notes

- This is a prototype. Financial heuristics are simplified for demo purposes.
- The ML model is synthetic and trained on a small sample in `model.py`.
