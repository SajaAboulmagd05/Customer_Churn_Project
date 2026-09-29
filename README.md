# Customer Intelligence: Churn, Value & Segmentation

A telecom (and a bit of banking) customer wants to know three things: **who's about to leave**, **what makes a customer valuable**, and **which natural groups exist in the customer base**. This project answers all three with classic ML — no deep learning, just solid preprocessing, honest evaluation, and models that actually get used.

## What's inside

**Classification** — predicts churn on the Telco dataset. Seven models tuned with GridSearchCV, evaluated on F1/recall/precision (never accuracy alone, since ~73% of customers don't churn anyway). Random Forest came out on top, catching ~80% of real churners.

**Regression** — predicts EstimatedSalary on the Bank Churn dataset (Kaggle Playground Series S4E1). Baseline Linear Regression, Polynomial, Ridge, and Lasso, compared head to head.

**Segmentation** — KMeans and DBSCAN group customers by tenure, spend, and services, with each cluster profiled and named for the retention team (e.g. "high-value at risk," "loyal low-spend").

**Streamlit app** — a small demo that scores a new customer on the spot using the saved models. No retraining, just load and predict.

## Why some things were dropped

A couple of columns looked useful and weren't:
- `Churn Score` correlated 0.66 with the actual churn outcome — it's a leaked proxy, not a real feature.
- `Churn Reason` is only filled in for customers who already churned, so it's the answer in disguise.
- `Total Charges` ≈ `Tenure × Monthly Charges` — kept the two components, dropped the redundant one.

Every drop is backed by a number, not a guess.

## Structure

```
data/          raw + cleaned datasets (gitignored)
notebooks/     01 preprocessing/EDA → 02 regression → 03 classification → 04 segmentation
models/        saved model, scaler, and feature columns (joblib)
app/           streamlit_app.py
report/        business summary for a non-technical audience
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
streamlit run app/streamlit_app.py
```

## Honest notes

CLTV turned out to have a low predictability ceiling with the Telco features available (R² ~0.17 even with tree-based models) — that's a real finding, not a modeling failure, and it's why regression moved to a dataset better suited to the task. Model comparisons everywhere in this project are picked on the metric that matches the business question, not the one that looks best.
