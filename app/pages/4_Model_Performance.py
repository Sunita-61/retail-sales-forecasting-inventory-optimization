"""Model Performance Page - Benchmark Evaluation and Model Cards."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BEST_MODEL_FILE, MODEL_METRICS_FILE
from src.forecasting.train_model import load_best_forecasting_model

st.set_page_config(page_title="Model Performance | RetailIQ", page_icon="🎯", layout="wide")

st.title("🎯 Model Evaluation & Benchmark Diagnostics")
st.markdown("Rigorous chronological validation comparing heuristic baselines against machine learning models on identical evaluation periods.")

if not MODEL_METRICS_FILE.exists() or not BEST_MODEL_FILE.exists():
    st.error("Model artifacts or metrics not found. Run `python main.py` first.")
    st.stop()


@st.cache_data
def load_metrics_and_model():
    metrics_df = pd.read_csv(MODEL_METRICS_FILE)
    artifact = load_best_forecasting_model()
    return metrics_df, artifact


metrics_df, artifact = load_metrics_and_model()

# Top Cards: Champion Model & Test Results
st.subheader("Champion Model Summary")
c1, c2, c3, c4 = st.columns(4)

champ_name = artifact["model_name"]
test_metrics = artifact["test_metrics"]

with c1:
    st.metric("Selected Model", champ_name)
with c2:
    st.metric("Test MAE", f"{test_metrics['MAE']:.3f} units")
with c3:
    st.metric("Test RMSE", f"{test_metrics['RMSE']:.3f} units")
with c4:
    st.metric("Test WAPE", f"{test_metrics['WAPE (%)']:.2f}%", delta="Held-out Test Period")

st.markdown("---")

# Comparative Metrics Table & Bar Charts
st.subheader("Candidate Model Benchmark Comparison (Validation Set)")

col_table, col_charts = st.columns([1, 1])

with col_table:
    st.markdown("**Validation Benchmark Metrics:**")
    st.dataframe(metrics_df.style.highlight_min(subset=["MAE", "RMSE", "WAPE (%)"], color="#DCFCE7").highlight_max(subset=["R2"], color="#DCFCE7"), use_container_width=True)

    st.markdown("""
    **Metric Definitions:**
    - **MAE (Mean Absolute Error)**: Average absolute magnitude of error in units sold.
    - **RMSE (Root Mean Squared Error)**: Penalizes larger deviations more heavily.
    - **WAPE (Weighted Absolute Percentage Error)**: $\\frac{\\sum |y - \\hat{y}|}{\\sum y} \\times 100\\%$. Standard industry KPI that avoids infinite divisions caused by zero sales in retail.
    - **$R^2$ Score**: Proportion of variance explained by model features.
    """)

with col_charts:
    fig_wape = px.bar(
        metrics_df.sort_values(by="WAPE (%)", ascending=False),
        x="WAPE (%)",
        y="Model",
        orientation="h",
        color="WAPE (%)",
        color_continuous_scale="Reds_r",
        title="Validation WAPE (%) — Lower is Better",
    )
    fig_wape.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_wape, use_container_width=True)

st.markdown("---")

# Feature Importance Analysis
st.subheader("Feature Importance Analysis (Random Forest Champion)")

model = artifact["model"]
feature_cols = artifact["feature_cols"]

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True).tail(12)

    fig_imp = px.bar(
        feat_imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top 12 Most Predictive Features",
    )
    fig_imp.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Feature importance is available for tree-based ensemble models.")

# Evaluation Strategy & Governance
st.subheader("Data Leakage Prevention & Chronological Splitting Strategy")
st.markdown("""
1. **Strict Chronological Splitting**:
   - **Training Set (70%)**: 2024-01-29 to 2025-06-30
   - **Validation Set (15%)**: 2025-07-01 to 2025-09-30 (Used exclusively for hyperparameter tuning & model selection)
   - **Test Set (15%)**: 2025-10-01 to 2025-12-31 (Unseen final performance test)
2. **Lag Calculation Integrity**:
   - All lag features ($t-1, t-7, t-14, t-21, t-28$) are calculated strictly on prior observations per `(store_id, product_id)`.
   - Rolling window averages and standard deviations apply `shift(1)` to ensure the current day's target is never visible during feature computation.
3. **Recursive Forecasting**:
   - Multi-step forward predictions dynamically roll forward previous forecasts as autoregressive lags, eliminating any dependency on future actuals.
""")
