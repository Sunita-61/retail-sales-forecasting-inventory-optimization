"""Sales Forecasting Page - Interactive Machine Learning Projections."""

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

from src.config import FORECASTS_FILE, PROCESSED_DATA_FILE

st.set_page_config(page_title="Sales Forecasting | RetailIQ", page_icon="🔮", layout="wide")

st.title("🔮 Predictive Demand & Sales Forecasting")
st.markdown("Autoregressive multi-step recursive forecasting delivering 30-day forward demand projections with 90% confidence bands.")

if not FORECASTS_FILE.exists() or not PROCESSED_DATA_FILE.exists():
    st.error("Forecast data or historical data not found. Please run `python main.py` first.")
    st.stop()


@st.cache_data
def load_forecast_data():
    hist_df = pd.read_csv(PROCESSED_DATA_FILE)
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    fc_df = pd.read_csv(FORECASTS_FILE)
    fc_df["date"] = pd.to_datetime(fc_df["date"])
    return hist_df, fc_df


hist_df, fc_df = load_forecast_data()

# Sidebar Filters
st.sidebar.header("Forecast Selector")
stores = sorted(fc_df["store_id"].unique().tolist())
selected_store = st.sidebar.selectbox("Select Store", stores)

store_products = fc_df[fc_df["store_id"] == selected_store]["product_name"].unique().tolist()
selected_product = st.sidebar.selectbox("Select Product", store_products)

history_days_to_show = st.sidebar.slider("Historical Lookback (Days)", min_value=14, max_value=120, value=60)

# Filter data
fc_item = fc_df[(fc_df["store_id"] == selected_store) & (fc_df["product_name"] == selected_product)].sort_values(by="date")
hist_item = hist_df[(hist_df["store_id"] == selected_store) & (hist_df["product_name"] == selected_product)].sort_values(by="date")

cutoff_date = hist_item["date"].max() - pd.Timedelta(days=history_days_to_show)
hist_recent = hist_item[hist_item["date"] >= cutoff_date]

# KPI Summary for Selected SKU
k1, k2, k3, k4 = st.columns(4)
total_fc_units = fc_item["forecasted_demand"].sum()
total_fc_rev = fc_item["forecast_revenue"].sum()
avg_daily_fc = fc_item["forecasted_demand"].mean()
unit_price = fc_item["unit_price"].iloc[0]

with k1:
    st.metric("Expected 30-Day Demand", f"{total_fc_units:,.0f} units")
with k2:
    st.metric("Projected 30-Day Revenue", f"${total_fc_rev:,.2f}")
with k3:
    st.metric("Avg Daily Projected Demand", f"{avg_daily_fc:.1f} units/day")
with k4:
    st.metric("Unit Retail Price", f"${unit_price:.2f}")

st.markdown("---")

# Main Interactive Forecast Chart
st.subheader(f"Historical Sales vs 30-Day Forward Forecast: {selected_product} ({selected_store})")

fig = go.Figure()

# Historical actuals
fig.add_trace(go.Scatter(
    x=hist_recent["date"],
    y=hist_recent["units_sold"],
    mode="lines+markers",
    name="Historical Actual Sales",
    line=dict(color="#2563EB", width=2),
    marker=dict(size=4),
))

# 30-day forward predictions
fig.add_trace(go.Scatter(
    x=fc_item["date"],
    y=fc_item["forecasted_demand"],
    mode="lines+markers",
    name="30-Day Forward Forecast",
    line=dict(color="#EA580C", width=2.5, dash="dash"),
    marker=dict(size=5),
))

# 90% Confidence Interval
if "forecast_upper_90ci" in fc_item.columns and "forecast_lower_90ci" in fc_item.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_item["date"], fc_item["date"][::-1]]),
        y=pd.concat([fc_item["forecast_upper_90ci"], fc_item["forecast_lower_90ci"][::-1]]),
        fill="toself",
        fillcolor="rgba(251, 146, 60, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="90% Prediction Interval",
    ))

# Vertical line dividing history and forecast
div_date = hist_item["date"].max()
fig.add_vline(
    x=div_date.timestamp() * 1000,
    line_width=1.5,
    line_dash="dot",
    line_color="#4B5563",
    annotation_text="Forecast Horizon Start",
    annotation_position="top left",
)

fig.update_layout(
    template="plotly_white",
    height=480,
    xaxis_title="Date",
    yaxis_title="Units Sold / Demanded",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# Secondary Chart: Chain-wide Category Forward Projections
st.subheader("Forward 30-Day Demand Projections Across Categories")
cat_fc = fc_df.groupby(["date", "category"])["forecasted_demand"].sum().reset_index()
fig_cat_fc = px.line(
    cat_fc,
    x="date",
    y="forecasted_demand",
    color="category",
    labels={"forecasted_demand": "Projected Daily Units", "date": "Date"},
    template="plotly_white",
)
fig_cat_fc.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_cat_fc, use_container_width=True)

# Forecast Table & CSV Download
st.subheader("Forecast Data Schedule")
st.dataframe(
    fc_item[["date", "store_id", "product_name", "forecasted_demand", "forecast_lower_90ci", "forecast_upper_90ci", "forecast_revenue"]],
    use_container_width=True,
)

fc_csv = fc_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Complete Chain Forecasts (CSV)",
    data=fc_csv,
    file_name="chain_sales_forecasts_30d.csv",
    mime="text/csv",
)
