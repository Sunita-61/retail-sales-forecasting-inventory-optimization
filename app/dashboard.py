"""Retail Sales Forecasting & Inventory Optimization System - Executive Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FORECASTS_FILE,
    INVENTORY_METRICS_FILE,
    MODEL_METRICS_FILE,
    PROCESSED_DATA_FILE,
    RECOMMENDATIONS_FILE,
)

# Page Configuration
st.set_page_config(
    page_title="RetailIQ | Sales & Inventory System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-sim {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not PROCESSED_DATA_FILE.exists():
        return None, None, None, None
    sales_df = pd.read_csv(PROCESSED_DATA_FILE)
    sales_df["date"] = pd.to_datetime(sales_df["date"])

    fc_df = pd.read_csv(FORECASTS_FILE) if FORECASTS_FILE.exists() else pd.DataFrame()
    if not fc_df.empty:
        fc_df["date"] = pd.to_datetime(fc_df["date"])

    inv_df = pd.read_csv(INVENTORY_METRICS_FILE) if INVENTORY_METRICS_FILE.exists() else pd.DataFrame()
    rec_df = pd.read_csv(RECOMMENDATIONS_FILE) if RECOMMENDATIONS_FILE.exists() else pd.DataFrame()

    return sales_df, fc_df, inv_df, rec_df


sales_df, fc_df, inv_df, rec_df = load_data()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">Retail Sales Forecasting & Inventory Optimization System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous Time-Series Demand Forecasting and Multi-Echelon Inventory Replenishment Platform</div>', unsafe_allow_html=True)
    st.caption("📅 Historical Data Cutoff: **2025-12-31** | Demonstration Forecast Horizon: **2026-01-01 to 2026-01-30** (30-day post-historical simulation)")
with col2:
    st.markdown("<br><span class='badge-sim'>SIMULATED RETAIL ENVIRONMENT</span>", unsafe_allow_html=True)

if sales_df is None:
    st.error("No processed data found. Please run `python main.py` in your terminal to initialize and populate the system.")
    st.stop()

# Key Metric Cards
st.markdown("### Executive Overview")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_revenue = sales_df["monetary_sales"].sum()
total_units = sales_df["units_sold"].sum()
total_stores = sales_df["store_id"].nunique()
total_skus = sales_df["product_id"].nunique()

reorders_needed = len(rec_df) if rec_df is not None and not rec_df.empty else 0
critical_stockouts = (inv_df["inventory_status"] == "Critical / Stockout").sum() if not inv_df.empty else 0

with kpi1:
    st.metric("Total Historical Revenue", f"${total_revenue:,.0f}", delta="2-Year Portfolio")
with kpi2:
    st.metric("Units Sold", f"{total_units:,.0f} units")
with kpi3:
    st.metric("Active Stores / SKUs", f"{total_stores} Stores | {total_skus} SKUs")
with kpi4:
    st.metric("Pending Reorders", f"{reorders_needed} SKUs", delta=f"{critical_stockouts} Stockouts", delta_color="inverse")
with kpi5:
    healthy_pct = round((inv_df["inventory_status"] == "Healthy Stock").mean() * 100, 1) if not inv_df.empty else 0
    st.metric("Stock Health Rate", f"{healthy_pct}%", delta="Target: 95% CSL")

st.markdown("---")

# Visual Summaries
row1_left, row1_right = st.columns([3, 2])

with row1_left:
    st.subheader("Aggregate Chain-Wide Daily Revenue Trend")
    daily_trend = sales_df.groupby("date").agg(
        revenue=("monetary_sales", "sum"),
        units=("units_sold", "sum"),
    ).reset_index()
    daily_trend["rolling_7"] = daily_trend["revenue"].rolling(7).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_trend["date"],
        y=daily_trend["revenue"],
        mode="lines",
        name="Daily Revenue ($)",
        line=dict(color="#93C5FD", width=1),
        opacity=0.6,
    ))
    fig_trend.add_trace(go.Scatter(
        x=daily_trend["date"],
        y=daily_trend["rolling_7"],
        mode="lines",
        name="7-Day Moving Avg",
        line=dict(color="#1D4ED8", width=2.5),
    ))
    fig_trend.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with row1_right:
    st.subheader("Current Inventory Health Status")
    if not inv_df.empty:
        status_counts = inv_df["inventory_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        color_map = {
            "Critical / Stockout": "#EF4444",
            "Reorder Required": "#F97316",
            "Low Stock": "#FBBF24",
            "Healthy Stock": "#10B981",
            "Overstock Risk": "#8B5CF6",
        }

        fig_pie = px.pie(
            status_counts,
            values="Count",
            names="Status",
            color="Status",
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig_pie.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="top", y=-0.05),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# Quick Navigation & Architecture Map
st.subheader("System Modules & Capabilities")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("#### 📊 1. Sales Analysis")
    st.write("Deep-dive historical sales velocity, product category distribution, store rankings, day-of-week trends, and promotional lift.")

with c2:
    st.markdown("#### 🔮 2. Sales Forecasting")
    st.write("Multi-step recursive 30-day forward demand projections with machine learning models and 90% uncertainty intervals.")

with c3:
    st.markdown("#### 📦 3. Inventory Optimization")
    st.write("Calculates safety stock, dynamic reorder points, order-up-to quantities, and flags stockout and overstock risks.")

with c4:
    st.markdown("#### 🎯 4. Model Performance")
    st.write("Fair benchmark comparisons of Seasonal Naive, 7-Day Moving Average, Ridge Regression, and Random Forest on held-out test data.")

st.info("💡 **Navigation Tip**: Use the left sidebar to navigate across the dedicated operational modules.")
