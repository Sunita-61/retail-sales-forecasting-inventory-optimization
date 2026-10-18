"""Sales Analysis Page - Exploratory Data Analysis and Business Trends."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DATA_FILE

st.set_page_config(page_title="Sales Analysis | RetailIQ", page_icon="📊", layout="wide")

st.title("📊 Sales & Demand Analysis")
st.markdown("Explore historical sales velocity, store-level performance, category dynamics, and promotional impact.")

if not PROCESSED_DATA_FILE.exists():
    st.error("Processed data file not found. Run `python main.py` first.")
    st.stop()

@st.cache_data
def load_sales():
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_sales()

# Sidebar Filters
st.sidebar.header("Filter Criteria")
stores = ["All Stores"] + sorted(df["store_id"].unique().tolist())
selected_store = st.sidebar.selectbox("Select Store", stores)

categories = ["All Categories"] + sorted(df["category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Select Category", categories)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Apply Filters
filtered_df = df.copy()
if selected_store != "All Stores":
    filtered_df = filtered_df[filtered_df["store_id"] == selected_store]
if selected_cat != "All Categories":
    filtered_df = filtered_df[filtered_df["category"] == selected_cat]

if len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[(filtered_df["date"].dt.date >= start_d) & (filtered_df["date"].dt.date <= end_d)]

# KPI Row
m1, m2, m3, m4 = st.columns(4)
total_rev = filtered_df["monetary_sales"].sum()
total_units = filtered_df["units_sold"].sum()
avg_price = filtered_df["unit_price"].mean()
total_stockouts = filtered_df["stockout_units"].sum() if "stockout_units" in filtered_df.columns else 0

with m1:
    st.metric("Filtered Revenue", f"${total_rev:,.2f}")
with m2:
    st.metric("Total Units Sold", f"{total_units:,.0f} units")
with m3:
    st.metric("Average Unit Price", f"${avg_price:.2f}")
with m4:
    st.metric("Unmet Demand (Stockouts)", f"{total_stockouts:,.0f} units", delta_color="inverse")

st.markdown("---")

# Charts: Trends and Categories
row1_left, row1_right = st.columns([3, 2])

with row1_left:
    st.subheader("Time Series Sales Trend")
    trend_type = st.radio("Aggregation Level", ["Daily", "Weekly", "Monthly"], horizontal=True)

    if trend_type == "Daily":
        t_data = filtered_df.groupby("date")["monetary_sales"].sum().reset_index()
        fig_ts = px.line(t_data, x="date", y="monetary_sales", labels={"monetary_sales": "Revenue ($)", "date": "Date"})
    elif trend_type == "Weekly":
        t_data = filtered_df.set_index("date").resample("W-MON")["monetary_sales"].sum().reset_index()
        fig_ts = px.bar(t_data, x="date", y="monetary_sales", labels={"monetary_sales": "Weekly Revenue ($)", "date": "Week Ending"})
    else:
        t_data = filtered_df.set_index("date").resample("MS")["monetary_sales"].sum().reset_index()
        fig_ts = px.bar(t_data, x="date", y="monetary_sales", labels={"monetary_sales": "Monthly Revenue ($)", "date": "Month"})

    fig_ts.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_ts, use_container_width=True)

with row1_right:
    st.subheader("Category Revenue Share")
    cat_summary = filtered_df.groupby("category")["monetary_sales"].sum().reset_index()
    fig_cat = px.pie(cat_summary, values="monetary_sales", names="category", hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
    fig_cat.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cat, use_container_width=True)

# Charts: Store Rankings & Day of Week
row2_left, row2_right = st.columns(2)

with row2_left:
    st.subheader("Store Revenue Comparison")
    store_rev = df.groupby(["store_id", "category"])["monetary_sales"].sum().reset_index()
    fig_store = px.bar(store_rev, x="store_id", y="monetary_sales", color="category", barmode="stack",
                       labels={"monetary_sales": "Revenue ($)", "store_id": "Store ID"},
                       color_discrete_sequence=px.colors.qualitative.Safe)
    fig_store.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_store, use_container_width=True)

with row2_right:
    st.subheader("Day-of-Week Seasonality")
    dow_df = filtered_df.copy()
    dow_df["dow"] = dow_df["date"].dt.day_name()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_summary = dow_df.groupby("dow")["units_sold"].mean().reindex(days_order).reset_index()
    fig_dow = px.bar(dow_summary, x="dow", y="units_sold", labels={"units_sold": "Avg Daily Units Sold", "dow": "Day of Week"},
                     color="units_sold", color_continuous_scale="Blues")
    fig_dow.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_dow, use_container_width=True)

# Top Products & Promotional Lift
row3_left, row3_right = st.columns(2)

with row3_left:
    st.subheader("Top Selling Products")
    top_p = filtered_df.groupby("product_name").agg(
        revenue=("monetary_sales", "sum"),
        units=("units_sold", "sum"),
    ).sort_values(by="revenue", ascending=False).head(8).reset_index()
    fig_top = px.bar(top_p, x="revenue", y="product_name", orientation="h", labels={"revenue": "Revenue ($)", "product_name": ""},
                     color="revenue", color_continuous_scale="Viridis")
    fig_top.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_top, use_container_width=True)

with row3_right:
    st.subheader("Promotional Demand Uplift")
    promo_stats = filtered_df.groupby(["category", "promotion_flag"])["units_sold"].mean().reset_index()
    promo_stats["Status"] = promo_stats["promotion_flag"].map({0: "Regular Price", 1: "On Promotion"})
    fig_promo = px.bar(promo_stats, x="category", y="units_sold", color="Status", barmode="group",
                       color_discrete_map={"Regular Price": "#94A3B8", "On Promotion": "#F97316"},
                       labels={"units_sold": "Mean Daily Units Sold", "category": "Category"})
    fig_promo.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_promo, use_container_width=True)

# Data Table & Export
st.subheader("Filtered Observations Table")
st.dataframe(filtered_df[["date", "store_id", "product_id", "product_name", "category", "unit_price", "units_sold", "monetary_sales", "on_hand_stock", "promotion_flag"]].head(200), use_container_width=True)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Filtered Sales Data (CSV)",
    data=csv_data,
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)
