"""Inventory Optimization Page - Dynamic Multi-Echelon Replenishment Cockpit."""

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

from src.config import INVENTORY_METRICS_FILE, RECOMMENDATIONS_FILE
from src.inventory.inventory_calculations import evaluate_inventory_row

st.set_page_config(page_title="Inventory Optimization | RetailIQ", page_icon="📦", layout="wide")

st.title("📦 Inventory Optimization & Replenishment")
st.markdown("Automated safety stock sizing, dynamic reorder point calculation, order-up-to policies, and stockout mitigation.")

if not INVENTORY_METRICS_FILE.exists():
    st.error("Inventory metrics not found. Run `python main.py` first.")
    st.stop()


@st.cache_data
def load_inventory():
    return pd.read_csv(INVENTORY_METRICS_FILE)


inv_base_df = load_inventory()

# Sidebar: Interactive Simulation Controls
st.sidebar.header("Policy & Simulation Controls")
service_level = st.sidebar.select_slider(
    "Target Cycle Service Level (CSL)",
    options=[0.85, 0.90, 0.95, 0.98, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x*100)}%",
)
# Map CSL to standard normal Z-score
z_mapping = {0.85: 1.036, 0.90: 1.282, 0.95: 1.645, 0.98: 2.054, 0.99: 2.326}
sim_z = z_mapping[service_level]

review_period_sim = st.sidebar.slider("Replenishment Review Period (Days)", min_value=1, max_value=14, value=7)

stores = ["All Stores"] + sorted(inv_base_df["store_id"].unique().tolist())
selected_store = st.sidebar.selectbox("Filter Store", stores)

statuses = ["All Statuses"] + sorted(inv_base_df["inventory_status"].unique().tolist())
selected_status = st.sidebar.selectbox("Filter Status", statuses)

# Real-time policy recomputation based on user-selected service level & review period
recomputed_rows = []
for _, row in inv_base_df.iterrows():
    calc = evaluate_inventory_row(
        on_hand=row["on_hand_stock"],
        on_order=row["on_order_stock"],
        backorders=row.get("backorders", 0),
        avg_daily_demand=row["avg_daily_demand"],
        std_daily_demand=row["std_daily_demand"],
        lead_time_days=int(row["lead_time_days"]),
        review_period=review_period_sim,
        z_score=sim_z,
    )
    rec = row.to_dict()
    rec.update(calc)
    rec["estimated_order_cost"] = round(calc["suggested_order_qty"] * float(row["unit_cost"]), 2)
    recomputed_rows.append(rec)

inv_df = pd.DataFrame(recomputed_rows)

# Filter
filtered_inv = inv_df.copy()
if selected_store != "All Stores":
    filtered_inv = filtered_inv[filtered_inv["store_id"] == selected_store]
if selected_status != "All Statuses":
    filtered_inv = filtered_inv[filtered_inv["inventory_status"] == selected_status]

# Operational KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
total_reorders = (filtered_inv["suggested_order_qty"] > 0).sum()
total_capital_needed = filtered_inv["estimated_order_cost"].sum()
critical_count = (filtered_inv["inventory_status"] == "Critical / Stockout").sum()
overstock_count = (filtered_inv["inventory_status"] == "Overstock Risk").sum()
avg_dos = filtered_inv["days_of_supply"].mean()

with c1:
    st.metric("Total Reorders Required", f"{total_reorders} SKUs")
with c2:
    st.metric("Replenishment Capital", f"${total_capital_needed:,.2f}")
with c3:
    st.metric("Critical Stockouts", f"{critical_count} SKUs", delta="Immediate Action", delta_color="inverse")
with c4:
    st.metric("Overstock Risks", f"{overstock_count} SKUs", delta="Holding Cost Risk")
with c5:
    st.metric("Average Days of Supply", f"{avg_dos:.1f} Days")

st.markdown("---")

# Visualizations: Health Status and Position vs Safety Stock
row1_left, row1_right = st.columns([1, 1])

with row1_left:
    st.subheader("Inventory Status Distribution")
    status_counts = filtered_inv["inventory_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    color_map = {
        "Critical / Stockout": "#DC2626",
        "Reorder Required": "#EA580C",
        "Low Stock": "#EAB308",
        "Healthy Stock": "#16A34A",
        "Overstock Risk": "#9333EA",
    }
    fig_status = px.bar(
        status_counts,
        x="Status",
        y="Count",
        color="Status",
        color_discrete_map=color_map,
        labels={"Count": "SKU Locations", "Status": ""},
    )
    fig_status.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig_status, use_container_width=True)

with row1_right:
    st.subheader("Inventory Position vs Reorder Point (Sample Top SKUs)")
    sample_skus = filtered_inv.head(10)
    fig_comp = go.Figure()

    fig_comp.add_trace(go.Bar(
        name="Inventory Position",
        x=sample_skus["product_name"],
        y=sample_skus["inventory_position"],
        marker_color="#3B82F6",
    ))
    fig_comp.add_trace(go.Bar(
        name="Reorder Point (ROP)",
        x=sample_skus["product_name"],
        y=sample_skus["reorder_point"],
        marker_color="#F97316",
    ))
    fig_comp.add_trace(go.Scatter(
        name="Target Stock (S)",
        x=sample_skus["product_name"],
        y=sample_skus["target_stock"],
        mode="markers",
        marker=dict(color="#10B981", size=10, symbol="diamond"),
    ))

    fig_comp.update_layout(
        template="plotly_white",
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# Replenishment Order Table
st.subheader("Actionable Replenishment Recommendations Schedule")
st.markdown("Orders prioritized by supply deficit and stockout severity:")

display_cols = [
    "store_id",
    "product_name",
    "category",
    "on_hand_stock",
    "inventory_position",
    "avg_daily_demand",
    "lead_time_days",
    "safety_stock",
    "reorder_point",
    "target_stock",
    "suggested_order_qty",
    "unit_cost",
    "estimated_order_cost",
    "days_of_supply",
    "inventory_status",
]

st.dataframe(
    filtered_inv[display_cols].style.highlight_between(
        subset=["suggested_order_qty"], left=1, right=100000, color="#FFEDD5"
    ),
    use_container_width=True,
)

# Export Recommendations CSV
csv_rec = filtered_inv[filtered_inv["suggested_order_qty"] > 0].to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Export Replenishment Purchase Orders (CSV)",
    data=csv_rec,
    file_name="purchase_order_recommendations.csv",
    mime="text/csv",
)
