"""Visualization utilities and figure generator for EDA, forecasting, and inventory."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger("visualization")

# Set consistent, modern styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.sans-serif": "DejaVu Sans",
        "font.family": "sans-serif",
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True,
    }
)


def plot_daily_sales_trend(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot aggregate daily retail sales trend over time with 7-day moving average."""
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    daily = df_copy.groupby("date")["units_sold"].sum().reset_index()
    daily["rolling_7"] = daily["units_sold"].rolling(7).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["units_sold"], color="#90CAF9", alpha=0.5, label="Daily Actual Units")
    ax.plot(daily["date"], daily["rolling_7"], color="#1E88E5", linewidth=2.2, label="7-Day Moving Avg")
    ax.set_title("Total Chain-Wide Daily Retail Sales Trend (2024-2025)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    ax.legend(frameon=True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_monthly_sales_trend(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot monthly total units sold and revenue."""
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy["year_month"] = df_copy["date"].dt.to_period("M").astype(str)

    monthly = df_copy.groupby("year_month").agg(
        total_units=("units_sold", "sum"),
        total_revenue=("monetary_sales", "sum"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    color1 = "#2b5c8f"
    color2 = "#d95f02"

    ax1.bar(monthly["year_month"], monthly["total_units"], color=color1, alpha=0.75, label="Units Sold")
    ax1.set_xlabel("Year-Month")
    ax1.set_ylabel("Total Units Sold", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(monthly["year_month"], monthly["total_revenue"], color=color2, marker="o", linewidth=2.5, label="Revenue ($)")
    ax2.set_ylabel("Monetary Revenue ($)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    plt.title("Monthly Sales Volume and Revenue Trend", fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sales_by_category(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot sales distribution across categories."""
    cat_sales = df.groupby("category").agg(
        total_units=("units_sold", "sum"),
        total_revenue=("monetary_sales", "sum"),
    ).sort_values(by="total_revenue", ascending=False).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sns.barplot(data=cat_sales, x="category", y="total_revenue", hue="category", legend=False, ax=ax1, palette="Blues_r")
    ax1.set_title("Total Revenue by Category ($)", fontweight="bold")
    ax1.set_ylabel("Revenue ($)")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=20)

    sns.barplot(data=cat_sales, x="category", y="total_units", hue="category", legend=False, ax=ax2, palette="mako")
    ax2.set_title("Total Units Sold by Category", fontweight="bold")
    ax2.set_ylabel("Units Sold")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=20)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sales_by_store(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot total volume and revenue across store locations."""
    store_sales = df.groupby(["store_id"]).agg(
        units=("units_sold", "sum"),
        revenue=("monetary_sales", "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=store_sales, x="store_id", y="revenue", hue="store_id", legend=False, palette="viridis", ax=ax)
    ax.set_title("Gross Revenue by Retail Store Location", fontweight="bold")
    ax.set_xlabel("Store Identifier")
    ax.set_ylabel("Total Revenue ($)")

    for p in ax.patches:
        ax.annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.9),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_top_products(df: pd.DataFrame, top_n: int = 10, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot top-performing products by total revenue."""
    top_p = df.groupby("product_name")["monetary_sales"].sum().sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_p.plot(kind="barh", ax=ax, color="#3F51B5")
    ax.set_title(f"Top {top_n} Revenue Generating Products", fontweight="bold")
    ax.set_xlabel("Total Revenue ($)")
    ax.set_ylabel("")

    for i, v in enumerate(top_p):
        ax.text(v * 0.98, i, f" ${v:,.0f}", color="white", va="center", ha="right", fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_day_of_week_pattern(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot average daily units sold by day of the week."""
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy["day_name"] = df_copy["date"].dt.day_name()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    dow_avg = df_copy.groupby("day_name")["units_sold"].mean().reindex(days_order).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=dow_avg, x="day_name", y="units_sold", hue="day_name", legend=False, palette="Blues_d", ax=ax)
    ax.set_title("Average Daily Sales by Day of Week (Weekend Lift)", fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Units Sold per SKU")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_promotion_impact(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Compare average daily demand during promotion vs non-promotion periods."""
    promo_stats = df.groupby(["category", "promotion_flag"])["units_sold"].mean().reset_index()
    promo_stats["promotion_flag"] = promo_stats["promotion_flag"].map({0: "Regular Price", 1: "On Promotion"})

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=promo_stats, x="category", y="units_sold", hue="promotion_flag", palette=["#78909C", "#E64A19"], ax=ax)
    ax.set_title("Promotional Demand Uplift by Product Category", fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Mean Daily Units Sold")
    ax.legend(title="Status")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_inventory_status_breakdown(rec_df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot distribution of inventory statuses and replenishment urgency."""
    status_counts = rec_df["inventory_status"].value_counts()
    colors = {
        "Critical / Stockout": "#D32F2F",
        "Reorder Required": "#F57C00",
        "Low Stock": "#FBC02D",
        "Healthy Stock": "#388E3C",
        "Overstock Risk": "#7B1FA2",
    }
    palette = [colors.get(s, "#757575") for s in status_counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(status_counts.index, status_counts.values, color=palette)
    ax.set_title("SKU Inventory Health Status Breakdown", fontweight="bold")
    ax.set_ylabel("Number of Store-SKU Locations")
    ax.tick_params(axis="x", rotation=25)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_forecast_vs_actual(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    store_id: str = "STR_01",
    product_id: str = "PRD_001",
    recent_history_days: int = 45,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot recent actual sales against forward forecasted values with confidence interval."""
    act_sub = actual_df[(actual_df["store_id"] == store_id) & (actual_df["product_id"] == product_id)].copy()
    act_sub["date"] = pd.to_datetime(act_sub["date"])
    cutoff = act_sub["date"].max() - pd.Timedelta(days=recent_history_days)
    act_sub = act_sub[act_sub["date"] >= cutoff]

    fc_sub = forecast_df[(forecast_df["store_id"] == store_id) & (forecast_df["product_id"] == product_id)].copy()
    fc_sub["date"] = pd.to_datetime(fc_sub["date"])

    p_name = act_sub["product_name"].iloc[0] if not act_sub.empty else product_id

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(act_sub["date"], act_sub["units_sold"], label="Historical Actual Sales", color="#1976D2", marker="o", markersize=3)
    ax.plot(fc_sub["date"], fc_sub["forecasted_demand"], label="30-Day Forward Forecast", color="#E65100", linestyle="--", linewidth=2.2)

    if "forecast_lower_90ci" in fc_sub.columns and "forecast_upper_90ci" in fc_sub.columns:
        ax.fill_between(
            fc_sub["date"],
            fc_sub["forecast_lower_90ci"],
            fc_sub["forecast_upper_90ci"],
            color="#FFE0B2",
            alpha=0.6,
            label="90% Prediction Interval",
        )

    ax.set_title(f"Sales Forecast vs Historicals: {p_name} ({store_id})", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Demanded / Sold")
    ax.legend(frameon=True, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_all_eda_figures(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Generate and persist all standard EDA and business summary charts."""
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating and saving all EDA visualization figures to %s...", out_dir)

    plot_daily_sales_trend(df, save_path=out_dir / "01_daily_sales_trend.png")
    plot_monthly_sales_trend(df, save_path=out_dir / "02_monthly_sales_trend.png")
    plot_sales_by_category(df, save_path=out_dir / "03_sales_by_category.png")
    plot_sales_by_store(df, save_path=out_dir / "04_sales_by_store.png")
    plot_top_products(df, save_path=out_dir / "05_top_products.png")
    plot_day_of_week_pattern(df, save_path=out_dir / "06_day_of_week_pattern.png")
    plot_promotion_impact(df, save_path=out_dir / "07_promotion_impact.png")

    logger.info("All EDA figures exported successfully.")
