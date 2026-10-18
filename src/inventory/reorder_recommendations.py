"""Inventory replenishment recommendations generator and exporter."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    INVENTORY_METRICS_FILE,
    INVENTORY_OUTPUT_DIR,
    RECOMMENDATIONS_FILE,
    RECOMMENDATIONS_OUTPUT_DIR,
    REVIEW_PERIOD_DAYS,
    SERVICE_LEVEL_Z_SCORE,
)
from src.inventory.demand_estimation import estimate_demand_parameters
from src.inventory.inventory_calculations import evaluate_inventory_row
from src.utils.logging_config import setup_logger

logger = setup_logger("reorder_recommendations")


def generate_replenishment_recommendations(
    sales_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    save_outputs: bool = True,
    service_level_z: float = SERVICE_LEVEL_Z_SCORE,
    review_period_days: int = REVIEW_PERIOD_DAYS,
    adjust_stockouts: bool = True,
) -> pd.DataFrame:
    """Generate store-level replenishment recommendations based on inventory position and demand.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Historical retail dataset with store, product, sales, and on-hand inventory.
    forecast_df : Optional[pd.DataFrame]
        Future forecast predictions. If provided, future demand rate can be factored in.
    save_outputs : bool
        Whether to export results to outputs/inventory and outputs/recommendations.
    service_level_z : float
        Normal distribution z-score for desired cycle service level.
    review_period_days : int
        Review period interval in days.
    adjust_stockouts : bool
        If True and stockout_units is available, un-censors demand (units_sold + stockout_units)
        to prevent under-sizing safety stock and ROP due to past stockouts. Default is True.

    Returns
    -------
    pd.DataFrame
        Detailed replenishment recommendations table.
    """
    logger.info("Generating inventory replenishment recommendations (adjust_stockouts=%s)...", adjust_stockouts)

    df_copy = sales_df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    latest_date = df_copy["date"].max()

    # 1. Estimate demand parameters (stockout-adjusted effective demand by default)
    demand_stats = estimate_demand_parameters(df_copy, window_days=60, adjust_stockouts=adjust_stockouts)

    # 2. Extract latest inventory state for each store-product series
    latest_state = (
        df_copy.sort_values(by="date")
        .groupby(["store_id", "product_id"])
        .last()
        .reset_index()
    )

    merged = pd.merge(latest_state, demand_stats, on=["store_id", "product_id"], how="left")

    recommendations = []
    for _, row in merged.iterrows():
        s_id = row["store_id"]
        p_id = row["product_id"]
        p_name = row.get("product_name", f"Product {p_id}")
        cat = row.get("category", "General")
        u_price = float(row.get("unit_price", 0.0))
        u_cost = float(row.get("unit_cost", u_price * 0.6))
        lead_time = int(row.get("lead_time_days", 5))

        on_hand = int(row.get("on_hand_stock", 0))
        # Simulated pipeline orders currently in transit (e.g., from recent stock_received patterns)
        on_order = int(row.get("stock_received", 0))
        backorders = 0

        # Demand parameters
        avg_demand = float(row["avg_daily_demand"])
        std_demand = float(row["std_daily_demand"])

        # If forward forecast is available, optionally blend with forecasted velocity
        if forecast_df is not None and not forecast_df.empty:
            fc_subset = forecast_df[
                (forecast_df["store_id"] == s_id) & (forecast_df["product_id"] == p_id)
            ]
            if not fc_subset.empty:
                fc_mean = float(fc_subset["forecasted_demand"].mean())
                # Blend 60% forecast, 40% recent empirical
                avg_demand = round(0.6 * fc_mean + 0.4 * avg_demand, 2)

        # Calculate optimization metrics
        calc_result = evaluate_inventory_row(
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            avg_daily_demand=avg_demand,
            std_daily_demand=std_demand,
            lead_time_days=lead_time,
            review_period=review_period_days,
            z_score=service_level_z,
        )

        order_qty = calc_result["suggested_order_qty"]
        est_order_cost = round(order_qty * u_cost, 2)

        rec_record = {
            "evaluation_date": latest_date.strftime("%Y-%m-%d"),
            "store_id": s_id,
            "product_id": p_id,
            "product_name": p_name,
            "category": cat,
            "unit_price": u_price,
            "unit_cost": u_cost,
            **calc_result,
            "estimated_order_cost": est_order_cost,
        }
        recommendations.append(rec_record)

    rec_df = pd.DataFrame(recommendations)

    # Sort so urgent reorders and stockouts are at the top
    status_order = {
        "Critical / Stockout": 0,
        "Reorder Required": 1,
        "Low Stock": 2,
        "Healthy Stock": 3,
        "Overstock Risk": 4,
    }
    rec_df["priority_rank"] = rec_df["inventory_status"].map(lambda x: status_order.get(x, 5))
    rec_df = rec_df.sort_values(by=["priority_rank", "days_of_supply"]).drop(columns=["priority_rank"]).reset_index(drop=True)

    if save_outputs:
        INVENTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RECOMMENDATIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        rec_df.to_csv(INVENTORY_METRICS_FILE, index=False)
        logger.info("Saved full inventory metrics to %s", INVENTORY_METRICS_FILE)

        # Filter actionable reorders for the recommendations table
        actionable_df = rec_df[rec_df["suggested_order_qty"] > 0].copy()
        actionable_df.to_csv(RECOMMENDATIONS_FILE, index=False)
        logger.info(
            "Saved %d actionable replenishment recommendations to %s",
            len(actionable_df),
            RECOMMENDATIONS_FILE,
        )

    return rec_df
