"""Demand rate and variability estimation for inventory optimization."""

from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("demand_estimation")


def estimate_demand_parameters(
    df: pd.DataFrame,
    window_days: int = 60,
    demand_col: str = "units_sold",
    group_keys: Optional[list] = None,
    adjust_stockouts: bool = False,
) -> pd.DataFrame:
    """Estimate historical mean daily demand and standard deviation per group.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales dataset.
    window_days : int
        Recent historical window length in days used to capture current demand velocity.
    demand_col : str
        Column containing unit demand observations (default: 'units_sold').
    group_keys : Optional[list]
        Group keys (default: ['store_id', 'product_id']).
    adjust_stockouts : bool
        If True and 'stockout_units' is present, un-censors demand by adding lost sales:
        effective_demand = units_sold + stockout_units.
        Default is False to adhere strictly to observable POS retail records.

    Returns
    -------
    pd.DataFrame
        Aggregated demand parameters with columns:
        ['store_id', 'product_id', 'avg_daily_demand', 'std_daily_demand']
    """
    if group_keys is None:
        group_keys = ["store_id", "product_id"]

    logger.info("Estimating empirical demand parameters over last %d days (adjust_stockouts=%s)...", window_days, adjust_stockouts)

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    max_date = df_copy["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=window_days)

    # Filter to recent window
    recent_df = df_copy[df_copy["date"] >= cutoff_date].copy()

    # Determine effective target series
    effective_col = demand_col
    if adjust_stockouts and "stockout_units" in recent_df.columns and demand_col == "units_sold":
        recent_df["effective_demand"] = recent_df["units_sold"] + recent_df["stockout_units"]
        effective_col = "effective_demand"

    # Calculate statistics per series
    agg_df = (
        recent_df.groupby(group_keys)[effective_col]
        .agg(
            avg_daily_demand="mean",
            std_daily_demand="std",
            sample_size="count",
        )
        .reset_index()
    )

    # Guard against NaN std or 0 std for low-demand / constant series
    agg_df["avg_daily_demand"] = agg_df["avg_daily_demand"].clip(lower=0.1).round(2)
    agg_df["std_daily_demand"] = agg_df["std_daily_demand"].fillna(agg_df["avg_daily_demand"] * 0.25).clip(lower=0.1).round(2)

    logger.info("Estimated demand parameters for %d series.", len(agg_df))
    return agg_df
