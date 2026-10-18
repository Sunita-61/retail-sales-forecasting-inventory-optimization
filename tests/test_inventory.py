"""Unit tests for inventory calculations and replenishment logic."""

import pytest
from src.inventory.inventory_calculations import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_target_stock_level,
    classify_inventory_status,
    evaluate_inventory_row,
)


def test_calculate_safety_stock_standard():
    """Verify standard safety stock formula: SS = ceil(Z * std * sqrt(L))."""
    # Z = 1.645, std = 4.0, L = 4 -> sqrt(4) = 2 -> 1.645 * 4 * 2 = 13.16 -> ceil = 14
    ss = calculate_safety_stock(daily_demand_std=4.0, lead_time_days=4, z_score=1.645)
    assert ss == 14


def test_calculate_safety_stock_invalid_inputs():
    """Verify safety stock rejects negative lead time or negative std."""
    with pytest.raises(ValueError):
        calculate_safety_stock(daily_demand_std=-2.0, lead_time_days=3)
    with pytest.raises(ValueError):
        calculate_safety_stock(daily_demand_std=2.0, lead_time_days=0)


def test_calculate_reorder_point():
    """Verify ROP = ceil(avg_daily_demand * lead_time + safety_stock)."""
    # avg_demand = 10.0, lead_time = 3, safety_stock = 8 -> 30 + 8 = 38
    rop = calculate_reorder_point(avg_daily_demand=10.0, lead_time_days=3, safety_stock=8)
    assert rop == 38


def test_calculate_target_stock_level():
    """Verify target stock S = ceil(avg_daily_demand * (review_period + lead_time) + safety_stock)."""
    # avg_demand = 10, review_period = 7, lead_time = 3, ss = 8 -> 10 * 10 + 8 = 108
    target_s = calculate_target_stock_level(
        avg_daily_demand=10.0, lead_time_days=3, safety_stock=8, review_period_days=7
    )
    assert target_s == 108


def test_classify_inventory_status():
    """Verify status classification thresholds."""
    assert classify_inventory_status(0, reorder_point=20, target_stock=50, safety_stock=10) == "Critical / Stockout"
    assert classify_inventory_status(15, reorder_point=20, target_stock=50, safety_stock=10) == "Reorder Required"
    assert classify_inventory_status(23, reorder_point=20, target_stock=50, safety_stock=10) == "Low Stock"
    assert classify_inventory_status(40, reorder_point=20, target_stock=50, safety_stock=10) == "Healthy Stock"
    assert classify_inventory_status(80, reorder_point=20, target_stock=50, safety_stock=10) == "Overstock Risk"


def test_evaluate_inventory_row_reorder_trigger():
    """Verify full evaluation triggers replenishment order when IP <= ROP."""
    result = evaluate_inventory_row(
        on_hand=15,
        on_order=0,
        backorders=0,
        avg_daily_demand=10.0,
        std_daily_demand=3.0,
        lead_time_days=3,
        review_period=7,
        z_score=1.645,
    )

    assert result["inventory_position"] == 15
    assert result["inventory_status"] == "Reorder Required"
    # Order quantity should be Target Stock - IP
    assert result["suggested_order_qty"] == result["target_stock"] - 15
    assert result["suggested_order_qty"] > 0
    assert result["stockout_risk"] is True  # 15 / 10 = 1.5 days supply < 3 days lead time


def test_evaluate_inventory_row_healthy_no_order():
    """Verify that when inventory is healthy (IP > ROP), suggested order quantity is 0."""
    result = evaluate_inventory_row(
        on_hand=80,
        on_order=0,
        backorders=0,
        avg_daily_demand=5.0,
        std_daily_demand=1.5,
        lead_time_days=2,
        review_period=7,
    )
    assert result["suggested_order_qty"] == 0
    assert result["inventory_status"] in ["Healthy Stock", "Overstock Risk"]
    assert result["stockout_risk"] is False


def test_evaluate_inventory_row_zero_demand_boundary():
    """Verify that zero daily demand does not cause ZeroDivisionError and safely defaults."""
    result = evaluate_inventory_row(
        on_hand=10,
        on_order=0,
        backorders=0,
        avg_daily_demand=0.0,
        std_daily_demand=0.0,
        lead_time_days=3,
        review_period=7,
    )
    assert result["days_of_supply"] > 0
    assert result["suggested_order_qty"] >= 0
    assert result["safety_stock"] >= 2  # Min safety stock buffer


def test_evaluate_inventory_row_negative_inputs_clamped():
    """Verify negative on-hand or on-order inputs are clamped safely to non-negative quantities."""
    result = evaluate_inventory_row(
        on_hand=-10,
        on_order=-5,
        backorders=-2,
        avg_daily_demand=5.0,
        std_daily_demand=1.0,
        lead_time_days=-3,
    )
    assert result["on_hand_stock"] == 0
    assert result["on_order_stock"] == 0
    assert result["lead_time_days"] >= 1
    assert result["inventory_position"] == 0
    assert result["inventory_status"] == "Critical / Stockout"


def test_estimate_demand_parameters_stockout_adjustment():
    """Verify estimate_demand_parameters un-censors demand when adjust_stockouts=True."""
    import pandas as pd
    from src.inventory.demand_estimation import estimate_demand_parameters

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="D"),
        "store_id": "STR_01",
        "product_id": "PRD_001",
        "units_sold": [10, 10, 0, 0, 10, 10, 10, 10, 10, 10],  # 2 days zero sales due to stockouts
        "stockout_units": [0, 0, 15, 15, 0, 0, 0, 0, 0, 0],     # 15 units demanded each day
    })

    # Standard (units_sold): mean is (80)/10 = 8.0
    res_censored = estimate_demand_parameters(df, window_days=10, adjust_stockouts=False)
    assert res_censored.iloc[0]["avg_daily_demand"] == 8.0

    # Adjusted (units_sold + stockout_units): mean is (80 + 30)/10 = 11.0
    res_uncensored = estimate_demand_parameters(df, window_days=10, adjust_stockouts=True)
    assert res_uncensored.iloc[0]["avg_daily_demand"] == 11.0


