"""Deterministic and stochastic retail inventory optimization calculations.

Assumptions & Mathematical Foundations:
1. Demand during lead time is treated as a sum of L independent daily demand distributions.
2. Under central limit approximations, standard deviation of lead time demand is:
       sigma_LTD = sigma_d * sqrt(L)
   where sigma_d is daily demand standard deviation and L is deterministic supplier lead time in days.
3. Safety Stock is determined to guarantee a target Cycle Service Level (CSL) Z:
       SS = max(MIN_BUFFER, ceil(Z * sigma_d * sqrt(L)))
   Here Z = 1.645 corresponds to a 95% cycle service level under a Gaussian demand distribution.
4. Reorder Point (ROP):
       ROP = ceil(d_avg * L + SS)
5. Periodic Review Order-Up-To Level (Target Stock Level S):
       S = ceil(d_avg * (R + L) + SS)
   where R is review period in days (e.g. 7 days weekly review).
6. Inventory Position (IP):
       IP = On_Hand + On_Order - Backorders
7. Suggested Order Quantity (Q):
       Q = max(0, S - IP) if IP <= ROP else 0
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_SERVICE_LEVEL,
    MIN_SAFETY_STOCK,
    REVIEW_PERIOD_DAYS,
    SERVICE_LEVEL_Z_SCORE,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("inventory_calculations")


def calculate_safety_stock(
    daily_demand_std: float,
    lead_time_days: int,
    z_score: float = SERVICE_LEVEL_Z_SCORE,
    min_safety_stock: int = MIN_SAFETY_STOCK,
) -> int:
    """Calculate buffer safety stock to protect against demand variability during lead time.

    Formula:
        SS = max(min_safety_stock, ceil(Z * sigma_d * sqrt(L)))

    Assumptions:
    - Lead time L is positive and deterministic.
    - Daily demand exhibits finite variance sigma_d.
    """
    if lead_time_days <= 0:
        raise ValueError(f"Lead time days must be positive, got {lead_time_days}")
    if daily_demand_std < 0:
        raise ValueError(f"Daily demand standard deviation cannot be negative, got {daily_demand_std}")

    ss_continuous = z_score * daily_demand_std * np.sqrt(lead_time_days)
    safety_stock = int(max(min_safety_stock, np.ceil(ss_continuous)))
    return safety_stock


def calculate_reorder_point(
    avg_daily_demand: float,
    lead_time_days: int,
    safety_stock: int,
) -> int:
    """Calculate Reorder Point (ROP).

    Formula:
        ROP = ceil(avg_daily_demand * lead_time_days + safety_stock)
    """
    if avg_daily_demand < 0:
        raise ValueError(f"Average daily demand cannot be negative, got {avg_daily_demand}")
    if lead_time_days <= 0:
        raise ValueError(f"Lead time days must be positive, got {lead_time_days}")

    lead_time_demand = avg_daily_demand * lead_time_days
    rop = int(np.ceil(lead_time_demand + safety_stock))
    return rop


def calculate_target_stock_level(
    avg_daily_demand: float,
    lead_time_days: int,
    safety_stock: int,
    review_period_days: int = REVIEW_PERIOD_DAYS,
) -> int:
    """Calculate periodic review Order-Up-To Target Stock Level (S).

    Formula:
        S = ceil(avg_daily_demand * (review_period_days + lead_time_days) + safety_stock)
    """
    exposure_period = review_period_days + lead_time_days
    target_stock = int(np.ceil(avg_daily_demand * exposure_period + safety_stock))
    return target_stock


def classify_inventory_status(
    inventory_position: float,
    reorder_point: int,
    target_stock: int,
    safety_stock: int,
) -> str:
    """Classify current stock position into actionable operational states.

    Categories:
    - 'Critical / Stockout': IP <= 0
    - 'Reorder Required': IP <= ROP
    - 'Low Stock': ROP < IP <= ROP + (0.5 * SS)
    - 'Healthy Stock': ROP + (0.5 * SS) < IP <= target_stock * 1.25
    - 'Overstock Risk': IP > target_stock * 1.25
    """
    if inventory_position <= 0:
        return "Critical / Stockout"
    elif inventory_position <= reorder_point:
        return "Reorder Required"
    elif inventory_position <= (reorder_point + 0.5 * safety_stock):
        return "Low Stock"
    elif inventory_position > (target_stock * 1.25):
        return "Overstock Risk"
    else:
        return "Healthy Stock"


def evaluate_inventory_row(
    on_hand: int,
    on_order: int,
    backorders: int,
    avg_daily_demand: float,
    std_daily_demand: float,
    lead_time_days: int,
    review_period: int = REVIEW_PERIOD_DAYS,
    z_score: float = SERVICE_LEVEL_Z_SCORE,
) -> Dict[str, any]:
    """Execute complete inventory policy calculation for a single SKU/store series."""
    # Ensure physical non-negative constraints
    on_hand = max(0, int(on_hand))
    on_order = max(0, int(on_order))
    backorders = max(0, int(backorders))
    lead_time_days = max(1, int(lead_time_days))
    avg_daily_demand = max(0.01, float(avg_daily_demand))
    std_daily_demand = max(0.01, float(std_daily_demand))

    # Inventory Position
    inv_pos = on_hand + on_order - backorders

    # Metrics
    ss = calculate_safety_stock(std_daily_demand, lead_time_days, z_score=z_score)
    rop = calculate_reorder_point(avg_daily_demand, lead_time_days, ss)
    target_stock = calculate_target_stock_level(avg_daily_demand, lead_time_days, ss, review_period)

    # Replenishment Order Quantity (Order-up-to policy triggered at ROP)
    if inv_pos <= rop:
        order_qty = max(0, target_stock - inv_pos)
    else:
        order_qty = 0

    # Days of supply based on on-hand stock and daily demand
    days_of_supply = round(inv_pos / avg_daily_demand, 1)

    # Stockout risk flag: supply expires before replenishment can arrive
    stockout_risk = bool(days_of_supply < lead_time_days)

    status = classify_inventory_status(inv_pos, rop, target_stock, ss)

    return {
        "on_hand_stock": on_hand,
        "on_order_stock": on_order,
        "backorders": backorders,
        "inventory_position": inv_pos,
        "avg_daily_demand": round(avg_daily_demand, 2),
        "std_daily_demand": round(std_daily_demand, 2),
        "lead_time_days": lead_time_days,
        "safety_stock": ss,
        "reorder_point": rop,
        "target_stock": target_stock,
        "suggested_order_qty": order_qty,
        "days_of_supply": days_of_supply,
        "inventory_status": status,
        "stockout_risk": stockout_risk,
    }
