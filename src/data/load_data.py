"""Synthetic retail sales and inventory data generator and loader."""

from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    FORECAST_HORIZON_DAYS,
    PRODUCTS_METADATA,
    RANDOM_SEED,
    RAW_DATA_FILE,
    SAMPLE_DATA_FILE,
    SERVICE_LEVEL_Z_SCORE,
    SIMULATION_END_DATE,
    SIMULATION_START_DATE,
    STORES_METADATA,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("load_data")


def generate_synthetic_retail_dataset(
    start_date: str = SIMULATION_START_DATE,
    end_date: str = SIMULATION_END_DATE,
    seed: int = RANDOM_SEED,
    save_raw: bool = True,
    save_sample: bool = True,
) -> pd.DataFrame:
    """Generate a realistic, reproducible daily synthetic retail dataset.

    Simulates daily demand with:
    - Store-level and product-level base demand
    - Day-of-week seasonality (weekend surges)
    - Monthly/Holiday seasonality (Q4 surges, summer patterns)
    - Long-term upward business growth trend
    - Promotional campaigns with demand uplifts
    - Realistic inventory replenishment, lead times, stock receipts, and stockouts
    - Explicit non-negative inventory constraints

    NOTE: This dataset is purely simulated for educational and portfolio demonstration.
    It does not reflect internal records of any commercial retailer.
    """
    logger.info("Initializing synthetic retail dataset generation (seed=%d)...", seed)
    rng = np.random.default_rng(seed)

    # Date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    total_days = len(date_range)
    logger.info("Simulation period: %s to %s (%d days)", start_date, end_date, total_days)

    all_records = []

    for store in STORES_METADATA:
        s_id = store["store_id"]
        s_mult = store["demand_multiplier"]

        for prod in PRODUCTS_METADATA:
            p_id = prod["product_id"]
            p_name = prod["product_name"]
            p_cat = prod["category"]
            p_price = prod["unit_price"]
            p_cost = prod["unit_cost"]
            p_base = prod["base_demand"]
            lead_time = prod["lead_time_days"]

            # Estimate historical variability to establish realistic initial stock
            daily_std_est = max(1.5, p_base * 0.25)
            safety_stock_est = int(np.ceil(SERVICE_LEVEL_Z_SCORE * daily_std_est * np.sqrt(lead_time)))
            reorder_point_est = int(np.ceil(p_base * s_mult * lead_time + safety_stock_est))
            target_stock_est = int(np.ceil(reorder_point_est + (p_base * s_mult * 7)))

            # State variables for inventory simulation
            current_on_hand = int(target_stock_est * 0.9)  # Start near target
            pipeline_orders = {}  # {delivery_date_index: order_quantity}

            # Generate promotional calendar (3-5 promotions a year, 3-7 days each)
            promo_days = np.zeros(total_days, dtype=int)
            num_promos = rng.integers(3, 7) * 2  # across 2 years
            for _ in range(num_promos):
                promo_start = rng.integers(10, total_days - 10)
                promo_len = rng.integers(3, 8)
                promo_days[promo_start : min(total_days, promo_start + promo_len)] = 1

            for day_idx, current_date in enumerate(date_range):
                # 1. Process arriving stock receipts
                stock_received = pipeline_orders.pop(day_idx, 0)
                current_on_hand += stock_received

                # 2. Demand calculations
                # Trend factor: 6% growth per year
                trend_factor = 1.0 + (0.06 * (day_idx / 365.25))

                # Day-of-week seasonality
                dow = current_date.dayofweek  # 0=Monday, 6=Sunday
                dow_factors = {0: 0.90, 1: 0.88, 2: 0.92, 3: 0.95, 4: 1.18, 5: 1.45, 6: 1.30}
                dow_mult = dow_factors.get(dow, 1.0)

                # Monthly/Holiday seasonality
                month = current_date.month
                month_factors = {
                    1: 0.92, 2: 0.90, 3: 0.98, 4: 1.00, 5: 1.02, 6: 1.05,
                    7: 1.08, 8: 1.06, 9: 0.98, 10: 1.05, 11: 1.20, 12: 1.40
                }
                # Electronics and apparel have higher surge in Q4
                if p_cat in ["Electronics", "Apparel"] and month in [11, 12]:
                    month_mult = month_factors[month] * 1.15
                elif p_cat == "Groceries" and month in [6, 7, 8]:
                    month_mult = month_factors[month] * 1.05
                else:
                    month_mult = month_factors.get(month, 1.0)

                # Promotion lift
                is_promo = int(promo_days[day_idx])
                promo_mult = 1.45 if is_promo == 1 else 1.0

                # Expected daily latent demand
                mean_demand = p_base * s_mult * trend_factor * dow_mult * month_mult * promo_mult

                # Stochastic realization (Poisson for low demand items, normal for high demand items)
                if mean_demand < 15:
                    simulated_demand = int(rng.poisson(lam=mean_demand))
                else:
                    simulated_demand = int(max(0, np.round(rng.normal(loc=mean_demand, scale=daily_std_est * 1.1))))

                # 3. Inventory execution and physical sales capping
                if simulated_demand <= current_on_hand:
                    units_sold = simulated_demand
                    stockout_units = 0
                    current_on_hand -= units_sold
                else:
                    # Stockout occurred! Demand exceeded available stock
                    units_sold = current_on_hand
                    stockout_units = simulated_demand - current_on_hand
                    current_on_hand = 0

                # Ensure non-negative stock strictly
                current_on_hand = max(0, current_on_hand)

                # 4. Replenishment ordering simulation
                # Pipeline inventory currently on order
                on_order = sum(pipeline_orders.values())
                inventory_position = current_on_hand + on_order

                # If inventory position falls to or below ROP, trigger order up to target stock
                if inventory_position <= reorder_point_est:
                    order_qty = max(0, target_stock_est - inventory_position)
                    if order_qty > 0:
                        delivery_day = day_idx + lead_time
                        pipeline_orders[delivery_day] = pipeline_orders.get(delivery_day, 0) + order_qty

                # Record observation
                all_records.append(
                    {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "store_id": s_id,
                        "product_id": p_id,
                        "product_name": p_name,
                        "category": p_cat,
                        "unit_price": float(round(p_price, 2)),
                        "unit_cost": float(round(p_cost, 2)),
                        "units_demanded": int(simulated_demand),
                        "units_sold": int(units_sold),
                        "stockout_units": int(stockout_units),
                        "on_hand_stock": int(current_on_hand),
                        "stock_received": int(stock_received),
                        "lead_time_days": int(lead_time),
                        "promotion_flag": int(is_promo),
                        "data_source": "Simulated Synthetic Data - Not Actual Company Records",
                    }
                )

    df = pd.DataFrame(all_records)
    logger.info("Generated %d synthetic records across %d stores and %d products.", len(df), len(STORES_METADATA), len(PRODUCTS_METADATA))

    if save_raw:
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_FILE, index=False)
        logger.info("Saved raw synthetic dataset to %s", RAW_DATA_FILE)

    if save_sample:
        SAMPLE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Store representative sample (first 100 rows + some promo rows)
        sample_df = pd.concat([df.head(50), df[df["promotion_flag"] == 1].head(50)]).drop_duplicates()
        sample_df.to_csv(SAMPLE_DATA_FILE, index=False)
        logger.info("Saved sample dataset to %s (%d rows)", SAMPLE_DATA_FILE, len(sample_df))

    return df


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load raw dataset from disk or generate if not present."""
    target_path = Path(filepath) if filepath else RAW_DATA_FILE
    if not target_path.exists():
        logger.warning("File %s not found. Triggering synthetic generator...", target_path)
        return generate_synthetic_retail_dataset(save_raw=True)

    logger.info("Loading raw dataset from %s...", target_path)
    df = pd.read_csv(target_path)
    logger.info("Successfully loaded raw dataset with %d rows and %d columns.", len(df), len(df.columns))
    return df
