"""Configuration settings and constants for the Retail Sales Forecasting & Inventory Optimization System."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FORECASTS_OUTPUT_DIR = OUTPUTS_DIR / "forecasts"
INVENTORY_OUTPUT_DIR = OUTPUTS_DIR / "inventory"
RECOMMENDATIONS_OUTPUT_DIR = OUTPUTS_DIR / "recommendations"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Default File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "synthetic_retail_sales.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "retail_sales_cleaned.csv"
SAMPLE_DATA_FILE = SAMPLE_DATA_DIR / "sample_sales.csv"

BEST_MODEL_FILE = MODELS_DIR / "best_forecasting_model.joblib"
MODEL_METRICS_FILE = METRICS_DIR / "model_comparison.csv"
FORECASTS_FILE = FORECASTS_OUTPUT_DIR / "future_sales_forecasts.csv"
INVENTORY_METRICS_FILE = INVENTORY_OUTPUT_DIR / "inventory_metrics.csv"
RECOMMENDATIONS_FILE = RECOMMENDATIONS_OUTPUT_DIR / "replenishment_recommendations.csv"

# Global Random Seed
RANDOM_SEED = 42

# Simulation Parameters (Synthetic Retail Dataset)
SIMULATION_START_DATE = "2024-01-01"
SIMULATION_END_DATE = "2025-12-31"  # 2 full years (731 days)

STORES_METADATA = [
    {"store_id": "STR_01", "store_name": "Metro Flagship", "store_type": "Urban Flagship", "demand_multiplier": 1.35},
    {"store_id": "STR_02", "store_name": "Westside Mall", "store_type": "Suburban Hypermarket", "demand_multiplier": 1.15},
    {"store_id": "STR_03", "store_name": "Downtown Central", "store_type": "Downtown Express", "demand_multiplier": 0.85},
    {"store_id": "STR_04", "store_name": "North Suburban", "store_type": "Regional Center", "demand_multiplier": 1.00},
]

PRODUCTS_METADATA = [
    # Groceries
    {"product_id": "PRD_001", "product_name": "Organic Whole Milk 1L", "category": "Groceries", "unit_price": 3.99, "unit_cost": 2.20, "base_demand": 45, "lead_time_days": 2, "shelf_life_days": 14},
    {"product_id": "PRD_002", "product_name": "Artisan Sourdough Bread", "category": "Groceries", "unit_price": 4.50, "unit_cost": 2.00, "base_demand": 35, "lead_time_days": 2, "shelf_life_days": 7},
    {"product_id": "PRD_003", "product_name": "Greek Honey Yogurt 500g", "category": "Groceries", "unit_price": 4.25, "unit_cost": 2.30, "base_demand": 28, "lead_time_days": 3, "shelf_life_days": 21},
    # Electronics
    {"product_id": "PRD_004", "product_name": "Noise-Cancelling Headphones", "category": "Electronics", "unit_price": 149.99, "unit_cost": 85.00, "base_demand": 6, "lead_time_days": 7, "shelf_life_days": 365},
    {"product_id": "PRD_005", "product_name": "Smart Fitness Tracker Band", "category": "Electronics", "unit_price": 79.99, "unit_cost": 42.00, "base_demand": 10, "lead_time_days": 6, "shelf_life_days": 365},
    # Apparel
    {"product_id": "PRD_006", "product_name": "Classic Cotton Crew T-Shirt", "category": "Apparel", "unit_price": 19.99, "unit_cost": 8.50, "base_demand": 22, "lead_time_days": 5, "shelf_life_days": 365},
    {"product_id": "PRD_007", "product_name": "Stretch Slim-Fit Denim Jeans", "category": "Apparel", "unit_price": 49.99, "unit_cost": 22.00, "base_demand": 14, "lead_time_days": 7, "shelf_life_days": 365},
    # Home & Kitchen
    {"product_id": "PRD_008", "product_name": "Cast Iron Skillet 10-inch", "category": "Home & Kitchen", "unit_price": 34.99, "unit_cost": 16.50, "base_demand": 8, "lead_time_days": 6, "shelf_life_days": 365},
    {"product_id": "PRD_009", "product_name": "Ceramic Glazed Coffee Mug", "category": "Home & Kitchen", "unit_price": 12.50, "unit_cost": 4.50, "base_demand": 18, "lead_time_days": 4, "shelf_life_days": 365},
    {"product_id": "PRD_010", "product_name": "Ultrasonic Aroma Diffuser", "category": "Home & Kitchen", "unit_price": 29.99, "unit_cost": 13.00, "base_demand": 12, "lead_time_days": 5, "shelf_life_days": 365},
]

# Required Schema for Raw Data Validation
REQUIRED_RAW_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "product_name",
    "category",
    "unit_price",
    "units_sold",
    "on_hand_stock",
    "stock_received",
    "lead_time_days",
    "promotion_flag"
]

NUMERIC_COLUMNS_NON_NEGATIVE = [
    "unit_price",
    "units_sold",
    "on_hand_stock",
    "stock_received",
    "lead_time_days",
]

# Feature Engineering Settings
TIME_SERIES_GROUP_KEYS = ["store_id", "product_id"]
TARGET_VARIABLE = "units_sold"
LAG_PERIODS = [1, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]

# Chronological Split Cutoffs
# Total: 2024-01-01 to 2025-12-31 (24 months)
# Train: 2024-01-01 to 2025-06-30 (18 months)
# Validation: 2025-07-01 to 2025-09-30 (3 months)
# Test: 2025-10-01 to 2025-12-31 (3 months)
TRAIN_END_DATE = "2025-06-30"
VAL_END_DATE = "2025-09-30"

# Inventory Optimization Settings
DEFAULT_SERVICE_LEVEL = 0.95  # 95% cycle service level
SERVICE_LEVEL_Z_SCORE = 1.645  # Standard normal distribution critical value for 95%
REVIEW_PERIOD_DAYS = 7  # Weekly periodic replenishment review policy
MIN_SAFETY_STOCK = 2  # Absolute minimum safety buffer in units
HOLDING_COST_ANNUAL_RATE = 0.20  # 20% annual inventory holding rate
STOCKOUT_PENALTY_MULTIPLIER = 1.5  # Lost profit + reputational penalty factor
FORECAST_HORIZON_DAYS = 30  # Standard operational forecast horizon
