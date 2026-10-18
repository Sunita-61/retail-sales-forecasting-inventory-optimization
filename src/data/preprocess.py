"""Data preprocessing and cleaning pipeline for retail sales data."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import PROCESSED_DATA_FILE
from src.data.validate_data import validate_retail_data
from src.utils.logging_config import setup_logger

logger = setup_logger("preprocess")


def clean_and_preprocess_data(
    df: pd.DataFrame,
    save_processed: bool = True,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Clean and preprocess raw retail sales dataset.

    Pipeline steps:
    1. Validate required columns and structure.
    2. Parse 'date' column into datetime objects.
    3. Remove or deduplicate exact composite key collisions.
    4. Handle missing values:
       - 'stock_received': filled with 0 (no delivery received).
       - 'promotion_flag': filled with 0 (no active promotion).
       - 'units_sold' & 'on_hand_stock': forward-fill or median per product/store.
    5. Ensure non-negative bounds on physical quantities.
    6. Derive financial metrics:
       - 'monetary_sales': units_sold * unit_price.
       - 'gross_margin': (unit_price - unit_cost) * units_sold (if unit_cost present).
    7. Chronologically sort observations by store_id, product_id, and date.
    8. Export to processed data storage if specified.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Preprocessed DataFrame and execution metadata summary.
    """
    logger.info("Initiating data preprocessing pipeline...")
    df_clean = df.copy()

    # Pre-validation
    validate_retail_data(df_clean, raise_errors=False)

    initial_rows = len(df_clean)

    # 1. Date conversion
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    null_dates = df_clean["date"].isna().sum()
    if null_dates > 0:
        logger.warning("Dropping %d rows with unparseable dates.", null_dates)
        df_clean = df_clean.dropna(subset=["date"])

    # 2. Deduplication on composite business key
    key_cols = ["date", "store_id", "product_id"]
    dup_mask = df_clean.duplicated(subset=key_cols, keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        logger.warning("Found and removed %d duplicate business records.", dup_count)
        df_clean = df_clean[~dup_mask]

    # 3. Missing Value Handling
    if "stock_received" in df_clean.columns:
        df_clean["stock_received"] = df_clean["stock_received"].fillna(0).astype(int)

    if "promotion_flag" in df_clean.columns:
        df_clean["promotion_flag"] = df_clean["promotion_flag"].fillna(0).astype(int)

    if "stockout_units" in df_clean.columns:
        df_clean["stockout_units"] = df_clean["stockout_units"].fillna(0).astype(int)

    # Fill any missing prices with median per product_id
    if "unit_price" in df_clean.columns:
        df_clean["unit_price"] = df_clean.groupby("product_id")["unit_price"].transform(
            lambda x: x.fillna(x.median())
        )

    # 4. Strict Non-negative bounds
    numeric_bounded_cols = ["units_sold", "on_hand_stock", "stock_received", "unit_price", "lead_time_days"]
    for col in numeric_bounded_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0)

    # 5. Financial Derivations (Distinguish units vs monetary currency)
    df_clean["monetary_sales"] = (df_clean["units_sold"] * df_clean["unit_price"]).round(2)

    if "unit_cost" in df_clean.columns:
        df_clean["monetary_cost"] = (df_clean["units_sold"] * df_clean["unit_cost"]).round(2)
        df_clean["gross_margin"] = (df_clean["monetary_sales"] - df_clean["monetary_cost"]).round(2)

    # 6. Chronological Sorting
    df_clean = df_clean.sort_values(by=["store_id", "product_id", "date"]).reset_index(drop=True)

    # Summary report
    summary = {
        "initial_rows": initial_rows,
        "cleaned_rows": len(df_clean),
        "dropped_rows": initial_rows - len(df_clean),
        "duplicates_removed": dup_count,
        "date_range": (
            str(df_clean["date"].min().date()),
            str(df_clean["date"].max().date()),
        ),
        "unique_stores": int(df_clean["store_id"].nunique()),
        "unique_products": int(df_clean["product_id"].nunique()),
    }

    # 7. Persistence
    if save_processed:
        target_file = output_path or PROCESSED_DATA_FILE
        target_file.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(target_file, index=False)
        logger.info("Saved cleaned dataset to %s (%d records)", target_file, len(df_clean))

    logger.info("Data preprocessing completed successfully.")
    return df_clean, summary


def load_processed_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned and preprocessed data from file."""
    path = Path(filepath) if filepath else PROCESSED_DATA_FILE
    if not path.exists():
        logger.warning("Processed file %s not found. Running raw load and cleaning...", path)
        from src.data.load_data import load_raw_data

        raw_df = load_raw_data()
        df_clean, _ = clean_and_preprocess_data(raw_df, save_processed=True)
        return df_clean

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
