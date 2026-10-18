"""Data validation module for retail sales and inventory datasets."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import NUMERIC_COLUMNS_NON_NEGATIVE, REQUIRED_RAW_COLUMNS
from src.utils.logging_config import setup_logger

logger = setup_logger("validate_data")


class DataValidationError(ValueError):
    """Custom exception raised when retail dataset validation fails."""
    pass


def validate_retail_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    raise_errors: bool = True,
) -> Dict[str, Any]:
    """Validate retail sales dataset structure, data types, nulls, duplicates, and business rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to validate.
    required_columns : Optional[List[str]]
        List of required columns. Defaults to REQUIRED_RAW_COLUMNS from config.
    raise_errors : bool
        If True, raises DataValidationError on critical failures. If False, returns report.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing detailed validation status and metrics.
    """
    logger.info("Starting dataset validation...")
    if required_columns is None:
        required_columns = REQUIRED_RAW_COLUMNS

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Non-empty check
    if df is None or df.empty:
        err = "Validation Failed: Dataset is empty or None."
        logger.error(err)
        if raise_errors:
            raise DataValidationError(err)
        return {"is_valid": False, "errors": [err], "warnings": []}

    # 2. Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        err = f"Validation Failed: Missing required columns: {missing_cols}"
        errors.append(err)
        logger.error(err)

    # 3. Check duplicate records on composite key [date, store_id, product_id]
    key_cols = ["date", "store_id", "product_id"]
    if all(col in df.columns for col in key_cols):
        duplicate_count = df.duplicated(subset=key_cols).sum()
        if duplicate_count > 0:
            err = f"Validation Failed: Found {duplicate_count} duplicate records for key: {key_cols}"
            errors.append(err)
            logger.error(err)

    # 4. Check date column parseability
    if "date" in df.columns:
        converted_dates = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates_count = converted_dates.isna().sum()
        if invalid_dates_count > 0:
            err = f"Validation Failed: Found {invalid_dates_count} unparseable dates in 'date' column."
            errors.append(err)
            logger.error(err)

    # 5. Check missing values
    null_summary = df[required_columns].isnull().sum().to_dict() if not missing_cols else df.isnull().sum().to_dict()
    cols_with_nulls = {k: v for k, v in null_summary.items() if v > 0}
    if cols_with_nulls:
        warning_msg = f"Dataset contains missing values: {cols_with_nulls}"
        warnings.append(warning_msg)
        logger.warning(warning_msg)

    # 6. Check negative values in non-negative numeric fields
    negative_counts = {}
    for col in NUMERIC_COLUMNS_NON_NEGATIVE:
        if col in df.columns:
            # check non-numeric or negative
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            neg_count = (numeric_series < 0).sum()
            if neg_count > 0:
                negative_counts[col] = int(neg_count)
                err = f"Validation Failed: Column '{col}' contains {neg_count} negative values (must be non-negative)."
                errors.append(err)
                logger.error(err)

    # 7. Check lead time positivity (lead time must be > 0)
    if "lead_time_days" in df.columns:
        lt_series = pd.to_numeric(df["lead_time_days"], errors="coerce")
        zero_or_neg_lt = (lt_series <= 0).sum()
        if zero_or_neg_lt > 0:
            err = f"Validation Failed: 'lead_time_days' must be strictly positive, but found {zero_or_neg_lt} non-positive values."
            errors.append(err)
            logger.error(err)

    is_valid = len(errors) == 0

    report = {
        "is_valid": is_valid,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "errors": errors,
        "warnings": warnings,
        "null_counts": cols_with_nulls,
        "negative_counts": negative_counts,
    }

    if not is_valid and raise_errors:
        raise DataValidationError("; ".join(errors))

    logger.info("Dataset validation passed successfully. Total rows: %d.", len(df))
    return report
