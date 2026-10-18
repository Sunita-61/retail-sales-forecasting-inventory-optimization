"""Unit tests for dataset validation module."""

import pytest
import pandas as pd
from src.data.validate_data import validate_retail_data, DataValidationError
from src.config import REQUIRED_RAW_COLUMNS


@pytest.fixture
def valid_sample_df():
    """Create a minimal valid synthetic DataFrame."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "store_id": ["STR_01", "STR_01", "STR_01"],
        "product_id": ["PRD_001", "PRD_001", "PRD_001"],
        "product_name": ["Organic Milk", "Organic Milk", "Organic Milk"],
        "category": ["Groceries", "Groceries", "Groceries"],
        "unit_price": [3.99, 3.99, 3.99],
        "units_sold": [10, 15, 12],
        "on_hand_stock": [50, 40, 28],
        "stock_received": [0, 0, 0],
        "lead_time_days": [2, 2, 2],
        "promotion_flag": [0, 0, 1],
    })


def test_validate_valid_dataframe(valid_sample_df):
    """Test that a well-formed DataFrame passes validation with is_valid=True."""
    report = validate_retail_data(valid_sample_df, raise_errors=False)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["row_count"] == 3


def test_validate_empty_dataframe():
    """Test that empty DataFrame raises DataValidationError."""
    with pytest.raises(DataValidationError):
        validate_retail_data(pd.DataFrame(), raise_errors=True)


def test_validate_missing_required_columns(valid_sample_df):
    """Test that missing required columns are flagged as errors."""
    incomplete_df = valid_sample_df.drop(columns=["units_sold", "lead_time_days"])
    with pytest.raises(DataValidationError):
        validate_retail_data(incomplete_df, raise_errors=True)


def test_validate_negative_quantities(valid_sample_df):
    """Test that negative physical units or prices trigger validation errors."""
    corrupt_df = valid_sample_df.copy()
    corrupt_df.loc[0, "units_sold"] = -5
    with pytest.raises(DataValidationError):
        validate_retail_data(corrupt_df, raise_errors=True)


def test_validate_invalid_lead_time(valid_sample_df):
    """Test that zero or negative lead time triggers validation error."""
    corrupt_df = valid_sample_df.copy()
    corrupt_df.loc[0, "lead_time_days"] = 0
    with pytest.raises(DataValidationError):
        validate_retail_data(corrupt_df, raise_errors=True)


def test_validate_duplicate_records(valid_sample_df):
    """Test that duplicate composite keys [date, store_id, product_id] are flagged."""
    dup_df = pd.concat([valid_sample_df, valid_sample_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate_retail_data(dup_df, raise_errors=True)
