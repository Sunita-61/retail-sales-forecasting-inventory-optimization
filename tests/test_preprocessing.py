"""Unit tests for preprocessing and data cleaning pipeline."""

import pytest
import pandas as pd
from src.data.preprocess import clean_and_preprocess_data


@pytest.fixture
def raw_test_df():
    """Create sample raw dataset with missing values and unsorted dates."""
    return pd.DataFrame({
        "date": ["2024-01-05", "2024-01-01", "2024-01-02", "2024-01-01"],
        "store_id": ["STR_01", "STR_01", "STR_01", "STR_01"],
        "product_id": ["PRD_001", "PRD_001", "PRD_001", "PRD_001"],
        "product_name": ["Milk", "Milk", "Milk", "Milk"],
        "category": ["Groceries", "Groceries", "Groceries", "Groceries"],
        "unit_price": [4.0, 4.0, 4.0, 4.0],
        "unit_cost": [2.5, 2.5, 2.5, 2.5],
        "units_sold": [10, 5, 8, 5],
        "on_hand_stock": [100, 110, 102, 110],
        "stock_received": [None, 0, None, 0],  # missing values
        "lead_time_days": [2, 2, 2, 2],
        "promotion_flag": [None, 0, 1, 0],
    })


def test_clean_and_preprocess_removes_duplicates(raw_test_df):
    """Verify that duplicate rows on [date, store_id, product_id] are removed."""
    cleaned_df, summary = clean_and_preprocess_data(raw_test_df, save_processed=False)
    assert len(cleaned_df) == 3
    assert summary["duplicates_removed"] == 1


def test_clean_and_preprocess_fills_missing(raw_test_df):
    """Verify missing stock_received and promotion_flag are imputed safely with 0."""
    cleaned_df, _ = clean_and_preprocess_data(raw_test_df, save_processed=False)
    assert cleaned_df["stock_received"].isna().sum() == 0
    assert cleaned_df["promotion_flag"].isna().sum() == 0
    assert (cleaned_df["stock_received"] >= 0).all()


def test_clean_and_preprocess_financial_metrics(raw_test_df):
    """Verify monetary sales and gross margin calculation."""
    cleaned_df, _ = clean_and_preprocess_data(raw_test_df, save_processed=False)
    assert "monetary_sales" in cleaned_df.columns
    assert "gross_margin" in cleaned_df.columns

    expected_sales = cleaned_df["units_sold"] * cleaned_df["unit_price"]
    assert (cleaned_df["monetary_sales"] == expected_sales).all()


def test_clean_and_preprocess_chronological_order(raw_test_df):
    """Verify output DataFrame is sorted chronologically."""
    cleaned_df, _ = clean_and_preprocess_data(raw_test_df, save_processed=False)
    dates = cleaned_df["date"].tolist()
    assert dates == sorted(dates)
