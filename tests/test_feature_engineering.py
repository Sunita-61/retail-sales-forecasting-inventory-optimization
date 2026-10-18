"""Unit tests for feature engineering and dataset splitting."""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import (
    create_calendar_features,
    create_lag_and_rolling_features,
    prepare_feature_dataset,
    split_chronological_data,
)


@pytest.fixture
def sequence_sales_df():
    """Create a 40-day chronological single-series sequence."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store_id": "STR_01",
        "product_id": "PRD_001",
        "product_name": "Milk",
        "category": "Groceries",
        "unit_price": 3.99,
        "units_sold": np.arange(10, 50),  # strictly increasing sequence
        "promotion_flag": 0,
    })


def test_create_calendar_features(sequence_sales_df):
    """Verify temporal calendar feature generation."""
    feat_df = create_calendar_features(sequence_sales_df)
    assert "day_of_week" in feat_df.columns
    assert "day_of_month" in feat_df.columns
    assert "month" in feat_df.columns
    assert "is_weekend" in feat_df.columns
    assert "sin_dow" in feat_df.columns
    assert "cos_dow" in feat_df.columns
    assert feat_df["is_weekend"].isin([0, 1]).all()


def test_create_lags_no_leakage(sequence_sales_df):
    """Verify lag_1 and rolling_mean_7 strictly reference past values, avoiding target leakage."""
    feat_df = create_lag_and_rolling_features(sequence_sales_df, lags=[1, 7], windows=[7])

    # lag_1 for day i must equal units_sold of day i-1
    assert np.isnan(feat_df.loc[0, "lag_1"])
    assert feat_df.loc[1, "lag_1"] == feat_df.loc[0, "units_sold"]
    assert feat_df.loc[5, "lag_1"] == feat_df.loc[4, "units_sold"]

    # rolling_mean_7 for day 7 must be average of days 0 to 6 (not including day 7 target)
    expected_mean_day7 = sequence_sales_df.loc[0:6, "units_sold"].mean()
    assert np.isclose(feat_df.loc[7, "rolling_mean_7"], expected_mean_day7)


def test_prepare_feature_dataset(sequence_sales_df):
    """Verify full feature preparation pipeline."""
    df_feat, feature_cols, encoder = prepare_feature_dataset(sequence_sales_df, drop_na_lags=True)
    assert len(feature_cols) > 10
    assert "lag_28" in feature_cols
    assert "rolling_mean_7" in feature_cols
    # Verify no NaN values in selected feature matrix
    assert df_feat[feature_cols].isna().sum().sum() == 0


def test_split_chronological_data(sequence_sales_df):
    """Verify train/val/test splits do not overlap and preserve time ordering."""
    splits = split_chronological_data(
        sequence_sales_df, train_end="2024-01-20", val_end="2024-01-30"
    )
    assert len(splits["train"]) == 20
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    assert splits["train"]["date"].max() < splits["val"]["date"].min()
    assert splits["val"]["date"].max() < splits["test"]["date"].min()
