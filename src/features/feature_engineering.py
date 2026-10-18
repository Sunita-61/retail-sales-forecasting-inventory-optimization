"""Feature engineering for retail time-series sales forecasting."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    LAG_PERIODS,
    ROLLING_WINDOWS,
    TARGET_VARIABLE,
    TIME_SERIES_GROUP_KEYS,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("feature_engineering")


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal and calendar features from 'date' column."""
    df_feat = df.copy()
    if not np.issubdtype(df_feat["date"].dtype, np.datetime64):
        df_feat["date"] = pd.to_datetime(df_feat["date"])

    df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
    df_feat["day_of_month"] = df_feat["date"].dt.day
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["year"] = df_feat["date"].dt.year
    df_feat["quarter"] = df_feat["date"].dt.quarter
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)

    # Cyclical representations
    df_feat["sin_dow"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["cos_dow"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    return df_feat


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    group_keys: Optional[List[str]] = None,
    target_col: str = TARGET_VARIABLE,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generate strictly past lagged and rolling statistics per group to prevent data leakage.

    NOTE: Rolling aggregations use shift(1) prior to calculating rolling window,
    ensuring that the current day's target is never leaked into the input features.
    """
    if group_keys is None:
        group_keys = TIME_SERIES_GROUP_KEYS
    if lags is None:
        lags = LAG_PERIODS
    if windows is None:
        windows = ROLLING_WINDOWS

    df_sorted = df.sort_values(by=group_keys + ["date"]).copy()

    # Lag features
    for lag in lags:
        df_sorted[f"lag_{lag}"] = (
            df_sorted.groupby(group_keys)[target_col].shift(lag)
        )

    # Rolling window statistics (shifted by 1 day to prevent leakage)
    for w in windows:
        shifted_target = df_sorted.groupby(group_keys)[target_col].shift(1)
        df_sorted[f"rolling_mean_{w}"] = (
            shifted_target.groupby([df_sorted[k] for k in group_keys])
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
        df_sorted[f"rolling_std_{w}"] = (
            shifted_target.groupby([df_sorted[k] for k in group_keys])
            .rolling(window=w, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(drop=True)
        )

    return df_sorted


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    encoder: Optional[OrdinalEncoder] = None,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    """Encode categorical columns using OrdinalEncoder fitted exclusively on training data."""
    if categorical_cols is None:
        categorical_cols = ["store_id", "product_id", "category"]

    df_encoded = df.copy()
    existing_cat = [c for c in categorical_cols if c in df_encoded.columns]

    if is_training or encoder is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_encoded[existing_cat] = encoder.fit_transform(df_encoded[existing_cat])
    else:
        df_encoded[existing_cat] = encoder.transform(df_encoded[existing_cat])

    return df_encoded, encoder


def prepare_feature_dataset(
    df: pd.DataFrame,
    drop_na_lags: bool = True,
    encoder: Optional[OrdinalEncoder] = None,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, List[str], OrdinalEncoder]:
    """Complete feature engineering pipeline.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], OrdinalEncoder]
        Feature-engineered DataFrame, list of feature column names, and the categorical encoder.
    """
    logger.info("Building calendar features...")
    df_feat = create_calendar_features(df)

    logger.info("Building lag and rolling window features (leakage-free)...")
    df_feat = create_lag_and_rolling_features(df_feat)

    logger.info("Encoding categorical attributes...")
    df_feat, enc = encode_categorical_features(df_feat, encoder=encoder, is_training=is_training)

    if drop_na_lags:
        max_lag = max(LAG_PERIODS)
        initial_len = len(df_feat)
        df_feat = df_feat.dropna(subset=[f"lag_{max_lag}"]).reset_index(drop=True)
        logger.info("Dropped initial %d rows with uninitialized lag features.", initial_len - len(df_feat))

    feature_cols = [
        "store_id",
        "product_id",
        "category",
        "unit_price",
        "promotion_flag",
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "is_weekend",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
    ] + [f"lag_{l}" for l in LAG_PERIODS] + [
        f"rolling_mean_{w}" for w in ROLLING_WINDOWS
    ] + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]

    # Ensure all feature columns exist
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    return df_feat, feature_cols, enc


def split_chronological_data(
    df: pd.DataFrame,
    train_end: str = TRAIN_END_DATE,
    val_end: str = VAL_END_DATE,
) -> Dict[str, pd.DataFrame]:
    """Perform chronological train / validation / test splits to prevent future information leakage."""
    train_mask = df["date"] <= pd.to_datetime(train_end)
    val_mask = (df["date"] > pd.to_datetime(train_end)) & (df["date"] <= pd.to_datetime(val_end))
    test_mask = df["date"] > pd.to_datetime(val_end)

    splits = {
        "train": df[train_mask].copy(),
        "val": df[val_mask].copy(),
        "test": df[test_mask].copy(),
    }

    logger.info(
        "Chronological splits created: Train=%d rows (%s to %s), Val=%d rows, Test=%d rows (%s to %s)",
        len(splits["train"]),
        splits["train"]["date"].min().date(),
        splits["train"]["date"].max().date(),
        len(splits["val"]),
        len(splits["test"]),
        splits["test"]["date"].min().date(),
        splits["test"]["date"].max().date(),
    )
    return splits
