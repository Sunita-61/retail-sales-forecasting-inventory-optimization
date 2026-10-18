"""Multi-step recursive future sales forecasting module."""

from datetime import timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON_DAYS, FORECASTS_FILE, FORECASTS_OUTPUT_DIR
from src.forecasting.train_model import load_best_forecasting_model
from src.utils.logging_config import setup_logger

logger = setup_logger("predict")


def generate_recursive_future_forecast(
    historical_df: pd.DataFrame,
    model_artifact: Optional[dict] = None,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    save_forecasts: bool = True,
) -> pd.DataFrame:
    """Generate multi-step future sales forecasts using an autoregressive recursive strategy.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Cleaned historical sales dataset.
    model_artifact : Optional[dict]
        Loaded model dictionary. If None, loaded from disk.
    horizon_days : int
        Number of forward days to forecast (default 30).
    save_forecasts : bool
        Whether to export predictions to CSV.

    Returns
    -------
    pd.DataFrame
        Table containing forecast dates, store, product, point predictions, and uncertainty intervals.
    """
    if model_artifact is None:
        model_artifact = load_best_forecasting_model()

    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    feature_medians = pd.Series(model_artifact["feature_medians"])
    encoder = model_artifact["encoder"]
    rmse = model_artifact["test_metrics"]["RMSE"]

    logger.info("Generating %d-day recursive forecast using %s...", horizon_days, model_artifact["model_name"])

    df_hist = historical_df.copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    last_hist_date = df_hist["date"].max()

    # Get distinct store-product series
    unique_series = df_hist[["store_id", "product_id", "product_name", "category", "unit_price"]].drop_duplicates()

    # Pre-build history dictionary for fast lag and rolling lookups:
    # {(store_id, product_id): {date: units_sold}}
    history_records = {}
    for (s_id, p_id), grp in df_hist.groupby(["store_id", "product_id"]):
        history_records[(s_id, p_id)] = dict(zip(grp["date"], grp["units_sold"]))

    future_dates = [last_hist_date + timedelta(days=d) for d in range(1, horizon_days + 1)]
    forecast_rows = []

    # Retrieve empirical validation residual standard deviation for uncertainty band
    residual_std = float(model_artifact.get("val_residual_std", model_artifact["test_metrics"]["RMSE"]))

    # Iterate day-by-day (recursive multi-step)
    for f_date in future_dates:
        day_features_list = []
        meta_list = []

        for _, row in unique_series.iterrows():
            s_id = row["store_id"]
            p_id = row["product_id"]
            p_name = row["product_name"]
            cat = row["category"]
            price = row["unit_price"]

            # Construct calendar features
            dow = f_date.dayofweek
            is_wknd = 1 if dow >= 5 else 0
            sin_dow = np.sin(2 * np.pi * dow / 7)
            cos_dow = np.cos(2 * np.pi * dow / 7)
            sin_mon = np.sin(2 * np.pi * f_date.month / 12)
            cos_mon = np.cos(2 * np.pi * f_date.month / 12)

            # Retrieve past values (lags)
            series_dict = history_records[(s_id, p_id)]

            feat_dict = {
                "store_id": s_id,
                "product_id": p_id,
                "category": cat,
                "unit_price": price,
                "promotion_flag": 0,
                "day_of_week": dow,
                "day_of_month": f_date.day,
                "month": f_date.month,
                "year": f_date.year,
                "is_weekend": is_wknd,
                "sin_dow": sin_dow,
                "cos_dow": cos_dow,
                "sin_month": sin_mon,
                "cos_month": cos_mon,
            }

            for lag in [1, 7, 14, 21, 28]:
                lookup_dt = f_date - timedelta(days=lag)
                feat_dict[f"lag_{lag}"] = series_dict.get(lookup_dt, np.nan)

            for w in [7, 14, 28]:
                past_vals = [series_dict.get(f_date - timedelta(days=d), np.nan) for d in range(1, w + 1)]
                valid_vals = [v for v in past_vals if not np.isnan(v)]
                feat_dict[f"rolling_mean_{w}"] = np.mean(valid_vals) if valid_vals else np.nan
                feat_dict[f"rolling_std_{w}"] = np.std(valid_vals) if len(valid_vals) > 1 else 0.0

            day_features_list.append(feat_dict)
            meta_list.append((s_id, p_id, p_name, cat, price))

        # Convert day features to DataFrame
        day_df = pd.DataFrame(day_features_list)

        # Categorical encoding
        day_encoded = day_df.copy()
        existing_cats = [c for c in ["store_id", "product_id", "category"] if c in day_encoded.columns]
        day_encoded[existing_cats] = encoder.transform(day_encoded[existing_cats])

        # Prepare X_step
        X_step = day_encoded[feature_cols].fillna(feature_medians)

        # Predict
        step_preds = model.predict(X_step)
        step_preds = np.maximum(0, np.round(step_preds, 1))

        # Update historical lookup dict with newly generated predictions for subsequent recursive steps
        for i, (s_id, p_id, p_name, cat, price) in enumerate(meta_list):
            pred_val = float(step_preds[i])
            history_records[(s_id, p_id)][f_date] = pred_val

            # Empirical residual-based uncertainty band (approx. 90% coverage using normal approximation: +/- 1.645 * sigma_resid)
            half_width = 1.645 * residual_std
            lower_ci = max(0.0, round(pred_val - half_width, 1))
            upper_ci = max(lower_ci, round(pred_val + half_width, 1))
            assert lower_ci <= upper_ci, f"Lower bound ({lower_ci}) exceeded upper bound ({upper_ci})"

            forecast_rows.append(
                {
                    "date": f_date.strftime("%Y-%m-%d"),
                    "store_id": s_id,
                    "product_id": p_id,
                    "product_name": p_name,
                    "category": cat,
                    "unit_price": price,
                    "forecasted_demand": pred_val,
                    "forecast_lower_90ci": lower_ci,
                    "forecast_upper_90ci": upper_ci,
                    "forecast_lower_bound": lower_ci,
                    "forecast_upper_bound": upper_ci,
                    "forecast_revenue": round(pred_val * price, 2),
                    "uncertainty_interval_method": "Empirical residual uncertainty band (~90% normal approximation)",
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    logger.info("Successfully generated %d forecast records across %d days.", len(forecast_df), horizon_days)

    if save_forecasts:
        FORECASTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(FORECASTS_FILE, index=False)
        logger.info("Exported forecast dataset to %s", FORECASTS_FILE)

    return forecast_df
