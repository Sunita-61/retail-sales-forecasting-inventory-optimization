"""Unit tests for forecasting models and evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
from src.forecasting.baseline import SeasonalNaiveBaseline, MovingAverageBaseline
from src.forecasting.evaluate_model import calculate_forecast_metrics, compare_models


def test_calculate_forecast_metrics_perfect_predictions():
    """Verify MAE, RMSE, WAPE are zero when predictions match ground truth exactly."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0])

    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["WAPE (%)"] == 0.0
    assert metrics["R2"] == 1.0


def test_calculate_forecast_metrics_known_error():
    """Verify WAPE calculation with known offsets."""
    y_true = np.array([10.0, 10.0, 10.0, 10.0])  # sum = 40
    y_pred = np.array([12.0, 8.0, 12.0, 8.0])    # sum |errors| = 8 -> WAPE = 8/40 = 20%

    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert metrics["MAE"] == 2.0
    assert metrics["WAPE (%)"] == 20.0


def test_seasonal_naive_baseline():
    """Verify SeasonalNaiveBaseline correctly outputs lag_7 values."""
    X = pd.DataFrame({
        "lag_7": [15.0, 22.0, 18.0],
        "lag_1": [10.0, 12.0, 11.0],
    })
    y = pd.Series([16.0, 20.0, 19.0])

    model = SeasonalNaiveBaseline().fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, np.array([15.0, 22.0, 18.0]))


def test_moving_average_baseline():
    """Verify MovingAverageBaseline correctly outputs rolling_mean_7."""
    X = pd.DataFrame({
        "rolling_mean_7": [14.5, 16.0, 17.5],
    })
    y = pd.Series([15.0, 16.0, 18.0])

    model = MovingAverageBaseline().fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, np.array([14.5, 16.0, 17.5]))


def test_compare_models_ranking():
    """Verify model comparison correctly ranks candidates by WAPE."""
    X = pd.DataFrame({"lag_7": [10.0, 20.0], "rolling_mean_7": [12.0, 22.0]})
    y = pd.Series([10.0, 20.0])  # Perfect match for lag_7

    models = {
        "Seasonal Naive": SeasonalNaiveBaseline(),
        "Moving Average": MovingAverageBaseline(),
    }
    for m in models.values():
        m.fit(X, y)

    comparison_df = compare_models(models, X, y, save_csv=False)
    assert comparison_df.iloc[0]["Model"] == "Seasonal Naive"
    assert comparison_df.iloc[0]["WAPE (%)"] == 0.0


def test_uncertainty_bounds_ordering():
    """Verify forecast uncertainty intervals satisfy non-negativity and lower <= upper."""
    from src.forecasting.predict import generate_recursive_future_forecast
    from src.forecasting.train_model import load_best_forecasting_model
    from src.data.preprocess import load_processed_data

    clean_df = load_processed_data()
    artifact = load_best_forecasting_model()
    
    assert "val_residual_std" in artifact
    assert artifact["val_residual_std"] >= 0

    fc_df = generate_recursive_future_forecast(clean_df.head(2000), model_artifact=artifact, horizon_days=3, save_forecasts=False)
    assert len(fc_df) > 0
    assert (fc_df["forecast_lower_bound"] <= fc_df["forecast_upper_bound"]).all()
    assert (fc_df["forecast_lower_bound"] >= 0).all()

