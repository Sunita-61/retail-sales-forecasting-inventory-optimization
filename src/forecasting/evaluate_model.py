"""Evaluation metrics and comparative model assessment for sales forecasting."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import METRICS_DIR, MODEL_METRICS_FILE
from src.utils.logging_config import setup_logger

logger = setup_logger("evaluate_model")


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, WAPE (Weighted Absolute Percentage Error), and R2.

    WAPE Formula:
        WAPE = (sum(|y_true - y_pred|) / sum(y_true)) * 100

    WAPE is preferred in retail demand forecasting over MAPE because it avoids
    infinite or explosive divisions when actual sales are zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))  # Predictions cannot be negative units

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    sum_true = float(np.sum(y_true))
    if sum_true > 0:
        wape = float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100.0)
    else:
        wape = 0.0

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE (%)": round(wape, 2),
        "R2": round(r2, 3),
    }


def compare_models(
    models_dict: Dict[str, any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Evaluate multiple trained models on validation set and compile comparison table."""
    logger.info("Evaluating models across validation set (%d samples)...", len(y_val))
    results = []

    for model_name, model in models_dict.items():
        preds = model.predict(X_val)
        metrics = calculate_forecast_metrics(y_val.to_numpy(), preds)
        results.append({"Model": model_name, **metrics})
        logger.info(
            "Model: %s | MAE: %.3f | RMSE: %.3f | WAPE: %.2f%% | R2: %.3f",
            model_name,
            metrics["MAE"],
            metrics["RMSE"],
            metrics["WAPE (%)"],
            metrics["R2"],
        )

    results_df = pd.DataFrame(results).sort_values(by="WAPE (%)").reset_index(drop=True)

    if save_csv:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(MODEL_METRICS_FILE, index=False)
        logger.info("Saved model comparison metrics to %s", MODEL_METRICS_FILE)

    return results_df
