"""Baseline heuristic models for time-series sales forecasting."""

from typing import Optional
import numpy as np
import pandas as pd


class SeasonalNaiveBaseline:
    """Seasonal Naive baseline: predicts demand equal to the same day from prior week (lag_7).

    If lag_7 is missing, falls back to lag_1 or series mean.
    """

    def __init__(self, fallback_col: str = "lag_1"):
        self.fallback_col = fallback_col
        self.default_mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit baseline by storing overall training target mean for edge cases."""
        self.default_mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions based on lag_7."""
        if "lag_7" in X.columns:
            preds = X["lag_7"].copy()
            if self.fallback_col in X.columns:
                preds = preds.fillna(X[self.fallback_col])
            preds = preds.fillna(self.default_mean)
            return np.maximum(0, preds.to_numpy())
        elif self.fallback_col in X.columns:
            return np.maximum(0, X[self.fallback_col].fillna(self.default_mean).to_numpy())
        else:
            return np.full(len(X), self.default_mean)


class MovingAverageBaseline:
    """Moving Average baseline: predicts demand equal to recent 7-day rolling mean."""

    def __init__(self, rolling_col: str = "rolling_mean_7"):
        self.rolling_col = rolling_col
        self.default_mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit baseline by storing overall training target mean."""
        self.default_mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions based on rolling mean."""
        if self.rolling_col in X.columns:
            preds = X[self.rolling_col].fillna(self.default_mean).to_numpy()
            return np.maximum(0, preds)
        return np.full(len(X), self.default_mean)
