"""Model training, selection, and serialization pipeline for retail sales forecasting."""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import BEST_MODEL_FILE, METRICS_DIR, MODELS_DIR, RANDOM_SEED, TARGET_VARIABLE
from src.features.feature_engineering import (
    prepare_feature_dataset,
    split_chronological_data,
)
from src.forecasting.baseline import MovingAverageBaseline, SeasonalNaiveBaseline
from src.forecasting.evaluate_model import calculate_forecast_metrics, compare_models
from src.utils.logging_config import setup_logger

logger = setup_logger("train_model")


def train_and_select_best_model(
    df: pd.DataFrame,
    save_model: bool = True,
) -> Tuple[any, pd.DataFrame, Dict[str, any]]:
    """Train candidate models, compare on validation set, evaluate on test set, and save champion model.

    Returns
    -------
    Tuple[any, pd.DataFrame, Dict[str, any]]
        Best model, validation comparison DataFrame, and test evaluation summary.
    """
    logger.info("Starting model training pipeline...")

    # 1. Feature Engineering
    df_feat, feature_cols, encoder = prepare_feature_dataset(df, drop_na_lags=True, is_training=True)

    # 2. Chronological Split
    splits = split_chronological_data(df_feat)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_VARIABLE]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_VARIABLE]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_VARIABLE]

    # Impute any edge NaN in features with training median
    feature_medians = X_train.median()
    X_train_clean = X_train.fillna(feature_medians)
    X_val_clean = X_val.fillna(feature_medians)
    X_test_clean = X_test.fillna(feature_medians)

    # 3. Instantiate Candidates
    candidates = {
        "Seasonal Naive (Lag 7)": SeasonalNaiveBaseline(),
        "7-Day Moving Average": MovingAverageBaseline(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_SEED),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    # 4. Train Candidates on Training Split
    logger.info("Training candidate models...")
    trained_models = {}
    for name, model in candidates.items():
        logger.info("Fitting %s...", name)
        model.fit(X_train_clean, y_train)
        trained_models[name] = model

    # 5. Evaluate and Compare on Validation Split
    comparison_df = compare_models(trained_models, X_val_clean, y_val, save_csv=True)

    # 6. Select Champion Model based on lowest WAPE
    champion_name = comparison_df.iloc[0]["Model"]
    champion_model = trained_models[champion_name]
    logger.info("Selected Champion Model: %s (Validation WAPE: %.2f%%)", champion_name, comparison_df.iloc[0]["WAPE (%)"])

    # 7. Evaluate Champion on Held-Out Test Split
    test_preds = champion_model.predict(X_test_clean)
    test_metrics = calculate_forecast_metrics(y_test.to_numpy(), test_preds)
    logger.info("Held-Out Test Results for %s: %s", champion_name, test_metrics)

    # 8. Save Model and Metadata Artifact
    artifact = {
        "model_name": champion_name,
        "model": champion_model,
        "feature_cols": feature_cols,
        "feature_medians": feature_medians.to_dict(),
        "encoder": encoder,
        "validation_metrics": comparison_df.iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "target_col": TARGET_VARIABLE,
    }

    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, BEST_MODEL_FILE)
        logger.info("Champion model artifact saved to %s", BEST_MODEL_FILE)

    return champion_model, comparison_df, artifact


def load_best_forecasting_model(model_path: Optional[Path] = None) -> dict:
    """Load serialized champion model and accompanying feature metadata."""
    target_path = model_path or BEST_MODEL_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {target_path}. Please train models first.")
    return joblib.load(target_path)
