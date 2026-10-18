"""Script to generate future forecasts and inventory replenishment recommendations."""

import argparse
from src.data.preprocess import load_processed_data
from src.forecasting.predict import generate_recursive_future_forecast
from src.forecasting.train_model import load_best_forecasting_model
from src.inventory.reorder_recommendations import generate_replenishment_recommendations
from src.utils.logging_config import setup_logger
from src.visualization.plots import plot_forecast_vs_actual, plot_inventory_status_breakdown
from src.config import FIGURES_DIR

logger = setup_logger("generate_forecast_script")


def main():
    parser = argparse.ArgumentParser(description="Generate future forecasts and replenishment recommendations.")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days (default: 30)")
    args = parser.parse_args()

    logger.info("Loading preprocessed sales data and trained model...")
    clean_df = load_processed_data()
    model_artifact = load_best_forecasting_model()

    logger.info("=== STEP 1: GENERATE MULTI-STEP FUTURE FORECASTS ===")
    forecast_df = generate_recursive_future_forecast(
        clean_df, model_artifact=model_artifact, horizon_days=args.horizon, save_forecasts=True
    )

    logger.info("=== STEP 2: GENERATE REPLENISHMENT RECOMMENDATIONS ===")
    rec_df = generate_replenishment_recommendations(
        clean_df, forecast_df=forecast_df, save_outputs=True
    )

    logger.info("=== STEP 3: EXPORT OPERATIONAL CHARTS ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_forecast_vs_actual(clean_df, forecast_df, store_id="STR_01", product_id="PRD_001", save_path=FIGURES_DIR / "08_forecast_vs_actual_sample.png")
    plot_inventory_status_breakdown(rec_df, save_path=FIGURES_DIR / "09_inventory_status_breakdown.png")

    logger.info("Forecasting and inventory optimization pipeline finished successfully.")


if __name__ == "__main__":
    main()
