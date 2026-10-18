"""Main orchestration pipeline for Retail Sales Forecasting & Inventory Optimization System."""

import argparse
import sys
from src.data.load_data import generate_synthetic_retail_dataset, load_raw_data
from src.data.preprocess import clean_and_preprocess_data, load_processed_data
from src.data.validate_data import validate_retail_data
from src.forecasting.predict import generate_recursive_future_forecast
from src.forecasting.train_model import load_best_forecasting_model, train_and_select_best_model
from src.inventory.reorder_recommendations import generate_replenishment_recommendations
from src.utils.logging_config import setup_logger
from src.visualization.plots import generate_all_eda_figures, plot_forecast_vs_actual, plot_inventory_status_breakdown
from src.config import FIGURES_DIR

logger = setup_logger("main_pipeline")


def run_pipeline(
    regenerate_data: bool = False,
    run_training: bool = True,
    run_forecasting: bool = True,
    run_inventory: bool = True,
    generate_figures: bool = True,
):
    """Execute complete end-to-end pipeline."""
    print("=" * 80)
    print("  RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION SYSTEM - END-TO-END PIPELINE")
    print("=" * 80)

    # 1. Data Generation / Loading
    logger.info(">>> STEP 1: DATA INGESTION & GENERATION")
    if regenerate_data:
        raw_df = generate_synthetic_retail_dataset(save_raw=True)
    else:
        raw_df = load_raw_data()

    # 2. Validation
    logger.info(">>> STEP 2: DATA VALIDATION")
    val_report = validate_retail_data(raw_df, raise_errors=True)
    logger.info("Validation successful: %d records, 0 critical schema errors.", val_report["row_count"])

    # 3. Preprocessing
    logger.info(">>> STEP 3: PREPROCESSING & CLEANING")
    clean_df, summary = clean_and_preprocess_data(raw_df, save_processed=True)
    logger.info("Data cleaned and saved: %d records.", len(clean_df))

    # 4. EDA Figures
    if generate_figures:
        logger.info(">>> STEP 4: EXPLORATORY DATA ANALYSIS & REPORT FIGURES")
        generate_all_eda_figures(clean_df)

    # 5. Model Training & Evaluation
    model_artifact = None
    if run_training:
        logger.info(">>> STEP 5: TIME-SERIES MODEL TRAINING & EVALUATION")
        champion_model, comparison_df, model_artifact = train_and_select_best_model(clean_df, save_model=True)
        print("\n--- Model Benchmark Results ---")
        print(comparison_df.to_string(index=False))
        print("--------------------------------\n")
    else:
        try:
            model_artifact = load_best_forecasting_model()
        except FileNotFoundError:
            logger.warning("No saved model found. Forcing model training...")
            champion_model, comparison_df, model_artifact = train_and_select_best_model(clean_df, save_model=True)

    # 6. Forecasting
    forecast_df = None
    if run_forecasting:
        logger.info(">>> STEP 6: 30-DAY RECURSIVE FUTURE SALES FORECASTING")
        forecast_df = generate_recursive_future_forecast(clean_df, model_artifact=model_artifact, horizon_days=30, save_forecasts=True)
        logger.info("Generated %d future forecast predictions.", len(forecast_df))

    # 7. Inventory Optimization
    if run_inventory:
        logger.info(">>> STEP 7: INVENTORY SIMULATION & REPLENISHMENT RECOMMENDATIONS")
        rec_df = generate_replenishment_recommendations(clean_df, forecast_df=forecast_df, save_outputs=True)
        actionable_count = (rec_df["suggested_order_qty"] > 0).sum()
        logger.info("Generated replenishment metrics: %d SKUs evaluated, %d reorders recommended.", len(rec_df), actionable_count)

        if generate_figures and forecast_df is not None:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plot_forecast_vs_actual(clean_df, forecast_df, store_id="STR_01", product_id="PRD_001", save_path=FIGURES_DIR / "08_forecast_vs_actual_sample.png")
            plot_inventory_status_breakdown(rec_df, save_path=FIGURES_DIR / "09_inventory_status_breakdown.png")

    print("\n" + "=" * 80)
    print("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("  To launch the interactive Streamlit dashboard, run:")
    print("    streamlit run app/dashboard.py")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Retail Sales Forecasting & Inventory Optimization Pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline from scratch.")
    parser.add_argument("--regenerate-data", action="store_true", help="Force regenerate synthetic data.")
    parser.add_argument("--train", action="store_true", help="Run training pipeline only.")
    parser.add_argument("--forecast", action="store_true", help="Run future forecasting only.")
    parser.add_argument("--inventory", action="store_true", help="Run inventory optimization only.")
    args = parser.parse_args()

    if args.train:
        clean_df = load_processed_data()
        train_and_select_best_model(clean_df, save_model=True)
    elif args.forecast:
        clean_df = load_processed_data()
        generate_recursive_future_forecast(clean_df, save_forecasts=True)
    elif args.inventory:
        clean_df = load_processed_data()
        generate_replenishment_recommendations(clean_df, save_outputs=True)
    else:
        # Default or --all
        run_pipeline(regenerate_data=args.regenerate_data)


if __name__ == "__main__":
    main()
