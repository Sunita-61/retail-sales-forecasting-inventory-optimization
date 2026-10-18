import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocess import load_processed_data
from src.forecasting.train_model import train_and_select_best_model
from src.utils.logging_config import setup_logger

logger = setup_logger("train_script")


def main():
    parser = argparse.ArgumentParser(description="Train retail sales forecasting models.")
    parser.parse_args()

    logger.info("Loading preprocessed sales data...")
    clean_df = load_processed_data()

    logger.info("Training and evaluating forecasting models...")
    champion_model, comparison_df, artifact = train_and_select_best_model(clean_df, save_model=True)

    logger.info("=== MODEL BENCHMARK RESULTS ===")
    print("\n" + comparison_df.to_string(index=False) + "\n")
    logger.info("Held-out test performance of %s: %s", artifact["model_name"], artifact["test_metrics"])
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
