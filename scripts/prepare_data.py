import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_data import generate_synthetic_retail_dataset, load_raw_data
from src.data.preprocess import clean_and_preprocess_data
from src.data.validate_data import validate_retail_data
from src.utils.logging_config import setup_logger
from src.visualization.plots import generate_all_eda_figures

logger = setup_logger("prepare_data_script")


def main():
    parser = argparse.ArgumentParser(description="Prepare retail dataset: generate, validate, and preprocess.")
    parser.add_argument("--force-generate", action="store_true", help="Force regenerate synthetic data.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip generating EDA figures.")
    args = parser.parse_args()

    logger.info("=== STEP 1: LOAD / GENERATE DATASET ===")
    if args.force_generate:
        raw_df = generate_synthetic_retail_dataset(save_raw=True)
    else:
        raw_df = load_raw_data()

    logger.info("=== STEP 2: VALIDATE DATASET ===")
    val_report = validate_retail_data(raw_df, raise_errors=True)
    logger.info("Validation Report: Valid=%s | Rows=%d | Warnings=%d", val_report["is_valid"], val_report["row_count"], len(val_report["warnings"]))

    logger.info("=== STEP 3: PREPROCESS & CLEAN DATASET ===")
    clean_df, summary = clean_and_preprocess_data(raw_df, save_processed=True)
    logger.info("Preprocessing complete: Initial=%d rows -> Final=%d rows.", summary["initial_rows"], summary["cleaned_rows"])

    if not args.skip_figures:
        logger.info("=== STEP 4: GENERATE EDA FIGURES ===")
        generate_all_eda_figures(clean_df)

    logger.info("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
