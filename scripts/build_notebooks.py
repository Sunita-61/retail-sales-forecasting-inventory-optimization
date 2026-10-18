"""Script to generate all 6 portfolio Jupyter notebooks."""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11.0"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in text.split("\n")],
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in code.split("\n")],
    }


def main():
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    # 1. 01_data_understanding.ipynb
    nb1 = make_notebook(
        [
            md_cell(
                "# 01. Data Understanding & Ingestion\n\n"
                "This notebook demonstrates loading raw retail sales and simulated inventory data, "
                "inspecting schema, data types, and initial business summary statistics."
            ),
            code_cell(
                "import pandas as pd\n"
                "import numpy as np\n"
                "from src.config import RAW_DATA_FILE\n"
                "from src.data.load_data import load_raw_data\n\n"
                "df_raw = load_raw_data(RAW_DATA_FILE)\n"
                "print(f'Dataset shape: {df_raw.shape}')\n"
                "df_raw.head()"
            ),
            code_cell("df_raw.info()"),
            code_cell("df_raw.describe().T"),
            md_cell(
                "### Verification of Simulation vs Real Retail Records\n"
                "> **Notice**: As documented in the project specifications, this dataset is generated through stochastic simulation "
                "to reflect realistic seasonality, trend, and inventory dynamics without accessing internal proprietary company databases."
            ),
        ]
    )

    with open(notebooks_dir / "01_data_understanding.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    # 2. 02_data_cleaning.ipynb
    nb2 = make_notebook(
        [
            md_cell(
                "# 02. Data Cleaning & Validation\n\n"
                "This notebook covers data validation, missing value imputation, business key duplicate removal, "
                "non-negative bounding, and date parsing."
            ),
            code_cell(
                "from src.data.validate_data import validate_retail_data\n"
                "from src.data.preprocess import clean_and_preprocess_data\n"
                "from src.data.load_data import load_raw_data\n\n"
                "raw_df = load_raw_data()\n"
                "val_report = validate_retail_data(raw_df, raise_errors=False)\n"
                "print('Validation Report:', val_report['is_valid'], 'Errors:', len(val_report['errors']))"
            ),
            code_cell(
                "cleaned_df, summary = clean_and_preprocess_data(raw_df, save_processed=True)\n"
                "print('Preprocessing Summary:', summary)\n"
                "cleaned_df.head()"
            ),
            code_cell("cleaned_df.isnull().sum()"),
        ]
    )

    with open(notebooks_dir / "02_data_cleaning.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    # 3. 03_exploratory_data_analysis.ipynb
    nb3 = make_notebook(
        [
            md_cell(
                "# 03. Exploratory Data Analysis (EDA)\n\n"
                "In-depth visual analysis of demand trends, day-of-week seasonality, monthly patterns, "
                "store comparisons, and promotional uplifts."
            ),
            code_cell(
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from src.data.preprocess import load_processed_data\n"
                "from src.visualization.plots import plot_daily_sales_trend, plot_sales_by_category, plot_day_of_week_pattern\n\n"
                "df = load_processed_data()\n"
                "fig1 = plot_daily_sales_trend(df)\n"
                "plt.show()"
            ),
            code_cell(
                "fig2 = plot_sales_by_category(df)\n"
                "plt.show()"
            ),
            code_cell(
                "fig3 = plot_day_of_week_pattern(df)\n"
                "plt.show()"
            ),
        ]
    )

    with open(notebooks_dir / "03_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    # 4. 04_feature_engineering.ipynb
    nb4 = make_notebook(
        [
            md_cell(
                "# 04. Time-Series Feature Engineering\n\n"
                "Constructs temporal calendar indicators, cyclical sine/cosine encoders, strictly past lagged demand (1, 7, 14, 21, 28 days), "
                "and rolling window statistics without lookahead data leakage."
            ),
            code_cell(
                "from src.data.preprocess import load_processed_data\n"
                "from src.features.feature_engineering import prepare_feature_dataset, split_chronological_data\n\n"
                "df = load_processed_data()\n"
                "df_feat, feature_cols, encoder = prepare_feature_dataset(df, drop_na_lags=True)\n"
                "print(f'Feature matrix shape: {df_feat.shape}')\n"
                "print(f'Engineered features ({len(feature_cols)}):', feature_cols)"
            ),
            code_cell(
                "splits = split_chronological_data(df_feat)\n"
                "print('Train rows:', len(splits['train']))\n"
                "print('Validation rows:', len(splits['val']))\n"
                "print('Test rows:', len(splits['test']))"
            ),
        ]
    )

    with open(notebooks_dir / "04_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    # 5. 05_model_experimentation.ipynb
    nb5 = make_notebook(
        [
            md_cell(
                "# 05. Model Experimentation & Comparative Evaluation\n\n"
                "Trains and compares candidate models (Seasonal Naive Baseline, 7-Day Moving Average, Ridge Regression, Random Forest Regressor) "
                "on validation data using MAE, RMSE, and WAPE."
            ),
            code_cell(
                "from src.data.preprocess import load_processed_data\n"
                "from src.forecasting.train_model import train_and_select_best_model\n\n"
                "df = load_processed_data()\n"
                "champ_model, comp_df, artifact = train_and_select_best_model(df, save_model=True)\n"
                "comp_df"
            ),
            code_cell(
                "print('Champion Model:', artifact['model_name'])\n"
                "print('Held-out Test Metrics:', artifact['test_metrics'])"
            ),
        ]
    )

    with open(notebooks_dir / "05_model_experimentation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=2)

    # 6. 06_inventory_analysis.ipynb
    nb6 = make_notebook(
        [
            md_cell(
                "# 06. Inventory Simulation & Replenishment Optimization\n\n"
                "Calculates safety stocks using normal quantile Z and lead time demand standard deviations, dynamic Reorder Points (ROP), "
                "periodic review Order-Up-To levels (S), and exports actionable purchase order recommendations."
            ),
            code_cell(
                "from src.data.preprocess import load_processed_data\n"
                "from src.forecasting.predict import generate_recursive_future_forecast\n"
                "from src.inventory.reorder_recommendations import generate_replenishment_recommendations\n\n"
                "df = load_processed_data()\n"
                "fc_df = generate_recursive_future_forecast(df, horizon_days=30, save_forecasts=True)\n"
                "rec_df = generate_replenishment_recommendations(df, forecast_df=fc_df, save_outputs=True)\n"
                "rec_df.head(10)"
            ),
        ]
    )

    with open(notebooks_dir / "06_inventory_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb6, f, indent=2)

    print("All 6 Jupyter notebooks built successfully.")


if __name__ == "__main__":
    main()
