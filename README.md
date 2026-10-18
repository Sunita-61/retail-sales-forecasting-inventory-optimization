# Retail Sales Forecasting & Inventory Optimization System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](tests/)

An enterprise-grade, end-to-end data science and supply chain optimization platform designed to forecast multi-store retail demand and automate inventory replenishment policies.

---

## 📌 Executive Summary

Retail inventory management balances two opposing financial forces: **stockouts** (resulting in lost sales, damaged customer loyalty, and forfeited profits) and **overstocks** (incurring capital holding costs, obsolescence, and warehouse crowding).

This repository implements an autonomous system that bridges **predictive machine learning time-series forecasting** with **stochastic inventory replenishment control**:
* **Data Simulation & Ingestion**: Generates reproducible daily retail data across multiple stores and categories with trend, seasonality, promotions, and lead times.
* **Leakage-Free Feature Engineering**: Constructs temporal calendar encodings, cyclical transforms, strictly past lags ($t-1, \dots, t-28$), and rolling window statistics without lookahead data leakage.
* **Competitive Model Benchmarking**: Compares Seasonal Naive, Moving Average, Ridge Regression, and Random Forest Regressor models on identical validation and test partitions.
* **30-Day Forward Autoregressive Forecasting**: Generates recursive multi-step daily projections with 90% confidence bands.
* **Multi-Echelon Inventory Optimization**: Automates safety stock sizing (95% Cycle Service Level), dynamic Reorder Points (ROP), periodic review Order-Up-To levels ($S$), and actionable purchase order recommendations.
* **Interactive Web Dashboard**: Streamlit application featuring an executive KPI overview, sales trend analysis, interactive forecast projections, and an inventory replenishment cockpit with real-time policy simulations.

> **Disclaimer**: This project operates on a realistic, mathematically modeled synthetic retail dataset designed for portfolio demonstration and algorithmic validation. It does not contain or expose internal proprietary records of any commercial enterprise.

---

## 🏗️ Project Architecture & Repository Structure

```
Retail-Sales-Forecasting-Inventory-Optimization/
│
├── app/                                    # Interactive Streamlit Web Application
│   ├── dashboard.py                        # Executive KPI Dashboard & Navigation
│   └── pages/
│       ├── 1_Sales_Analysis.py             # Historical EDA, Category & Store Analysis
│       ├── 2_Sales_Forecasting.py          # Interactive 30-Day Forward Forecast Explorer
│       ├── 3_Inventory_Optimization.py     # Replenishment Cockpit & Policy Simulator
│       └── 4_Model_Performance.py          # ML Model Benchmarks & Feature Importance
│
├── data/                                   # Data Storage (Git-Ignored / Sampled)
│   ├── raw/                                # Raw synthetic observations (29,240 rows)
│   ├── processed/                          # Cleaned and feature-augmented dataset
│   └── sample/                             # Representative sample CSV
│
├── notebooks/                              # Jupyter Analytical Portfolio Notebooks
│   ├── 01_data_understanding.ipynb         # Data ingestion, schema, and statistics
│   ├── 02_data_cleaning.ipynb              # Validation, missing values, deduplication
│   ├── 03_exploratory_data_analysis.ipynb  # Seasonality, trends, and promotional lift
│   ├── 04_feature_engineering.ipynb        # Calendar, lag, and rolling statistics
│   ├── 05_model_experimentation.ipynb      # ML benchmark and validation evaluation
│   └── 06_inventory_analysis.ipynb         # Safety stock, ROP, and replenishment orders
│
├── src/                                    # Production Source Code Package
│   ├── __init__.py
│   ├── config.py                           # System constants, metadata, filepaths
│   ├── data/                               # Data subpackage
│   │   ├── load_data.py                    # Stochastic data generator & loader
│   │   ├── validate_data.py                # Schema & business rule validation
│   │   └── preprocess.py                   # Cleaning, financial metrics, sorting
│   ├── features/
│   │   └── feature_engineering.py          # Temporal, cyclical, lag & rolling features
│   ├── forecasting/
│   │   ├── baseline.py                     # Heuristic benchmark models (Lag 7, 7-Day MA)
│   │   ├── train_model.py                  # Model training, comparison & persistence
│   │   ├── predict.py                      # 30-day recursive forward forecasting engine
│   │   └── evaluate_model.py               # MAE, RMSE, WAPE (%), and R2 metrics
│   ├── inventory/
│   │   ├── demand_estimation.py            # Historical demand velocity & variance
│   │   ├── inventory_calculations.py       # Safety stock, ROP, target stock formulas
│   │   └── reorder_recommendations.py      # Actionable replenishment order generator
│   ├── visualization/
│   │   └── plots.py                        # Reusable publication-grade figure generator
│   └── utils/
│       └── logging_config.py               # Standardized logging utility
│
├── models/                                 # Serialized model artifacts (.joblib)
├── outputs/                                # Operational Business Deliverables
│   ├── forecasts/                          # 30-day forward forecasts CSV
│   ├── inventory/                          # SKU-level inventory health metrics CSV
│   └── recommendations/                    # Actionable purchase orders CSV
│
├── reports/                                # Formal Reports & Figures
│   ├── figures/                            # High-resolution EDA and operational charts
│   ├── metrics/                            # Benchmark model comparison CSV
│   └── final_project_report.md             # Comprehensive final technical report
│
├── docs/                                   # In-Depth Technical Documentation
│   ├── project_overview.md                 # System overview and architecture
│   ├── data_dictionary.md                  # Comprehensive schema dictionary
│   ├── methodology.md                      # Feature engineering & ML methodology
│   ├── inventory_formulas.md               # Mathematical inventory formulations
│   ├── model_card.md                       # Machine learning model card
│   └── limitations.md                      # Assumptions, constraints, and roadmap
│
├── scripts/                                # Standalone Execution Scripts
│   ├── prepare_data.py                     # Data generation, cleaning, and EDA
│   ├── train.py                            # Model training and benchmark export
│   ├── generate_forecast.py                # Forward forecasting & inventory run
│   └── build_notebooks.py                  # Notebook compilation utility
│
├── tests/                                  # Automated Pytest Test Suite (34 Tests)
│   ├── test_data_validation.py             # Schema and constraint tests
│   ├── test_preprocessing.py               # Cleaning and deduplication tests
│   ├── test_feature_engineering.py         # Leakage prevention and lag tests
│   ├── test_forecasting.py                 # Baseline, metrics, and training tests
│   └── test_inventory.py                   # Safety stock, ROP, and order tests
│
├── .gitignore
├── .env.example
├── LICENSE                                 # MIT Open Source License
├── main.py                                 # Master End-to-End Orchestrator Pipeline
├── requirements.txt                        # Dependency definitions
└── pytest.ini                              # Pytest configuration
```

---

## ⚡ Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/retail-sales-forecasting-inventory-optimization.git
cd retail-sales-forecasting-inventory-optimization

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Execute the master orchestrator to generate data, validate schema, train models, forecast demand, optimize inventory, and export reports:
```bash
python main.py
```

Optional CLI flags:
* `python main.py --all` : Complete pipeline execution.
* `python main.py --regenerate-data` : Force regenerate the raw synthetic dataset.
* `python main.py --train` : Run feature engineering and model training only.
* `python main.py --forecast` : Generate 30-day forward predictions only.
* `python main.py --inventory` : Recompute inventory metrics and recommendations only.

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501` to explore the interactive visual analytics.

---

## 🧪 Automated Testing

The repository includes a comprehensive automated test suite with **34 passing unit and integration tests** verifying data validation, preprocessing, leakage-free feature computation, forecasting metrics, and inventory algorithms:

```bash
pytest -v
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
collected 34 items

tests\test_data_validation.py .......                                    [ 20%]
tests\test_feature_engineering.py .....                                  [ 35%]
tests\test_forecasting.py .......                                        [ 55%]
tests\test_inventory.py ...........                                      [ 88%]
tests\test_preprocessing.py ....                                         [100%]

============================= 34 passed in 1.55s ==============================
```

---

## 📊 Model Evaluation & Benchmarking

Models were trained on historical data from `2024-01-29` to `2025-06-30` and benchmarked on an identical validation window (`2025-07-01` to `2025-09-30`, 3,680 samples):

| Model Candidate | MAE (Units) | RMSE (Units) | WAPE (%) | $R^2$ Score | Rank / Selection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **5.365** | **7.879** | **19.99%** | **0.853** | **Champion Model** |
| Ridge Regression ($\alpha=10.0$) | 5.682 | 8.187 | 21.17% | 0.841 | Runner-Up |
| 7-Day Moving Average | 7.039 | 10.201 | 26.23% | 0.753 | Heuristic Baseline |
| Seasonal Naive (Lag 7) | 7.510 | 10.843 | 27.99% | 0.721 | Heuristic Baseline |

* **Held-Out Test Set (Oct–Dec 2025)**: On the unseen final test set capturing the volatile Q4 holiday surge, the champion Random Forest model maintained **WAPE = 25.77%** and $R^2 = 0.793$.
* **Why WAPE**: Retail demand frequently features zero sales, which causes undefined divisions in MAPE. WAPE ($\frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$) provides a stable, volume-weighted accuracy metric.
* **Empirical Uncertainty Bands**: Projections include empirical residual error bands ($\pm 1.645 \cdot \sigma_{\text{resid}}$). Measured empirical coverage is **91.17% on validation** and **82.50% on the Q4 test period** (average interval width: 25.03 units). These bands reflect empirical historical error distributions and do not claim or guarantee an unconditional 90% confidence under structural distribution shifts.

---

## 🔮 Forecasting Horizon & Historical Simulation Design (Option A)

* **Historical Data Cutoff**: `2025-12-31` (exactly 2 complete calendar years: 731 days from `2024-01-01` to `2025-12-31`, preserving leap-year calendar alignment and two complete annual seasonal cycles).
* **Forecast Horizon**: 30-day post-historical demonstration horizon spanning **`2026-01-01` through `2026-01-30`** (1,200 forecast records across 4 stores and 10 SKUs).
* **Simulation Framework**: The project is intentionally designed as a **historical simulation study**. It demonstrates out-of-sample forward demand projection immediately following the historical observation cutoff, avoiding mid-cycle synthetic extensions that would distort clean annual seasonality and established benchmark evaluation splits.

---

## 📦 Inventory Optimization & Stockout Adjustment Policies

The replenishment module calculates multi-echelon policies with documented assumptions:

1. **Target Distinction**:
   * **Sales Forecasting Target**: Strictly models observable Point-of-Sale (POS) transactions (`units_sold`), preserving actual registered consumer transactions and preventing synthetic target leakage.
   * **Inventory Replenishment Target**: Replenishment calculations use **stockout-adjusted unconstrained demand** (`units_sold + stockout_units`) via `adjust_stockouts=True` in `generate_replenishment_recommendations`. This un-censors demand velocity (chain-wide average demand rises from 31.37 to 36.00 units/day) to prevent the self-reinforcing under-replenishment spiral caused by past supply shortages.
2. **Safety Stock ($SS$)**:
   $$SS = \max\left(2, \left\lceil Z \cdot \sigma_d \cdot \sqrt{L} \right\rceil\right)$$
   Where $Z = 1.645$ corresponds to a 95% Cycle Service Level (CSL), $\sigma_d$ is daily demand standard deviation, and $L$ is lead time in days.
3. **Reorder Point ($ROP$)**:
   $$ROP = \lceil d_{avg} \cdot L + SS \rceil$$
4. **Periodic Review Order-Up-To Level ($S$)**:
   $$S = \lceil d_{avg} \cdot (R + L) + SS \rceil$$
   Where $R$ is review period (default $R = 7$ days).
5. **Suggested Order Quantity ($Q$)**:
   $$Q = \max(0, S - IP) \quad \text{if } IP \le ROP \text{ else } 0$$
6. **Stockout Risk Trigger**:
   Identifies SKUs where Days of Supply ($IP / d_{avg}$) is less than lead time $L$, indicating impending stock exhaustion before supplier delivery.

---

## 📄 License & Attribution

This project is released under the [MIT License](LICENSE). Designed and implemented for portfolio demonstration, supply chain analytics research, and production data science showcase.
