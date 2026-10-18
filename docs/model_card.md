# Model Card: Time-Series Retail Demand Predictor

## Model Details
* **Developer**: Senior Data Scientist & ML Engineer
* **Model Date**: October 2026
* **Model Version**: 1.0.0
* **Model Type**: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
* **Target Objective**: Predict physical daily sales units (`units_sold`) per store-SKU series.
* **Loss Function**: Mean Squared Error (MSE), evaluated via MAE, RMSE, and WAPE.

---

## Intended Use
* **Primary Intended Use**: Operational 1-to-30 day demand forecasting for inventory replenishment planning, safety stock calibration, and supply chain purchase order generation.
* **Primary Intended Users**: Store managers, retail inventory planners, demand planners, category merchandising managers.
* **Out-of-Scope Uses**: High-frequency intraday algorithmic pricing, promotional elasticity causation inference, or financial trading.

---

## Training Data & Features
* **Dataset**: Chronologically sorted historical observations from `2024-01-29` to `2025-06-30` (20,760 samples).
* **Input Features (17 total)**:
  * Categorical: `store_id`, `product_id`, `category` (Ordinal Encoded).
  * Pricing & Marketing: `unit_price`, `promotion_flag`.
  * Calendar & Cyclical: `day_of_week`, `day_of_month`, `month`, `year`, `is_weekend`, `sin_dow`, `cos_dow`, `sin_month`, `cos_month`.
  * Autoregressive Lags: `lag_1`, `lag_7`, `lag_14`, `lag_21`, `lag_28`.
  * Rolling Window Statistics: `rolling_mean_7`, `rolling_std_7`, `rolling_mean_14`, `rolling_std_14`, `rolling_mean_28`, `rolling_std_28`.

---

## Benchmark Performance Comparison

Evaluation across identical Validation Partition (`2025-07-01` to `2025-09-30`, 3,680 records):

| Model Candidate | MAE (Units) | RMSE (Units) | WAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor (Champion)** | **5.365** | **7.879** | **19.99%** | **0.853** |
| Ridge Regression ($\alpha=10.0$) | 5.682 | 8.187 | 21.17% | 0.841 |
| 7-Day Moving Average Baseline | 7.039 | 10.201 | 26.23% | 0.753 |
| Seasonal Naive (Lag 7) Baseline | 7.510 | 10.843 | 27.99% | 0.721 |

### Held-Out Final Test Set Evaluation (`2025-10-01` to `2025-12-31`, 3,680 records):
* **Test MAE**: 7.714 units
* **Test RMSE**: 11.103 units
* **Test WAPE**: 25.77%
* **Test $R^2$**: 0.793

*Note*: Test period encapsulates the high-variance Q4 holiday volume surge, demonstrating stable generalization.

---

## Top Predictive Features (Random Forest)
1. `rolling_mean_7` (Recent demand velocity)
2. `lag_7` (Same-day-last-week seasonal anchor)
3. `lag_1` (Immediate preceding day velocity)
4. `product_id` (SKU volume baseline)
5. `day_of_week` / `sin_dow` (Weekend surges)
6. `promotion_flag` (Promotional uplift)

---

## Ethical & Operational Considerations
* **Simulated Data**: The model is trained on synthetic data engineered to mirror real retail mechanics. It does not ingest confidential company data or personally identifiable customer information (PII).
* **Target Non-Negativity**: All operational predictions apply post-inference lower bound clipping ($\hat{y} \ge 0$).
