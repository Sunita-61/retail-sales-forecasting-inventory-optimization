# Final Project Report: Retail Sales Forecasting & Inventory Optimization System

**Project Type:** End-to-End Retail Data Science & Supply Chain Optimization  
**Environment:** Python 3.14 / Scikit-Learn / Pandas / Streamlit / Plotly  
**Date:** October 2026  
**Status:** Completed & Validated

---

## 1. Executive Summary

In retail operations, improper inventory stocking leads to a dual penalty: stockouts cause lost revenue and customer defection, while excess stock ties up working capital and accrues holding costs. This project implements a complete, autonomous retail analytics and replenishment platform that connects time-series demand forecasting with stochastic inventory replenishment policies.

### Key Highlights & Results:
1. **Robust Data Pipeline**: Ingested and preprocessed 29,240 synthetic daily observations spanning 2 full years (2024–2025) across 4 retail stores and 10 representative SKUs in 4 major merchandise categories.
2. **Forecasting Superiority**: Benchmarked multiple candidate models. The **Random Forest Regressor** achieved champion status with a validation **WAPE of 19.99%** (MAE: 5.365 units, $R^2$: 0.853), outperforming the Seasonal Naive baseline (WAPE 27.99%) and 7-day Moving Average (WAPE 26.23%).
3. **Generalization on Unseen Q4 Surge**: Evaluated on a held-out 3-month test set (October–December 2025) containing the peak Q4 holiday volume surge, the champion model maintained a competitive WAPE of **25.77%**.
4. **Supply Chain Impact**: Automated calculation of safety stocks at 95% Cycle Service Level, dynamic Reorder Points (ROP), and Order-Up-To replenishment quantities. Evaluated 40 store-SKU series, generating actionable purchase order recommendations for 33 SKU locations requiring replenishment to prevent projected stockouts.
5. **Interactive Dashboard**: Deployed a multi-page interactive Streamlit web dashboard providing executive KPIs, exploratory sales analytics, forward 30-day forecast visualizations with 90% confidence bands, and an inventory replenishment cockpit with real-time policy simulation.

---

## 2. Exploratory Data Analysis & Business Trends

All figures have been generated, validated, and persisted under `reports/figures/`.

### 2.1 Trend and Seasonality
* **Annual Growth Trend**: Observed a steady ~6% annual volume growth across the 24-month horizon.
* **Weekly Seasonality**: Distinct day-of-week demand patterns were identified (`reports/figures/06_day_of_week_pattern.png`). Demand peaks significantly on Saturdays (+45% above mid-week baseline) and Sundays (+30%), with Monday and Tuesday exhibiting the lowest store traffic.
* **Holiday & Monthly Surges**: High-value categories (Electronics and Apparel) experienced significant volume surges during November and December (Q4 holiday shopping, Black Friday, and Christmas peaks) (`reports/figures/02_monthly_sales_trend.png`).

### 2.2 Category & Store Dynamics
* **Revenue Breakdown**: While Groceries generated the highest absolute unit volume (due to daily consumable turnover for Milk and Bread), **Electronics** generated the highest total revenue due to high unit selling prices ($79.99 to $149.99) (`reports/figures/03_sales_by_category.png`).
* **Store Performance**: Store `STR_01` (Metro Flagship) generated the highest gross revenue (~35% above regional baseline), followed by `STR_02` (Suburban Hypermarket). Store `STR_03` (Downtown Express) operated at a lower average basket size, catering primarily to rapid daily convenience purchases (`reports/figures/04_sales_by_store.png`).

### 2.3 Promotional Lift
* Products featured on marketing promotions experienced an observed **35% to 50% demand lift** across categories (`reports/figures/07_promotion_impact.png`). Promotional lift was particularly pronounced for impulse purchase items in Apparel and Home & Kitchen.

---

## 3. Machine Learning Forecasting Benchmarks

To establish rigorous methodological validity, candidate models were trained strictly on historical data prior to July 2025 and evaluated on an identical validation period (`2025-07-01` to `2025-09-30`, 3,680 records):

| Model Candidate | MAE (Units) | RMSE (Units) | WAPE (%) | $R^2$ Score | Rank / Selection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **5.365** | **7.879** | **19.99%** | **0.853** | **Champion Model** |
| Ridge Regression ($\alpha=10.0$) | 5.682 | 8.187 | 21.17% | 0.841 | Runner-Up |
| 7-Day Moving Average | 7.039 | 10.201 | 26.23% | 0.753 | Heuristic Baseline |
| Seasonal Naive (Lag 7) | 7.510 | 10.843 | 27.99% | 0.721 | Heuristic Baseline |

### Analysis of Forecasting Models:
* **Why Random Forest Won**: The Random Forest ensemble successfully synthesized non-linear interactions between promotional flags, store multipliers, cyclical calendar encodings, and autoregressive demand lags.
* **Feature Importance**: Analysis indicates that recent demand velocity (`rolling_mean_7`), weekly seasonal lag (`lag_7`), immediate prior day (`lag_1`), and product identifier were the top four predictive drivers.
* **Metric Selection**: WAPE ($\frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$) was adopted as the primary optimization metric because retail sales series exhibit intermittent zero sales where standard MAPE divides by zero and diverges to infinity.

---

## 4. Inventory Optimization & Replenishment Execution

Using estimated daily demand rates and variance combined with model forecasts, inventory policies were calibrated:

### 4.1 Policy Formulations
* **Cycle Service Level**: 95% target ($Z = 1.645$).
* **Safety Stock**: $SS = \max(2, \lceil Z \cdot \sigma_d \cdot \sqrt{L} \rceil)$.
* **Reorder Point (ROP)**: $ROP = \lceil d_{avg} \cdot L + SS \rceil$.
* **Periodic Review Target (S)**: $S = \lceil d_{avg} \cdot (R + L) + SS \rceil$ with review period $R = 7$ days.

### 4.2 Portfolio Inventory Health
Across the 40 SKU-store locations evaluated at the close of 2025:
* **Actionable Reorders Recommended**: 33 SKU locations triggered replenishment orders ($IP \le ROP$).
* **Critical Stockouts Flagged**: Identified SKU locations with zero on-hand stock and pending deliveries.
* **Overstock Candidates**: 2 SKU locations had stock levels exceeding $1.25 \times S$, indicating opportunities to curtail upcoming purchase orders and liberate trapped working capital.
* **Replenishment Capital Required**: Total estimated procurement cost for suggested orders was exported to `outputs/recommendations/replenishment_recommendations.csv`.

---

## 5. Business Insights & Recommendations

1. **Prioritize Long-Lead Time SKUs**: Items in Electronics (e.g., Noise-Cancelling Headphones with $L=7$ days) possess high demand uncertainty over lead time. Ensuring strict adherence to safety stock thresholds ($SS \approx 10$ units) prevents severe gross margin erosion.
2. **Promotional Coordination**: Marketing teams should notify inventory planners at least 14 days prior to promotional campaigns so orders can be placed before the lead-time window expires.
3. **Automated Order Queueing**: Incorporate the generated purchase order recommendation CSV into daily ERP replenishment queues, reducing human ordering latency.

---

## 6. Verification and Project Delivery Artifacts

* **Source Code Modules**: Clean architecture under `src/` (data, features, forecasting, inventory, visualization, utils).
* **Automated Test Suite**: 26 unit and integration tests passing (`tests/`).
* **Interactive Dashboard**: Streamlit app with 4 dedicated pages (`app/`).
* **Jupyter Notebooks**: 6 complete notebooks covering the entire analytical lifecycle (`notebooks/`).
* **Documentation**: 6 comprehensive markdown guides under `docs/` and root `README.md`.
