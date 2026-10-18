# Project Overview: Retail Sales Forecasting & Inventory Optimization System

## Executive Summary
The **Retail Sales Forecasting & Inventory Optimization System** is an end-to-end data science and supply chain engineering solution designed to assist retail merchandising and supply chain teams in demand prediction and inventory control.

The system addresses the fundamental retail dilemma: **balancing inventory holding costs against stockout risks and lost sales**. By integrating machine learning-driven time-series forecasting with stochastic inventory replenishment policies, the platform automates purchase order recommendations across multi-store, multi-category retail operations.

---

## Business Objectives
1. **Demand Forecasting Accuracy**: Deliver accurate multi-step forward demand projections using machine learning models benchmarked against heuristic baselines.
2. **Stockout Risk Mitigation**: Proactively detect SKUs facing imminent stock exhaustion before supplier replenishment arrives.
3. **Working Capital Efficiency**: Prevent excessive buffer accumulation and overstock holding penalties by deploying dynamic safety stock sizing and order-up-to policies.
4. **Autonomous Decision Support**: Translate mathematical models into actionable purchase orders through an interactive Streamlit operational cockpit.

---

## Simulated Retail Environment
To ensure full portability and demonstrate production architecture without exposing proprietary or confidential corporate records, this project uses a **reproducible stochastic simulation engine**.

* **Store Network**: 4 representative store profiles (Urban Flagship, Suburban Hypermarket, Downtown Express, Regional Center).
* **Product Catalog**: 10 distinct products across 4 categories (Groceries, Electronics, Apparel, Home & Kitchen).
* **Horizon**: 2 full historical calendar years (2024-01-01 to 2025-12-31, 731 days per series).
* **Stochastic Mechanics**: Modeled annual business growth trend (+6%/yr), day-of-week seasonality, Q4 holiday surges, multi-day promotional campaigns, supplier lead times, delivery pipeline queues, and non-negative stock constraints.

---

## Architecture Flow

```
[ Synthetic Simulation / Raw Ingestion ]
                  │
                  ▼
[ Data Validation & Integrity Checks ]
                  │
                  ▼
[ Cleaning & Preprocessing (Monetary vs Units) ]
                  │
                  ▼
[ Leakage-Free Time-Series Feature Engineering ]
                  │
                  ▼
[ Model Training & Benchmark Comparison ]
  ├── Seasonal Naive (Lag 7)
  ├── 7-Day Moving Average
  ├── Ridge Regression
  └── Random Forest Regressor (Champion)
                  │
                  ▼
[ 30-Day Autoregressive Recursive Forecasting ]
                  │
                  ▼
[ Multi-Echelon Stochastic Inventory Optimization ]
  ├── Demand Variance & Lead-Time Demand
  ├── Safety Stock (95% Cycle Service Level)
  ├── Dynamic Reorder Points (ROP)
  └── Periodic Review Order-Up-To Level (S)
                  │
                  ▼
[ Interactive Streamlit Management Cockpit ]
```
