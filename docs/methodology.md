# Methodology: Time-Series Modeling & Inventory Control

This document articulates the end-to-end engineering and statistical methodology implemented across data preprocessing, feature engineering, model training, and multi-echelon replenishment.

---

## 1. Data Ingestion & Integrity Validation
Data integrity is enforced programmatically prior to downstream ingestion:
1. **Schema Verification**: Confirms all 11 requisite raw columns exist with appropriate data types.
2. **Business Key Uniqueness**: Verifies uniqueness on composite key `(date, store_id, product_id)`.
3. **Physical Feasibility Bounds**: Asserts physical non-negativity for units sold, on-hand inventory, unit pricing, and strictly positive supplier lead times ($L \ge 1$).
4. **Missing Value Protocol**: Documented domain imputation (missing receipts = 0 units received; missing promotions = 0 non-promoted regular price; missing prices filled via product group median).

---

## 2. Leakage-Free Time-Series Feature Engineering
Time-series forecasting models are vulnerable to lookahead data leakage if features inadvertently encode future outcomes. The following architectural guards are implemented:

1. **Chronological Sorting**: Data is sorted by `(store_id, product_id, date)`.
2. **Lagged Target Features**: Lags are constructed strictly within individual `(store_id, product_id)` time-series partitions:
   $$X_{t}^{\text{lag\_k}} = y_{t-k}, \quad k \in \{1, 7, 14, 21, 28\}$$
3. **Rolling Window Aggregations**: Rolling statistics apply an explicit 1-day backward shift:
   $$\text{rolling\_mean}_w(t) = \frac{1}{w} \sum_{i=1}^{w} y_{t-i}$$
   This guarantees that $y_t$ is excluded from the feature vector at time $t$.
4. **Temporal Calendar & Cyclical Encodings**:
   $$\sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
5. **Categorical Encoders**: Scikit-learn `OrdinalEncoder` fitted strictly on training data splits, preventing categorical target leakage into validation and test partitions.

---

## 3. Chronological Train / Validation / Test Strategy
Traditional cross-validation (e.g. random K-Fold) corrupts time series by training on future records to predict the past. We implement chronological time-ordered partitioning:

* **Training Window (70%)**: `2024-01-29` to `2025-06-30` (20,760 observations)
* **Validation Window (15%)**: `2025-07-01` to `2025-09-30` (3,680 observations) — Used strictly for hyperparameter selection and model benchmarking.
* **Held-Out Test Window (15%)**: `2025-10-01` to `2025-12-31` (3,680 observations) — Unseen final operational assessment.

---

## 4. Forecasting Model Benchmarking
Four candidate algorithms were trained on the identical training set and evaluated on the identical validation window:

1. **Seasonal Naive (Lag 7)**:
   Predicts $\hat{y}_t = y_{t-7}$. Exploits retail weekly shopping cycles.
2. **7-Day Moving Average**:
   Predicts $\hat{y}_t = \frac{1}{7} \sum_{i=1}^7 y_{t-i}$. Smooths high-frequency noise.
3. **Ridge Regression**:
   Linear model regularized with an $L_2$ penalty ($\alpha=10.0$) over standardized features.
4. **Random Forest Regressor**:
   Ensemble of 100 decision trees (`max_depth=12`, `min_samples_split=5`, `random_state=42`) capable of capturing non-linear interactions between promotional flags, store profiles, calendar cycles, and lagged demand.

---

## 5. Multi-Step Recursive Forecasting
For operational planning over a 30-day horizon ($H=30$):
* For day $t+1$: Compute features from actual history, predict $\hat{y}_{t+1}$.
* For day $t+k$ ($k > 1$): Features requiring lags $\{t+k-1, \dots\}$ consume prior predictions $\{\hat{y}_{t+k-1}, \dots\}$.
* Non-negativity is enforced via $\hat{y} = \max(0, \hat{y})$.
* 90% Confidence intervals are generated as $\hat{y} \pm 1.645 \cdot \text{RMSE}_{\text{test}}$.
