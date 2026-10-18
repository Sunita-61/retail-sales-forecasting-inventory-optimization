# Independent Technical Validation Audit Report

**System Name:** Retail Sales Forecasting & Inventory Optimization System  
**Audit Date:** 2026-10-18  
**Auditor:** Senior Python Developer & Machine Learning Engineer  
**Audit Scope:** End-to-End Pipeline, Data Integrity, Leakage Prevention, Model Reproduction, Inventory Calculations, Test Suite, and Dashboard  

---

## 1. Executive Summary

An independent, rigorous technical validation audit of the Retail Sales Forecasting & Inventory Optimization System was conducted across all codebase components, data artifacts, machine learning models, and inventory formulas.

### Summary Assessment
* **Automated Tests**: **31 / 31 tests passed** (including 5 new regression tests added during this audit).
* **Dataset Integrity**: Exactly **29,240 records** across 4 stores and 10 products over 731 days (2024-01-01 to 2025-12-31). 100% reproducible with fixed random seed 42. Zero null values, zero duplicate keys, zero negative inventory.
* **Forecasting Reproducibility**: Champion model (**Random Forest Regressor**) reproduced on validation set with **WAPE = 19.99%** (MAE: 5.365 units, RMSE: 7.879 units, $R^2$: 0.853), outperforming Ridge Regression (WAPE 21.17%), 7-Day Moving Average (WAPE 26.23%), and Seasonal Naive (WAPE 27.99%). Held-out test performance confirmed at **WAPE = 25.77%**.
* **Key Defects Identified & Resolved**:
  1. *Grouped Rolling Index Alignment Defect* in `src/features/feature_engineering.py`: Positional index alignment bug during `.reset_index(drop=True)` after groupby rolling was corrected using grouped `.transform()`.
  2. *Encoder Fit Timing Smell* in `src/forecasting/train_model.py`: Categorical `OrdinalEncoder` was refactored to fit strictly on the training partition (`train_df`) after chronological splitting.
  3. *Missing Path Import* in `src/data/load_data.py`: Added `from pathlib import Path` to allow calling `load_raw_data(filepath)` with string or Path arguments.
  4. *Script sys.path Resolution* in `scripts/`: Added root directory insertion to `sys.path` so scripts can be invoked directly from any shell directory.
  5. *Uncertainty Interval Calibration Clarification* in `src/forecasting/predict.py`: Labeled and implemented as an **Empirical Residual Uncertainty Band (~90% normal approximation)** using measured validation residual variance, with programmatic verification that $\text{lower\_bound} \le \text{upper\_bound}$ and $\text{lower\_bound} \ge 0$.
* **Overall Audit Recommendation**: **PASS (GO FOR PORTFOLIO PUBLICATION)**.

---

## 2. Environment Details

| Component | Documented / Expected | Observed / Verified | Status |
| :--- | :--- | :--- | :---: |
| **Operating System** | Windows 10/11 | Windows (x86_64) | **PASS** |
| **Python Version** | Python 3.11+ / 3.14 compatible | Python 3.14.8 | **PASS** |
| **Package Dependency Check** | `python -m pip check` clean | `No broken requirements found.` | **PASS** |
| **Core Libraries** | pandas, numpy, scikit-learn, joblib | pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0 | **PASS** |
| **Visualization & App** | matplotlib, seaborn, plotly, streamlit | matplotlib 3.11.2, seaborn 0.13.2, plotly 7.1.0, streamlit 1.66.0 | **PASS** |
| **Testing Framework** | pytest >= 7.4.0 | pytest 9.1.1, pytest-cov 7.1.0 | **PASS** |
| **Git Repository** | Initialized and committed | Git repo initialized (commit `d9979e4`) | **PASS** |

---

## 3. Dataset Validation

Audit performed on `data/raw/synthetic_retail_sales.csv` and verified against fresh generation with `RANDOM_SEED = 42`:

* **Total Row Count**: 29,240 records (**PASS**).
* **Time Series Granularity**: 4 stores $\times$ 10 products = 40 distinct series (**PASS**).
* **Calendar Range**: 2024-01-01 to 2025-12-31 (731 consecutive days per series) (**PASS**).
* **Missing Values**: 0 nulls across all 15 columns (**PASS**).
* **Duplicate Business Keys**: 0 duplicate composite keys on `['date', 'store_id', 'product_id']` (**PASS**).
* **Negative Quantities**:
  * `units_sold < 0`: 0 violations (**PASS**).
  * `on_hand_stock < 0`: 0 violations (**PASS**).
  * `unit_price < 0`: 0 violations (**PASS**).
  * `lead_time_days <= 0`: 0 violations (**PASS**).
* **Stockout Consistency**: Verified that `units_demanded - units_sold == stockout_units` holds across 100% of rows (29,240 / 29,240 records) (**PASS**).
* **Physical Stock Decrementing**: Realized sales never exceed available on-hand stock plus inbound receipts (**PASS**).
* **Synthetic Attribution**: Column `data_source` explicitly identifies: `'Simulated Synthetic Data - Not Actual Company Records'` on every row (**PASS**).
* **Saved Validation Report**: Available at `reports/metrics/data_validation_report.txt` (**PASS**).

---

## 4. Data Leakage Audit

A line-by-line inspection of feature engineering and modeling scripts was performed:

1. **Chronological Sorting**: `df.sort_values(by=group_keys + ["date"])` is enforced prior to creating lag or rolling features (**PASS**).
2. **Lag Grouping**: Lags $t-1, t-7, t-14, t-21, t-28$ are calculated with `groupby(['store_id', 'product_id'])`, strictly isolating time series (**PASS**).
3. **Rolling Window Leakage Protection**: All rolling features apply `shift(1)` prior to rolling calculation, ensuring current target $y_t$ is excluded (**PASS**).
4. **Grouped Rolling Alignment (Resolved Defect)**:
   * *File*: `src/features/feature_engineering.py`
   * *Problem*: `.reset_index(drop=True)` after multi-group rolling caused positional index misalignment between sorted DataFrame and calculated rolling values.
   * *Fix Applied*: Replaced with `shifted_target.groupby(...).transform(lambda s: s.rolling(window=w, min_periods=1).mean())` on a reset-indexed DataFrame, guaranteeing 100% group alignment.
   * *Status*: **PASS** (Verified with regression test `test_multi_group_rolling_alignment`).
5. **Encoder Leakage Protection (Resolved Smell)**:
   * *File*: `src/forecasting/train_model.py`
   * *Problem*: `OrdinalEncoder` was previously fitted on the entire DataFrame prior to chronological splitting.
   * *Fix Applied*: Restructured pipeline to perform chronological split first, then fit `encoder.fit(train_df[cat_cols])` strictly on `train_df`, followed by `encoder.transform()` on `val_df` and `test_df`.
   * *Status*: **PASS**.
6. **Chronological Splits**:
   * Train: 2024-01-29 to 2025-06-30 (20,760 samples)
   * Validation: 2025-07-01 to 2025-09-30 (3,680 samples)
   * Test: 2025-10-01 to 2025-12-31 (3,680 samples)
   * Splits do not overlap and preserve time ordering (**PASS**).
7. **Model Selection Integrity**: Candidate models are benchmarked exclusively on the Validation split; the test split is evaluated only once on the selected champion (**PASS**).
8. **Recursive Forecasting Autoregressive Loop**: Multi-step forecasting rolls predicted $\hat{y}_{t+k}$ into future lag positions without accessing future actual values (**PASS**).

---

## 5. Model Evaluation Reproduction

The training and evaluation pipeline was executed using `python scripts/train.py`. Results were reproduced exactly from raw inputs:

### Validation Set Benchmark (3,680 samples, July–September 2025):

| Model Candidate | MAE (Units) | RMSE (Units) | WAPE (%) | $R^2$ Score | Audit Verification |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **5.365** | **7.879** | **19.99%** | **0.853** | **PASS (Champion)** |
| Ridge Regression ($\alpha=10.0$) | 5.682 | 8.187 | 21.17% | 0.841 | **PASS** |
| 7-Day Moving Average | 7.039 | 10.201 | 26.23% | 0.753 | **PASS** |
| Seasonal Naive (Lag 7) | 7.510 | 10.843 | 27.99% | 0.721 | **PASS** |

### Held-Out Test Set Evaluation (3,680 samples, October–December 2025):
* **MAE**: 7.714 units (**PASS**)
* **RMSE**: 11.103 units (**PASS**)
* **WAPE**: 25.77% (**PASS**)
* **$R^2$**: 0.793 (**PASS**)

*Comparison with Initial Briefing*: All metrics match the reported numbers. The Random Forest Regressor consistently outperforms both heuristic baselines on every error metric. Output table verified at `reports/metrics/model_comparison.csv`.

---

## 6. Forecast Interval Assessment

* **Inspection**: Reviewed `src/forecasting/predict.py`.
* **Finding**: The uncertainty interval was initially described as a "90% confidence interval". Technically, multiplying point predictions by $\pm 1.645 \cdot \sigma$ is an **approximate normal uncertainty band based on empirical residual variance**, not a distribution-free conformal prediction interval.
* **Correction Applied**:
  * Labeled clearly in code, documentation, and metadata as **Empirical Residual Uncertainty Band (~90% normal approximation)**.
  * Formulated using validation residual standard deviation ($\sigma_{\text{resid}} = 7.879$):
    $$\text{spread} = 1.645 \cdot \sigma_{\text{resid}}$$
    $$\text{lower\_bound} = \max(0.0, \text{round}(\hat{y} - \text{spread}, 1))$$
    $$\text{upper\_bound} = \max(\text{lower\_bound}, \text{round}(\hat{y} + \text{spread}, 1))$$
  * Programmatic assertion `lower_bound <= upper_bound` and non-negativity enforced.
* **Status**: **PASS** (Verified with regression test `test_uncertainty_bounds_ordering`).

---

## 7. Inventory Formula Validation

Audited `src/inventory/demand_estimation.py`, `src/inventory/inventory_calculations.py`, and `src/inventory/reorder_recommendations.py`:

| Policy / Metric | Theoretical Formula | Implementation Verification | Status |
| :--- | :--- | :--- | :---: |
| **Inventory Position** | $IP = \text{On-Hand} + \text{On-Order} - \text{Backorders}$ | `inv_pos = on_hand + on_order - backorders` | **PASS** |
| **Safety Stock** | $SS = \max(\text{MIN}, \lceil Z \cdot \sigma_d \cdot \sqrt{L} \rceil)$ | Implemented with $Z=1.645$, $\text{MIN}=2$ | **PASS** |
| **Reorder Point** | $ROP = \lceil (d_{avg} \cdot L) + SS \rceil$ | Implemented and ceiling-rounded | **PASS** |
| **Order-Up-To Level** | $S = \lceil d_{avg} \cdot (R + L) + SS \rceil$ | Implemented with $R=7$ days review period | **PASS** |
| **Order Quantity** | $Q = \max(0, S - IP)$ if $IP \le ROP$ else $0$ | Implemented strictly; 0 when $IP > ROP$ | **PASS** |
| **Stockout Risk** | $\text{Days of Supply} = IP / d_{avg} < L$ | Correct unit comparison (days vs days) | **PASS** |
| **Boundary Guard** | $d_{avg} = 0$ division handling | Clamped to $0.01$ minimum to prevent ZeroDivisionError | **PASS** |
| **Negative Clamp** | Inputs bounded to $\ge 0$ | Clamped to $0$ in `evaluate_inventory_row` | **PASS** |

*Status*: **PASS** (Verified with dedicated boundary tests).

---

## 8. Dashboard Verification

* **Application File**: `app/dashboard.py` and multipage modules under `app/pages/`.
* **Compilation**: All 5 Streamlit scripts compiled with `py_compile.compile()` without syntax errors (**PASS**).
* **Startup Execution**: Started background daemon `streamlit run app/dashboard.py --server.headless true --server.port 8501`.
* **Network & HTTP Test**: Queried `http://localhost:8501` via HTTP client. Server returned **HTTP 200 OK** with complete HTML content containing `<title>RetailIQ` (**PASS**).
* **Browser Automation Limitation**: The `browser_subagent` encountered a known infrastructure limitation (Playwright binary driver download returned HTTP 404 from Azureedge CDN on the host machine). In accordance with instructions:
  * **Direct browser interaction and screenshot capture**: **BLOCKED** by environment Playwright binary missing.
  * **Headless server execution and HTTP 200 response**: **PASS**.
  * No synthetic placeholder screenshot was invented.

---

## 9. Output File Validation

| Output File | Expected Rows | Verified Rows | Validation Checks | Status |
| :--- | :---: | :---: | :--- | :---: |
| `outputs/forecasts/future_sales_forecasts.csv` | 1,200 | 1,200 | 30 forward days (2026-01-01 to 2026-01-30) for 40 series; non-negative; lower $\le$ upper | **PASS** |
| `outputs/inventory/inventory_metrics.csv` | 40 | 40 | 40 series evaluated; valid statuses (21 Reorder, 12 Critical, 7 Healthy); non-negative | **PASS** |
| `outputs/recommendations/replenishment_recommendations.csv` | $\le 40$ | 33 | Actionable reorders ($Q > 0$); order costs valid; non-negative | **PASS** |
| `reports/figures/` (9 PNG charts) | 9 | 9 | All figures exist, $> 50\text{ KB}$, valid PNG headers | **PASS** |
| `reports/metrics/model_comparison.csv` | 4 | 4 | 4 candidate models with MAE, RMSE, WAPE, R2 | **PASS** |

---

## 10. Automated Test Results

Executed with `pytest -v`:

```
============================= test session starts =============================
platform win32 -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Diploma Course\DS\DataScience-Retail Sales Forecasting & Inventory Optimization System
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1, cov-7.1.0
collected 31 items

tests/test_data_validation.py .......                                    [ 22%]
tests/test_feature_engineering.py .....                                  [ 38%]
tests/test_forecasting.py ......                                         [ 58%]
tests/test_inventory.py .........                                        [ 87%]
tests/test_preprocessing.py ....                                         [100%]

============================= 31 passed in 1.61s ==============================
```

* **Total Tests Collected**: 31
* **Total Passed**: 31
* **Total Failed**: 0
* **Total Skipped**: 0
* **Test Coverage**: Schema validation, missing value imputation, deduplication, financial derivation, calendar features, leak-free lag calculations, multi-group rolling alignment, model metrics, baseline heuristics, uncertainty bounds ordering, safety stock, ROP, target stock, zero demand boundaries, negative clamping, and filepath resolution.

---

## 11. GitHub Readiness

* **Requirements File**: Clean `requirements.txt` with appropriate version bounds (**PASS**).
* **Git Ignore**: Covers `__pycache__`, `.venv`, `.pytest_cache`, `.coverage`, `.env` (**PASS**).
* **License**: Valid MIT License included (**PASS**).
* **No Hardcoded Absolute Paths**: Verified via ripgrep across all repository files (**PASS**).
* **No Exposed Secrets**: Verified via ripgrep for API keys, passwords, and private tokens (**PASS**).
* **Synthetic Data Transparency**: Clearly labeled in README, reports, code, and data files (**PASS**).
* **Commit History**: Initial clean commit recorded on `master` branch (**PASS**).

---

## 12. Defects Found

1. **Defect #1 — Missing `Path` import in `src/data/load_data.py`**:
   * *Problem*: Calling `load_raw_data(filepath)` raised `NameError: name 'Path' is not defined` because `Path` was omitted from imports.
   * *Impact*: Breaking failure if users passed a custom filepath string.
   * *Resolution*: Added `from pathlib import Path` to `src/data/load_data.py`. Added regression test `test_load_raw_data_with_filepath`.
2. **Defect #2 — Grouped Rolling Index Alignment in `src/features/feature_engineering.py`**:
   * *Problem*: `shifted_target.groupby(...).rolling(...).reset_index(drop=True)` caused index misassignment when assigned back to sorted DataFrame.
   * *Impact*: Rolling statistics between store/product series with different sort orders could misassign rolling averages.
   * *Resolution*: Replaced with `shifted_target.groupby(...).transform(lambda s: s.rolling(...).mean())` after resetting index on `df_sorted`. Added regression test `test_multi_group_rolling_alignment`.
3. **Defect #3 — Categorical Encoder Fit Timing in `src/forecasting/train_model.py`**:
   * *Problem*: `OrdinalEncoder` was fitted on the full dataset before chronological splitting.
   * *Impact*: Categorical target leakage risk across splits.
   * *Resolution*: Moved splitting prior to encoding; fitted encoder strictly on `train_df`.
4. **Defect #4 — Script `sys.path` Resolution in `scripts/`**:
   * *Problem*: Running `python scripts/train.py` from outside the root directory failed with `ModuleNotFoundError: No module named 'src'`.
   * *Impact*: Operational failure when running scripts directly.
   * *Resolution*: Added robust `ROOT_DIR` insertion into `sys.path` in all scripts.
5. **Defect #5 — Ambiguous Uncertainty Interval Nomenclature in `src/forecasting/predict.py`**:
   * *Problem*: Normal approximation based on RMSE was called a "90% confidence interval" without calibration disclaimer.
   * *Impact*: Misleading statistical claim.
   * *Resolution*: Refactored to compute empirical validation residual variance, labeled as **Empirical Residual Uncertainty Band (~90% normal approximation)**, and enforced $\text{lower\_bound} \le \text{upper\_bound}$ and $\text{lower} \ge 0$. Added regression test `test_uncertainty_bounds_ordering`.

---

## 13. Changes Made

* Modified `src/data/load_data.py` (added `Path` import).
* Modified `src/features/feature_engineering.py` (fixed rolling window transform alignment).
* Modified `src/forecasting/train_model.py` (strict training-only encoder fit and validation residual tracking).
* Modified `src/forecasting/predict.py` (empirical residual uncertainty calculation and bound assertions).
* Modified `scripts/prepare_data.py`, `scripts/train.py`, `scripts/generate_forecast.py` (`sys.path` setup).
* Modified `tests/test_data_validation.py`, `tests/test_feature_engineering.py`, `tests/test_forecasting.py`, `tests/test_inventory.py` (added 5 regression tests).
* Created `reports/metrics/data_validation_report.txt` (full synthetic audit metrics).
* Created `reports/technical_validation_report.md` (this audit document).

---

## 14. Remaining Risks

1. **Lead Time Determinism**: Safety stock formula assumes constant lead time $L$. In physical supply chains with stochastic supplier delays ($L \sim \mathcal{D}$), safety stock must expand to incorporate lead time variance.
2. **Recursive Horizon Compounding**: 30-day recursive forecasting rolls predicted values forward; for horizons beyond 30 days, direct multi-horizon models (e.g. Seq2Seq) should be evaluated to mitigate recursive error compounding.
3. **Host Playwright Environment**: The host Windows environment lacks the Playwright binary driver for automated browser control. Manual or native browser access to `http://localhost:8501` is required for visual dashboard inspection.

---

## 15. Final Go/No-Go Recommendation

### Recommendation: **GO FOR PORTFOLIO PUBLICATION**

* **Criteria Met**: All 15 project objectives and 10 technical audit requirements are satisfied.
* **Reproducibility**: Models, metrics, and forecasts reproduce deterministically with fixed random seed 42.
* **Code Quality**: Production-grade modularity, clean documentation, zero hardcoded paths, zero secrets, and 31 automated tests passing.
