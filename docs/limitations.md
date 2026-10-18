# System Limitations, Assumptions & Future Roadmap

This document provides transparent documentation of known limitations, modeling assumptions, and engineering recommendations for enterprise-scale deployment.

---

## 1. Simulation vs Commercial Reality
* **Synthetic Generator**: The system is calibrated on synthetic observations generated from stochastic process equations rather than proprietary ERP/POS databases. While trends, holiday patterns, lead times, and stockouts are faithfully simulated, real-world retail displays complex tail events, store cannibalization, and cross-elasticities not modeled here.
* **Censored Latent Demand**: In physical retail, when an item stocks out, true customer demand is censored (unobserved). In our simulation, physical `units_sold` is strictly capped by `on_hand_stock`. For production deployment with real store POS data, expectation-maximization (EM) or Kaplan-Meier survival curves should be applied to un-censor demand prior to training.

---

## 2. Supply Chain & Inventory Assumptions
1. **Deterministic Lead Time**: The safety stock formulation ($SS = Z \cdot \sigma_d \cdot \sqrt{L}$) assumes lead time $L$ is constant. In real international supply chains, supplier lead time is itself a stochastic random variable ($L \sim \mathcal{D}$). When lead time variance $\sigma_L^2 > 0$, the expanded formula must be utilized:
   $$SS = Z \cdot \sqrt{L \cdot \sigma_d^2 + d_{avg}^2 \cdot \sigma_L^2}$$
2. **Infinite Supplier Capacity**: The periodic review model assumes suppliers can fulfill any order quantity $Q$ without minimum order quantities (MOQ), pallet batch size constraints, or supplier stockouts.
3. **Linear Holding Costs**: Holding costs are modeled as an annualized percentage of unit cost (20%). Warehouse cubic volume constraints, perishability spoilage, and obsolescence discounting are not currently modeled dynamically.

---

## 3. Modeling & Algorithmic Considerations
1. **Autoregressive Multi-Step Compounding**: Recursive multi-step forecasting ($H=30$) rolls predicted values into subsequent lag features. If day $t+1$ prediction carries error, that error can compound across long horizons. Direct multi-horizon forecasting (e.g. Seq2Seq, Temporal Fusion Transformers, or separate models for $t+1, \dots, t+30$) would mitigate recursive degradation.
2. **Cold-Start SKUs**: Feature engineering requires a minimum 28-day lag warm-up. Brand new products without sales history must leverage hierarchical cross-sectional category medians until sufficient history accumulates.

---

## 4. Production Roadmap
* [ ] Integrate stochastic supplier lead time modeling.
* [ ] Support Multi-Echelon Inventory Optimization (MEIO) coordinating central Distribution Centers (DC) with regional spoke stores.
* [ ] Incorporate macroeconomic external regressors (local weather anomalies, CPI, competitor promotional scraping).
* [ ] Implement automated ML retraining pipelines with Feast feature store and MLflow model registry.
