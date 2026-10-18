# Inventory Optimization Formulas & Supply Chain Policy

This reference document outlines the mathematical formulas, inventory policies, assumptions, and constraints implemented in the system.

---

## 1. Key Definitions & State Variables

* **$On\_Hand$**: Physical inventory currently present in the store / distribution center.
* **$On\_Order$**: Pipeline inventory purchased from supplier currently in transit.
* **$Backorders$**: Unfulfilled demand owed to customers (assumed 0 in lost-sales retail).
* **$Inventory\_Position\ (IP)$**: Effective inventory available to meet future demand:
  $$IP = On\_Hand + On\_Order - Backorders$$
* **$L$**: Supplier replenishment lead time in days ($L \ge 1$).
* **$R$**: Replenishment review period in days (e.g., $R=7$ days for weekly periodic review).
* **$d_{avg}$**: Expected daily demand rate (estimated via empirical historical velocity or ML forecast).
* **$\sigma_d$**: Standard deviation of daily demand.

---

## 2. Lead Time Demand & Uncertainty

Under the standard assumption that daily demand increments $D_1, D_2, \dots, D_L$ are independent and identically distributed (i.i.d.) random variables with mean $d_{avg}$ and variance $\sigma_d^2$:

* **Expected Lead Time Demand ($LTD$)**:
  $$E[D_L] = \sum_{i=1}^L E[D_i] = d_{avg} \cdot L$$
* **Variance of Lead Time Demand**:
  $$\text{Var}(D_L) = \sum_{i=1}^L \text{Var}(D_i) = L \cdot \sigma_d^2$$
* **Standard Deviation of Lead Time Demand**:
  $$\sigma_L = \sqrt{L} \cdot \sigma_d$$

> **Documented Assumptions**:
> 1. Replenishment lead time $L$ is deterministic and constant.
> 2. Daily demand follows approximately finite normal variance (Central Limit Theorem applied over lead time).
> 3. Demand across consecutive days during lead time exhibits zero or minimal autocorrelation.

---

## 3. Safety Stock ($SS$)

Safety stock protects against stochastic demand surges during the supplier lead time window:

$$SS = \max\left(\text{MIN\_BUFFER}, \left\lceil Z \cdot \sigma_d \cdot \sqrt{L} \right\rceil\right)$$

Where:
* $Z$ is the standard normal distribution quantile corresponding to the target **Cycle Service Level (CSL)**:
  * 85% Service Level: $Z = 1.036$
  * 90% Service Level: $Z = 1.282$
  * **95% Service Level (Default)**: $Z = 1.645$
  * 98% Service Level: $Z = 2.054$
  * 99% Service Level: $Z = 2.326$
* $\text{MIN\_BUFFER} = 2$ units (ensures physical coverage for low-demand SKUs).

---

## 4. Reorder Point ($ROP$)

The replenishment threshold that triggers a purchase order:

$$ROP = \left\lceil (d_{avg} \cdot L) + SS \right\rceil$$

If $IP \le ROP$, a replenishment order is triggered immediately.

---

## 5. Periodic Review Order-Up-To Level ($S$)

In a periodic review $(R, S)$ policy where inventory is reviewed every $R$ days, the system must buffer against demand over the combined **exposure period** $(R + L)$:

$$S = \left\lceil d_{avg} \cdot (R + L) + SS \right\rceil$$

---

## 6. Suggested Order Quantity ($Q$)

When $IP \le ROP$, the recommended order quantity elevates the inventory position back to target stock $S$:

$$Q = \begin{cases} \max(0, S - IP), & \text{if } IP \le ROP \\ 0, & \text{if } IP > ROP \end{cases}$$

---

## 7. Days of Supply ($DOS$) & Stockout Risk

* **Days of Supply**:
  $$DOS = \frac{IP}{d_{avg}}$$
* **Stockout Risk Flag**:
  $$\text{Risk} = \begin{cases} \text{True}, & \text{if } DOS < L \\ \text{False}, & \text{otherwise} \end{cases}$$
  *Interpretation*: If the remaining inventory coverage is shorter than the supplier's delivery lead time, the store will stock out before new merchandise arrives unless an emergency expedited delivery is scheduled.

---

## 8. Inventory Status Classification

| Operational State | Mathematical Condition | Recommended Action |
| :--- | :--- | :--- |
| **Critical / Stockout** | $IP \le 0$ | Immediate emergency reorder; assess supplier expediting. |
| **Reorder Required** | $0 < IP \le ROP$ | Issue standard purchase order of quantity $Q = S - IP$. |
| **Low Stock** | $ROP < IP \le ROP + 0.5 \cdot SS$ | Monitor closely; queue for next scheduled review cycle. |
| **Healthy Stock** | $ROP + 0.5 \cdot SS < IP \le S \cdot 1.25$ | Ideal stock posture; no action required. |
| **Overstock Risk** | $IP > S \cdot 1.25$ | Excess holding capital; curtail upcoming orders, consider promotion. |
