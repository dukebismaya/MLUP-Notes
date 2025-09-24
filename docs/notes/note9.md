---
title: "Evaluation Metrics in Regression Models"
description: "Comprehensive overview of regression evaluation metrics: error-based, goodness-of-fit, and advanced metrics with Python examples"
tags:
  - regression
  - evaluation
  - metrics
  - fundamentals
---
# Evaluation Metrics in Regression Models

Regression metrics measure how well a model predicts **continuous** outcomes. They fall broadly into: **error magnitude metrics** and **goodness-of-fit metrics**, with some specialized variants for particular data characteristics.

## 1. Error-Based Metrics

### Mean Absolute Error (MAE)
$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
Average absolute deviation; intuitive and in original units. Less sensitive to large outliers than squared metrics.

**Use when:** All errors should contribute proportionally; robust-ish interpretation.

### Mean Squared Error (MSE)
$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
Squares residuals → emphasizes large mistakes heavily.

**Use when:** Large deviations are very costly (risk-sensitive domains).

### Root Mean Squared Error (RMSE)
$\text{RMSE} = \sqrt{\text{MSE}}$
Same units as target; balances interpretability with sensitivity to large errors.

### Mean Absolute Percentage Error (MAPE)
$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$
Expresses error as a percentage. **Avoid when $y_i = 0$ or near zero**, as it explodes.

**Use when:** Stakeholders care about relative error (e.g., demand forecasting).

### Mean Squared Logarithmic Error (MSLE)
$\text{MSLE} = \frac{1}{n} \sum_{i=1}^{n} (\log(1+y_i) - \log(1+\hat{y}_i))^2$
Penalizes underestimation more than overestimation; smooths exponential growth.

**Use when:** Targets grow multiplicatively (population, viral adoption, sales scaling).

### Huber Loss (Metric Perspective)
Hybrid of MAE and MSE with threshold $\delta$:
$L_\delta(e) = \begin{cases} \frac{1}{2} e^2 & |e| \le \delta \\ \delta(|e| - \frac{1}{2}\delta) & |e| > \delta \end{cases}$
Less sensitive to outliers than MSE; smoother than pure MAE.

**Use when:** Some outliers exist but complete robustness of MAE not desired.

## 2. Goodness-of-Fit Metrics

### $R^2$ (Coefficient of Determination)
$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}, \quad SS_{res} = \sum (y_i - \hat{y}_i)^2, \quad SS_{tot} = \sum (y_i - \bar{y})^2$
Proportion of variance explained. Negative values mean model underperforms a naive mean predictor.

### Adjusted $R^2$
$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$
Corrects $R^2$ inflation when adding non-informative predictors (with $p$ features, $n$ samples).

## 3. Python Example
```python
import numpy as np
from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, r2_score,
	mean_absolute_percentage_error, mean_squared_log_error
)
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate sample regression data
X, y = make_regression(n_samples=80, n_features=1, noise=12, random_state=42)
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Avoid negative or zero values before MSLE / MAPE (example guard)
if (y > -1).all() and (y_pred > -1).all():
	try:
		msle = mean_squared_log_error(y, y_pred)
	except ValueError:
		msle = None
else:
	msle = None

mape = mean_absolute_percentage_error(y, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"MSLE: {msle if msle is not None else 'n/a (invalid for negative values)'}")
```

## 4. Choosing the Right Metric
| Scenario | Preferred Metrics | Rationale |
|----------|------------------|-----------|
| Balanced general performance | RMSE + MAE | Two perspectives on error magnitude |
| Outliers present | MAE / Huber | Reduced sensitivity to large errors |
| Large errors very costly | MSE / RMSE | Squares large residuals |
| Percentage importance | MAPE (if no zeros) | Relative interpretability |
| Multiplicative growth | MSLE | Log-scaling stabilizes growth |
| Feature set comparison | Adjusted $R^2$ | Penalizes complexity |

## 5. Practical Tips
1. Always pair at least one absolute error metric (MAE/RMSE) with a relative/explanatory one ($R^2$ / Adjusted $R^2$).
2. Inspect residual distributions; metrics can hide structure (see residual diagnostics in earlier notes).
3. Prefer cross-validation averages over single split scores for model selection.
4. For skewed targets, consider log-transform + back-transform carefully when reporting.
5. Avoid MAPE when actual values can be zero or extremely small.

## 6. Summary
- **Error Metrics:** MAE, MSE, RMSE, MAPE, MSLE, Huber quantify prediction error specifics.
- **Goodness-of-Fit:** $R^2$, Adjusted $R^2$ explain variance captured.
- **No universal best metric**—selection depends on domain cost structure and data distribution.

> Next: Dive deeper into **residual analysis**, **prediction intervals**, or **probabilistic regression metrics** (Pinball loss, CRPS) if needed.