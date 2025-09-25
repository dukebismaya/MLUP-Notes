---
title: "Model Evaluation in Regression"
description: "Key regression evaluation metrics: MAE, MSE, RMSE, R², Adjusted R² with formulas, Python example, and selection guidance"
tags:
  - regression
  - evaluation
  - metrics
  - fundamentals
---
# Model Evaluation in Regression

Evaluating a regression model requires looking at **error magnitude** and **variance explained**. No single metric tells the whole story—using a combination helps avoid misleading conclusions.

## 1. Core Regression Metrics

### (a) Mean Absolute Error (MAE)
$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
Average absolute deviation. Interpretable (same units as target). Robust to large outliers relative to squared metrics.

### (b) Mean Squared Error (MSE)
$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
Squares residuals → penalizes larger errors more. Smooth gradient (useful for optimization). Sensitive to outliers.

### &#40;c&#41; Root Mean Squared Error (RMSE)
$\text{RMSE} = \sqrt{\text{MSE}}$
Same units as target. More interpretable than MSE; still emphasizes larger errors.

### (d) Coefficient of Determination ($R^2$)
$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
Where:
$SS_{res} = \sum (y_i - \hat{y}_i)^2 \qquad SS_{tot} = \sum (y_i - \bar{y})^2$
Represents proportion of variance explained. Range roughly: $(-\infty, 1]$. Negative means model is worse than predicting the mean.

### (e) Adjusted $R^2$ (Multiple Regression)
$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$
Penalizes addition of non-informative predictors (where $p$ = number of features, $n$ = samples).

## 2. Python Example
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1.5, 3.5, 4.2, 5.0, 7.8, 8.5])

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

n = len(y)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
print(f"Adjusted R^2: {adj_r2:.4f}")
```

## 3. When to Use Which Metric
| Metric | Use When | Strengths | Cautions |
|--------|----------|-----------|----------|
| MAE | Need interpretability | Linear penalty, robust-ish | Under-penalizes rare large errors |
| MSE | Large errors costly | Smooth & differentiable | Inflated by outliers |
| RMSE | Stakeholders expect natural units | Balances frequency & size | Still sensitive to outliers |
| $R^2$ | Compare baseline vs model | Scale-free | Misleading with non-linear patterns, can be high with bias |
| Adjusted $R^2$ | Feature set comparison | Penalizes complexity | Not for non-linear/cross-validated model selection |

## 4. Additional (Optional) Metrics
- **Median Absolute Error (MedAE)**: Robust to extreme outliers.
- **MAPE**: Percentage error; avoid when $y_i$ can be zero or near zero.
- **SMAPE**: Symmetric variant of MAPE.
- **$R^2$ on test folds**: Use cross-validation for reliable generalization estimate.

## 5. Practical Guidance
1. Always inspect residual plots alongside metrics.
2. Report both an absolute error (MAE/RMSE) and a relative/explanatory metric ($R^2$).
3. Beware high $R^2$ with systematic bias (check residual structure).
4. Use cross-validation for model comparison instead of single train-test splits where feasible.
5. For skewed targets, consider log-transforming before evaluation (then invert carefully when interpreting).

## 6. Summary
- **MAE / MSE / RMSE** quantify average error magnitude.
- **$R^2$ / Adjusted $R^2$** quantify variance explained.
- No single metric is sufficient—combine them and validate with residual diagnostics.

> Next: We can explore error analysis (residual stratification, prediction intervals) or evaluation under distribution shift.