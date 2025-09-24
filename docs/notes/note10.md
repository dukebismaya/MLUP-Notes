---
title: "Multiple Linear Regression"
description: "Definition, formula, OLS solution, assumptions, diagnostics, Python example, interpretation, and applications"
tags:
	- regression
	- linear-models
	- supervised-learning
	- fundamentals
---
# Multiple Linear Regression (MLR)

**Multiple Linear Regression** extends simple linear regression to model the relationship between a continuous target variable and **two or more** predictor variables.

## 1. Mathematical Model
$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \varepsilon$
Where:
- $Y$ = dependent (target) variable
- $X_1, X_2, \dots, X_p$ = independent features
- $\beta_0$ = intercept
- $\beta_i$ = coefficient (marginal effect of feature $i$ holding others constant)
- $\varepsilon$ = error term (unexplained noise)

### Matrix Form
Let $X \in \mathbb{R}^{n \times (p+1)}$ with first column of ones (intercept), $\beta \in \mathbb{R}^{p+1}$, $y \in \mathbb{R}^n$:
$y = X\beta + \varepsilon$

### Ordinary Least Squares (OLS) Solution
Minimize $\|y - X\beta\|_2^2$. Closed-form (when $X^T X$ invertible):
$\hat{\beta} = (X^T X)^{-1} X^T y$

> In practice, libraries use QR decomposition / SVD for numerical stability.

## 2. Core Assumptions
| Assumption | Description | Violation Indicator | Potential Remedies |
|------------|-------------|---------------------|--------------------|
| Linearity | Expected value of $Y$ is linear in coefficients | Curvature in residual vs fitted | Add transforms / interactions |
| Independence | Observations are independent | Autocorrelation in residuals | Use time-series models / cluster robust SE |
| Homoscedasticity | Constant error variance | Funnel pattern, Breusch-Pagan p < 0.05 | Transform target, weighted / robust regression |
| Normality (errors) | Residuals ~ Normal (for inference) | Heavy tails in Q-Q | Larger sample CLT, transform, robust SE |
| No Multicollinearity | Predictors not redundant | High VIF (>5 or >10) | Drop / combine features, regularize |

See earlier notes on residual diagnostics (note6) for plots & tests.

## 3. Python Example
Predict student exam score from **hours studied** and **hours slept**.
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = {
		"Hours_Studied": [5, 10, 8, 12, 15],
		"Hours_Slept":   [7,  6, 8,  5,  4],
		"Exam_Score":    [75, 85, 82, 90, 95]
}
df = pd.DataFrame(data)

X = df[["Hours_Studied", "Hours_Slept"]]
y = df["Exam_Score"]

model = LinearRegression().fit(X, y)
pred = model.predict(X)

print("Intercept (β0):", model.intercept_)
print("Coefficients (β1, β2):", model.coef_)
print("R²:", r2_score(y, pred))
print("MSE:", mean_squared_error(y, pred))

# Adjusted R²
n = len(y)
p = X.shape[1]
r2 = r2_score(y, pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R²:", adj_r2)

# VIF Check (small sample just illustrative)
X_with_const = np.column_stack([np.ones(len(X)), X.values])
vif = []
for i in range(1, X_with_const.shape[1]):  # skip intercept column
		vif_val = variance_inflation_factor(X_with_const, i)
		vif.append((X.columns[i-1], vif_val))
print("VIF:", vif)

# Predict for a new student
new_student = np.array([[14, 6]])
print("Prediction (14 study hrs, 6 sleep hrs):", model.predict(new_student)[0])
```
> NOTE: Sample size is extremely small (n=5); metrics are not reliable—example is pedagogical.

## 4. Interpretation of Results
- **Intercept ($\beta_0$)**: Expected exam score if all predictors are 0 (may be extrapolation; interpret cautiously).
- **Coefficient $\beta_1$ (Hours_Studied)**: Expected change in score per extra study hour holding sleep constant.
- **Coefficient $\beta_2$ (Hours_Slept)**: Expected change in score per extra sleep hour holding study constant.
- **$R^2$ / Adjusted $R^2$**: Variance explained; adjusted penalizes complexity.
- **VIF**: Gauges multicollinearity; near 1 is good, high values flag redundancy.

## 5. Common Applications
- House price prediction (size, rooms, location, age)
- Sales forecasting (ad spend, seasonality, discounting)
- Salary estimation (experience, education, role features)
- Medical outcomes (biomarkers, lifestyle indicators)
- Energy consumption (weather, occupancy, equipment status)

## 6. Pitfalls & Remedies
| Issue | Symptom | Strategy |
|-------|---------|----------|
| Multicollinearity | Unstable coefficients, high VIF | Drop / combine features, Ridge regression |
| Omitted variable bias | Residual structure, low $R^2$ | Add relevant predictors, domain consultation |
| Overfitting | High train $R^2$, low test $R^2$ | Cross-validation, regularization |
| Heteroscedasticity | Patterned residual spread | Transform target, robust SE, WLS |
| Non-linearity | Curved residual patterns | Add interactions, polynomial terms, splines |

## 7. Relation to Other Linear Models
- **Simple Linear Regression**: Special case with $p=1$.
- **Polynomial Regression**: Linear model on expanded (non-linear) basis.
- **Regularized Models**: Ridge/Lasso mitigate multicollinearity & overfitting.
- **Generalized Linear Models (GLMs)**: Extend linear predictor to non-Gaussian targets via link functions.

## 8. Summary
Multiple Linear Regression provides a transparent baseline for modeling continuous outcomes with multiple predictors. Its validity depends on assumptions; diagnostics (see note6) and careful feature engineering improve reliability.

> Next: We can explore **regularization (Ridge/Lasso)** or **interaction terms & feature engineering**.