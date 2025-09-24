---
title: "What is Non-Linear Regression?"
description: "Overview of non-linear regression models: polynomial, exponential, logarithmic, logistic with Python examples"
tags:
  - regression
  - non-linear
  - modeling
  - fundamentals
---
# What is Non-Linear Regression?

Linear Regression assumes a straight-line relationship between input feature(s) and output target. In many real-world scenarios, relationships are **curved, saturating, multiplicative, or otherwise non-linear**. When a linear (affine) model cannot adequately capture the pattern, we turn to **Non-Linear Regression**.

## General Form
$Y = f(X) + \varepsilon$
Where $f(X)$ is a non-linear function of the predictors and $\varepsilon$ is the error term.

## Examples of Non-Linear Relationships
- **Exponential growth / decay**: Population growth, radioactive decay
- **Polynomial trends**: U-shaped or inverted U-shapes (economics, sales over time)
- **Logarithmic**: Diminishing returns (learning curves, efficiency vs resources)
- **Logistic (sigmoidal)**: Growth saturating at a carrying capacity (population, adoption curves)

## Common Non-Linear Model Types
### 1. Polynomial Regression
Extends linear regression by adding higher-order terms of features. Though the model is *linear in parameters*, the relationship between original $X$ and $Y$ becomes non-linear.

General (single feature):
$Y = b_0 + b_1 X + b_2 X^2 + b_3 X^3 + \dots + b_d X^d + \varepsilon$

### 2. Exponential Regression
Captures multiplicative / exponential growth or decay:
$Y = a e^{bX}$
Often log-transformable: $\ln Y = \ln a + bX$ when $Y>0$.

### 3. Logarithmic Regression
Useful for diminishing returns:
$Y = a + b \ln(X)$

### 4. Logistic Function (S-curve)
For bounded growth (continuous) or as the link in **Logistic Regression** for classification probability modeling:
$p = \frac{1}{1 + e^{-(w_0 + w_1 X_1 + \dots + w_n X_n)}}$
While *logistic regression* is typically presented as a **classification** technique, it is conceptually a non-linear regression on probability space via the sigmoid.

> Note: Many other non-linear forms exist (power law, Michaelis–Menten, Gompertz, spline-based, kernel methods, neural networks).

## Example 1: Polynomial Regression (Python)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([2.3, 2.9, 7.6, 15.3, 26.8, 40.5, 61.1, 85.2, 120.3])

# Polynomial features (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression (Degree 2)")
plt.show()
```

## Example 2: Exponential Regression (Curve Fitting)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_func(x, a, b):
		return a * np.exp(b * x)

x_data = np.array([0, 1, 2, 3, 4, 5])
y_data = np.array([2, 3, 7, 20, 55, 148])

params, _ = curve_fit(exp_func, x_data, y_data)
a, b = params

x_line = np.linspace(0, 5, 100)
y_line = exp_func(x_line, a, b)

plt.scatter(x_data, y_data, color='blue')
plt.plot(x_line, y_line, color='red')
plt.title("Exponential Regression Example")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
print(f"Estimated parameters: a={a:.4f}, b={b:.4f}")
```

## Linear vs Non-Linear Regression (Comparison)
| Aspect | Linear Regression | Non-Linear Regression |
|--------|------------------|------------------------|
| Model Shape | Straight line / hyperplane | Curve (polynomial, exponential, logistic, etc.) |
| Equation Form | $Y = a + bX$ | $Y = f(X)$ (non-linear $f$) |
| Use Case | Simple linear trends | Complex / curved patterns |
| Computation | Closed-form (OLS) or simple | May require iterative curve fitting |
| Interpretability | High (coefficients) | Varies (may be harder) |
| Risk | Underfit curved data | Overfit if too flexible |

## Practical Notes
- Start with exploratory plots (scatter + smoothing) to judge linear vs non-linear structure.
- Polynomial degree should be chosen carefully (use validation or regularization like Ridge/Lasso on polynomial features).
- For exponential/log models, check domain constraints ($Y>0$, $X>0$ for logs).
- Consider transformations (log, Box-Cox) before jumping to complex non-linear solvers.
- For highly flexible unknown forms, splines or tree-based / kernel / neural models may outperform parametric curves.

## Summary
- Linear Regression only fits straight-line relationships.
- Non-Linear Regression captures curved, saturating, or multiplicative patterns.
- Model choice depends on observed data patterns and interpretability needs.

> Next: We can explore logistic regression (classification) or regularization of polynomial models if desired.