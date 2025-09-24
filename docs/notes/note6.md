---
title: "What is Linear Regression?"
description: "Introduction to linear regression: concept, formula, types, steps, Python example, and evaluation metrics"
tags:
  - regression
  - supervised-learning
  - fundamentals
  - linear-models
---

# What is Linear Regression?

**Linear Regression** is a supervised learning algorithm used to predict a **continuous** outcome. It models the relationship between one or more independent variables (features) and a dependent variable (target) by fitting a line (in higher dimensions: a hyperplane) that minimizes error.

## Core Idea / Formula
For a simple (single feature) linear regression the model is:

$$Y = mX + c + \varepsilon$$

Where:
- $Y$ = Dependent variable (target)
- $X$ = Independent variable (feature)
- $m$ = Slope (coefficient / weight)
- $c$ = Intercept (bias term)
- $\varepsilon$ = Error term (residual noise)

For multiple features ($n$ features):

$$Y = w_0 + w_1 X_1 + w_2 X_2 + \dots + w_n X_n + \varepsilon$$

Vector form: $\hat{y} = Xw$ (with intercept handled via bias or augmented feature vector).

## Types of Linear Regression
- **Simple Linear Regression** – One independent variable ($X$).  
  Example: Predicting house price using only square footage.
- **Multiple Linear Regression** – Multiple independent variables ($X_1, X_2, \dots, X_n$).  
  Example: Predicting house price using size, rooms, location, age.

> Variants (later study): Ridge, Lasso, Elastic Net (regularized linear models).

## Steps in Linear Regression
1. Collect dataset.
2. Define dependent variable ($Y$) and independent features ($X$).
3. Split data (train/test) if doing generalization evaluation.
4. Fit model using **Ordinary Least Squares (OLS)** → minimizes Sum of Squared Errors.
5. Generate predictions on training (and test) data.
6. Evaluate performance (MSE, RMSE, $R^2$).
7. Inspect residuals for patterns (check assumptions).

## Python Example (Simple Linear Regression)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset: Study Hours vs Scores
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Scores': [10, 20, 30, 40, 50, 55, 65, 80, 90]
}

df = pd.DataFrame(data)

# Feature matrix (must be 2D) and target vector
X = df[['Hours']]  # shape (n, 1)
y = df['Scores']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R^2 Score:", r2_score(y, y_pred))

# Visualization
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Fitted Line')
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
```

## Key Evaluation Metrics
| Metric | Definition | Notes |
|--------|------------|-------|
| **MSE** (Mean Squared Error) | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Penalizes larger errors strongly |
| **RMSE** (Root MSE) | $\sqrt{\text{MSE}}$ | Same units as target |
| **$R^2$ Score** | $1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}$ | Proportion of variance explained |

## Residual Diagnostics
Understanding residuals (errors) is critical to validate linear regression assumptions.

### What Are Residuals?
Residual $e_i = y_i - \hat{y}_i$ (difference between actual and predicted). A good linear model leaves **random, structureless** residuals.

### Key Assumptions Checked via Residuals
| Assumption | What to Look For | Violation Pattern |
|------------|------------------|-------------------|
| Linearity | No systematic curve in residuals vs fitted | U-shape or curvature |
| Homoscedasticity (constant variance) | Even vertical spread | Funnel (widening/narrowing) |
| Independence | No obvious time/order pattern | Trend / cycles |
| Normality (for inference) | Residuals approx. normal | Heavy tails / skew in histogram & Q-Q |
| No high leverage / influence | Points not dominating fit | Outliers far from trend |

### Common Diagnostic Plots
| Plot | Purpose | Healthy Pattern |
|------|---------|-----------------|
| Residuals vs Fitted | Detect non-linearity & heteroscedasticity | Horizontal cloud around 0 |
| Histogram / KDE of Residuals | Assess normality | Bell-shaped, centered at 0 |
| Q-Q Plot | Quantify deviation from normal | Points ~ straight line |
| Residuals vs Feature (each X) | Check overlooked non-linearity | No structure |
| Scale-Location (|resid|^0.5 vs fitted) | Homoscedasticity | Flat horizontal band |
| Leverage vs Residual (Cook’s Distance) | Influential points | Few, small influence |

### Remedies When Issues Found
| Issue | Potential Fixes |
|-------|----------------|
| Non-linearity | Add polynomial terms, interactions, transform features |
| Heteroscedasticity | Transform target (log), weighted least squares, robust regression |
| Non-normal residuals | Often harmless with large n; else transform or use robust methods |
| Autocorrelation | Use time-series models (ARIMA), add lag features |
| Influential points | Investigate data quality, robust regression |

### Mini Example (Residual Plot)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.linspace(1, 9, 9)
y = np.array([10, 20, 30, 40, 50, 55, 65, 80, 90])
X_2d = X.reshape(-1, 1)
model = LinearRegression().fit(X_2d, y)
preds = model.predict(X_2d)
resid = y - preds

plt.scatter(preds, resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()
```

> Interpreting residuals is as important as the initial accuracy metrics—use diagnostics before trusting model conclusions.

---
**Next Deep Dive:** An in-repo companion Jupyter notebook `notebooks/linear_regression_diagnostics.ipynb` includes:

### Interactive Exploration Notebook

Features provided there:
- Full residual diagnostic panel (residuals vs fitted, histogram/KDE, Q-Q, scale-location, leverage vs residuals with Cook's distance, autocorrelation bars).
- Statistical assumption tests: Jarque-Bera, Shapiro-Wilk, Breusch-Pagan, Durbin-Watson, Ljung-Box.
- Multicollinearity check via Variance Inflation Factor (VIF).
- Interactive refit panel (Linear / Ridge / Lasso) with alpha slider & feature selection to observe how diagnostics change.
- Figure export helper (`save_current_figures()`) saving timestamped PNGs under `notebooks/reports/`.
- Residuals export (CSV always; Parquet if optional dependency installed) enriched with leverage + Cook’s distance.

#### Enabling Parquet Export
Install either `pyarrow` or `fastparquet` inside the active virtual environment:
```
pip install pyarrow
# or
pip install fastparquet
```
Then re-run the export cell (Section 13) in the notebook.

> Tip: Run cells sequentially the first time; for exploration you can just re-run the interactive cell (Section 11) after adjusting controls.