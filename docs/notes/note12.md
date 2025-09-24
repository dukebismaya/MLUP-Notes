---
title: "Regression Model Implementations on Diverse Datasets"
description: "Practical implementations: simple & multiple linear regression, polynomial regression, logistic classification contrast, and a real-world housing dataset"
tags: [regression, linear-regression, polynomial-regression, logistic-regression, implementation]
---

# Implementation of Regression Models on Various Datasets

This note walks through hands-on implementations of common regression (and a related classification) techniques using small synthetic and real datasets. You will see how to construct models, evaluate them, and understand where each approach fits.

---
## 1. Simple Linear Regression (Salary vs. Experience)
Goal: Predict salary from years of experience.

**Concept**: Fit a line \( y = \beta_0 + \beta_1 x + \varepsilon \) minimizing squared error.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Sample dataset
data = {
	"YearsExperience": [1,2,3,4,5,6,7,8,9,10],
	"Salary": [40000,45000,50000,60000,65000,70000,75000,80000,85000,90000]
}
df = pd.DataFrame(data)

X = df[["YearsExperience"]]  # feature matrix (2D)
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("R²:", r2_score(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))

plt.scatter(X, y, color="steelblue", label="Actual")
plt.plot(X, y_pred, color="crimson", label="Predicted")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()
```

> Note: This synthetic dataset is nearly linear; real-world salary data usually requires more robust modeling and additional predictors.

---
## 2. Multiple Linear Regression (Boston Housing – Deprecated Example)
The classic Boston Housing dataset (median house value vs. socioeconomic & structural features) is **deprecated in scikit-learn** due to ethical concerns and should not be used in production or teaching without context. We'll illustrate the API, then recommend an alternative.

```python
import pandas as pd
from sklearn.datasets import load_boston  # Deprecated in recent versions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Warning: load_boston may be removed; wrap in try/except
try:
	boston = load_boston()
	X = pd.DataFrame(boston.data, columns=boston.feature_names)
	y = boston.target

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = LinearRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	print("Test MSE:", mean_squared_error(y_test, y_pred))
	print("Test R²:", r2_score(y_test, y_pred))
except Exception as e:
	print("Boston dataset not available (expected in newer sklearn). Use California Housing instead.")
	print("Error:", e)
```

### Recommended Replacement: California Housing (see section 5)

Key points in multiple regression:
- Check multicollinearity (VIF)
- Scale features if magnitudes differ significantly
- Consider regularization (Ridge/Lasso) when many correlated predictors

---
## 3. Polynomial Regression (Modeling Non-linear Relationships)
Polynomial regression augments features with polynomial terms then applies linear regression in the expanded feature space.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array([1,2,3,4,5,6]).reshape(-1, 1)
y = np.array([2, 6, 14, 28, 45, 66])  # Quadratic-like growth

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

print("R² (degree=2):", r2_score(y, y_pred))
print("Feature names:", poly.get_feature_names_out(["X"]))

plt.scatter(X, y, color="royalblue", label="Actual")
plt.plot(X, y_pred, color="darkorange", label="Polynomial Fit (deg 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression")
plt.legend()
plt.tight_layout()
plt.show()
```

> Caution: High-degree polynomials can overfit; use cross-validation and consider regularization or splines.

### Using a Pipeline (Good Practice)
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
	("poly", PolynomialFeatures(degree=3, include_bias=False)),
	("linreg", LinearRegression())
])
pipeline.fit(X, y)
print("R² (degree=3):", pipeline.score(X, y))
```

---
## 4. Logistic Regression (Classification Contrast – Iris Dataset)
Although named "regression", Logistic Regression is a **classification** algorithm modeling log-odds.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, :2]  # use two features for visualization simplicity
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.25, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=300, multi_class='auto')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

> For multi-class probability calibration or imbalanced classes, consider `solver='lbfgs'` (default), penalty adjustments, or class weights.

---
## 5. Real-World Regression: California Housing Dataset
Predict median house value in California districts. Each row aggregates census block-level stats (continuous target: `MedHouseValue`).

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Test MSE:", mean_squared_error(y_test, preds))
print("Test RMSE:", mean_squared_error(y_test, preds, squared=False))
print("Test R²:", r2_score(y_test, preds))
```

### Potential Enhancements
- Add feature scaling + regularization (Ridge/Lasso)
- Log-transform skewed features (e.g., population)
- Evaluate residual diagnostics (see earlier residuals note)
- Compare tree-based models (RandomForest, GradientBoosting)

---
## 6. Summary
| Scenario | Technique | Notes |
|----------|-----------|-------|
| Single numeric predictor | Simple Linear Regression | Interpret slope directly |
| Multiple correlated predictors | Multiple Linear Regression | Watch multicollinearity & consider regularization |
| Curved relationship | Polynomial Regression | Control degree to avoid overfit |
| Categorical target (multi-class) | Logistic Regression | Outputs class probabilities/log-odds |
| Real estate aggregated census data | California Housing Regression | Baseline; test advanced models |

---
### Key Takeaways
- Always separate training and test sets for realistic performance estimates.
- Use pipelines for transformations + models to avoid leakage.
- Prefer modern, ethically appropriate datasets (California over Boston).
- Polynomial features increase dimensionality quickly—apply regularization and validation.
- Logistic Regression belongs in classification despite its name—don’t use for continuous targets.

---
### Next Directions
- Add Ridge/Lasso/ElasticNet comparisons
- Introduce regularization paths & bias-variance trade-off
- Move to tree ensembles & boosting for non-linearities
- Implement cross-validation & hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

> Tip: Wrap repeated evaluation logic in utility functions (e.g., `evaluate_regression(model, X_train, X_test, y_train, y_test)`) to standardize metrics across experiments.
