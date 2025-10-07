---
title: "Module 3 – Model Evaluation Metrics with scikit-learn"
description: "Hands-on walkthrough of core classification, regression, and imbalanced-learning metrics using scikit-learn"
tags:
  - evaluation
  - metrics
  - scikit-learn
  - module3
---

# Model Evaluation Metrics with scikit-learn

Evaluating a model is as important as training it. scikit-learn ships with metrics that make it easy to quantify performance for classification, regression, and imbalanced datasets. This note collects the most common ones alongside runnable examples.

## Classification Metrics

| Metric | Description |
| --- | --- |
| Accuracy | Fraction of correctly predicted samples |
| Precision | $\text{TP} / (\text{TP} + \text{FP})$ — how many predicted positives were correct |
| Recall (Sensitivity) | $\text{TP} / (\text{TP} + \text{FN})$ — how many actual positives were found |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the Receiver Operating Characteristic curve |
| Confusion Matrix | Table showing counts for TP, TN, FP, and FN |

### Example 1 — Binary Classification (Breast Cancer Dataset)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train SVM (probability=True enables predict_proba for ROC-AUC)
clf = SVC(kernel="rbf", C=1, gamma="scale", probability=True)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

<small>Tip: If you only have decision scores, replace `predict_proba` with `decision_function` inside `roc_auc_score`.</small>

### Example 2 — Multi-class Classification (Iris Dataset)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train SVM with one-vs-one strategy
clf = SVC(kernel="rbf", C=1, gamma="scale", decision_function_shape="ovo")
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
```

`classification_report` includes precision, recall, and F1-score for each class plus macro/micro averages.

## Regression Metrics

| Metric | Description |
| --- | --- |
| Mean Absolute Error (MAE) | Average of absolute differences between predictions and targets |
| Mean Squared Error (MSE) | Average of squared differences |
| Root Mean Squared Error (RMSE) | Square root of MSE; interpretable in the target units |
| $R^2$ Score | Proportion of variance explained by the model |

### Example 3 — Regression (Boston Housing Dataset)

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train SVM regressor
svr = SVR(kernel="rbf", C=100, gamma=0.1)
svr.fit(X_train, y_train)

# Predictions
y_pred = svr.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
```

!!! note
    `load_boston` is deprecated in newer versions of scikit-learn. Switch to `fetch_california_housing` or a custom dataset for production work.

## Additional Metrics for Imbalanced Datasets

Imbalanced datasets emphasise minority classes; standard accuracy can be misleading. Use metrics that penalise false negatives/positives appropriately.

```python
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)

# precision, recall, thresholds relate back to the positive class
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
print("Average Precision:", average_precision_score(y_test, y_proba))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
```

Balanced accuracy computes the average recall for each class, making it robust when class frequencies differ. Plotting `precision_recall_curve` helps visualise trade-offs at different decision thresholds.

## Summary

- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Regression:** MAE, MSE, RMSE, $R^2$
- **Imbalanced Learning:** Prefer Balanced Accuracy, Precision-Recall curves, and Average Precision

These evaluation utilities are consistent across scikit-learn estimators, making it straightforward to swap datasets or algorithms while preserving diagnostics.
