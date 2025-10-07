---
title: "Module 3 – K-Nearest Neighbours: scikit-learn Implementation"
description: "Practical walkthrough of training and evaluating a KNN classifier on the Iris dataset using scikit-learn"
tags:
  - knn
  - scikit-learn
  - iris
  - classification
  - module3
---

# K-Nearest Neighbours (KNN): scikit-learn Implementation

The classic Iris dataset is a gentle starting point for experimenting with KNN. Below is a minimal but production-ready skeleton that scales features, trains a $k=5$ classifier, and reports model diagnostics using scikit-learn.

```python
# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling (important for distance-based algorithms like KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## 🔑 Key Points
- Feature scaling is critical because KNN relies on distance metrics such as Euclidean distance.
- `n_neighbors=5` is a sensible default; tune it with cross-validation to balance bias and variance.
- `load_iris()` ships with scikit-learn, letting you reproduce the workflow quickly for demos or unit tests.
