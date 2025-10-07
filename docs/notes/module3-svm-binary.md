---
title: "Module 3 – Support Vector Machines: Binary Classification"
description: "Maximum-margin hyperplanes, soft margins, kernels, and an RBF implementation for two-class problems"
tags:
  - svm
  - binary-classification
  - kernel-methods
  - margin
  - scikit-learn
  - module3
---

# Support Vector Machines for Binary Classification

Support Vector Machines (SVMs) search for the separating hyperplane that maximises the margin between two classes. Only a handful of points — the **support vectors** — determine the final decision boundary.

## Working principle
- **Input**: Feature vectors $X \in \mathbb{R}^n$ and labels $y \in \{-1, +1\}$.
- **Hyperplane**: $w^\top x + b = 0$ separates the two classes.
- **Prediction rule**: $\hat{y} = \operatorname{sign}(w^\top x + b)$.
- **Margin**: Twice the distance from the hyperplane to the closest support vectors; maximising it improves generalisation.

## Optimisation problem

### Hard-margin (perfect separability)
$
\min_{w, b} \; \frac{1}{2}\lVert w \rVert^2
$
subject to
$$
y_i (w^\top x_i + b) \ge 1, \quad \forall i
$$

### Soft-margin (allows violations)
Introduce slack variables $\xi_i \ge 0$ and penalty $C$:
$$
\min_{w, b, \xi} \; \frac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i
$$
subject to
$$
y_i (w^\top x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
$$
- Large $C$ → low bias, high variance (fits training data tightly).
- Small $C$ → high bias, low variance (wider margin, more tolerant to errors).

## Kernel trick
When classes are not linearly separable in the original space, map inputs into a higher-dimensional feature space using a kernel $K(x, z)$:
- **Linear**: $K(x, z) = x^\top z$
- **Polynomial**: $K(x, z) = (\gamma x^\top z + r)^d$
- **RBF / Gaussian**: $K(x, z) = \exp(-\gamma \lVert x - z \rVert^2)$
- **Sigmoid**: $K(x, z) = \tanh(\gamma x^\top z + r)$
The optimisation happens implicitly in the transformed space without ever computing the coordinates explicitly.

## scikit-learn example (RBF kernel)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Iris dataset – convert to binary problem (class 0 vs non-zero)
iris = datasets.load_iris()
X = iris.data[:, :2]  # two features for plotting convenience
y = (iris.target != 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = SVC(kernel="rbf", C=1.0, gamma=0.5)
clf.fit(X_train, y_train)

print("Training accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))


def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.title("SVM Decision Boundary (RBF kernel)")
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()


plot_decision_boundary(X_test, y_test, clf)
```

Results include training/test accuracy and a decision boundary visualisation generated from the RBF kernel.

## Tuning key `SVC` parameters

scikit-learn's `SVC` exposes knobs that balance margin width, boundary flexibility, and computational trade-offs:

- **`C` (regularisation)**: Controls tolerance for violations.
  - `SVC(C=0.1)` → wider margin, more tolerance to mistakes (higher bias).
  - `SVC(C=100)` → narrow margin, tries to classify everything correctly (higher variance).
- **`kernel`**: Shape of the separating surface (`'linear'`, `'poly'`, `'rbf'` (default), `'sigmoid'`).
- **`degree`**: Polynomial degree when `kernel='poly'` (e.g., `degree=3` for cubic).
- **`gamma`**: Influence radius for `'rbf'`, `'poly'`, `'sigmoid'` kernels.
  - Smaller `gamma` → smoother boundary, more generalisation.
  - Larger `gamma` → complex boundary, risk of overfitting.
- **`shrinking`**: Enable/disable shrinking heuristic (usually `True`).
- **`probability`**: Turn on probability estimates (slower due to internal cross-validation).
- **`class_weight`**: Rebias the margin for imbalanced datasets (e.g., `'balanced'` or `{0:1, 1:10}`).
- **`max_iter`**: Cap on optimisation iterations (`-1` means no limit).
- **`decision_function_shape`**: `'ovo'` (one-vs-one, default) or `'ovr'` (one-vs-rest) for multiclass.

```python
clf = SVC(
    C=10,
    kernel="rbf",
    gamma=0.5,
    class_weight="balanced",
    probability=True,
)
clf.fit(X_train, y_train)
print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))
```

## Extending SVMs to multi-class problems

Although SVM is inherently binary, scikit-learn lifts it to multi-class via:

- **One-vs-One (OvO)**: Train a classifier for every pair of classes (default in `SVC`). Predictions use majority voting across pairwise models.
- **One-vs-Rest (OvR)**: Train one classifier per class vs the rest; choose the class with the highest decision score.

### OvO / OvR in practice

```python
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Default OvO strategy
ovo_clf = SVC(kernel="rbf", C=1, gamma=0.5, decision_function_shape="ovo")
ovo_clf.fit(X_train, y_train)
print("OvO report:\n", classification_report(y_test, ovo_clf.predict(X_test)))

# Switch to OvR
ovr_clf = SVC(kernel="rbf", C=1, gamma=0.5, decision_function_shape="ovr")
ovr_clf.fit(X_train, y_train)
print("OvR report:\n", classification_report(y_test, ovr_clf.predict(X_test)))
```

Both strategies optimise the same SVM objective internally; OvO can be faster for small numbers of classes, while OvR is often easier to interpret when class imbalance is present.

## SVM implementations across popular datasets

To complement the theory, the following mini-walkthroughs show how `SVC` behaves on diverse datasets—binary, multi-class, non-linear, and high-dimensional image data. Each snippet reuses the familiar `train_test_split → fit → evaluate` workflow so you can mix and match kernels and parameters quickly.

### 1️⃣ Binary classification – Breast Cancer dataset

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train SVM with RBF kernel
clf = SVC(kernel="rbf", C=1, gamma="scale")
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

print("Breast Cancer Dataset - Binary Classification")
print(classification_report(y_test, y_pred))
```

✅ Notes:

- Binary classification (0 vs 1)
- `C` regulates margin width vs. misclassification penalties
- `gamma` controls how far the influence of each support vector reaches

### 2️⃣ Multi-class classification – Iris dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target  # 3 classes

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Multi-class SVM (default = One-vs-One)
clf = SVC(kernel="rbf", C=1, gamma="scale", decision_function_shape="ovo")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Iris Dataset - Multi-Class Classification")
print(classification_report(y_test, y_pred))
```

✅ Notes:

- OvO strategy is default in scikit-learn for multi-class SVM
- Switch to `decision_function_shape="ovr"` for one-vs-rest when classes are imbalanced

### 3️⃣ Non-linear classification – Moons dataset

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Create dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train SVM with RBF kernel
clf = SVC(kernel="rbf", C=1, gamma=2)
clf.fit(X_train, y_train)

# Evaluate accuracy
print("Training accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))


# Visualize decision boundary
def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.title("SVM with RBF Kernel - Moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary(X_test, y_test, clf)
```

✅ Notes:

- Non-linear dataset with intertwined classes
- RBF kernel captures curved boundaries
- `gamma` and `C` jointly control boundary flexibility

### 4️⃣ Polynomial kernel – Digits dataset

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Scale features (important for SVMs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# SVM with Polynomial kernel
clf = SVC(kernel="poly", degree=3, C=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Digits Dataset - Multi-Class Classification with Polynomial Kernel")
print(classification_report(y_test, y_pred))
```

✅ Notes:

- Polynomial kernel models complex high-dimensional boundaries
- Feature scaling keeps all pixels on comparable scales
- Tune `degree` to control polynomial complexity

## ✅ Key points
- SVM maximises the margin defined by support vectors, improving robustness to outliers.
- Soft-margin formulation (via $C$) balances misclassification tolerance and margin width.
- Kernel choice shapes the boundary: linear for separable data, RBF for curved boundaries, polynomial for high-dimensional structure.
- Binary SVM models extend naturally to multiclass through one-vs-one or one-vs-rest (`decision_function_shape`).
- Real datasets benefit from consistent preprocessing—especially feature scaling for RBF/poly kernels and careful tuning of `C` and `gamma`.

## 📺 Watch next

<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/5cWlZf0VzsM" title="Binary SVM decision boundaries" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">Binary SVM intuition and decision boundary visualisation</div>
</div>

<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/1vsmaEfbnoE" title="Soft margins and parameter C" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">Soft-margin SVM explained with practical tuning tips</div>
</div>

## 📚 Further reading
- scikit-learn documentation – [SVMs for classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification)
- Cortes & Vapnik (1995) – *Support-Vector Networks*
- Andrew Ng (CS229) – Lecture notes on SVMs and kernel methods
