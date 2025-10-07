---
title: "Module 3 – Support Vector Machines: Classification"
description: "Maximum-margin intuition, mathematical formulation, and scikit-learn implementation with decision boundaries"
tags:
  - svm
  - classification
  - kernel-methods
  - margin
  - scikit-learn
  - module3
---

# Classification using Support Vector Machines (SVM)

Support Vector Machines hunt for the decision boundary with the widest possible margin while respecting the labels of the training data. The data points that touch this margin are the **support vectors** that ultimately define the classifier.

## Key idea
- Find the hyperplane that maximizes the distance to the closest points of each class.
- Margin width provides robustness; wider margins typically generalize better.
- Kernel functions project inputs into higher-dimensional feature spaces to handle non-linear separability.

## Mathematical formulation
Given labelled data $(x_i, y_i)$ with $x_i \in \mathbb{R}^n$ and $y_i \in \{-1, +1\}$, SVM searches for a hyperplane $w^\top x + b = 0$.

### Hard-margin optimisation (linearly separable case)

$$
\min_{w, b} \; \frac{1}{2} \lVert w \rVert^2
$$

subject to

$$
y_i (w^\top x_i + b) \ge 1, \quad \forall i
$$

The constraints ensure all samples fall on the correct side of the margin while the objective widens the margin.

### Soft margin & kernels
When data is not perfectly separable, introduce slack variables $\xi_i$ and a penalty parameter $C$:

$$
\min_{w, b, \xi} \; \frac{1}{2} \lVert w \rVert^2 + C \sum_i \xi_i, \quad
\text{s.t. } y_i (w^\top x_i + b) \ge 1 - \xi_i, \; \xi_i \ge 0
$$

Kernel functions implicitly map inputs into higher-dimensional spaces:
- Linear: $K(x, z) = x^\top z$
- Polynomial: $K(x, z) = (\gamma x^\top z + r)^d$
- Radial Basis Function (RBF): $K(x, z) = \exp(-\gamma \lVert x - z \rVert^2)$
- Sigmoid: $K(x, z) = \tanh(\gamma x^\top z + r)$

## scikit-learn implementation (binary Iris subset)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset (Iris dataset – first two features for visualisation)
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Convert to binary classification (class 0 vs class 1)
mask = y != 2
X, y = X[mask], y[mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Linear SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


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
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Paired")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="Paired")
    plt.title("SVM Decision Boundary (linear kernel)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary(X_test, y_test, model)
```

- Confusion matrix and classification report expose precision/recall performance.
- Visual decision boundary illustrates the maximum-margin separator.

## Beyond the basics
- **Multiclass**: scikit-learn automatically applies one-vs-one; one-vs-rest is also available.
- **Non-linear kernels**: swap `kernel` to `"rbf"`, `"poly"`, or `"sigmoid"` for curved boundaries.
- **Regularisation (`C`)**: smaller $C$ widens the margin (more bias), larger $C$ fits the training data tighter (less bias, more variance).
- **Feature scaling**: standardise features so that the margin reflects meaningful distances.

## ✅ Key takeaways
- SVM maximises margin, leaning on support vectors only.
- Kernel trick enables non-linear decision boundaries without explicit feature engineering.
- Regularisation balances margin width and misclassification tolerance.
- Works well on medium-sized datasets with clear margins; consider linear models or SGD for massive datasets.

## 📺 Watch next

<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/efR1C6CvhmE" title="SVM Intuition" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">SVM intuition and maximum-margin geometry</div>
</div>

<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/1NxnPkZM9bc" title="Kernel trick explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">Kernel tricks and non-linear decision boundaries</div>
</div>

## 📚 Further reading
- scikit-learn: [Support Vector Machines user guide](https://scikit-learn.org/stable/modules/svm.html)
- Stanford CS229 notes: [SVMs and kernels](https://cs229.stanford.edu/notes2023fall/cs229-notes3.pdf)
- Bishop, C. M. *Pattern Recognition and Machine Learning* – Chapter 7
