---
title: "Module 3 – K‑Nearest Neighbours: Mathematical Formulation"
description: "Formal KNN setup, distance, neighbour selection, voting, and probability view (binary classification)"
tags:
  - knn
  - classification
  - mathematics
  - module3
---

# K‑Nearest Neighbours (KNN): Mathematical Formulation

We consider binary classification $y \in \{0,1\}$; extensions to multi‑class follow similarly by one‑vs‑rest or majority over labels $\{0,\dots,C-1\}$.

## 1) Problem setup
Training dataset $\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$ with
- $x_i \in \mathbb{R}^d$ (feature vectors)
- $y_i \in \{0,1\}$ (labels)

Given a new point $x \in \mathbb{R}^d$, predict $\hat{y}(x)$.

## 2) Distance function
KNN measures similarity by a distance $d(\cdot,\cdot)$, commonly Euclidean:

$$
 d(x, x_i) = \sqrt{\sum_{j=1}^d (x_j - x_{ij})^2}.
$$

More generally, $d : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\ge 0}$ (e.g., Euclidean, Manhattan, Minkowski, cosine distance for normalized vectors).

## 3) Selection of nearest neighbours
Let $N_k(x)$ denote the index set of the $k$ nearest neighbours of $x$ in $\mathcal{D}$:

$$
 N_k(x) = \operatorname*{arg\,min}_{S \subseteq \{1,\dots,n\},\; |S|=k} \sum_{i \in S} d(x, x_i).
$$

In practice, this means: sort all $i$ by $d(x, x_i)$ and take the first $k$ indices.

## 4) Majority voting rule
The predicted class is the majority among the $k$ neighbours:

$$
 \hat{y}(x) = \begin{cases}
 1, & \text{if } \sum_{i \in N_k(x)} y_i \ge \tfrac{k}{2} \\
 0, & \text{otherwise.}
 \end{cases}
$$

Equivalently using the indicator $\mathbf{1}[\cdot]$:

$$
 \hat{y}(x) = \mathbf{1}\!\left[ \frac{1}{k} \sum_{i \in N_k(x)} y_i \ge 0.5 \right].
$$

## 5) Probability view (optional)
Define a non‑parametric posterior estimate:

$$
 \hat{P}(y=1\mid x) = \frac{1}{k} \sum_{i \in N_k(x)} y_i, \qquad
 \hat{P}(y=0\mid x) = 1 - \hat{P}(y=1\mid x).
$$

Then the classifier is the MAP decision:

$$
 \hat{y}(x) = \operatorname*{arg\,max}_{c \in \{0,1\}} \hat{P}(y=c\mid x).
$$

---

### Notes
- Choose $k$ via cross‑validation or heuristics (odd $k$ avoids ties in binary case).
- Scale features (e.g., StandardScaler) so distances are meaningful across dimensions.
- For large $n$, use approximate nearest neighbours (e.g., KD‑trees, Ball trees, HNSW).
- For multi‑class, vote by counts per class label.

---

## Appendix: Quick scikit‑learn example
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Toy binary dataset
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)

# Pipeline: scale + KNN
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
model.fit(X, y)

# Decision boundary
xx, yy = np.meshgrid(
  np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
  np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, Z, levels=2, cmap='Pastel2', alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y, cmap='Set1', s=18, edgecolor='k', alpha=0.8)
plt.title('KNN (k=7) decision regions')
plt.show()
```
