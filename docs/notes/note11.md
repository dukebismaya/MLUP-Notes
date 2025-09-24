---
title: "Feature Reduction Using PCA"
description: "Principal Component Analysis: intuition, mathematics, workflow, Python implementation, and practical guidance"
tags: [pca, dimensionality-reduction, feature-engineering, unsupervised-learning]
---

# Feature Reduction Using Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a linear, unsupervised technique that projects data into a lower-dimensional space while retaining as much variance (information) as possible. It creates new orthogonal axes (principal components) ordered by the amount of variance they capture.

---
## 1. Why Feature Reduction?
High-dimensional data can cause:
- Redundant / correlated features (multicollinearity)
- Increased model variance (overfitting risk)
- Higher computational costs
- Harder visualization & interpretability

PCA addresses these by creating a smaller set of uncorrelated components still carrying most of the signal.

---
## 2. Core Intuition
Imagine fitting a line through a 2D point cloud so that when you project points onto that line, the spread (variance) is maximized. That line is the first principal component. The second component is the next axis (orthogonal to the first) capturing the next largest remaining variance, and so on.

PCA is fundamentally about variance maximization under orthogonality constraints.

---
## 3. Mathematical Foundations
Given a centered data matrix $X \in \mathbb{R}^{n \times d}$ (n samples, d features):

1. Center features: subtract mean of each column: $X_c = X - \mu$.
2. (Optional) Scale (standardize) features if they are on different units.
3. Compute covariance matrix: $\Sigma = \frac{1}{n-1} X_c^T X_c$.
4. Eigen-decompose: $\Sigma = Q \Lambda Q^T$ where:
   - Columns of $Q$ are eigenvectors (principal directions)
   - Diagonal of $\Lambda$ has eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$
5. Variance explained by component $k$: $\text{VE}_k = \frac{\lambda_k}{\sum_{j=1}^d \lambda_j}$
6. Project data onto first $K$ components: $Z = X_c Q_K$ (where $Q_K$ are the first $K$ eigenvectors)

### Relation to Singular Value Decomposition (SVD)
Instead of forming $\Sigma$, compute SVD: $X_c = U S V^T$.
- Right singular vectors (columns of $V$) are eigenvectors of $\Sigma$.
- Eigenvalues relate to singular values: $\lambda_k = \frac{S_k^2}{n-1}$.
SVD is numerically more stable for large or sparse matrices.

### Orthogonality & Uncorrelated Components
Because $Q^T Q = I$, the transformed features (principal components) are uncorrelated. This helps regression models sensitive to multicollinearity (e.g., linear regression) but at the cost of losing direct feature semantics.

---
## 4. Step-by-Step PCA Workflow
1. Collect feature matrix $X$ (shape: n samples × d features)
2. Optional: Handle missing values / outliers (extreme outliers inflate variance)
3. Decide whether to standardize (usually yes if units differ)
4. Center (and scale) the data
5. Compute covariance (or use SVD directly)
6. Sort eigenvalues/eigenvectors
7. Compute cumulative explained variance
8. Select number of components K (threshold, elbow, downstream performance)
9. Project data: $Z = X_c Q_K$
10. Use $Z$ in modeling / visualization

---
## 5. Choosing the Number of Components
Common heuristics:
- Cumulative variance threshold (e.g., retain 90–95%)
- Scree plot elbow (point where marginal gain drops)
- Cross-validation of downstream model performance
- Information criteria (rarely in practice for PCA directly)

No universal rule—depends on performance vs. interpretability trade-offs.

---
## 6. Practical Considerations
| Issue | Guidance |
|-------|----------|
| Scaling | Always scale if units differ; PCA is variance-sensitive. |
| Outliers | Can dominate components; consider robust scaling / winsorization. |
| Interpretability | Components are linear mixtures; inspect loadings. |
| Sparsity | PCA does not produce sparse features; consider Sparse PCA if needed. |
| Non-linearity | PCA only captures linear correlations; use Kernel PCA / t-SNE / UMAP for non-linear structure. |
| Leakage | Fit PCA only on training set; apply same transform to test set. |

---
## 7. Interpreting Component Loadings
Loadings = eigenvectors (or columns of the rotation matrix). Large absolute value => strong contribution of original feature to that component.

For standardized data, the squared loading approximates the proportion of variance of a feature captured by the component.

---
## 8. Python Example (Iris Dataset)
Below: Standardize, fit PCA, inspect explained variance, plot scree & 2D projection.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
iris = load_iris()
X = iris.data  # shape (150, 4)
y = iris.target
feature_names = iris.feature_names

# 2. Standardize (mean=0, var=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit PCA (all components first)
pca = PCA(n_components=None, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("Explained variance ratio per component:")
for i, (ev, cv) in enumerate(zip(explained, cumulative), start=1):
	print(f"PC{i}: {ev:.4f} (cumulative: {cv:.4f})")

# 4. Scree plot
plt.figure(figsize=(6,4))
plt.plot(range(1, len(explained)+1), explained, marker='o', label='Individual')
plt.plot(range(1, len(explained)+1), cumulative, marker='s', label='Cumulative')
plt.xticks(range(1, len(explained)+1))
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('PCA Scree Plot (Iris)')
plt.legend()
plt.tight_layout()
plt.show()

# 5. 2D projection (first two components)
pc_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
pc_df['target'] = y
plt.figure(figsize=(6,5))
sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue='target', palette='viridis', s=60)
plt.title('Iris Projected onto First Two Principal Components')
plt.tight_layout()
plt.show()

# 6. Component loadings (contribution of original features)
loadings = pd.DataFrame(pca.components_.T, index=feature_names,
						columns=[f'PC{i}' for i in range(1, len(feature_names)+1)])
print("\nComponent Loadings (first two PCs):")
print(loadings.iloc[:, :2])
```

---
## 9. When PCA Helps
- Visualization: 2D/3D scatter of high-dimensional data
- Noise reduction (drop low-variance components)
- Mitigating multicollinearity before linear models
- Preprocessing before clustering or regression
- Speedup for algorithms sensitive to high dimensionality

---
## 10. When PCA May Not Help
- Strong non-linear manifolds (consider Kernel PCA, t-SNE, UMAP)
- Features already low-dimensional and well-engineered
- Interpretability is critical (components obscure original meanings)
- Presence of many categorical features (PCA requires numeric, usually continuous)

---
## 11. Limitations & Assumptions
- Linear method: captures only linear covariance
- Maximizes variance, not class separation (unsupervised)
- Sensitive to scaling and outliers
- Components can be hard to interpret in regulated or explainability-critical domains

---
## 12. Summary
PCA re-expresses data through orthogonal axes sorted by variance. Properly applied (with scaling and careful component selection), it improves efficiency and can enhance downstream performance. Always validate whether the reduction preserves task-relevant information.

---
## 13. Next Topics (Suggestions)
- Linear Discriminant Analysis (LDA) – supervised dimensionality reduction
- Kernel PCA – non-linear extension
- t-SNE / UMAP – manifold visualization
- Feature Selection vs Feature Extraction – conceptual distinctions
- Sparse PCA – interpretability via sparsity constraints

---
### Quick Reference
| Concept | Formula / Idea |
|---------|----------------|
| Covariance Matrix | $\Sigma = \frac{1}{n-1} X_c^T X_c$ |
| Eigen Decomposition | $\Sigma = Q \Lambda Q^T$ |
| Variance Explained | $\text{VE}_k = \lambda_k / \sum_j \lambda_j$ |
| Projection | $Z = X_c Q_K$ |
| SVD Link | $X_c = U S V^T$, $\lambda_k = S_k^2/(n-1)$ |

---
> Tip: Always fit PCA on the training split, then transform validation/test sets with the fitted object to avoid data leakage.
