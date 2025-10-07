---
title: "Module 4 – Clustering Fundamentals"
description: "Definition, algorithms, applications, and K-Means in practice (with code)"
tags:
  - clustering
  - unsupervised-learning
  - kmeans
  - module4
---

# Clustering: Definition and Applications

Clustering is an unsupervised learning technique that groups similar data points into clusters so that:

- Points in the same cluster are more similar to each other (high intra‑cluster similarity)
- Points in different clusters are less similar (low inter‑cluster similarity)

Similarity is typically measured with a distance metric, such as Euclidean, Manhattan, or cosine distance.

## When to Use Clustering
- No labels are available (unlike classification/regression)
- You want to discover hidden structure or natural groupings
- You need to compress, segment, or summarize data

## Common Algorithms
- K‑Means: partitions data into k clusters around centroids
- Hierarchical clustering: builds a dendrogram (agglomerative/divisive)
- DBSCAN: density-based; can find arbitrary shapes and label noise
- Gaussian Mixture Models (GMM): probabilistic mixture of Gaussians

## Real‑world Applications
- Customer segmentation (marketing, product)
- Image segmentation and color quantization
- Anomaly/outlier detection
- Document/topic clustering (NLP)
- Recommender systems (user/product grouping)
- Healthcare & biology (patient, gene expression grouping)
- Geographical data analysis (regions by climate/socioeconomic factors)

---

# K‑Means Clustering

K‑Means partitions the dataset into k clusters by minimizing within‑cluster variance (sum of squared distances to the cluster centroid).

## Algorithm (Lloyd's algorithm)
1. Initialize: choose k initial centroids (often k‑means++).
2. Assignment step: assign each point to the nearest centroid.
3. Update step: recompute each centroid as the mean of the assigned points.
4. Repeat steps 2–3 until convergence (centroids stop moving or max iterations reached).

> Objective function: minimize $\sum_{j=1}^k \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert_2^2$.

## Choosing k (number of clusters)
- Elbow method: plot WCSS (inertia) vs k; look for the elbow.
- Silhouette score: higher is better; measures cohesion vs separation.
- Gap statistic: compare to a null reference distribution.
- Domain knowledge: interpretability and business constraints often drive k.

> See also: [Centroids](module4-centroids.md) for geometric intuition and formulas used by K‑Means.

### Why k matters
- Too small: distinct groups get merged (underfitting).
- Too large: natural groups get split (overfitting).

---

# Examples

## 1) Basic K‑Means on Iris (numeric)
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
labels = kmeans.fit_predict(X)

# Prepare for plotting
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = labels

plt.figure(figsize=(8,5))
sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], hue='Cluster', data=df, palette='Set1')
plt.title('K-Means Clusters on Iris (first two features)')
plt.show()
```
Notes:
- Iris has 3 species; k=3 is a reasonable demonstration choice.
- `fit_predict` trains and assigns labels in one call.

## 2) Mixed numeric + categorical
```python
import pandas as pd
from sklearn.cluster import KMeans

# Example dataset
data = pd.DataFrame({
    'Age':[25, 45, 35, 50, 23],
    'Income':[50000, 80000, 60000, 100000, 45000],
    'Gender':['Male','Female','Female','Male','Female']
})

# One-hot encode categoricals
encoded = pd.get_dummies(data, columns=['Gender'], drop_first=True)

kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
data['Cluster'] = kmeans.fit_predict(encoded)
print(data)
```
Notes:
- K‑Means requires numeric features; one‑hot encode before clustering.

## 3) Image color quantization (segmentation)
```python
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
labels = kmeans.fit_predict(pixels)
centers = kmeans.cluster_centers_.astype(np.uint8)
segmented = centers[labels].reshape(image.shape)

plt.imshow(segmented)
plt.axis('off')
plt.show()
```
Notes:
- Reduces the palette to k colors; useful for compression/segmentation.

---

# Choosing k with the Elbow method
<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/wW1tgWtkj4I" title="Elbow Method for KMeans" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">Video: Elbow method intuition and steps</div>
  
</div>
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)  # within-cluster sum of squares

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (inertia)')
plt.title('Elbow Method')
plt.show()
```
Tip:
- Look for the “elbow” where adding clusters yields diminishing returns.

### References
- K‑Means introduction (GeeksforGeeks): https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/
- Elbow method (GeeksforGeeks): https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/

---

# Practical notes
- Standardize features when scales differ significantly (use `StandardScaler`).
- K‑Means assumes roughly spherical, similar‑sized clusters; consider DBSCAN/GMM otherwise.
- Initialize with `k-means++` (default in scikit‑learn) for better convergence.
- Run with several initializations (`n_init`) to avoid poor local minima.
- Inspect inertia, silhouette score, and cluster profiles to choose k.

