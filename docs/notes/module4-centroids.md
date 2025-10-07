---
title: "Module 4 – Centroids"
description: "Geometric centroid (center of mass), triangle median intersection, and relation to K‑Means"
tags:
  - centroid
  - geometry
  - kmeans
  - module4
---

# Centroid: geometric center

A centroid is the geometric center (mean position) of all points in a figure. For uniform density, it coincides with the center of mass and acts as the balance point.

## Key aspects
- Geometric center: average location of all points in the figure
- Center of mass (uniform density): same as balance point
- Always inside a triangle: the centroid of any triangle lies within the triangle

## Triangle centroid
The centroid is the intersection of the three medians. Each median connects a vertex to the midpoint of the opposite side, and the centroid divides each median in a 2:1 ratio (the longer segment is closer to the vertex).

Given triangle vertices $(x_1, y_1), (x_2, y_2), (x_3, y_3)$, the centroid $(\bar{x}, \bar{y})$ is

$$
\bar{x} = \frac{x_1 + x_2 + x_3}{3}, \qquad
\bar{y} = \frac{y_1 + y_2 + y_3}{3}.
$$

More generally, for points $\{x_i\}_{i=1}^n$ in $\mathbb{R}^d$,

$$
\mu = \frac{1}{n} \sum_{i=1}^n x_i.
$$

## Relation to K‑Means
In K‑Means clustering, each cluster is represented by its centroid $\mu_j$, the arithmetic mean of all points assigned to that cluster. The update step recomputes $\mu_j$ exactly like the centroid formulas above, but in higher dimensions and across many points.

### Tiny example (triangle centroid)
```python
import numpy as np

triangle = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 6.0]])
centroid = triangle.mean(axis=0)
print(centroid)  # [1. 2.]
```

> Tip: When features have very different scales, standardize before computing centroids (and before K‑Means) so no dimension dominates the mean.
