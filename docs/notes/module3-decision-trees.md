---
title: "Module 3 – Decision Trees: Steps for Classification"
description: "Entropy-driven splits, Play Tennis walkthrough, and practical scikit-learn pipelines"
tags:
  - decision-trees
  - classification
  - entropy
  - information-gain
  - scikit-learn
  - module3
---

# Decision Trees: Steps for Classification

Decision Trees break decisions down into a hierarchy of feature-based questions. They remain a favorite for their interpretability, ability to mix numeric/categorical inputs, and support for both binary and multi-class targets.

## Step-by-step workflow
1. **Select the best feature** to split the current subset (Entropy/Information Gain, Gini, or Chi-square are typical choices).
2. **Partition the data** according to that feature’s values (for numerics, choose an optimal threshold).
3. **Recurse on each child subset** until the node is pure or further splits no longer improve the objective.
4. **Label leaf nodes** using the majority class or estimated class probabilities.

## Entropy and Information Gain

For a dataset $S$ with class proportions $p_i$ across $k$ classes:

$$
\text{Entropy}(S) = - \sum_{i=1}^{k} p_i \log_2 p_i
$$

- Entropy $= 0$ when a node is pure.
- Entropy is maximal when classes are evenly distributed.

Splitting on attribute $A$ yields subsets $S_v$ for each value $v$:

$$
\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \, \text{Entropy}(S_v)
$$

ID3 selects the attribute with the highest information gain at each step.

## Worked example: Play Tennis

| Outlook   | Temperature | Humidity | Wind   | Play |
|-----------|-------------|----------|--------|------|
| Sunny     | Hot         | High     | Weak   | No   |
| Sunny     | Hot         | High     | Strong | No   |
| Overcast  | Hot         | High     | Weak   | Yes  |
| Rain      | Mild        | High     | Weak   | Yes  |
| Rain      | Cool        | Normal   | Weak   | Yes  |
| Rain      | Cool        | Normal   | Strong | No   |
| Overcast  | Cool        | Normal   | Strong | Yes  |
| Sunny     | Mild        | High     | Weak   | No   |
| Sunny     | Cool        | Normal   | Weak   | Yes  |
| Rain      | Mild        | Normal   | Weak   | Yes  |
| Sunny     | Mild        | Normal   | Strong | Yes  |
| Overcast  | Mild        | High     | Strong | Yes  |
| Overcast  | Hot         | Normal   | Weak   | Yes  |
| Rain      | Mild        | High     | Strong | No   |

- Overall entropy: $\text{Entropy}(S) = 0.940$ (9 “Yes”, 5 “No”).
- Highest information gain comes from **Outlook**, so it becomes the root.
- Subsequent splits deliver the classic tree:

```
Outlook
 ├── Overcast → Play = Yes
 ├── Sunny
 │     ├── Humidity = High   → Play = No
 │     └── Humidity = Normal → Play = Yes
 └── Rain
       ├── Wind = Weak   → Play = Yes
       └── Wind = Strong → Play = No
```

That’s entropy-driven tree construction in action.

## Quickstart: Iris classification with scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Decision Tree
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
clf.fit(X_train, y_train)

# Accuracy on hold-out set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Visualize tree
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names
)
plt.show()
```

Accuracy captures generalisation on unseen data, while the tree visualisation reveals decision logic.

## Implementing trees across multiple datasets

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

configs = [
    {
        "loader": datasets.load_iris,
        "type": "classification",
        "name": "Iris",
        "estimator": DecisionTreeClassifier(
            criterion="entropy", max_depth=3, random_state=42
        ),
        "test_size": 0.3,
    },
    {
        "loader": datasets.load_wine,
        "type": "classification",
        "name": "Wine",
        "estimator": DecisionTreeClassifier(
            criterion="gini", max_depth=4, random_state=42
        ),
        "test_size": 0.3,
    },
    {
        "loader": datasets.load_breast_cancer,
        "type": "classification",
        "name": "Breast Cancer",
        "estimator": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, random_state=42
        ),
        "test_size": 0.25,
    },
]

for cfg in configs:
    data = cfg["loader"]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=cfg["test_size"],
        random_state=42,
        stratify=data.target
    )

    estimator = cfg["estimator"]
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{cfg['name']} accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# Regression example: California Housing
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data,
    housing.target,
    test_size=0.3,
    random_state=42
)

reg = DecisionTreeRegressor(max_depth=5, random_state=42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"California Housing MSE: {mse:.3f}")
```

- Classification demos span Iris, Wine, and Breast Cancer datasets.
- Regression demo swaps in `DecisionTreeRegressor` on California Housing.
- Tune `max_depth`, `min_samples_leaf`, or prune to rein in overfitting.

## ✅ Key takeaways
- Decision Trees are interpretable, handle mixed data types, and support multi-class outputs.
- Entropy/information gain and Gini both aim to produce purer child nodes.
- Trees often overfit—limit depth, prune, or ensemble (Random Forests, Gradient Boosting) for better generalisation.
- Visualising splits clarifies feature importance and rule paths.

## 📺 Watch next

<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/coOTEc-0OGw" title="Decision Tree Intuition" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">Decision Tree intuition and high-level walkthrough</div>
</div>

<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/xyDv3DLYjfM" title="Decision Tree Example" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">Worked example with entropy calculations</div>
</div>

<div class="video-embed" style="margin:1rem 0">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/vySR1fBSpRg" title="Decision Tree Overfitting & Pruning" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <div class="mdx-caption">Managing overfitting and pruning strategies</div>
</div>

## 📚 Further reading
- GeeksforGeeks – [Decision Tree in Machine Learning](https://www.geeksforgeeks.org/machine-learning/decision-tree/)
- Video playlist – [Decision Trees Deep Dive](https://www.youtube.com/watch?v=coOTEc-0OGw&list=PL4gu8xQu0_5K858LBik5BQfDVutvawEFU)
