# Supervised vs Unsupervised Learning (Based on Problem Definition)

## 1. Supervised Learning

**Problem Definition**  
- The problem is defined in terms of **input variables (features)** and **output variables (labels/targets)**.  
- The task is to **learn a mapping function** from inputs → outputs.  
- The goal is to **predict outcomes** for unseen data.  

**Examples of Problems**  
- **Classification** → Predict categories (e.g., spam/not spam, disease/no disease).  
- **Regression** → Predict continuous values (e.g., house price, stock value).  

!!! tip "Key Point"
    In supervised learning, we know the **"correct answers"** during training.  

---

## 2. Unsupervised Learning

**Problem Definition**  
- The problem is defined **only in terms of input variables (features)**.  
- No output/labels are given.  
- The task is to **discover hidden patterns, structures, or relationships** in the data.  

**Examples of Problems**  
- **Clustering** → Grouping similar items (e.g., customer segmentation, document clustering).  
- **Dimensionality Reduction** → Reducing features while preserving patterns (e.g., PCA).  
- **Association Rule Mining** → Finding relations (e.g., "people who buy bread often buy butter").  

!!! tip "Key Point"
    In unsupervised learning, the system learns **without knowing the correct answers**.  

---

## ✅ Comparison Table (Based on Problem Definition)

| Aspect            | Supervised Learning                   | Unsupervised Learning                  |
|-------------------|---------------------------------------|----------------------------------------|
| **Problem Setup** | Input + Output (labeled data)         | Only Input (unlabeled data)            |
| **Goal**          | Learn mapping from input → output     | Discover hidden patterns/structure      |
| **Training Data** | Requires labeled data                 | Works with unlabeled data              |
| **Problem Type**  | Prediction (classification, regression) | Exploration (clustering, association, dimensionality reduction) |
| **Examples**      | Predict house prices, spam detection, disease diagnosis | Customer segmentation, market basket analysis, PCA |
