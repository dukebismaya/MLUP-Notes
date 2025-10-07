---
title: "Module 3 – Iris Classification (Note 1)"
description: "Dataset overview, classification framing, and formal problem statement for the classic Iris dataset"
tags:
  - iris
  - classification
  - multiclass
  - module3
---

# Iris Dataset and Problem Framing

## 1) What is the Iris Dataset?
The Iris dataset (Fisher, 1936) contains 150 flower samples from three species:
- Iris‑setosa
- Iris‑versicolor
- Iris‑virginica

Each sample has 4 numeric features (independent variables):
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The target variable is the species label.

## 2) Classification Problem Definition
A classification task assigns inputs to one of several predefined categories.

Inputs (features):
- Sepal length, sepal width, petal length, petal width

Output (target class):
- 0 → Iris‑setosa
- 1 → Iris‑versicolor
- 2 → Iris‑virginica

Goal: Given sepal and petal measurements, predict the flower species.

## 3) Why it’s a classification problem
- The target is categorical (species)
- We predict class labels, not continuous values
- It’s multi‑class (3 classes)

## 4) Formal problem statement
“Given measurements of iris flower sepals and petals, build a machine learning model to classify the flower into one of three species: Setosa, Versicolor, or Virginica.”
