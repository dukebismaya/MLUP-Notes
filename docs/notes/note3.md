# Understanding the Problem and Its Possible Solutions Using IRIS and Various Datasets

## Step 1: Understanding the Problem
When working with a dataset (like IRIS, Titanic, MNIST, etc.), the first step is to define:

- **What is given?** → Input features  
- **What do we want to find?** → Output / Patterns  
- **Is the output labeled?**  

!!! tip "Learning Type"
    - If **YES** → Supervised Learning  
    - If **NO** → Unsupervised Learning  

---

## Step 2: IRIS Dataset Example
The IRIS dataset has:

- **Input (features):** Sepal length, Sepal width, Petal length, Petal width  
- **Output (label):** Species (Setosa, Versicolor, Virginica)  

### Possible Problems & Solutions

- **Classification (Supervised)**  
  - **Problem:** Predict flower species from features  
  - **Solution:** Logistic Regression, SVM, Decision Tree, KNN  

- **Clustering (Unsupervised)**  
  - **Problem:** Group flowers without knowing species labels  
  - **Solution:** K-Means, Hierarchical Clustering  

- **Dimensionality Reduction (Unsupervised)**  
  - **Problem:** Visualize high-dimensional flower data in 2D  
  - **Solution:** PCA (Principal Component Analysis)  

---

## Step 3: Other Dataset Examples

### 1. Titanic Dataset (Passenger Survival)
- **Features:** Age, Sex, Ticket class, Fare, etc.  
- **Label:** Survival (Yes/No)  
- **Problem:** Predict survival  
- **Solution:** Classification → Logistic Regression, Decision Tree  

### 2. MNIST Dataset (Handwritten Digits)
- **Features:** Pixel values of images (28×28)  
- **Label:** Digit (0–9)  
- **Problem:** Digit recognition  
- **Solution:** Classification (Supervised, Deep Learning CNNs)  

### 3. Supermarket Sales Dataset
- **Features:** Product, Price, Quantity, Date, etc.  
- **Label:** Not always given  

**Problems & Solutions:**  
- Predict sales revenue → Regression (Supervised)  
- Customer segmentation → Clustering (Unsupervised)  
- Market basket analysis → Association Rule Mining (Unsupervised)  

### 4. COVID-19 Dataset
- **Features:** Cases, Deaths, Vaccinations, Date, Country  

**Problems & Solutions:**  
- Forecast future cases → Time Series Prediction (Supervised)  
- Cluster countries by impact → Clustering (Unsupervised)  
- Identify main factors influencing spread → Feature Selection + Regression  

---

## Step 4: General Rule for Problem → Solution
- **If labels (target variable) are present** → Supervised Learning (Classification/Regression)  
- **If no labels are present** → Unsupervised Learning (Clustering/Association)  
- **If the task is decision-making via rewards** → Reinforcement Learning  

---

## ✅ In Summary
- **IRIS** → Classification (Supervised), Clustering (Unsupervised), PCA (Dimensionality Reduction)  
- **Titanic** → Classification  
- **MNIST** → Classification (Deep Learning)  
- **Supermarket Sales** → Regression, Clustering, Association Rules  
- **COVID-19** → Time Series Forecasting, Clustering  
