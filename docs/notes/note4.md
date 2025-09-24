# Machine Learning Notes with Python Examples

---

## 1. Supervised vs Unsupervised Learning

### Supervised Learning Example (Classification)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load IRIS dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

# Python Libraries Suitable for Machine Learning

In Python, Machine Learning (ML) is supported by a powerful ecosystem of libraries spanning data handling, model building, visualization, experimentation, and deployment.

Below is a categorized, skimmable overview you can reference while choosing tools for a project.

---
## 🔹 1. Core Libraries for Data Handling
| Library | Purpose | Notes |
|---------|---------|-------|
| **NumPy** | Numerical computing (n‑d arrays, linear algebra, vectorization) | Foundation for most libraries |
| **Pandas** | Data manipulation (DataFrames, IO: CSV/Parquet/Excel) | Feature engineering & preprocessing |
| **SciPy** | Scientific computing (stats, optimization, signal processing) | Builds on NumPy; specialized routines |

---
## 🔹 2. Machine Learning Libraries (Classical & Gradient Boosting)
| Library | Focus | Strength |
|---------|-------|----------|
| **Scikit-learn** | Classification, regression, clustering, preprocessing, model selection | Standard, consistent API |
| **XGBoost** | Gradient boosting (trees) | Strong for tabular; regularization controls |
| **LightGBM** | Fast gradient boosting | Large datasets, leaf-wise growth |
| **CatBoost** | Gradient boosting with native categorical support | Minimal encoding needed |

---
## 🔹 3. Deep Learning Libraries
| Library | Focus | Notes |
|---------|-------|-------|
| **TensorFlow** | Scalable deep learning & deployment | Ecosystem: TF Serving, TF Lite |
| **Keras** | High-level API (now core TensorFlow) | Fast prototyping |
| **PyTorch** | Dynamic computation graphs; research-friendly | Strong community & ecosystem (TorchVision, TorchText) |
| **MXNet** | Scalable distributed DL | Multi-language bindings |

> Tip: For research → PyTorch; for production edge / mobile → TensorFlow (TF Lite); for quick educational prototypes → Keras.

---
## 🔹 4. Visualization & Model Insight
| Library | Focus | When to Use |
|---------|-------|-------------|
| **Matplotlib** | Base plotting primitives | Low-level control |
| **Seaborn** | Statistical & aesthetic wrappers over Matplotlib | Fast EDA |
| **Plotly** | Interactive charts & dashboards | Exploratory analytics / web |
| **Yellowbrick** | Model diagnostic visualizers | Confusion matrix, ROC, residuals |

---
## 🔹 5. Natural Language Processing (NLP)
| Library | Focus | Notes |
|---------|-------|-------|
| **NLTK** | Classical NLP (tokenization, stemming, POS) | Education & traditional pipelines |
| **spaCy** | Industrial NLP (NER, parsing, vectors) | Efficient, production-friendly |
| **Transformers (Hugging Face)** | Pre-trained large models (BERT, GPT, T5) | Transfer learning powerhouse |
| **Gensim** | Topic modeling & embeddings (Word2Vec, Doc2Vec, LSI) | Lightweight semantic modeling |

---
## 🔹 6. Computer Vision (CV)
| Library | Focus | Notes |
|---------|-------|-------|
| **OpenCV** | Image processing & classical CV | Pre/post-processing, feature ops |
| **scikit-image** | Image filtering, transforms, feature extraction | Complement to OpenCV |
| **Detectron2** | Object detection & segmentation | PyTorch; state-of-the-art reference |
| **MMCV / MMDetection** | OpenMMLab modular vision frameworks | Flexible experiment structure |

---
## 🔹 7. Reinforcement Learning
| Library | Focus | Notes |
|---------|-------|-------|
| **Gym (OpenAI Gym / Gymnasium)** | Standardized RL environments | Benchmarking tasks |
| **Stable-Baselines3** | Ready-to-use RL algorithms (PPO, DQN, A2C, etc.) | Quick baseline training |
| **RLlib** | Scalable RL (Ray ecosystem) | Distributed training |

---
## 🔹 8. Model Deployment & Utilities
| Library / Tool | Focus | Notes |
|----------------|-------|-------|
| **ONNX** | Model exchange format | Interoperability across runtimes |
| **MLflow** | Experiment tracking, model registry, deployment | Reproducibility & lifecycle |
| **TensorFlow Serving** | Serving TensorFlow models | High-performance inference |
| **FastAPI / Flask** | Serving models via REST APIs | Lightweight microservices |

> Deployment pathways often combine: Train (PyTorch) → Export (ONNX) → Serve (ONNX Runtime / FastAPI) or Train (TensorFlow) → Serve (TF Serving / Vertex / SageMaker).

---
## ✅ Summary Cheat Sheet
| Category | Primary Picks |
|----------|---------------|
| Data Handling | NumPy, Pandas, SciPy |
| ML (Classical + Boosting) | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Deep Learning | PyTorch, TensorFlow, Keras |
| Visualization | Matplotlib, Seaborn, Plotly, Yellowbrick |
| NLP | spaCy, Transformers, NLTK, Gensim |
| Computer Vision | OpenCV, scikit-image, Detectron2 |
| RL | Gym, Stable-Baselines3, RLlib |
| Deployment | MLflow, FastAPI, ONNX |

---
### Selection Guidance
| Need | Likely Choice |
|------|---------------|
| Tabular baseline | Scikit-learn |
| Tabular SOTA model | LightGBM / CatBoost |
| Prototype deep model quickly | Keras |
| Research flexibility | PyTorch |
| Production pipeline w/ tracking | MLflow + FastAPI |
| Classic statistical transforms | SciPy |
| Explain model predictions | Yellowbrick (viz) / SHAP (not listed above) |
| Transfer learning NLP | Transformers |

---
**Next Expansion Ideas:** Add time-series specific libs (Prophet, statsmodels, darts), MLOps orchestration (Prefect, Airflow, Dagster), and model explainability (SHAP, LIME, Captum) in a future revision.
