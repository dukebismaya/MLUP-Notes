---
title: "Environment Setup & Installation of Important ML Libraries"
description: "Step-by-step guide to setting up Python environments and installing core Machine Learning, Deep Learning, NLP, CV, RL, and deployment libraries"
tags:
  - setup
  - environment
  - python
  - installation
  - tooling
---

# Environment Setup & Installation of Important ML Libraries

This guide walks through installing Python, creating an isolated environment, and installing essential libraries for machine learning, deep learning, NLP, computer vision, reinforcement learning, and deployment.

---
## 1. Install Python
Download from: https://www.python.org (use version 3.8+; 3.10–3.12 commonly supported).  
Or install **Anaconda / Miniconda** (recommended for ML—many packages pre-built, easier dependency handling).

Check version:
```bash
python --version
```

---
## 2. Set Up a Virtual Environment
Creating an isolated environment prevents dependency conflicts across projects.

### Using `venv` (built-in)
```bash
python -m venv ml_env
# Activate (Windows PowerShell)
ml_env\Scripts\Activate.ps1
# Or Windows cmd
ml_env\Scripts\activate.bat
# Or macOS / Linux
source ml_env/bin/activate
```

### Using `conda`
```bash
conda create --name ml_env python=3.9
conda activate ml_env
```
> Choose conda if you need complex native dependencies (e.g., some CV or GPU builds) without manual compilation.

---
## 3. Install Essential Libraries
Install in logical groups so failures are easier to diagnose.

### ✅ Core Libraries
```bash
pip install numpy pandas scipy
```

### ✅ Machine Learning (Classical & Boosting)
```bash
pip install scikit-learn xgboost lightgbm catboost
```

### ✅ Deep Learning
```bash
pip install tensorflow keras torch torchvision
```
> If you have a CUDA-capable GPU, consult PyTorch / TensorFlow install pages for GPU-specific wheels.

### ✅ Data Visualization
```bash
pip install matplotlib seaborn plotly yellowbrick
```

### ✅ Natural Language Processing
```bash
pip install nltk spacy transformers gensim
```
Download a spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### ✅ Computer Vision
```bash
pip install opencv-python scikit-image
```

### ✅ Reinforcement Learning
```bash
pip install gym stable-baselines3
```
> For Atari or MuJoCo environments additional extras may be required (e.g., `pip install gym[atari]`).

### ✅ Deployment & Utilities
```bash
pip install flask fastapi uvicorn mlflow onnx
```
(Optional) ONNX runtime for inference:
```bash
pip install onnxruntime
```

---
## 4. Jupyter Notebook / IDE Setup
Install Jupyter Lab / Notebook:
```bash
pip install jupyterlab notebook
```
Run Notebook:
```bash
jupyter notebook
```

### IDE Options
- **VS Code** (Python + Jupyter extensions)
- **PyCharm** (robust project tooling)
- **Google Colab** (no local install, free GPU tier)
- **Kaggle Notebooks** (datasets + accelerators)

---
## 5. Verify Installation
Open Python and try imports:
```python
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
import matplotlib.pyplot as plt

print("All libraries loaded successfully!")
```
If no errors appear, your environment is ready.

---
## ✅ Summary
| Step | Action |
|------|--------|
| 1 | Install Python / Anaconda |
| 2 | Create virtual or conda environment |
| 3 | Install core + ML + DL + NLP + CV + RL + deployment libraries |
| 4 | Set up Jupyter / IDE |
| 5 | Test imports |

---
### Optional Next Steps
| Goal | Command / Tool |
|------|----------------|
| Freeze dependencies | `pip freeze > requirements.txt` |
| Upgrade a package | `pip install -U package_name` |
| Check outdated | `pip list --outdated` |
| Manage experiments | `mlflow ui` |
| Serve FastAPI app | `uvicorn app:app --reload` |

> Keep environments lean. Only install what the specific project requires; create new environments for materially different dependency stacks (e.g., GPU deep learning vs lightweight analytics).