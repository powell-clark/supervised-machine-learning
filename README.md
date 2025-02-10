# 🎓 Supervised Machine Learning

This repository teaches machine learning from first principles using Python. Starting with foundational mathematics (derivatives, exp/log, probability), each notebook builds complete understanding before exploring modern tools like scikit-learn and PyTorch. All mathematical concepts are derived step-by-step, making the content accessible to anyone with basic calculus knowledge. Part of a broader machine learning series, with companion repositories for unsupervised, reinforcement and other types of learning in development.

## 📚 Notebooks

🚀 Quick Start: Click any "Open in Colab" link below to run notebooks directly in your browser - no setup required!

### 1a_logistic_regression_theory.ipynb  
Theory & from-scratch implementation of logistic regression using the Wisconsin Breast Cancer dataset.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/1a_logistic_regression_theory.ipynb)
* [View Source](notebooks/1a_logistic_regression_theory.ipynb)

### 1b_logistic_regression_practical.ipynb
Production-grade PyTorch implementation with modern ML engineering practices.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/1b_logistic_regression_practical.ipynb)
* [View Source](notebooks/1b_logistic_regression_practical.ipynb)

### 2a_decision_trees_theory.ipynb
Deep dive into decision tree theory with a complete from-scratch implementation.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/2a_decision_trees_theory.ipynb)
* [View Source](notebooks/2a_decision_trees_theory.ipynb)

### 2b_decision_trees_practical.ipynb
Real-world application building a London housing market price predictor.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/2b_decision_trees_practical.ipynb)
* [View Source](notebooks/2b_decision_trees_practical.ipynb)

### 2c_decision_trees_ATLAS.ipynb
Automated Tree Learning Analysis System (ATLAS) for feature engineering and model comparison.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/2c_decision_trees_ATLAS.ipynb)
* [View Source](notebooks/2c_decision_trees_ATLAS.ipynb)


### 🧠 Neural Networks (Coming Soon)
Implementation of neural networks and deep learning fundamentals.

## 📊 Datasets

### [Wisconsin Breast Cancer (1995)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
* Binary classification task
* 569 samples × 30 features
* Medical diagnosis application

### [London Housing Prices (2020)](https://www.kaggle.com/datasets/arnavkulkarni/housing-prices-in-london)
* Regression problem
* 3,479 samples × 9 features
* Geographic feature encoding

## 💻 Local Setup
For those who prefer to run notebooks locally:
```bash
git clone https://github.com/powell-clark/supervised-machine-learning.git
cd supervised-machine-learning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
---
© 2025 Powell-Clark Limited