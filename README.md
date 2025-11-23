# 🎓 Supervised Machine Learning from First Principles

This repository teaches supervised learning through rigorous mathematical derivation and from-scratch implementation.

Each lesson follows the pattern: derive the mathematics, implement from scratch in NumPy, then build production systems with modern libraries (Scikit-learn, PyTorch). You'll understand why algorithms work, not just how to use them.

**Curriculum:**
- **Lesson 0:** Linear Regression (Normal Equation, Gradient Descent)
- **Lesson 1:** Logistic Regression (Binary Classification, BCE Loss)
- **Lesson 2:** Decision Trees (Entropy, Information Gain, Ensembles)
- **Lesson 3:** Neural Networks (Backpropagation, Optimization)

Requires calculus (derivatives), linear algebra (matrices, dot products), and probability. Suitable for undergraduate ML courses at MIT, Stanford, Caltech.

## 📚 Notebooks

🚀 Quick Start:  Run notebooks directly in your browser - no setup required!
1. Click any "Open in Colab" button below
2. In Colab: Click "Connect" (top-right)
3. Click "Runtime" > "Run all" (top menu)

### Foundation
**Lesson 0: Linear Regression** - The foundation of machine learning

#### 0a_linear_regression_theory.ipynb
Mathematical derivation of Normal Equation and Gradient Descent with NumPy implementation.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/0a_linear_regression_theory.ipynb)
* [View Source](notebooks/0a_linear_regression_theory.ipynb)

---

### Core Algorithms
**Lesson 1: Logistic Regression** - Binary classification from first principles

#### 1a_logistic_regression_theory.ipynb
Theory & from-scratch implementation using the Wisconsin Breast Cancer dataset.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/1a_logistic_regression_theory.ipynb)
* [View Source](notebooks/1a_logistic_regression_theory.ipynb)

#### 1b_logistic_regression_practical.ipynb
Production-grade PyTorch implementation with modern ML engineering practices.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/1b_logistic_regression_practical.ipynb)
* [View Source](notebooks/1b_logistic_regression_practical.ipynb)

**Lesson 2: Decision Trees & Ensembles** - From single trees to Random Forests and XGBoost

#### 2a_decision_trees_theory.ipynb
Deep dive into decision tree theory with complete from-scratch implementation.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/2a_decision_trees_theory.ipynb)
* [View Source](notebooks/2a_decision_trees_theory.ipynb)

#### 2b_decision_trees_practical.ipynb
Real-world London housing price prediction with Scikit-learn, Random Forests, and XGBoost.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/2b_decision_trees_practical.ipynb)
* [View Source](notebooks/2b_decision_trees_practical.ipynb)

#### 2c_decision_trees_ATLAS_model_comparison.ipynb
Automated Tree Learning Analysis System (ATLAS) for feature engineering and model comparison.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/2c_decision_trees_ATLAS_model_comparison.ipynb)
* [View Source](notebooks/2c_decision_trees_ATLAS_model_comparison.ipynb)

**Lesson 3: Neural Networks** - Backpropagation and gradient-based learning

#### 3a_neural_networks_theory.ipynb
Backpropagation derivation via chain rule, from-scratch NumPy implementation, MNIST classification.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/3a_neural_networks_theory.ipynb)
* [View Source](notebooks/3a_neural_networks_theory.ipynb)

---

## 📊 Datasets

### [California Housing (1990)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
* Regression problem
* 20,640 samples × 8 features
* Predicting median house values
* **Used in:** Lesson 0 (Linear Regression)

### [Wisconsin Breast Cancer (1995)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
* Binary classification task
* 569 samples × 30 features
* Medical diagnosis application
* **Used in:** Lesson 1 (Logistic Regression)

### [London Housing Prices (2020)](https://www.kaggle.com/datasets/arnavkulkarni/housing-prices-in-london)
* Regression problem
* 3,479 samples × 9 features
* Geographic feature encoding
* **Used in:** Lesson 2 (Decision Trees)

### [MNIST Handwritten Digits (1998)](http://yann.lecun.com/exdb/mnist/)
* Multi-class classification (10 classes)
* 70,000 samples (60k train, 10k test) × 784 features (28×28 pixels)
* Handwritten digit recognition (0-9)
* **Used in:** Lesson 3 (Neural Networks)

## 💻 Local Setup
For those who prefer to run notebooks locally:
```bash
git clone https://github.com/powell-clark/supervised-machine-learning.git
cd supervised-machine-learning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Contributing & License

Copyright © 2025 Powell-Clark Limited  
Licensed under the Apache License, Version 2.0

### Contributing
We welcome community contributions to improve these educational materials! Here's how you can help:

- Spot a typo or unclear explanation? Open a pull request
- Have an idea for improvement? Create an issue
- Want to add new examples or exercises? We'd love to see them
- Found a bug in the code? Let us know

Contributors will be acknowledged in our CONTRIBUTORS.md file which will be created with our first contribution.

### Citation
If you use these materials in your work, please cite as:
```
Powell-Clark (2025). Supervised Machine Learning from First Principles.
GitHub: https://github.com/powell-clark/supervised-machine-learning
```

### Contact
Questions or feedback? Contact emmanuel@powellclark.com