# 🎓 Supervised Machine Learning

This repository teaches machine learning from first principles using Python. 

Starting with foundational mathematics (derivatives, exp/log, probability), each notebook builds complete understanding before exploring modern tools like Scikit-learn and PyTorch. All mathematical concepts are derived step-by-step, making the content accessible to anyone with basic high school calculus knowledge. 

This aims to be part of a broader machine learning series, with companion repositories for unsupervised, reinforcement and other state-of-the-art machine learning domains in development.

## 📚 Notebooks

🚀 Quick Start:  Run notebooks directly in your browser - no setup required!
1. Click any "Open in Colab" button below 
2. In Colab: Click "Connect" (top-right) 
3. Click "Runtime" > "Run all" (top menu)


### 1a_logistic_regression_theory.ipynb  
Theory & from-scratch implementation of logistic regression using the Wisconsin Breast Cancer dataset.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/1a_logistic_regression_theory.ipynb)
* [View Source](notebooks/1a_logistic_regression_theory.ipynb)

### 1b_logistic_regression_practical.ipynb
Production-grade PyTorch implementation with modern ML engineering practices.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/1b_logistic_regression_practical.ipynb)
* [View Source](notebooks/1b_logistic_regression_practical.ipynb)

### 2a_decision_trees_theory.ipynb
Deep dive into decision tree theory with a complete from-scratch implementation.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/2a_decision_trees_theory.ipynb)
* [View Source](notebooks/2a_decision_trees_theory.ipynb)

### 2b_decision_trees_practical.ipynb
Real-world prediction of London housing market prices with Scikit-learn, random forests and XGBoost.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/2b_decision_trees_practical.ipynb)
* [View Source](notebooks/2b_decision_trees_practical.ipynb)

### 2c_decision_trees_ATLAS.ipynb
Automated Tree Learning Analysis System (ATLAS) for feature engineering and model comparison.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/2c_decision_trees_ATLAS.ipynb)
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

## Contributing & License

This project is licensed under the Apache License 2.0  
Copyright (c) 2025 Powell-Clark Limited

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
Questions or feedback? Contact emmanuel@powell-clark.com