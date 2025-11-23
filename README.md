# 🎓 Supervised Machine Learning: From First Principles to Transformers

This repository teaches machine learning from mathematical foundations through modern deep learning.

Starting with derivatives, logarithms, and probability, each notebook builds understanding step-by-step using Python, NumPy, and production tools (Scikit-learn, PyTorch, TensorFlow). You'll implement algorithms from scratch to understand them deeply, then use industry-standard libraries for practical work.

**What's covered:**
- 9 classical algorithms (linear regression → anomaly detection)
- Modern deep learning (CNNs, RNNs, Transformers)
- Production practices (hyperparameter tuning, model evaluation)
- Model interpretability (SHAP, LIME)
- Ethics and bias detection

Assumes basic high school calculus. Companion repositories for unsupervised and reinforcement learning in development.

## 📚 Notebooks

🚀 Quick Start:  Run notebooks directly in your browser - no setup required!
1. Click any "Open in Colab" button below
2. In Colab: Click "Connect" (top-right)
3. Click "Runtime" > "Run all" (top menu)

### Foundation
**Lesson 0: Linear Regression** - The foundation of machine learning

#### 0a_linear_regression_theory.ipynb
Theory & from-scratch implementation with Normal Equation and Gradient Descent.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/0a_linear_regression_theory.ipynb)
* [View Source](notebooks/0a_linear_regression_theory.ipynb)

#### 0b_linear_regression_practical.ipynb
Production implementations with Scikit-learn, polynomial features, and Ridge/Lasso regularization.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/0b_linear_regression_practical.ipynb)
* [View Source](notebooks/0b_linear_regression_practical.ipynb)

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

**Lesson 3: Neural Networks** - Deep learning from backpropagation to production PyTorch

#### 3a_neural_networks_theory.ipynb
Theory & from-scratch implementation with forward and backpropagation on MNIST digits.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/3a_neural_networks_theory.ipynb)
* [View Source](notebooks/3a_neural_networks_theory.ipynb)

#### 3b_neural_networks_practical.ipynb
Production PyTorch: modern optimizers, regularization, deeper architectures, GPU acceleration.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/3b_neural_networks_practical.ipynb)
* [View Source](notebooks/3b_neural_networks_practical.ipynb)

**Lesson 4: Support Vector Machines** - Maximum margin classification with kernels

#### 4a_svm_theory.ipynb
Maximum margin theory, support vectors, kernel trick, and from-scratch implementation.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/4a_svm_theory.ipynb)
* [View Source](notebooks/4a_svm_theory.ipynb)

#### 4b_svm_practical.ipynb
Scikit-learn SVM with kernel comparison and hyperparameter tuning.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/4b_svm_practical.ipynb)
* [View Source](notebooks/4b_svm_practical.ipynb)

**Lesson 5: K-Nearest Neighbors** - Instance-based learning and distance metrics

#### 5a_knn_theory.ipynb
Distance metrics, choosing K, curse of dimensionality, and from-scratch implementation.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/5a_knn_theory.ipynb)
* [View Source](notebooks/5a_knn_theory.ipynb)

#### 5b_knn_practical.ipynb
Optimized KNN with scikit-learn, finding optimal K, and algorithm comparison.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/5b_knn_practical.ipynb)
* [View Source](notebooks/5b_knn_practical.ipynb)

**Lesson 6: Naive Bayes** - Probabilistic classification with Bayes' Theorem

#### 6a_naive_bayes_theory.ipynb
Bayes' Theorem, conditional independence, Gaussian/Multinomial variants, from-scratch implementation.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/6a_naive_bayes_theory.ipynb)
* [View Source](notebooks/6a_naive_bayes_theory.ipynb)

#### 6b_naive_bayes_practical.ipynb
Text classification with scikit-learn, CountVectorizer/TF-IDF, and 20 Newsgroups dataset.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/6b_naive_bayes_practical.ipynb)
* [View Source](notebooks/6b_naive_bayes_practical.ipynb)

**Lesson 7: Ensemble Methods Mastery** - Advanced ensemble techniques

#### 7a_ensemble_methods_theory.ipynb
Bagging, boosting, stacking theory - why ensembles outperform single models.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/7a_ensemble_methods_theory.ipynb)
* [View Source](notebooks/7a_ensemble_methods_theory.ipynb)

#### 7b_ensemble_practical.ipynb
XGBoost, LightGBM, stacking, and hyperparameter tuning for production.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/7b_ensemble_practical.ipynb)
* [View Source](notebooks/7b_ensemble_practical.ipynb)

**Lesson 8: Anomaly Detection** - Detecting outliers, fraud, and rare events

#### 8a_anomaly_detection_theory.ipynb
Statistical methods, Isolation Forest, One-Class SVM theory and applications.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/8a_anomaly_detection_theory.ipynb)
* [View Source](notebooks/8a_anomaly_detection_theory.ipynb)

#### 8b_anomaly_detection_practical.ipynb
Production anomaly detection systems for fraud detection and monitoring.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/8b_anomaly_detection_practical.ipynb)
* [View Source](notebooks/8b_anomaly_detection_practical.ipynb)

---

### Professional ML Practice (X-Series)
**Cross-cutting skills that apply to all algorithms**

#### X1_feature_engineering.ipynb
Encoding, scaling, transformations, interaction features, time-based features.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/X1_feature_engineering.ipynb)
* [View Source](notebooks/X1_feature_engineering.ipynb)

#### X2_model_evaluation.ipynb
Complete evaluation framework: metrics, cross-validation, ROC curves, statistical testing.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/X2_model_evaluation.ipynb)
* [View Source](notebooks/X2_model_evaluation.ipynb)

#### X3_hyperparameter_tuning.ipynb
Grid search, random search, Bayesian optimization, and AutoML best practices.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/X3_hyperparameter_tuning.ipynb)
* [View Source](notebooks/X3_hyperparameter_tuning.ipynb)

#### X4_imbalanced_data.ipynb
SMOTE, class weights, cost-sensitive learning for handling imbalanced datasets.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/X4_imbalanced_data.ipynb)
* [View Source](notebooks/X4_imbalanced_data.ipynb)

#### X5_interpretability_explainability.ipynb
SHAP, LIME, permutation importance, partial dependence plots for model explainability and EU AI Act compliance.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/X5_interpretability_explainability.ipynb)
* [View Source](notebooks/X5_interpretability_explainability.ipynb)

#### X6_ethics_bias_detection.ipynb
Fairness metrics, bias detection and mitigation, ethical frameworks, and responsible AI deployment.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/X6_ethics_bias_detection.ipynb)
* [View Source](notebooks/X6_ethics_bias_detection.ipynb)

---

### Modern Deep Learning (Lesson 9)
**CNNs, RNNs, and Transformers**

**Lesson 9: Advanced Deep Learning** - CNNs, RNNs, and Transformers

#### 9a_cnns_transfer_learning.ipynb
Convolutional Neural Networks, transfer learning with VGG/ResNet/MobileNet, data augmentation, and computer vision production best practices.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/9a_cnns_transfer_learning.ipynb)
* [View Source](notebooks/9a_cnns_transfer_learning.ipynb)

#### 9b_rnns_sequences.ipynb
Recurrent Neural Networks, LSTM, GRU, bidirectional RNNs, sequence-to-sequence models, and time series forecasting.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/9b_rnns_sequences.ipynb)
* [View Source](notebooks/9b_rnns_sequences.ipynb)

#### 9c_transformers_attention.ipynb
Transformers, attention mechanisms, BERT vs GPT, fine-tuning with Hugging Face, and Vision Transformers.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/powell-clark/supervised-machine-learning/blob/main/notebooks/9c_transformers_attention.ipynb)
* [View Source](notebooks/9c_transformers_attention.ipynb)

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
* **Used in:** Lessons 1 (Logistic Regression), 4 (SVM), 5 (KNN), 6 (Naive Bayes)

### [Iris (1936)](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
* Multi-class classification (3 classes)
* 150 samples × 4 features
* Classic ML dataset
* **Used in:** Lesson 5 (KNN)

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

### [20 Newsgroups (1995)](http://qwone.com/~jason/20Newsgroups/)
* Multi-class text classification (20 classes)
* ~18,000 newsgroup documents
* Natural language processing
* **Used in:** Lesson 6 (Naive Bayes)

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