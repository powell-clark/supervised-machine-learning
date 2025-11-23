# Supervised Machine Learning Curriculum Roadmap

## Current State (7 Notebooks)

**Completed - Academic Quality:**
- **Lesson 0:** Linear Regression (0a theory)
- **Lesson 1:** Logistic Regression (1a theory, 1b practical)
- **Lesson 2:** Decision Trees (2a theory, 2b practical, 2c ATLAS)
- **Lesson 3:** Neural Networks (3a theory)

**Quality Standard:**
- Theory notebooks: Mathematical derivations (>100 LaTeX symbols), from-scratch NumPy implementations
- Practical notebooks: Production code with substantial implementations (>20 math symbols)
- Benchmark: 1a has 194 math symbols, 7 implementations, 133KB
- No emojis, no corporate buzzwords, no tool tutorials

---

## Salvageable Content (In Git History at 366684d)

### Quick Wins - Classic Algorithms (~40 hours each)

**Lesson 4: Support Vector Machines**
- Current state: 5.4KB stub, 0 math symbols
- Needs: Maximum margin derivation, Lagrangian dual, kernel trick mathematics, SMO algorithm
- From-scratch: Implement SVM with gradient descent on hinge loss
- Practical: Kernel comparison (linear, RBF, polynomial), hyperparameter C/gamma tuning
- References: MIT 6.034, Stanford CS229 lectures on SVM

**Lesson 5: K-Nearest Neighbors**
- Current state: 5.7KB stub, 6 math symbols
- Needs: Distance metrics (Euclidean, Manhattan, Minkowski), KD-tree mathematics, curse of dimensionality
- From-scratch: Implement KNN with KD-tree for efficiency
- Practical: Optimal K selection via cross-validation, weighted voting
- References: ESL Chapter 13, Hastie et al.

**Lesson 6: Naive Bayes**
- Current state: 6.2KB stub, 8 math symbols
- Needs: Bayes' theorem derivation, conditional independence assumption, Gaussian/Multinomial/Bernoulli variants
- From-scratch: Implement Gaussian NB with MLE parameter estimation
- Practical: Text classification with TF-IDF, Laplace smoothing
- References: Murphy's "Machine Learning: A Probabilistic Perspective" Chapter 3

### Medium Effort (~40-50 hours each)

**Lesson 7: Ensemble Methods**
- Current state: 7.9KB stub, 4 math symbols
- Needs: Bias-variance decomposition, bagging mathematics, AdaBoost derivation, gradient boosting theory
- From-scratch: Implement AdaBoost from scratch
- Practical: XGBoost, LightGBM with hyperparameter tuning strategies
- References: ESL Chapter 10, Friedman's gradient boosting papers

**Lesson 8: Anomaly Detection**
- Current state: 6.0KB stub, 3 math symbols
- Needs: Gaussian distribution modeling, Mahalanobis distance, Isolation Forest mathematics, One-Class SVM theory
- From-scratch: Implement Gaussian anomaly detection
- Practical: Fraud detection case study, ROC curve analysis for imbalanced data
- References: Chandola et al. "Anomaly Detection: A Survey"

### Major Rewrites - Deep Learning (~60-80 hours each)

**Lesson 9a: Convolutional Neural Networks**
- Current state: 0 math, PyTorch tutorial with emojis (🚀✅)
- Needs complete rewrite:
  - Discrete convolution mathematical definition
  - Backpropagation through convolutional layers (chain rule application)
  - Pooling layer gradient derivation
  - Weight sharing and parameter reduction mathematics
- From-scratch: CNN in NumPy (forward + backward pass)
- Practical: Image classification, transfer learning theory (feature reuse mathematics)
- References: Stanford CS231n, Goodfellow's Deep Learning Book Chapter 9

**Lesson 9b: Recurrent Neural Networks**
- Current state: 0 math, PyTorch tutorial
- Needs complete rewrite:
  - Backpropagation Through Time (BPTT) derivation
  - Vanishing/exploding gradient mathematics
  - LSTM gate equations and gradient flow
  - GRU simplification and performance trade-offs
- From-scratch: RNN + LSTM in NumPy
- Practical: Sequence modeling, time series forecasting
- References: Goodfellow Chapter 10, Hochreiter & Schmidhuber LSTM paper

**Lesson 9c: Transformers & Attention**
- Current state: 0 math, marketing language ("MOST IMPORTANT lesson")
- Needs complete rewrite:
  - Scaled dot-product attention mathematical derivation
  - Multi-head attention mathematics (parallel attention computations)
  - Positional encoding theory (sinusoidal vs learned)
  - Self-attention vs cross-attention mathematics
  - Transformer architecture (encoder-decoder) from first principles
- From-scratch: Attention mechanism in NumPy, scaled dot-product implementation
- Practical: Sequence-to-sequence tasks, pre-trained model mathematics
- References: "Attention Is All You Need" paper, Harvard NLP Annotated Transformer

### Not Worth Salvaging - X-Series

**Why delete X-series:**
- Wrong pedagogical format (meta-lessons about tools vs mathematical foundations)
- Corporate training approach (slideshows, not derivations)
- Should be integrated into practical notebooks, not separate lessons

**Better approach:**
- **Feature engineering** → Integrate into 2b (decision trees practical) and other "b" notebooks
- **Model evaluation** → Cover in each practical notebook (confusion matrix, ROC, precision/recall)
- **Hyperparameter tuning** → Show grid search/Bayesian optimization in context (e.g., 4b SVM)
- **Imbalanced data** → Discuss in 8b (anomaly detection practical)
- **Interpretability** → Add SHAP/LIME to 2b (tree-based interpretability)
- **Ethics/bias** → Dedicated section in 1b or 6b (classification fairness)

---

## Proposed Full Curriculum (Academic Quality)

### Core Supervised Learning (Lessons 0-8)
0. Linear Regression ✅
1. Logistic Regression ✅
2. Decision Trees ✅
3. Neural Networks ✅ (theory only)
4. Support Vector Machines ⏳ (salvageable, ~40 hours)
5. K-Nearest Neighbors ⏳ (salvageable, ~40 hours)
6. Naive Bayes ⏳ (salvageable, ~40 hours)
7. Ensemble Methods ⏳ (salvageable, ~50 hours)
8. Anomaly Detection ⏳ (salvageable, ~50 hours)

### Advanced Deep Learning (Lessons 9a-c)
9a. CNNs & Computer Vision ⏳ (needs complete rewrite, ~60 hours)
9b. RNNs & Sequences ⏳ (needs complete rewrite, ~60 hours)
9c. Transformers & Attention ⏳ (needs complete rewrite, ~80 hours)

**Total effort to complete:** ~500 hours

---

## Quality Checklist for New Lessons

**Theory Notebooks (a):**
- [ ] Mathematical derivations with LaTeX (>100 symbols minimum)
- [ ] From-scratch NumPy implementation (no libraries except NumPy/matplotlib)
- [ ] Step-by-step derivations (chain rule, gradients, optimization)
- [ ] Real-world dataset application
- [ ] Convergence analysis or theoretical properties
- [ ] No emojis, no hype language, no corporate buzzwords

**Practical Notebooks (b):**
- [ ] Substantial code (>20 math symbols for mathematical explanations)
- [ ] Production libraries (Scikit-learn, PyTorch) with understanding of underlying math
- [ ] Hyperparameter tuning and model selection
- [ ] Performance analysis and visualization
- [ ] Comparison to from-scratch implementation
- [ ] No "industry-standard" or marketing language

**Benchmarks:**
- 1a_logistic_regression_theory: 194 math symbols, 7 implementations, 133KB
- 2a_decision_trees_theory: 130 math symbols, 13 implementations, 136KB
- 3a_neural_networks_theory: 120 math symbols, 5 implementations, 55KB

---

## Academic References

**Textbooks:**
- **ESL:** Hastie, Tibshirani, Friedman - "Elements of Statistical Learning"
- **Murphy:** Kevin Murphy - "Machine Learning: A Probabilistic Perspective"
- **Goodfellow:** Ian Goodfellow et al. - "Deep Learning"
- **Bishop:** Christopher Bishop - "Pattern Recognition and Machine Learning"

**University Courses:**
- **MIT 6.036:** Introduction to Machine Learning
- **Stanford CS229:** Machine Learning (Andrew Ng)
- **Stanford CS231n:** Convolutional Neural Networks (Karpathy)
- **Caltech CS156:** Learning From Data (Abu-Mostafa)

**Papers:**
- Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- Vaswani et al. (2017) - "Attention Is All You Need"
- Breiman (2001) - "Random Forests"
- Cortes & Vapnik (1995) - "Support-Vector Networks"

---

## Recovery Instructions

To recover deleted content from git history:

```bash
# View what was deleted
git show 366684d:notebooks/4a_svm_theory.ipynb

# Restore specific notebook
git checkout 366684d -- notebooks/4a_svm_theory.ipynb

# Restore all Lessons 4-6
git checkout 366684d -- notebooks/4*.ipynb notebooks/5*.ipynb notebooks/6*.ipynb
```

**Note:** Restored content will need complete rewrite to meet academic standards.
