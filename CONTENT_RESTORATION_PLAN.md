# Content Restoration Plan: Lessons 4-8

**Purpose**: Concrete implementation guide to bring stub lessons to Lessons 1a-2c quality standard

**Target**: Transform 20% stubs → 100% complete (matching Lessons 1a-2c excellence)

---

## 📊 Current State Analysis

### Completion Metrics by Lesson

| Lesson | Topic | Current Lines | Target Lines | Gap | Priority |
|--------|-------|--------------|--------------|-----|----------|
| 4a | SVM Theory | 189 | 1,200 | -1,011 | 🔴 Critical |
| 4b | SVM Practical | 140 | 800 | -660 | 🔴 Critical |
| 5a | KNN Theory | 219 | 1,000 | -781 | 🟠 High |
| 5b | KNN Practical | 167 | 700 | -533 | 🟠 High |
| 6a | Naive Bayes Theory | 218 | 1,000 | -782 | 🟠 High |
| 6b | Naive Bayes Practical | 190 | 700 | -510 | 🟠 High |
| 7a | Ensemble Theory | 256 | 1,500 | -1,244 | 🔴 Critical |
| 7b | Ensemble Practical | 134 | 1,200 | -1,066 | 🔴 Critical |
| 8a | Anomaly Theory | 212 | 1,000 | -788 | 🟡 Medium |
| 8b | Anomaly Practical | 113 | 800 | -687 | 🟡 Medium |
| **Total** | **All Stubs** | **1,838** | **9,900** | **-8,062** | **8-12 weeks** |

---

## 🎯 Lesson 4a (SVM Theory) - Detailed Restoration Plan

### Week 1-2 Sprint: SVM Theory Completion

**Current**: 189 lines (10 cells)
**Target**: 1,200+ lines (45-50 cells)
**Effort**: 10-12 hours of focused development

### 1. Introduction Enhancement (+150 lines, 3-4 cells)

**Current Cell 1** (Stub):
```markdown
# Lesson 4A: Support Vector Machines Theory
SVMs find the optimal decision boundary by maximizing the margin between classes.
```

**Replace With** (Following Lesson 1a's cancer diagnosis story):
```markdown
# Lesson 4A: Support Vector Machines Theory

<a name="introduction"></a>
## Introduction

Picture yourself as a radiologist reviewing a tumor scan. You've identified two key measurements: tumor size and cell density. Plotting these, you see a pattern emerging.

Some tumors cluster in the "clearly benign" region - small size, low cell density. Others cluster in "clearly malignant" territory - large, dense, aggressive. But between these clusters lies a gray zone of borderline cases where one wrong decision could mean unnecessary surgery or, worse, undetected cancer.

You need more than just a line separating the two groups. You need the *safest possible* boundary - one that stays as far away from borderline cases as possible, giving you the widest "confidence margin" for your diagnosis.

This is exactly what Support Vector Machines do mathematically. They find the classification boundary with **maximum margin** from both classes, creating the safest possible decision rule.

In this lesson, we'll:
1. Understand the geometric intuition behind maximum margin classification
2. Derive the mathematical formulation (primal and dual)
3. Discover the kernel trick for non-linear boundaries
4. Implement SVM from scratch using quadratic programming
5. Apply it to cancer diagnosis using the Wisconsin Breast Cancer dataset
6. Compare linear, polynomial, and RBF kernels
7. Understand when to use SVM vs other algorithms

Then in Lesson 4B, we'll:
1. Use scikit-learn's production SVM implementation
2. Master hyperparameter tuning (C, gamma, kernel selection)
3. Apply SVMs to multi-class problems
4. Handle imbalanced datasets
5. Optimize for large-scale applications
```

### 2. Mathematical Foundation (+600 lines, 15-18 cells)

**Add Complete Derivation Sequence**:

**Cell: "The Margin Concept"** (80 lines)
```python
"""
## What is the Margin?

Given a hyperplane w·x + b = 0, the margin is the distance from the
hyperplane to the nearest point from either class.

Mathematical Definition:
For a point x_i with label y_i ∈ {-1, +1}, the functional margin is:
    γ̂_i = y_i(w·x_i + b)

If γ̂_i > 0, the point is classified correctly.
If γ̂_i >> 0, we're confident about the classification.

The geometric margin is:
    γ_i = γ̂_i / ||w||

Why divide by ||w||? Because the distance from point x to hyperplane w·x + b = 0 is:
    distance = |w·x + b| / ||w||

[Complete geometric proof with diagrams]
"""

# Visualization: Show how ||w|| affects margin width
X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y = np.array([-1, -1, -1, 1, 1, 1])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Show three hyperplanes with different ||w||
for idx, (w_norm, ax) in enumerate(zip([0.5, 1.0, 2.0], axes)):
    # Plot points
    ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', label='Class -1', s=100)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class +1', s=100)

    # Plot hyperplane and margins
    # [Complete visualization code showing margin width]

    ax.set_title(f'||w|| = {w_norm:.1f}, Margin = {2/w_norm:.1f}')

plt.suptitle('Margin Width = 2/||w||: Smaller ||w|| = Wider Margin', fontsize=14)
plt.show()

print("Key Insight: To maximize margin, we minimize ||w||!")
```

**Cell: "Primal Formulation"** (120 lines)
```python
"""
## The Primal Optimization Problem

Goal: Find the hyperplane that maximizes the margin.

Maximizing margin = Maximizing 2/||w|| = Minimizing ||w||

For mathematical convenience, we minimize (1/2)||w||²

Subject to: All points classified correctly with margin ≥ 1
    y_i(w·x_i + b) ≥ 1  for all i

This is a **convex quadratic programming** problem!

Complete Formulation:
    minimize: (1/2)||w||²
    subject to: y_i(w·x_i + b) ≥ 1, i = 1,...,n

Why is this convex?
- Objective: (1/2)||w||² is a convex quadratic function
- Constraints: y_i(w·x_i + b) ≥ 1 are linear (affine) constraints
- Convex objective + convex feasible region = global optimum guaranteed!

[Proof of convexity]
[Visualization of constraint regions]
"""

# Worked Example: 2D case with 4 points
X_example = np.array([[1, 1], [2, 2], [3, 1], [4, 3]])
y_example = np.array([-1, -1, 1, 1])

print("Example Problem:")
print(f"Points: {X_example}")
print(f"Labels: {y_example}")
print("\nPrimal Problem:")
print("minimize: (1/2)(w₁² + w₂²)")
print("subject to:")
for i in range(len(X_example)):
    x1, x2 = X_example[i]
    y_val = y_example[i]
    sign = '+' if y_val > 0 else '-'
    print(f"  {sign}(w₁·{x1} + w₂·{x2} + b) ≥ 1")

# [Solve this simple case analytically to show solution]
```

**Cell: "Lagrangian Dual Formulation"** (200 lines)
```python
"""
## The Dual Problem

Why go to the dual?
1. Easier to solve when n_features >> n_samples
2. Enables the kernel trick
3. Only support vectors matter (sparse solution)

Lagrangian:
    L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w·xᵢ + b) - 1]

where α = [α₁, α₂, ..., αₙ] are Lagrange multipliers (αᵢ ≥ 0)

Karush-Kuhn-Tucker (KKT) Conditions:
1. Stationarity: ∇_w L = 0, ∂L/∂b = 0
2. Primal feasibility: yᵢ(w·xᵢ + b) ≥ 1
3. Dual feasibility: αᵢ ≥ 0
4. Complementary slackness: αᵢ[yᵢ(w·xᵢ + b) - 1] = 0

Derivation:

Step 1: ∇_w L = 0
    w - Σᵢ αᵢyᵢxᵢ = 0
    ⟹ w = Σᵢ αᵢyᵢxᵢ  ← w expressed in terms of training data!

Step 2: ∂L/∂b = 0
    Σᵢ αᵢyᵢ = 0  ← constraint on α

Step 3: Substitute back into L
    L = (1/2)||Σᵢ αᵢyᵢxᵢ||² - Σᵢ αᵢ[yᵢ((Σⱼ αⱼyⱼxⱼ)·xᵢ + b) - 1]

    [Algebraic manipulation over 30 lines]

    = Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)

Dual Problem:
    maximize: Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
    subject to: αᵢ ≥ 0, Σᵢ αᵢyᵢ = 0

Key Insight: Only inner products xᵢ·xⱼ appear! (Enables kernel trick)
"""

# Worked Example: Solve dual for 4-point case
print("Dual Problem for 4-point example:")
print("maximize: Σαᵢ - (1/2)ΣΣ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)")
print("\nKernel Matrix K (xᵢ·xⱼ):")
K = X_example @ X_example.T
print(K)

# [Complete solution showing optimal α values]
# [Visualization showing which points become support vectors]
```

**Cell: "The Kernel Trick"** (150 lines)
```python
"""
## The Kernel Trick: Non-Linear Classification

Problem: What if classes aren't linearly separable?

Solution: Map data to higher-dimensional space where they ARE separable!

    φ: ℝᵈ → ℝᴰ  (d << D, maybe D = ∞!)

Example: 2D → 3D transformation
    φ([x₁, x₂]) = [x₁², √2·x₁x₂, x₂²]

In the dual problem, we only need inner products φ(xᵢ)·φ(xⱼ)

The Trick: Compute inner products in high-D space WITHOUT computing φ!

Kernel function: K(x, x') = φ(x)·φ(x')

Common Kernels:

1. Linear: K(x, x') = x·x'
   (No transformation, original space)

2. Polynomial: K(x, x') = (x·x' + c)ᵈ
   (Polynomial features of degree d)

3. RBF (Radial Basis Function): K(x, x') = exp(-γ||x - x'||²)
   (Infinite-dimensional feature space!)
   γ controls "width" - larger γ = more local influence

4. Sigmoid: K(x, x') = tanh(κx·x' + c)
   (Similar to neural network activation)

Proof: RBF kernel corresponds to infinite dimensions

K(x, x') = exp(-γ||x - x'||²)
         = exp(-γ(||x||² + ||x'||² - 2x·x'))
         = exp(-γ||x||²)·exp(-γ||x'||²)·exp(2γx·x')

Taylor expansion of exp(2γx·x'):
    = Σₙ (2γx·x')ⁿ/n!

This is an infinite sum of polynomial features!
"""

# Visualization: Show kernel transformations
from mpl_toolkits.mplot3d import Axes3D

# Generate non-linearly separable data (two circles)
theta = np.linspace(0, 2*np.pi, 50)
r1, r2 = 1, 3

# Inner circle (class -1)
X_inner = np.column_stack([r1*np.cos(theta), r1*np.sin(theta)])
# Outer circle (class +1)
X_outer = np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])

X_circles = np.vstack([X_inner, X_outer])
y_circles = np.array([-1]*50 + [1]*50)

fig = plt.figure(figsize=(15, 5))

# Plot 1: Original 2D space (not linearly separable)
ax1 = fig.add_subplot(131)
ax1.scatter(X_circles[y_circles==-1, 0], X_circles[y_circles==-1, 1],
            c='blue', label='Class -1')
ax1.scatter(X_circles[y_circles==1, 0], X_circles[y_circles==1, 1],
            c='red', label='Class +1')
ax1.set_title('Original Space (2D)\nNot Linearly Separable')
ax1.legend()

# Plot 2: Transform to 3D using φ(x) = [x₁, x₂, x₁² + x₂²]
ax2 = fig.add_subplot(132, projection='3d')
X_transformed = np.column_stack([
    X_circles[:, 0],
    X_circles[:, 1],
    X_circles[:, 0]**2 + X_circles[:, 1]**2
])
ax2.scatter(X_transformed[y_circles==-1, 0],
           X_transformed[y_circles==-1, 1],
           X_transformed[y_circles==-1, 2], c='blue')
ax2.scatter(X_transformed[y_circles==1, 0],
           X_transformed[y_circles==1, 1],
           X_transformed[y_circles==1, 2], c='red')
ax2.set_title('Transformed Space (3D)\nLinearly Separable!')

# Plot 3: Decision boundary back in 2D (using RBF kernel)
ax3 = fig.add_subplot(133)
# [Show RBF SVM decision boundary]
ax3.set_title('RBF Kernel Decision Boundary\n(Computed Without Explicit Transform!)')

plt.tight_layout()
plt.show()

print("✅ Kernel trick: Get high-D power at low-D cost!")
```

**Cell: "Soft Margin SVM"** (100 lines)
```python
"""
## Soft Margin: Handling Imperfect Separation

Problem: What if data is noisy or overlapping?

Hard margin requires: yᵢ(w·xᵢ + b) ≥ 1 (all points must be correctly classified)
→ No solution if data isn't perfectly separable!

Solution: Allow some violations with penalty

Introduce slack variables ξᵢ ≥ 0 for each point:
    yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ

ξᵢ = 0: Point outside or on correct margin
ξᵢ ∈ (0, 1]: Point inside margin but correctly classified
ξᵢ > 1: Point misclassified

Soft Margin Objective:
    minimize: (1/2)||w||² + C·Σᵢ ξᵢ
    subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

C is the regularization parameter:
- C → ∞: Hard margin (no violations allowed)
- C → 0: Very soft margin (prioritize large margin over accuracy)
- C moderate: Trade-off between margin width and violations

Effect of C:
- High C: Fewer violations, risk overfitting (small margin)
- Low C: More violations, risk underfitting (large margin)

[Visualization showing effect of different C values]
"""

# Demo: Show how C affects decision boundary
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, C_val in enumerate([0.1, 1.0, 100.0]):
    ax = axes[idx]

    # Train SVM with this C
    from sklearn.svm import SVC
    svm = SVC(kernel='linear', C=C_val)
    svm.fit(X_circles[:20], y_circles[:20])  # Use subset with overlap

    # Plot decision boundary and margins
    # [Visualization code]

    ax.set_title(f'C = {C_val}\n' +
                 ('Small margin, few violations' if C_val > 10 else
                  'Large margin, more violations' if C_val < 1 else
                  'Balanced'))

plt.tight_layout()
plt.show()
```

### 3. From-Scratch Implementation (+400 lines, 8-10 cells)

**Cell: "SVM Class Structure"** (250 lines)
```python
class SVMFromScratch:
    """
    Support Vector Machine with complete dual formulation.

    This implementation solves the dual optimization problem using
    quadratic programming, supporting multiple kernel functions.

    Mathematical Background
    -----------------------
    SVM finds the hyperplane w·x + b = 0 that maximizes the margin 2/||w||.

    In the dual formulation, we solve:
        maximize: L(α) = Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        subject to: 0 ≤ αᵢ ≤ C, Σᵢ αᵢyᵢ = 0

    where K(xᵢ,xⱼ) is the kernel function.

    Support vectors are points with αᵢ > 0. These are the only points
    that matter for the final decision function:
        f(x) = Σᵢ∈SV αᵢyᵢK(xᵢ,x) + b

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel function:
        - 'linear': K(x,x') = x·x'
        - 'poly': K(x,x') = (γx·x' + r)^d
        - 'rbf': K(x,x') = exp(-γ||x-x'||²)
        - 'sigmoid': K(x,x') = tanh(γx·x' + r)

    C : float, default=1.0
        Regularization parameter. Higher C → fewer margin violations.
        Lower C → wider margin with more violations.

    gamma : float or 'auto', default='auto'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        If 'auto', uses 1/n_features.

    degree : int, default=3
        Degree for polynomial kernel.

    coef0 : float, default=0.0
        Independent term in poly and sigmoid kernels.

    Attributes
    ----------
    alpha_ : ndarray of shape (n_support_vectors,)
        Lagrange multipliers for support vectors.

    support_vectors_ : ndarray of shape (n_support_vectors, n_features)
        Support vectors (points with α > 0).

    support_vector_labels_ : ndarray of shape (n_support_vectors,)
        Labels of support vectors.

    b_ : float
        Bias term in decision function.

    Examples
    --------
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> svm = SVMFromScratch(kernel='linear', C=1.0)
    >>> svm.fit(X, y)
    >>> svm.predict([[0.5, 0.5]])
    array([0])
    """

    def __init__(self, kernel='rbf', C=1.0, gamma='auto',
                 degree=3, coef0=0.0, tol=1e-3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol

        # Fitted parameters
        self.alpha_ = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.b_ = None

    def _kernel_function(self, X1, X2):
        """
        Compute kernel matrix K[i,j] = K(X1[i], X2[j])

        Parameters
        ----------
        X1 : ndarray of shape (n_samples_1, n_features)
        X2 : ndarray of shape (n_samples_2, n_features)

        Returns
        -------
        K : ndarray of shape (n_samples_1, n_samples_2)
            Kernel matrix
        """
        if self.kernel == 'linear':
            # K(x, x') = x·x'
            return X1 @ X2.T

        elif self.kernel == 'rbf':
            # K(x, x') = exp(-γ||x-x'||²)
            # Efficient computation using:
            # ||x-x'||² = ||x||² + ||x'||² - 2x·x'

            # Shape: (n_samples_1, 1)
            X1_norm_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            # Shape: (1, n_samples_2)
            X2_norm_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            # Shape: (n_samples_1, n_samples_2)
            distances_sq = X1_norm_sq + X2_norm_sq - 2 * (X1 @ X2.T)

            return np.exp(-self.gamma * distances_sq)

        elif self.kernel == 'poly':
            # K(x, x') = (γx·x' + r)^d
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree

        elif self.kernel == 'sigmoid':
            # K(x, x') = tanh(γx·x' + r)
            return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """
        Fit SVM by solving the dual optimization problem.

        Steps:
        1. Compute kernel matrix K
        2. Solve QP: maximize L(α) subject to constraints
        3. Identify support vectors (α > 0)
        4. Compute bias b

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target labels (must be -1 or +1)

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Convert labels to {-1, +1}
        y = np.where(y <= 0, -1, 1)

        # Set gamma if 'auto'
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features

        # Compute kernel matrix
        K = self._kernel_function(X, X)

        # Solve dual problem using quadratic programming
        # We need to solve:
        # maximize: Σα_i - (1/2)ΣΣ α_i α_j y_i y_j K(x_i,x_j)
        # subject to: 0 ≤ α_i ≤ C, Σ α_i y_i = 0

        # Convert to minimization problem (negate objective)
        # minimize: (1/2)α^T P α + q^T α
        # where P[i,j] = y_i y_j K(x_i, x_j), q = -1

        P = np.outer(y, y) * K  # Shape: (n_samples, n_samples)
        q = -np.ones(n_samples)

        # Equality constraint: y^T α = 0
        A_eq = y.reshape(1, -1)
        b_eq = np.array([0.0])

        # Inequality constraints: 0 ≤ α_i ≤ C
        # We use bounds instead of inequality constraints
        bounds = [(0, self.C) for _ in range(n_samples)]

        # Solve using scipy.optimize.minimize
        from scipy.optimize import minimize

        # Initial guess: α = 0
        alpha_init = np.zeros(n_samples)

        def objective(alpha):
            return 0.5 * alpha @ P @ alpha + q @ alpha

        def jac_objective(alpha):
            return P @ alpha + q

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda alpha: A_eq @ alpha,
                      'jac': lambda alpha: A_eq.flatten()}

        # Solve
        result = minimize(
            objective,
            alpha_init,
            method='SLSQP',
            jac=jac_objective,
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )

        alpha = result.x

        # Support vectors have α > threshold
        sv_idx = alpha > self.tol
        self.alpha_ = alpha[sv_idx]
        self.support_vectors_ = X[sv_idx]
        self.support_vector_labels_ = y[sv_idx]

        # Compute bias b using support vectors with 0 < α < C
        # These are points exactly on the margin
        margin_sv_idx = (alpha > self.tol) & (alpha < self.C - self.tol)

        if np.sum(margin_sv_idx) > 0:
            # b = y_k - Σ α_i y_i K(x_i, x_k) for any margin SV x_k
            K_sv = self._kernel_function(X[sv_idx], X[margin_sv_idx])
            b_values = y[margin_sv_idx] - (self.alpha_ * self.support_vector_labels_) @ K_sv
            self.b_ = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            K_sv = self._kernel_function(self.support_vectors_, self.support_vectors_)
            self.b_ = np.mean(self.support_vector_labels_ -
                             (self.alpha_ * self.support_vector_labels_) @ K_sv)

        return self

    def decision_function(self, X):
        """
        Compute decision function f(x) = Σ α_i y_i K(x_i, x) + b

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision function values (distance from hyperplane)
        """
        K = self._kernel_function(X, self.support_vectors_)
        return (K @ (self.alpha_ * self.support_vector_labels_)) + self.b_

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

print("✅ Complete SVM implementation ready!")
```

[Continue with remaining cells for implementation testing, visualization, etc.]

---

## 🎯 Quick Win: Template Files

Create these template notebooks to accelerate development:

### Template: Theory Lesson Structure
```python
# Save as: notebooks/TEMPLATE_theory.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lesson Xa: [Algorithm] Theory\n",
    "\n",
    "## Introduction\n",
    "[Story-driven motivation - 150 lines]\n",
    "## Table of Contents\n",
    "[Complete TOC with anchors]\n",
    "## Mathematical Foundation\n",
    "[600-800 lines of derivations]\n",
    "## From-Scratch Implementation\n",
    "[400-600 lines with full class]\n",
    "## Real-World Application\n",
    "[500-700 lines dataset analysis]\n",
    "## When to Use\n",
    "[200-300 lines guidance]\n",
    "## Conclusion\n",
    "[50-100 lines summary]"
   ]
  }
 ]
}
```

---

## 📅 Implementation Schedule

### Sprint 1 (Weeks 1-2): Lesson 4 (SVM)
- **Days 1-3**: 4a Introduction + Math derivations (600 lines)
- **Days 4-6**: 4a From-scratch implementation (400 lines)
- **Day 7**: 4a Application + visualization (300 lines)
- **Days 8-10**: 4b Scikit-learn practical (600 lines)
- **Days 11-12**: 4b Hyperparameter tuning + multi-class (200 lines)
- **Days 13-14**: Review, test, polish both notebooks

### Sprint 2 (Weeks 3-4): Lesson 7 (Ensemble Methods)
[Similar breakdown]

---

**Status**: Ready for implementation
**Next Action**: Begin Lesson 4a restoration following this plan
