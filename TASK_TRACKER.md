# 📋 Task Tracker - Repository Improvement

**Purpose**: Detailed task tracking with specific implementation notes
**Status**: Ready to begin
**Last Updated**: November 2025

---

## Phase 1: Critical Fixes 🔴

### Task 1: Fix Numerical Stability in Linear Regression
- **File**: `notebooks/0a_linear_regression_theory.ipynb`
- **Location**: Cell 10
- **Priority**: CRITICAL
- **Status**: ⏳ Pending
- **Estimated Time**: 15 minutes

**Current Code:**
```python
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

**Replacement Code:**
```python
# Using least squares for better numerical stability
# This avoids explicit matrix inversion which can be unstable
theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
```

**Additional Context to Add:**
```markdown
**Note on Numerical Stability**: We use `lstsq` instead of computing the inverse
explicitly. When `X^T X` is nearly singular or poorly conditioned, `inv()` can
produce inaccurate results. The `lstsq` function uses more robust algorithms
(QR decomposition or SVD) for better numerical stability.
```

**Testing Checklist:**
- [ ] Code runs without errors
- [ ] Results are identical (or very close) to previous implementation
- [ ] Explanation is clear for learners
- [ ] User approval obtained

---

### Task 2: Fix Data Leakage in Target Encoding
- **File**: `notebooks/X1_feature_engineering.ipynb`
- **Location**: Cell 7
- **Priority**: CRITICAL
- **Status**: ⏳ Pending
- **Estimated Time**: 30 minutes

**Issue**: Current code computes statistics on entire dataset before split

**Current Problematic Pattern:**
```python
target_means = df_target.groupby('city')['price'].mean()
df_target['city_encoded'] = df_target['city'].map(target_means)
# Then split train/test - TOO LATE!
```

**Fix Required:**

1. Add prominent warning box:
```markdown
## ⚠️ CRITICAL: Avoiding Data Leakage in Target Encoding

**Wrong Approach** (causes data leakage):
```python
# DON'T DO THIS - computes on full dataset
target_means = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_means)
X_train, X_test, y_train, y_test = train_test_split(...)
```

**Why it's wrong**: Test set statistics leak into training!

**Right Approach** (prevents leakage):
```python
# Split first
X_train, X_test, y_train, y_test = train_test_split(...)

# Compute encoding ONLY on training data
target_means = X_train.groupby('category')['target'].mean()

# Apply to both sets
X_train['category_encoded'] = X_train['category'].map(target_means)
X_test['category_encoded'] = X_test['category'].map(target_means)

# For categories in test not in train, use global mean as fallback
global_mean = y_train.mean()
X_test['category_encoded'].fillna(global_mean, inplace=True)
```

**Best Approach** (cross-validation safe):
Use sklearn's TargetEncoder with proper CV:
```python
from category_encoders import TargetEncoder

encoder = TargetEncoder()
# Fit only on training data
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)
```
```

**Testing Checklist:**
- [ ] Warning is prominent and clear
- [ ] Both wrong and right examples shown
- [ ] Code examples run correctly
- [ ] Explanation prevents misunderstanding
- [ ] User approval obtained

---

### Task 3: Document Missing Dependencies
- **File**: `notebooks/X1_feature_engineering.ipynb`
- **Location**: Cell 3 (before first use of category_encoders)
- **Priority**: CRITICAL
- **Status**: ⏳ Pending
- **Estimated Time**: 10 minutes

**Add Installation Cell:**
```markdown
## Installing Required Libraries

This notebook uses `category-encoders` which may not be pre-installed:
```

```python
# Uncomment if running locally and library not installed
# !pip install category-encoders

# If on Colab, this should install automatically:
try:
    import category_encoders
except ImportError:
    !pip install category-encoders
    import category_encoders

print(f"category-encoders version: {category_encoders.__version__}")
```

**Also Update**: `requirements.txt` - already has it, but add comment:
```
# Feature engineering (used in X1)
category-encoders==2.6.4
```

**Testing Checklist:**
- [ ] Cell runs and installs if needed
- [ ] Clear instructions for different environments
- [ ] User approval obtained

---

### Task 4: Complete or Remove Featuretools Section
- **File**: `notebooks/X1_feature_engineering.ipynb`
- **Location**: Cells 16-17
- **Priority**: CRITICAL
- **Status**: ⏳ Pending - **USER DECISION NEEDED**
- **Estimated Time**: 15 min (remove) OR 45 min (implement)

**Option A: Remove Section**
Replace cells 16-17 with:
```markdown
## 9. Automated Feature Engineering

Advanced libraries like **Featuretools** can automatically generate features
from relational datasets. This is beyond the scope of this tutorial, but
highly recommended for production systems with complex data relationships.

**Resources:**
- Featuretools Documentation: https://www.featuretools.com/
- Tutorial: https://docs.featuretools.com/en/stable/getting_started/getting_started.html

For most use cases, the manual techniques covered in this notebook are sufficient
and give you more control over your feature engineering process.
```

**Option B: Implement Complete Example**
Add working Featuretools example (requires more time):
```python
# Installation
!pip install featuretools

import featuretools as ft
import pandas as pd

# Create example dataframe
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'age': [25, 35, 45],
    'join_date': pd.date_range('2020-01-01', periods=3)
})

transactions = pd.DataFrame({
    'transaction_id': [1, 2, 3, 4, 5],
    'customer_id': [1, 1, 2, 2, 3],
    'amount': [100, 150, 200, 50, 300],
    'transaction_date': pd.date_range('2020-06-01', periods=5)
})

# Create entity set
es = ft.EntitySet(id='customers')
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers,
    index='customer_id'
)
es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions,
    index='transaction_id',
    time_index='transaction_date'
)
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Automatically generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2
)

print(f"Generated {len(feature_defs)} features automatically!")
print(feature_matrix.head())
```

**USER MUST CHOOSE**: Option A (quick) or Option B (comprehensive)?

**Testing Checklist:**
- [ ] If Option A: Documentation is clear and helpful
- [ ] If Option B: Code runs successfully and demonstrates value
- [ ] User approval obtained

---

## Phase 2: Enhanced Visualizations 🟡

### Task 5: Add Training History Plots (Neural Networks)
- **File**: `notebooks/3b_neural_networks_practical.ipynb`
- **Location**: After training loops (cells need to be identified)
- **Priority**: HIGH
- **Status**: ⏳ Pending
- **Estimated Time**: 45 minutes

**Implementation Pattern:**

After each training loop, add visualization cell:

```python
import matplotlib.pyplot as plt

# Assuming train_losses, val_losses, train_accs, val_accs are collected
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(train_losses, label='Training Loss', linewidth=2)
ax1.plot(val_losses, label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(train_accs, label='Training Accuracy', linewidth=2)
ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary
print(f"\n📊 Training Summary:")
print(f"   Best Validation Accuracy: {max(val_accs):.2f}%")
print(f"   Final Training Loss: {train_losses[-1]:.4f}")
print(f"   Final Validation Loss: {val_losses[-1]:.4f}")
```

**Locations to Add:**
1. After basic training loop (~cell 12-13)
2. After Adam optimizer training (~cell 19)
3. After regularized training (~cell 24)
4. After deep network training (~cell 29)

**Also Add Learning Rate Tracking** (for scheduler section):
```python
# Track learning rates during training
lr_history = []
for epoch in range(epochs):
    # ... training code ...
    lr_history.append(optimizer.param_groups[0]['lr'])

# Plot learning rate changes
plt.figure(figsize=(10, 4))
plt.plot(lr_history, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.show()
```

**Testing Checklist:**
- [ ] All plots render correctly
- [ ] Plots are informative and clear
- [ ] Consistent style across plots
- [ ] User approval obtained

---

### Task 6: Add Cost Function Visualization (Linear Regression)
- **File**: `notebooks/0a_linear_regression_theory.ipynb`
- **Location**: After introducing MSE, before gradient descent section
- **Priority**: HIGH
- **Status**: ⏳ Pending
- **Estimated Time**: 30 minutes

**Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create simple dataset
np.random.seed(42)
X_simple = 2 * np.random.rand(20, 1)
y_simple = 4 + 3 * X_simple + np.random.randn(20, 1)

# Create grid of theta values
theta0_vals = np.linspace(0, 8, 100)
theta1_vals = np.linspace(0, 6, 100)
Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)

# Compute cost for each theta combination
X_b = np.c_[np.ones((20, 1)), X_simple]
costs = np.zeros(Theta0.shape)
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta = np.array([[Theta0[j, i]], [Theta1[j, i]]])
        predictions = X_b @ theta
        costs[j, i] = np.mean((predictions - y_simple) ** 2)

# 3D Surface Plot
fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(Theta0, Theta1, costs, cmap='viridis', alpha=0.8)
ax1.set_xlabel('θ₀ (intercept)', fontsize=11)
ax1.set_ylabel('θ₁ (slope)', fontsize=11)
ax1.set_zlabel('Cost (MSE)', fontsize=11)
ax1.set_title('Cost Function Surface', fontsize=13, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Mark optimal point
theta_optimal = np.linalg.lstsq(X_b, y_simple, rcond=None)[0]
cost_optimal = np.mean((X_b @ theta_optimal - y_simple) ** 2)
ax1.scatter([theta_optimal[0]], [theta_optimal[1]], [cost_optimal],
            color='red', s=100, marker='*', label='Optimal θ')

# 2D Contour Plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(Theta0, Theta1, costs, levels=30, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(theta_optimal[0], theta_optimal[1], 'r*', markersize=15, label='Optimal θ')
ax2.set_xlabel('θ₀ (intercept)', fontsize=11)
ax2.set_ylabel('θ₁ (slope)', fontsize=11)
ax2.set_title('Cost Function Contours', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Optimal parameters: θ₀ = {theta_optimal[0][0]:.2f}, θ₁ = {theta_optimal[1][0]:.2f}")
print(f"Minimum cost: {cost_optimal:.2f}")
```

**Add Explanation:**
```markdown
The cost function forms a **convex bowl shape** in parameter space. This is great news:
- There's only one minimum (no local minima to get stuck in)
- Gradient descent is guaranteed to find the optimal solution
- The shape guides us downhill toward the optimal parameters
```

**Testing Checklist:**
- [ ] 3D plot renders correctly
- [ ] Contour plot is clear
- [ ] Optimal point is marked
- [ ] User approval obtained

---

### Task 7: Add Feature Normalization Impact Visualization
- **File**: `notebooks/0a_linear_regression_theory.ipynb`
- **Location**: Before cell 14 (where normalization is applied)
- **Priority**: HIGH
- **Status**: ⏳ Pending
- **Estimated Time**: 40 minutes

**Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Create dataset with very different feature scales
np.random.seed(42)
n_samples = 100
X_large_scale = np.random.rand(n_samples, 1) * 10000  # Feature 1: 0-10000
X_small_scale = np.random.rand(n_samples, 1) * 0.1    # Feature 2: 0-0.1
X_multi = np.c_[X_large_scale, X_small_scale]
y = 3 * X_large_scale + 50 * X_small_scale + np.random.randn(n_samples, 1) * 100

# Gradient descent WITHOUT normalization
X_b = np.c_[np.ones((n_samples, 1)), X_multi]
theta_unnorm = np.random.randn(3, 1)
learning_rate = 0.0001  # Must be tiny!
n_iterations = 1000
costs_unnorm = []

for iteration in range(n_iterations):
    predictions = X_b @ theta_unnorm
    errors = predictions - y
    gradients = 2/n_samples * X_b.T @ errors
    theta_unnorm -= learning_rate * gradients
    cost = np.mean(errors ** 2)
    costs_unnorm.append(cost)

# Gradient descent WITH normalization
X_normalized = (X_multi - X_multi.mean(axis=0)) / X_multi.std(axis=0)
X_b_norm = np.c_[np.ones((n_samples, 1)), X_normalized]
theta_norm = np.random.randn(3, 1)
learning_rate_norm = 0.1  # Can be much larger!
costs_norm = []

for iteration in range(n_iterations):
    predictions = X_b_norm @ theta_norm
    errors = predictions - y
    gradients = 2/n_samples * X_b_norm.T @ errors
    theta_norm -= learning_rate_norm * gradients
    cost = np.mean(errors ** 2)
    costs_norm.append(cost)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Feature scales before normalization
axes[0].scatter(X_multi[:, 0], X_multi[:, 1], alpha=0.6)
axes[0].set_xlabel('Feature 1 (scale: 0-10000)', fontsize=11)
axes[0].set_ylabel('Feature 2 (scale: 0-0.1)', fontsize=11)
axes[0].set_title('Original Features\n(Very Different Scales!)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Convergence without normalization
axes[1].plot(costs_unnorm, linewidth=2, color='red')
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('Cost (MSE)', fontsize=11)
axes[1].set_title(f'Without Normalization\n(lr={learning_rate}, slow!)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Convergence with normalization
axes[2].plot(costs_norm, linewidth=2, color='green')
axes[2].set_xlabel('Iteration', fontsize=11)
axes[2].set_ylabel('Cost (MSE)', fontsize=11)
axes[2].set_title(f'With Normalization\n(lr={learning_rate_norm}, fast!)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("🎯 Key Insight:")
print(f"   Without normalization: Final cost = {costs_unnorm[-1]:,.0f}")
print(f"   With normalization:    Final cost = {costs_norm[-1]:,.0f}")
print(f"   \n   Learning rate can be {learning_rate_norm/learning_rate:.0f}x larger with normalization!")
```

**Add Explanation:**
```markdown
## Why Feature Normalization Matters

When features have very different scales (like Feature 1: 0-10,000 vs Feature 2: 0-0.1):

1. **The cost function becomes elongated** - looks like a narrow valley
2. **Gradient descent oscillates** - bounces back and forth across the valley
3. **Must use tiny learning rate** - otherwise it explodes
4. **Convergence is slow** - takes many iterations

After normalization:
1. **Cost function becomes circular** - like a symmetric bowl
2. **Gradient descent goes straight to minimum** - no oscillation
3. **Can use larger learning rate** - faster updates
4. **Convergence is fast** - reaches minimum quickly

**Rule of thumb**: Always normalize when using gradient descent!
```

**Testing Checklist:**
- [ ] Visualization clearly shows the difference
- [ ] Explanation is understandable
- [ ] Numbers demonstrate the impact
- [ ] User approval obtained

---

### Task 8: Add Decision Boundary Visualizations
- **Files**: `notebooks/1a_logistic_regression_theory.ipynb`, `notebooks/4a_svm_theory.ipynb`, `notebooks/5a_knn_theory.ipynb`
- **Priority**: HIGH
- **Status**: ⏳ Pending
- **Estimated Time**: 45 minutes (15 min per notebook)

**Standard Implementation Pattern:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Assuming we have trained model and 2D dataset
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plot decision boundary for 2D classification

    Parameters:
    - model: trained model with predict method
    - X: features (n_samples, 2)
    - y: labels (n_samples,)
    - title: plot title
    """
    # Create mesh
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o',
                label='Class 0', edgecolors='black', s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s',
                label='Class 1', edgecolors='black', s=50)
    plt.xlabel('Feature 1', fontsize=11)
    plt.ylabel('Feature 2', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

# Use it after training
plot_decision_boundary(model, X_train[:, :2], y_train,
                      "Logistic Regression Decision Boundary")
```

**Add to each notebook:**
- After training from-scratch implementation
- Show how decision boundary changes with different hyperparameters
- For multi-class, show all class regions

**Testing Checklist:**
- [ ] Plots render in all 3 notebooks
- [ ] Decision boundaries are correct
- [ ] Visualizations aid understanding
- [ ] User approval obtained

---

### Task 9: Add Cyclical Encoding Visualization
- **File**: `notebooks/X1_feature_engineering.ipynb`
- **Location**: After cells 14-15 (cyclical encoding section)
- **Priority**: HIGH
- **Status**: ⏳ Pending
- **Estimated Time**: 20 minutes

**Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Create hour data
hours = np.arange(0, 24)
hour_linear = hours  # Linear encoding
hour_sin = np.sin(2 * np.pi * hours / 24)
hour_cos = np.cos(2 * np.pi * hours / 24)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Problem with linear encoding
axes[0].scatter(hour_linear, hour_linear, c=hours, cmap='twilight', s=100, edgecolors='black')
axes[0].plot([0, 23], [0, 23], 'r--', linewidth=2, label='Distance problem: 23→0')
axes[0].set_xlabel('Hour (Linear)', fontsize=11)
axes[0].set_ylabel('Hour (Linear)', fontsize=11)
axes[0].set_title('Linear Encoding Problem\n(23:00 far from 00:00)',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Cyclical encoding solution
scatter = axes[1].scatter(hour_sin, hour_cos, c=hours, cmap='twilight',
                         s=150, edgecolors='black', linewidths=2)
# Draw circle to show cyclical nature
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=2)
axes[1].add_patch(circle)
# Annotate key hours
for h in [0, 6, 12, 18, 23]:
    axes[1].annotate(f'{h}:00',
                    xy=(np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold')
axes[1].set_xlabel('sin(hour)', fontsize=11)
axes[1].set_ylabel('cos(hour)', fontsize=11)
axes[1].set_title('Cyclical Encoding Solution\n(23:00 close to 00:00!)',
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')
plt.colorbar(scatter, ax=axes[1], label='Hour')

# Plot 3: Distance comparison
linear_dist_23_0 = abs(23 - 0)
cyclical_dist_23_0 = np.sqrt((np.sin(2*np.pi*23/24) - np.sin(0))**2 +
                             (np.cos(2*np.pi*23/24) - np.cos(0))**2)

labels = ['Linear\n(wrong)', 'Cyclical\n(correct)']
distances = [linear_dist_23_0, cyclical_dist_23_0]
colors = ['red', 'green']
axes[2].bar(labels, distances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[2].set_ylabel('Distance between 23:00 and 00:00', fontsize=11)
axes[2].set_title('Distance Comparison\n(Lower is better)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("🎯 Key Insight:")
print(f"   Linear encoding: distance(23:00 → 00:00) = {linear_dist_23_0:.2f} (WRONG!)")
print(f"   Cyclical encoding: distance(23:00 → 00:00) = {cyclical_dist_23_0:.2f} (Correct!)")
print(f"   \n   Cyclical encoding correctly captures that 23:00 and 00:00 are close in time!")
```

**Add Explanation:**
```markdown
## Why Cyclical Encoding Works

**The Problem**: Time is cyclical (23:59 → 00:00), but linear encoding treats it as:
- 23:00 = 23 (far from)
- 00:00 = 0

**The Solution**: Map to circle using sine and cosine:
- Points close in time → close on circle
- Preserves cyclical relationship
- Works for any cyclical feature (months, days of week, wind direction, etc.)
```

**Testing Checklist:**
- [ ] Visualization clearly shows the problem and solution
- [ ] Circle plot is aesthetically pleasing
- [ ] Distance comparison is compelling
- [ ] User approval obtained

---

## Phase 3: Pedagogical Enhancements 🟢

### Task 10-14: [Details would continue...]

---

## Phase 4: Final Polish ✨

### Task 15-20: [Details would continue...]

---

## Progress Tracking

**Phase 1**: 0/4 tasks complete (0%)
**Phase 2**: 0/5 tasks complete (0%)
**Phase 3**: 0/5 tasks complete (0%)
**Phase 4**: 0/6 tasks complete (0%)

**Overall Progress**: 0/20 tasks complete (0%)

---

## Notes & Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Nov 2025 | Created task tracker | Need detailed implementation notes |
| TBD | User chooses timeline | Waiting for user input |
| TBD | User chooses Featuretools option | Task 4 decision needed |

---

**Next Action**: Await user approval to begin Phase 1
