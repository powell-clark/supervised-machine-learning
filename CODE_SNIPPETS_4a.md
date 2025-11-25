# Ready-to-Use Code Snippets for Lesson 4a (SVM Theory)

**Purpose**: Copy-paste these code cells directly into `notebooks/4a_svm_theory.ipynb`

**Usage**: Each section below is a complete, tested code cell ready for insertion.

---

## Section 3: The Margin Concept

### Cell: "Visualizing the Margin"

```python
"""
## The Margin Concept: Geometric Intuition

The margin is the perpendicular distance from the separating hyperplane
to the nearest training points on either side.

For a hyperplane defined by w·x + b = 0:
- Points with w·x + b > 0 are on one side (predict +1)
- Points with w·x + b < 0 are on the other side (predict -1)
- The margin width is 2/||w||

Why? The decision boundary is w·x + b = 0
The parallel hyperplanes are:
- w·x + b = +1 (margin for positive class)
- w·x + b = -1 (margin for negative class)

Distance between these parallel hyperplanes = 2/||w||
"""

# Create a simple 2D dataset
np.random.seed(42)
X_positive = np.random.randn(20, 2) + [2, 2]
X_negative = np.random.randn(20, 2) + [-2, -2]
X_toy = np.vstack([X_positive, X_negative])
y_toy = np.array([1]*20 + [-1]*20)

# Define three different hyperplanes with different ||w||
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

hyperplanes = [
    {'w': np.array([0.5, 0.5]), 'b': 0, 'name': 'Small ||w||'},
    {'w': np.array([1.0, 1.0]), 'b': 0, 'name': 'Medium ||w||'},
    {'w': np.array([2.0, 2.0]), 'b': 0, 'name': 'Large ||w||'}
]

for idx, (hp, ax) in enumerate(zip(hyperplanes, axes)):
    w = hp['w']
    b = hp['b']
    w_norm = np.linalg.norm(w)
    margin_width = 2 / w_norm

    # Plot data points
    ax.scatter(X_toy[y_toy==1, 0], X_toy[y_toy==1, 1],
               c='red', s=100, alpha=0.6, edgecolors='darkred',
               linewidth=2, label='Class +1')
    ax.scatter(X_toy[y_toy==-1, 0], X_toy[y_toy==-1, 1],
               c='blue', s=100, alpha=0.6, edgecolors='darkblue',
               linewidth=2, label='Class -1')

    # Create grid for decision boundary visualization
    x_min, x_max = X_toy[:, 0].min() - 1, X_toy[:, 0].max() + 1
    y_min, y_max = X_toy[:, 1].min() - 1, X_toy[:, 1].max() + 1

    # Plot decision boundary: w·x + b = 0
    # Rewrite as x2 = -(w1*x1 + b)/w2
    x1_boundary = np.array([x_min, x_max])
    x2_boundary = -(w[0] * x1_boundary + b) / w[1]
    ax.plot(x1_boundary, x2_boundary, 'k-', linewidth=3, label='Decision Boundary')

    # Plot margin boundaries: w·x + b = ±1
    x2_margin_pos = -(w[0] * x1_boundary + b - 1) / w[1]
    x2_margin_neg = -(w[0] * x1_boundary + b + 1) / w[1]
    ax.plot(x1_boundary, x2_margin_pos, 'r--', linewidth=2, alpha=0.7, label='Margin (+1)')
    ax.plot(x1_boundary, x2_margin_neg, 'b--', linewidth=2, alpha=0.7, label='Margin (-1)')

    # Fill margin region
    ax.fill_between(x1_boundary, x2_margin_neg, x2_margin_pos,
                     alpha=0.1, color='green')

    # Add text annotations
    ax.text(0, 4.5, f'||w|| = {w_norm:.2f}', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(0, 3.8, f'Margin = 2/||w|| = {margin_width:.2f}', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'{hp["name"]}: ||w|| = {w_norm:.2f}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

plt.suptitle('Margin Width = 2/||w||: Smaller ||w|| → Wider Margin',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("✅ Key Insight: To maximize the margin, we minimize ||w||!")
print(f"\n📊 Margin widths:")
for hp in hyperplanes:
    w_norm = np.linalg.norm(hp['w'])
    print(f"  {hp['name']}: margin = {2/w_norm:.3f}")
```

### Cell: "Mathematical Definition of Margin"

```python
"""
## Mathematical Definition

For a point x with label y ∈ {-1, +1}, we define:

**Functional Margin** (unnormalized):
    γ̂ = y(w·x + b)

If γ̂ > 0: correctly classified
If γ̂ >> 0: confidently classified
If γ̂ < 0: misclassified

**Geometric Margin** (actual distance):
    γ = γ̂ / ||w|| = y(w·x + b) / ||w||

This is the perpendicular distance from point x to the hyperplane w·x + b = 0

**Why divide by ||w||?**
The distance from point x₀ to hyperplane w·x + b = 0 is:
    distance = |w·x₀ + b| / ||w||

Proof:
- Let x_proj be the projection of x₀ onto the hyperplane
- Then x₀ = x_proj + t·(w/||w||) for some scalar t
- Since x_proj is on the hyperplane: w·x_proj + b = 0
- Substituting: w·(x₀ - t·w/||w||) + b = 0
- w·x₀ - t·||w|| + b = 0
- t = (w·x₀ + b) / ||w||
- Distance = |t| = |w·x₀ + b| / ||w|| ✓
"""

# Worked example with actual numbers
print("Worked Example: Computing Margins")
print("="*60)

# Define hyperplane
w_example = np.array([3, 4])  # Normal vector
b_example = -1  # Bias
w_norm_example = np.linalg.norm(w_example)  # ||w|| = 5

print(f"\nHyperplane: {w_example[0]}x₁ + {w_example[1]}x₂ + {b_example} = 0")
print(f"||w|| = {w_norm_example}")

# Test point 1: x = [2, 1], y = +1
x1 = np.array([2, 1])
y1 = 1
functional_margin_1 = y1 * (np.dot(w_example, x1) + b_example)
geometric_margin_1 = functional_margin_1 / w_norm_example

print(f"\nPoint 1: x = {x1}, y = {y1}")
print(f"  w·x + b = {w_example[0]}·{x1[0]} + {w_example[1]}·{x1[1]} + {b_example}")
print(f"           = {np.dot(w_example, x1) + b_example}")
print(f"  Functional margin: γ̂ = {y1} × {np.dot(w_example, x1) + b_example} = {functional_margin_1}")
print(f"  Geometric margin:  γ = {functional_margin_1} / {w_norm_example} = {geometric_margin_1:.3f}")
print(f"  Status: {'✅ Correctly classified' if functional_margin_1 > 0 else '❌ Misclassified'}")

# Test point 2: x = [-1, 1], y = -1
x2 = np.array([-1, 1])
y2 = -1
functional_margin_2 = y2 * (np.dot(w_example, x2) + b_example)
geometric_margin_2 = functional_margin_2 / w_norm_example

print(f"\nPoint 2: x = {x2}, y = {y2}")
print(f"  w·x + b = {w_example[0]}·{x2[0]} + {w_example[1]}·{x2[1]} + {b_example}")
print(f"           = {np.dot(w_example, x2) + b_example}")
print(f"  Functional margin: γ̂ = {y2} × {np.dot(w_example, x2) + b_example} = {functional_margin_2}")
print(f"  Geometric margin:  γ = {functional_margin_2} / {w_norm_example} = {geometric_margin_2:.3f}")
print(f"  Status: {'✅ Correctly classified' if functional_margin_2 > 0 else '❌ Misclassified'}")

print("\n" + "="*60)
print("💡 Key Insight: Geometric margin is scale-invariant!")
print("   Multiplying w by constant doesn't change geometric margin")
```

---

## Section 4: Primal Formulation

### Cell: "The Optimization Problem"

```python
"""
## Primal Formulation: The Optimization Problem

Goal: Find hyperplane w·x + b = 0 that maximizes the margin

Since margin = 2/||w||:
- Maximizing 2/||w|| ↔ Minimizing ||w|| ↔ Minimizing ||w||²/2

**Complete Primal Problem:**

    minimize: (1/2)||w||²

    subject to: yᵢ(w·xᵢ + b) ≥ 1  for all i = 1,...,n

Why ≥ 1 and not ≥ 0?
- We fix the functional margin at 1 (normalization)
- This makes the problem well-defined
- Still captures the maximum margin solution

**Why is this convex?**

Objective function: f(w) = (1/2)||w||² = (1/2)w^T w
- This is a quadratic function
- Hessian: ∇²f = I (identity matrix)
- I is positive definite → f is strictly convex ✓

Constraints: gᵢ(w,b) = yᵢ(w·xᵢ + b) - 1 ≥ 0
- These are affine (linear) functions
- Affine functions are convex ✓

Convex objective + convex constraints = **global optimum guaranteed!**
"""

# Visualization: Convexity of ||w||²
w_vals = np.linspace(-3, 3, 100)
objective_1d = 0.5 * w_vals**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1D case
axes[0].plot(w_vals, objective_1d, 'b-', linewidth=3)
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Global minimum')
axes[0].scatter([0], [0], c='red', s=200, zorder=5, edgecolors='darkred', linewidth=2)
axes[0].set_xlabel('w', fontsize=12)
axes[0].set_ylabel('(1/2)w²', fontsize=12)
axes[0].set_title('Convex Objective: (1/2)||w||² in 1D', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 2D case
w1_vals = np.linspace(-3, 3, 50)
w2_vals = np.linspace(-3, 3, 50)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Objective_2d = 0.5 * (W1**2 + W2**2)

contour = axes[1].contour(W1, W2, Objective_2d, levels=15, cmap='viridis')
axes[1].clabel(contour, inline=True, fontsize=8)
axes[1].scatter([0], [0], c='red', s=200, zorder=5,
                marker='*', edgecolors='darkred', linewidth=2,
                label='Global minimum at w=[0,0]')
axes[1].set_xlabel('w₁', fontsize=12)
axes[1].set_ylabel('w₂', fontsize=12)
axes[1].set_title('Convex Objective: (1/2)||w||² in 2D', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Convex optimization guarantees we'll find the global optimum!")
print("   No local minima to get stuck in.")
```

---

## Section 5: Lagrangian Dual Formulation

### Cell: "Lagrangian Setup and KKT Conditions"

```python
"""
## The Dual Problem: Why and How

**Why go to the dual?**

1. **Computational efficiency**: When n_features >> n_samples, dual is faster
2. **Kernel trick**: Only inner products xᵢ·xⱼ appear in dual
3. **Sparsity**: Many αᵢ = 0, only support vectors matter

**Lagrangian Function:**

L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w·xᵢ + b) - 1]

where α = [α₁, ..., αₙ] are Lagrange multipliers, αᵢ ≥ 0

**Karush-Kuhn-Tucker (KKT) Conditions:**

For optimal (w*, b*, α*), we need:

1. **Stationarity**: ∇w L = 0, ∂L/∂b = 0
2. **Primal feasibility**: yᵢ(w·xᵢ + b) ≥ 1 for all i
3. **Dual feasibility**: αᵢ ≥ 0 for all i
4. **Complementary slackness**: αᵢ[yᵢ(w·xᵢ + b) - 1] = 0 for all i

**Deriving the Dual:**

Step 1: Stationarity wrt w
    ∇w L = w - Σᵢ αᵢyᵢxᵢ = 0

    ⟹ w* = Σᵢ αᵢyᵢxᵢ  ← **w expressed in terms of training data!**

Step 2: Stationarity wrt b
    ∂L/∂b = -Σᵢ αᵢyᵢ = 0

    ⟹ Σᵢ αᵢyᵢ = 0  ← **constraint on α**

Step 3: Substitute back into L(w, b, α)

    L = (1/2)||Σᵢ αᵢyᵢxᵢ||² - Σᵢ αᵢ[yᵢ((Σⱼ αⱼyⱼxⱼ)·xᵢ + b) - 1]

    = (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ) - ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ) - b·Σᵢαᵢyᵢ + Σᵢ αᵢ

    = Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)  (using Σαᵢyᵢ = 0)

**Dual Problem:**

    maximize: L(α) = Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)

    subject to: αᵢ ≥ 0, Σᵢ αᵢyᵢ = 0

**Key Insight:** Only inner products xᵢ·xⱼ appear! This enables the kernel trick.
"""

# Worked example with 4 points
print("Worked Example: Dual Problem for 4-Point Dataset")
print("="*70)

# Simple linearly separable dataset
X_small = np.array([
    [1, 1],   # Point 1, class +1
    [2, 2],   # Point 2, class +1
    [3, 1],   # Point 3, class -1
    [4, 3]    # Point 4, class -1
])
y_small = np.array([1, 1, -1, -1])

n_points = len(X_small)

# Compute kernel matrix K[i,j] = xᵢ·xⱼ
K = X_small @ X_small.T

print("Dataset:")
for i, (x, label) in enumerate(zip(X_small, y_small)):
    print(f"  Point {i+1}: x = {x}, y = {label:+d}")

print(f"\nKernel Matrix K (xᵢ·xⱼ):")
print(K)

print(f"\nDual Problem:")
print("maximize: Σαᵢ - (1/2)ΣΣ αᵢαⱼyᵢyⱼK[i,j]")
print("subject to: αᵢ ≥ 0, Σαᵢyᵢ = 0")

print(f"\nExpanded form:")
print("maximize: α₁ + α₂ + α₃ + α₄")
print("         - (1/2)[")
for i in range(n_points):
    for j in range(n_points):
        term = f"α{i+1}α{j+1}·({y_small[i]:+d})·({y_small[j]:+d})·{K[i,j]:.0f}"
        if j < n_points - 1:
            term += " + "
        elif i < n_points - 1:
            term += " +"
        print(f"           {term}")

print("         ]")
print("subject to:")
print(f"  α₁ + α₂ - α₃ - α₄ = 0")
print(f"  α₁, α₂, α₃, α₄ ≥ 0")

print("\n✅ This is a quadratic programming problem we can solve!")
```

---

## Usage Instructions

1. **Copy entire cell** from above (including docstring and code)
2. **Paste into notebook** at appropriate section
3. **Run cell** to verify it works
4. **Iterate**: Modify as needed for your specific lesson flow

**Next snippets to add**:
- Kernel trick visualization (3D transformation)
- Soft margin C parameter comparison
- Complete SVMFromScratch class
- Real-world application on Breast Cancer dataset

**See**: CONTENT_RESTORATION_PLAN.md for complete templates
