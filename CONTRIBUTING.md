# Contributing to Supervised Machine Learning

Thank you for your interest in contributing to this educational machine learning repository! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Style Guides](#style-guides)

## Code of Conduct

This project is intended for education. We expect all contributors to:

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn
- Assume good intentions

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Code snippets** or error messages

**Example:**
```markdown
**Title:** Linear Regression Normal Equation crashes on singular matrices

**Description:** When X.T @ X is singular, the normal equation implementation crashes
without a helpful error message.

**Steps to Reproduce:**
1. Create dataset with duplicate features
2. Run LinearRegressionNormal().fit(X, y)
3. Observe LinAlgError

**Expected:** Graceful fallback to pseudo-inverse with warning
**Actual:** Crash with cryptic error

**Environment:** Python 3.10, Ubuntu 22.04
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear use case**: Why is this enhancement useful?
- **Detailed description**: What should it do?
- **Examples**: Show expected usage
- **Alternatives considered**: Other approaches you've thought about

### Improving Documentation

Documentation improvements are highly valued:

- Fix typos or unclear explanations
- Add missing docstrings
- Improve code comments
- Add examples to notebooks
- Update README with new information

### Adding Code

Code contributions should:

- **Add educational value**: Help learners understand ML concepts
- **Include documentation**: Docstrings, comments, markdown explanations
- **Follow style guides**: PEP 8, type hints, clear naming
- **Include tests**: Verify correctness
- **Work in Colab**: All notebooks must run in Google Colab

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/supervised-machine-learning.git
cd supervised-machine-learning
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

### 5. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Coding Standards

### Python Code Style

- **Follow PEP 8**: Use `black` for formatting
- **Type hints**: Add type annotations to all functions
- **Docstrings**: Use Google or NumPy style
- **Max line length**: 100 characters (notebooks), 88 (Python files)

**Example:**
```python
from numpy.typing import NDArray
from typing import Tuple


def compute_accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)

    Returns:
        Accuracy score between 0 and 1

    Raises:
        ValueError: If y_true and y_pred have different lengths

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> compute_accuracy(y_true, y_pred)
        0.75
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")

    return float(np.mean(y_true == y_pred))
```

### Notebook Style

Notebooks should be **educational** and follow this structure:

#### 1. **Clear Introduction**
```markdown
# Lesson N: Topic Name

## Introduction

[Intuitive explanation of what the topic is and why it matters]

**What you'll learn:**
1. Core concepts
2. Mathematical foundations
3. Implementation from scratch
4. Real-world applications
```

#### 2. **Table of Contents**
```markdown
## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Theory](#theory)
3. [Implementation](#implementation)
...
```

#### 3. **Code Cells**
- Add docstrings to all functions
- Include type hints
- Add comments for complex operations
- Print informative output
- Visualize results

#### 4. **Markdown Explanations**
- Explain the "why" not just the "what"
- Use analogies and examples
- Include mathematical notation (LaTeX)
- Add warnings for common pitfalls

**Example:**
```markdown
### Gradient Descent Intuition

Think of gradient descent like hiking down a mountain in fog:

- You can't see the bottom (global minimum)
- You can only feel the slope under your feet (gradient)
- You take steps downhill (update weights)
- Smaller steps = slower but safer (learning rate)

**Mathematical formulation:**

$$w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$$

Where:
- $w$: Model weights
- $\alpha$: Learning rate (step size)
- $\frac{\partial L}{\partial w}$: Gradient (slope direction)

⚠️ **Common mistake:** Learning rate too large → overshoot minimum!
```

### Testing Standards

All new functions should have tests:

```python
# tests/test_ml_utils.py
import pytest
import numpy as np
from src.ml_utils import validate_input


def test_validate_input_valid():
    """Test validation with valid inputs."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    assert validate_input(X, y) is True


def test_validate_input_shape_mismatch():
    """Test validation catches shape mismatch."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1, 2])  # Wrong length

    with pytest.raises(ValueError, match="same number of samples"):
        validate_input(X, y)


def test_validate_input_nan():
    """Test validation catches NaN values."""
    X = np.array([[1, 2], [np.nan, 4]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="NaN"):
        validate_input(X, y, allow_nan=False)
```

## Submitting Changes

### 1. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 2. Run Linting

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy src/
```

### 3. Test Notebooks

```bash
# Test that notebooks run without errors
jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: Add cross-validation to linear regression

- Implement k-fold cross-validation
- Add example usage in notebook
- Include tests for edge cases"
```

**Commit message format:**
```
<type>: <short summary>

<detailed description>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Clear title**: Summarize the change
- **Description**: Explain what and why
- **Testing**: Describe how you tested
- **Screenshots**: If applicable (notebook outputs)
- **Checklist**:
  - [ ] Tests pass
  - [ ] Code is formatted
  - [ ] Documentation updated
  - [ ] Notebooks run in Colab

## Style Guides

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be ≤ 50 characters
- Reference issues: "Fixes #123" or "Closes #456"

### Python Naming Conventions

- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

### Mathematical Notation

Use LaTeX in markdown cells:

```markdown
**Inline math:** The loss function $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

**Display math:**
$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
```

### Comments

```python
# Good: Explains WHY
# Use log1p to handle zeros in the data
log_transformed = np.log1p(data)

# Bad: Explains WHAT (obvious from code)
# Take the log
log_transformed = np.log1p(data)
```

## Recognition

Contributors will be acknowledged in:

- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors page

Significant contributions may also be mentioned in:

- Notebook acknowledgments
- README credits

## Questions?

- **Open an issue** for questions about contributing
- **Start a discussion** for general questions
- **Email** emmanuel@powellclark.com for private inquiries

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for helping make machine learning education better! 🎓
