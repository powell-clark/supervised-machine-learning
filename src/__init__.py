"""
Supervised Machine Learning Utilities

This package contains shared utility functions used across notebooks.
"""

__version__ = "1.0.0"

from .ml_utils import (
    validate_input,
    compute_metrics,
    safe_matrix_inverse,
    check_convergence,
)

__all__ = [
    "validate_input",
    "compute_metrics",
    "safe_matrix_inverse",
    "check_convergence",
]
