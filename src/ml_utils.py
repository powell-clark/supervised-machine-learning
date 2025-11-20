"""
Machine Learning Utility Functions

Shared functions for validation, metrics, and common operations.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
import warnings


def validate_input(
    X: NDArray,
    y: Optional[NDArray] = None,
    allow_nan: bool = False,
    require_2d: bool = True
) -> bool:
    """
    Validate input data for machine learning models.

    Args:
        X: Feature matrix
        y: Target vector (optional)
        allow_nan: Whether to allow NaN values
        require_2d: Whether X must be 2D

    Returns:
        True if validation passes

    Raises:
        TypeError: If input is not numpy array
        ValueError: If validation fails
    """
    # Check types
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be numpy array, got {type(X)}")

    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError(f"y must be numpy array, got {type(y)}")

    # Check dimensions
    if require_2d and X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")

    if y is not None and y.ndim > 2:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")

    # Check for NaN/Inf
    if not allow_nan:
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")

        if y is not None and (np.any(np.isnan(y)) or np.any(np.isinf(y))):
            raise ValueError("y contains NaN or Inf values")

    # Check shape compatibility
    if y is not None:
        if y.ndim == 1:
            if X.shape[0] != len(y):
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X: {X.shape[0]}, y: {len(y)}"
                )
        else:  # y.ndim == 2
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X: {X.shape[0]}, y: {y.shape[0]}"
                )

    # Check for empty arrays
    if X.size == 0:
        raise ValueError("X is empty")

    if y is not None and y.size == 0:
        raise ValueError("y is empty")

    return True


def compute_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    task: str = "regression"
) -> Dict[str, float]:
    """
    Compute common evaluation metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        task: "regression" or "classification"

    Returns:
        Dictionary of metric names and values
    """
    validate_input(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))

    if task == "regression":
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        # R² score with division-by-zero handling
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

    elif task == "classification":
        accuracy = np.mean(y_true == y_pred)

        return {
            "accuracy": float(accuracy)
        }

    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")


def safe_matrix_inverse(
    matrix: NDArray,
    method: str = "auto",
    cond_threshold: float = 1e10
) -> Tuple[NDArray, bool]:
    """
    Safely compute matrix inverse with fallback to pseudo-inverse.

    Args:
        matrix: Square matrix to invert
        method: "auto", "inverse", or "pinv"
        cond_threshold: Condition number threshold for ill-conditioned matrices

    Returns:
        Tuple of (inverted matrix, used_pinv flag)
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}")

    used_pinv = False

    if method == "pinv":
        inv_matrix = np.linalg.pinv(matrix)
        used_pinv = True

    elif method == "inverse":
        try:
            inv_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Matrix is singular and cannot be inverted. "
                f"Try method='auto' or method='pinv'. Original error: {e}"
            )

    else:  # method == "auto"
        try:
            # Check condition number
            cond_num = np.linalg.cond(matrix)

            if cond_num > cond_threshold:
                warnings.warn(
                    f"Matrix is ill-conditioned (cond={cond_num:.2e}). "
                    f"Using pseudo-inverse.",
                    RuntimeWarning
                )
                inv_matrix = np.linalg.pinv(matrix)
                used_pinv = True
            else:
                # Standard inverse
                inv_matrix = np.linalg.inv(matrix)

        except np.linalg.LinAlgError:
            warnings.warn(
                "Matrix is singular. Using pseudo-inverse.",
                RuntimeWarning
            )
            inv_matrix = np.linalg.pinv(matrix)
            used_pinv = True

    return inv_matrix, used_pinv


def check_convergence(
    losses: list,
    patience: int = 5,
    min_delta: float = 1e-4,
    check_nan: bool = True
) -> Tuple[bool, str]:
    """
    Check if training has converged or diverged.

    Args:
        losses: List of loss values
        patience: Number of epochs without improvement
        min_delta: Minimum change to qualify as improvement
        check_nan: Whether to check for NaN/Inf

    Returns:
        Tuple of (should_stop, reason)
    """
    if len(losses) == 0:
        return False, ""

    current_loss = losses[-1]

    # Check for NaN/Inf
    if check_nan and (np.isnan(current_loss) or np.isinf(current_loss)):
        return True, f"Loss became {current_loss} (training diverged)"

    # Need at least patience+1 losses to check for early stopping
    if len(losses) <= patience:
        return False, ""

    # Check if loss hasn't improved in last 'patience' epochs
    recent_losses = losses[-(patience + 1):]
    best_recent = min(recent_losses[:-1])

    if current_loss > best_recent - min_delta:
        best_overall = min(losses)
        best_epoch = losses.index(best_overall) + 1
        return True, f"No improvement for {patience} epochs. Best loss: {best_overall:.4f} at epoch {best_epoch}"

    return False, ""


def clip_gradients(
    gradients: Dict[str, NDArray],
    max_norm: float = 5.0
) -> Dict[str, NDArray]:
    """
    Clip gradients by global norm to prevent exploding gradients.

    Args:
        gradients: Dictionary of gradient arrays
        max_norm: Maximum norm for gradients

    Returns:
        Clipped gradients
    """
    # Compute global norm
    total_norm = 0.0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        gradients = {k: v * clip_coef for k, v in gradients.items()}

    return gradients


def train_test_split_stratified(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Split data into train/test with stratification for classification.

    Args:
        X: Features
        y: Labels
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    validate_input(X, y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)

    train_indices = []
    test_indices = []

    # Stratified sampling
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_cls_test = max(1, int(len(cls_indices) * test_size))

        # Shuffle and split
        np.random.shuffle(cls_indices)
        test_indices.extend(cls_indices[:n_cls_test])
        train_indices.extend(cls_indices[n_cls_test:])

    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )
