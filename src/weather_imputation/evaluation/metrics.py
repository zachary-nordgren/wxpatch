"""Evaluation metrics for imputation quality using PyTorch tensors.

All metrics follow the convention:
- Input tensors have shape (N, T, V) where N=samples, T=timesteps, V=variables
- mask tensor: True=evaluate this position, False=ignore
- Metrics compute only on masked positions (typically the synthetic gaps)
"""

import torch
from torch import Tensor


def compute_rmse(y_true: Tensor, y_pred: Tensor, mask: Tensor) -> float:
    """Compute Root Mean Squared Error on masked positions only.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate, False=ignore

    Returns:
        RMSE value (scalar float). Returns NaN if no valid positions.

    Example:
        >>> y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        >>> y_pred = torch.tensor([[[1.5, 2.0], [3.0, 5.0]]])
        >>> mask = torch.tensor([[[True, False], [False, True]]])
        >>> rmse = compute_rmse(y_true, y_pred, mask)
        >>> # Only positions (0,0,0) and (0,1,1) are evaluated
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool dtype, got {mask.dtype}")

    # Select only masked positions
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if y_true_masked.numel() == 0:
        return float("nan")

    # Compute RMSE
    squared_errors = (y_true_masked - y_pred_masked) ** 2
    mse = squared_errors.mean()
    rmse_value = torch.sqrt(mse)

    return rmse_value.item()


def compute_mae(y_true: Tensor, y_pred: Tensor, mask: Tensor) -> float:
    """Compute Mean Absolute Error on masked positions only.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate, False=ignore

    Returns:
        MAE value (scalar float). Returns NaN if no valid positions.
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool dtype, got {mask.dtype}")

    # Select only masked positions
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if y_true_masked.numel() == 0:
        return float("nan")

    # Compute MAE
    absolute_errors = torch.abs(y_true_masked - y_pred_masked)
    mae_value = absolute_errors.mean()

    return mae_value.item()


def compute_bias(y_true: Tensor, y_pred: Tensor, mask: Tensor) -> float:
    """Compute mean bias (systematic error) on masked positions only.

    Bias = mean(y_pred - y_true)
    Positive bias means predictions are systematically too high.
    Negative bias means predictions are systematically too low.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate, False=ignore

    Returns:
        Bias value (scalar float). Returns NaN if no valid positions.
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool dtype, got {mask.dtype}")

    # Select only masked positions
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if y_true_masked.numel() == 0:
        return float("nan")

    # Compute bias
    errors = y_pred_masked - y_true_masked
    bias_value = errors.mean()

    return bias_value.item()


def compute_r2_score(y_true: Tensor, y_pred: Tensor, mask: Tensor) -> float:
    """Compute R² (coefficient of determination) on masked positions only.

    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum of squared residuals
          SS_tot = total sum of squares (variance)

    R² = 1.0 means perfect predictions
    R² = 0.0 means predictions are as good as mean baseline
    R² < 0.0 means predictions are worse than mean baseline

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate, False=ignore

    Returns:
        R² score (scalar float). Returns NaN if no valid positions or zero variance.
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool dtype, got {mask.dtype}")

    # Select only masked positions
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if y_true_masked.numel() == 0:
        return float("nan")

    # Total sum of squares (variance around mean)
    mean_y_true = y_true_masked.mean()
    ss_tot = ((y_true_masked - mean_y_true) ** 2).sum()

    if ss_tot == 0:
        # Zero variance in ground truth (constant values)
        return float("nan")

    # Residual sum of squares
    ss_res = ((y_true_masked - y_pred_masked) ** 2).sum()

    # R² score
    r2 = 1 - (ss_res / ss_tot)

    return r2.item()


def compute_mse(y_true: Tensor, y_pred: Tensor, mask: Tensor) -> float:
    """Compute Mean Squared Error on masked positions only.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate, False=ignore

    Returns:
        MSE value (scalar float). Returns NaN if no valid positions.
    """
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"mask {mask.shape}"
        )

    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool dtype, got {mask.dtype}")

    # Select only masked positions
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if y_true_masked.numel() == 0:
        return float("nan")

    # Compute MSE
    squared_errors = (y_true_masked - y_pred_masked) ** 2
    mse_value = squared_errors.mean()

    return mse_value.item()


def compute_all_metrics(
    y_true: Tensor, y_pred: Tensor, mask: Tensor
) -> dict[str, float]:
    """Compute all standard point metrics on masked positions.

    This is a convenience function that computes RMSE, MAE, MSE, Bias, and R²
    in a single call.

    Args:
        y_true: Ground truth values, shape (N, T, V)
        y_pred: Predicted values, shape (N, T, V)
        mask: Evaluation mask, shape (N, T, V), True=evaluate, False=ignore

    Returns:
        Dictionary with keys: rmse, mae, mse, bias, r2
        All values are scalar floats (may be NaN if no valid positions)
    """
    return {
        "rmse": compute_rmse(y_true, y_pred, mask),
        "mae": compute_mae(y_true, y_pred, mask),
        "mse": compute_mse(y_true, y_pred, mask),
        "bias": compute_bias(y_true, y_pred, mask),
        "r2": compute_r2_score(y_true, y_pred, mask),
    }
