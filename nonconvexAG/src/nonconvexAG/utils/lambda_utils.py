"""Lambda parameter utilities."""

from typing import Optional, Union
import numpy as np
from numba import jit
import numpy.typing as npt

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def lambda_max_LM(X: ArrayLike, y: ArrayLike, 
                  penalty: str = "SCAD", 
                  a: float = 3.7, 
                  gamma: float = 2.0) -> float:
    """Calculate lambda_max for linear models.
    
    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Design matrix with intercept column.
    y : array_like of shape (n_samples,)
        Response vector.
    penalty : {'SCAD', 'MCP'}, default='SCAD'
        Penalty type.
    a : float, default=3.7
        SCAD parameter.
    gamma : float, default=2.0
        MCP parameter.
        
    Returns
    -------
    float
        Lambda max value.
    """
    # Ensure arrays are float64 for numba
    X_float = X.astype(np.float64)
    y_float = y.astype(np.float64)
    
    n = X_float.shape[0]
    
    # Calculate gradient at beta = 0 (only intercept)
    intercept = np.mean(y_float)
    residuals = y_float - intercept
    gradients = np.abs(X_float[:, 1:].T @ residuals) / n
    
    # For both SCAD and MCP, the initial slope is lambda
    lambda_max = np.max(gradients)
    
    # Add small epsilon for numerical stability
    return lambda_max * 1.0001


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def lambda_max_logistic(X: ArrayLike, y: ArrayLike,
                        penalty: str = "SCAD",
                        a: float = 3.7,
                        gamma: float = 2.0) -> float:
    """Calculate minimum lambda to set all penalized coefficients to zero for logistic regression.
    
    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Design matrix including intercept column.
    y : array_like of shape (n_samples,)
        Binary response vector (0 or 1).
    penalty : {'SCAD', 'MCP'}, default='SCAD'
        Type of nonconvex penalty.
    a : float, default=3.7
        SCAD parameter (ignored for MCP).
    gamma : float, default=2.0
        MCP parameter (ignored for SCAD).
        
    Returns
    -------
    float
        Minimum lambda value.
    """
    # Ensure arrays are float64 for numba
    X_float = X.astype(np.float64)
    y_float = y.astype(np.float64)
    
    n = X_float.shape[0]
    
    # Calculate gradient at beta = 0 (only intercept)
    # For logistic regression: intercept = log(p/(1-p)) where p = mean(y)
    p_bar = np.mean(y_float)
    
    # Avoid numerical issues
    if p_bar <= 0:
        p_bar = 0.001
    elif p_bar >= 1:
        p_bar = 0.999
        
    # Predicted probabilities with only intercept
    pred_probs = np.full(n, p_bar)
    
    # Gradient of log-likelihood
    residuals = y_float - pred_probs
    gradients = np.abs(X_float[:, 1:].T @ residuals) / n
    
    lambda_max = np.max(gradients)
    
    # Add small epsilon for numerical stability
    return lambda_max * 1.0001


def generate_lambda_sequence(lambda_max: float, 
                           lambda_min_ratio: float = 0.01,
                           n_lambda: int = 100,
                           log_scale: bool = True) -> np.ndarray:
    """Generate a sequence of lambda values for solution path.
    
    Parameters
    ----------
    lambda_max : float
        Maximum lambda value (typically from lambda_max_LM or lambda_max_logistic).
    lambda_min_ratio : float, default=0.01
        Ratio of minimum to maximum lambda.
    n_lambda : int, default=100
        Number of lambda values.
    log_scale : bool, default=True
        Whether to use log-scale spacing.
        
    Returns
    -------
    np.ndarray
        Decreasing sequence of lambda values.
    """
    lambda_min = lambda_max * lambda_min_ratio
    
    if log_scale:
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)
    else:
        lambdas = np.linspace(lambda_max, lambda_min, n_lambda)
        
    return lambdas