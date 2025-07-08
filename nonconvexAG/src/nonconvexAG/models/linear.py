"""Linear regression model implementation."""

from typing import Union
import numpy as np
from numba import jit
import numpy.typing as npt

from .base import BaseModel

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


class LinearModel(BaseModel):
    """Linear regression model for use with nonconvex penalties.
    
    Implements the loss function and gradient for:
        L(β) = (1/2n) ||y - Xβ||²₂
    """
    
    def __init__(self):
        """Initialize linear model."""
        super().__init__()
        
    def loss(self, X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> float:
        """Calculate squared error loss.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        y : array_like of shape (n_samples,)
            Response vector.
        beta : array_like of shape (n_features,)
            Parameter vector.
            
        Returns
        -------
        float
            Loss value: (1/2n) * ||y - Xβ||²
        """
        n = X.shape[0]
        residuals = y - X @ beta
        return 0.5 * np.sum(residuals**2) / n
        
    def gradient(self, X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> np.ndarray:
        """Calculate gradient of squared error loss.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        y : array_like of shape (n_samples,)
            Response vector.
        beta : array_like of shape (n_features,)
            Parameter vector.
            
        Returns
        -------
        np.ndarray of shape (n_features,)
            Gradient vector: -(1/n) * X'(y - Xβ)
        """
        n = X.shape[0]
        residuals = y - X @ beta
        return -X.T @ residuals / n
        
    def predict(self, X: ArrayLike, beta: ArrayLike) -> np.ndarray:
        """Make predictions.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        beta : array_like of shape (n_features,)
            Parameter vector.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted values: Xβ
        """
        return X @ beta
        
    def _compute_intercept(self, y: ArrayLike) -> float:
        """Compute initial intercept as mean of y.
        
        Parameters
        ----------
        y : array_like of shape (n_samples,)
            Response vector.
            
        Returns
        -------
        float
            Mean of y.
        """
        return np.mean(y)


# Numba-optimized functions for performance
@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def linear_loss_jit(X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> float:
    """JIT-compiled squared error loss."""
    n = X.shape[0]
    residuals = y - X @ beta
    return 0.5 * np.sum(residuals**2) / n


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def linear_gradient_jit(X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> np.ndarray:
    """JIT-compiled gradient of squared error loss."""
    n = X.shape[0]
    residuals = y - X @ beta
    return -X.T @ residuals / n


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_convex_LM(smooth_grad: ArrayLike, 
                                X: ArrayLike, 
                                y: ArrayLike,
                                beta: ArrayLike) -> np.ndarray:
    """Update gradient for the smooth convex component of linear model.
    
    Parameters
    ----------
    smooth_grad : array_like
        Current gradient (will be modified in-place).
    X : array_like
        Design matrix.
    y : array_like
        Response vector.
    beta : array_like
        Current parameter vector.
        
    Returns
    -------
    np.ndarray
        Updated gradient.
    """
    n = X.shape[0]
    smooth_grad[:] = X.T @ (X @ beta - y) / n
    return smooth_grad


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def eval_obj_linear(X: ArrayLike, y: ArrayLike, beta: ArrayLike,
                   lambda_: float, penalty_func) -> float:
    """Evaluate objective function for linear model with penalty.
    
    Parameters
    ----------
    X : array_like
        Design matrix.
    y : array_like
        Response vector.
    beta : array_like
        Parameter vector.
    lambda_ : float
        Regularization parameter.
    penalty_func : callable
        Penalty function (SCAD or MCP).
        
    Returns
    -------
    float
        Objective value: loss + penalty
    """
    loss = linear_loss_jit(X, y, beta)
    penalty = penalty_func(beta, lambda_)
    return loss + penalty