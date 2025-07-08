"""Logistic regression model implementation."""

from typing import Union
import numpy as np
from numba import jit
import numpy.typing as npt

from .base import BaseModel

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


class LogisticModel(BaseModel):
    """Logistic regression model for use with nonconvex penalties.
    
    Implements the negative log-likelihood loss and gradient for:
        L(β) = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
    where pᵢ = 1 / (1 + exp(-xᵢ'β))
    """
    
    def __init__(self):
        """Initialize logistic model."""
        super().__init__()
        
    def loss(self, X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> float:
        """Calculate negative log-likelihood loss.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        y : array_like of shape (n_samples,)
            Binary response vector (0 or 1).
        beta : array_like of shape (n_features,)
            Parameter vector.
            
        Returns
        -------
        float
            Negative log-likelihood loss.
        """
        n = X.shape[0]
        linear_pred = X @ beta
        
        # Clip to prevent overflow in exp
        linear_pred = np.clip(linear_pred, -500, 500)
        
        # Negative log-likelihood
        loss = np.sum(np.log(1 + np.exp(-linear_pred))) + np.sum((1 - y) * linear_pred)
        return loss / n
        
    def gradient(self, X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> np.ndarray:
        """Calculate gradient of negative log-likelihood.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        y : array_like of shape (n_samples,)
            Binary response vector (0 or 1).
        beta : array_like of shape (n_features,)
            Parameter vector.
            
        Returns
        -------
        np.ndarray of shape (n_features,)
            Gradient vector: (1/n) * X'(p - y)
        """
        n = X.shape[0]
        linear_pred = X @ beta
        
        # Compute predicted probabilities
        prob = self._sigmoid(linear_pred)
        
        # Gradient
        return X.T @ (prob - y) / n
        
    def predict_proba(self, X: ArrayLike, beta: ArrayLike) -> np.ndarray:
        """Predict class probabilities.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        beta : array_like of shape (n_features,)
            Parameter vector.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted probabilities for class 1.
        """
        linear_pred = X @ beta
        return self._sigmoid(linear_pred)
        
    def predict(self, X: ArrayLike, beta: ArrayLike) -> np.ndarray:
        """Predict classes.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        beta : array_like of shape (n_features,)
            Parameter vector.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted classes (0 or 1).
        """
        prob = self.predict_proba(X, beta)
        return (prob >= 0.5).astype(int)
        
    def _compute_intercept(self, y: ArrayLike) -> float:
        """Compute initial intercept using log odds.
        
        Parameters
        ----------
        y : array_like of shape (n_samples,)
            Binary response vector.
            
        Returns
        -------
        float
            Log odds: log(p/(1-p)) where p = mean(y).
        """
        p = np.mean(y)
        # Clip to avoid log(0)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))
        
    def _sigmoid(self, z: ArrayLike) -> np.ndarray:
        """Stable sigmoid function.
        
        Parameters
        ----------
        z : array_like
            Linear predictions.
            
        Returns
        -------
        np.ndarray
            Sigmoid values.
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


# Numba-optimized functions for performance
@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def logistic_loss_jit(X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> float:
    """JIT-compiled negative log-likelihood loss."""
    n = X.shape[0]
    linear_pred = X @ beta
    
    # Clip to prevent overflow
    for i in range(len(linear_pred)):
        if linear_pred[i] > 500:
            linear_pred[i] = 500
        elif linear_pred[i] < -500:
            linear_pred[i] = -500
    
    loss = 0.0
    for i in range(n):
        loss += np.log(1 + np.exp(-linear_pred[i])) + (1 - y[i]) * linear_pred[i]
    
    return loss / n


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def sigmoid_jit(z: ArrayLike) -> np.ndarray:
    """JIT-compiled stable sigmoid function."""
    # Convert to array and flatten to handle scalars
    z_flat = np.atleast_1d(z).astype(np.float64)
    result = np.empty_like(z_flat, dtype=np.float64)
    
    # Process each element
    for i in range(z_flat.shape[0]):
        if z_flat[i] > 500:
            result[i] = 1.0
        elif z_flat[i] < -500:
            result[i] = 0.0
        else:
            result[i] = 1.0 / (1.0 + np.exp(-z_flat[i]))
    
    return result


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def logistic_gradient_jit(X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> np.ndarray:
    """JIT-compiled gradient of negative log-likelihood."""
    n = X.shape[0]
    linear_pred = X @ beta
    prob = sigmoid_jit(linear_pred)
    return X.T @ (prob - y) / n


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_convex_logistic(smooth_grad: ArrayLike,
                                      X: ArrayLike,
                                      y: ArrayLike,
                                      beta: ArrayLike) -> np.ndarray:
    """Update gradient for the smooth convex component of logistic model.
    
    Parameters
    ----------
    smooth_grad : array_like
        Current gradient (will be modified in-place).
    X : array_like
        Design matrix.
    y : array_like
        Binary response vector.
    beta : array_like
        Current parameter vector.
        
    Returns
    -------
    np.ndarray
        Updated gradient.
    """
    n = X.shape[0]
    linear_pred = X @ beta
    prob = sigmoid_jit(linear_pred)
    smooth_grad[:] = X.T @ (prob - y) / n
    return smooth_grad