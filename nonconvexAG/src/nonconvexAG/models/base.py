"""Base class for models."""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import numpy as np
import numpy.typing as npt

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


class BaseModel(ABC):
    """Base class for regression models."""
    
    def __init__(self):
        pass
        self.n_samples: Optional[int] = None
        self.n_features: Optional[int] = None
        self.L_convex: Optional[float] = None
        
    @abstractmethod
    def loss(self, X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> float:
        """Calculate loss."""
        pass
        
    @abstractmethod
    def gradient(self, X: ArrayLike, y: ArrayLike, beta: ArrayLike) -> np.ndarray:
        """Calculate gradient."""
        pass
        
    @abstractmethod
    def predict(self, X: ArrayLike, beta: ArrayLike) -> np.ndarray:
        """Make predictions."""
        pass
        
    def calculate_L_convex(self, X: ArrayLike) -> float:
        """Calculate L-smoothness constant."""
        n = X.shape[0]
        # Largest eigenvalue of X'X/n
        if hasattr(np.linalg, 'eigvalsh'):
            L = np.linalg.eigvalsh(X.T @ X / n)[-1]
        else:
            L = np.linalg.eigvals(X.T @ X / n).real.max()
        return L
        
    def init_beta(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Initialize parameters."""
        # Default: zeros except for intercept
        beta = np.zeros(X.shape[1])
        beta[0] = self._compute_intercept(y)
        return beta
        
    @abstractmethod
    def _compute_intercept(self, y: ArrayLike) -> float:
        """Compute initial intercept value.
        
        Parameters
        ----------
        y : array_like of shape (n_samples,)
            Response vector.
            
        Returns
        -------
        float
            Initial intercept.
        """
        pass