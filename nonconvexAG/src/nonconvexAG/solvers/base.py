"""Base solver class."""

from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, Dict, Any
import numpy as np
import numpy.typing as npt
import time

from ..utils.validation import check_convergence

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


class BaseSolver(ABC):
    """Abstract base class for optimization solvers.
    
    Parameters
    ----------
    penalty : {'SCAD', 'MCP'}
        Type of nonconvex penalty to use.
    a : float, default=3.7
        SCAD parameter (only used when penalty='SCAD').
    gamma : float, default=2.0
        MCP parameter (only used when penalty='MCP').
    tol : float, default=1e-4
        Convergence tolerance.
    maxit : int, default=1000
        Maximum number of iterations.
    verbose : bool, default=False
        Whether to print progress messages.
    """
    
    def __init__(self,
                 penalty: Literal["SCAD", "MCP"] = "SCAD",
                 a: float = 3.7,
                 gamma: float = 2.0,
                 tol: float = 1e-4,
                 maxit: int = 1000,
                 verbose: bool = False):
        """Initialize base solver."""
        self.penalty = penalty
        self.a = a
        self.gamma = gamma
        self.tol = tol
        self.maxit = maxit
        self.verbose = verbose
        
        # Runtime statistics
        self.n_iter_ = 0
        self.converged_ = False
        self.runtime_ = 0.0
        self.objective_values_ = []
        
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike, lambda_: float,
            beta_init: Optional[ArrayLike] = None) -> 'BaseSolver':
        """Fit the model.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        y : array_like of shape (n_samples,)
            Response vector.
        lambda_ : float
            Regularization parameter.
        beta_init : array_like of shape (n_features,), optional
            Initial coefficient values.
            
        Returns
        -------
        self
            Fitted solver instance.
        """
        pass
        
    def _check_convergence(self, beta_old: ArrayLike, beta_new: ArrayLike,
                          iteration: int) -> bool:
        """Check if algorithm has converged.
        
        Parameters
        ----------
        beta_old : array_like
            Previous coefficient values.
        beta_new : array_like
            Current coefficient values.
        iteration : int
            Current iteration number.
            
        Returns
        -------
        bool
            True if converged.
        """
        return check_convergence(beta_old, beta_new, self.tol, 
                               iteration, self.maxit)
                               
    def _print_progress(self, iteration: int, obj_value: float,
                       beta_diff: float) -> None:
        """Print optimization progress.
        
        Parameters
        ----------
        iteration : int
            Current iteration.
        obj_value : float
            Current objective value.
        beta_diff : float
            Change in coefficients.
        """
        if self.verbose and iteration % 10 == 0:
            print(f"Iter {iteration:4d}: obj = {obj_value:.6e}, "
                  f"||β_new - β_old||∞ = {beta_diff:.6e}")
                  
    def get_params(self) -> Dict[str, Any]:
        """Get solver parameters.
        
        Returns
        -------
        dict
            Solver parameters.
        """
        return {
            'penalty': self.penalty,
            'a': self.a,
            'gamma': self.gamma,
            'tol': self.tol,
            'maxit': self.maxit,
            'verbose': self.verbose
        }
        
    def set_params(self, **params) -> 'BaseSolver':
        """Set solver parameters.
        
        Parameters
        ----------
        **params
            Solver parameters to set.
            
        Returns
        -------
        self
            Solver instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self