"""Unified Accelerated Gradient (UAG) solver implementation."""

from typing import Optional, Union, Literal, Tuple
import numpy as np
from numba import jit
import numpy.typing as npt
import time

from .base import BaseSolver
from ..penalties import (soft_thresholding, SCAD_concave_grad, MCP_concave_grad)
from ..models import LinearModel, LogisticModel
from ..models.linear import update_smooth_grad_convex_LM
from ..models.logistic import update_smooth_grad_convex_logistic
from ..utils.validation import validate_inputs, validate_model_type

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


class UAG(BaseSolver):
    """UAG solver for nonconvex sparse learning.
    
    Parameters
    ----------
    model_type : {'linear', 'logistic'}
        Regression model type.
    penalty : {'SCAD', 'MCP'}
        Penalty type.
    a : float, default=3.7
        SCAD parameter.
    gamma : float, default=2.0
        MCP parameter.
    tol : float, default=1e-4
        Convergence tolerance.
    maxit : int, default=1000
        Max iterations.
    L_convex : float, optional
        L-smoothness constant.
    verbose : bool, default=False
        Print progress.
        
    Attributes
    ----------
    coef_ : np.ndarray
        Fitted coefficients.
    intercept_ : float
        Fitted intercept.
    n_iter_ : int
        Iterations performed.
    converged_ : bool
        Convergence flag.
    runtime_ : float
        Runtime in seconds.
    """
    
    def __init__(self,
                 model_type: Literal["linear", "logistic"] = "linear",
                 penalty: Literal["SCAD", "MCP"] = "SCAD",
                 a: float = 3.7,
                 gamma: float = 2.0,
                 tol: float = 1e-4,
                 maxit: int = 1000,
                 L_convex: Optional[float] = None,
                 verbose: bool = False):
        pass
        super().__init__(penalty=penalty, a=a, gamma=gamma, 
                        tol=tol, maxit=maxit, verbose=verbose)
        
        self.model_type = model_type
        self.L_convex = L_convex
        
        # Initialize model
        if model_type == "linear":
            self.model = LinearModel()
        else:
            self.model = LogisticModel()
            
        # Results
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X: ArrayLike, y: ArrayLike, lambda_: float,
            beta_init: Optional[ArrayLike] = None) -> 'UAG':
        """Fit the model using UAG algorithm.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix (should include intercept column).
        y : array_like of shape (n_samples,)
            Response vector.
        lambda_ : float
            Regularization parameter.
        beta_init : array_like of shape (n_features,), optional
            Initial coefficients. If None, will be initialized automatically.
            
        Returns
        -------
        self
            Fitted UAG instance.
        """
        start_time = time.time()
        
        # Validate inputs
        validate_inputs(X, y, self.penalty, lambda_, self.a, self.gamma, 
                       self.tol, self.maxit)
        validate_model_type(y, self.model_type)
        
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n, p = X.shape
        
        # Calculate L-convex if not provided
        if self.L_convex is None:
            self.L_convex = self.model.calculate_L_convex(X)
            
        # Initialize beta
        if beta_init is None:
            beta = self.model.init_beta(X, y)
        else:
            beta = np.asarray(beta_init, dtype=np.float64).copy()
            
        # Run optimization
        beta, n_iter, converged = self._optimize(X, y, beta, lambda_)
        
        # Store results
        self.coef_ = beta[1:]  # Exclude intercept
        self.intercept_ = beta[0]
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.runtime_ = time.time() - start_time
        
        if self.verbose:
            status = "converged" if converged else "reached max iterations"
            print(f"\nOptimization {status} in {n_iter} iterations "
                  f"({self.runtime_:.3f} seconds)")
        
        return self
        
    def _optimize(self, X: ArrayLike, y: ArrayLike, beta: ArrayLike,
                  lambda_: float) -> Tuple[np.ndarray, int, bool]:
        """Run UAG optimization algorithm.
        
        Parameters
        ----------
        X : array_like
            Design matrix.
        y : array_like
            Response vector.
        beta : array_like
            Initial coefficients.
        lambda_ : float
            Regularization parameter.
            
        Returns
        -------
        beta : np.ndarray
            Optimized coefficients.
        n_iter : int
            Number of iterations performed.
        converged : bool
            Whether algorithm converged.
        """
        # Select appropriate functions based on model type and penalty
        if self.model_type == "linear":
            update_grad_func = update_smooth_grad_convex_LM
            if self.penalty == "SCAD":
                update_grad_concave_func = _update_smooth_grad_SCAD_LM
            else:
                update_grad_concave_func = _update_smooth_grad_MCP_LM
        else:
            update_grad_func = update_smooth_grad_convex_logistic
            if self.penalty == "SCAD":
                update_grad_concave_func = _update_smooth_grad_SCAD_logistic
            else:
                update_grad_concave_func = _update_smooth_grad_MCP_logistic
                
        # Get penalty-specific parameters
        if self.penalty == "SCAD":
            penalty_param = self.a
        else:
            penalty_param = self.gamma
            
        # Run core UAG algorithm
        return _uag_core(X, y, beta, lambda_, penalty_param,
                        self.L_convex, self.tol, self.maxit,
                        update_grad_func, update_grad_concave_func,
                        self.verbose)
                        
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Make predictions.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix (should include intercept column).
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predictions.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        X = np.asarray(X)
        beta = np.concatenate([[self.intercept_], self.coef_])
        return self.model.predict(X, beta)
        
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Calculate model score.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        y : array_like of shape (n_samples,)
            True response values.
            
        Returns
        -------
        float
            R² score for linear regression, accuracy for logistic.
        """
        predictions = self.predict(X)
        
        if self.model_type == "linear":
            # R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            # Accuracy
            return np.mean((predictions >= 0.5) == y)


# Numba-optimized core UAG algorithm
@jit(nopython=True, cache=True)
def _uag_core(X: ArrayLike, y: ArrayLike, beta_init: ArrayLike,
              lambda_: float, penalty_param: float, L_convex: float,
              tol: float, maxit: int,
              update_grad_convex, update_grad_concave,
              verbose: bool = False) -> Tuple[np.ndarray, int, bool]:
    """Core UAG optimization routine."""
    n, p = X.shape
    
    # Initialize
    beta = beta_init.copy()
    beta_ag = beta.copy()
    beta_md = beta.copy()
    
    # Gradient storage
    smooth_grad = np.zeros(p)
    smooth_grad_md = np.zeros(p)
    
    # Momentum parameters
    t = 1.0
    converged = False
    
    for k in range(maxit):
        # Store previous values
        beta_old = beta.copy()
        beta_ag_old = beta_ag.copy()
        t_old = t
        
        # Update smooth gradient at momentum point
        smooth_grad = update_grad_convex(smooth_grad, X, y, beta_md)
        smooth_grad_md = update_grad_concave(smooth_grad_md, X, y, beta_md,
                                            lambda_, penalty_param)
        
        # Full gradient
        full_grad = smooth_grad + smooth_grad_md
        
        # Proximal gradient step
        beta_ag = soft_thresholding(beta_md - full_grad / L_convex, 
                                   lambda_ / L_convex)
        
        # Check restart condition
        if np.dot(beta_ag - beta_ag_old, beta_ag - beta_old) < 0:
            # Restart
            t = 1.0
            beta = beta_ag.copy()
        else:
            # Continue with momentum
            t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
            beta = beta_ag.copy()
            
        # Update momentum point
        momentum = (t_old - 1) / t
        beta_md = beta + momentum * (beta - beta_old)
        
        # Check convergence
        beta_diff = np.max(np.abs(beta - beta_old))
        if beta_diff < tol:
            converged = True
            break
            
    return beta, k + 1, converged


# Model-specific gradient update functions
@jit(nopython=True, cache=True)
def _update_smooth_grad_SCAD_LM(smooth_grad: ArrayLike, X: ArrayLike,
                               y: ArrayLike, beta: ArrayLike,
                               lambda_: float, a: float) -> np.ndarray:
    """Update SCAD concave gradient for linear model."""
    smooth_grad[:] = SCAD_concave_grad(beta, lambda_, a)
    return smooth_grad


@jit(nopython=True, cache=True)
def _update_smooth_grad_MCP_LM(smooth_grad: ArrayLike, X: ArrayLike,
                              y: ArrayLike, beta: ArrayLike,
                              lambda_: float, gamma: float) -> np.ndarray:
    """Update MCP concave gradient for linear model."""
    smooth_grad[:] = MCP_concave_grad(beta, lambda_, gamma)
    return smooth_grad


@jit(nopython=True, cache=True)
def _update_smooth_grad_SCAD_logistic(smooth_grad: ArrayLike, X: ArrayLike,
                                     y: ArrayLike, beta: ArrayLike,
                                     lambda_: float, a: float) -> np.ndarray:
    """Update SCAD concave gradient for logistic model."""
    smooth_grad[:] = SCAD_concave_grad(beta, lambda_, a)
    return smooth_grad


@jit(nopython=True, cache=True)
def _update_smooth_grad_MCP_logistic(smooth_grad: ArrayLike, X: ArrayLike,
                                    y: ArrayLike, beta: ArrayLike,
                                    lambda_: float, gamma: float) -> np.ndarray:
    """Update MCP concave gradient for logistic model."""
    smooth_grad[:] = MCP_concave_grad(beta, lambda_, gamma)
    return smooth_grad