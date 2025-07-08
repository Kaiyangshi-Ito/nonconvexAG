"""Input validation utilities."""

from typing import Optional, Union, Literal
import numpy as np
import numpy.typing as npt

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


def validate_inputs(X: ArrayLike, 
                   y: ArrayLike,
                   penalty: Literal["SCAD", "MCP"] = "SCAD",
                   lambda_: Optional[Union[float, ArrayLike]] = None,
                   a: float = 3.7,
                   gamma: float = 2.0,
                   tol: float = 1e-4,
                   maxit: int = 1000) -> None:
    """Validate inputs for optimization functions.
    
    Parameters
    ----------
    X : array_like
        Design matrix.
    y : array_like
        Response vector.
    penalty : {'SCAD', 'MCP'}
        Penalty type.
    lambda_ : float or array_like, optional
        Regularization parameter(s).
    a : float
        SCAD parameter.
    gamma : float
        MCP parameter.
    tol : float
        Convergence tolerance.
    maxit : int
        Maximum iterations.
        
    Raises
    ------
    ValueError
        If any inputs are invalid.
    TypeError
        If inputs have wrong type.
    """
    # Check arrays
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
        
    # Check dimensions
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {X.ndim}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got {y.ndim}")
        
    # Check shapes match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples: "
                        f"{X.shape[0]} != {y.shape[0]}")
    
    # Check penalty
    if penalty not in ["SCAD", "MCP"]:
        raise ValueError(f"penalty must be 'SCAD' or 'MCP', got {penalty}")
        
    # Check lambda
    if lambda_ is not None:
        lambda_arr = np.asarray(lambda_)
        if np.any(lambda_arr < 0):
            raise ValueError("lambda must be non-negative")
            
    # Check parameters
    if penalty == "SCAD" and a <= 2:
        raise ValueError(f"SCAD parameter 'a' must be > 2, got {a}")
    if penalty == "MCP" and gamma <= 0:
        raise ValueError(f"MCP parameter 'gamma' must be > 0, got {gamma}")
        
    # Check convergence parameters
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if maxit < 1:
        raise ValueError(f"maxit must be at least 1, got {maxit}")
        

def validate_model_type(y: ArrayLike, model_type: Literal["linear", "logistic"]) -> None:
    """Validate that response matches model type.
    
    Parameters
    ----------
    y : array_like
        Response vector.
    model_type : {'linear', 'logistic'}
        Type of model.
        
    Raises
    ------
    ValueError
        If response doesn't match model type.
    """
    if model_type == "logistic":
        unique_vals = np.unique(y)
        if not np.array_equal(unique_vals, [0, 1]) and not np.array_equal(unique_vals, [0.0, 1.0]):
            raise ValueError(f"For logistic regression, y must contain only 0 and 1. "
                           f"Got unique values: {unique_vals}")
                           

def check_convergence(beta_old: ArrayLike, 
                     beta_new: ArrayLike, 
                     tol: float,
                     iteration: int,
                     maxit: int) -> bool:
    """Check convergence of optimization algorithm.
    
    Parameters
    ----------
    beta_old : array_like
        Previous parameter vector.
    beta_new : array_like
        Current parameter vector.
    tol : float
        Convergence tolerance.
    iteration : int
        Current iteration number.
    maxit : int
        Maximum iterations allowed.
        
    Returns
    -------
    bool
        True if converged, False otherwise.
        
    Warns
    -----
    UserWarning
        If maximum iterations reached without convergence.
    """
    if iteration >= maxit:
        import warnings
        warnings.warn(f"Maximum iterations ({maxit}) reached without convergence. "
                     "Consider increasing maxit or tol.", UserWarning)
        return True
        
    # Check uniform norm convergence
    return np.max(np.abs(beta_new - beta_old)) < tol