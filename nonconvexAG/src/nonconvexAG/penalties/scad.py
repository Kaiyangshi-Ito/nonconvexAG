"""SCAD (Smoothly Clipped Absolute Deviation) penalty functions."""

from typing import Union
import numpy as np
from numba import jit
import numpy.typing as npt

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SCAD(beta: ArrayLike, lambda_: float, a: float = 3.7) -> float:
    """Calculate SCAD penalty value.
    
    Parameters
    ----------
    beta : array_like
        Parameters (first element unpenalized).
    lambda_ : float
        Regularization parameter.
    a : float, default=3.7
        SCAD parameter.
        
    Returns
    -------
    float
        Penalty value.
    """
    beta_1 = beta[1:]  # Exclude intercept
    abs_beta = np.abs(beta_1)
    
    penalty = 0.0
    for i in range(len(beta_1)):
        if abs_beta[i] <= lambda_:
            penalty += lambda_ * abs_beta[i]
        elif abs_beta[i] <= a * lambda_:
            penalty += (2 * a * lambda_ * abs_beta[i] - abs_beta[i]**2 - lambda_**2) / (2 * (a - 1))
        else:
            penalty += lambda_**2 * (a + 1) / 2
            
    return penalty


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SCAD_grad(beta: ArrayLike, lambda_: float, a: float = 3.7) -> np.ndarray:
    """Calculate gradient of SCAD penalty.
    
    Parameters
    ----------
    beta : array_like
        Parameter vector (first element is not penalized).
    lambda_ : float
        Regularization parameter.
    a : float, default=3.7
        SCAD parameter.
        
    Returns
    -------
    np.ndarray
        Gradient vector with same shape as beta.
    """
    grad = np.zeros_like(beta)
    
    for i in range(1, len(beta)):  # Skip intercept
        abs_beta_i = np.abs(beta[i])
        
        if abs_beta_i <= lambda_:
            grad[i] = lambda_ * np.sign(beta[i])
        elif abs_beta_i <= a * lambda_:
            grad[i] = (a * lambda_ * np.sign(beta[i]) - beta[i]) / (a - 1)
        else:
            grad[i] = 0.0
            
    return grad


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True) 
def SCAD_concave(beta: ArrayLike, lambda_: float, a: float = 3.7) -> float:
    """Calculate concave part of SCAD penalty.
    
    SCAD = L1 penalty - SCAD_concave
    
    Parameters
    ----------
    beta : array_like
        Parameter vector (first element is not penalized).
    lambda_ : float
        Regularization parameter.
    a : float, default=3.7
        SCAD parameter.
        
    Returns
    -------
    float
        Value of concave component.
    """
    beta_1 = beta[1:]  # Exclude intercept
    abs_beta = np.abs(beta_1)
    
    concave_part = 0.0
    for i in range(len(beta_1)):
        if abs_beta[i] <= lambda_:
            concave_part += 0.0
        elif abs_beta[i] <= a * lambda_:
            concave_part += (abs_beta[i] - lambda_)**2 / (2 * (a - 1))
        else:
            concave_part += (a + 1) * lambda_**2 / 2 - lambda_ * abs_beta[i]
            
    return concave_part


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SCAD_concave_grad(beta: ArrayLike, lambda_: float, a: float = 3.7) -> np.ndarray:
    """Calculate gradient of concave part of SCAD penalty.
    
    Parameters
    ----------
    beta : array_like
        Parameter vector (first element is not penalized).
    lambda_ : float
        Regularization parameter.
    a : float, default=3.7
        SCAD parameter.
        
    Returns
    -------
    np.ndarray
        Gradient vector with same shape as beta.
    """
    grad = np.zeros_like(beta)
    
    for i in range(1, len(beta)):  # Skip intercept
        abs_beta_i = np.abs(beta[i])
        
        if abs_beta_i <= lambda_:
            grad[i] = 0.0
        elif abs_beta_i <= a * lambda_:
            grad[i] = (beta[i] - lambda_ * np.sign(beta[i])) / (a - 1)
        else:
            grad[i] = -lambda_ * np.sign(beta[i])
            
    return grad