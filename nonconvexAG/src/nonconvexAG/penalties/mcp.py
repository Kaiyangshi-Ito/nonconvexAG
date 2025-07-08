"""MCP (Minimax Concave Penalty) functions."""

from typing import Union
import numpy as np
from numba import jit
import numpy.typing as npt

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP(beta: ArrayLike, lambda_: float, gamma: float = 2.0) -> float:
    """Calculate MCP penalty value.
    
    The MCP penalty is defined as:
    - λ|β| - β²/(2γ) for |β| ≤ γλ
    - γλ²/2 for |β| > γλ
    
    Parameters
    ----------
    beta : array_like
        Parameter vector (first element is not penalized).
    lambda_ : float
        Regularization parameter.
    gamma : float, default=2.0
        MCP parameter controlling the concavity.
        
    Returns
    -------
    float
        MCP penalty value.
        
    References
    ----------
    Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax 
    concave penalty. Annals of Statistics 38(2), 894-942.
    """
    beta_1 = beta[1:]  # Exclude intercept
    abs_beta = np.abs(beta_1)
    
    penalty = 0.0
    for i in range(len(beta_1)):
        if abs_beta[i] <= gamma * lambda_:
            penalty += lambda_ * abs_beta[i] - abs_beta[i]**2 / (2 * gamma)
        else:
            penalty += gamma * lambda_**2 / 2
            
    return penalty


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP_grad(beta: ArrayLike, lambda_: float, gamma: float = 2.0) -> np.ndarray:
    """Calculate gradient of MCP penalty.
    
    Parameters
    ----------
    beta : array_like
        Parameter vector (first element is not penalized).
    lambda_ : float
        Regularization parameter.
    gamma : float, default=2.0
        MCP parameter.
        
    Returns
    -------
    np.ndarray
        Gradient vector with same shape as beta.
    """
    grad = np.zeros_like(beta)
    
    for i in range(1, len(beta)):  # Skip intercept
        abs_beta_i = np.abs(beta[i])
        
        if abs_beta_i <= gamma * lambda_:
            grad[i] = lambda_ * np.sign(beta[i]) - beta[i] / gamma
        else:
            grad[i] = 0.0
            
    return grad


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP_concave(beta: ArrayLike, lambda_: float, gamma: float = 2.0) -> float:
    """Calculate concave part of MCP penalty.
    
    MCP = L1 penalty - MCP_concave
    
    Parameters
    ----------
    beta : array_like
        Parameter vector (first element is not penalized).
    lambda_ : float
        Regularization parameter.
    gamma : float, default=2.0
        MCP parameter.
        
    Returns
    -------
    float
        Value of concave component.
    """
    beta_1 = beta[1:]  # Exclude intercept
    abs_beta = np.abs(beta_1)
    
    concave_part = 0.0
    for i in range(len(beta_1)):
        if abs_beta[i] <= gamma * lambda_:
            concave_part += abs_beta[i]**2 / (2 * gamma)
        else:
            concave_part += lambda_ * abs_beta[i] - gamma * lambda_**2 / 2
            
    return concave_part


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP_concave_grad(beta: ArrayLike, lambda_: float, gamma: float = 2.0) -> np.ndarray:
    """Calculate gradient of concave part of MCP penalty.
    
    Parameters
    ----------
    beta : array_like
        Parameter vector (first element is not penalized).
    lambda_ : float
        Regularization parameter.
    gamma : float, default=2.0
        MCP parameter.
        
    Returns
    -------
    np.ndarray
        Gradient vector with same shape as beta.
    """
    grad = np.zeros_like(beta)
    
    for i in range(1, len(beta)):  # Skip intercept
        abs_beta_i = np.abs(beta[i])
        
        if abs_beta_i <= gamma * lambda_:
            grad[i] = beta[i] / gamma
        else:
            grad[i] = lambda_ * np.sign(beta[i])
            
    return grad