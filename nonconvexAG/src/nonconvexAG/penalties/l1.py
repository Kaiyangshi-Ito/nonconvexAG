"""L1 penalty and soft thresholding operations."""

from typing import Union
import numpy as np
from numba import jit
import numpy.typing as npt

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def soft_thresholding(x: ArrayLike, lambda_: float) -> np.ndarray:
    """Apply soft-thresholding operator for L1 penalty.
    
    Calculates the soft-thresholding mapping of a given vector, 
    excluding the first element (intercept term).
    
    Parameters
    ----------
    x : array_like
        Input vector where x[0] is the intercept (not penalized).
    lambda_ : float
        Regularization parameter for L1 penalty.
        
    Returns
    -------
    np.ndarray
        Soft-thresholded vector with same shape as input.
        
    Notes
    -----
    The soft-thresholding operator is defined as:
        S(x, λ) = sign(x) * max(|x| - λ, 0)
    
    The first element (intercept) is not penalized and passes through unchanged.
    """
    return np.hstack((np.array([x[0]]),
                     np.where(np.abs(x[1:]) > lambda_,
                             x[1:] - np.sign(x[1:]) * lambda_, 
                             0)))