"""Data preprocessing utilities."""

from typing import Tuple, Optional, Union
import numpy as np
import numpy.typing as npt

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


def add_intercept(X: ArrayLike) -> np.ndarray:
    """Add intercept column to design matrix.
    
    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Design matrix without intercept.
        
    Returns
    -------
    np.ndarray of shape (n_samples, n_features + 1)
        Design matrix with intercept column prepended.
    """
    n = X.shape[0]
    X_with_intercept = np.column_stack((np.ones(n), X))
    return X_with_intercept


def standardize_data(X: ArrayLike, 
                    y: Optional[ArrayLike] = None,
                    center: bool = True,
                    scale: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Standardize features (not intercept column).
    
    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Data to standardize.
    y : array_like of shape (n_samples,), optional
        Target values to center.
    center : bool, default=True
        Center the data.
    scale : bool, default=True
        Scale to unit variance.
        
    Returns
    -------
    X_std : np.ndarray
        Standardized features.
    y_centered : np.ndarray, optional
        Centered y (if provided).
    """
    # Convert to float to avoid casting issues
    X_std = X.astype(np.float64, copy=True)
    
    # Don't standardize intercept column
    if X.shape[1] > 1:
        if center:
            X_mean = np.mean(X_std[:, 1:], axis=0)
            X_std[:, 1:] -= X_mean
            
        if scale:
            X_scale = np.std(X_std[:, 1:], axis=0)
            # Avoid division by zero
            X_scale[X_scale == 0] = 1.0
            X_std[:, 1:] /= X_scale
    
    if y is not None:
        if center:
            y_centered = y - np.mean(y)
            return X_std, y_centered
        else:
            return X_std, y.copy()
    
    return X_std


def check_design_matrix(X: ArrayLike) -> None:
    """Validate design matrix.
    
    Parameters
    ----------
    X : array_like
        Design matrix to validate.
        
    Raises
    ------
    ValueError
        If X contains NaN, inf values or has other issues.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
        
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
        
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")
        
    if X.shape[0] < X.shape[1]:
        import warnings
        warnings.warn(f"n_samples ({X.shape[0]}) < n_features ({X.shape[1]}). "
                     "This might lead to overfitting.", UserWarning)