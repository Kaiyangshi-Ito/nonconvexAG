"""Solution path computation for nonconvex sparse learning."""

from typing import Optional, Union, Literal, List
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .base import BaseSolver
from .uag import UAG
from ..utils import lambda_max_LM, lambda_max_logistic, add_intercept
from ..utils.lambda_utils import generate_lambda_sequence

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


class SolutionPath(BaseSolver):
    """Compute regularization path for nonconvex sparse learning.
    
    This class computes solutions for a sequence of lambda values,
    using warm starts for computational efficiency.
    
    Parameters
    ----------
    model_type : {'linear', 'logistic'}
        Type of regression model.
    penalty : {'SCAD', 'MCP'}
        Type of nonconvex penalty.
    a : float, default=3.7
        SCAD parameter.
    gamma : float, default=2.0
        MCP parameter.
    tol : float, default=1e-4
        Convergence tolerance.
    maxit : int, default=1000
        Maximum iterations per lambda.
    verbose : bool, default=False
        Whether to show progress.
        
    Attributes
    ----------
    coef_path_ : np.ndarray of shape (n_features, n_lambdas)
        Coefficients along the regularization path.
    intercept_path_ : np.ndarray of shape (n_lambdas,)
        Intercepts along the regularization path.
    lambda_path_ : np.ndarray of shape (n_lambdas,)
        The lambda values used.
    n_iter_path_ : List[int]
        Number of iterations for each lambda.
    """
    
    def __init__(self,
                 model_type: Literal["linear", "logistic"] = "linear",
                 penalty: Literal["SCAD", "MCP"] = "SCAD",
                 a: float = 3.7,
                 gamma: float = 2.0,
                 tol: float = 1e-4,
                 maxit: int = 1000,
                 verbose: bool = False):
        """Initialize solution path solver."""
        super().__init__(penalty=penalty, a=a, gamma=gamma,
                        tol=tol, maxit=maxit, verbose=verbose)
        
        self.model_type = model_type
        
        # Results storage
        self.coef_path_ = None
        self.intercept_path_ = None
        self.lambda_path_ = None
        self.n_iter_path_ = []
        
    def fit(self, X: ArrayLike, y: ArrayLike,
            lambdas: Optional[ArrayLike] = None,
            n_lambdas: int = 100,
            lambda_min_ratio: float = 0.01,
            standardize: bool = True) -> 'SolutionPath':
        """Compute the regularization path.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix (should include intercept if desired).
        y : array_like of shape (n_samples,)
            Response vector.
        lambdas : array_like of shape (n_lambdas,), optional
            Lambda values to use. If None, will be generated automatically.
        n_lambdas : int, default=100
            Number of lambda values if lambdas is None.
        lambda_min_ratio : float, default=0.01
            Ratio of smallest to largest lambda if lambdas is None.
        standardize : bool, default=True
            Whether to standardize features before fitting.
            
        Returns
        -------
        self
            Fitted SolutionPath instance.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        # Standardization tracking
        self._X_mean = None
        self._X_scale = None
        self._y_mean = None
        
        if standardize and n_features > 1:
            # Don't standardize intercept column if present
            self._X_mean = np.mean(X[:, 1:], axis=0)
            self._X_scale = np.std(X[:, 1:], axis=0)
            self._X_scale[self._X_scale == 0] = 1.0
            
            X = X.copy()
            X[:, 1:] = (X[:, 1:] - self._X_mean) / self._X_scale
            
            if self.model_type == "linear":
                self._y_mean = np.mean(y)
                y = y - self._y_mean
        
        # Generate lambda sequence if not provided
        if lambdas is None:
            if self.model_type == "linear":
                lambda_max = lambda_max_LM(X, y, self.penalty, self.a, self.gamma)
            else:
                lambda_max = lambda_max_logistic(X, y, self.penalty, self.a, self.gamma)
                
            lambdas = generate_lambda_sequence(lambda_max, lambda_min_ratio,
                                             n_lambdas, log_scale=True)
        else:
            lambdas = np.asarray(lambdas)
            # Ensure decreasing order
            if lambdas[0] < lambdas[-1]:
                lambdas = lambdas[::-1]
                
        n_lambdas = len(lambdas)
        self.lambda_path_ = lambdas
        
        # Initialize storage
        self.coef_path_ = np.zeros((n_features - 1, n_lambdas))  # Exclude intercept
        self.intercept_path_ = np.zeros(n_lambdas)
        self.n_iter_path_ = []
        
        # Create UAG solver
        solver = UAG(model_type=self.model_type, penalty=self.penalty,
                    a=self.a, gamma=self.gamma, tol=self.tol, 
                    maxit=self.maxit, verbose=False)
        
        # Warm start with zeros for first lambda
        beta_init = None
        
        # Progress bar if verbose
        iterator = enumerate(lambdas)
        if self.verbose:
            iterator = tqdm(iterator, total=n_lambdas, desc="Computing path")
        
        for i, lambda_val in iterator:
            # Fit for current lambda
            solver.fit(X, y, lambda_val, beta_init=beta_init)
            
            # Store results
            self.coef_path_[:, i] = solver.coef_
            self.intercept_path_[i] = solver.intercept_
            self.n_iter_path_.append(solver.n_iter_)
            
            # Warm start for next lambda
            beta_init = np.concatenate([[solver.intercept_], solver.coef_])
            
            if self.verbose and i % 10 == 0:
                n_active = np.sum(np.abs(solver.coef_) > 1e-10)
                tqdm.write(f"Lambda {i+1}/{n_lambdas}: "
                          f"λ={lambda_val:.4f}, active={n_active}, "
                          f"iter={solver.n_iter_}")
        
        # Transform coefficients back to original scale if standardized
        if standardize and self._X_scale is not None:
            self.coef_path_ = self.coef_path_ / self._X_scale.reshape(-1, 1)
            if self.model_type == "linear" and self._y_mean is not None:
                # Adjust intercept for centering
                adjustment = np.sum(self.coef_path_ * self._X_mean.reshape(-1, 1), axis=0)
                self.intercept_path_ = self.intercept_path_ + self._y_mean - adjustment
        
        return self
        
    def predict(self, X: ArrayLike, lambda_idx: int = -1) -> np.ndarray:
        """Make predictions for a specific lambda.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        lambda_idx : int, default=-1
            Index of lambda value to use (default: smallest lambda).
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predictions.
        """
        if self.coef_path_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        X = np.asarray(X)
        coef = self.coef_path_[:, lambda_idx]
        intercept = self.intercept_path_[lambda_idx]
        
        # Create temporary UAG solver for predictions
        solver = UAG(model_type=self.model_type)
        solver.coef_ = coef
        solver.intercept_ = intercept
        
        return solver.predict(X)
        
    def get_support(self, lambda_idx: int = -1, threshold: float = 1e-6) -> np.ndarray:
        """Get mask of non-zero coefficients.
        
        Parameters
        ----------
        lambda_idx : int, default=-1
            Index of lambda value to use.
        threshold : float, default=1e-6
            Threshold below which coefficients are considered zero.
            
        Returns
        -------
        np.ndarray of shape (n_features,)
            Boolean mask of active features.
        """
        if self.coef_path_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        return np.abs(self.coef_path_[:, lambda_idx]) > threshold
        
    def plot_path(self, feature_names: Optional[List[str]] = None,
                  log_lambda: bool = True, **kwargs):
        """Plot the regularization path.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names for features.
        log_lambda : bool, default=True
            Whether to use log scale for lambda axis.
        **kwargs
            Additional arguments passed to matplotlib.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. "
                            "Install with: pip install matplotlib")
            
        if self.coef_path_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # X-axis values
        x_vals = np.log(self.lambda_path_) if log_lambda else self.lambda_path_
        
        # Plot each coefficient trajectory
        n_features = self.coef_path_.shape[0]
        for i in range(n_features):
            if feature_names is not None and i < len(feature_names):
                label = feature_names[i]
            else:
                label = f"Feature {i+1}"
                
            # Only label features that are ever non-zero
            if np.any(np.abs(self.coef_path_[i, :]) > 1e-10):
                ax.plot(x_vals, self.coef_path_[i, :], label=label, **kwargs)
            else:
                ax.plot(x_vals, self.coef_path_[i, :], color='gray', 
                       alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('log(λ)' if log_lambda else 'λ')
        ax.set_ylabel('Coefficient value')
        ax.set_title(f'Regularization Path ({self.penalty} penalty)')
        ax.grid(True, alpha=0.3)
        
        # Only show legend for active features
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) <= 20:  # Don't show legend if too many features
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        return fig