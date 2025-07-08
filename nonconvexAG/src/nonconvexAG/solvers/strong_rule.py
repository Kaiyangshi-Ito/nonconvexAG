"""Strong rule implementation for efficient variable screening."""

from typing import Optional, Union, Literal, Set, Tuple
import numpy as np
from numba import jit
import numpy.typing as npt

from .uag import UAG
from ..utils import validate_inputs
from ..models.linear import linear_gradient_jit
from ..models.logistic import logistic_gradient_jit

ArrayLike = Union[np.ndarray, npt.NDArray[np.float64]]


class StrongRuleSolver(UAG):
    """UAG solver with strong rule for efficient variable screening.
    
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
    strong_rule_freq : int, default=1
        Strong rule check frequency.
    violation_check : bool, default=True
        Check for violations.
    verbose : bool, default=False
        Print progress.
        
    Attributes
    ----------
    active_set_ : Set[int]
        Active variable indices.
    n_strong_rule_checks_ : int
        Strong rule check count.
    n_violations_ : int
        Violation count.
    """
    
    def __init__(self,
                 model_type: Literal["linear", "logistic"] = "linear",
                 penalty: Literal["SCAD", "MCP"] = "SCAD",
                 a: float = 3.7,
                 gamma: float = 2.0,
                 tol: float = 1e-4,
                 maxit: int = 1000,
                 strong_rule_freq: int = 1,
                 violation_check: bool = True,
                 L_convex: Optional[float] = None,
                 verbose: bool = False):
        """Initialize strong rule solver."""
        super().__init__(model_type=model_type, penalty=penalty, a=a, gamma=gamma,
                        tol=tol, maxit=maxit, L_convex=L_convex, verbose=verbose)
        
        self.strong_rule_freq = strong_rule_freq
        self.violation_check = violation_check
        
        # Strong rule statistics
        self.active_set_ = set()
        self.n_strong_rule_checks_ = 0
        self.n_violations_ = 0
        
    def fit(self, X: ArrayLike, y: ArrayLike, lambda_: float,
            beta_init: Optional[ArrayLike] = None,
            lambda_prev: Optional[float] = None,
            beta_prev: Optional[ArrayLike] = None) -> 'StrongRuleSolver':
        """Fit model using UAG with strong rule.
        
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Design matrix.
        y : array_like of shape (n_samples,)
            Response vector.
        lambda_ : float
            Current regularization parameter.
        beta_init : array_like of shape (n_features,), optional
            Initial coefficients.
        lambda_prev : float, optional
            Previous lambda value (for sequential strong rule).
        beta_prev : array_like of shape (n_features,), optional
            Coefficients from previous lambda.
            
        Returns
        -------
        self
            Fitted solver instance.
        """
        # Input validation
        validate_inputs(X, y, self.penalty, lambda_, self.a, self.gamma,
                       self.tol, self.maxit)
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        
        # Calculate L-convex if needed
        if self.L_convex is None:
            self.L_convex = self.model.calculate_L_convex(X)
        
        # Initialize
        if beta_init is None:
            beta = self.model.init_beta(X, y)
        else:
            beta = np.asarray(beta_init, dtype=np.float64).copy()
            
        # Apply strong rule to get initial active set
        if lambda_prev is not None and beta_prev is not None:
            # Sequential strong rule
            active_indices = self._sequential_strong_rule(
                X, y, beta_prev, lambda_prev, lambda_
            )
        else:
            # Basic strong rule
            active_indices = self._basic_strong_rule(X, y, beta, lambda_)
            
        self.active_set_ = set(active_indices)
        self.n_strong_rule_checks_ = 1
        
        if self.verbose:
            print(f"Initial active set size: {len(self.active_set_)}/{p}")
        
        # Main optimization loop with active set
        converged = False
        for iteration in range(self.maxit):
            beta_old = beta.copy()
            
            # Update only on active set
            beta = self._update_active_set(X, y, beta, lambda_, self.maxit - iteration)
            
            # Check convergence
            beta_diff = np.max(np.abs(beta - beta_old))
            if beta_diff < self.tol:
                converged = True
                break
                
            # Periodically check for violations
            if self.violation_check and iteration % self.strong_rule_freq == 0:
                violations = self._check_violations(X, y, beta, lambda_)
                if violations:
                    self.active_set_.update(violations)
                    self.n_violations_ += len(violations)
                    if self.verbose:
                        print(f"Iter {iteration}: Found {len(violations)} violations, "
                              f"active set size: {len(self.active_set_)}")
        
        # Final violation check
        if self.violation_check:
            final_violations = self._check_violations(X, y, beta, lambda_)
            if final_violations:
                # Re-optimize with expanded active set
                self.active_set_.update(final_violations)
                beta = self._update_active_set(X, y, beta, lambda_, self.maxit)
                
        # Store results
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]
        self.n_iter_ = iteration + 1
        self.converged_ = converged
        
        if self.verbose:
            print(f"Final active set size: {len(self.active_set_)}/{p}")
            print(f"Total violations: {self.n_violations_}")
            
        return self
        
    def _basic_strong_rule(self, X: ArrayLike, y: ArrayLike, 
                          beta: ArrayLike, lambda_: float) -> np.ndarray:
        """Apply basic strong rule for initial screening.
        
        Parameters
        ----------
        X : array_like
            Design matrix.
        y : array_like
            Response vector.
        beta : array_like
            Current coefficients.
        lambda_ : float
            Regularization parameter.
            
        Returns
        -------
        np.ndarray
            Indices of potentially active variables.
        """
        n = X.shape[0]
        
        # Calculate gradient at current beta
        if self.model_type == "linear":
            grad = linear_gradient_jit(X, y, beta)
        else:
            grad = logistic_gradient_jit(X, y, beta)
            
        # Strong rule threshold
        threshold = 2 * lambda_ - self.tol
        
        # Variables that pass the threshold
        active = np.where(np.abs(grad[1:]) > threshold)[0] + 1  # +1 for intercept
        
        # Always include intercept
        return np.concatenate([[0], active])
        
    def _sequential_strong_rule(self, X: ArrayLike, y: ArrayLike,
                               beta_prev: ArrayLike, lambda_prev: float,
                               lambda_curr: float) -> np.ndarray:
        """Apply sequential strong rule using previous solution.
        
        Parameters
        ----------
        X : array_like
            Design matrix.
        y : array_like
            Response vector.
        beta_prev : array_like
            Coefficients from previous lambda.
        lambda_prev : float
            Previous lambda value.
        lambda_curr : float
            Current lambda value.
            
        Returns
        -------
        np.ndarray
            Indices of potentially active variables.
        """
        # Include previously active variables
        active_prev = np.where(np.abs(beta_prev[1:]) > 1e-10)[0] + 1
        
        # Calculate gradient at previous solution
        if self.model_type == "linear":
            grad_prev = linear_gradient_jit(X, y, beta_prev)
        else:
            grad_prev = logistic_gradient_jit(X, y, beta_prev)
            
        # Sequential strong rule threshold
        threshold = 2 * lambda_curr - lambda_prev
        
        # Variables that pass the threshold
        active_new = np.where(np.abs(grad_prev[1:]) > threshold)[0] + 1
        
        # Combine with previously active
        active = np.unique(np.concatenate([active_prev, active_new]))
        
        # Always include intercept
        return np.concatenate([[0], active])
        
    def _update_active_set(self, X: ArrayLike, y: ArrayLike,
                          beta: ArrayLike, lambda_: float,
                          max_iter: int) -> np.ndarray:
        """Update coefficients only on active set.
        
        Parameters
        ----------
        X : array_like
            Design matrix.
        y : array_like
            Response vector. 
        beta : array_like
            Current coefficients.
        lambda_ : float
            Regularization parameter.
        max_iter : int
            Maximum iterations.
            
        Returns
        -------
        np.ndarray
            Updated coefficients.
        """
        # Create active set mask
        active_mask = np.zeros(len(beta), dtype=bool)
        active_mask[list(self.active_set_)] = True
        
        # Extract active subset
        active_indices = np.array(list(self.active_set_))
        X_active = X[:, active_indices]
        beta_active = beta[active_indices]
        
        # Create temporary solver for active set
        solver_active = UAG(model_type=self.model_type, penalty=self.penalty,
                           a=self.a, gamma=self.gamma, tol=self.tol,
                           maxit=max_iter, L_convex=self.L_convex, verbose=False)
        
        # Solve on active set
        solver_active.fit(X_active, y, lambda_, beta_init=beta_active)
        
        # Update full beta
        beta_new = beta.copy()
        beta_new[active_indices] = np.concatenate([[solver_active.intercept_], 
                                                   solver_active.coef_])
        beta_new[~active_mask] = 0.0  # Set inactive to exact zero
        
        return beta_new
        
    def _check_violations(self, X: ArrayLike, y: ArrayLike,
                         beta: ArrayLike, lambda_: float) -> Set[int]:
        """Check for strong rule violations.
        
        Parameters
        ----------
        X : array_like
            Design matrix.
        y : array_like
            Response vector.
        beta : array_like
            Current coefficients.
        lambda_ : float
            Regularization parameter.
            
        Returns
        -------
        Set[int]
            Indices of variables that violate the strong rule.
        """
        n, p = X.shape
        
        # Calculate gradient
        if self.model_type == "linear":
            grad = linear_gradient_jit(X, y, beta)
        else:
            grad = logistic_gradient_jit(X, y, beta)
            
        violations = set()
        
        # Check inactive variables
        for j in range(1, p):  # Skip intercept
            if j not in self.active_set_:
                # Check KKT condition for inactive variable
                if np.abs(grad[j]) > lambda_ + self.tol:
                    violations.add(j)
                    
        self.n_strong_rule_checks_ += 1
        
        return violations