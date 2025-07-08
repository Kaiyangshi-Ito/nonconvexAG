"""Accelerated gradient methods for nonconvex sparse learning with SCAD and MCP penalties."""

__version__ = "0.9.6"
__author__ = "Kai Yang"
__email__ = "kai.yang2@mail.mcgill.ca"

# Import required libraries
import numpy as np

# Import main user-facing classes and functions
from .solvers import UAG, SolutionPath, StrongRuleSolver
from .penalties import (
    soft_thresholding,
    SCAD, SCAD_grad, 
    MCP, MCP_grad
)
from .utils import (
    lambda_max_LM,
    lambda_max_logistic,
    add_intercept,
    standardize_data
)

# Import models for advanced users
from .models import LinearModel, LogisticModel

# For backward compatibility, create wrapper functions
def UAG_LM_SCAD_MCP(design_matrix, outcome, beta_0=None, tol=1e-4, maxit=1000,
                    lambda_=0.1, penalty="SCAD", a=3.7, gamma=2.0, 
                    L_convex=None, add_intercept_column=False):
    """Unified Accelerated Gradient for linear models with SCAD/MCP penalty.
    
    .. deprecated:: 0.9.6
        Use UAG class instead: UAG(model_type="linear").fit(X, y, lambda_)
    """
    import warnings
    warnings.warn("UAG_LM_SCAD_MCP is deprecated. Use UAG(model_type='linear') instead.",
                  DeprecationWarning, stacklevel=2)
    
    # Handle intercept
    if add_intercept_column:
        design_matrix = add_intercept(design_matrix)
        
    solver = UAG(model_type="linear", penalty=penalty, a=a, gamma=gamma,
                 tol=tol, maxit=maxit, L_convex=L_convex)
    solver.fit(design_matrix, outcome, lambda_, beta_init=beta_0)
    
    return np.concatenate([[solver.intercept_], solver.coef_])


def UAG_logistic_SCAD_MCP(design_matrix, outcome, beta_0=None, tol=1e-4, maxit=1000,
                          lambda_=0.1, penalty="SCAD", a=3.7, gamma=2.0,
                          L_convex=None, add_intercept_column=False):
    """Unified Accelerated Gradient for logistic models with SCAD/MCP penalty.
    
    .. deprecated:: 0.9.6
        Use UAG class instead: UAG(model_type="logistic").fit(X, y, lambda_)
    """
    import warnings
    warnings.warn("UAG_logistic_SCAD_MCP is deprecated. Use UAG(model_type='logistic') instead.",
                  DeprecationWarning, stacklevel=2)
    
    # Handle intercept
    if add_intercept_column:
        design_matrix = add_intercept(design_matrix)
        
    solver = UAG(model_type="logistic", penalty=penalty, a=a, gamma=gamma,
                 tol=tol, maxit=maxit, L_convex=L_convex)
    solver.fit(design_matrix, outcome, lambda_, beta_init=beta_0)
    
    return np.concatenate([[solver.intercept_], solver.coef_])


def solution_path_LM(design_matrix, outcome, lambda_, **kwargs):
    """Compute solution path for linear models.
    
    .. deprecated:: 0.9.6
        Use SolutionPath class instead: SolutionPath(model_type="linear").fit(X, y, lambdas)
    """
    import warnings
    warnings.warn("solution_path_LM is deprecated. Use SolutionPath(model_type='linear') instead.",
                  DeprecationWarning, stacklevel=2)
    
    solver = SolutionPath(model_type="linear", **kwargs)
    solver.fit(design_matrix, outcome, lambda_)
    # Return full coefficient matrix including intercept (old API format)
    return np.vstack([solver.intercept_path_, solver.coef_path_])


def solution_path_logistic(design_matrix, outcome, lambda_, **kwargs):
    """Compute solution path for logistic models.
    
    .. deprecated:: 0.9.6
        Use SolutionPath class instead: SolutionPath(model_type="logistic").fit(X, y, lambdas)
    """
    import warnings
    warnings.warn("solution_path_logistic is deprecated. Use SolutionPath(model_type='logistic') instead.",
                  DeprecationWarning, stacklevel=2)
    
    solver = SolutionPath(model_type="logistic", **kwargs)
    solver.fit(design_matrix, outcome, lambda_)
    # Return full coefficient matrix including intercept (old API format)
    return np.vstack([solver.intercept_path_, solver.coef_path_])


# List all public APIs
__all__ = [
    # Main solvers
    'UAG',
    'SolutionPath', 
    'StrongRuleSolver',
    
    # Penalties
    'soft_thresholding',
    'SCAD',
    'SCAD_grad',
    'MCP', 
    'MCP_grad',
    
    # Utilities
    'lambda_max_LM',
    'lambda_max_logistic',
    'add_intercept',
    'standardize_data',
    
    # Models
    'LinearModel',
    'LogisticModel',
    
    # Deprecated functions for backward compatibility
    'UAG_LM_SCAD_MCP',
    'UAG_logistic_SCAD_MCP',
    'solution_path_LM',
    'solution_path_logistic',
]