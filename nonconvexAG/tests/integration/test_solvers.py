"""Integration tests for optimization solvers."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import time

from nonconvexAG.solvers import UAG
from nonconvexAG.utils import add_intercept, lambda_max_LM, lambda_max_logistic


class TestUAGSolver:
    """Test UAG solver end-to-end."""
    
    def test_linear_regression_convergence(self, linear_regression_data):
        """Test UAG convergence on linear regression."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        # Calculate appropriate lambda
        lambda_max = lambda_max_LM(X, y)
        lambda_ = lambda_max * 0.1
        
        # Fit model
        solver = UAG(model_type="linear", penalty="SCAD", tol=1e-6)
        solver.fit(X, y, lambda_)
        
        # Check convergence
        assert solver.converged_
        assert solver.n_iter_ < solver.maxit
        
        # Check sparsity
        n_nonzero = np.sum(np.abs(solver.coef_) > 1e-6)
        assert n_nonzero <= data['n_informative'] * 2  # Some tolerance
        
    def test_logistic_regression_convergence(self, logistic_regression_data):
        """Test UAG convergence on logistic regression."""
        data = logistic_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        # Calculate appropriate lambda
        lambda_max = lambda_max_logistic(X, y)
        lambda_ = lambda_max * 0.05
        
        # Fit model
        solver = UAG(model_type="logistic", penalty="MCP", gamma=3.0, tol=1e-5)
        solver.fit(X, y, lambda_)
        
        # Check convergence
        assert solver.converged_
        
        # Check predictions are valid probabilities
        X_test = add_intercept(data['X_test'])
        predictions = solver.predict(X_test)
        assert np.all((predictions >= 0) & (predictions <= 1))
        
    def test_different_penalties(self, linear_regression_data, penalty_type):
        """Test different penalty types."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_max = lambda_max_LM(X, y)
        lambda_ = lambda_max * 0.1
        
        # Fit with different penalties
        solver = UAG(model_type="linear", penalty=penalty_type)
        solver.fit(X, y, lambda_)
        
        assert solver.converged_
        assert solver.coef_ is not None
        
    def test_warm_start(self, linear_regression_data):
        """Test warm start functionality."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_max = lambda_max_LM(X, y)
        
        # First fit
        solver = UAG(model_type="linear", penalty="SCAD")
        solver.fit(X, y, lambda_max * 0.2)
        beta_first = np.concatenate([[solver.intercept_], solver.coef_])
        
        # Second fit with warm start
        solver.fit(X, y, lambda_max * 0.15, beta_init=beta_first)
        
        # Should converge faster with warm start
        assert solver.n_iter_ < 80  # Allow some flexibility
        
    def test_score_methods(self, linear_regression_data, logistic_regression_data):
        """Test scoring methods."""
        # Linear regression - R² score
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        solver = UAG(model_type="linear")
        solver.fit(X, y, lambda_max_LM(X, y) * 0.1)
        
        r2_train = solver.score(X, y)
        assert 0 <= r2_train <= 1
        
        # Logistic regression - accuracy
        data = logistic_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        solver = UAG(model_type="logistic")
        solver.fit(X, y, lambda_max_logistic(X, y) * 0.05)
        
        accuracy = solver.score(X, y)
        assert 0 <= accuracy <= 1
        assert accuracy > 0.5  # Better than random
        
    def test_verbose_output(self, small_data, capsys):
        """Test verbose output."""
        X, y = small_data
        X = add_intercept(X)
        
        solver = UAG(model_type="linear", verbose=True, maxit=20)
        solver.fit(X, y, 0.1)
        
        captured = capsys.readouterr()
        # Check that verbose output shows optimization status
        assert "Optimization" in captured.out
        assert "iterations" in captured.out
        
    def test_l_convex_provided(self, linear_regression_data):
        """Test providing L_convex manually."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        # Calculate L_convex manually
        L_manual = np.linalg.eigvalsh(X.T @ X / len(y))[-1]
        
        # Fit with provided L_convex
        solver = UAG(model_type="linear", L_convex=L_manual)
        solver.fit(X, y, 0.1)
        
        assert solver.converged_
        assert solver.L_convex == L_manual


class TestSolverRobustness:
    """Test solver robustness to edge cases."""
    
    def test_perfect_separation(self):
        """Test logistic regression with perfect separation."""
        # Create perfectly separable data
        X = np.array([[1, -2], [1, -1], [1, 1], [1, 2]])
        y = np.array([0, 0, 1, 1])
        
        solver = UAG(model_type="logistic", penalty="SCAD", tol=1e-3, maxit=5000)
        solver.fit(X, y, 0.1)  # Use larger lambda for regularization
        
        # Should still converge
        assert solver.converged_
        
        # Should separate classes
        predictions = solver.predict(X)
        assert np.array_equal(predictions >= 0.5, y)
        
    def test_multicollinearity(self):
        """Test with highly correlated features."""
        n = 100
        X1 = np.random.randn(n, 1)
        X2 = X1 + 0.01 * np.random.randn(n, 1)  # Almost identical
        X = np.column_stack([np.ones(n), X1, X2, np.random.randn(n, 5)])
        
        true_beta = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y = X @ true_beta + 0.1 * np.random.randn(n)
        
        solver = UAG(model_type="linear", penalty="MCP")
        solver.fit(X, y, 0.1)
        
        # Should converge despite multicollinearity
        assert solver.converged_
        
        # One of the correlated features should be selected
        assert (abs(solver.coef_[0]) > 0.1) or (abs(solver.coef_[1]) > 0.1)
        
    def test_all_zeros_lambda(self):
        """Test with lambda that zeros all coefficients."""
        X = np.random.randn(50, 10)
        X = add_intercept(X)
        y = np.random.randn(50)
        
        # Use very large lambda
        lambda_huge = lambda_max_LM(X, y) * 10
        
        solver = UAG(model_type="linear")
        solver.fit(X, y, lambda_huge)
        
        # All coefficients except intercept should be zero
        assert np.allclose(solver.coef_, 0, atol=1e-10)
        assert solver.intercept_ == pytest.approx(np.mean(y), abs=1e-6)