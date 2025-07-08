"""Unit tests for utility functions."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import warnings

from nonconvexAG.utils import (
    lambda_max_LM, lambda_max_logistic,
    add_intercept, standardize_data, check_design_matrix,
    validate_inputs, validate_model_type, check_convergence
)
from nonconvexAG.utils.lambda_utils import generate_lambda_sequence


class TestLambdaUtils:
    """Test lambda parameter utilities."""
    
    def test_lambda_max_lm(self):
        """Test lambda_max calculation for linear models."""
        # Simple case where we know the answer
        X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        
        lambda_max = lambda_max_LM(X, y)
        
        # Should be positive
        assert lambda_max > 0
        
        # With standardized data centered at 0
        X_centered = X.astype(np.float64, copy=True)
        X_centered[:, 1] -= X_centered[:, 1].mean()
        y_centered = y - y.mean()
        
        lambda_max_centered = lambda_max_LM(X_centered, y_centered)
        assert lambda_max_centered > 0
        
    def test_lambda_max_logistic(self):
        """Test lambda_max calculation for logistic models."""
        X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
        y = np.array([0, 0, 1, 1])
        
        lambda_max = lambda_max_logistic(X, y)
        
        # Should be positive
        assert lambda_max > 0
        
        # Edge cases
        y_all_ones = np.ones(4)
        lambda_max_edge = lambda_max_logistic(X, y_all_ones)
        assert lambda_max_edge > 0
        
    def test_generate_lambda_sequence(self):
        """Test lambda sequence generation."""
        lambda_max = 1.0
        
        # Log scale
        seq_log = generate_lambda_sequence(lambda_max, lambda_min_ratio=0.01, 
                                         n_lambda=10, log_scale=True)
        assert len(seq_log) == 10
        assert seq_log[0] == pytest.approx(lambda_max)
        assert seq_log[-1] == pytest.approx(0.01)
        assert np.all(np.diff(seq_log) < 0)  # Decreasing
        
        # Linear scale
        seq_lin = generate_lambda_sequence(lambda_max, lambda_min_ratio=0.01,
                                         n_lambda=10, log_scale=False)
        assert len(seq_lin) == 10
        assert seq_lin[0] == pytest.approx(lambda_max)
        assert seq_lin[-1] == pytest.approx(0.01)


class TestDataUtils:
    """Test data preprocessing utilities."""
    
    def test_add_intercept(self):
        """Test adding intercept column."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_with_intercept = add_intercept(X)
        
        assert X_with_intercept.shape == (3, 3)
        assert_array_almost_equal(X_with_intercept[:, 0], np.ones(3))
        assert_array_almost_equal(X_with_intercept[:, 1:], X)
        
    def test_standardize_data(self):
        """Test data standardization."""
        # With intercept column
        X = np.array([[1, 2, 4], [1, 4, 8], [1, 6, 12]])
        y = np.array([1, 2, 3])
        
        X_std, y_centered = standardize_data(X, y, center=True, scale=True)
        
        # Intercept should be unchanged
        assert_array_almost_equal(X_std[:, 0], np.ones(3))
        
        # Other columns should have mean 0, std 1
        assert_almost_equal(X_std[:, 1].mean(), 0)
        assert_almost_equal(X_std[:, 1].std(), 1)
        assert_almost_equal(X_std[:, 2].mean(), 0)
        assert_almost_equal(X_std[:, 2].std(), 1)
        
        # y should be centered
        assert_almost_equal(y_centered.mean(), 0)
        
    def test_check_design_matrix(self):
        """Test design matrix validation."""
        # Valid matrix
        X_valid = np.array([[1, 2], [3, 4]])
        check_design_matrix(X_valid)  # Should not raise
        
        # Invalid cases
        with pytest.raises(ValueError, match="2-dimensional"):
            check_design_matrix(np.array([1, 2, 3]))
            
        with pytest.raises(ValueError, match="NaN"):
            check_design_matrix(np.array([[1, np.nan], [3, 4]]))
            
        with pytest.raises(ValueError, match="infinite"):
            check_design_matrix(np.array([[1, np.inf], [3, 4]]))
            
        # Warning for n < p
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_design_matrix(np.ones((2, 5)))
            assert len(w) == 1
            assert "overfitting" in str(w[0].message)


class TestValidation:
    """Test input validation functions."""
    
    def test_validate_inputs(self):
        """Test comprehensive input validation."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        
        # Valid inputs
        validate_inputs(X, y, "SCAD", 0.1, a=3.7, gamma=2.0, 
                       tol=1e-4, maxit=1000)
        
        # Invalid penalty
        with pytest.raises(ValueError, match="penalty must be"):
            validate_inputs(X, y, "INVALID", 0.1)
            
        # Negative lambda
        with pytest.raises(ValueError, match="non-negative"):
            validate_inputs(X, y, "SCAD", -0.1)
            
        # Invalid SCAD parameter
        with pytest.raises(ValueError, match="a.*must be > 2"):
            validate_inputs(X, y, "SCAD", 0.1, a=1.5)
            
        # Invalid MCP parameter
        with pytest.raises(ValueError, match="gamma.*must be > 0"):
            validate_inputs(X, y, "MCP", 0.1, gamma=-1)
            
    def test_validate_model_type(self):
        """Test model type validation."""
        # Valid linear model
        y_continuous = np.array([1.5, 2.5, 3.5])
        validate_model_type(y_continuous, "linear")  # Should not raise
        
        # Valid logistic model
        y_binary = np.array([0, 1, 0, 1])
        validate_model_type(y_binary, "logistic")  # Should not raise
        
        # Invalid logistic model
        y_invalid = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            validate_model_type(y_invalid, "logistic")
            
    def test_check_convergence(self):
        """Test convergence checking."""
        beta_old = np.array([1.0, 2.0, 3.0])
        beta_new = np.array([1.0001, 2.0001, 3.0001])
        
        # Should converge with loose tolerance
        assert check_convergence(beta_old, beta_new, tol=0.001, 
                               iteration=10, maxit=100)
        
        # Should not converge with tight tolerance
        assert not check_convergence(beta_old, beta_new, tol=0.00001,
                                   iteration=10, maxit=100)
        
        # Should warn and return True at max iterations
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_convergence(beta_old, beta_new, tol=0.00001,
                                     iteration=100, maxit=100)
            assert result
            assert len(w) == 1
            assert "Maximum iterations" in str(w[0].message)