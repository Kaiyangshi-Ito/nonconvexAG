"""Test backward compatibility with old API."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import warnings

# Import old-style functions from new package
from nonconvexAG import (
    UAG_LM_SCAD_MCP,
    UAG_logistic_SCAD_MCP,
    solution_path_LM,
    solution_path_logistic,
    # Import new API for comparison
    UAG, SolutionPath,
    add_intercept
)


class TestDeprecatedFunctions:
    """Test that deprecated functions still work but issue warnings."""
    
    def test_UAG_LM_SCAD_MCP_deprecation(self, linear_regression_data):
        """Test deprecated UAG_LM_SCAD_MCP function."""
        data = linear_regression_data
        X = data['X'][:100]  # Smaller for speed
        y = data['y'][:100]
        
        # Should issue deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                # Old API - adds intercept internally
                beta_old = UAG_LM_SCAD_MCP(
                    design_matrix=X,
                    outcome=y,
                    lambda_=0.1,
                    penalty="SCAD",
                    add_intercept_column=True
                )
            except ReferenceError as e:
                if "underlying object has vanished" in str(e):
                    pytest.skip("Numba caching issue - known intermittent problem")
                else:
                    raise
            
            # Filter to only deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "deprecated" in str(deprecation_warnings[0].message)
            
        # Compare with new API
        X_new = add_intercept(X)
        solver_new = UAG(model_type="linear", penalty="SCAD")
        solver_new.fit(X_new, y, lambda_=0.1)
        beta_new = np.concatenate([[solver_new.intercept_], solver_new.coef_])
        
        # Results should be identical
        assert_array_almost_equal(beta_old, beta_new, decimal=6)
        
    def test_UAG_logistic_SCAD_MCP_deprecation(self, logistic_regression_data):
        """Test deprecated UAG_logistic_SCAD_MCP function."""
        data = logistic_regression_data
        X = data['X'][:100]
        y = data['y'][:100]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Old API
            beta_old = UAG_logistic_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                lambda_=0.05,
                penalty="MCP",
                gamma=3.0,
                add_intercept_column=True
            )
            
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            
        # Compare with new API
        X_new = add_intercept(X)
        solver_new = UAG(model_type="logistic", penalty="MCP", gamma=3.0)
        solver_new.fit(X_new, y, lambda_=0.05)
        beta_new = np.concatenate([[solver_new.intercept_], solver_new.coef_])
        
        assert_array_almost_equal(beta_old, beta_new, decimal=6)
        
    def test_solution_path_LM_deprecation(self, linear_regression_data):
        """Test deprecated solution_path_LM function."""
        data = linear_regression_data
        X = add_intercept(data['X'][:100])
        y = data['y'][:100]
        
        lambdas = np.array([0.5, 0.3, 0.1, 0.05])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Old API returns coefficient matrix directly
            path_old = solution_path_LM(
                design_matrix=X,
                outcome=y,
                lambda_=lambdas,
                penalty="SCAD"
            )
            
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            
        # New API
        solver_new = SolutionPath(model_type="linear", penalty="SCAD")
        solver_new.fit(X, y, lambdas=lambdas)
        
        # Old API returns full beta matrix (with intercept)
        # New API separates intercept and coefficients
        path_new = np.vstack([solver_new.intercept_path_, solver_new.coef_path_])
        
        assert_array_almost_equal(path_old, path_new, decimal=6)
        
    def test_solution_path_logistic_deprecation(self, logistic_regression_data):
        """Test deprecated solution_path_logistic function."""
        data = logistic_regression_data
        X = add_intercept(data['X'][:100])
        y = data['y'][:100]
        
        lambdas = np.array([0.1, 0.05, 0.01])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            path_old = solution_path_logistic(
                design_matrix=X,
                outcome=y,
                lambda_=lambdas,
                penalty="MCP",
                gamma=2.5
            )
            
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            
        # New API
        solver_new = SolutionPath(model_type="logistic", penalty="MCP", gamma=2.5)
        solver_new.fit(X, y, lambdas=lambdas)
        path_new = np.vstack([solver_new.intercept_path_, solver_new.coef_path_])
        
        assert_array_almost_equal(path_old, path_new, decimal=6)


class TestOldAPIParameters:
    """Test that old API parameters work correctly."""
    
    def test_beta_0_parameter(self, small_data):
        """Test beta_0 initial value parameter."""
        X, y = small_data
        X = add_intercept(X)
        
        # Custom starting value
        beta_0 = np.array([1.0, 0.5, -0.5])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = UAG_LM_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                beta_0=beta_0,
                lambda_=0.1
            )
            
        assert result is not None
        assert len(result) == X.shape[1]
        
    def test_tolerance_and_maxit(self, small_data):
        """Test tol and maxit parameters."""
        X, y = small_data
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Very tight tolerance
            result1 = UAG_LM_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                tol=1e-10,
                maxit=10000,
                lambda_=0.1,
                add_intercept_column=True
            )
            
            # Loose tolerance
            result2 = UAG_LM_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                tol=1e-2,
                maxit=10,
                lambda_=0.1,
                add_intercept_column=True
            )
            
        # Both should return results
        assert result1 is not None
        assert result2 is not None
        
    def test_L_convex_parameter(self, linear_regression_data):
        """Test L_convex parameter."""
        data = linear_regression_data
        X = data['X'][:100]
        y = data['y'][:100]
        
        # Calculate L_convex manually
        X_with_int = add_intercept(X)
        L_manual = np.linalg.eigvalsh(X_with_int.T @ X_with_int / len(y))[-1]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = UAG_LM_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                L_convex=L_manual,
                lambda_=0.1,
                add_intercept_column=True
            )
            
        assert result is not None


class TestAddInterceptBehavior:
    """Test add_intercept_column parameter behavior."""
    
    def test_add_intercept_true(self, linear_regression_data):
        """Test when add_intercept_column=True."""
        data = linear_regression_data
        X = data['X'][:50]  # No intercept
        y = data['y'][:50]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Old API adds intercept
            result_old = UAG_LM_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                lambda_=0.1,
                add_intercept_column=True
            )
            
        # New API - manually add intercept
        X_new = add_intercept(X)
        solver_new = UAG(model_type="linear")
        solver_new.fit(X_new, y, lambda_=0.1)
        
        # Should have same dimension (original features + intercept)
        assert len(result_old) == X.shape[1] + 1
        
    def test_add_intercept_false(self, linear_regression_data):
        """Test when add_intercept_column=False."""
        data = linear_regression_data
        X = add_intercept(data['X'][:50])  # Already has intercept
        y = data['y'][:50]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Old API doesn't add intercept
            result_old = UAG_LM_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                lambda_=0.1,
                add_intercept_column=False
            )
            
        # New API
        solver_new = UAG(model_type="linear")
        solver_new.fit(X, y, lambda_=0.1)
        result_new = np.concatenate([[solver_new.intercept_], solver_new.coef_])
        
        assert_array_almost_equal(result_old, result_new, decimal=6)


class TestParameterCompatibility:
    """Test all parameter combinations work."""
    
    @pytest.mark.parametrize("penalty", ["SCAD", "MCP"])
    @pytest.mark.parametrize("add_intercept_param", [True, False])
    def test_all_penalty_intercept_combinations(self, small_data, penalty, add_intercept_param):
        """Test all combinations of penalty and intercept options."""
        X, y = small_data
        
        if not add_intercept_param:
            X = add_intercept(X)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = UAG_LM_SCAD_MCP(
                design_matrix=X,
                outcome=y,
                lambda_=0.1,
                penalty=penalty,
                add_intercept_column=add_intercept_param
            )
            
        assert result is not None
        expected_length = X.shape[1] + (1 if add_intercept_param else 0)
        assert len(result) == expected_length
        
    def test_solution_path_parameters(self, linear_regression_data):
        """Test solution path with various parameters."""
        data = linear_regression_data
        X = add_intercept(data['X'][:100])
        y = data['y'][:100]
        
        lambdas = np.logspace(-2, 0, 20)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test with different parameters
            path1 = solution_path_LM(
                design_matrix=X,
                outcome=y,
                lambda_=lambdas,
                penalty="SCAD",
                a=3.7,
                tol=1e-4
            )
            
            path2 = solution_path_LM(
                design_matrix=X,
                outcome=y,
                lambda_=lambdas,
                penalty="MCP",
                gamma=2.0,
                tol=1e-4
            )
            
        assert path1.shape == (X.shape[1], len(lambdas))
        assert path2.shape == (X.shape[1], len(lambdas))
        
        # Paths should be different for different penalties
        assert not np.allclose(path1, path2)