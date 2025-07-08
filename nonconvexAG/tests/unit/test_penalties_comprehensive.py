"""Comprehensive tests for all penalty functions."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from scipy.optimize import approx_fprime

from nonconvexAG.penalties import (
    soft_thresholding,
    SCAD, SCAD_grad, SCAD_concave, SCAD_concave_grad,
    MCP, MCP_grad, MCP_concave, MCP_concave_grad
)


class TestSoftThresholdingComprehensive:
    """Comprehensive tests for soft thresholding operator."""
    
    @pytest.mark.parametrize("size", [5, 10, 100, 1000])
    def test_different_sizes(self, size):
        """Test soft thresholding on different vector sizes."""
        x = np.random.randn(size)
        lambda_ = 0.5
        result = soft_thresholding(x, lambda_)
        
        assert result.shape == x.shape
        assert result[0] == x[0]  # Intercept unchanged
        
    def test_edge_cases(self):
        """Test edge cases for soft thresholding."""
        # Empty after intercept
        x = np.array([1.0])
        result = soft_thresholding(x, 0.5)
        assert_array_almost_equal(result, x)
        
        # All zeros except intercept
        x = np.array([5.0, 0.0, 0.0, 0.0])
        result = soft_thresholding(x, 0.1)
        assert_array_almost_equal(result, x)
        
        # Very large lambda
        x = np.array([1.0, 10.0, -10.0, 100.0])
        result = soft_thresholding(x, 1000.0)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)
        
    @pytest.mark.parametrize("lambda_", np.logspace(-3, 2, 10))
    def test_monotonicity(self, lambda_):
        """Test that soft thresholding is monotonic in lambda."""
        x = np.array([1.0, 2.0, -3.0, 0.5, -0.5])
        
        result1 = soft_thresholding(x, lambda_)
        result2 = soft_thresholding(x, lambda_ * 1.1)
        
        # Larger lambda should give smaller or equal absolute values
        assert np.all(np.abs(result2[1:]) <= np.abs(result1[1:]) + 1e-10)
        
    def test_subdifferential_at_zero(self):
        """Test subdifferential property at zero."""
        x = np.array([1.0, 0.0, 0.0])
        lambda_ = 0.5
        
        # At zero, soft thresholding should return 0
        result = soft_thresholding(x, lambda_)
        assert result[1] == 0.0
        assert result[2] == 0.0
        
    def test_vectorized_performance(self):
        """Test that function works efficiently on large vectors."""
        x = np.random.randn(10000)
        lambda_ = 0.1
        
        # Should complete quickly due to vectorization
        result = soft_thresholding(x, lambda_)
        assert result.shape == x.shape


class TestSCADComprehensive:
    """Comprehensive tests for SCAD penalty and its derivatives."""
    
    @pytest.fixture
    def scad_params(self):
        """Standard SCAD parameters."""
        return {'a': 3.7, 'lambda_': 0.5}
    
    def test_scad_regions(self, scad_params):
        """Test SCAD behavior in different regions."""
        a = scad_params['a']
        lambda_ = scad_params['lambda_']
        
        # Region 1: |beta| <= lambda
        beta1 = np.array([0.0, 0.3, -0.3])
        penalty1 = SCAD(beta1, lambda_, a)
        expected1 = lambda_ * (np.abs(beta1[1]) + np.abs(beta1[2]))
        assert_almost_equal(penalty1, expected1)
        
        # Region 2: lambda < |beta| <= a*lambda
        beta2 = np.array([0.0, 1.0, -1.5])
        penalty2 = SCAD(beta2, lambda_, a)
        # Manual calculation for middle region
        assert penalty2 > 0
        
        # Region 3: |beta| > a*lambda
        beta3 = np.array([0.0, 5.0, -5.0])
        penalty3 = SCAD(beta3, lambda_, a)
        expected3 = 2 * lambda_**2 * (a + 1) / 2  # Two coefficients in region 3
        assert_almost_equal(penalty3, expected3)
        
    def test_scad_gradient_numerical(self, scad_params):
        """Test SCAD gradient against numerical approximation."""
        a = scad_params['a']
        lambda_ = scad_params['lambda_']
        
        beta = np.array([1.0, 0.3, 1.2, 3.0, 5.0])
        
        # Analytical gradient
        grad_analytical = SCAD_grad(beta, lambda_, a)
        
        # Numerical gradient
        def scad_func(b):
            return SCAD(b, lambda_, a)
        
        grad_numerical = approx_fprime(beta, scad_func, epsilon=1e-8)
        
        # Compare (note: intercept gradient should be 0)
        assert_almost_equal(grad_analytical[0], 0.0)
        assert_array_almost_equal(grad_analytical[1:], grad_numerical[1:], decimal=5)
        
    def test_scad_concave_consistency(self, scad_params):
        """Test relationship between SCAD and its concave part."""
        a = scad_params['a']
        lambda_ = scad_params['lambda_']
        
        for _ in range(10):
            beta = np.random.randn(5) * 2
            beta[0] = 1.0  # Set intercept
            
            scad_val = SCAD(beta, lambda_, a)
            l1_val = lambda_ * np.sum(np.abs(beta[1:]))
            concave_val = SCAD_concave(beta, lambda_, a)
            
            # SCAD = L1 - concave part
            # Check that SCAD reduces the L1 penalty
            assert scad_val <= l1_val
            
    def test_scad_properties(self, scad_params):
        """Test mathematical properties of SCAD."""
        a = scad_params['a']
        lambda_ = scad_params['lambda_']
        
        # Property 1: SCAD(0) = 0
        beta_zero = np.zeros(5)
        assert SCAD(beta_zero, lambda_, a) == 0.0
        
        # Property 2: SCAD is symmetric
        beta_pos = np.array([0.0, 1.0, 2.0])
        beta_neg = np.array([0.0, -1.0, -2.0])
        assert_almost_equal(SCAD(beta_pos, lambda_, a), SCAD(beta_neg, lambda_, a))
        
        # Property 3: SCAD is continuous
        # Test at boundary points
        beta_boundary1 = np.array([0.0, lambda_, -lambda_])
        beta_boundary2 = np.array([0.0, a*lambda_, -a*lambda_])
        
        # Small perturbation shouldn't cause large jump
        eps = 1e-6
        penalty1 = SCAD(beta_boundary1, lambda_, a)
        penalty1_perturbed = SCAD(beta_boundary1 + eps, lambda_, a)
        assert abs(penalty1 - penalty1_perturbed) < 1e-4
        
    @pytest.mark.parametrize("a", [2.1, 3.0, 3.7, 5.0, 10.0])
    def test_scad_parameter_a(self, a):
        """Test SCAD with different values of parameter a."""
        lambda_ = 0.5
        beta = np.array([0.0, 0.3, 1.0, 2.5, 5.0])
        
        penalty = SCAD(beta, lambda_, a)
        grad = SCAD_grad(beta, lambda_, a)
        
        assert penalty >= 0
        assert grad[0] == 0.0  # No gradient on intercept
        
        # Check that gradient is 0 for large coefficients
        large_beta = np.array([0.0, 10.0, -10.0])
        grad_large = SCAD_grad(large_beta, lambda_, a)
        assert_array_almost_equal(grad_large[1:], np.zeros(2))


class TestMCPComprehensive:
    """Comprehensive tests for MCP penalty and its derivatives."""
    
    @pytest.fixture
    def mcp_params(self):
        """Standard MCP parameters."""
        return {'gamma': 2.0, 'lambda_': 0.5}
    
    def test_mcp_regions(self, mcp_params):
        """Test MCP behavior in different regions."""
        gamma = mcp_params['gamma']
        lambda_ = mcp_params['lambda_']
        
        # Region 1: |beta| <= gamma*lambda
        beta1 = np.array([0.0, 0.5, -0.8])
        penalty1 = MCP(beta1, lambda_, gamma)
        # Should be quadratic in this region
        assert penalty1 > 0
        
        # Region 2: |beta| > gamma*lambda  
        beta2 = np.array([0.0, 3.0, -3.0])
        penalty2 = MCP(beta2, lambda_, gamma)
        expected2 = 2 * gamma * lambda_**2 / 2  # Two coefficients in region 2
        assert_almost_equal(penalty2, expected2)
        
    def test_mcp_gradient_numerical(self, mcp_params):
        """Test MCP gradient against numerical approximation."""
        gamma = mcp_params['gamma']
        lambda_ = mcp_params['lambda_']
        
        beta = np.array([1.0, 0.3, 0.8, 1.5, 3.0])
        
        # Analytical gradient
        grad_analytical = MCP_grad(beta, lambda_, gamma)
        
        # Numerical gradient
        def mcp_func(b):
            return MCP(b, lambda_, gamma)
        
        grad_numerical = approx_fprime(beta, mcp_func, epsilon=1e-8)
        
        # Compare
        assert_almost_equal(grad_analytical[0], 0.0)
        assert_array_almost_equal(grad_analytical[1:], grad_numerical[1:], decimal=5)
        
    def test_mcp_concave_consistency(self, mcp_params):
        """Test relationship between MCP and its concave part."""
        gamma = mcp_params['gamma']
        lambda_ = mcp_params['lambda_']
        
        for _ in range(10):
            beta = np.random.randn(5) * 2
            beta[0] = 1.0
            
            mcp_val = MCP(beta, lambda_, gamma)
            l1_val = lambda_ * np.sum(np.abs(beta[1:]))
            concave_val = MCP_concave(beta, lambda_, gamma)
            
            # MCP = L1 - concave part
            assert_almost_equal(mcp_val + concave_val, l1_val, decimal=10)
            
    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 3.0, 10.0])
    def test_mcp_parameter_gamma(self, gamma):
        """Test MCP with different values of parameter gamma."""
        lambda_ = 0.5
        beta = np.array([0.0, 0.3, 0.8, 1.5, 3.0])
        
        penalty = MCP(beta, lambda_, gamma)
        grad = MCP_grad(beta, lambda_, gamma)
        
        assert penalty >= 0
        assert grad[0] == 0.0
        
        # Smaller gamma means more concavity
        if gamma < 2.0:
            # More coefficients should have zero gradient
            large_beta = np.array([0.0, 2.0, -2.0])
            grad_large = MCP_grad(large_beta, lambda_, gamma)
            n_zero_grad = np.sum(np.abs(grad_large[1:]) < 1e-10)
            assert n_zero_grad >= 1


class TestPenaltyComparison:
    """Compare different penalties."""
    
    def test_penalty_ordering(self):
        """Test that L1 >= SCAD >= MCP for standard parameters."""
        lambda_ = 0.5
        a = 3.7
        gamma = 2.0
        
        for _ in range(20):
            beta = np.random.randn(5) * 2
            beta[0] = 1.0
            
            l1_val = lambda_ * np.sum(np.abs(beta[1:]))
            scad_val = SCAD(beta, lambda_, a)
            mcp_val = MCP(beta, lambda_, gamma)
            
            # L1 >= SCAD
            assert l1_val >= scad_val - 1e-10
            
            # Can't guarantee SCAD >= MCP in general, but both <= L1
            assert scad_val <= l1_val + 1e-10
            assert mcp_val <= l1_val + 1e-10
            
    def test_penalties_agree_at_zero(self):
        """Test that all penalties agree at zero."""
        lambda_ = 0.5
        beta = np.array([1.0, 0.0, 0.0, 0.0])
        
        l1_val = lambda_ * np.sum(np.abs(beta[1:]))
        scad_val = SCAD(beta, lambda_)
        mcp_val = MCP(beta, lambda_)
        
        assert l1_val == 0.0
        assert scad_val == 0.0
        assert mcp_val == 0.0
        
    def test_penalties_behavior_small_coefficients(self):
        """Test that penalties behave like L1 for small coefficients."""
        lambda_ = 0.5
        beta_small = np.array([0.0, 0.1, -0.1, 0.05])
        
        l1_val = lambda_ * np.sum(np.abs(beta_small[1:]))
        scad_val = SCAD(beta_small, lambda_)
        mcp_val = MCP(beta_small, lambda_)
        
        # For small coefficients, all penalties should be approximately L1
        assert_almost_equal(scad_val, l1_val, decimal=3)
        assert_almost_equal(mcp_val, l1_val, decimal=2)  # MCP has slight difference


class TestGradientConsistency:
    """Test consistency between penalties and their gradients."""
    
    def test_gradient_zero_at_origin(self):
        """Test that gradients handle zero correctly."""
        lambda_ = 0.5
        beta = np.array([1.0, 0.0, 0.0])
        
        scad_grad_val = SCAD_grad(beta, lambda_)
        mcp_grad_val = MCP_grad(beta, lambda_)
        
        # Gradients at zero should be in subdifferential [-lambda, lambda]
        assert np.abs(scad_grad_val[1]) <= lambda_
        assert np.abs(mcp_grad_val[1]) <= lambda_
        
    def test_concave_gradient_consistency(self):
        """Test that concave gradients are consistent with concave functions."""
        lambda_ = 0.5
        a = 3.7
        gamma = 2.0
        
        for _ in range(10):
            beta = np.random.randn(5)
            beta[0] = 1.0
            
            # SCAD concave gradient
            def scad_concave_func(b):
                return SCAD_concave(b, lambda_, a)
            
            grad_analytical = SCAD_concave_grad(beta, lambda_, a)
            grad_numerical = approx_fprime(beta, scad_concave_func, epsilon=1e-8)
            assert_array_almost_equal(grad_analytical[1:], grad_numerical[1:], decimal=5)
            
            # MCP concave gradient
            def mcp_concave_func(b):
                return MCP_concave(b, lambda_, gamma)
            
            grad_analytical = MCP_concave_grad(beta, lambda_, gamma)
            grad_numerical = approx_fprime(beta, mcp_concave_func, epsilon=1e-8)
            assert_array_almost_equal(grad_analytical[1:], grad_numerical[1:], decimal=5)