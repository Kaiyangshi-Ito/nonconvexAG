"""Unit tests for penalty functions."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from nonconvexAG.penalties import (
    soft_thresholding, 
    SCAD, SCAD_grad, SCAD_concave, SCAD_concave_grad,
    MCP, MCP_grad, MCP_concave, MCP_concave_grad
)


class TestSoftThresholding:
    """Test L1 soft thresholding operator."""
    
    def test_basic_functionality(self):
        """Test basic soft thresholding behavior."""
        x = np.array([5.0, 2.0, -3.0, 0.5, -0.5])
        lambda_ = 1.0
        
        result = soft_thresholding(x, lambda_)
        
        # First element (intercept) should be unchanged
        assert result[0] == x[0]
        
        # Check soft thresholding logic
        assert result[1] == 1.0  # 2 - 1 = 1
        assert result[2] == -2.0  # -3 + 1 = -2
        assert result[3] == 0.0   # |0.5| < 1
        assert result[4] == 0.0   # |-0.5| < 1
        
    def test_zero_lambda(self):
        """Test with lambda = 0 (no shrinkage)."""
        x = np.array([1.0, 2.0, -3.0])
        result = soft_thresholding(x, 0.0)
        assert_array_almost_equal(result, x)
        
    def test_large_lambda(self):
        """Test with large lambda (all zeros except intercept)."""
        x = np.array([1.0, 2.0, -3.0, 0.5])
        result = soft_thresholding(x, 10.0)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)


class TestSCAD:
    """Test SCAD penalty functions."""
    
    def test_scad_value(self):
        """Test SCAD penalty calculation."""
        beta = np.array([1.0, 0.5, 2.0, 5.0])  # intercept + 3 coefficients
        lambda_ = 1.0
        a = 3.7
        
        penalty = SCAD(beta, lambda_, a)
        
        # Manually calculate expected value
        # beta[1] = 0.5: |0.5| <= 1, so penalty = 1 * 0.5 = 0.5
        # beta[2] = 2.0: 1 < |2| <= 3.7, so penalty = (2*3.7*1*2 - 4 - 1)/(2*(3.7-1)) ≈ 3.074
        # beta[3] = 5.0: |5| > 3.7, so penalty = 1²*(3.7+1)/2 = 2.35
        # Total ≈ 0.5 + 3.074 + 2.35 = 5.924
        
        assert penalty > 0
        
    def test_scad_gradient(self):
        """Test SCAD gradient calculation."""
        beta = np.array([1.0, 0.5, 2.0, 5.0])
        lambda_ = 1.0
        a = 3.7
        
        grad = SCAD_grad(beta, lambda_, a)
        
        # Intercept gradient should be 0
        assert grad[0] == 0.0
        
        # Check gradient signs match beta signs
        assert np.sign(grad[1]) == np.sign(beta[1])
        assert grad[3] == 0.0  # |beta[3]| > a*lambda
        
    def test_scad_concave_consistency(self):
        """Test that SCAD = L1 - SCAD_concave."""
        beta = np.array([1.0, 0.5, 2.0, -3.0])
        lambda_ = 1.0
        a = 3.7
        
        scad_val = SCAD(beta, lambda_, a)
        l1_val = lambda_ * np.sum(np.abs(beta[1:]))
        concave_val = SCAD_concave(beta, lambda_, a)
        
        # SCAD should equal L1 minus concave part
        assert_almost_equal(scad_val, l1_val - concave_val, decimal=6)


class TestMCP:
    """Test MCP penalty functions."""
    
    def test_mcp_value(self):
        """Test MCP penalty calculation."""
        beta = np.array([1.0, 0.5, 3.0, 5.0])  # intercept + 3 coefficients
        lambda_ = 1.0
        gamma = 2.0
        
        penalty = MCP(beta, lambda_, gamma)
        
        # Manually calculate expected value
        # beta[1] = 0.5: |0.5| <= 2*1, so penalty = 1*0.5 - 0.25/2 = 0.375
        # beta[2] = 3.0: |3| > 2*1, so penalty = 2*1²/2 = 1
        # beta[3] = 5.0: |5| > 2*1, so penalty = 2*1²/2 = 1
        # Total = 0.375 + 1 + 1 = 2.375
        
        assert penalty > 0
        
    def test_mcp_gradient(self):
        """Test MCP gradient calculation."""
        beta = np.array([1.0, 0.5, 3.0, -4.0])
        lambda_ = 1.0
        gamma = 2.0
        
        grad = MCP_grad(beta, lambda_, gamma)
        
        # Intercept gradient should be 0
        assert grad[0] == 0.0
        
        # Check gradient for |beta| > gamma*lambda
        assert grad[2] == 0.0  # |3| > 2*1
        assert grad[3] == 0.0  # |-4| > 2*1
        
    def test_mcp_concave_consistency(self):
        """Test that MCP = L1 - MCP_concave."""
        beta = np.array([1.0, 0.5, 1.5, -3.0])
        lambda_ = 1.0
        gamma = 2.0
        
        mcp_val = MCP(beta, lambda_, gamma)
        l1_val = lambda_ * np.sum(np.abs(beta[1:]))
        concave_val = MCP_concave(beta, lambda_, gamma)
        
        # MCP should equal L1 minus concave part
        assert_almost_equal(mcp_val, l1_val - concave_val, decimal=6)


class TestPenaltyProperties:
    """Test general properties of penalty functions."""
    
    def test_penalties_zero_at_origin(self):
        """Test that penalties are zero when all coefficients are zero."""
        beta = np.zeros(5)
        lambda_ = 1.0
        
        assert SCAD(beta, lambda_) == 0.0
        assert MCP(beta, lambda_) == 0.0
        
    def test_penalties_nonnegative(self):
        """Test that penalties are always non-negative."""
        beta = np.random.randn(10)
        lambda_ = 0.5
        
        assert SCAD(beta, lambda_) >= 0
        assert MCP(beta, lambda_) >= 0
        
    def test_gradient_zero_at_origin(self):
        """Test that gradients have correct behavior at origin."""
        beta = np.array([1.0, 0.0, 0.0, 0.0])
        lambda_ = 1.0
        
        scad_grad = SCAD_grad(beta, lambda_)
        mcp_grad = MCP_grad(beta, lambda_)
        
        # Gradients should be subgradients at 0
        assert scad_grad[0] == 0  # intercept
        assert abs(scad_grad[1]) <= lambda_
        assert abs(mcp_grad[1]) <= lambda_