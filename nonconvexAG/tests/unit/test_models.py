"""Unit tests for model implementations."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from nonconvexAG.models import LinearModel, LogisticModel
from nonconvexAG.models.linear import (
    linear_loss_jit, linear_gradient_jit, update_smooth_grad_convex_LM
)
from nonconvexAG.models.logistic import (
    logistic_loss_jit, logistic_gradient_jit, sigmoid_jit,
    update_smooth_grad_convex_logistic
)


class TestLinearModel:
    """Test linear regression model."""
    
    def test_loss_calculation(self):
        """Test squared error loss calculation."""
        model = LinearModel()
        
        # Simple case
        X = np.array([[1, 0], [1, 1], [1, 2]])
        y = np.array([1.0, 2.0, 3.0])
        beta = np.array([1.0, 1.0])
        
        loss = model.loss(X, y, beta)
        
        # Predictions: [1, 2, 3]
        # Residuals: [0, 0, 0]
        # Loss: 0
        assert loss == pytest.approx(0.0)
        
        # Non-zero loss
        beta_wrong = np.array([0.0, 1.0])
        loss_wrong = model.loss(X, y, beta_wrong)
        assert loss_wrong > 0
        
    def test_gradient_calculation(self):
        """Test gradient calculation."""
        model = LinearModel()
        
        X = np.array([[1, 0], [1, 1], [1, 2]])
        y = np.array([1.0, 3.0, 5.0])
        beta = np.array([0.0, 2.0])
        
        grad = model.gradient(X, y, beta)
        
        # Should have same shape as beta
        assert grad.shape == beta.shape
        
        # At optimal beta, gradient should be near zero
        beta_opt = np.array([1.0, 2.0])
        grad_opt = model.gradient(X, y, beta_opt)
        assert_array_almost_equal(grad_opt, np.zeros(2), decimal=10)
        
    def test_predict(self):
        """Test prediction."""
        model = LinearModel()
        
        X = np.array([[1, 2], [1, 3], [1, 4]])
        beta = np.array([1.0, 0.5])
        
        predictions = model.predict(X, beta)
        expected = np.array([2.0, 2.5, 3.0])
        assert_array_almost_equal(predictions, expected)
        
    def test_jit_functions(self):
        """Test JIT-compiled functions match class methods."""
        model = LinearModel()
        
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        beta = np.random.randn(10)
        
        # Loss
        loss_class = model.loss(X, y, beta)
        loss_jit = linear_loss_jit(X, y, beta)
        assert_almost_equal(loss_class, loss_jit)
        
        # Gradient
        grad_class = model.gradient(X, y, beta)
        grad_jit = linear_gradient_jit(X, y, beta)
        assert_array_almost_equal(grad_class, grad_jit)
        
    def test_L_convex_calculation(self):
        """Test L-smoothness constant calculation."""
        model = LinearModel()
        
        # Identity matrix case
        X = np.eye(5)
        L = model.calculate_L_convex(X)
        assert L == pytest.approx(1/5)
        
        # General case
        X = np.random.randn(100, 10)
        L = model.calculate_L_convex(X)
        assert L > 0


class TestLogisticModel:
    """Test logistic regression model."""
    
    def test_loss_calculation(self):
        """Test negative log-likelihood calculation."""
        model = LogisticModel()
        
        # Perfect predictions
        X = np.array([[1, -10], [1, 10]])
        y = np.array([0, 1])
        beta = np.array([0.0, 1.0])
        
        loss = model.loss(X, y, beta)
        assert loss == pytest.approx(0.0, abs=1e-4)  # Allow small numerical error
        
        # Worst predictions
        beta_wrong = np.array([0.0, -1.0])
        loss_wrong = model.loss(X, y, beta_wrong)
        assert loss_wrong > loss
        
    def test_gradient_calculation(self):
        """Test gradient of negative log-likelihood."""
        model = LogisticModel()
        
        X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
        y = np.array([0, 0, 1, 1])
        beta = np.array([0.0, 0.0])
        
        grad = model.gradient(X, y, beta)
        
        # Should have same shape as beta
        assert grad.shape == beta.shape
        
        # Gradient should point in direction of improvement
        beta_updated = beta - 0.1 * grad
        loss_before = model.loss(X, y, beta)
        loss_after = model.loss(X, y, beta_updated)
        assert loss_after < loss_before
        
    def test_predict_proba(self):
        """Test probability predictions."""
        model = LogisticModel()
        
        X = np.array([[1, -2], [1, 0], [1, 2]])
        beta = np.array([0.0, 1.0])
        
        probs = model.predict_proba(X, beta)
        
        # Check bounds
        assert np.all((probs >= 0) & (probs <= 1))
        
        # Check ordering (higher X -> higher prob)
        assert probs[0] < probs[1] < probs[2]
        
    def test_predict(self):
        """Test class predictions."""
        model = LogisticModel()
        
        X = np.array([[1, -2], [1, 0], [1, 2]])
        beta = np.array([0.0, 1.0])
        
        predictions = model.predict(X, beta)
        # At X=0, probability is exactly 0.5, which rounds to 1
        expected = np.array([0, 1, 1])  # Based on 0.5 threshold
        assert_array_almost_equal(predictions, expected)
        
    def test_sigmoid_stability(self):
        """Test sigmoid function numerical stability."""
        # Extreme values
        z_extreme = np.array([-1000, -500, 0, 500, 1000])
        probs = sigmoid_jit(z_extreme)
        
        # Should not have NaN or inf
        assert np.all(np.isfinite(probs))
        
        # Should be in [0, 1]
        assert np.all((probs >= 0) & (probs <= 1))
        
        # Check approximate values
        assert probs[0] == pytest.approx(0.0, abs=1e-10)
        assert probs[2] == pytest.approx(0.5)
        assert probs[4] == pytest.approx(1.0, abs=1e-10)
        
    def test_jit_functions(self):
        """Test JIT-compiled functions match class methods."""
        model = LogisticModel()
        
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)
        beta = np.random.randn(10) * 0.1  # Small values for stability
        
        # Loss
        loss_class = model.loss(X, y, beta)
        loss_jit = logistic_loss_jit(X, y, beta)
        assert_almost_equal(loss_class, loss_jit, decimal=5)
        
        # Gradient
        grad_class = model.gradient(X, y, beta)
        grad_jit = logistic_gradient_jit(X, y, beta)
        assert_array_almost_equal(grad_class, grad_jit, decimal=5)
        
    def test_intercept_initialization(self):
        """Test intercept initialization."""
        model = LogisticModel()
        
        # Balanced case
        y_balanced = np.array([0, 0, 1, 1])
        intercept = model._compute_intercept(y_balanced)
        assert intercept == pytest.approx(0.0)  # log(0.5/0.5) = 0
        
        # Imbalanced case
        y_imbalanced = np.array([0, 0, 0, 1])
        intercept = model._compute_intercept(y_imbalanced)
        assert intercept < 0  # log(0.25/0.75) < 0