"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from scipy.linalg import toeplitz


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def linear_regression_data(random_seed):
    """Generate synthetic data for linear regression tests.
    
    Returns
    -------
    dict with keys:
        X : np.ndarray of shape (1000, 100)
            Design matrix (without intercept).
        y : np.ndarray of shape (1000,)
            Response vector.
        true_beta : np.ndarray of shape (100,)
            True coefficients (sparse).
        X_test : np.ndarray of shape (200, 100)
            Test design matrix.
        y_test : np.ndarray of shape (200,)
            Test response.
    """
    n_samples = 1000
    n_features = 100
    n_informative = 10
    
    # True sparse coefficients
    true_beta = np.zeros(n_features)
    true_beta[:n_informative] = np.random.randn(n_informative) * 3
    
    # Correlation structure
    rho = 0.5
    cov = toeplitz(rho ** np.arange(n_features))
    
    # Generate features
    X = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)
    
    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Generate response
    signal = X @ true_beta
    signal_std = np.std(signal)
    if signal_std == 0:
        noise_std = 0.1  # If signal is zero, use fixed noise
    else:
        noise_std = signal_std / 3  # SNR ≈ 3
    y = signal + np.random.normal(0, noise_std, n_samples)
    
    # Test data
    X_test = np.random.multivariate_normal(np.zeros(n_features), cov, 200)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    y_test = X_test @ true_beta + np.random.normal(0, noise_std, 200)
    
    return {
        'X': X,
        'y': y,
        'true_beta': true_beta,
        'X_test': X_test,
        'y_test': y_test,
        'n_informative': n_informative
    }


@pytest.fixture
def logistic_regression_data(random_seed):
    """Generate synthetic data for logistic regression tests.
    
    Returns
    -------
    dict with keys:
        X : np.ndarray of shape (1000, 50)
            Design matrix (without intercept).
        y : np.ndarray of shape (1000,)
            Binary response vector.
        true_beta : np.ndarray of shape (50,)
            True coefficients (sparse).
        X_test : np.ndarray of shape (200, 50)
            Test design matrix.
        y_test : np.ndarray of shape (200,)
            Test binary response.
    """
    n_samples = 1000
    n_features = 50
    n_informative = 8
    
    # True sparse coefficients
    true_beta = np.zeros(n_features)
    true_beta[:n_informative] = np.random.randn(n_informative) * 1.5
    
    # Correlation structure
    rho = 0.3
    cov = toeplitz(rho ** np.arange(n_features))
    
    # Generate features
    X = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)
    
    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Generate binary response
    logits = X @ true_beta
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)
    
    # Test data
    X_test = np.random.multivariate_normal(np.zeros(n_features), cov, 200)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    logits_test = X_test @ true_beta
    probs_test = 1 / (1 + np.exp(-logits_test))
    y_test = np.random.binomial(1, probs_test)
    
    return {
        'X': X,
        'y': y,
        'true_beta': true_beta,
        'X_test': X_test,
        'y_test': y_test,
        'n_informative': n_informative
    }


@pytest.fixture
def small_data():
    """Small dataset for quick tests."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    y = np.array([1.5, 3.5, 5.5, 7.5])
    return X, y


@pytest.fixture(params=["SCAD", "MCP"])
def penalty_type(request):
    """Parametrized fixture for penalty types."""
    return request.param


@pytest.fixture(params=[0.01, 0.1, 0.5])
def lambda_value(request):
    """Parametrized fixture for lambda values."""
    return request.param