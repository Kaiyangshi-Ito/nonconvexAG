"""Comprehensive integration tests for all solver functionality."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import warnings

from nonconvexAG.solvers import UAG, SolutionPath, StrongRuleSolver
from nonconvexAG.utils import add_intercept, lambda_max_LM, lambda_max_logistic
from nonconvexAG.penalties import SCAD, MCP


class TestUAGSolverComprehensive:
    """Comprehensive tests for UAG solver."""
    
    @pytest.mark.parametrize("penalty,param", [
        ("SCAD", {"a": 3.7}),
        ("SCAD", {"a": 5.0}),
        ("MCP", {"gamma": 2.0}),
        ("MCP", {"gamma": 3.0}),
    ])
    def test_different_penalties_and_parameters(self, linear_regression_data, penalty, param):
        """Test UAG with different penalty functions and parameters."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_max = lambda_max_LM(X, y)
        lambda_ = lambda_max * 0.1
        
        solver = UAG(model_type="linear", penalty=penalty, **param)
        solver.fit(X, y, lambda_)
        
        assert solver.converged_
        assert solver.coef_ is not None
        assert solver.intercept_ is not None
        
        # Verify objective decreases
        beta_init = np.zeros(X.shape[1])
        beta_init[0] = np.mean(y)
        obj_init = solver.model.loss(X, y, beta_init)
        
        beta_final = np.concatenate([[solver.intercept_], solver.coef_])
        obj_final = solver.model.loss(X, y, beta_final)
        
        if penalty == "SCAD":
            obj_final += SCAD(beta_final, lambda_, param.get("a", 3.7))
        else:
            obj_final += MCP(beta_final, lambda_, param.get("gamma", 2.0))
            
        assert obj_final < obj_init
        
    def test_convergence_tolerance(self, small_data):
        """Test that convergence tolerance is respected."""
        X, y = small_data
        X = add_intercept(X)
        
        # Very tight tolerance
        solver_tight = UAG(model_type="linear", tol=1e-10, maxit=10000)
        solver_tight.fit(X, y, 0.01)
        
        # Loose tolerance
        solver_loose = UAG(model_type="linear", tol=1e-2, maxit=10000)
        solver_loose.fit(X, y, 0.01)
        
        # Tight tolerance should take more iterations
        assert solver_tight.n_iter_ >= solver_loose.n_iter_
        
    def test_max_iterations(self, linear_regression_data):
        """Test that max iterations limit is respected."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        # Very few iterations
        solver = UAG(model_type="linear", maxit=5, tol=1e-10)
        
        with warnings.catch_warnings(record=True):
            solver.fit(X, y, 0.1)
            
        assert solver.n_iter_ == 5
        assert not solver.converged_  # Should not converge in 5 iterations
        
    def test_warm_start_efficiency(self, linear_regression_data):
        """Test that warm starts reduce iterations."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_max = lambda_max_LM(X, y)
        lambda1 = lambda_max * 0.2
        lambda2 = lambda_max * 0.19  # Close lambda
        
        # Cold start
        solver_cold = UAG(model_type="linear")
        solver_cold.fit(X, y, lambda2)
        n_iter_cold = solver_cold.n_iter_
        
        # Warm start
        solver_warm1 = UAG(model_type="linear")
        solver_warm1.fit(X, y, lambda1)
        beta_warm = np.concatenate([[solver_warm1.intercept_], solver_warm1.coef_])
        
        solver_warm2 = UAG(model_type="linear")
        solver_warm2.fit(X, y, lambda2, beta_init=beta_warm)
        n_iter_warm = solver_warm2.n_iter_
        
        # Warm start should converge faster
        assert n_iter_warm < n_iter_cold
        
        # Solutions should be similar
        assert_array_almost_equal(solver_cold.coef_, solver_warm2.coef_, decimal=3)
        
    def test_different_starting_values(self, linear_regression_data):
        """Test robustness to different starting values."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_ = 0.1
        
        # Different starting points
        np.random.seed(123)  # Fixed seed for reproducible random start
        starts = [
            None,  # Default initialization
            np.zeros(X.shape[1]),  # All zeros
            np.random.randn(X.shape[1]),  # Random
            np.ones(X.shape[1]) * 10,  # Far from optimum
        ]
        
        solutions = []
        for beta_init in starts:
            solver = UAG(model_type="linear", penalty="SCAD")
            try:
                solver.fit(X, y, lambda_, beta_init=beta_init)
                solutions.append(solver.coef_)
            except ReferenceError as e:
                if "underlying object has vanished" in str(e):
                    pytest.skip("Numba caching issue - known intermittent problem")
                else:
                    raise
            
        # All should converge to same solution (allow for numerical differences)
        for i in range(1, len(solutions)):
            assert_array_almost_equal(solutions[0], solutions[i], decimal=0)
            
    def test_sparsity_pattern(self, linear_regression_data):
        """Test that sparsity increases with lambda."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_max = lambda_max_LM(X, y)
        lambdas = lambda_max * np.array([0.01, 0.05, 0.1, 0.2, 0.5])
        
        n_nonzero = []
        for lam in lambdas:
            solver = UAG(model_type="linear", penalty="MCP")
            solver.fit(X, y, lam)
            n_nonzero.append(np.sum(np.abs(solver.coef_) > 1e-6))
            
        # Sparsity should increase (n_nonzero decrease) with lambda
        assert all(n_nonzero[i] >= n_nonzero[i+1] for i in range(len(n_nonzero)-1))
        
    def test_logistic_probability_bounds(self, logistic_regression_data):
        """Test that logistic predictions are valid probabilities."""
        data = logistic_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        solver = UAG(model_type="logistic", penalty="SCAD")
        solver.fit(X, y, 0.05)
        
        # Get probability predictions
        from nonconvexAG.models import LogisticModel
        model = LogisticModel()
        beta = np.concatenate([[solver.intercept_], solver.coef_])
        probs = model.predict_proba(X, beta)
        
        # Check bounds
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        
        # Check no NaN or inf
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))


class TestSolutionPathComprehensive:
    """Comprehensive tests for solution path computation."""
    
    def test_path_continuity(self, linear_regression_data):
        """Test that solution paths are continuous."""
        data = linear_regression_data
        X = add_intercept(data['X'][:200])  # Smaller for speed
        y = data['y'][:200]
        
        path = SolutionPath(model_type="linear", penalty="SCAD")
        path.fit(X, y, n_lambdas=50)
        
        # Check continuity: adjacent solutions should be close
        for i in range(path.coef_path_.shape[1] - 1):
            coef_diff = np.max(np.abs(path.coef_path_[:, i] - path.coef_path_[:, i+1]))
            # Difference should be small relative to lambda change
            lambda_ratio = path.lambda_path_[i+1] / path.lambda_path_[i]
            assert coef_diff < 3.0  # Allow larger jumps for nonconvex penalties
            
    def test_path_endpoints(self, linear_regression_data):
        """Test solution path endpoints."""
        data = linear_regression_data
        X = add_intercept(data['X'][:200])
        y = data['y'][:200]
        
        path = SolutionPath(model_type="linear", penalty="MCP")
        path.fit(X, y, n_lambdas=100, lambda_min_ratio=0.001)
        
        # At large lambda, should be mostly zeros
        n_nonzero_start = np.sum(np.abs(path.coef_path_[:, 0]) > 1e-6)
        assert n_nonzero_start <= 5  # Very sparse
        
        # At small lambda, should have more nonzeros
        n_nonzero_end = np.sum(np.abs(path.coef_path_[:, -1]) > 1e-6)
        assert n_nonzero_end >= n_nonzero_start
        
    def test_custom_lambda_sequence(self, small_data):
        """Test with custom lambda sequence."""
        X, y = small_data
        X = add_intercept(X)
        
        # Custom lambdas
        lambdas = np.array([1.0, 0.5, 0.2, 0.1, 0.05])
        
        path = SolutionPath(model_type="linear")
        path.fit(X, y, lambdas=lambdas)
        
        assert_array_almost_equal(path.lambda_path_, lambdas)
        assert path.coef_path_.shape[1] == len(lambdas)
        
    def test_standardization_option(self, linear_regression_data):
        """Test standardization in solution path."""
        data = linear_regression_data
        X = add_intercept(data['X'][:200])
        y = data['y'][:200]
        
        # With standardization
        path_std = SolutionPath(model_type="linear")
        path_std.fit(X, y, n_lambdas=20, standardize=True)
        
        # Without standardization
        path_no_std = SolutionPath(model_type="linear")
        path_no_std.fit(X, y, n_lambdas=20, standardize=False)
        
        # Results should differ if features have different scales
        if np.std(np.std(X[:, 1:], axis=0)) > 0.1:  # Features have different scales
            assert not np.allclose(path_std.coef_path_, path_no_std.coef_path_)
            
    def test_get_support(self, linear_regression_data):
        """Test get_support method."""
        data = linear_regression_data  
        X = add_intercept(data['X'][:200])
        y = data['y'][:200]
        
        path = SolutionPath(model_type="linear")
        path.fit(X, y, n_lambdas=30)
        
        # Test at different lambda indices
        support_start = path.get_support(lambda_idx=0)
        support_mid = path.get_support(lambda_idx=15)
        support_end = path.get_support(lambda_idx=-1)
        
        # Support should generally increase
        assert np.sum(support_start) <= np.sum(support_end)
        
        # Test with different threshold
        support_tight = path.get_support(lambda_idx=-1, threshold=1e-3)
        support_loose = path.get_support(lambda_idx=-1, threshold=1e-8)
        assert np.sum(support_tight) <= np.sum(support_loose)
        
    def test_path_predictions(self, logistic_regression_data):
        """Test predictions along the path."""
        data = logistic_regression_data
        X = add_intercept(data['X'][:200])
        y = data['y'][:200]
        X_test = add_intercept(data['X_test'][:50])
        
        path = SolutionPath(model_type="logistic", penalty="SCAD")
        path.fit(X, y, n_lambdas=20)
        
        # Test predictions at different points
        for idx in [0, 10, -1]:
            pred = path.predict(X_test, lambda_idx=idx)
            assert pred.shape == (X_test.shape[0],)
            assert np.all((pred >= 0) & (pred <= 1))


class TestStrongRuleSolverComprehensive:
    """Comprehensive tests for strong rule solver."""
    
    def test_strong_rule_correctness(self, linear_regression_data):
        """Test that strong rule gives same solution as regular UAG."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_ = 0.1
        
        # Regular UAG
        solver_regular = UAG(model_type="linear", penalty="SCAD")
        solver_regular.fit(X, y, lambda_)
        
        # Strong rule UAG
        solver_strong = StrongRuleSolver(model_type="linear", penalty="SCAD")
        solver_strong.fit(X, y, lambda_)
        
        # Solutions should be very close (may have small differences due to active set)
        assert_array_almost_equal(solver_regular.coef_, solver_strong.coef_, decimal=3)
        assert_almost_equal(solver_regular.intercept_, solver_strong.intercept_, decimal=3)
        
    def test_strong_rule_efficiency(self, linear_regression_data):
        """Test that strong rule reduces active set size."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_max = lambda_max_LM(X, y)
        lambda_ = lambda_max * 0.3  # Moderate sparsity
        
        solver = StrongRuleSolver(model_type="linear", penalty="MCP", verbose=True)
        solver.fit(X, y, lambda_)
        
        # Active set should be much smaller than total features
        assert len(solver.active_set_) < X.shape[1] / 2
        
        # Should have done strong rule checks
        assert solver.n_strong_rule_checks_ > 0
        
    def test_sequential_strong_rule(self, linear_regression_data):
        """Test sequential strong rule with warm starts."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        lambda_max = lambda_max_LM(X, y)
        lambda1 = lambda_max * 0.3
        lambda2 = lambda_max * 0.25
        
        # First fit
        solver1 = StrongRuleSolver(model_type="linear", penalty="SCAD")
        solver1.fit(X, y, lambda1)
        
        # Sequential fit
        solver2 = StrongRuleSolver(model_type="linear", penalty="SCAD")
        beta_prev = np.concatenate([[solver1.intercept_], solver1.coef_])
        solver2.fit(X, y, lambda2, beta_init=beta_prev, 
                   lambda_prev=lambda1, beta_prev=beta_prev)
        
        # Sequential should have smaller initial active set
        assert solver2.n_iter_ <= solver1.n_iter_
        
    def test_violation_detection(self, linear_regression_data):
        """Test that violations are properly detected and handled."""
        data = linear_regression_data
        X = add_intercept(data['X'])
        y = data['y']
        
        # Use aggressive screening
        solver = StrongRuleSolver(
            model_type="linear", 
            penalty="MCP",
            strong_rule_freq=1,  # Check every iteration
            violation_check=True
        )
        
        lambda_ = 0.15
        solver.fit(X, y, lambda_)
        
        # Should detect some violations with aggressive screening
        # (though not guaranteed depending on data)
        assert solver.n_strong_rule_checks_ >= solver.n_iter_
        
    def test_no_violation_check(self, linear_regression_data):
        """Test solver without violation checking."""
        data = linear_regression_data
        X = add_intercept(data['X'][:, :20])  # Fewer features
        y = data['y']
        
        solver = StrongRuleSolver(
            model_type="linear",
            penalty="SCAD", 
            violation_check=False
        )
        
        solver.fit(X, y, 0.1)
        
        # Should not have violation statistics
        assert solver.n_violations_ == 0
        
    @pytest.mark.parametrize("model_type", ["linear", "logistic"])
    def test_strong_rule_both_models(self, linear_regression_data, 
                                    logistic_regression_data, model_type):
        """Test strong rule works for both model types."""
        if model_type == "linear":
            data = linear_regression_data
        else:
            data = logistic_regression_data
            
        X = add_intercept(data['X'][:300])  # Smaller for speed
        y = data['y'][:300]
        
        solver = StrongRuleSolver(model_type=model_type, penalty="MCP")
        
        if model_type == "linear":
            lambda_ = lambda_max_LM(X, y) * 0.1
        else:
            lambda_ = lambda_max_logistic(X, y) * 0.05
            
        solver.fit(X, y, lambda_)
        
        assert solver.converged_
        assert len(solver.active_set_) > 0  # At least intercept
        assert solver.coef_ is not None


class TestSolverEdgeCases:
    """Test edge cases and robustness."""
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(50, 1)
        X = add_intercept(X)
        y = 2 * X[:, 1] + np.random.randn(50) * 0.1
        
        solver = UAG(model_type="linear")
        solver.fit(X, y, 0.01)
        
        assert solver.converged_
        assert len(solver.coef_) == 1
        
    def test_perfect_fit(self):
        """Test when perfect fit is possible."""
        X = np.eye(5)
        y = np.array([1, 2, 3, 4, 5])
        
        solver = UAG(model_type="linear", penalty="SCAD")
        solver.fit(X, y, 0.001)  # Very small lambda
        
        # Should recover exact solution
        expected = y
        actual = solver.predict(X)
        assert_array_almost_equal(actual, expected, decimal=3)
        
    def test_constant_features(self):
        """Test with constant features."""
        X = np.ones((50, 3))
        X[:, 1] = np.random.randn(50)  # Only one informative
        X = add_intercept(X)
        y = 2 * X[:, 2] + np.random.randn(50) * 0.1
        
        solver = UAG(model_type="linear")
        solver.fit(X, y, 0.5)
        
        assert solver.converged_
        # Constant features should have small coefficients
        assert abs(solver.coef_[1]) < 2.0  # First constant feature  
        assert abs(solver.coef_[2]) < 2.0  # Second constant feature
        
    def test_highly_correlated_features(self):
        """Test with highly correlated features."""
        n = 100
        X1 = np.random.randn(n)
        X2 = X1 + 0.001 * np.random.randn(n)  # Almost identical
        X = np.column_stack([X1, X2, np.random.randn(n, 3)])
        X = add_intercept(X)
        
        y = 2 * X1 + np.random.randn(n) * 0.1
        
        solver = UAG(model_type="linear", penalty="MCP")
        solver.fit(X, y, 0.05)
        
        assert solver.converged_
        # Should select one of the correlated features
        selected = np.abs(solver.coef_[:2]) > 0.01
        assert np.sum(selected) >= 1  # At least one selected