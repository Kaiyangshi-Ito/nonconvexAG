"""Performance and stress tests for nonconvexAG."""

import pytest
import numpy as np
import time
from scipy.linalg import toeplitz
import psutil
import os

from nonconvexAG import UAG, SolutionPath, StrongRuleSolver
from nonconvexAG.utils import add_intercept, lambda_max_LM, lambda_max_logistic


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks for solvers."""
    
    @pytest.mark.parametrize("n_samples,n_features", [
        (100, 50),
        (500, 100),
        (1000, 200),
        (2000, 500),
    ])
    def test_scaling_with_size(self, n_samples, n_features):
        """Test performance scaling with problem size."""
        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        true_beta = np.zeros(n_features)
        true_beta[:10] = np.random.randn(10) * 2
        y = X @ true_beta + np.random.randn(n_samples) * 0.5
        
        X = add_intercept(X)
        lambda_ = 0.1
        
        # Time the solver
        solver = UAG(model_type="linear", penalty="SCAD")
        
        start_time = time.time()
        solver.fit(X, y, lambda_)
        runtime = time.time() - start_time
        
        # Check performance
        assert solver.converged_
        assert runtime < 60  # Should complete within 60 seconds
        
        # Report performance
        print(f"\nSize: {n_samples}x{n_features}")
        print(f"Runtime: {runtime:.3f}s")
        print(f"Iterations: {solver.n_iter_}")
        print(f"Time per iteration: {runtime/solver.n_iter_*1000:.1f}ms")
        
    def test_strong_rule_speedup(self):
        """Benchmark strong rule speedup on large problems."""
        # Large sparse problem
        n_samples, n_features = 1000, 1000
        n_informative = 20
        
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        true_beta = np.zeros(n_features)
        idx = np.random.choice(n_features, n_informative, replace=False)
        true_beta[idx] = np.random.randn(n_informative) * 2
        
        y = X @ true_beta + np.random.randn(n_samples) * 0.5
        X = add_intercept(X)
        
        lambda_max = lambda_max_LM(X, y)
        lambda_ = lambda_max * 0.2
        
        # Run multiple times to get stable timing
        times_regular = []
        times_strong = []
        
        for _ in range(3):
            # Regular UAG
            solver_regular = UAG(model_type="linear", penalty="MCP")
            start_regular = time.time()
            solver_regular.fit(X, y, lambda_)
            times_regular.append(time.time() - start_regular)
            
            # Strong rule UAG
            solver_strong = StrongRuleSolver(model_type="linear", penalty="MCP")
            start_strong = time.time()
            solver_strong.fit(X, y, lambda_)
            times_strong.append(time.time() - start_strong)
        
        # Take minimum times for stable comparison
        time_regular = min(times_regular)
        time_strong = min(times_strong)
        
        # Check speedup
        speedup = time_regular / time_strong
        print(f"\nStrong Rule Performance:")
        print(f"Regular UAG: {time_regular:.3f}s")
        print(f"Strong Rule: {time_strong:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Active set size: {len(solver_strong.active_set_)}/{n_features+1}")
        
        # Strong rule should reduce active set even if not always faster
        assert len(solver_strong.active_set_) < n_features * 0.8  # Significant reduction
        
    def test_solution_path_performance(self):
        """Test solution path computation performance."""
        # Moderate size problem
        n_samples, n_features = 500, 200
        
        np.random.seed(42)
        cov = toeplitz(0.5 ** np.arange(n_features))
        X = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        true_beta = np.zeros(n_features)
        true_beta[:15] = np.random.randn(15) * 2
        y = X @ true_beta + np.random.randn(n_samples) * 0.5
        
        X = add_intercept(X)
        
        # Time solution path
        path_solver = SolutionPath(model_type="linear", penalty="SCAD")
        
        start_time = time.time()
        path_solver.fit(X, y, n_lambdas=100)
        runtime = time.time() - start_time
        
        # Check performance
        assert runtime < 120  # Should complete within 2 minutes
        
        # Calculate efficiency
        total_iterations = sum(path_solver.n_iter_path_)
        avg_iterations = np.mean(path_solver.n_iter_path_)
        
        print(f"\nSolution Path Performance:")
        print(f"Total runtime: {runtime:.3f}s")
        print(f"Lambdas: {len(path_solver.lambda_path_)}")
        print(f"Time per lambda: {runtime/len(path_solver.lambda_path_):.3f}s")
        print(f"Total iterations: {total_iterations}")
        print(f"Average iterations per lambda: {avg_iterations:.1f}")
        
    @pytest.mark.memory_intensive
    def test_memory_efficiency(self):
        """Test memory usage remains reasonable."""
        # Large problem
        n_samples, n_features = 2000, 1000
        
        initial_memory = get_memory_usage()
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)  # Use float32
        y = np.random.randn(n_samples).astype(np.float32)
        X = add_intercept(X)
        
        data_memory = get_memory_usage()
        data_size = data_memory - initial_memory
        
        # Fit model
        solver = UAG(model_type="linear", penalty="MCP")
        solver.fit(X, y, 0.1)
        
        peak_memory = get_memory_usage()
        solver_overhead = peak_memory - data_memory
        
        print(f"\nMemory Usage:")
        print(f"Data size: {data_size:.1f} MB")
        print(f"Solver overhead: {solver_overhead:.1f} MB")
        print(f"Total: {peak_memory - initial_memory:.1f} MB")
        
        # Solver shouldn't use more than 2x the data size
        assert solver_overhead < 2 * data_size


@pytest.mark.slow
class TestStressTests:
    """Stress tests for edge cases and robustness."""
    
    def test_high_dimensional(self):
        """Test with p >> n (more features than samples)."""
        n_samples = 100
        n_features = 1000
        
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Very sparse true model
        true_beta = np.zeros(n_features)
        true_beta[:5] = np.random.randn(5) * 3
        y = X @ true_beta + np.random.randn(n_samples) * 0.1
        
        X = add_intercept(X)
        lambda_max = lambda_max_LM(X, y)
        
        # Should handle high-dimensional case
        solver = StrongRuleSolver(model_type="linear", penalty="SCAD")
        solver.fit(X, y, lambda_max * 0.3)
        
        assert solver.converged_
        # Should find sparse solution
        n_selected = np.sum(np.abs(solver.coef_) > 1e-6)
        assert n_selected < 50  # Much less than p
        
    def test_highly_correlated_design(self):
        """Test with highly correlated features."""
        n_samples = 500
        n_features = 100
        
        np.random.seed(42)
        # Create blocks of correlated features
        n_blocks = 10
        block_size = n_features // n_blocks
        
        X = []
        for i in range(n_blocks):
            # Each block is highly correlated
            base = np.random.randn(n_samples, 1)
            block = base + 0.1 * np.random.randn(n_samples, block_size)
            X.append(block)
            
        X = np.hstack(X)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # True model uses one feature from each block
        true_beta = np.zeros(n_features)
        for i in range(n_blocks):
            true_beta[i * block_size] = np.random.randn() * 2
            
        y = X @ true_beta + np.random.randn(n_samples) * 0.5
        X = add_intercept(X)
        
        # Should handle correlation
        solver = UAG(model_type="linear", penalty="MCP", gamma=3.0)
        solver.fit(X, y, 3.0)
        
        assert solver.converged_
        # Should select roughly one per block
        n_selected = np.sum(np.abs(solver.coef_) > 1e-6)
        assert 5 <= n_selected <= 35
        
    def test_nearly_singular_design(self):
        """Test with nearly singular design matrix."""
        n_samples = 100
        n_features = 50
        
        np.random.seed(42)
        # Create nearly singular matrix
        X = np.random.randn(n_samples, n_features)
        # Make some columns nearly identical
        for i in range(5):
            X[:, i+10] = X[:, i] + 1e-6 * np.random.randn(n_samples)
            
        y = np.random.randn(n_samples)
        X = add_intercept(X)
        
        # Should handle near-singularity gracefully
        solver = UAG(model_type="linear", penalty="SCAD", tol=1e-4)
        solver.fit(X, y, 0.1)
        
        assert solver.converged_
        assert not np.any(np.isnan(solver.coef_))
        assert not np.any(np.isinf(solver.coef_))
        
    def test_extreme_lambda_values(self):
        """Test with very small and very large lambda values."""
        n_samples = 200
        n_features = 50
        
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        X = add_intercept(X)
        
        lambda_max = lambda_max_LM(X, y)
        
        # Very large lambda (should zero everything)
        solver_large = UAG(model_type="linear", penalty="MCP")
        solver_large.fit(X, y, lambda_max * 10)
        
        assert np.allclose(solver_large.coef_, 0, atol=1e-10)
        
        # Very small lambda (should be like unpenalized)
        solver_small = UAG(model_type="linear", penalty="MCP")
        solver_small.fit(X, y, lambda_max * 1e-6)
        
        # Compare with least squares
        beta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
        beta_small = np.concatenate([[solver_small.intercept_], solver_small.coef_])
        
        # Should be close to least squares
        rel_error = np.linalg.norm(beta_small - beta_ls) / np.linalg.norm(beta_ls)
        assert rel_error < 0.1
        
    def test_extreme_response_values(self):
        """Test with extreme values in response."""
        n_samples = 200
        n_features = 30
        
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        X = add_intercept(X)
        
        # Response with outliers
        y = np.random.randn(n_samples)
        y[0] = 1000  # Extreme outlier
        y[1] = -1000
        
        # Should handle outliers
        solver = UAG(model_type="linear", penalty="SCAD")
        solver.fit(X, y, 0.1)
        
        assert solver.converged_
        assert not np.any(np.isnan(solver.coef_))
        
    @pytest.mark.slow
    def test_many_lambda_path(self):
        """Test solution path with many lambda values."""
        n_samples = 300
        n_features = 100
        
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        X = add_intercept(X)
        
        # Many lambda values
        path_solver = SolutionPath(model_type="linear", penalty="MCP")
        path_solver.fit(X, y, n_lambdas=200)
        
        # Should complete and maintain continuity
        assert len(path_solver.lambda_path_) == 200
        assert len(path_solver.n_iter_path_) == 200
        
        # Check path smoothness
        for i in range(199):
            coef_change = np.max(np.abs(
                path_solver.coef_path_[:, i] - path_solver.coef_path_[:, i+1]
            ))
            # Changes should be gradual
            assert coef_change < 1.0