"""Basic usage examples for nonconvexAG package."""

import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# Import from the refactored package
from nonconvexAG import UAG, SolutionPath, StrongRuleSolver
from nonconvexAG.utils import add_intercept, lambda_max_LM, lambda_max_logistic


def generate_synthetic_data(n_samples=500, n_features=100, n_informative=10, 
                          correlation=0.5, noise_level=0.5, seed=42):
    """Generate synthetic regression data with correlated features."""
    np.random.seed(seed)
    
    # True sparse coefficients
    true_beta = np.zeros(n_features)
    informative_indices = np.random.choice(n_features, n_informative, replace=False)
    true_beta[informative_indices] = np.random.randn(n_informative) * 3
    
    # Correlation structure
    cov = toeplitz(correlation ** np.arange(n_features))
    X = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)
    
    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Generate response
    signal = X @ true_beta
    noise = np.random.normal(0, noise_level * np.std(signal), n_samples)
    y = signal + noise
    
    return X, y, true_beta, informative_indices


def example_basic_linear_regression():
    """Example 1: Basic linear regression with SCAD penalty."""
    print("=" * 60)
    print("Example 1: Linear Regression with SCAD Penalty")
    print("=" * 60)
    
    # Generate data
    X, y, true_beta, true_support = generate_synthetic_data()
    X_with_intercept = add_intercept(X)
    
    # Calculate lambda_max
    lambda_max = lambda_max_LM(X_with_intercept, y)
    lambda_val = lambda_max * 0.1
    
    # Fit model
    print(f"\nFitting UAG with λ = {lambda_val:.4f}")
    solver = UAG(model_type="linear", penalty="SCAD", verbose=True)
    solver.fit(X_with_intercept, y, lambda_val)
    
    # Results
    print(f"\nConverged: {solver.converged_}")
    print(f"Number of iterations: {solver.n_iter_}")
    print(f"Runtime: {solver.runtime_:.3f} seconds")
    
    # Check sparsity
    selected = np.where(np.abs(solver.coef_) > 1e-6)[0]
    print(f"\nTrue support size: {len(true_support)}")
    print(f"Selected features: {len(selected)}")
    print(f"Correct selections: {len(set(selected) & set(true_support))}")
    
    # Score
    train_score = solver.score(X_with_intercept, y)
    print(f"\nR² score: {train_score:.4f}")


def example_logistic_regression():
    """Example 2: Logistic regression with MCP penalty."""
    print("\n" + "=" * 60)
    print("Example 2: Logistic Regression with MCP Penalty")
    print("=" * 60)
    
    # Generate binary data
    X, y_continuous, true_beta, true_support = generate_synthetic_data(
        n_samples=800, n_features=50, n_informative=8
    )
    
    # Convert to binary
    probs = 1 / (1 + np.exp(-y_continuous))
    y = np.random.binomial(1, probs)
    
    X_with_intercept = add_intercept(X)
    
    # Fit model
    lambda_max = lambda_max_logistic(X_with_intercept, y)
    lambda_val = lambda_max * 0.05
    
    print(f"\nFitting UAG with λ = {lambda_val:.4f}")
    solver = UAG(model_type="logistic", penalty="MCP", gamma=3.0)
    solver.fit(X_with_intercept, y, lambda_val)
    
    # Results
    print(f"\nConverged: {solver.converged_}")
    print(f"Number of iterations: {solver.n_iter_}")
    
    # Check accuracy
    accuracy = solver.score(X_with_intercept, y)
    print(f"Training accuracy: {accuracy:.4f}")


def example_solution_path():
    """Example 3: Computing and plotting solution paths."""
    print("\n" + "=" * 60)
    print("Example 3: Solution Path Computation")
    print("=" * 60)
    
    # Generate data with fewer features for clearer visualization
    X, y, true_beta, true_support = generate_synthetic_data(
        n_samples=200, n_features=20, n_informative=5
    )
    X_with_intercept = add_intercept(X)
    
    # Compute solution path
    print("\nComputing solution path...")
    path_solver = SolutionPath(model_type="linear", penalty="SCAD", 
                              verbose=True, tol=1e-5)
    path_solver.fit(X_with_intercept, y, n_lambdas=50)
    
    print(f"\nPath computed for {len(path_solver.lambda_path_)} λ values")
    print(f"λ range: [{path_solver.lambda_path_[-1]:.4f}, {path_solver.lambda_path_[0]:.4f}]")
    
    # Plot
    feature_names = [f"X{i+1}" if i in true_support else None 
                    for i in range(X.shape[1])]
    fig = path_solver.plot_path(feature_names=feature_names)
    plt.show()


def example_strong_rule():
    """Example 4: Using strong rule for efficiency."""
    print("\n" + "=" * 60)
    print("Example 4: Strong Rule for Variable Screening")
    print("=" * 60)
    
    # Generate high-dimensional data
    X, y, true_beta, true_support = generate_synthetic_data(
        n_samples=300, n_features=500, n_informative=15, correlation=0.7
    )
    X_with_intercept = add_intercept(X)
    
    lambda_max = lambda_max_LM(X_with_intercept, y)
    lambda_val = lambda_max * 0.2
    
    # Compare with and without strong rule
    print("\nFitting WITHOUT strong rule...")
    solver_regular = UAG(model_type="linear", penalty="SCAD")
    solver_regular.fit(X_with_intercept, y, lambda_val)
    time_regular = solver_regular.runtime_
    
    print("\nFitting WITH strong rule...")
    solver_strong = StrongRuleSolver(model_type="linear", penalty="SCAD", 
                                    verbose=True)
    solver_strong.fit(X_with_intercept, y, lambda_val)
    time_strong = solver_strong.runtime_
    
    # Compare results
    print(f"\nRegular UAG time: {time_regular:.3f}s")
    print(f"Strong rule time: {time_strong:.3f}s")
    print(f"Speedup: {time_regular/time_strong:.2f}x")
    
    print(f"\nActive set size: {len(solver_strong.active_set_)}/{X.shape[1]+1}")
    print(f"Strong rule checks: {solver_strong.n_strong_rule_checks_}")
    print(f"Violations found: {solver_strong.n_violations_}")
    
    # Verify same solution
    coef_diff = np.max(np.abs(solver_regular.coef_ - solver_strong.coef_))
    print(f"\nMax coefficient difference: {coef_diff:.2e}")


def example_warm_starts():
    """Example 5: Using warm starts for sequential problems."""
    print("\n" + "=" * 60)
    print("Example 5: Warm Starts for Sequential Problems")
    print("=" * 60)
    
    X, y, _, _ = generate_synthetic_data(n_samples=400, n_features=50)
    X_with_intercept = add_intercept(X)
    
    lambda_max = lambda_max_LM(X_with_intercept, y)
    lambdas = lambda_max * np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    
    solver = UAG(model_type="linear", penalty="MCP")
    beta_current = None
    
    print("\nSequential fitting with warm starts:")
    for i, lam in enumerate(lambdas):
        solver.fit(X_with_intercept, y, lam, beta_init=beta_current)
        print(f"λ = {lam:.4f}: {solver.n_iter_:3d} iterations")
        
        # Prepare warm start for next lambda
        beta_current = np.concatenate([[solver.intercept_], solver.coef_])


if __name__ == "__main__":
    # Run all examples
    example_basic_linear_regression()
    example_logistic_regression()
    example_solution_path()
    example_strong_rule()
    example_warm_starts()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)