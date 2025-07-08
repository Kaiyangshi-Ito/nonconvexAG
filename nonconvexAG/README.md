# nonconvexAG

[![MIT License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/nonconvexAG.svg)](https://badge.fury.io/py/nonconvexAG)

**Kai Yang** <kai.yang2@mail.mcgill.ca>  
ORCID: [0000-0001-5505-6886](https://orcid.org/0000-0001-5505-6886)  
GPG: [B080 1753 189F BAFE 10B5 3E8A 0F6C F129 F618 CEEF](https://keys.openpgp.org/vks/v1/by-fingerprint/B0801753189FBAFE10B53E8A0F6CF129F618CEEF)

Accelerated gradient methods with strong rules for nonconvex sparse learning (SCAD/MCP penalties). [Paper](https://arxiv.org/abs/2009.10629).

## Install

```bash
pip install nonconvexAG
```

## How to Use This Package

### Step 1: Import Required Functions

```python
import numpy as np
from nonconvexAG import UAG, SolutionPath, StrongRuleSolver
from nonconvexAG.utils import add_intercept

# For legacy compatibility
from nonconvexAG import UAG_LM_SCAD_MCP, UAG_logistic_SCAD_MCP
```

### Step 2: Prepare Your Data

**CRITICAL**: Always add an intercept column to your design matrix X:

```python
# Your original data
X = np.random.randn(100, 20)  # 100 samples, 20 features
y = np.random.randn(100)       # response

# Add intercept column (REQUIRED)
X_with_intercept = add_intercept(X)  # Now shape is (100, 21)
```

### Step 3: Choose Your Model and Penalty

```python
# For regression problems
solver = UAG(model_type="linear", penalty="SCAD")

# For classification problems  
solver = UAG(model_type="logistic", penalty="MCP")

# Parameters:
# - penalty: "SCAD" or "MCP"
# - a: SCAD parameter (default 3.7)
# - gamma: MCP parameter (default 2.0)
```

### Step 4: Fit the Model

```python
# Single lambda value
solver.fit(X_with_intercept, y, lambda_=0.1)

# Access results
print(f"Intercept: {solver.intercept_}")
print(f"Coefficients: {solver.coef_}")
print(f"Iterations: {solver.n_iter_}")
```

### Step 5: Make Predictions

```python
# Linear regression
y_pred = solver.predict(X_with_intercept)

# Logistic regression
y_pred = solver.predict(X_with_intercept)  # class predictions (0 or 1)
```

## Complete Examples

### Example 1: Sparse Linear Regression

```python
import numpy as np
from nonconvexAG import UAG
from nonconvexAG.utils import add_intercept

# Generate sparse data
np.random.seed(42)
n, p = 200, 50
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = np.array([3, -2, 0, 1.5, -1])  # 5 true features
y = X @ true_beta + 0.5 * np.random.randn(n)

# Add intercept
X_with_intercept = add_intercept(X)

# Fit SCAD model
solver = UAG(model_type="linear", penalty="SCAD", a=3.7)
solver.fit(X_with_intercept, y, lambda_=0.1)

# Check sparsity
n_nonzero = np.sum(np.abs(solver.coef_) > 1e-6)
print(f"Non-zero coefficients: {n_nonzero}/{p}")

# Evaluate
from sklearn.metrics import mean_squared_error
y_pred = solver.predict(X_with_intercept)
mse = mean_squared_error(y, y_pred)
print(f"MSE: {mse:.4f}")
```

### Example 2: Cross-Validation for Lambda Selection

```python
from nonconvexAG import SolutionPath
from sklearn.model_selection import KFold

# Generate solution path
path = SolutionPath(model_type="linear", penalty="SCAD")
path.fit(X_with_intercept, y, n_lambdas=30)

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_errors = np.zeros(len(path.lambda_path_))

for train_idx, val_idx in kf.split(X):
    X_train = X_with_intercept[train_idx]
    y_train = y[train_idx]
    X_val = X_with_intercept[val_idx]
    y_val = y[val_idx]
    
    # Fit path on training data
    path_cv = SolutionPath(model_type="linear", penalty="SCAD")
    path_cv.fit(X_train, y_train, lambdas=path.lambda_path_)
    
    # Evaluate on validation data
    for i, lam in enumerate(path.lambda_path_):
        y_pred = X_val @ np.concatenate([[path_cv.intercept_path_[i]], 
                                         path_cv.coef_path_[:, i]])
        cv_errors[i] += np.mean((y_val - y_pred)**2)

cv_errors /= 5
best_lambda_idx = np.argmin(cv_errors)
best_lambda = path.lambda_path_[best_lambda_idx]

print(f"Best lambda by CV: {best_lambda:.4f}")

# Refit with best lambda
solver_best = UAG(model_type="linear", penalty="SCAD")
solver_best.fit(X_with_intercept, y, lambda_=best_lambda)
```

### Example 3: High-Dimensional Data with Strong Rules

```python
from nonconvexAG import StrongRuleSolver

# Very high dimensional
n, p = 200, 5000
X = np.random.randn(n, p)
true_beta = np.zeros(p)
# 20 true features randomly placed
true_indices = np.random.choice(p, 20, replace=False)
true_beta[true_indices] = np.random.randn(20) * 2
y = X @ true_beta + 0.1 * np.random.randn(n)

X_with_intercept = add_intercept(X)

# Standard solver would be slow
# Use strong rule for efficiency
solver = StrongRuleSolver(model_type="linear", penalty="MCP", gamma=2.0)
solver.fit(X_with_intercept, y, lambda_=0.5)

print(f"Active set size: {len(solver.active_set_)}")
print(f"Speedup factor: ~{p/len(solver.active_set_):.1f}x")

# Check feature selection accuracy
selected = np.where(np.abs(solver.coef_) > 1e-6)[0]
true_positives = len(set(selected) & set(true_indices))
print(f"True features found: {true_positives}/{len(true_indices)}")
```

## Usage

### Installation from Source

```bash
# Clone repository
git clone https://github.com/Kaiyangshi-Ito/nonconvexAG.git
cd nonconvexAG

# Install in development mode
pip install -e .

# Or build and install
python -m build
pip install dist/nonconvexAG-*.whl
```

### Basic Usage

```python
import numpy as np
from nonconvexAG import UAG
from nonconvexAG.utils import add_intercept

# Generate data
n, p = 100, 20
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [2, -1.5, 1, -0.5, 3]
y = X @ true_beta + 0.1 * np.random.randn(n)

# Add intercept column (IMPORTANT: always do this)
X_with_intercept = add_intercept(X)

# Fit
solver = UAG(model_type="linear", penalty="SCAD")
solver.fit(X_with_intercept, y, lambda_=0.1)

print(f"Intercept: {solver.intercept_}")
print(f"Coefficients: {solver.coef_}")
```

### Logistic Regression

```python
# Binary classification data
from nonconvexAG.utils import add_intercept

# Generate binary data
np.random.seed(42)
n, p = 100, 15
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:3] = [1.5, -2, 1]
logits = X @ true_beta
y = (np.random.random(n) < 1 / (1 + np.exp(-logits))).astype(int)

# Add intercept
X_with_intercept = add_intercept(X)

# Fit
solver = UAG(model_type="logistic", penalty="MCP")
solver.fit(X_with_intercept, y, lambda_=0.05)

# Predictions
y_pred = solver.predict(X_with_intercept)

accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.3f}")
```

### Solution Path

```python
from nonconvexAG import SolutionPath
from nonconvexAG.utils import add_intercept

# Prepare data
X_with_intercept = add_intercept(X)

# Compute path
path = SolutionPath(model_type="linear", penalty="SCAD")
path.fit(X_with_intercept, y, n_lambdas=50)

# Access results
print(f"Lambda values: {path.lambda_path_}")
print(f"Coefficients shape: {path.coef_path_.shape}")  # (n_features, n_lambdas)

# Find best lambda (example with simple validation)
mse_values = []
for i, lam in enumerate(path.lambda_path_):
    coef = path.coef_path_[:, i]
    y_pred = X @ coef  # Note: X without intercept for prediction
    mse = np.mean((y - y_pred)**2)
    mse_values.append(mse)

best_idx = np.argmin(mse_values)
print(f"Best lambda: {path.lambda_path_[best_idx]:.4f}")
```

### Strong Rules (High-Dimensional)

```python
from nonconvexAG import StrongRuleSolver
from nonconvexAG.utils import add_intercept

# High-dimensional case (p >> n)
n, p = 100, 1000
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:10] = np.random.randn(10) * 2  # 10 true features
y = X @ true_beta + 0.1 * np.random.randn(n)

# Add intercept
X_with_intercept = add_intercept(X)

# Fit with strong rule
solver = StrongRuleSolver(model_type="linear", penalty="SCAD")
solver.fit(X_with_intercept, y, lambda_=0.1)

print(f"Active features: {len(solver.active_set_)}/{p}")
print(f"True features recovered: {np.sum(np.abs(solver.coef_[:10]) > 1e-6)}/10")
```

## Key Parameters

- `model_type`: "linear" or "logistic"
- `penalty`: "SCAD" or "MCP"
- `lambda_`: regularization parameter (larger = more sparse)
- `a`: SCAD parameter (default: 3.7, recommended)
- `gamma`: MCP parameter (default: 2.0)

## Important Notes

1. **Always add intercept column** using `add_intercept(X)` before fitting
2. The package assumes the first column is the intercept (not penalized)
3. Use cross-validation to select optimal `lambda_`
4. For high-dimensional data (p >> n), use `StrongRule` for speed

## Legacy Functions

Still supported for backward compatibility:
- `UAG_LM_SCAD_MCP`, `UAG_logistic_SCAD_MCP`
- `solution_path_LM`, `solution_path_logistic`
- `UAG_LM_SCAD_MCP_strongrule`, `UAG_logistic_SCAD_MCP_strongrule`
- Memory mapping versions: `memmap_*` functions

## Citation

```bibtex
@article{yang2020restarting,
  title={Restarting accelerated gradient methods with a rough strong convexity estimate},
  author={Yang, Kai},
  journal={arXiv preprint arXiv:2009.10629},
  year={2020}
}
```

## Quick Reference

### Classes
```python
# Main solvers
UAG(model_type="linear", penalty="SCAD")           # Standard solver
SolutionPath(model_type="linear", penalty="MCP")   # Compute path
StrongRuleSolver(model_type="logistic", penalty="SCAD")  # High-dimensional

# Utilities
add_intercept(X)                    # Add intercept column
standardize_data(X, y)              # Standardize features
lambda_max_LM(X, y)                 # Get lambda_max
```

### Common Workflows
```python
# 1. Basic fitting
X_with_int = add_intercept(X)
solver = UAG(model_type="linear", penalty="SCAD")
solver.fit(X_with_int, y, lambda_=0.1)

# 2. Solution path
path = SolutionPath(model_type="linear", penalty="MCP")
path.fit(X_with_int, y, n_lambdas=50)

# 3. High-dimensional
solver = StrongRuleSolver(model_type="linear", penalty="SCAD")
solver.fit(X_with_int, y, lambda_=0.1)
```

### Tips
- Always add intercept with `add_intercept(X)`
- Use `StrongRule` when p > n
- Try both SCAD and MCP penalties
- Use cross-validation for lambda selection
- Check `solver.converged_` after fitting

## License

GNU Affero General Public License v3.0