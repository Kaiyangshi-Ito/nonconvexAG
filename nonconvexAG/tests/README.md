# nonconvexAG Test Suite

This directory contains comprehensive tests for the nonconvexAG package, organized into unit and integration tests.

## Test Organization

### Unit Tests (`tests/unit/`)
- **`test_penalties.py`**: Basic penalty function tests
- **`test_penalties_comprehensive.py`**: Exhaustive penalty function tests including:
  - Edge cases and boundary conditions
  - Numerical gradient verification
  - Mathematical properties (non-negativity, symmetry, continuity)
  - Parameter sensitivity
  - Subdifferential behavior

- **`test_models.py`**: Tests for LinearModel and LogisticModel classes
  - Loss and gradient calculations
  - Predictions and probability bounds
  - Numerical stability
  - JIT function consistency

- **`test_utils.py`**: Tests for utility functions
  - Lambda parameter calculations
  - Data preprocessing (standardization, intercept)
  - Input validation
  - Convergence checking

### Integration Tests (`tests/integration/`)
- **`test_solvers.py`**: Basic solver integration tests
- **`test_solvers_comprehensive.py`**: Exhaustive solver tests including:
  - Different penalty parameters
  - Convergence behavior
  - Warm starts
  - Sparsity patterns
  - Edge cases (perfect separation, multicollinearity)

- **`test_backward_compatibility.py`**: Tests for deprecated API
  - Old function signatures still work
  - Deprecation warnings are issued
  - Results match new API

- **`test_performance.py`**: Performance benchmarks and stress tests
  - Scaling with problem size
  - Strong rule efficiency
  - Memory usage
  - High-dimensional problems
  - Extreme values

### Test Fixtures (`tests/conftest.py`)
- `linear_regression_data`: Synthetic linear regression dataset
- `logistic_regression_data`: Synthetic logistic regression dataset
- `small_data`: Small dataset for quick tests
- `penalty_type`: Parametrized fixture for SCAD/MCP
- `lambda_value`: Parametrized fixture for regularization values

## Running Tests

### Basic Usage
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nonconvexAG --cov-report=html

# Run specific test file
pytest tests/unit/test_penalties.py

# Run tests in parallel
pytest -n auto
```

### Using the Test Runner Script
```bash
# Run all tests with coverage
python run_tests.py --coverage

# Run only unit tests
python run_tests.py --unit

# Run only integration tests  
python run_tests.py --integration

# Skip slow tests
python run_tests.py --fast

# Run only slow tests (performance/stress)
python run_tests.py --slow

# Run in parallel with 4 workers
python run_tests.py -n 4
```

### Test Markers
- `@pytest.mark.slow`: Marks slow tests (performance benchmarks)
- `@pytest.mark.memory_intensive`: Marks tests using lots of memory
- `@pytest.mark.parametrize`: Used for testing multiple parameter combinations

## Test Coverage

The test suite aims for comprehensive coverage including:

1. **Functionality Tests**
   - All public APIs are tested
   - Both positive and negative test cases
   - Edge cases and boundary conditions

2. **Mathematical Properties**
   - Penalties are non-negative
   - Gradients match numerical approximations
   - Optimization objectives decrease
   - Solutions are stable

3. **Performance Tests**
   - Scaling with problem size
   - Memory efficiency
   - Strong rule effectiveness

4. **Robustness Tests**
   - High-dimensional problems (p >> n)
   - Highly correlated features
   - Nearly singular matrices
   - Extreme parameter values

## Adding New Tests

When adding new features, ensure you:

1. Add unit tests for individual functions
2. Add integration tests for end-to-end workflows
3. Test edge cases and error conditions
4. Add performance benchmarks if relevant
5. Update this README

## Continuous Integration

Tests are automatically run on:
- Every push to main/develop branches
- Every pull request
- Multiple OS (Ubuntu, macOS, Windows)
- Multiple Python versions (3.7-3.11)

See `.github/workflows/tests.yml` for CI configuration.