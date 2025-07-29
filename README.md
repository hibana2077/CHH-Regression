# CHH-Regression: Combined Huber-Correntropy Hybrid Regression with MCC

A robust regression implementation based on the CHH loss function, combining the bounded influence property of Huber loss with the fast down-weighting capability of Correntropy. The system also includes a complete Maximum Correntropy Criterion (MCC) regressor for comparison.

## System Architecture

This system adopts a modular design with comprehensive robust regression methods:

```
src/
├── __init__.py              # Package initialization
├── loss_functions.py        # CHH & Welsch/MCC loss functions (150 lines)
├── optimizer.py             # CHH MM-IRLS optimization algorithm (99 lines)
├── mcc_regressor.py         # MCC/Welsch regressor implementation (300 lines)
├── data_loader.py           # Data loading module (98 lines)
├── evaluation.py            # Evaluation metrics and benchmarks (420 lines)
├── visualization.py         # Visualization tools (99 lines)
├── utils.py                 # Auxiliary utility functions (410 lines)
├── main.py                  # Main execution script (96 lines)
└── test_system.py           # System testing script (160 lines)
```

## Core Features

### 1. CHH Loss Function (`loss_functions.py`)
- **Hybrid Loss**: ρ_CHH(r) = H_δ(r) + β(1 - exp(-r²/2σ²))
- **Bounded Influence Function**: Combines the linear bound of Huber and the exponential down-weighting of correntropy
- **Welsch/MCC Loss**: ρ_MCC(r) = 1 - exp(-r²/2σ²) for direct comparison
- **Automatic Parameter Estimation**: Based on MAD scale estimation and 95% efficiency setting

### 2. Optimization Algorithms
- **CHH MM-IRLS** (`optimizer.py`): Monotonic convergence for CHH loss
- **MCC IRLS** (`mcc_regressor.py`): Maximum Correntropy Criterion with sklearn-style API
- **Auto Parameter Selection**: Automatic sigma estimation for MCC regressor

### 3. Comprehensive Evaluation (`evaluation.py`)
- **Multi-Method Comparison**: OLS, Huber, MCC, and CHH side-by-side
- **Contamination Test**: Robustness evaluation under different outlier ratios
- **Cross-Validation**: Supports parameter grid search and performance comparison
- **Statistical Metrics**: RMSE, MAE, R², quantile residuals, etc.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Usage

#### CHH Regression
```python
from src.optimizer import CHHRegressor
from src.utils import generate_synthetic_data

# Generate data containing outliers
X, y, _ = generate_synthetic_data(n_samples=200, outlier_fraction=0.15)

# Fit CHH regression
chh = CHHRegressor(beta=1.0, max_iter=50)
chh.fit(X, y)

# Predict
y_pred = chh.predict(X)
```

#### MCC/Welsch Regression
```python
from src.mcc_regressor import MCCRegressor, auto_mcc_regressor

# Manual sigma setting
mcc = MCCRegressor(sigma=1.0, max_iter=100)
mcc.fit(X, y)

# Automatic sigma estimation
auto_mcc = auto_mcc_regressor(X, y)
auto_mcc.fit(X, y)

# Predict
y_pred_mcc = mcc.predict(X)
```

#### Compare Methods
```python
from src.evaluation import compare_regressors

# Compare all robust methods
results = compare_regressors(
    X, y,
    contamination_rates=[0.0, 0.1, 0.2],
    outlier_magnitude=8.0
)
```

### 3. Run Tests
```bash
cd src
python test_system.py
```

### 4. Run MCC Demo
```bash
python demo_mcc.py
```

### 5. Complete Experiment
```bash
cd src  
python main.py
```

### 6. Usage Example
```bash
python example_usage.py
```

## Datasets

The system supports three public datasets:

1. **UCI Airfoil Self-Noise** (`src/dataset/airfoil_self_noise.data`)
    - 1503 samples, 5 features
    - NASA wind tunnel noise measurement data

2. **UCI Yacht Hydrodynamics** (`src/dataset/yacht_hydrodynamics.data`)
    - 308 samples, 6 features
    - Hull resistance prediction data

3. **California Housing** (sklearn built-in)
    - 20640 samples, 8 features
    - California housing price data, can be sampled for use

## Main Parameters

### CHHRegressor Parameters
- `delta`: Huber threshold (default: 1.345 * estimated scale)
- `sigma`: Correntropy kernel width (default: estimated scale)
- `beta`: Correntropy weight (default: 1.0)
  - `beta=0`: Pure Huber regression
  - `beta>0`: CHH hybrid regression
  - A larger `beta` is closer to pure correntropy behavior

### MCCRegressor Parameters
- `sigma`: Gaussian kernel width (default: 1.0)
  - Smaller `sigma`: Stronger outlier suppression
  - Larger `sigma`: More similar to OLS behavior
- `alpha`: L2 regularization coefficient (default: 0.0)
- `max_iter`: Maximum iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)
- `init`: Initialization method ("ols" or "zeros")
- `sample_weight_mode`: How to handle external sample weights

## Experimental Results

### Contamination Test
Performance comparison under different outlier contamination rates:

| Method | 0% | 5% | 10% | 20% |
|------|----|----|-----|-----|
| OLS | Best | Poor | Very Poor | Fails |
| Huber | Good | Good | Better | Better |
| MCC | Good | Very Good | Best | Excellent |
| CHH | Best | Best | Best | Best |

### Convergence
- **MM-IRLS (CHH)**: Monotonic convergence, 10-30 iterations
- **IRLS (MCC)**: Fast convergence, typically 5-15 iterations
- Both methods only require solving WLS problems in each iteration

## Theoretical Basis

### 1. Loss Function Properties
- **Convex-Concave Hybrid**: Maintains convexity in the small residual region and introduces concavity in the large residual region
- **Bounded Influence Function**: |ψ(r)| ≤ δ, providing robustness guarantee
- **Smooth Transition**: Smooth transition from Huber behavior to correntropy behavior

### 2. MM Algorithm Theory
- Uses the tangent inequality of the exponential function to construct an upper bound
- Minimizes the reachable upper bound function in each step
- Guarantees that the original objective function value is monotonically non-increasing

### 3. Statistical Properties
- **Consistency**: Has consistency under symmetric noise
- **High Efficiency**: Approaches 95% efficiency under normal distribution
- **Breakdown Point**: Can handle up to 30% outlier contamination

## Visualization Features

The system provides rich visualization tools:

1. **Loss Function Comparison**: Shows the shape and influence function of different loss functions
2. **Convergence Process**: Displays the convergence history of the MM-IRLS algorithm
3. **Residual Analysis**: Includes residual plots, weight distribution, etc.
4. **Benchmark Comparison**: Performance comparison of different methods under various contamination rates
5. **Parameter Sensitivity**: Impact of δ, σ, β parameters on performance

## Extension Features

### 1. Automatic Parameter Tuning
```python
from src.utils import cross_validate_parameters

param_grid = {
     'beta': [0.5, 1.0, 2.0],
     'delta': [1.0, 1.345, 2.0]
}

best_params = cross_validate_parameters(X, y, param_grid)
```

### 2. Influence Diagnostics
```python
from src.utils import compute_influence_diagnostics

diagnostics = compute_influence_diagnostics(X, y, fitted_model)
# Includes: residuals, weights, leverage values, Cook's distance, etc.
```

### 3. Bootstrap Confidence Intervals
```python
from src.utils import bootstrap_confidence_intervals

ci_results = bootstrap_confidence_intervals(X, y, estimator, n_bootstrap=100)
```

## File Description

### Core Modules
- **`loss_functions.py`**: CHH & Welsch/MCC loss functions implementation
- **`optimizer.py`**: MM-IRLS algorithm and CHHRegressor class
- **`mcc_regressor.py`**: Maximum Correntropy Criterion regressor with sklearn API
- **`data_loader.py`**: Unified data loading interface
- **`evaluation.py`**: Comprehensive evaluation framework with multi-method comparison
- **`visualization.py`**: Rich visualization tools for loss functions and results
- **`utils.py`**: Collection of auxiliary functions including outlier injection

### Execution Scripts
- **`main.py`**: Complete experimental demonstration
- **`demo_mcc.py`**: MCC regressor demonstration and comparison
- **`test_system.py`**: System function testing including MCC
- **`example_usage.py`**: Usage examples for both CHH and MCC

## Technical Details

### MM-IRLS Algorithm Steps
1. **Initialization**: Use OLS or Huber regression to obtain initial estimates
2. **Weight Calculation**: Calculate Huber weights and correntropy weights
3. **WLS Solution**: Solve (X^T W X)β = X^T W y
4. **Convergence Check**: Check if the relative change is less than the tolerance value
5. **Iteration**: Repeat steps 2-4 until convergence

### Parameter Setting Recommendations
- **δ**: Use 1.345 * MAD to obtain 95% normal efficiency
- **σ**: Use MAD estimation of initial residuals
- **β**: Adjust according to data characteristics, use larger values for heavy-tailed/impulse noise

## Performance Characteristics

### Computational Complexity
- Each iteration: O(np² + p³), where n is the number of samples and p is the number of features
- Total complexity: O(k·np² + k·p³), where k is the number of iterations
- Usually k < 30, suitable for medium-scale problems

### Memory Usage
- Main memory requirements: Design matrix X and weight matrix W
- Space complexity: O(np + p²)
- Supports in-place operations to reduce memory usage

## Citation

If you use this implementation, please cite the CHH loss function and MM-IRLS algorithm from the original paper.

## License

This project follows the MIT license terms.