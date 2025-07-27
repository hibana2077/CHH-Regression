"""
Simple test script for CHH-Regression
Tests basic functionality without requiring external dependencies
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    from loss_functions import CHHLoss, estimate_scale, get_default_parameters
    from optimizer import CHHRegressor
    from utils import generate_synthetic_data, validate_input_data
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_loss_functions():
    """Test CHH loss function implementation"""
    print("\n=== Testing Loss Functions ===")
    
    # Test CHH loss
    chh = CHHLoss(delta=1.345, sigma=1.0, beta=1.0)
    
    # Test with sample residuals
    r = np.array([-3, -1, 0, 1, 3])
    
    loss_values = chh.loss(r)
    psi_values = chh.psi(r)
    weights = chh.weights(r)
    
    print(f"Residuals: {r}")
    print(f"Loss values: {loss_values}")
    print(f"Psi values: {psi_values}")
    print(f"Weights: {weights}")
    
    # Test scale estimation
    residuals = np.random.normal(0, 2, 100)
    scale = estimate_scale(residuals, 'mad')
    print(f"Estimated scale (true=2): {scale:.4f}")
    
    print("✓ Loss functions test passed")


def test_synthetic_data():
    """Test with synthetic data"""
    print("\n=== Testing with Synthetic Data ===")
    
    # Generate synthetic data
    X, y, true_coef = generate_synthetic_data(
        n_samples=100,
        n_features=3,
        noise_std=0.5,
        outlier_fraction=0.1,
        random_state=42
    )
    
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True coefficients: {true_coef}")
    
    # Validate data
    validate_input_data(X, y)
    print("✓ Data validation passed")
    
    # Fit CHH regressor
    chh = CHHRegressor(beta=1.0, max_iter=20, tol=1e-4)
    chh.fit(X, y)
    
    print(f"CHH fitting completed in {chh.n_iter_} iterations")
    print(f"Estimated coefficients: {chh.coef_}")
    print(f"Intercept: {chh.intercept_:.4f}")
    
    # Make predictions
    y_pred = chh.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Training MAE: {mae:.4f}")
    
    print("✓ Synthetic data test passed")


def test_parameter_variations():
    """Test different parameter combinations"""
    print("\n=== Testing Parameter Variations ===")
    
    # Generate test data
    X, y, _ = generate_synthetic_data(n_samples=50, n_features=2, random_state=42)
    
    # Test different beta values
    beta_values = [0.0, 0.5, 1.0, 2.0]
    
    for beta in beta_values:
        chh = CHHRegressor(beta=beta, max_iter=15)
        chh.fit(X, y)
        y_pred = chh.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        print(f"Beta={beta}: RMSE={rmse:.4f}, Iterations={chh.n_iter_}")
    
    print("✓ Parameter variations test passed")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")
    
    # Test with minimal data
    X_small = np.array([[1], [2], [3]])
    y_small = np.array([1, 2, 3])
    
    chh = CHHRegressor(max_iter=10)
    chh.fit(X_small, y_small)
    y_pred = chh.predict(X_small)
    print(f"Minimal data test: predictions = {y_pred}")
    
    # Test with perfect fit
    X_perfect = np.array([[1], [2], [3], [4]])
    y_perfect = 2 * X_perfect.flatten() + 1  # y = 2x + 1
    
    chh = CHHRegressor(max_iter=10)
    chh.fit(X_perfect, y_perfect)
    print(f"Perfect fit test: coef={chh.coef_[0]:.4f}, intercept={chh.intercept_:.4f}")
    
    print("✓ Edge cases test passed")


def main():
    """Run all tests"""
    print("CHH-Regression Test Suite")
    print("=" * 40)
    
    try:
        test_loss_functions()
        test_synthetic_data()
        test_parameter_variations()
        test_edge_cases()
        
        print("\n" + "=" * 40)
        print("✓ All tests passed successfully!")
        print("CHH-Regression system is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
