"""
Example usage of CHH-Regression
Demonstrates basic usage and key features
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from optimizer import CHHRegressor
from mcc_regressor import MCCRegressor
from data_loader import DataLoader
from evaluation import compute_metrics
from utils import generate_synthetic_data


def basic_example():
    """Basic usage example"""
    print("=== Basic CHH-Regression Example ===\n")
    
    # Generate synthetic data with outliers
    print("1. Generating synthetic data with outliers...")
    X, y, true_coef = generate_synthetic_data(
        n_samples=200,
        n_features=4,
        noise_std=1.0,
        outlier_fraction=0.15,  # 15% outliers
        outlier_magnitude=5.0,
        random_state=42
    )
    print(f"   Data shape: {X.shape}")
    print(f"   True coefficients: {true_coef}")
    
    # Fit CHH regressor
    print("\n2. Fitting CHH regressor...")
    chh = CHHRegressor(
        beta=1.0,           # Balance between Huber and correntropy
        max_iter=50,        # Maximum iterations
        tol=1e-6           # Convergence tolerance
    )
    chh.fit(X, y)
    
    print(f"   Converged in {chh.n_iter_} iterations")
    print(f"   Estimated coefficients: {chh.coef_}")
    print(f"   Intercept: {chh.intercept_:.4f}")
    
    # Make predictions and evaluate
    print("\n3. Evaluating performance...")
    y_pred = chh.predict(X)
    metrics = compute_metrics(y, y_pred)
    
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R²: {metrics['r2']:.4f}")
    print(f"   Median Absolute Error: {metrics['median_abs_residual']:.4f}")
    
    return chh, X, y


def comparison_example():
    """Compare CHH with MCC and different beta values"""
    print("\n=== Comparison of CHH, MCC and Different Beta Values ===\n")
    
    # Generate data
    X, y, _ = generate_synthetic_data(
        n_samples=150,
        n_features=3,
        outlier_fraction=0.2,
        random_state=123
    )
    
    print(f"{'Method':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Iterations':<12}")
    print("-" * 65)
    
    # Test MCC regressor
    mcc = MCCRegressor(sigma=1.0, max_iter=30)
    mcc.fit(X, y)
    y_pred_mcc = mcc.predict(X)
    metrics_mcc = compute_metrics(y, y_pred_mcc)
    
    print(f"{'MCC':<15} {metrics_mcc['rmse']:<8.4f} {metrics_mcc['mae']:<8.4f} "
          f"{metrics_mcc['r2']:<8.4f} {mcc.n_iter_:<12}")
    
    # Test CHH with different beta values
    beta_values = [0.0, 0.5, 1.0, 2.0]  # 0.0 = pure Huber
    
    for beta in beta_values:
        chh = CHHRegressor(beta=beta, max_iter=30)
        chh.fit(X, y)
        
        y_pred = chh.predict(X)
        metrics = compute_metrics(y, y_pred)
        
        method_name = f"CHH (β={beta})"
        print(f"{method_name:<15} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} "
              f"{metrics['r2']:<8.4f} {chh.n_iter_:<12}")


def real_data_example():
    """Example with real dataset (if available)"""
    print("\n=== Real Dataset Example ===\n")
    
    try:
        # Try to load real data
        loader = DataLoader(standardize=True)
        
        # Try airfoil dataset first
        try:
            X, y = loader.load_dataset('airfoil')
            dataset_name = "Airfoil Self-Noise"
        except:
            # Fallback to California housing
            try:
                X, y = loader.load_dataset('california', sample_fraction=0.1)
                dataset_name = "California Housing (10% sample)"
            except:
                print("   No real datasets available, skipping this example.")
                return
        
        print(f"   Dataset: {dataset_name}")
        print(f"   Shape: {X.shape}")
        
        # Fit CHH regressor
        chh = CHHRegressor(beta=1.0, max_iter=50)
        chh.fit(X, y)
        
        y_pred = chh.predict(X)
        metrics = compute_metrics(y, y_pred)
        
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   Converged in {chh.n_iter_} iterations")
        
    except Exception as e:
        print(f"   Error loading real data: {e}")


def robustness_demo():
    """Demonstrate robustness to outliers"""
    print("\n=== Robustness Demonstration ===\n")
    
    # Generate clean data
    X, y_clean, _ = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        noise_std=0.5,
        outlier_fraction=0.0,  # No outliers initially
        random_state=42
    )
    
    from sklearn.linear_model import LinearRegression
    
    print("Comparing OLS vs CHH with increasing outlier contamination:\n")
    print(f"{'Contamination':<14} {'OLS RMSE':<10} {'CHH RMSE':<10} {'Improvement':<12}")
    print("-" * 50)
    
    contamination_rates = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    for cont_rate in contamination_rates:
        # Add outliers
        from data_loader import add_outliers
        y_contaminated, _ = add_outliers(
            y_clean, 
            contamination_rate=cont_rate,
            outlier_magnitude=8.0,
            random_state=42
        )
        
        # Fit OLS
        ols = LinearRegression()
        ols.fit(X, y_contaminated)
        y_pred_ols = ols.predict(X)
        rmse_ols = np.sqrt(np.mean((y_clean - y_pred_ols)**2))  # Evaluate on clean data
        
        # Fit CHH
        chh = CHHRegressor(beta=1.0, max_iter=30)
        chh.fit(X, y_contaminated)
        y_pred_chh = chh.predict(X)
        rmse_chh = np.sqrt(np.mean((y_clean - y_pred_chh)**2))  # Evaluate on clean data
        
        improvement = (rmse_ols - rmse_chh) / rmse_ols * 100
        
        print(f"{cont_rate:<14.1%} {rmse_ols:<10.4f} {rmse_chh:<10.4f} {improvement:<12.1f}%")


def main():
    """Run all examples"""
    print("CHH-Regression Usage Examples")
    print("=" * 50)
    
    try:
        # Basic example
        chh, X, y = basic_example()
        
        # Comparison example
        comparison_example()
        
        # Real data example
        real_data_example()
        
        # Robustness demonstration
        robustness_demo()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nKey takeaways:")
        print("• CHH combines Huber loss robustness with correntropy re-descending")
        print("• Beta parameter controls the balance (0=pure Huber, higher=more correntropy)")
        print("• Particularly effective for data with heavy-tailed noise and outliers")
        print("• MM-IRLS algorithm ensures monotonic convergence")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
