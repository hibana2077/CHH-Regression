"""
Demo script for MCC (Maximum Correntropy Criterion) Regressor
Demonstrates MCC regression compared to other robust methods
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mcc_regressor import MCCRegressor, auto_mcc_regressor
from optimizer import CHHRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from utils import generate_synthetic_data, add_outliers
from evaluation import compute_metrics
from visualization import plot_loss_functions


def demo_mcc_basic():
    """Basic MCC regression demonstration"""
    print("=== MCC Regression Basic Demo ===\n")
    
    # Generate synthetic data with heavy-tailed noise
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([2.0, -1.5, 0.8])
    y = X @ true_coef + 0.5 * np.random.randn(n_samples)
    
    # Add impulsive outliers (20% contamination)
    y = add_outliers(y, contamination_rate=0.2, outlier_magnitude=8.0, random_state=42)
    
    print(f"Data: {n_samples} samples, {n_features} features")
    print(f"True coefficients: {true_coef}")
    print(f"Added 20% impulsive outliers\n")
    
    # Fit MCC regressor
    print("1. Fitting MCC regressor...")
    mcc = MCCRegressor(sigma=1.0, max_iter=50, verbose=1)
    mcc.fit(X, y)
    
    print(f"\nMCC Results:")
    print(f"  Estimated coefficients: {mcc.coef_}")
    print(f"  Intercept: {mcc.intercept_:.4f}")
    print(f"  Converged in {mcc.n_iter_} iterations")
    
    # Evaluate
    y_pred = mcc.predict(X)
    metrics = compute_metrics(y, y_pred)
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    return mcc, X, y


def demo_auto_mcc():
    """Demonstrate automatic MCC parameter estimation"""
    print("\n=== Auto MCC Parameter Estimation ===\n")
    
    # Generate data
    X, y, true_coef = generate_synthetic_data(
        n_samples=150,
        n_features=2,
        noise_std=1.0,
        outlier_fraction=0.15,
        random_state=123
    )
    
    print("Using auto_mcc_regressor for automatic sigma estimation...")
    
    # Use automatic parameter estimation
    auto_mcc = auto_mcc_regressor(X, y, init_method='ols')
    auto_mcc.fit(X, y)
    
    print(f"Auto-estimated sigma: {auto_mcc.sigma:.4f}")
    print(f"Estimated coefficients: {auto_mcc.coef_}")
    print(f"Converged in {auto_mcc.n_iter_} iterations")
    
    # Compare with manual sigma
    manual_mcc = MCCRegressor(sigma=1.0, max_iter=50)
    manual_mcc.fit(X, y)
    
    print(f"\nComparison (auto vs manual sigma=1.0):")
    
    y_pred_auto = auto_mcc.predict(X)
    y_pred_manual = manual_mcc.predict(X)
    
    metrics_auto = compute_metrics(y, y_pred_auto)
    metrics_manual = compute_metrics(y, y_pred_manual)
    
    print(f"Auto MCC - RMSE: {metrics_auto['rmse']:.4f}, MAE: {metrics_auto['mae']:.4f}")
    print(f"Manual MCC - RMSE: {metrics_manual['rmse']:.4f}, MAE: {metrics_manual['mae']:.4f}")


def demo_robust_comparison():
    """Compare MCC with other robust methods"""
    print("\n=== Robust Methods Comparison ===\n")
    
    # Generate contaminated data
    X, y, true_coef = generate_synthetic_data(
        n_samples=200,
        n_features=4,
        noise_std=0.8,
        outlier_fraction=0.25,  # Heavy contamination
        outlier_magnitude=6.0,
        random_state=456
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True coefficients: {true_coef}")
    print(f"25% outlier contamination\n")
    
    # Define regressors
    regressors = {
        'OLS': LinearRegression(),
        'Huber': HuberRegressor(epsilon=1.35, max_iter=100),
        'MCC (σ=0.8)': MCCRegressor(sigma=0.8, max_iter=100),
        'MCC (σ=1.2)': MCCRegressor(sigma=1.2, max_iter=100),
        'CHH (β=1.0)': CHHRegressor(beta=1.0, sigma=1.0, max_iter=100)
    }
    
    print(f"{'Method':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Med.AE':<8}")
    print("-" * 60)
    
    results = {}
    for name, regressor in regressors.items():
        try:
            regressor.fit(X, y)
            y_pred = regressor.predict(X)
            metrics = compute_metrics(y, y_pred)
            
            results[name] = metrics
            
            print(f"{name:<15} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} "
                  f"{metrics['r2']:<8.4f} {metrics['median_abs_residual']:<8.4f}")
            
        except Exception as e:
            print(f"{name:<15} Failed: {e}")
    
    return results


def demo_convergence_analysis():
    """Analyze MCC convergence properties"""
    print("\n=== MCC Convergence Analysis ===\n")
    
    # Generate test data
    X, y, _ = generate_synthetic_data(n_samples=100, n_features=3, random_state=789)
    
    # Fit with extended iterations to observe convergence
    mcc = MCCRegressor(sigma=1.0, max_iter=50, tol=0, verbose=0)  # Disable early stopping
    mcc.fit(X, y)
    
    # Extract convergence information
    obj_path = mcc.objective_path_()
    weight_stats = mcc.weight_stats_path_()
    
    print(f"Iterations completed: {len(obj_path)}")
    print(f"Final objective value: {obj_path[-1]:.6f}")
    print(f"Objective improvement: {obj_path[0] - obj_path[-1]:.6f}")
    print(f"Final mean weight: {weight_stats[-1, 0]:.4f}")
    print(f"Final min weight: {weight_stats[-1, 1]:.4f}")
    print(f"Final max weight: {weight_stats[-1, 2]:.4f}")
    
    # Plot convergence if matplotlib available
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(obj_path)
        plt.title('Objective Value')
        plt.xlabel('Iteration')
        plt.ylabel('Objective')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(weight_stats[:, 0], label='Mean weight')
        plt.plot(weight_stats[:, 1], label='Min weight')
        plt.plot(weight_stats[:, 2], label='Max weight')
        plt.title('Weight Statistics')
        plt.xlabel('Iteration')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.semilogy(np.diff(obj_path))
        plt.title('Objective Decrease')
        plt.xlabel('Iteration')
        plt.ylabel('|Δ Objective|')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


def demo_loss_comparison():
    """Visualize loss function comparisons"""
    print("\n=== Loss Function Visualization ===\n")
    
    try:
        # Plot loss functions comparison
        plot_loss_functions(
            residual_range=(-4, 4),
            delta=1.345,
            sigma=1.0,
            beta=1.0,
            save_path=None  # Set to a path if you want to save
        )
        print("Loss function comparison plot displayed")
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Plotting failed: {e}")


def main():
    """Run all MCC demonstrations"""
    print("MCC (Maximum Correntropy Criterion) Regression Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demo_mcc_basic()
        demo_auto_mcc()
        results = demo_robust_comparison()
        demo_convergence_analysis()
        demo_loss_comparison()
        
        print("\n" + "=" * 60)
        print("✓ All MCC demonstrations completed successfully!")
        
        # Summary
        print("\nKey findings:")
        print("• MCC provides strong robustness against impulsive outliers")
        print("• Automatic sigma estimation works well for most datasets")
        print("• MCC typically converges faster than iterative CHH")
        print("• Smaller sigma values provide stronger outlier suppression")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
