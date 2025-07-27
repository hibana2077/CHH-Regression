"""
CHH-Regression Visualization Demo
Demonstrates the key visualizations for understanding CHH regression
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from visualization import plot_loss_functions, plot_convergence, plot_residuals_analysis
    from optimizer import CHHRegressor
    from utils import generate_synthetic_data
    print("✓ Successfully imported visualization modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def demo_loss_functions():
    """Demonstrate CHH loss function visualization"""
    print("\n=== CHH Loss Function Visualization ===")
    
    # Show how CHH loss changes with different beta values
    plt.figure(figsize=(15, 10))
    
    for i, beta in enumerate([0.0, 0.5, 1.0, 2.0]):
        plt.subplot(2, 2, i+1)
        
        r = np.linspace(-4, 4, 1000)
        
        # Create CHH loss with different beta
        from loss_functions import CHHLoss
        chh = CHHLoss(delta=1.345, sigma=1.0, beta=beta)
        
        # Compute losses
        chh_loss = chh.loss(r)
        huber_loss = CHHLoss(1.345, 1.0, 0.0).loss(r)  # Pure Huber
        
        plt.plot(r, huber_loss, 'g--', label='Huber', alpha=0.7)
        plt.plot(r, chh_loss, 'r-', label=f'CHH (β={beta})', linewidth=2)
        
        plt.xlabel('Residual')
        plt.ylabel('Loss')
        plt.title(f'CHH Loss (β={beta})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 8)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Loss function visualization complete")


def demo_convergence_analysis():
    """Demonstrate convergence analysis"""
    print("\n=== Convergence Analysis ===")
    
    # Generate data
    X, y, _ = generate_synthetic_data(
        n_samples=150, 
        n_features=3,
        outlier_fraction=0.15,
        random_state=42
    )
    
    # Fit CHH with different beta values
    beta_values = [0.0, 0.5, 1.0, 2.0]
    
    plt.figure(figsize=(12, 8))
    
    for i, beta in enumerate(beta_values):
        plt.subplot(2, 2, i+1)
        
        chh = CHHRegressor(beta=beta, max_iter=30, tol=1e-8)
        chh.fit(X, y)
        
        plt.plot(chh.loss_history_, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('CHH Loss')
        plt.title(f'Convergence (β={beta}, {chh.n_iter_} iterations)')
        plt.grid(True, alpha=0.3)
        
        # Add final loss text
        final_loss = chh.loss_history_[-1]
        plt.text(0.7, 0.9, f'Final: {final_loss:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Convergence analysis complete")


def demo_residuals_analysis():
    """Demonstrate residuals analysis"""
    print("\n=== Residuals Analysis ===")
    
    # Generate data with outliers
    X, y, _ = generate_synthetic_data(
        n_samples=200,
        n_features=3,
        outlier_fraction=0.2,
        outlier_magnitude=5.0,
        random_state=42
    )
    
    # Fit CHH regressor
    chh = CHHRegressor(beta=1.0, max_iter=30)
    chh.fit(X, y)
    
    # Get predictions and weights
    y_pred = chh.predict(X)
    weights = chh.get_weights(X, y)
    
    # Create residuals analysis plot
    plot_residuals_analysis(y, y_pred, weights, "CHH Regression")
    
    print("✓ Residuals analysis complete")


def demo_robustness_comparison():
    """Demonstrate robustness comparison"""
    print("\n=== Robustness Comparison ===")
    
    from sklearn.linear_model import LinearRegression, HuberRegressor
    from data_loader import add_outliers
    
    # Generate clean data
    X, y_clean, _ = generate_synthetic_data(
        n_samples=150,
        n_features=2,
        noise_std=0.5,
        outlier_fraction=0.0,
        random_state=42
    )
    
    contamination_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    # Store results
    ols_rmse = []
    huber_rmse = []
    chh_rmse = []
    
    for cont_rate in contamination_rates:
        # Add outliers
        y_cont, _ = add_outliers(y_clean, cont_rate, 6.0, random_state=42)
        
        # Fit models
        ols = LinearRegression()
        ols.fit(X, y_cont)
        y_pred_ols = ols.predict(X)
        
        huber = HuberRegressor(epsilon=1.35, max_iter=100)
        huber.fit(X, y_cont)
        y_pred_huber = huber.predict(X)
        
        chh = CHHRegressor(beta=1.0, max_iter=30)
        chh.fit(X, y_cont)
        y_pred_chh = chh.predict(X)
        
        # Compute RMSE on clean data
        ols_rmse.append(np.sqrt(np.mean((y_clean - y_pred_ols)**2)))
        huber_rmse.append(np.sqrt(np.mean((y_clean - y_pred_huber)**2)))
        chh_rmse.append(np.sqrt(np.mean((y_clean - y_pred_chh)**2)))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot([c*100 for c in contamination_rates], ols_rmse, 'b-o', label='OLS', linewidth=2)
    plt.plot([c*100 for c in contamination_rates], huber_rmse, 'g-s', label='Huber', linewidth=2)
    plt.plot([c*100 for c in contamination_rates], chh_rmse, 'r-^', label='CHH', linewidth=2)
    
    plt.xlabel('Contamination Rate (%)')
    plt.ylabel('RMSE (on clean data)')
    plt.title('Robustness Comparison: CHH vs OLS vs Huber')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("✓ Robustness comparison complete")


def main():
    """Run all visualization demos"""
    print("CHH-Regression Visualization Demo")
    print("=" * 40)
    
    try:
        demo_loss_functions()
        demo_convergence_analysis()
        demo_residuals_analysis()
        demo_robustness_comparison()
        
        print("\n" + "=" * 40)
        print("✓ All visualization demos completed!")
        print("\nThe plots demonstrate:")
        print("• How CHH loss function changes with β parameter")
        print("• Convergence behavior of MM-IRLS algorithm")
        print("• Residuals analysis and weight distribution")
        print("• Superior robustness compared to OLS and Huber")
        
    except Exception as e:
        print(f"Error in visualization demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
