"""
Main execution script for CHH-Regression experiments
Demonstrates the complete CHH-Regression system with comprehensive evaluation
"""

import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader, get_dataset_info
from optimizer import CHHRegressor
from evaluation import RegressionBenchmark, parameter_sensitivity_analysis, convergence_analysis
from visualization import (
    plot_loss_functions, plot_convergence, plot_residuals_analysis,
    plot_benchmark_results, plot_parameter_sensitivity, create_results_report
)


def run_single_experiment(dataset_name: str = 'airfoil') -> None:
    """
    Run a single experiment on specified dataset
    
    Args:
        dataset_name: Name of dataset ('airfoil', 'yacht', 'california')
    """
    print(f"\n=== CHH-Regression Experiment: {dataset_name.upper()} ===")
    
    # Load data
    loader = DataLoader(standardize=True)
    try:
        if dataset_name == 'california':
            X, y = loader.load_dataset(dataset_name, sample_fraction=0.1)  # 10% sample
        else:
            X, y = loader.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create and fit CHH regressor
    chh = CHHRegressor(beta=1.0, max_iter=50)
    chh.fit(X, y)
    
    print(f"CHH fitting completed in {chh.n_iter_} iterations")
    print(f"Final loss: {chh.loss_history_[-1]:.6f}")
    
    # Make predictions and compute metrics
    y_pred = chh.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Training MAE: {mae:.4f}")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Plot convergence
    plot_convergence(chh.loss_history_, f"CHH Convergence - {dataset_name.title()}")
    
    # Plot residuals analysis
    weights = chh.get_weights(X, y)
    plot_residuals_analysis(y, y_pred, weights, f"CHH - {dataset_name.title()}")


def run_comprehensive_benchmark(dataset_name: str = 'airfoil') -> None:
    """
    Run comprehensive benchmark comparing multiple methods
    
    Args:
        dataset_name: Name of dataset to benchmark
    """
    print(f"\n=== Comprehensive Benchmark: {dataset_name.upper()} ===")
    
    # Load data
    loader = DataLoader(standardize=True)
    try:
        if dataset_name == 'california':
            X, y = loader.load_dataset(dataset_name, sample_fraction=0.1)
        else:
            X, y = loader.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run benchmark
    benchmark = RegressionBenchmark(random_state=42)
    results = benchmark.run_benchmark(
        X, y,
        contamination_rates=[0.0, 0.05, 0.1, 0.2]
    )
    
    # Print results summary
    print("\n" + "="*60)
    benchmark.print_summary(results, 'rmse')
    benchmark.print_summary(results, 'mae')
    
    # Plot results
    print("\nGenerating benchmark plots...")
    plot_benchmark_results(results, 'rmse')
    plot_benchmark_results(results, 'mae')


def run_parameter_analysis(dataset_name: str = 'yacht') -> None:
    """
    Run parameter sensitivity analysis
    
    Args:
        dataset_name: Name of dataset for analysis
    """
    print(f"\n=== Parameter Sensitivity Analysis: {dataset_name.upper()} ===")
    
    # Load data
    loader = DataLoader(standardize=True)
    try:
        X, y = loader.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run sensitivity analysis
    print("Running parameter sensitivity analysis...")
    sensitivity_results = parameter_sensitivity_analysis(
        X, y,
        delta_range=[0.8, 1.0, 1.345, 2.0],
        sigma_range=[0.8, 1.0, 1.2],
        beta_range=[0.5, 1.0, 2.0],
        cv_folds=3
    )
    
    print(f"Tested {len(sensitivity_results)} parameter combinations")
    
    # Find best parameters
    best_result = max(sensitivity_results, key=lambda x: x['mean_score'])
    print(f"\nBest parameters:")
    print(f"  Delta: {best_result['delta']}")
    print(f"  Sigma: {best_result['sigma']}")
    print(f"  Beta: {best_result['beta']}")
    print(f"  Score: {-best_result['mean_score']:.4f}")
    
    # Plot sensitivity
    plot_parameter_sensitivity(sensitivity_results)


def demonstrate_loss_functions() -> None:
    """Demonstrate CHH loss function properties"""
    print("\n=== CHH Loss Function Demonstration ===")
    
    # Plot loss functions with different parameters
    print("Plotting loss function comparisons...")
    
    plot_loss_functions(
        residual_range=(-4, 4),
        delta=1.345,
        sigma=1.0,
        beta=1.0
    )
    
    # Show effect of different beta values
    for beta in [0.5, 1.0, 2.0]:
        plot_loss_functions(
            residual_range=(-4, 4),
            delta=1.345,
            sigma=1.0,
            beta=beta
        )


def main():
    """Main execution function"""
    print("CHH-Regression: Combined Huber-Correntropy Hybrid Regression")
    print("=" * 60)
    
    # Display available datasets
    dataset_info = get_dataset_info()
    print("\nAvailable datasets:")
    for name, info in dataset_info.items():
        print(f"  {name}: {info['description']} (n={info['n_samples']}, p={info['n_features']})")
    
    # Demonstrate loss functions
    demonstrate_loss_functions()
    
    # Run experiments on different datasets
    for dataset in ['airfoil', 'yacht']:
        try:
            run_single_experiment(dataset)
        except Exception as e:
            print(f"Error in {dataset} experiment: {e}")
    
    # Run comprehensive benchmark
    try:
        run_comprehensive_benchmark('airfoil')
    except Exception as e:
        print(f"Error in comprehensive benchmark: {e}")
    
    # Run parameter sensitivity analysis
    try:
        run_parameter_analysis('yacht')
    except Exception as e:
        print(f"Error in parameter analysis: {e}")
    
    print("\n" + "="*60)
    print("CHH-Regression experiments completed!")
    print("Check the generated plots for detailed analysis.")


if __name__ == "__main__":
    main()
