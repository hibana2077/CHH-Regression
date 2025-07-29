"""
Main execution script for CHH-Regression experiments
Demonstrates the complete CHH-Regression system with comprehensive evaluation
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
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
from data_export import ExperimentDataExporter


def run_single_experiment(dataset_name: str = 'airfoil', exporter: ExperimentDataExporter = None) -> Dict[str, Any]:
    """
    Run a single experiment on specified dataset
    
    Args:
        dataset_name: Name of dataset ('airfoil', 'yacht', 'california')
        exporter: Data exporter instance
        
    Returns:
        Dictionary containing experiment results
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
        return {}
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create and fit CHH regressor with Huber initialization
    chh = CHHRegressor(beta=1.0, max_iter=50, init_method='huber')
    chh.fit(X, y)
    
    print(f"CHH fitting completed in {chh.n_iter_} iterations")
    print(f"Final loss: {chh.loss_history_[-1]:.6f}")
    
    # Make predictions and compute metrics
    y_pred = chh.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Training MAE: {mae:.4f}")
    
    # Collect experiment results
    experiment_results = {
        'dataset_info': {
            'name': dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
        },
        'model_parameters': {
            'beta': chh.beta,
            'delta': chh.delta,
            'sigma': chh.sigma,
            'max_iter': chh.max_iter,
            'tol': chh.tol
        },
        'training_results': {
            'n_iterations': chh.n_iter_,
            'final_loss': chh.loss_history_[-1],
            'loss_history': chh.loss_history_,
            'rmse': rmse,
            'mae': mae,
            'coefficients': chh.coef_.tolist(),
            'intercept': chh.intercept_
        },
        'predictions': {
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'residuals': (y - y_pred).tolist()
        }
    }
    
    # Export data if exporter provided
    if exporter:
        exporter.export_single_experiment(dataset_name, experiment_results)
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Plot convergence
    plot_convergence(chh.loss_history_, f"CHH Convergence - {dataset_name.title()}")
    
    # Plot residuals analysis
    weights = chh.get_weights(X, y)
    plot_residuals_analysis(y, y_pred, weights, f"CHH - {dataset_name.title()}")
    
    return experiment_results


def run_comprehensive_benchmark(dataset_name: str = 'airfoil', exporter: ExperimentDataExporter = None) -> Dict[str, Any]:
    """
    Run comprehensive benchmark comparing multiple methods
    
    Args:
        dataset_name: Name of dataset to benchmark
        exporter: Data exporter instance
        
    Returns:
        Dictionary containing benchmark results
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
        return {}
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run benchmark
    benchmark = RegressionBenchmark(random_state=42)
    results = benchmark.run_benchmark(
        X, y,
        contamination_rates=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    
    # Export benchmark results
    if exporter:
        exporter.export_benchmark_results(results, dataset_name)
    
    # Print results summary
    print("\n" + "="*60)
    benchmark.print_summary(results, 'rmse')
    benchmark.print_summary(results, 'mae')
    
    # Plot results
    print("\nGenerating benchmark plots...")
    plot_benchmark_results(results, 'rmse')
    plot_benchmark_results(results, 'mae')
    
    return results


def run_parameter_analysis(dataset_name: str = 'yacht', exporter: ExperimentDataExporter = None) -> Dict[str, Any]:
    """
    Run parameter sensitivity analysis
    
    Args:
        dataset_name: Name of dataset for analysis
        exporter: Data exporter instance
        
    Returns:
        Dictionary containing sensitivity analysis results
    """
    print(f"\n=== Parameter Sensitivity Analysis: {dataset_name.upper()} ===")
    
    # Load data
    loader = DataLoader(standardize=True)
    try:
        X, y = loader.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return {}
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run sensitivity analysis
    print("Running parameter sensitivity analysis...")
    sensitivity_results = parameter_sensitivity_analysis(
        X, y,
        delta_range=[1.0, 1.345, 2.0],  # delta/sigma_hat - more conservative
        sigma_range=[0.8, 1.0, 1.5],  # sigma/sigma_hat - closer to 1.0
        beta_range=[0.02, 0.05, 0.1, 0.2],  # more balanced beta values
        cv_folds=3,
        scoring='neg_mean_absolute_error'  # Use MAE as primary metric
    )
    
    print(f"Tested {len(sensitivity_results)} parameter combinations")
    
    # Find best parameters - only if we have results
    if sensitivity_results:
        best_result = max(sensitivity_results, key=lambda x: x['mean_score'])
        print(f"\nBest parameters:")
        print(f"  Delta scale: {best_result['delta_scale']} (actual: {best_result['delta_actual']:.4f})")
        print(f"  Sigma scale: {best_result['sigma_scale']} (actual: {best_result['sigma_actual']:.4f})")
        print(f"  Beta: {best_result['beta']}")
        print(f"  Sigma_hat: {best_result['sigma_hat']:.4f}")
        print(f"  MAE Score: {-best_result['mean_score']:.4f}")
    else:
        print("\n⚠️  No successful parameter combinations found!")
        print("This may indicate:")
        print("  • Parameter ranges are too extreme")
        print("  • Numerical instability in the optimization")
        print("  • Dataset characteristics require different parameter settings")
        best_result = None
    
    # Export sensitivity results
    if exporter:
        exporter.export_parameter_sensitivity(sensitivity_results, dataset_name)
    
    # Plot sensitivity
    plot_parameter_sensitivity(sensitivity_results)
    
    return {
        'sensitivity_results': sensitivity_results,
        'best_parameters': best_result
    }


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
    # for beta in [0.5, 1.0, 2.0]:
    for beta in [0.02, 0.05, 0.1, 0.2]:  # Smaller beta values for robustness
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
    
    # Initialize data exporter
    exporter = ExperimentDataExporter(output_dir="../results")
    print(f"✓ Data exporter initialized - results will be saved to: {exporter.output_dir}")
    
    # Display available datasets
    dataset_info = get_dataset_info()
    print("\nAvailable datasets:")
    for name, info in dataset_info.items():
        print(f"  {name}: {info['description']} (n={info['n_samples']}, p={info['n_features']})")
    
    # Store all results for comprehensive export
    all_results = {}
    
    # Demonstrate loss functions
    demonstrate_loss_functions()
    
    # Run experiments on different datasets
    for dataset in ['airfoil', 'yacht']:
        try:
            print(f"\n{'='*20} Processing {dataset.upper()} {'='*20}")
            
            # Single experiment
            experiment_results = run_single_experiment(dataset, exporter)
            
            # Benchmark comparison
            benchmark_results = run_comprehensive_benchmark(dataset, exporter)
            
            # Parameter sensitivity (run on both datasets for comprehensive analysis)
            # sensitivity_results = run_parameter_analysis(dataset, exporter)
            
            # Store all results for this dataset
            all_results[dataset] = {
                'dataset_info': dataset_info[dataset],
                'experiment_results': experiment_results,
                'benchmark_results': benchmark_results,
                # 'sensitivity_results': sensitivity_results
            }
            
        except Exception as e:
            print(f"Error in {dataset} experiment: {e}")
            all_results[dataset] = {'error': str(e)}
    
    # California housing (if time permits)
    try:
        print(f"\n{'='*20} Processing CALIFORNIA HOUSING {'='*20}")
        california_results = run_single_experiment('california', exporter)
        california_benchmark = run_comprehensive_benchmark('california', exporter)
        # california_sensitivity = run_parameter_analysis('california', exporter)
        all_results['california'] = {
            'dataset_info': dataset_info['california'],
            'experiment_results': california_results,
            'benchmark_results': california_benchmark,
            # 'sensitivity_results': california_sensitivity
        }
    except Exception as e:
        print(f"Error in California Housing experiment: {e}")
        all_results['california'] = {'error': str(e)}
    
    # Create comprehensive summary report
    print(f"\n{'='*20} Generating Summary Report {'='*20}")
    summary_path = exporter.create_summary_report(all_results)
    
    # Export complete results as JSON
    complete_results_file = f"complete_experiment_results_{exporter.timestamp}.json"
    exporter._export_json(all_results, complete_results_file)
    
    print("\n" + "="*60)
    print("CHH-Regression experiments completed!")
    print("📊 Exported data includes:")
    print("  • Individual experiment results (JSON, CSV, Pickle)")
    print("  • Benchmark comparisons (Excel, CSV, JSON)")
    print("  • Parameter sensitivity analysis (Excel, CSV)")
    print("  • Convergence data (CSV, JSON)")
    print("  • Complete results (JSON)")
    print(f"  • Summary report (Markdown): {summary_path}")
    print(f"\n📁 All results saved to: {exporter.output_dir}")
    print("🎨 Check the generated plots for detailed visual analysis.")
    
    return all_results


if __name__ == "__main__":
    main()
