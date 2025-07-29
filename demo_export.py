"""
Data Export Demo for CHH-Regression
Demonstrates how to export experimental data in various formats
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from data_export import ExperimentDataExporter
    from optimizer import CHHRegressor
    from evaluation import RegressionBenchmark
    from utils import generate_synthetic_data
    from data_loader import DataLoader
    print("✓ Successfully imported export modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def demo_single_experiment_export():
    """Demonstrate exporting single experiment results"""
    print("\n=== Single Experiment Export Demo ===")
    
    # Generate synthetic data
    X, y, true_coef = generate_synthetic_data(
        n_samples=200,
        n_features=3,
        outlier_fraction=0.1,
        random_state=42
    )
    
    # Fit CHH regressor
    chh = CHHRegressor(beta=1.0, max_iter=30)
    chh.fit(X, y)
    
    # Prepare results
    y_pred = chh.predict(X)
    experiment_results = {
        'dataset_info': {
            'name': 'synthetic_demo',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'true_coefficients': true_coef.tolist()
        },
        'model_parameters': {
            'beta': chh.beta,
            'delta': chh.delta,
            'sigma': chh.sigma
        },
        'training_results': {
            'n_iterations': chh.n_iter_,
            'final_loss': chh.loss_history_[-1],
            'loss_history': chh.loss_history_,
            'rmse': np.sqrt(np.mean((y - y_pred)**2)),
            'mae': np.mean(np.abs(y - y_pred)),
            'coefficients': chh.coef_.tolist(),
            'intercept': chh.intercept_
        },
        'predictions': {
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'residuals': (y - y_pred).tolist()
        }
    }
    
    # Export data
    exporter = ExperimentDataExporter(output_dir="demo_exports")
    exporter.export_single_experiment('synthetic_demo', experiment_results)
    
    print("✓ Single experiment export completed")


def demo_benchmark_export():
    """Demonstrate exporting benchmark results"""
    print("\n=== Benchmark Export Demo ===")
    
    # Generate data
    X, y, _ = generate_synthetic_data(
        n_samples=150,
        n_features=2,
        outlier_fraction=0.15,
        random_state=42
    )
    
    # Run benchmark
    benchmark = RegressionBenchmark(random_state=42)
    results = benchmark.run_benchmark(
        X, y,
        contamination_rates=[0.0, 0.1, 0.2]  # Reduced for demo
    )
    
    # Export benchmark results
    exporter = ExperimentDataExporter(output_dir="demo_exports")
    exporter.export_benchmark_results(results, 'benchmark_demo')
    
    print("✓ Benchmark export completed")


def demo_parameter_sensitivity_export():
    """Demonstrate exporting parameter sensitivity results"""
    print("\n=== Parameter Sensitivity Export Demo ===")
    
    # Generate smaller dataset for quick demo
    X, y, _ = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        random_state=42
    )
    
    # Simulate parameter sensitivity results
    from evaluation import parameter_sensitivity_analysis
    
    sensitivity_results = parameter_sensitivity_analysis(
        X, y,
        delta_range=[1.0, 1.345],
        sigma_range=[0.8, 1.0],
        beta_range=[0.5, 1.0],
        cv_folds=2  # Reduced for demo
    )
    
    # Export sensitivity analysis
    exporter = ExperimentDataExporter(output_dir="demo_exports")
    exporter.export_parameter_sensitivity(sensitivity_results, 'sensitivity_demo')
    
    print("✓ Parameter sensitivity export completed")


def demo_real_data_export():
    """Demonstrate exporting with real dataset"""
    print("\n=== Real Data Export Demo ===")
    
    try:
        # Load real dataset
        loader = DataLoader(standardize=True)
        X, y = loader.load_dataset('yacht')
        
        print(f"Loaded yacht dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Quick CHH experiment
        chh = CHHRegressor(beta=1.0, max_iter=20)
        chh.fit(X, y)
        
        y_pred = chh.predict(X)
        
        # Prepare results
        experiment_results = {
            'dataset_info': {
                'name': 'yacht_hydrodynamics',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'source': 'UCI Machine Learning Repository'
            },
            'model_parameters': {
                'beta': chh.beta,
                'delta': chh.delta,
                'sigma': chh.sigma
            },
            'training_results': {
                'n_iterations': chh.n_iter_,
                'final_loss': chh.loss_history_[-1],
                'loss_history': chh.loss_history_,
                'rmse': np.sqrt(np.mean((y - y_pred)**2)),
                'mae': np.mean(np.abs(y - y_pred)),
                'coefficients': chh.coef_.tolist(),
                'intercept': chh.intercept_
            }
        }
        
        # Export data
        exporter = ExperimentDataExporter(output_dir="demo_exports")
        exporter.export_single_experiment('yacht_real', experiment_results)
        
        print("✓ Real data export completed")
        
    except Exception as e:
        print(f"Real data export failed: {e}")


def demo_comprehensive_export():
    """Demonstrate comprehensive export with summary report"""
    print("\n=== Comprehensive Export Demo ===")
    
    # Simulate multiple experiment results
    all_results = {}
    
    # Synthetic experiment
    X1, y1, _ = generate_synthetic_data(n_samples=100, n_features=2, random_state=42)
    chh1 = CHHRegressor(beta=0.5, max_iter=20)
    chh1.fit(X1, y1)
    
    all_results['experiment_1'] = {
        'dataset_info': {'name': 'synthetic_1', 'n_samples': 100, 'n_features': 2},
        'training_results': {
            'rmse': np.sqrt(np.mean((y1 - chh1.predict(X1))**2)),
            'mae': np.mean(np.abs(y1 - chh1.predict(X1))),
            'n_iterations': chh1.n_iter_
        }
    }
    
    # Another synthetic experiment
    X2, y2, _ = generate_synthetic_data(n_samples=100, n_features=2, random_state=123)
    chh2 = CHHRegressor(beta=1.0, max_iter=20)
    chh2.fit(X2, y2)
    
    all_results['experiment_2'] = {
        'dataset_info': {'name': 'synthetic_2', 'n_samples': 100, 'n_features': 2},
        'training_results': {
            'rmse': np.sqrt(np.mean((y2 - chh2.predict(X2))**2)),
            'mae': np.mean(np.abs(y2 - chh2.predict(X2))),
            'n_iterations': chh2.n_iter_
        }
    }
    
    # Create comprehensive export
    exporter = ExperimentDataExporter(output_dir="demo_exports")
    
    # Export complete results
    exporter._export_json(all_results, "comprehensive_demo_results.json")
    
    # Create summary report
    summary_path = exporter.create_summary_report(all_results)
    
    print(f"✓ Comprehensive export completed")
    print(f"  Summary report: {summary_path}")


def main():
    """Run all export demos"""
    print("CHH-Regression Data Export Demo")
    print("=" * 40)
    
    try:
        demo_single_experiment_export()
        demo_benchmark_export()
        demo_parameter_sensitivity_export()
        demo_real_data_export()
        demo_comprehensive_export()
        
        print("\n" + "=" * 40)
        print("✓ All export demos completed!")
        print("\nExported files in 'demo_exports' directory:")
        print("• JSON files: Raw data in structured format")
        print("• CSV files: Tabular data for analysis")
        print("• Excel files: Multi-sheet workbooks for detailed review")
        print("• Pickle files: Python objects for later loading")
        print("• Markdown report: Human-readable summary")
        
        # List exported files
        from pathlib import Path
        export_dir = Path("demo_exports")
        if export_dir.exists():
            print(f"\nFiles created:")
            for file in export_dir.iterdir():
                print(f"  📄 {file.name}")
        
    except Exception as e:
        print(f"Error in export demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
