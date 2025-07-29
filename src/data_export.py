"""
Data Export Module for CHH-Regression
Handles exporting experimental results to various formats
"""

import json
import csv
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ExperimentDataExporter:
    """Export experimental results to various formats"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize exporter
        
        Args:
            output_dir: Directory to save exported data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_single_experiment(
        self,
        dataset_name: str,
        model_results: Dict[str, Any],
        format: str = "all"
    ) -> None:
        """
        Export single experiment results
        
        Args:
            dataset_name: Name of the dataset
            model_results: Dictionary containing model results
            format: Export format ('json', 'csv', 'pickle', 'all')
        """
        base_filename = f"{dataset_name}_experiment_{self.timestamp}"
        
        if format in ["json", "all"]:
            self._export_json(model_results, f"{base_filename}.json")
        
        if format in ["csv", "all"]:
            self._export_csv(model_results, f"{base_filename}.csv")
        
        if format in ["pickle", "all"]:
            self._export_pickle(model_results, f"{base_filename}.pkl")
    
    def export_benchmark_results(
        self,
        benchmark_results: Dict[str, Any],
        dataset_name: str,
        format: str = "all"
    ) -> None:
        """
        Export benchmark comparison results
        
        Args:
            benchmark_results: Results from benchmark comparison
            dataset_name: Name of the dataset
            format: Export format
        """
        base_filename = f"{dataset_name}_benchmark_{self.timestamp}"
        
        # Create structured data for export
        structured_data = self._structure_benchmark_data(benchmark_results)
        
        if format in ["json", "all"]:
            self._export_json(structured_data, f"{base_filename}.json")
        
        if format in ["csv", "all"]:
            self._export_benchmark_csv(structured_data, f"{base_filename}.csv")
        
        # if format in ["excel", "all"]:
        #     self._export_benchmark_excel(structured_data, f"{base_filename}.xlsx")
    
    def export_parameter_sensitivity(
        self,
        sensitivity_results: List[Dict[str, Any]],
        dataset_name: str,
        format: str = "all"
    ) -> None:
        """
        Export parameter sensitivity analysis results
        
        Args:
            sensitivity_results: List of parameter sensitivity results
            dataset_name: Name of the dataset
            format: Export format
        """
        base_filename = f"{dataset_name}_sensitivity_{self.timestamp}"
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(sensitivity_results)
        
        if format in ["csv", "all"]:
            df.to_csv(self.output_dir / f"{base_filename}.csv", index=False)
        
        # if format in ["excel", "all"]:
        #     df.to_excel(self.output_dir / f"{base_filename}.xlsx", index=False)
        
        if format in ["json", "all"]:
            self._export_json(sensitivity_results, f"{base_filename}.json")
    
    def export_convergence_data(
        self,
        convergence_data: Dict[str, Any],
        dataset_name: str,
        format: str = "all"
    ) -> None:
        """
        Export convergence analysis data
        
        Args:
            convergence_data: Convergence analysis results
            dataset_name: Name of the dataset
            format: Export format
        """
        base_filename = f"{dataset_name}_convergence_{self.timestamp}"
        
        if format in ["json", "all"]:
            self._export_json(convergence_data, f"{base_filename}.json")
        
        if format in ["csv", "all"]:
            # Export loss history as CSV
            loss_df = pd.DataFrame({
                'iteration': range(len(convergence_data['loss_history'])),
                'loss': convergence_data['loss_history']
            })
            loss_df.to_csv(self.output_dir / f"{base_filename}.csv", index=False)
    
    def _export_json(self, data: Any, filename: str) -> None:
        """Export data as JSON"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self._make_json_serializable(data)
        
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported JSON: {filename}")
    
    def _export_csv(self, data: Dict[str, Any], filename: str) -> None:
        """Export data as CSV"""
        # Flatten dictionary for CSV export
        flattened = self._flatten_dict(data)
        
        with open(self.output_dir / filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Key', 'Value'])
            for key, value in flattened.items():
                writer.writerow([key, value])
        
        print(f"✓ Exported CSV: {filename}")
    
    def _export_pickle(self, data: Any, filename: str) -> None:
        """Export data as pickle"""
        with open(self.output_dir / filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Exported Pickle: {filename}")
    
    def _export_benchmark_csv(self, data: Dict[str, Any], filename: str) -> None:
        """Export benchmark results as structured CSV"""
        # Create a comprehensive CSV with all metrics
        rows = []
        
        for method_name, method_data in data.items():
            if method_name == 'metadata':
                continue
                
            test_results = method_data.get('test_results', {})
            for contamination, metrics in test_results.items():
                row = {
                    'method': method_name,
                    'contamination_rate': contamination.replace('contamination_', ''),
                    **metrics
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / filename, index=False)
        print(f"✓ Exported benchmark CSV: {filename}")
    
    def _export_benchmark_excel(self, data: Dict[str, Any], filename: str) -> None:
        """Export benchmark results as Excel with multiple sheets"""
        with pd.ExcelWriter(self.output_dir / filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_rows = []
            for method_name, method_data in data.items():
                if method_name == 'metadata':
                    continue
                    
                test_results = method_data.get('test_results', {})
                for contamination, metrics in test_results.items():
                    row = {
                        'method': method_name,
                        'contamination_rate': contamination.replace('contamination_', ''),
                        **metrics
                    }
                    summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual method sheets
            for method_name, method_data in data.items():
                if method_name == 'metadata':
                    continue
                
                method_df = pd.DataFrame.from_dict(
                    method_data.get('test_results', {}), 
                    orient='index'
                )
                method_df.to_excel(writer, sheet_name=method_name[:31])  # Excel sheet name limit
        
        print(f"✓ Exported benchmark Excel: {filename}")
    
    def _structure_benchmark_data(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Structure benchmark data for export"""
        structured = {
            'metadata': {
                'timestamp': self.timestamp,
                'export_date': datetime.now().isoformat(),
                'total_methods': len(benchmark_results)
            }
        }
        
        for method_name, method_data in benchmark_results.items():
            structured[method_name] = {
                'test_results': method_data.get('test_results', {}),
                'cv_results_summary': self._summarize_cv_results(
                    method_data.get('cv_results', {})
                )
            }
        
        return structured
    
    def _summarize_cv_results(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize cross-validation results"""
        summary = {}
        
        for contamination, metrics in cv_results.items():
            summary[contamination] = {}
            for metric_name, values in metrics.items():
                if isinstance(values, list):
                    summary[contamination][f"{metric_name}_mean"] = np.mean(values)
                    summary[contamination][f"{metric_name}_std"] = np.std(values)
                else:
                    summary[contamination][metric_name] = values
        
        return summary
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        else:
            return data
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def create_summary_report(self, all_results: Dict[str, Any]) -> str:
        """Create a comprehensive summary report"""
        report_filename = f"experiment_summary_{self.timestamp}.md"
        
        with open(self.output_dir / report_filename, 'w', encoding='utf-8') as f:
            f.write("# CHH-Regression Experiment Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset information
            f.write("## Datasets Tested\n\n")
            for dataset_name, results in all_results.items():
                if 'dataset_info' in results:
                    info = results['dataset_info']
                    f.write(f"### {dataset_name.title()}\n")
                    f.write(f"- Samples: {info.get('n_samples', 'N/A')}\n")
                    f.write(f"- Features: {info.get('n_features', 'N/A')}\n")
                    f.write(f"- Description: {info.get('description', 'N/A')}\n\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            for dataset_name, results in all_results.items():
                if 'benchmark_results' in results:
                    f.write(f"### {dataset_name.title()} Dataset\n\n")
                    benchmark = results['benchmark_results']
                    
                    # Create performance table
                    f.write("| Method | Clean RMSE | 5% Contamination | 10% Contamination | 20% Contamination | 30% Contamination | 40% Contamination | 50% Contamination |\n")
                    f.write("|--------|------------|------------------|-------------------|-------------------|-------------------|-------------------|-------------------|\n")

                    for method_name, method_data in benchmark.items():
                        if method_name == 'metadata':
                            continue
                        test_results = method_data.get('test_results', {})
                        row = f"| {method_name} |"
                        for cont in ['contamination_0.00', 'contamination_0.05', 'contamination_0.10', 'contamination_0.20', 'contamination_0.30', 'contamination_0.40', 'contamination_0.50']:
                            if cont in test_results:
                                rmse = test_results[cont].get('rmse', 'N/A')
                                row += f" {rmse:.4f} |" if isinstance(rmse, (int, float)) else f" {rmse} |"
                            else:
                                row += " N/A |"
                        f.write(row + "\n")
                    f.write("\n")
        
        print(f"✓ Created summary report: {report_filename}")
        return str(self.output_dir / report_filename)
