"""
Data Loading Module for CHH-Regression
Handles loading of UCI datasets and sklearn California Housing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
import warnings


def load_airfoil_data(data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UCI Airfoil Self-Noise dataset
    
    Args:
        data_path: Path to airfoil_self_noise.data file
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if data_path is None:
        # Default path relative to this file
        current_dir = Path(__file__).parent
        data_path = current_dir / "dataset" / "airfoil_self_noise.data"
    
    try:
        # Load data (tab-separated, no header)
        data = pd.read_csv(data_path, sep='\t', header=None)
        
        # Features: frequency, angle of attack, chord length, free-stream velocity, displacement thickness
        # Target: scaled sound pressure level
        X = data.iloc[:, :-1].values  # First 5 columns are features
        y = data.iloc[:, -1].values   # Last column is target
        
        return X, y
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Airfoil data file not found at {data_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading airfoil data: {e}")


def load_yacht_data(data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UCI Yacht Hydrodynamics dataset
    
    Args:
        data_path: Path to yacht_hydrodynamics.data file
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if data_path is None:
        # Default path relative to this file
        current_dir = Path(__file__).parent
        data_path = current_dir / "dataset" / "yacht_hydrodynamics.data"
    
    try:
        # Load data (space-separated, no header)
        data = pd.read_csv(data_path, sep='\s+', header=None)
        
        # Features: 6 yacht design parameters
        # Target: residuary resistance per unit weight
        X = data.iloc[:, :-1].values  # First 6 columns are features
        y = data.iloc[:, -1].values   # Last column is target
        
        return X, y
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Yacht data file not found at {data_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading yacht data: {e}")


def load_california_housing(sample_fraction: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load California Housing dataset from sklearn
    
    Args:
        sample_fraction: Fraction of data to sample (for computational efficiency)
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    try:
        from sklearn.datasets import fetch_california_housing
        
        # Load full dataset
        data = fetch_california_housing()
        X, y = data.data, data.target
        
        # Sample if requested
        if sample_fraction < 1.0:
            n_samples = int(len(X) * sample_fraction)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        return X, y
    
    except ImportError:
        raise ImportError("sklearn is required to load California Housing dataset")
    except Exception as e:
        raise RuntimeError(f"Error loading California Housing data: {e}")


def standardize_features(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Standardize features and target for better numerical stability
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Returns:
        Tuple of (X_scaled, y_scaled, scaler_info)
    """
    # Standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std == 0, 1, X_std)  # Avoid division by zero
    X_scaled = (X - X_mean) / X_std
    
    # Center target (but don't scale to preserve interpretability)
    y_mean = np.mean(y)
    y_scaled = y - y_mean
    
    scaler_info = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean
    }
    
    return X_scaled, y_scaled, scaler_info


def add_outliers(
    y: np.ndarray, 
    contamination_rate: float = 0.1, 
    outlier_magnitude: float = 8.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add artificial outliers to target variable for robustness testing
    
    Args:
        y: Original target vector
        contamination_rate: Fraction of samples to contaminate
        outlier_magnitude: Magnitude of outliers (in units of standard deviation)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (y_contaminated, outlier_mask)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y)
    n_outliers = int(n_samples * contamination_rate)
    
    # Select random samples to contaminate
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # Add outliers
    y_contaminated = y.copy()
    y_std = np.std(y)
    outlier_values = np.random.choice([-1, 1], n_outliers) * outlier_magnitude * y_std
    y_contaminated[outlier_indices] += outlier_values
    
    return y_contaminated, outlier_mask


def get_dataset_info() -> dict:
    """
    Get information about available datasets
    
    Returns:
        Dictionary with dataset information
    """
    return {
        'airfoil': {
            'name': 'UCI Airfoil Self-Noise',
            'n_samples': 1503,
            'n_features': 5,
            'description': 'NASA wind tunnel noise measurements'
        },
        'yacht': {
            'name': 'UCI Yacht Hydrodynamics', 
            'n_samples': 308,
            'n_features': 6,
            'description': 'Yacht hull resistance prediction'
        },
        'california': {
            'name': 'California Housing (sklearn)',
            'n_samples': 20640,
            'n_features': 8,
            'description': 'California housing prices'
        }
    }


class DataLoader:
    """Unified data loader for all datasets"""
    
    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.scaler_info = None
    
    def load_dataset(self, name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset by name
        
        Args:
            name: Dataset name ('airfoil', 'yacht', 'california')
            **kwargs: Additional arguments for specific loaders
        
        Returns:
            Tuple of (X, y)
        """
        if name == 'airfoil':
            X, y = load_airfoil_data(kwargs.get('data_path'))
        elif name == 'yacht':
            X, y = load_yacht_data(kwargs.get('data_path'))
        elif name == 'california':
            X, y = load_california_housing(kwargs.get('sample_fraction', 1.0))
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        if self.standardize:
            X, y, self.scaler_info = standardize_features(X, y)
        
        return X, y
