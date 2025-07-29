"""
CHH Loss Functions Module
Implements the Combined Huber-Correntropy Hybrid (CHH) loss function
"""

import numpy as np
from typing import Tuple, Optional


class CHHLoss:
    """
    Combined Huber-Correntropy Hybrid Loss Function
    
    Combines the robustness of Huber loss in small residuals with
    the rapid re-descending properties of correntropy for large residuals.
    """
    
    def __init__(self, delta: float = 1.345, sigma: float = 1.0, beta: float = 1.0):
        """
        Initialize CHH loss parameters
        
        Args:
            delta: Huber threshold (typically 1.345 * scale for 95% efficiency)
            sigma: Correntropy kernel width
            beta: Weight of correntropy component (beta=0 reduces to Huber)
        """
        self.delta = delta
        self.sigma = sigma
        self.beta = beta
    
    def huber_loss(self, r: np.ndarray) -> np.ndarray:
        """Compute Huber loss component"""
        abs_r = np.abs(r)
        return np.where(
            abs_r <= self.delta,
            0.5 * r**2,
            self.delta * abs_r - 0.5 * self.delta**2
        )
    
    def correntropy_loss(self, r: np.ndarray) -> np.ndarray:
        """Compute correntropy/Welsch loss component"""
        return 1 - np.exp(-r**2 / (2 * self.sigma**2))
    
    def loss(self, r: np.ndarray) -> np.ndarray:
        """Compute CHH loss: rho_CHH(r) = H_delta(r) + beta * C(r)"""
        return self.huber_loss(r) + self.beta * self.correntropy_loss(r)
    
    def psi(self, r: np.ndarray) -> np.ndarray:
        """
        Compute influence function (derivative of loss)
        psi(r) = psi_Huber(r) + beta * psi_correntropy(r)
        """
        # Huber psi function
        psi_huber = np.where(
            np.abs(r) <= self.delta,
            r,
            self.delta * np.sign(r)
        )
        
        # Correntropy psi function
        psi_corr = (r / self.sigma**2) * np.exp(-r**2 / (2 * self.sigma**2))
        
        return psi_huber + self.beta * psi_corr
    
    def weights(self, r: np.ndarray) -> np.ndarray:
        """
        Compute IRLS weights for CHH loss
        w = w_Huber + w_correntropy
        """
        # Avoid division by zero
        r_safe = np.where(np.abs(r) < 1e-12, 1e-12, r)
        
        # Huber weights
        w_huber = np.where(
            np.abs(r) <= self.delta,
            1.0,
            self.delta / np.abs(r_safe)
        )
        
        # Correntropy weights
        w_corr = (self.beta / (2 * self.sigma**2)) * np.exp(-r**2 / (2 * self.sigma**2))
        
        return w_huber + w_corr


def estimate_scale(residuals: np.ndarray, method: str = 'mad') -> float:
    """
    Estimate scale parameter from residuals
    
    Args:
        residuals: Array of residuals
        method: Scale estimation method ('mad' or 'std')
    
    Returns:
        Estimated scale parameter
    """
    if method == 'mad':
        # Median Absolute Deviation with consistency factor for normal distribution
        return 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    elif method == 'std':
        return np.std(residuals)
    else:
        raise ValueError(f"Unknown scale estimation method: {method}")


class WelschLoss:
    """
    Welsch/MCC (Maximum Correntropy Criterion) Loss Function
    
    Implements correntropy-induced loss: rho(r) = 1 - exp(-r^2 / (2*sigma^2))
    This is equivalent to Welsch M-estimator and MCC objective.
    """
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize Welsch/MCC loss parameters
        
        Args:
            sigma: Correntropy kernel width (scale parameter)
        """
        self.sigma = sigma
    
    def loss(self, r: np.ndarray) -> np.ndarray:
        """Compute Welsch/MCC loss"""
        return 1 - np.exp(-r**2 / (2 * self.sigma**2))
    
    def psi(self, r: np.ndarray) -> np.ndarray:
        """
        Compute influence function (derivative of loss)
        psi(r) = (r/sigma^2) * exp(-r^2 / (2*sigma^2))
        """
        return (r / self.sigma**2) * np.exp(-r**2 / (2 * self.sigma**2))
    
    def weights(self, r: np.ndarray) -> np.ndarray:
        """
        Compute IRLS weights for Welsch/MCC loss
        w(r) = exp(-r^2 / (2*sigma^2))
        """
        return np.exp(-r**2 / (2 * self.sigma**2))


def get_default_parameters(initial_residuals: np.ndarray) -> Tuple[float, float, float]:
    """
    Get default CHH parameters based on initial residuals
    
    Args:
        initial_residuals: Initial residuals from OLS or simple fit
    
    Returns:
        Tuple of (delta, sigma, beta) parameters
    """
    scale = estimate_scale(initial_residuals, 'mad')
    delta = 1.345 * scale  # 95% efficiency under normality
    sigma = scale  # Use estimated scale as kernel width
    beta = 1.0  # Balanced weight between Huber and correntropy
    
    return delta, sigma, beta
