"""
Maximum Correntropy Criterion (MCC) Regressor
Implementation based on sklearn style API with IRLS optimization
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from numpy.linalg import LinAlgError
import warnings

try:
    from .loss_functions import WelschLoss, estimate_scale
except ImportError:
    from loss_functions import WelschLoss, estimate_scale


@dataclass
class _IterRecord:
    """Record for iteration history"""
    obj: float
    mean_w: float
    min_w: float
    max_w: float
    rmse: float
    param_delta: float


def _solve_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float,
               fit_intercept: bool) -> Tuple[np.ndarray, float]:
    """
    Solve weighted ridge least squares.
    
    Args:
        X: Feature matrix
        y: Target vector
        w: Sample weights
        alpha: L2 regularization parameter
        fit_intercept: Whether to fit intercept
        
    Returns:
        Tuple of (coefficients, intercept)
    """
    # Apply sample weights sqrt to rows
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    if fit_intercept:
        Xw_aug = np.hstack([Xw, sw[:, None]])  # last column = weights for intercept
        p = X.shape[1]
        A = Xw_aug.T @ Xw_aug
        if alpha > 0:
            A[:p, :p] += alpha * np.eye(p)
        b = Xw_aug.T @ yw
        try:
            sol = np.linalg.solve(A, b)
        except LinAlgError:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
        coef = sol[:p]
        intercept = sol[p]
        return coef, float(intercept)
    else:
        A = Xw.T @ Xw
        if alpha > 0:
            A += alpha * np.eye(X.shape[1])
        b = Xw.T @ yw
        try:
            coef = np.linalg.solve(A, b)
        except LinAlgError:
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
        return coef, 0.0


class MCCRegressor:
    """
    Maximum Correntropy Criterion Linear Regressor (IRLS/MM algorithm)
    
    Minimizes correntropy-induced C-loss with Gaussian kernel:
        J(β) = Σ_i [1 − exp(− r_i(β)^2 / (2 σ^2))] + α ||β||_2^2,
    where r_i = y_i − x_i^T β − b.
    
    Uses IRLS iterations: at step t, compute weights w_i = exp(− r_i^2 / (2 σ^2))
    from residuals r^(t), then solve weighted ridge least squares.
    
    Parameters
    ----------
    sigma : float, default=1.0
        Gaussian kernel scale parameter σ (>0). Smaller σ provides stronger
        suppression of large residuals and improved robustness.
    alpha : float, default=0.0
        L2 regularization coefficient (does not penalize intercept).
    fit_intercept : bool, default=True
        Whether to fit intercept term.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for parameter updates (measured by L2 norm).
    obj_tol : float, default=0.0
        Tolerance for objective improvement; when >0, also checks |ΔJ| < obj_tol
        for convergence.
    init : {"ols", "zeros"}, default="ols"
        Parameter initialization method.
    sample_weight_mode : {"multiply", "normalize"}, default="multiply"
        How to combine with external sample_weight:
        - "multiply": directly multiply with MCC weights.
        - "normalize": normalize external weights to mean 1 then multiply,
          avoiding scale effects.
    verbose : int, default=0
        If >0, print iteration summary.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted coefficients.
    intercept_ : float
        Fitted intercept.
    n_iter_ : int
        Number of iterations performed.
    converged_ : bool
        Whether algorithm converged.
    history_ : List[dict]
        Per-iteration metrics: obj, mean_w, min_w, max_w, rmse, param_delta.
    """
    
    def __init__(self,
                 sigma: float = 1.0,
                 alpha: float = 0.0,
                 fit_intercept: bool = True,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 obj_tol: float = 0.0,
                 init: str = "ols",
                 sample_weight_mode: str = "multiply",
                 verbose: int = 0):
        self.sigma = sigma
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.obj_tol = obj_tol
        self.init = init
        self.sample_weight_mode = sample_weight_mode
        self.verbose = verbose
        
        # Initialize loss function
        self.loss_fn = WelschLoss(sigma=sigma)
    
    def _check_params(self):
        """Validate parameters"""
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.init not in ("ols", "zeros"):
            raise ValueError("init must be 'ols' or 'zeros'")
        if self.sample_weight_mode not in ("multiply", "normalize"):
            raise ValueError("sample_weight_mode must be 'multiply' or 'normalize'")
    
    def _initialize(self, X: np.ndarray, y: np.ndarray, 
                   sw: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        """Initialize parameters"""
        n_features = X.shape[1]
        if self.init == "zeros":
            coef = np.zeros(n_features, dtype=float)
            if sw is not None and self.fit_intercept:
                intercept = float(np.average(y, weights=sw))
            else:
                intercept = float(np.mean(y)) if self.fit_intercept else 0.0
            return coef, intercept
        
        # OLS/Ridge initialization
        w = np.ones_like(y) if sw is None else sw
        coef, intercept = _solve_wls(X, y, w, self.alpha, self.fit_intercept)
        return coef, intercept
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None) -> 'MCCRegressor':
        """
        Fit MCC regressor
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self._check_params()
        
        # Input validation
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        n_samples, n_features = X.shape
        
        # Handle sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight has incorrect length")
            if self.sample_weight_mode == "normalize":
                sample_weight = sample_weight * (n_samples / sample_weight.sum())
        else:
            sample_weight = np.ones(n_samples, dtype=float)
        
        # Update loss function with current sigma
        self.loss_fn = WelschLoss(sigma=self.sigma)
        
        # Initialize parameters
        coef, intercept = self._initialize(X, y, sample_weight)
        self.history_: List[Dict[str, float]] = []
        prev_obj = np.inf
        converged = False
        
        # IRLS iterations
        for t in range(1, self.max_iter + 1):
            # Compute residuals with current parameters
            r = y - (X @ coef + (intercept if self.fit_intercept else 0.0))
            
            # Compute MCC weights
            w_mcc = self.loss_fn.weights(r)
            w = w_mcc * sample_weight
            
            # Update parameters by weighted ridge LS
            new_coef, new_intercept = _solve_wls(X, y, w, self.alpha, self.fit_intercept)
            
            # Compute parameter change
            delta = np.linalg.norm(np.r_[
                new_coef - coef,
                (new_intercept - intercept) if self.fit_intercept else 0.0
            ])
            
            # Update parameters
            coef, intercept = new_coef, new_intercept
            
            # Compute diagnostics
            r = y - (X @ coef + (intercept if self.fit_intercept else 0.0))
            obj = float(self.loss_fn.loss(r).dot(sample_weight)) + \
                  self.alpha * float(np.dot(coef, coef))
            rmse = float(np.sqrt(np.average(r ** 2, weights=sample_weight)))
            
            # Store iteration record
            rec = _IterRecord(
                obj=obj,
                mean_w=float(np.average(w_mcc, weights=sample_weight)),
                min_w=float(w_mcc.min()),
                max_w=float(w_mcc.max()),
                rmse=rmse,
                param_delta=float(delta)
            )
            self.history_.append(rec.__dict__)
            
            if self.verbose:
                print(f"[MCCRegressor] iter={t:03d} obj={obj:.6f} rmse={rmse:.6f} "
                      f"mean_w={rec.mean_w:.4f} Δparam={delta:.3e}")
            
            # Check convergence
            if delta < self.tol:
                converged = True
                break
            if self.obj_tol > 0 and abs(prev_obj - obj) < self.obj_tol:
                converged = True
                break
            prev_obj = obj
        
        # Store results
        self.coef_ = coef
        self.intercept_ = float(intercept) if self.fit_intercept else 0.0
        self.n_iter_ = t
        self.converged_ = bool(converged)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using fitted model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, 'coef_'):
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Return coefficient of determination R^2
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
            
        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(X)
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            y_mean = np.average(y, weights=sample_weight)
            ss_tot = np.average((y - y_mean) ** 2, weights=sample_weight)
            ss_res = np.average((y - y_pred) ** 2, weights=sample_weight)
        else:
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def objective_path_(self) -> np.ndarray:
        """Return numpy array of objective values over iterations."""
        if not hasattr(self, 'history_'):
            raise ValueError("Model not fitted yet. Call fit() first.")
        return np.array([h["obj"] for h in self.history_], dtype=float)
    
    def weight_stats_path_(self) -> np.ndarray:
        """Return array with columns [mean_w, min_w, max_w] over iterations."""
        if not hasattr(self, 'history_'):
            raise ValueError("Model not fitted yet. Call fit() first.")
        return np.array([[h["mean_w"], h["min_w"], h["max_w"]] 
                        for h in self.history_], dtype=float)


def auto_mcc_regressor(X: np.ndarray, y: np.ndarray, 
                      init_method: str = 'ols') -> MCCRegressor:
    """
    Create MCC regressor with automatically estimated sigma parameter
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    init_method : str, default='ols'
        Method to estimate initial residuals ('ols' or 'mean').
        
    Returns
    -------
    regressor : MCCRegressor
        MCC regressor with auto-estimated sigma.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Get initial residuals for scale estimation
    if init_method == 'ols':
        try:
            # Simple OLS fit for initial residuals
            if X.shape[1] < X.shape[0]:  # Check for overdetermined system
                coef_init = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred_init = X @ coef_init
            else:
                y_pred_init = np.mean(y)
        except:
            y_pred_init = np.mean(y)
    else:
        y_pred_init = np.mean(y)
    
    initial_residuals = y - y_pred_init
    sigma_auto = estimate_scale(initial_residuals, method='mad')
    
    return MCCRegressor(sigma=sigma_auto)
