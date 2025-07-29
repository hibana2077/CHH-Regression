# MCC

```py
from __future__ import annotations
import numpy as np
from numpy.linalg import LinAlgError
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils import check_array


@dataclass
class _IterRecord:
    obj: float
    mean_w: float
    min_w: float
    max_w: float
    rmse: float
    param_delta: float


def _correntropy_loss(residuals: np.ndarray, sigma: float) -> np.ndarray:
    """Correntropy‑induced loss (C‑loss) for MCC.
    ρ(r) = 1 − exp(− r^2 / (2 σ^2)).
    Returns per‑sample losses.
    """
    s2 = 2.0 * (sigma ** 2)
    return 1.0 - np.exp(-(residuals ** 2) / s2)


def _mcc_weights(residuals: np.ndarray, sigma: float) -> np.ndarray:
    """IRLS weights for MCC.
    w(r) = exp(− r^2 / (2 σ^2)).
    (起自 ψ(r)= (r/σ^2) exp(− r^2/(2σ^2)) ，權重 = ψ(r)/r = exp(− r^2/(2σ^2))).
    """
    s2 = 2.0 * (sigma ** 2)
    return np.exp(-(residuals ** 2) / s2)


def _solve_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float,
               fit_intercept: bool) -> Tuple[np.ndarray, float]:
    """Solve weighted ridge least squares.

    If fit_intercept, we augment X with ones and set zero penalty on intercept.
    Returns (coef, intercept).
    """
    # Apply sample weights sqrt to rows
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    if fit_intercept:
        Xw_aug = np.hstack([Xw, sw[:, None]])  # last column = weights for intercept (ones * sqrt(w))
        # Build (p+1) x (p+1) system with zero penalty for intercept
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


class MCCRegressor(BaseEstimator, RegressorMixin):
    """Maximum Correntropy Criterion 線性回歸器（IRLS / MM）。

    以高斯核 correntropy 誘導之 C‑loss 最小化：
        J(β) = Σ_i [1 − exp(− r_i(β)^2 / (2 σ^2))] + α ||β||_2^2,
    其中 r_i = y_i − x_i^T β − b。

    透過 IRLS 反覆：在第 t 次以 r^(t) 計算權重 w_i = exp(− r_i^2 / (2 σ^2))，
    解帶樣本權重的加權（帶 L2 正則）最小平方法。

    參數
    ------
    sigma : float, default=1.0
        高斯核尺度參數 σ (>0)。σ 較小會更強烈抑制大殘差、提升穩健性。
    alpha : float, default=0.0
        L2 正則化係數（不懲罰截距）。
    fit_intercept : bool, default=True
        是否擬合截距。
    max_iter : int, default=100
        最大迭代次數。
    tol : float, default=1e-6
        參數更新的容忍度（以 L2 範數量測）。
    obj_tol : float, default=0.0
        目標值改善的容忍度；>0 時亦會檢查 |ΔJ| < obj_tol 觸發收斂。
    init : {"ols", "zeros"}, default="ols"
        參數初始化方式。
    sample_weight_mode : {"multiply", "normalize"}, default="multiply"
        與外部 sample_weight 的結合方式：
        - "multiply": 直接與 MCC 權重相乘。
        - "normalize": 將外部權重正規化到平均 1 後再相乘，避免尺度影響。
    verbose : int, default=0
        >0 時輸出每次迭代摘要。

    屬性
    ------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    n_iter_ : int
    converged_ : bool
    history_ : List[dict]
        每回合的指標：obj, mean_w, min_w, max_w, rmse, param_delta。
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

    def _more_tags(self):
        return {"requires_y": True}

    def _check_params(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.init not in ("ols", "zeros"):
            raise ValueError("init must be 'ols' or 'zeros'")
        if self.sample_weight_mode not in ("multiply", "normalize"):
            raise ValueError("sample_weight_mode must be 'multiply' or 'normalize'")

    def _initialize(self, X: np.ndarray, y: np.ndarray, sw: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        n_features = X.shape[1]
        if self.init == "zeros":
            coef = np.zeros(n_features, dtype=float)
            intercept = float(np.average(y, weights=sw) if (sw is not None and self.fit_intercept) else 0.0)
            return coef, intercept
        # OLS/Ridge 初始化（若 alpha>0 則 ridge）
        w = np.ones_like(y) if sw is None else sw
        coef, intercept = _solve_wls(X, y, w, self.alpha, self.fit_intercept)
        return coef, intercept

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        self._check_params()
        X, y = check_X_y(X, y, accept_sparse=False, dtype=float, y_numeric=True)
        n_samples, n_features = X.shape
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight has incorrect length")
            if self.sample_weight_mode == "normalize":
                sample_weight = sample_weight * (n_samples / sample_weight.sum())
        else:
            sample_weight = np.ones(n_samples, dtype=float)

        coef, intercept = self._initialize(X, y, sample_weight)
        self.history_: List[Dict[str, float]] = []
        prev_obj = np.inf
        converged = False

        for t in range(1, self.max_iter + 1):
            # residuals with current params
            r = y - (X @ coef + (intercept if self.fit_intercept else 0.0))
            w_mcc = _mcc_weights(r, self.sigma)
            w = w_mcc * sample_weight

            # update by weighted ridge LS
            new_coef, new_intercept = _solve_wls(X, y, w, self.alpha, self.fit_intercept)

            # diagnostics
            delta = np.linalg.norm(np.r_[new_coef - coef, (new_intercept - intercept) if self.fit_intercept else 0.0])
            coef, intercept = new_coef, new_intercept
            r = y - (X @ coef + (intercept if self.fit_intercept else 0.0))
            obj = float(_correntropy_loss(r, self.sigma).dot(sample_weight)) + self.alpha * float(np.dot(coef, coef))
            rmse = float(np.sqrt(np.average(r ** 2, weights=sample_weight)))
            rec = _IterRecord(obj=obj,
                              mean_w=float(np.average(w_mcc, weights=sample_weight)),
                              min_w=float(w_mcc.min()),
                              max_w=float(w_mcc.max()),
                              rmse=rmse,
                              param_delta=float(delta))
            self.history_.append(rec.__dict__)

            if self.verbose:
                print(f"[MCCRegressor] iter={t:03d} obj={obj:.6f} rmse={rmse:.6f} "
                      f"mean_w={rec.mean_w:.4f} Δparam={delta:.3e}")

            # convergence tests
            if delta < self.tol:
                converged = True
                break
            if self.obj_tol > 0 and abs(prev_obj - obj) < self.obj_tol:
                converged = True
                break
            prev_obj = obj

        self.coef_ = coef
        self.intercept_ = float(intercept) if self.fit_intercept else 0.0
        self.n_iter_ = t
        self.converged_ = bool(converged)
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = check_array(X, accept_sparse=False, dtype=float)
        return X @ self.coef_ + self.intercept_

    def objective_path_(self) -> np.ndarray:
        """Return numpy array of objective values over iterations."""
        check_is_fitted(self, attributes=["history_"])
        return np.array([h["obj"] for h in self.history_], dtype=float)

    def weight_stats_path_(self) -> np.ndarray:
        """Return array with columns [mean_w, min_w, max_w] over iterations."""
        check_is_fitted(self, attributes=["history_"])
        return np.array([[h["mean_w"], h["min_w"], h["max_w"]] for h in self.history_], dtype=float)


# 使用示例（測試時可取消註解）
# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     n, p = 200, 5
#     X = rng.standard_normal((n, p))
#     beta_true = rng.standard_normal(p)
#     y = X @ beta_true + 0.5 * rng.standard_normal(n)
#     # 注入脈衝雜訊
#     idx = rng.choice(n, size=int(0.2 * n), replace=False)
#     y[idx] += rng.normal(0, 20, size=idx.size)
#     model = MCCRegressor(sigma=1.0, alpha=0.0, max_iter=100, tol=1e-6, verbose=1)
#     model.fit(X, y)
#     print("Converged:", model.converged_, "iters:", model.n_iter_)
#     print("R2:", model.score(X, y))
```