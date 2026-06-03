"""
Torch-free core for the reference-aware K=1 estimator (Theorem 1).
"""
from __future__ import annotations
import numpy as np

try:
    from sklearn.linear_model import Lasso as _SkLasso
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


def centered_design(Z: np.ndarray) -> np.ndarray:
    return 2.0 * (Z - 0.5)


def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float,
              n_iter: int = 50000, tol: float = 1e-6):
    if _HAVE_SK:
        m = _SkLasso(alpha=max(lam, 1e-9), fit_intercept=True,
                     max_iter=n_iter, tol=tol)
        m.fit(X, y)
        return m.coef_.copy(), float(m.intercept_)
    return _lasso_cd_numpy(X, y, lam, n_iter, tol)


def _lasso_cd_numpy(X, y, lam, n_iter=1000, tol=1e-8):
    N, d = X.shape
    y_mean = y.mean()
    r = y - y_mean
    beta = np.zeros(d)
    col_sq = (X ** 2).sum(axis=0) + 1e-12
    for _ in range(n_iter):
        max_delta = 0.0
        for j in range(d):
            xj = X[:, j]
            rho_j = xj @ r + beta[j] * col_sq[j]
            z = rho_j / col_sq[j]
            thr = lam * N / col_sq[j]
            new = np.sign(z) * max(abs(z) - thr, 0.0)
            if new != beta[j]:
                r += (beta[j] - new) * xj
                max_delta = max(max_delta, abs(new - beta[j]))
                beta[j] = new
        if max_delta < tol:
            break
    return beta, y_mean


def empirical_leakage_batch(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = 2.0 * (Z - 0.5)
    N = X.shape[1]
    XtY = np.einsum("bnd,bn->bd", X, Y) / N
    return np.max(np.abs(XtY), axis=1)