"""
Torch-free core for the reference-aware K=1 estimator (Theorem 1).

Provides:
  * centered_design          : {0,1} masks -> centered orthonormal +-1 design
  * lasso_fit                : Lasso surrogate fit. Uses sklearn if available
                               (C-backed, fast), else a numpy coordinate-descent
                               fallback. Returns (beta, intercept).
  * empirical_leakage_batch  : VECTORIZED ||(1/N) X^T r||_inf for Lemma 1 checks.

Vectorization notes
-------------------
- Coordinate descent over the d coordinates is inherently SEQUENTIAL (update j
  depends on the residual updated at j-1), so it cannot be vectorized *within*
  one fit. Two real speedups instead:
    (a) use sklearn.linear_model.Lasso (compiled coordinate descent) -- default;
    (b) vectorize across the many INDEPENDENT fits (trials/beta/m). The leakage
        diagnostic below is fully vectorized with no Python inner loop, and the
        synthetic script batches trials.
"""
from __future__ import annotations
import numpy as np

try:
    from sklearn.linear_model import Lasso as _SkLasso
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


def centered_design(Z: np.ndarray) -> np.ndarray:
    """Z in {0,1} (N,d) -> centered orthonormal columns chi_i = 2(z_i-1/2) in {-1,+1}."""
    return 2.0 * (Z - 0.5)


# --------------------------------------------------------------------------- #
#  Lasso fit.  Convention: objective = (1/2N)||y - X b||^2 + lam ||b||_1.
#  With +-1 columns (||col||^2 = N) the coefficient soft-threshold equals lam,
#  so to make the active set coincide with a detection `floor`, pass lam = floor.
# --------------------------------------------------------------------------- #
def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float,
              n_iter: int = 50000, tol: float = 1e-6):
    if _HAVE_SK:
        m = _SkLasso(alpha=max(lam, 1e-9), fit_intercept=True,
                     max_iter=n_iter, tol=tol)
        m.fit(X, y)
        return m.coef_.copy(), float(m.intercept_)
    return _lasso_cd_numpy(X, y, lam, n_iter, tol)


def _lasso_cd_numpy(X, y, lam, n_iter=1000, tol=1e-8):
    """No-dependency fallback. Sequential CD (cannot be vectorized over coords)."""
    N, d = X.shape
    y_mean = y.mean()
    r = y - y_mean
    beta = np.zeros(d)
    col_sq = (X ** 2).sum(axis=0) + 1e-12   # ~= N for +-1 columns
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


# --------------------------------------------------------------------------- #
#  VECTORIZED leakage:  eta = ||(1/N) X^T r||_inf  for B independent draws.
#  Z: (B,N,d) in {0,1}.  Y: (B,N).  -> (B,) etas, no Python loop over B.
# --------------------------------------------------------------------------- #
def empirical_leakage_batch(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = 2.0 * (Z - 0.5)                        # (B,N,d)
    N = X.shape[1]
    XtY = np.einsum("bnd,bn->bd", X, Y) / N    # (B,d)
    return np.max(np.abs(XtY), axis=1)