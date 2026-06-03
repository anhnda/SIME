"""Verify Section 5.2(A): Monotone recovery in N (trust boundary) -- image model.

This is the *real* 5.2 setup, not the synthetic 5.1 abstraction. We treat an
ImageNet classifier (ResNet-50 by default) as a query-only black box, partition
the image into a d = gh*gw grid of cells, mask off-cells with a blur reference
(on-manifold, matching the LIMEExplainer in xai_suff), and recover the K=1
reference-conditioned coefficients with a column-standardized Lasso.

There is NO ground-truth beta in this setting. The "certified set" is defined
purely operationally, exactly as the paper states the claim:

    C(N) = { cell j : |beta_hat_j(N)| > floor(N, rho) }, with stable sign.

The three things 5.2(A) asserts, and what we check:

  1. floor(N, rho) = Cest * lambda_rho is monotonically NON-INCREASING in N.
     (lambda_rho = Clam * sigma_eff * sqrt(log p_K / N); sigma_eff measured from
      a pilot via Eq.(10), NOT assumed.)
  2. The certified set C(N) is monotonically NON-DECREASING:
        N1 < N2  =>  C(N1) subset of C(N2).
     Each cell appears once the dropping floor passes below |beta_hat_j| and
     remains thereafter.
  3. Per-cell stability: once a cell enters C(N) it keeps a consistent SIGN and
     a stable magnitude (no flicker out, no sign flips) at larger N.

sigma_eff is split correctly into its two roles (the bug in the earlier draft):
  - lambda_rho  = Clam * sigma_eff * sqrt(log p_K / N)   -> the Lasso penalty (fit)
  - floor(N)    = Cest * lambda_rho                       -> the detection boundary
These differ by Cest and are separate knobs; certification is judged against the
FLOOR, the fit uses the PENALTY.

No torch is run automatically. Execute yourself:

    python verify_5_2_a.py --image /path/to/img.jpg
    python verify_5_2_a.py --image img.jpg --grid 12 12 --sigma 11 \
                           --N-grid 200 400 800 1600 3200 6400 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from xai_suff.backbone import (
    get_class_names,
    load_backbone,
    load_image,
)
from xai_suff.explainers import blur_reference


# --------------------------------------------------------------------------- #
# candidate count / penalty / floor
# --------------------------------------------------------------------------- #
def p_K(d: int, K: int = 1) -> int:
    return sum(math.comb(d, k) for k in range(0, K + 1))


def lambda_value(sigma_eff: float, d: int, N: int, K: int = 1,
                 c_lambda: float = 0.30) -> float:
    """Lasso penalty lambda_rho = Clam * sigma_eff * sqrt(log p_K / N), Eq.(6)."""
    return c_lambda * sigma_eff * math.sqrt(math.log(p_K(d, K)) / N)


def floor_value(sigma_eff: float, d: int, N: int, K: int = 1,
                c_lambda: float = 0.30, c_est: float = 1.0) -> float:
    """Detection floor(N, rho) = Cest * lambda_rho, Eq.(8). Monotone in N."""
    return c_est * lambda_value(sigma_eff, d, N, K, c_lambda)


# --------------------------------------------------------------------------- #
# cell-id map (mirrors xai_suff LIMEExplainer._cell_id_map)
# --------------------------------------------------------------------------- #
def cell_id_map(H, W, grid):
    gh, gw = grid
    ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
    xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
    return ys.view(-1, 1) * gw + xs.view(1, -1)  # (H,W) in [0, gh*gw)


# --------------------------------------------------------------------------- #
# masked-response oracle: query the model on blur-composite masks
# --------------------------------------------------------------------------- #
@torch.no_grad()
def query_masked_response(model, x, b, cell_ids, Z, target, device,
                          batch_size=64):
    """g_rho(z) for each mask row in Z (N, n_cells): target-class probability.

    keep cell -> sharp pixels; off cell -> blur reference (on-manifold).
    """
    N = Z.shape[0]
    _, C, H, W = x.shape
    y = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        zb = torch.as_tensor(
            Z[start:start + batch_size], dtype=torch.float32, device=device)
        keep = zb[:, cell_ids].unsqueeze(1)             # (B,1,H,W)
        comp = keep * x + (1.0 - keep) * b              # (B,C,H,W)
        p = F.softmax(model(comp), dim=1)[:, target]
        y[start:start + zb.shape[0]] = p.detach().cpu().numpy()
    return y


# --------------------------------------------------------------------------- #
# standardized-design K=1 Lasso (centered +-1 design, unit-norm columns)
# --------------------------------------------------------------------------- #
def centered_design(Z):
    return 2.0 * (Z - 0.5)


def standardized_lasso_fit(Z, y, lam, n_iter=20000, tol=1e-7):
    """K=1 Lasso on the centered +-1 design; coefficients in the chi_S basis.

    The centered +-1 design already has equal column norms (||X_j|| = sqrt(N)
    exactly, since entries are +-1), so column standardization is a no-op up to
    the global sqrt(N) factor. That factor belongs in the penalty (sklearn's
    objective is 1/(2N)||y-Xb||^2 + alpha||b||_1, which already carries the 1/N),
    NOT divided out of beta. Dividing beta by col_norm ~ sqrt(N) was the bug that
    crushed every coefficient by ~2 orders of magnitude. We fit directly and
    return coefficients on the same natural scale as floor(N, rho).
    """
    X = centered_design(Z)
    try:
        from sklearn.linear_model import Lasso
        m = Lasso(alpha=max(lam, 1e-9), fit_intercept=True,
                  max_iter=n_iter, tol=tol)
        m.fit(X, y)
        return m.coef_.copy()
    except Exception:
        return _lasso_cd(X, y, lam, n_iter, tol)


def _lasso_cd(X, y, lam, n_iter=20000, tol=1e-7):
    N, d = X.shape
    r = y - y.mean()
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
    return beta


# --------------------------------------------------------------------------- #
# pilot: measure sigma_eff via Eq.(10) (held-out unexplained variance)
# --------------------------------------------------------------------------- #
def measure_sigma_eff(model, x, b, cell_ids, n_cells, target, device,
                      d, K, n_pilot, c_lambda, sigma_obs=0.0, seed=0,
                      batch_size=64):
    """sigma_eff = sigma_obs + Cm*sqrt(m_>K), with m_>K from the held-out
    residual of a degree-K fit on fresh masks (Eq.10). sigma_obs is 0 for a
    deterministic model+reference; pass >0 only for a stochastic reference.

    Cm folded in implicitly: we report sigma_eff directly as
        sqrt( max(resid_var - sigma_obs^2, 0) )  + sigma_obs
    i.e. the measured perturbation-noise scale that drives the floor. This is the
    conservative, upper-biased proxy the paper describes.
    """
    g = np.random.default_rng(seed)
    # fit split
    Zf = (g.random((n_pilot, n_cells)) > 0.5).astype(np.float64)
    Zf[0] = 1.0
    yf = query_masked_response(model, x, b, cell_ids, Zf, target, device,
                               batch_size)
    lam = lambda_value(max(sigma_obs, 1e-3), d, n_pilot, K, c_lambda)
    beta = standardized_lasso_fit(Zf, yf, lam)
    intercept = yf.mean() - centered_design(Zf) @ beta
    b0 = float(np.mean(intercept))
    # held-out split
    Zv = (g.random((n_pilot, n_cells)) > 0.5).astype(np.float64)
    yv = query_masked_response(model, x, b, cell_ids, Zv, target, device,
                               batch_size)
    pred = centered_design(Zv) @ beta + b0
    resid_var = float(np.mean((yv - pred) ** 2))
    m_hat = max(resid_var - sigma_obs ** 2, 0.0)
    sigma_eff = sigma_obs + math.sqrt(m_hat)
    return sigma_eff, {"resid_var": resid_var, "m_hat": m_hat, "b0": b0}


# --------------------------------------------------------------------------- #
# main verification over the N-grid
# --------------------------------------------------------------------------- #
def verify(model, x, b, cell_ids, n_cells, target, class_name, device,
           N_grid, K=1, c_lambda=0.30, c_est=1.0, n_pilot=512,
           sigma_obs=0.0, mag_eps=0.20, seed=0, batch_size=64, out=None):
    d = n_cells

    print("=" * 76)
    print("Section 5.2(A): monotone recovery in N (trust boundary) -- image model")
    print("=" * 76)
    print(f"  target class = {target} ({class_name})")
    print(f"  grid cells d = {d}   K = {K}   p_K = {p_K(d, K)}")
    print(f"  N grid       = {N_grid}")

    # ---- pilot: sigma_eff (measured, not assumed) ------------------------ #
    sigma_eff, pinfo = measure_sigma_eff(
        model, x, b, cell_ids, n_cells, target, device, d, K, n_pilot,
        c_lambda, sigma_obs=sigma_obs, seed=seed, batch_size=batch_size)
    print(f"  pilot N0     = {n_pilot}   resid_var = {pinfo['resid_var']:.5f}   "
          f"m_hat = {pinfo['m_hat']:.5f}")
    print(f"  sigma_eff    = {sigma_eff:.5f}  (Eq.10 pilot; sigma_obs={sigma_obs})")
    print(f"  certified iff |beta_hat_j| > floor(N) with stable sign\n")

    # ---- fit at each budget --------------------------------------------- #
    grid = list(N_grid)
    n_grid = len(grid)
    floors = [floor_value(sigma_eff, d, N, K, c_lambda, c_est) for N in grid]
    beta_hist = np.zeros((n_grid, d))
    cert_indicator = np.zeros((n_grid, d), dtype=bool)
    rng = np.random.default_rng(seed + 999)

    for k, N in enumerate(grid):
        Z = (rng.random((N, n_cells)) > 0.5).astype(np.float64)
        Z[0] = 1.0
        y = query_masked_response(model, x, b, cell_ids, Z, target, device,
                                  batch_size)
        lam = lambda_value(sigma_eff, d, N, K, c_lambda)
        beta_hat = standardized_lasso_fit(Z, y, lam)
        beta_hist[k] = beta_hat
        floor = floors[k]
        cert_indicator[k] = np.abs(beta_hat) > floor

    # diagnostic: coefficient magnitudes vs floor at the largest budget.
    # A healthy run has max|beta| well above the floor; if max|beta| < floor at
    # every N, certification is vacuous and the "PASS" below is meaningless.
    max_abs_final = float(np.max(np.abs(beta_hist[-1])))
    print(f"  diag: max|beta_hat| at N={grid[-1]} = {max_abs_final:.5f}  "
          f"vs floor = {floors[-1]:.5f}  "
          f"(ratio {max_abs_final / (floors[-1] + 1e-12):.1f}x)")
    if max_abs_final < floors[-1]:
        print("  WARNING: no coefficient clears the floor at ANY budget -- the "
              "certified set is empty and the PASS results are VACUOUS. Check "
              "the fit scale / sigma_eff / c_lambda before trusting this.\n")
    else:
        print()

    # ---- Claim 1: floor non-increasing ----------------------------------- #
    floor_arr = np.array(floors)
    floor_mono = bool(np.all(np.diff(floor_arr) <= 1e-12))
    print("[Claim 1] floor(N) non-increasing in N")
    print(f"  floor at N={grid[0]:>5}: {floor_arr[0]:.5f}   "
          f"floor at N={grid[-1]:>5}: {floor_arr[-1]:.5f}")
    print(f"  monotone non-increasing       : "
          f"{'PASS' if floor_mono else 'FAIL'}\n")

    # ---- Claim 2: certified set non-decreasing --------------------------- #
    set_mono = True
    first_violation = None
    for k in range(1, n_grid):
        prev = set(np.where(cert_indicator[k - 1])[0])
        cur = set(np.where(cert_indicator[k])[0])
        if not prev.issubset(cur):
            set_mono = False
            if first_violation is None:
                first_violation = (grid[k - 1], grid[k], sorted(prev - cur))
    print("[Claim 2] certified set C(N) non-decreasing")
    print(f"  {'N':>6} {'floor':>9} {'#cert':>6}  certified cell idx")
    for k, N in enumerate(grid):
        idx = sorted(np.where(cert_indicator[k])[0])
        show = idx if len(idx) <= 14 else idx[:14] + ["..."]
        print(f"  {N:>6} {floor_arr[k]:>9.5f} {int(cert_indicator[k].sum()):>6}"
              f"  {show}")
    print(f"  C(N) monotone non-decreasing  : "
          f"{'PASS' if set_mono else 'CHECK'}")
    if not set_mono:
        a, bb, lost = first_violation
        print(f"    (cells dropped between N={a} and N={bb}: {lost})")
    print()

    # ---- Claim 3: per-cell sign + magnitude stability once certified ----- #
    sign_flips = 0
    mag_unstable = 0
    flicker = 0
    n_entered = 0
    for j in range(d):
        traj = cert_indicator[:, j]
        if not traj.any():
            continue
        n_entered += 1
        first = int(np.argmax(traj))
        tail = slice(first, n_grid)
        # flicker: ever drops back out after first certification
        if not traj[tail].all():
            flicker += 1
        # sign consistency across the certified tail
        signs = np.sign(beta_hist[tail, j][cert_indicator[tail, j]])
        if signs.size and not np.all(signs == signs[0]):
            sign_flips += 1
        # magnitude stability: relative spread across certified tail
        mags = np.abs(beta_hist[tail, j][cert_indicator[tail, j]])
        if mags.size >= 2:
            spread = (mags.max() - mags.min()) / (mags.mean() + 1e-12)
            if spread > 1.0:                      # >100% swing = unstable
                mag_unstable += 1
    print("[Claim 3] per-cell stability once certified")
    print(f"  cells that ever certify        : {n_entered}")
    print(f"  flicker OFF after entering      : {flicker}")
    print(f"  sign flips in certified tail    : {sign_flips}")
    print(f"  magnitude-unstable (>100% swing): {mag_unstable}")
    flicker_rate = flicker / n_entered if n_entered else 0.0
    sign_ok = sign_flips == 0
    print(f"  no flicker (>=90%)             : "
          f"{'PASS' if flicker_rate <= 0.10 else 'CHECK'} "
          f"(rate {flicker_rate*100:.1f}%)")
    print(f"  no sign flips                  : "
          f"{'PASS' if sign_ok else 'CHECK'}\n")

    non_vacuous = max_abs_final >= floors[-1] and n_entered > 0
    overall = (floor_mono and set_mono and flicker_rate <= 0.10
               and sign_ok and non_vacuous)
    print("=" * 76)
    status = "PASS" if overall else ("VACUOUS" if not non_vacuous else "CHECK")
    print(f"  OVERALL 5.2(A)                : {status}")
    print("=" * 76)

    if out:
        os.makedirs(out, exist_ok=True)
        rec = {
            "target": target, "class_name": class_name,
            "d": d, "K": K, "sigma_eff": sigma_eff,
            "N_grid": grid, "floors": floor_arr.tolist(),
            "cert_count": [int(c.sum()) for c in cert_indicator],
            "floor_monotone": floor_mono, "set_monotone": set_mono,
            "flicker_rate": flicker_rate, "sign_flips": sign_flips,
        }
        with open(os.path.join(out, "verify_5_2_a.json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  -> {os.path.join(out, 'verify_5_2_a.json')}")
    return overall


def main():
    ap = argparse.ArgumentParser(
        description="Verify Section 5.2(A): monotone recovery in N (image model).")
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--target", type=int, default=None,
                    help="target class (default: model top-1)")
    ap.add_argument("--grid", type=int, nargs=2, default=(12, 12),
                    metavar=("GH", "GW"))
    ap.add_argument("--sigma", type=float, default=11.0,
                    help="blur reference bandwidth")
    ap.add_argument("--N-grid", type=int, nargs="+",
                    default=[200, 400, 800, 1600, 3200, 6400])
    ap.add_argument("--K", type=int, default=1, choices=[1])
    ap.add_argument("--c-lambda", type=float, default=0.30,
                    help="penalty constant Clam")
    ap.add_argument("--c-est", type=float, default=1.0,
                    help="floor constant Cest (floor = Cest*lambda)")
    ap.add_argument("--n-pilot", type=int, default=512)
    ap.add_argument("--sigma-obs", type=float, default=0.0,
                    help="query-noise scale (0 for deterministic model+ref)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)
    _, _, H, W = x.shape

    with torch.no_grad():
        top1 = int(model(x).argmax(1).item())
    target = args.target if args.target is not None else top1

    cell_ids = cell_id_map(H, W, tuple(args.grid)).to(device)
    n_cells = args.grid[0] * args.grid[1]
    b = blur_reference(x, args.sigma).to(device)

    verify(model, x, b, cell_ids, n_cells, target, class_names[target], device,
           N_grid=args.N_grid, K=args.K, c_lambda=args.c_lambda,
           c_est=args.c_est, n_pilot=args.n_pilot, sigma_obs=args.sigma_obs,
           seed=args.seed, batch_size=args.batch_size, out=args.out)


if __name__ == "__main__":
    main()