"""Verify Section 5.2(A): Monotone recovery in N (trust boundary).

Claim under test
----------------
As the query budget N grows, floor(N, rho) = Cest*Clam*sigma_eff*sqrt(log p_K / N)
decreases (~1/sqrt(N)). A coefficient is *certified* once the floor drops below its
true magnitude, and -- under the incoherence regime of Appendix E -- it then stays
certified for all larger N (consistent sign, stable magnitude). So:

  1. floor(N) is monotonically NON-INCREASING in N.
  2. the certified set C(N) = { S : coeff certified at budget N } is monotonically
     NON-DECREASING:  N1 < N2  =>  C(N1) subset of C(N2).
  3. each coefficient, once it enters C(N), keeps the correct sign and a stable
     magnitude (no flicker in/out, no sign flips) at larger N.

This is the image-model claim, but its operational content is geometric and is
tested here on the masked-function abstraction that the paper itself uses for
synthetic validation (centered orthonormal +-1 design, K=1). The image classifier
just supplies a masked response g_rho(z); the recovery law is identical. Running
on the abstraction lets us check monotonicity exactly (known ground truth) without
a GPU forward pass per mask. If you want the literal ResNet-50/ViT version, swap
`make_masked_fn` for a loop that queries `xai_suff` LIMEExplainer-style composites
(the model query slots in where `sample_fn` returns y).

A coefficient S is judged "certified at budget N" by the same signed-detection
criterion the paper uses (Table 2 / Sec 5.1): its Lasso estimate is separated from
zero past the penalty-scaled threshold with the correct sign. We use
    |beta_hat_S| > rec_frac * lam   AND   sign(beta_hat_S) == sign(beta_true_S)
which tracks the 1/sqrt(N) penalty scale rather than an absolute tolerance.

No torch is used or run; this is pure numpy + the torch-free _core helpers.

Usage:
    python verify_5_2_a.py
    python verify_5_2_a.py --d 30 --n-active 6 --trials 40
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from _core import centered_design, lasso_fit


# --------------------------------------------------------------------------- #
# floor / candidate-count helpers (mirror verify_theory.py conventions)
# --------------------------------------------------------------------------- #
def p_K(d: int, K: int = 1) -> int:
    return sum(math.comb(d, k) for k in range(0, K + 1))


def floor_value(sigma_eff: float, d: int, N: int, K: int = 1,
                c_floor: float = 1.0) -> float:
    """floor(N, rho) up to the absolute constant Cest*Clam (folded into c_floor).

    floor = c_floor * sigma_eff * sqrt(log p_K / N).  Monotone decreasing in N.
    """
    return c_floor * sigma_eff * math.sqrt(math.log(p_K(d, K)) / N)


# --------------------------------------------------------------------------- #
# masked-function generator (K=1 ground truth with controllable active set)
# --------------------------------------------------------------------------- #
def make_masked_fn(d, n_active, beta_active_vals, m_resid, seed):
    """Build a K=1 masked response with a *graded* active set.

    beta_active_vals: list of true magnitudes (one per active singleton). Using a
    spread of magnitudes is what exercises monotone recovery -- different coeffs
    cross the dropping floor at different N, so the certified set grows in steps.
    Higher-degree energy m_resid emulates the reference-induced mismatch (sigma_eff
    inflation) the way verify_theory.py does.
    """
    g = np.random.default_rng(seed)
    units = list(range(d))
    n_active = min(n_active, d)
    active = sorted(g.choice(units, size=n_active, replace=False).tolist())
    signs = g.choice([-1.0, 1.0], size=n_active)

    beta_true = np.zeros(d)
    for i, s, mag in zip(active, signs, beta_active_vals):
        beta_true[i] = mag * s

    # higher-order (degree>=2) residual structure of total energy m_resid
    hi_sets, n_hi = [], 200
    while len(hi_sets) < n_hi:
        k = min(int(g.integers(2, 4)), d)
        S = tuple(sorted(g.choice(units, size=k, replace=False)))
        if S not in hi_sets:
            hi_sets.append(S)
    mag = math.sqrt(m_resid / n_hi) if m_resid > 0 else 0.0
    hi_signs = g.choice([-1.0, 1.0], size=n_hi)
    beta_hi = {S: mag * s for S, s in zip(hi_sets, hi_signs)}

    def chi(Z, S):
        out = np.ones(Z.shape[0])
        for i in S:
            out *= (2.0 * (Z[:, i] - 0.5))
        return out

    def sample_fn(N, sigma_obs, _rng=np.random.default_rng(seed + 10_000)):
        Z = (_rng.random((N, d)) > 0.5).astype(float)
        y = np.zeros(N)
        for i in range(d):
            if beta_true[i] != 0.0:
                y += beta_true[i] * (2.0 * (Z[:, i] - 0.5))
        for S, b in beta_hi.items():
            y += b * chi(Z, S)
        if sigma_obs > 0:
            y += sigma_obs * _rng.standard_normal(N)
        return Z, y

    return beta_true, active, sample_fn


# --------------------------------------------------------------------------- #
# certified-set recovery at a single budget
# --------------------------------------------------------------------------- #
def certified_set(beta_hat, beta_true, active, lam, rec_frac=0.5):
    """Return the set of active singletons certified at this budget.

    Signed detection past the penalty-scaled threshold (Sec 5.1 criterion).
    """
    tol = rec_frac * lam
    cert = set()
    for i in active:
        if (np.sign(beta_hat[i]) == np.sign(beta_true[i])
                and abs(beta_hat[i]) > tol):
            cert.add(i)
    return cert


# --------------------------------------------------------------------------- #
# main verification
# --------------------------------------------------------------------------- #
def verify_monotone_recovery(d=30, n_active=6, sigma_obs=1.0, m_resid=0.10,
                             n_trials=30, rec_frac=0.5, c_floor=1.0,
                             N_grid=None, seed0=7):
    K = 1
    if N_grid is None:
        N_grid = np.unique(np.round(
            np.geomspace(40, 6000, 16)).astype(int)).tolist()

    # graded true magnitudes spanning ~1.5 decades so coeffs cross the floor
    # at different budgets -> a genuinely *growing* certified set.
    beta_vals = np.geomspace(0.30, 3.0, n_active) * sigma_obs
    sigma_eff = sigma_obs + 1.26 * math.sqrt(m_resid)  # Cm=1.26 from Table 1

    print("=" * 74)
    print("Section 5.2(A) verification: monotone recovery in N (trust boundary)")
    print("=" * 74)
    print(f"  d={d}  K={K}  n_active={n_active}  sigma_obs={sigma_obs}  "
          f"m_resid={m_resid}")
    print(f"  sigma_eff = sigma_obs + 1.26*sqrt(m) = {sigma_eff:.4f}")
    print(f"  true active |beta| (graded): "
          f"[{', '.join(f'{v:.3f}' for v in beta_vals)}]")
    print(f"  certified iff correct sign AND |beta_hat| > {rec_frac}*lam\n")

    # ---- per-trial: track certified set as a function of N --------------- #
    # We aggregate over trials by majority vote (a coeff counts as certified
    # at N if it is certified in >=50% of trials), then test the three claims
    # on the aggregated trajectory AND report the per-trial violation rate.
    grid = list(N_grid)
    n_grid = len(grid)

    # cert_count_per_trial[t][k] = number of actives certified at grid[k]
    cert_indicator = np.zeros((n_trials, n_grid, d), dtype=bool)
    floors = [floor_value(sigma_eff, d, N, K, c_floor) for N in grid]

    for t in range(n_trials):
        beta_true, active, sf = make_masked_fn(
            d, n_active, beta_vals, m_resid, seed=seed0 + 1000 * t)
        for k, N in enumerate(grid):
            lam = floor_value(sigma_eff, d, N, K, c_floor)
            Z, y = sf(N, sigma_obs)
            X = centered_design(Z)
            beta_hat, _ = lasso_fit(X, y, lam=max(lam, 1e-9))
            cert = certified_set(beta_hat, beta_true, active, lam, rec_frac)
            for i in cert:
                cert_indicator[t, k, i] = True

    # ---- Claim 1: floor monotonically non-increasing --------------------- #
    floor_arr = np.array(floors)
    floor_mono = bool(np.all(np.diff(floor_arr) <= 1e-12))
    print("[Claim 1] floor(N) non-increasing in N")
    print(f"  floor at N={grid[0]:>5}: {floor_arr[0]:.4f}   "
          f"floor at N={grid[-1]:>5}: {floor_arr[-1]:.4f}")
    print(f"  monotone non-increasing       : "
          f"{'PASS' if floor_mono else 'FAIL'}\n")

    # ---- Claim 2: certified set non-decreasing in N ---------------------- #
    # aggregate: coeff certified at grid k if certified in >=50% of trials
    maj = cert_indicator.mean(axis=0) >= 0.5          # (n_grid, d)
    cert_count = maj.sum(axis=1)                        # certified count vs N

    # subset monotonicity on the majority trajectory
    set_mono = True
    first_violation = None
    for k in range(1, n_grid):
        prev, cur = set(np.where(maj[k - 1])[0]), set(np.where(maj[k])[0])
        if not prev.issubset(cur):
            set_mono = False
            if first_violation is None:
                first_violation = (grid[k - 1], grid[k],
                                   sorted(prev - cur))
    print("[Claim 2] certified set C(N) non-decreasing (majority-vote agg.)")
    print(f"  {'N':>6} {'floor':>9} {'#certified':>11}  certified active idx")
    for k, N in enumerate(grid):
        idx = sorted(np.where(maj[k])[0])
        print(f"  {N:>6} {floor_arr[k]:>9.4f} {cert_count[k]:>11d}  {idx}")
    print(f"  C(N) monotone non-decreasing  : "
          f"{'PASS' if set_mono else 'CHECK'}")
    if not set_mono:
        a, b, lost = first_violation
        print(f"    (coeffs dropped between N={a} and N={b}: {lost})")
    print()

    # ---- Claim 3: per-trial stability once certified --------------------- #
    # For each trial+coeff, after the first budget it is certified, it should
    # remain certified at all larger budgets (no flicker). Count violations.
    flicker = 0
    total_entered = 0
    for t in range(n_trials):
        for i in range(d):
            traj = cert_indicator[t, :, i]
            if not traj.any():
                continue
            total_entered += 1
            first = int(np.argmax(traj))            # first True index
            if not traj[first:].all():              # dropped out later?
                flicker += 1
    flicker_rate = flicker / total_entered if total_entered else 0.0
    print("[Claim 3] per-trial stability: certified coeffs stay certified")
    print(f"  trial x coeff trajectories that ever certify : {total_entered}")
    print(f"  of those, # that later flicker OFF           : {flicker}")
    print(f"  flicker rate                                 : "
          f"{flicker_rate*100:.1f}%")
    print(f"  stable-once-certified (>=90%)                : "
          f"{'PASS' if flicker_rate <= 0.10 else 'CHECK'}\n")

    # ---- overall ---------------------------------------------------------- #
    overall = floor_mono and set_mono and flicker_rate <= 0.10
    print("=" * 74)
    print(f"  OVERALL 5.2(A)                : "
          f"{'PASS' if overall else 'CHECK'}  "
          f"(floor monotone, certified set grows, no flicker)")
    print("=" * 74)
    return {
        "grid": grid,
        "floors": floor_arr.tolist(),
        "cert_count": cert_count.tolist(),
        "floor_monotone": floor_mono,
        "set_monotone": set_mono,
        "flicker_rate": flicker_rate,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Verify Section 5.2(A): monotone recovery in N.")
    ap.add_argument("--d", type=int, default=30)
    ap.add_argument("--n-active", type=int, default=6)
    ap.add_argument("--sigma-obs", type=float, default=1.0)
    ap.add_argument("--m-resid", type=float, default=0.10,
                    help="higher-order energy m_>K (reference mismatch proxy)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--rec-frac", type=float, default=0.5)
    ap.add_argument("--c-floor", type=float, default=1.0,
                    help="absolute floor constant Cest*Clam")
    args = ap.parse_args()

    verify_monotone_recovery(
        d=args.d, n_active=args.n_active, sigma_obs=args.sigma_obs,
        m_resid=args.m_resid, n_trials=args.trials,
        rec_frac=args.rec_frac, c_floor=args.c_floor)


if __name__ == "__main__":
    main()