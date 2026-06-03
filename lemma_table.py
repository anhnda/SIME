"""
lemma1_table.py -- populate Table 1 (Proposition 1 / Lemma 1 leakage check).

eta_{rho,N} = ||(1/N) X^T r||_inf  for a PURE higher-degree residual r of
controlled energy m (no degree-<=K active part), should track
sqrt(m log p_K / N) with a constant ratio C_m across m and N.

Uses _core.empirical_leakage_batch (vectorized over B independent draws).
No torch. K=1, d=30.
"""
from __future__ import annotations
import math
import numpy as np
from _core import empirical_leakage_batch

RNG = np.random.default_rng(0)
d, K = 30, 1
p_K = sum(math.comb(d, k) for k in range(0, K + 1))   # = d+1 = 31
log_pK = math.log(p_K)
B = 400          # independent draws per (m,N) cell -> average eta
n_hi = 200       # number of degree>=2 residual terms carrying energy m


def make_residual_terms(seed):
    """Fixed set of degree-2/3 subsets and +-1 signs; magnitude set per-m."""
    g = np.random.default_rng(seed)
    units = list(range(d))
    sets = []
    while len(sets) < n_hi:
        k = int(g.integers(2, 4))
        S = tuple(sorted(g.choice(units, size=k, replace=False)))
        if S not in sets:
            sets.append(S)
    signs = g.choice([-1.0, 1.0], size=n_hi)
    return sets, signs


def chi_batch(Z, S):
    # Z: (B,N,d) -> (B,N) character for subset S
    out = np.ones(Z.shape[:2])
    for i in S:
        out *= (2.0 * (Z[:, :, i] - 0.5))
    return out


def residual_response(Z, sets, signs, m):
    """Pure higher-degree residual of total energy m: r(z)=sum mag*sign*chi_S."""
    mag = math.sqrt(m / n_hi)
    Y = np.zeros(Z.shape[:2])
    for S, s in zip(sets, signs):
        Y += mag * s * chi_batch(Z, S)
    return Y


def main():
    sets, signs = make_residual_terms(seed=12345)
    m_vals = [0.05, 0.20, 0.80]            # 16x range
    N_vals = [500, 1000, 2000]             # 4x range
    rows = []
    for m in m_vals:
        for N in N_vals:
            Z = (RNG.random((B, N, d)) > 0.5).astype(float)
            Y = residual_response(Z, sets, signs, m)   # pure residual, no active part
            etas = empirical_leakage_batch(Z, Y)        # (B,)
            eta_emp = float(etas.mean())
            pred = math.sqrt(m * log_pK / N)
            rows.append((m, N, eta_emp, pred, eta_emp / pred))

    print(f"{'m':>6} {'N':>6} {'eta_emp':>10} {'sqrt(m logpK/N)':>16} {'ratio':>8}")
    for m, N, e, p, r in rows:
        print(f"{m:>6.2f} {N:>6d} {e:>10.5f} {p:>16.5f} {r:>8.4f}")
    ratios = np.array([r[4] for r in rows])
    print(f"\nC_m = ratio: mean {ratios.mean():.4f}, "
          f"std {ratios.std():.4f}, CoV {ratios.std()/ratios.mean():.4f}")
    print(f"constant across 16x m, 4x N : "
          f"{'PASS' if ratios.std()/ratios.mean() < 0.10 else 'CHECK'} "
          f"(target CoV<0.10)")

    # LaTeX-ready rows
    print("\n--- LaTeX table body ---")
    for m, N, e, p, r in rows:
        print(f"{m:.2f} & {N} & {e:.4f} & {p:.4f} & {r:.3f} \\\\")


if __name__ == "__main__":
    main()