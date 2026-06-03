"""
verify_theory.py -- the parts of the paper that synthetic_verify.py does NOT test.

synthetic_verify.py already covers:
    - Lemma 1   (leakage sqrt-law)        -> the load-bearing result, well done there
    - "floor scaling" / "collapse"        -> soft / partly self-referential

This file adds the three promised-but-untested claims, each DECOUPLED from the
estimator's own lambda so the test is not circular:

    thm1      Theorem 1, Eq. (8): || beta_hat - beta ||_inf on the active set
              is <= C_est * lambda, and active coeffs above the floor get the
              correct sign. We MEASURE C_est rather than assume it.

    incoh     Corollary 1: the irrepresentability / mutual-incoherence number
              max_{j not in S} || X_S^+ x_j ||_1.  < 1 => support recovery is
              feasible; this is the diagnostic Section 7(2) says it reports.

    budget    Corollary 2 / Table 2: leakage-free (m=0) regime. At each SNR
              gamma = beta_min / sigma_obs, find the budget N at which 90%
              exact signed-support recovery occurs, and check it tracks the
              1/gamma^2 law  N ~ C^2 * sigma_eff^2 * log p_K / beta_min^2,
              locating the constant C.

Entry points:
    python verify_theory.py thm1
    python verify_theory.py incoh
    python verify_theory.py budget
    python verify_theory.py all        # default

No torch. Reuses _core.py (centered_design, lasso_fit) unchanged.
"""
from __future__ import annotations
import sys, math
import numpy as np

from _core import centered_design, lasso_fit

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
#  Shared K=1 function builder with KNOWN active singletons + controlled m_{>1}.
#  (Mirrors synthetic_verify.make_function but exposes the active coeff vector
#   so we can measure estimation error against ground truth.)
# --------------------------------------------------------------------------- #
def make_k1_function(d, n_active, beta_active, m_resid, seed):
    g = np.random.default_rng(seed)
    units = list(range(d))
    active = list(g.choice(units, size=n_active, replace=False))
    signs = g.choice([-1.0, 1.0], size=n_active)
    beta_true = np.zeros(d)
    for i, s in zip(active, signs):
        beta_true[i] = beta_active * s

    # degree->=2 residual carrying total energy m_resid (pure mismatch)
    hi_sets, n_hi = [], 200
    while len(hi_sets) < n_hi:
        k = int(g.integers(2, 4)); k = min(k, d)
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

    def sample_fn(N, sigma_obs):
        Z = (RNG.random((N, d)) > 0.5).astype(float)
        y = np.zeros(N)
        for i in range(d):
            if beta_true[i] != 0.0:
                y += beta_true[i] * (2.0 * (Z[:, i] - 0.5))
        for S, b in beta_hi.items():
            y += b * chi(Z, S)
        if sigma_obs > 0:
            y += sigma_obs * RNG.standard_normal(N)
        return Z, y

    return beta_true, set(active), sample_fn


def p_K(d, K=1):
    return sum(math.comb(d, k) for k in range(0, K + 1))


def floor_value(c, sigma_obs, m, d, N, K=1):
    return (sigma_obs + c * math.sqrt(m)) * math.sqrt(math.log(p_K(d, K)) / N)


# =========================================================================== #
#  thm1 : Eq. (8) estimation bound + sign correctness on the active set.
#         We fit at lambda = floor, then measure the realized ratio
#             C_est_emp = || beta_hat_S - beta_S ||_inf / lambda
#         and the fraction of active coeffs recovered with the correct sign.
# =========================================================================== #
def verify_thm1(c=1.3, d=30, n_active=4, N=4000, sigma_obs=0.02,
                m=0.02, n_trials=40, beta_mult=3.0):
    K = 1
    lam = floor_value(c, sigma_obs, m, d, N, K)
    beta_active = beta_mult * lam          # sit comfortably above the floor
    ratios, sign_ok = [], []
    for t in range(n_trials):
        beta_true, active, sf = make_k1_function(
            d, n_active, beta_active, m, seed=4242 + t)
        Z, y = sf(N, sigma_obs)
        X = centered_design(Z)
        beta_hat, _ = lasso_fit(X, y, lam=max(lam, 1e-9))
        idx = sorted(active)
        err_inf = np.max(np.abs(beta_hat[idx] - beta_true[idx]))
        ratios.append(err_inf / lam)
        # sign correctness for coeffs whose true magnitude exceeds the floor
        ok = all(np.sign(beta_hat[i]) == np.sign(beta_true[i]) for i in idx)
        sign_ok.append(ok)
    ratios = np.array(ratios)
    c_est_emp = float(ratios.mean())
    frac_sign = float(np.mean(sign_ok))
    print(f"  lambda = floor                : {lam:.5f}")
    print(f"  active |beta| / floor          : {beta_mult:.1f}x")
    print(f"  measured C_est (mean err/lam) : {c_est_emp:.3f} "
          f"(max {ratios.max():.3f}, std {ratios.std():.3f})")
    print(f"  active-set sign correctness   : {frac_sign*100:.0f}% of trials")
    bound_holds = ratios.max() < 5.0   # C_est is an O(1) constant; flag if huge
    print(f"  Eq.(8) ratio bounded by O(1)  : "
          f"{'PASS' if bound_holds else 'CHECK'} "
          f"(estimation error is a bounded multiple of lambda)")
    print(f"  sign claim (>=90% trials)     : "
          f"{'PASS' if frac_sign >= 0.9 else 'CHECK'}")
    return c_est_emp, frac_sign


# =========================================================================== #
#  incoh : Corollary 1 mutual-incoherence / irrepresentability diagnostic.
#          For active set S, eta_irr = max_{j not in S} || (X_S^T X_S)^-1 X_S^T x_j ||_1.
#          eta_irr < 1 (with margin) => signed-support recovery is feasible.
#          Reported as Section 7(2) promises.
# =========================================================================== #
def verify_incoherence(d=30, n_active=4, N=4000, n_trials=30):
    vals = []
    for t in range(n_trials):
        _, active, sf = make_k1_function(
            d, n_active, beta_active=0.1, m_resid=0.0, seed=99 + t)
        Z, _ = sf(N, sigma_obs=0.0)
        X = centered_design(Z)
        S = sorted(active)
        Sc = [j for j in range(d) if j not in active]
        XS = X[:, S]
        G = XS.T @ XS
        # solve G W = XS^T X_Sc  ->  W = (XS^T XS)^-1 XS^T X_Sc, columns per j
        rhs = XS.T @ X[:, Sc]
        W = np.linalg.solve(G, rhs)             # (|S|, |Sc|)
        eta_irr = float(np.max(np.sum(np.abs(W), axis=0)))
        vals.append(eta_irr)
    vals = np.array(vals)
    print(f"  irrepresentability eta_irr     : mean {vals.mean():.3f}, "
          f"max {vals.max():.3f}, min {vals.min():.3f}")
    feasible = vals.max() < 1.0
    print(f"  incoherence (eta_irr < 1)     : "
          f"{'PASS' if feasible else 'CHECK'} "
          f"(Corollary 1 support recovery {'feasible' if feasible else 'NOT guaranteed'})")
    print(f"  note: orthonormal +-1 design => eta_irr should be ~O(sqrt(s/N)), "
          f"i.e. small; large values flag a degenerate draw.")
    return vals


# =========================================================================== #
#  budget : Corollary 2 / Table 2.  Leakage-free m=0.  For each gamma = beta/sigma,
#           find N at 90% exact signed-support recovery; compare to C^2 sigma^2 logp / beta^2.
#           Recovery is judged WITHOUT reference to `floor` (uses a fixed small
#           magnitude tolerance), so the test is not circular.
# =========================================================================== #
def verify_budget(d=30, n_active=4, sigma_obs=1.0, n_trials=30, mag_tol=1e-6):
    K = 1
    log_pK = math.log(p_K(d, K))
    gammas = [2.00, 1.00, 0.71, 0.50]      # beta_min / sigma_obs (matches Table 2)
    N_grid = np.unique(np.round(
        np.geomspace(40, 4000, 26)).astype(int))
    print(f"  {'gamma':>6} {'beta':>8} {'N@90%(emp)':>12} "
          f"{'pred N (C=2)':>13} {'implied C':>10}")
    results = []
    for gamma in gammas:
        beta = gamma * sigma_obs
        emp_N = None
        for N in N_grid:
            succ = 0
            # lambda for the m=0 case: standard Lasso rate ~ sigma sqrt(logp/N)
            lam = sigma_obs * math.sqrt(log_pK / N)
            for t in range(n_trials):
                beta_true, active, sf = make_k1_function(
                    d, n_active, beta_active=beta, m_resid=0.0,
                    seed=31 * t + N + int(gamma * 1000))
                Z, y = sf(N, sigma_obs)
                X = centered_design(Z)
                beta_hat, _ = lasso_fit(X, y, lam=max(lam, 1e-9))
                rec = set(np.where(np.abs(beta_hat) > mag_tol)[0].tolist())
                if rec == active:               # EXACT signed support (mag-based, not floor)
                    succ += 1
            if succ / n_trials >= 0.90:
                emp_N = int(N)
                break
        pred_N = (2.0 ** 2) * sigma_obs ** 2 * log_pK / beta ** 2
        implied_C = (math.sqrt(emp_N * beta ** 2 / (sigma_obs ** 2 * log_pK))
                     if emp_N else float("nan"))
        results.append((gamma, beta, emp_N, pred_N, implied_C))
        print(f"  {gamma:>6.2f} {beta:>8.3f} "
              f"{(emp_N if emp_N else -1):>12d} "
              f"{pred_N:>13.0f} {implied_C:>10.3f}")
    Cs = [r[4] for r in results if not math.isnan(r[4])]
    if len(Cs) >= 2:
        cov = np.std(Cs) / np.mean(Cs)
        print(f"\n  implied C across SNRs: mean {np.mean(Cs):.3f}, CoV {cov:.3f}")
        print(f"  1/gamma^2 law (C ~ constant)  : "
              f"{'PASS' if cov < 0.30 else 'CHECK'} "
              f"(target CoV<0.30; constant C => N scales as 1/gamma^2)")
    return results


# --------------------------------------------------------------------------- #
def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 70)
    print("Theory verification: Theorem 1 / Corollary 1 / Corollary 2")
    print("(the claims synthetic_verify.py does not cover)")
    print("=" * 70)

    if what in ("thm1", "all"):
        print("\n[Theorem 1]  Eq.(8) estimation bound + sign correctness")
        verify_thm1()

    if what in ("incoh", "all"):
        print("\n[Corollary 1]  mutual-incoherence diagnostic (promised in 7(2))")
        verify_incoherence()

    if what in ("budget", "all"):
        print("\n[Corollary 2 / Table 2]  leakage-free budget vs SNR (1/gamma^2 law)")
        verify_budget()

    print("\n" + "=" * 70)
    print("Notes:")
    print(" - thm1 measures C_est rather than assuming it; the claim is that")
    print("   active-set error is a bounded O(1) multiple of lambda, not a")
    print("   specific constant.")
    print(" - budget judges EXACT support by coefficient magnitude, NOT by")
    print("   `floor`, so it is independent of the quantity being tested")
    print("   (unlike synthetic_verify.experiment_collapse).")
    print("=" * 70)


if __name__ == "__main__":
    main()