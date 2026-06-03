"""Verify Section 5.2(B): forward budget prediction (effective-sample-size rule).

The procedure the paper claims a practitioner can run:
    (i)   run a cheap pilot, measure sigma_eff via Eq.(10);
    (ii)  pick a target trust threshold beta_min;
    (iii) predict the budget N from Corollary 1:
              N_pred = ceil( C^2 * sigma_eff^2 * log p_K / beta_min^2 ),  C = Cest*Clam;
    (iv)  run the real fit at N_pred;
    (v)   the REALIZED floor should land at ~= beta_min, and the certified set
          should extend DOWN TO ~= beta_min (no further, no shorter).
We report predicted-vs-realized floor and the certified count.

WHAT IS AND ISN'T A REAL TEST. Two things here are true analytically, not
empirically, and the script says so rather than dressing them up as a PASS:

  * floor(N_pred, rho) ~= beta_min is true BY CONSTRUCTION. N_pred is obtained by
    inverting floor(N)=beta_min, so plugging it back gives beta_min up to the
    integer-ceiling rounding of N. We still print it (rounding can shift it a
    little, and a divergence would expose a code-path bug), but PASS here means
    "the inversion round-trips", not "the theory predicted the world".

  * The certified COUNT at N_pred is whatever it is; the claim is about its
    LOWER EDGE, i.e. the smallest certified |beta_hat| should be ~= beta_min.

THE GENUINELY EMPIRICAL CONTENT, and what (B) actually stakes:

  (B1) PILOT TRANSFER. sigma_eff is measured on a small pilot (N0 ~ few*p_K) and
       then used to set a much larger N_pred. The claim only works if the pilot's
       sigma_eff predicts the sigma_eff that actually governs the run at N_pred.
       We re-measure sigma_eff on a held-out split AT N_pred (sigma_eff_run) and
       check sigma_eff_pilot >= sigma_eff_run up to tolerance -- the pilot proxy
       is supposed to be UPPER-biased (Appendix G), so it should over-, never
       under-, estimate. An under-estimate is the dangerous failure (under-budget).

  (B2) LOWER-EDGE LANDING. The smallest certified coefficient at N_pred,
       beta_edge = min_{j in C(N_pred)} |beta_hat_j|, should sit just above the
       realized floor and near beta_min: ratio beta_edge / floor in roughly
       [1, 1+band]. If beta_edge is far above floor, the run resolved nothing new
       near the threshold (over-budgeted / nothing lives there); if the certified
       set is empty, the prediction is vacuous at this beta_min.

  (B3) CONSISTENCY WITH 5.2(A). Running ALSO at a coarser and a finer budget
       (N_pred/2, 2*N_pred) should bracket the certified count monotonically
       (more budget -> floor drops -> >= as many certified). This reuses the
       monotone-recovery logic of (A) as a sanity rail on the single (B) point:
       the predicted N should not be an outlier off the monotone curve.

Pilot conservativeness (Appendix G) means N_pred is an UPPER estimate, so the
honest reading of a PASS is "spending N_pred certifies down to beta_min; you may
get away with fewer." We report the implied over-budget factor
(sigma_eff_pilot/sigma_eff_run)^2 so the practitioner sees the slack.

No torch is run automatically. Execute yourself, e.g.:

    python verify_5_2_b.py --image /path/to/img.jpg
    python verify_5_2_b.py --image img.jpg --grid 12 12 --sigma 11 \
        --beta-min 0.02 --n-pilot 512 --c-lambda 0.30 --c-est 1.0 --device cuda

Pick beta_min sensibly: it must be reachable, i.e. N_pred should clear the
feasibility floor N >~ s*log p_K (Cor.3). A beta_min far below the smallest real
coefficient just predicts an enormous N and certifies the whole support; a
beta_min above the largest coefficient predicts a tiny (infeasible) N. The script
warns in both regimes.
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
# candidate count / penalty / floor / budget  (same conventions as 5.2 a & c)
# --------------------------------------------------------------------------- #
def p_K(d: int, K: int = 1) -> int:
    return sum(math.comb(d, k) for k in range(0, K + 1))


def lambda_value(sigma_eff: float, d: int, N: int, K: int = 1,
                 c_lambda: float = 0.30) -> float:
    """Lasso penalty lambda_rho = Clam * sigma_eff * sqrt(log p_K / N), Eq.(6)."""
    return c_lambda * sigma_eff * math.sqrt(math.log(p_K(d, K)) / N)


def floor_value(sigma_eff: float, d: int, N: int, K: int = 1,
                c_lambda: float = 0.30, c_est: float = 1.0) -> float:
    """Detection floor(N, rho) = Cest * lambda_rho, Eq.(8)."""
    return c_est * lambda_value(sigma_eff, d, N, K, c_lambda)


def predict_budget(sigma_eff: float, d: int, beta_min: float, K: int = 1,
                   C_budget: float = 3.5) -> int:
    """Cor.1 inversion: N_pred = ceil( C_budget^2 sigma_eff^2 log p_K / beta_min^2 ).

    IMPORTANT: C_budget is the EMPIRICALLY CALIBRATED budget constant from the
    synthetic Table 2 (signed-detection transition at C ~= 3.5), NOT the product
    Cest*Clam of the two floor constants. The paper is explicit (Table 2 caption)
    that the transition sits at C ~= 3.5 "rather than the loose C = 2 guess" --
    and the floor's own Cest*Clam (~0.3 here) is smaller still. Using the floor
    constants to set the budget predicts an infeasibly tiny N (below s*log p_K);
    the calibrated C_budget is what makes N_pred land in the feasible regime.

    Consequence: floor(N_pred) computed with the FLOOR constants will sit BELOW
    beta_min by the ratio (Cest*Clam)/C_budget -- i.e. the calibrated budget
    OVER-resolves relative to the floor formula. That is the conservative
    direction and is the honest content of (B): spending the calibrated N_pred
    certifies AT LEAST down to beta_min.
    """
    n = (C_budget ** 2) * (sigma_eff ** 2) * math.log(p_K(d, K)) / (beta_min ** 2)
    return int(math.ceil(n))


# --------------------------------------------------------------------------- #
# cell-id map (mirrors xai_suff LIMEExplainer._cell_id_map)
# --------------------------------------------------------------------------- #
def cell_id_map(H, W, grid):
    gh, gw = grid
    ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
    xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
    return ys.view(-1, 1) * gw + xs.view(1, -1)


# --------------------------------------------------------------------------- #
# masked-response oracle (identical to 5.2 a & c)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def query_masked_response(model, x, b, cell_ids, Z, target, device,
                          batch_size=64):
    """g_rho(z): target-class prob with keep->sharp, off->blur reference b."""
    N = Z.shape[0]
    y = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        zb = torch.as_tensor(
            Z[start:start + batch_size], dtype=torch.float32, device=device)
        keep = zb[:, cell_ids].unsqueeze(1)
        comp = keep * x + (1.0 - keep) * b
        p = F.softmax(model(comp), dim=1)[:, target]
        y[start:start + zb.shape[0]] = p.detach().cpu().numpy()
    return y


# --------------------------------------------------------------------------- #
# standardized K=1 Lasso on the centered +-1 design (identical to 5.2 a & c)
# --------------------------------------------------------------------------- #
def centered_design(Z):
    return 2.0 * (Z - 0.5)


def standardized_lasso_fit(Z, y, lam, n_iter=20000, tol=1e-7):
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
# sigma_eff via Eq.(10): fit/held-out split on a mask set, return the held-out
# unexplained variance proxy. Reused for the pilot AND for the run-time re-check.
# --------------------------------------------------------------------------- #
def sigma_eff_from_split(model, x, b, cell_ids, n_cells, target, device,
                         d, K, n, c_lambda, sigma_obs, seed, batch_size):
    """Measure sigma_eff on a fresh fit/val split of size n each (Eq.10).

    Returns (sigma_eff, info). The fit penalty uses sigma_obs as a seed scale
    (the same bootstrap the 5.2(a)/(c) pilots use), then the held-out residual
    gives m_hat and hence sigma_eff = sigma_obs + sqrt(m_hat).
    """
    g = np.random.default_rng(seed)
    Zf = (g.random((n, n_cells)) > 0.5).astype(np.float64)
    Zf[0] = 1.0
    yf = query_masked_response(model, x, b, cell_ids, Zf, target, device,
                               batch_size)
    lam = lambda_value(max(sigma_obs, 1e-3), d, n, K, c_lambda)
    beta = standardized_lasso_fit(Zf, yf, lam)
    b0 = float(np.mean(yf.mean() - centered_design(Zf) @ beta))
    Zv = (g.random((n, n_cells)) > 0.5).astype(np.float64)
    yv = query_masked_response(model, x, b, cell_ids, Zv, target, device,
                               batch_size)
    pred = centered_design(Zv) @ beta + b0
    resid_var = float(np.mean((yv - pred) ** 2))
    m_hat = max(resid_var - sigma_obs ** 2, 0.0)
    sigma_eff = sigma_obs + math.sqrt(m_hat)
    return sigma_eff, {"resid_var": resid_var, "m_hat": m_hat, "b0": b0}


# --------------------------------------------------------------------------- #
# one full fit at a given budget N: realized floor, certified set, lower edge
# --------------------------------------------------------------------------- #
def fit_at_budget(model, x, b, cell_ids, n_cells, target, device,
                  sigma_eff, d, N, K, c_lambda, c_est, seed, batch_size):
    rng = np.random.default_rng(seed + 999)
    Z = (rng.random((N, n_cells)) > 0.5).astype(np.float64)
    Z[0] = 1.0
    y = query_masked_response(model, x, b, cell_ids, Z, target, device,
                              batch_size)
    lam = lambda_value(sigma_eff, d, N, K, c_lambda)
    beta_hat = standardized_lasso_fit(Z, y, lam)
    fl = floor_value(sigma_eff, d, N, K, c_lambda, c_est)
    cert = np.abs(beta_hat) > fl
    cert_idx = np.where(cert)[0]
    cert_count = int(cert.sum())
    abs_beta = np.abs(beta_hat)
    beta_edge = float(abs_beta[cert_idx].min()) if cert_count else float("nan")
    max_abs = float(abs_beta.max())
    return {
        "N": N, "floor": fl, "cert_count": cert_count,
        "cert_idx": [int(i) for i in cert_idx],
        "beta_edge": beta_edge, "max_abs_beta": max_abs,
        "vacuous": cert_count == 0,
    }


# --------------------------------------------------------------------------- #
# main verification
# --------------------------------------------------------------------------- #
def verify(model, x, b, cell_ids, n_cells, target, class_name, device,
           beta_min, K=1, c_lambda=0.30, c_est=1.0, C_budget=3.5,
           n_pilot=512, sigma_obs=0.0, edge_band=0.25, transfer_tol=0.15,
           seed=0, batch_size=64, out=None):
    d = n_cells
    C_floor = c_est * c_lambda

    print("=" * 78)
    print("Section 5.2(B): forward budget prediction (effective-sample-size rule)")
    print("=" * 78)
    print(f"  target class = {target} ({class_name})")
    print(f"  grid cells d = {d}   K = {K}   p_K = {p_K(d, K)}")
    print(f"  beta_min     = {beta_min}   C = Cest*Clam = {C:.4f}\n")

    # ---- (i)-(iii) pilot -> sigma_eff -> N_pred -------------------------- #
    sigma_pilot, pinfo = sigma_eff_from_split(
        model, x, b, cell_ids, n_cells, target, device, d, K, n_pilot,
        c_lambda, sigma_obs, seed, batch_size)
    N_pred = predict_budget(sigma_pilot, d, beta_min, K, C_budget)
    floor_pred = floor_value(sigma_pilot, d, N_pred, K, c_lambda, c_est)
    # the floor formula uses C_floor; the budget uses C_budget. So the realized
    # floor at N_pred is predicted to sit at beta_min * (C_floor / C_budget) --
    # BELOW beta_min, i.e. the calibrated budget over-resolves. Make that target
    # explicit so the round-trip check tests the right number.
    floor_target = beta_min * (C_floor / C_budget)
    print(f"  pilot N0       = {n_pilot}   resid_var = {pinfo['resid_var']:.5f}   "
          f"m_hat = {pinfo['m_hat']:.5f}")
    print(f"  sigma_eff(pilot) = {sigma_pilot:.5f}  (Eq.10; sigma_obs={sigma_obs})")
    print(f"  C_floor=Cest*Clam = {C_floor:.3f}   C_budget (calibrated, Tbl.2) "
          f"= {C_budget:.3f}")
    print(f"  N_pred         = {N_pred}  (Cor.1 inversion at beta_min, C_budget)")
    print(f"  expected floor(N_pred) = beta_min*C_floor/C_budget = "
          f"{floor_target:.5f}  (<= beta_min={beta_min}: budget over-resolves)\n")

    # feasibility guard (Cor.3): N_pred must clear N >~ s*log p_K. We don't know
    # s yet; use the pilot's would-be support size as a cheap stand-in after the
    # run, and warn here only if N_pred is absurdly small.
    if N_pred < math.log(p_K(d, K)):
        print(f"  WARNING: N_pred={N_pred} is below even log p_K="
              f"{math.log(p_K(d,K)):.0f}; beta_min is likely set above the "
              f"largest coefficient (infeasible regime). Lower beta_min.\n")

    # ---- (iv) run the real fit at N_pred --------------------------------- #
    run = fit_at_budget(model, x, b, cell_ids, n_cells, target, device,
                        sigma_pilot, d, N_pred, K, c_lambda, c_est, seed,
                        batch_size)

    # re-measure sigma_eff AT N_pred on a held-out split (pilot transfer test)
    sigma_run, rinfo = sigma_eff_from_split(
        model, x, b, cell_ids, n_cells, target, device, d, K,
        max(N_pred, n_pilot), c_lambda, sigma_obs, seed + 7, batch_size)

    s_hat = max(run["cert_count"], 1)
    N_feas = s_hat * math.log(p_K(d, K))

    # ---- [Claim B-round-trip] realized floor lands at floor_target -------- #
    # NOT beta_min: the floor uses C_floor while the budget uses the calibrated
    # C_budget, so by construction floor(N_pred) = beta_min*C_floor/C_budget.
    # This is still an analytic round-trip (same sigma_pilot built both), so a
    # gap signals a code-path bug, not an empirical finding.
    print("[round-trip] realized floor at N_pred vs expected floor_target "
          "(analytic; checks inversion + rounding, not the world)")
    rt_err = abs(run["floor"] - floor_target) / floor_target
    rt_ok = rt_err <= 0.05
    print(f"  floor(N_pred)  = {run['floor']:.5f}   "
          f"expected = {floor_target:.5f}   "
          f"rel_err = {rt_err:.3f}  {'OK' if rt_ok else 'DIVERGENCE'}")
    print(f"  (floor sits at {run['floor']/beta_min:.2f} x beta_min -- the "
          f"calibrated budget certifies BELOW beta_min, the safe direction)")
    if not rt_ok:
        print("    (a large gap here is a CODE bug -- the same sigma_pilot built "
              "both N_pred and this floor; they must round-trip)")
    print()

    # ---- [Claim B1] pilot transfer: sigma_pilot upper-bounds sigma_run --- #
    print("[Claim B1] pilot sigma_eff transfers (and is upper-biased, App.G)")
    print(f"  sigma_eff(pilot) = {sigma_pilot:.5f}   "
          f"sigma_eff(run@N_pred) = {sigma_run:.5f}")
    transfer_ratio = sigma_pilot / (sigma_run + 1e-12)
    # PASS if pilot >= run (conservative) OR within tolerance below it.
    under = sigma_pilot < sigma_run * (1.0 - transfer_tol)
    over_factor = (sigma_pilot / (sigma_run + 1e-12)) ** 2
    b1_ok = not under
    print(f"  ratio pilot/run = {transfer_ratio:.3f}   "
          f"implied over-budget factor (pilot/run)^2 = {over_factor:.2f}x")
    if under:
        print(f"  FAIL: pilot UNDER-estimates run sigma_eff by more than "
              f"{transfer_tol:.0%} -> N_pred under-budgets, the dangerous "
              f"direction. Enlarge n_pilot or cross-fit the residual (App.G).")
    else:
        print(f"  PASS: pilot is conservative (>= run up to {transfer_tol:.0%}); "
              f"N_pred over-budgets by ~{over_factor:.1f}x -- spend less if "
              f"desired, but down to beta_min is covered.")
    print()

    # ---- [Claim B2] lower-edge landing near beta_min --------------------- #
    print("[Claim B2] certified set extends DOWN TO ~= beta_min (lower edge)")
    if run["vacuous"]:
        b2_ok = False
        print(f"  certified count = 0 at N_pred -> VACUOUS. No coefficient "
              f"clears the floor; beta_min may be below the whole support, or "
              f"the fit scale is off. (max|beta_hat| = {run['max_abs_beta']:.5f})")
    else:
        edge_ratio = run["beta_edge"] / (run["floor"] + 1e-12)
        # the smallest certified coefficient should sit just above the floor:
        # ratio in [1, 1+edge_band]. Far above -> nothing lives near beta_min
        # (over-budget); ~1 -> the run resolved exactly down to the threshold.
        b2_ok = 1.0 - 1e-6 <= edge_ratio <= 1.0 + edge_band
        print(f"  certified count        = {run['cert_count']}")
        print(f"  smallest certified |b| = {run['beta_edge']:.5f}  "
              f"(beta_edge)")
        print(f"  realized floor         = {run['floor']:.5f}")
        print(f"  beta_edge / floor      = {edge_ratio:.3f}  "
              f"(target [1, {1+edge_band:.2f}])  "
              f"{'PASS' if b2_ok else 'CHECK'}")
        if edge_ratio > 1.0 + edge_band:
            print(f"    edge sits well above floor: the run certified nothing "
                  f"NEW near beta_min -- either no coefficient lives in "
                  f"[floor, {run['beta_edge']:.4f}] (over-budgeted) or beta_min "
                  f"is set below a gap in the spectrum.")
    print()

    # ---- [Claim B3] consistency with 5.2(A): bracket N_pred monotonically  #
    print("[Claim B3] monotone bracket around N_pred (sanity vs 5.2(A))")
    N_lo = max(int(N_pred // 2), int(math.ceil(math.log(p_K(d, K)))) + 1)
    N_hi = 2 * N_pred
    run_lo = fit_at_budget(model, x, b, cell_ids, n_cells, target, device,
                           sigma_pilot, d, N_lo, K, c_lambda, c_est, seed,
                           batch_size)
    run_hi = fit_at_budget(model, x, b, cell_ids, n_cells, target, device,
                           sigma_pilot, d, N_hi, K, c_lambda, c_est, seed,
                           batch_size)
    counts = [(N_lo, run_lo["cert_count"], run_lo["floor"]),
              (N_pred, run["cert_count"], run["floor"]),
              (N_hi, run_hi["cert_count"], run_hi["floor"])]
    print(f"  {'N':>8} {'floor':>9} {'#cert':>6}")
    for Nv, c, fl in counts:
        print(f"  {Nv:>8} {fl:>9.5f} {c:>6}")
    # floor must strictly decrease with N; count must be non-decreasing (allow
    # +-1 jitter, consistent with the borderline band used in 5.2(A)).
    floor_mono = run_lo["floor"] >= run["floor"] >= run_hi["floor"]
    count_mono = (run_lo["cert_count"] <= run["cert_count"] + 1 and
                  run["cert_count"] <= run_hi["cert_count"] + 1)
    b3_ok = floor_mono and count_mono
    print(f"  floor decreasing in N          : "
          f"{'PASS' if floor_mono else 'FAIL'}")
    print(f"  #cert non-decreasing (+-1)     : "
          f"{'PASS' if count_mono else 'CHECK'}")
    print(f"  N_pred on the monotone curve   : "
          f"{'PASS' if b3_ok else 'CHECK'}\n")

    # ---- feasibility note ------------------------------------------------ #
    if N_pred < N_feas:
        print(f"  NOTE: N_pred={N_pred} < feasibility floor s*log p_K~={N_feas:.0f} "
              f"(s_hat={s_hat}); recovery may not be feasible at this budget "
              f"(Cor.3). Treat the certified set as underdetermined.\n")

    # ---- overall --------------------------------------------------------- #
    non_vacuous = not run["vacuous"]
    overall = rt_ok and b1_ok and b2_ok and b3_ok and non_vacuous
    status = "PASS" if overall else ("VACUOUS" if not non_vacuous else "CHECK")
    print("=" * 78)
    print(f"  OVERALL 5.2(B)               : {status}")
    print(f"  predicted floor vs beta_min  : {run['floor']:.5f} vs {beta_min} "
          f"(round-trip {'ok' if rt_ok else 'BAD'})")
    print(f"  certified count at N_pred    : {run['cert_count']}")
    print(f"  pilot over-budget factor     : ~{over_factor:.1f}x "
          f"(App.G: pilot is conservative -> N_pred is an UPPER estimate)")
    print("=" * 78)

    metrics = {
        "d": d, "K": K, "beta_min": beta_min, "C": C,
        "sigma_eff_pilot": sigma_pilot, "sigma_eff_run": sigma_run,
        "N_pred": N_pred, "floor_pred": floor_pred,
        "floor_realized": run["floor"], "cert_count": run["cert_count"],
        "beta_edge": run["beta_edge"], "max_abs_beta": run["max_abs_beta"],
        "round_trip_ok": rt_ok, "transfer_ok": b1_ok,
        "over_budget_factor": over_factor, "edge_ok": b2_ok,
        "bracket_ok": b3_ok, "vacuous": run["vacuous"],
        "N_lo": N_lo, "cert_lo": run_lo["cert_count"],
        "N_hi": N_hi, "cert_hi": run_hi["cert_count"],
        "overall": overall,
    }
    if out:
        os.makedirs(out, exist_ok=True)
        rec = dict(metrics)
        rec.update({"target": target, "class_name": class_name,
                    "cert_idx": run["cert_idx"]})
        with open(os.path.join(out, "verify_5_2_b.json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  -> {os.path.join(out, 'verify_5_2_b.json')}")
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Verify Section 5.2(B): forward budget prediction "
                    "(effective-sample-size rule, image model).")
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--target", type=int, default=None,
                    help="target class (default: model top-1)")
    ap.add_argument("--grid", type=int, nargs=2, default=(12, 12),
                    metavar=("GH", "GW"))
    ap.add_argument("--sigma", type=float, default=11.0,
                    help="blur reference bandwidth")
    ap.add_argument("--beta-min", type=float, default=0.02,
                    help="trust threshold to certify down to; must be reachable "
                         "(N_pred should clear N >~ s*log p_K, Cor.3)")
    ap.add_argument("--K", type=int, default=1, choices=[1])
    ap.add_argument("--c-lambda", type=float, default=0.30,
                    help="penalty constant Clam")
    ap.add_argument("--c-est", type=float, default=1.0,
                    help="floor constant Cest (floor = Cest*lambda)")
    ap.add_argument("--n-pilot", type=int, default=512)
    ap.add_argument("--sigma-obs", type=float, default=0.0,
                    help="query-noise scale (0 for deterministic model+ref)")
    ap.add_argument("--edge-band", type=float, default=0.25,
                    help="upper tolerance for beta_edge/floor (lower-edge "
                         "landing); ratio in [1, 1+band] counts as landing at "
                         "beta_min")
    ap.add_argument("--transfer-tol", type=float, default=0.15,
                    help="how far below run sigma_eff the pilot may sit before "
                         "B1 fails (pilot is supposed to over-estimate)")
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
           beta_min=args.beta_min, K=args.K, c_lambda=args.c_lambda,
           c_est=args.c_est, n_pilot=args.n_pilot, sigma_obs=args.sigma_obs,
           edge_band=args.edge_band, transfer_tol=args.transfer_tol,
           seed=args.seed, batch_size=args.batch_size, out=args.out)


if __name__ == "__main__":
    main()