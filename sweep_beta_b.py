"""Sweep beta_min on a single image to test Corollary 1's scaling law.

The single-point 5.2(B) check confirms the budget rule is *operable* on one
beta_min: the pilot transfers conservatively (B1) and the inversion round-trips.
What it CANNOT show is that the rule obeys the law it claims -- Corollary 1 says

    N_pred  ~  C_budget^2 * sigma_eff^2 * log p_K / beta_min^2,

i.e. N_pred should scale as beta_min^(-2): halving beta_min quadruples the budget
(the paper's headline 1/gamma^2 statement, Table 2). A single beta_min can't
exhibit a slope. This wrapper runs verify() across a grid of beta_min on the SAME
image (same pilot draws up to seed reuse) and checks two things the point cannot:

  (S1) BUDGET SCALING. Regress log N_pred on log beta_min. The slope should be
       ~= -2. (sigma_eff is re-measured per call but on the same image, so it is
       roughly constant across the sweep; any drift is reported so a slope that
       deviates from -2 can be attributed to sigma_eff drift vs. a real break.)

  (S2) FLOOR TRACKING. The realized floor should sit at beta_min*C_floor/C_budget
       at every beta_min -- i.e. floor_realized / beta_min should be ~constant
       (= C_floor/C_budget). This is the round-trip holding ACROSS the sweep, not
       just at one point: a beta_min where it breaks exposes a regime issue
       (infeasible N, or beta_min outside the resolvable spectrum).

  (S3) TRANSFER STABILITY. B1 (pilot >= run sigma_eff) should hold at EVERY
       beta_min, not just the one we happened to pick. We report the pilot/run
       ratio per row; any row < 1 is the dangerous (under-budget) direction.

This does NOT make the result multi-image -- that's the separate dataset loop.
It upgrades the single image from "the rule runs here" to "the rule obeys its
scaling law here." Multi-image generalization is still required for a population
claim.

No torch is run automatically. Execute yourself, e.g.:

    python sweep_beta_min_5_2_b.py --image /path/to/church.jpg
    python sweep_beta_min_5_2_b.py --image img.jpg --grid 12 12 --sigma 11 \
        --beta-mins 0.005 0.01 0.02 0.04 0.08 --n-pilot 512 --device cuda

Choose the grid so the smallest beta_min still clears feasibility (N >~ s*log p_K,
Cor.3) and the largest stays below the biggest real coefficient; verify() warns
per row when a beta_min falls outside the resolvable band, and the summary flags
any row that went vacuous or infeasible so it can be excluded from the fit.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os

import numpy as np
import torch

from xai_suff.backbone import (
    get_class_names,
    load_backbone,
    load_image,
)
from xai_suff.explainers import blur_reference

# reuse the single-point verifier verbatim -- same floor/budget/transfer logic
from verify_5_2_b import (
    cell_id_map,
    p_K,
    verify,
)


def _loglog_slope(xs, ys):
    """OLS slope of log ys on log xs (theory: N_pred vs beta_min -> -2)."""
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2)) + 1e-18
    r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def sweep(model, x, b, cell_ids, n_cells, target, class_name, device,
          beta_mins, K=1, c_lambda=0.30, c_est=1.0, C_budget=3.5,
          n_pilot=512, sigma_obs=0.0, edge_band=0.25, transfer_tol=0.15,
          seed=0, batch_size=64, out=None, verbose=False):
    d = n_cells
    C_floor = c_est * c_lambda
    expected_floor_ratio = C_floor / C_budget  # floor_realized / beta_min target

    print("#" * 78)
    print("Section 5.2(B) sweep: Corollary 1 scaling on a single image")
    print("#" * 78)
    print(f"  target class = {target} ({class_name})")
    print(f"  grid cells d = {d}   K = {K}   p_K = {p_K(d, K)}")
    print(f"  C_floor = {C_floor:.3f}   C_budget = {C_budget:.3f}   "
          f"expected floor/beta_min = {expected_floor_ratio:.4f}")
    print(f"  beta_min grid = {list(beta_mins)}\n")

    rows = []
    for bm in beta_mins:
        # run the full single-point verifier; capture its console block unless
        # --verbose, so the sweep table stays readable.
        sink = io.StringIO()
        ctx = contextlib.redirect_stdout(sink) if not verbose else \
            contextlib.nullcontext()
        with ctx:
            m = verify(model, x, b, cell_ids, n_cells, target, class_name,
                       device, beta_min=bm, K=K, c_lambda=c_lambda,
                       c_est=c_est, C_budget=C_budget, n_pilot=n_pilot,
                       sigma_obs=sigma_obs, edge_band=edge_band,
                       transfer_tol=transfer_tol, seed=seed,
                       batch_size=batch_size, out=None)
        if verbose:
            print(sink.getvalue() if False else "")  # already printed live

        floor_over_bmin = m["floor_realized"] / bm
        pilot_over_run = m["sigma_eff_pilot"] / (m["sigma_eff_run"] + 1e-12)
        N_feas = max(m["cert_count"], 1) * math.log(p_K(d, K))
        feasible = m["N_pred"] >= N_feas
        rows.append({
            "beta_min": bm,
            "N_pred": m["N_pred"],
            "sigma_eff_pilot": m["sigma_eff_pilot"],
            "sigma_eff_run": m["sigma_eff_run"],
            "floor_realized": m["floor_realized"],
            "floor_over_bmin": floor_over_bmin,
            "pilot_over_run": pilot_over_run,
            "over_budget_factor": m["over_budget_factor"],
            "cert_count": m["cert_count"],
            "beta_edge": m["beta_edge"],
            "edge_ok": m["edge_ok"],
            "vacuous": m["vacuous"],
            "feasible": feasible,
            "overall": m["overall"],
        })

    # ---- per-row table --------------------------------------------------- #
    print(f"  {'beta_min':>9} {'N_pred':>8} {'sig_pilot':>9} {'sig_run':>8} "
          f"{'floor':>9} {'fl/bmin':>8} {'pil/run':>8} {'#cert':>6} "
          f"{'edge':>5} {'feas':>5}")
    for r in rows:
        print(f"  {r['beta_min']:>9.4f} {r['N_pred']:>8} "
              f"{r['sigma_eff_pilot']:>9.5f} {r['sigma_eff_run']:>8.5f} "
              f"{r['floor_realized']:>9.5f} {r['floor_over_bmin']:>8.4f} "
              f"{r['pilot_over_run']:>8.3f} {r['cert_count']:>6} "
              f"{'ok' if r['edge_ok'] else 'CHK':>5} "
              f"{'ok' if r['feasible'] else 'NO':>5}")
    print()

    # only rows that are usable for the scaling fit (non-vacuous, feasible)
    usable = [r for r in rows if not r["vacuous"] and r["feasible"]]
    excluded = [r for r in rows if r not in usable]
    if excluded:
        print(f"  excluded from fit (vacuous/infeasible): "
              f"{[r['beta_min'] for r in excluded]}\n")

    # ---- [S1] budget scaling: slope of log N_pred vs log beta_min ~ -2 --- #
    print("[S1] budget scaling  (Cor.1: N_pred ~ beta_min^-2, slope = -2)")
    if len(usable) >= 2:
        bms = [r["beta_min"] for r in usable]
        nps = [r["N_pred"] for r in usable]
        slope, intercept, r2 = _loglog_slope(bms, nps)
        # sigma_eff drift across the sweep: if sigma_eff moves, N_pred carries an
        # extra sigma_eff^2 factor that is NOT the beta_min law -- report it so a
        # slope off -2 can be attributed correctly.
        sig = [r["sigma_eff_pilot"] for r in usable]
        sig_cov = float(np.std(sig) / (np.mean(sig) + 1e-12))
        s1_ok = abs(slope - (-2.0)) <= 0.15 and r2 >= 0.98
        print(f"  log-log slope = {slope:.3f}   (target -2.000)   R^2 = {r2:.4f}")
        print(f"  sigma_eff(pilot) CoV across sweep = {sig_cov:.3f}  "
              f"(low -> slope is the beta_min law, not sigma_eff drift)")
        print(f"  {'PASS' if s1_ok else 'CHECK'}: "
              f"{'slope matches the 1/beta_min^2 law' if s1_ok else 'slope deviates from -2 -- check sigma_eff drift / feasibility at the small-beta_min end'}")
    else:
        slope = intercept = r2 = sig_cov = float("nan")
        s1_ok = False
        print("  CHECK: fewer than 2 usable rows -- cannot fit a slope. "
              "Widen the grid or move it into the feasible band.")
    print()

    # ---- [S2] floor tracking: floor/beta_min constant = C_floor/C_budget - #
    print("[S2] floor tracking  (round-trip holds across the sweep)")
    if usable:
        ratios = [r["floor_over_bmin"] for r in usable]
        ratio_mean = float(np.mean(ratios))
        ratio_cov = float(np.std(ratios) / (ratio_mean + 1e-12))
        rel_to_expected = abs(ratio_mean - expected_floor_ratio) / \
            expected_floor_ratio
        s2_ok = ratio_cov <= 0.02 and rel_to_expected <= 0.05
        print(f"  floor/beta_min  = {ratio_mean:.4f} +- (CoV {ratio_cov:.3f})   "
              f"expected = {expected_floor_ratio:.4f}   "
              f"rel_err = {rel_to_expected:.3f}")
        print(f"  {'PASS' if s2_ok else 'CHECK'}: "
              f"{'floor tracks beta_min at the calibrated ratio everywhere' if s2_ok else 'ratio drifts -- a beta_min may be outside the resolvable band'}")
    else:
        ratio_mean = ratio_cov = float("nan")
        s2_ok = False
        print("  CHECK: no usable rows.")
    print()

    # ---- [S3] transfer stability: B1 holds at every beta_min ------------- #
    print("[S3] transfer stability  (B1: pilot >= run sigma_eff at every row)")
    under_rows = [r["beta_min"] for r in rows
                  if r["pilot_over_run"] < (1.0 - transfer_tol)]
    s3_ok = len(under_rows) == 0
    if s3_ok:
        ratios = [r["pilot_over_run"] for r in rows]
        print(f"  pilot/run in [{min(ratios):.3f}, {max(ratios):.3f}] "
              f"across all rows (all >= {1-transfer_tol:.2f})")
        print(f"  PASS: pilot is conservative at every beta_min -- no row "
              f"under-budgets.")
    else:
        print(f"  FAIL: pilot UNDER-estimates run sigma_eff at beta_min="
              f"{under_rows} (dangerous / under-budget direction). "
              f"Enlarge n_pilot or cross-fit (App.G).")
    print()

    # ---- overall --------------------------------------------------------- #
    overall = s1_ok and s2_ok and s3_ok
    print("#" * 78)
    print(f"  SWEEP OVERALL                : "
          f"{'PASS' if overall else 'CHECK'}")
    print(f"  budget slope (target -2)     : "
          f"{slope:.3f} (R^2 {r2:.3f})" if not math.isnan(slope)
          else "  budget slope                 : n/a")
    print(f"  floor/beta_min (target {expected_floor_ratio:.3f}) : "
          f"{ratio_mean:.4f}" if not math.isnan(ratio_mean)
          else "  floor/beta_min               : n/a")
    print(f"  transfer conservative all rows: {'yes' if s3_ok else 'NO'}")
    print("#" * 78)

    summary = {
        "d": d, "K": K, "C_floor": C_floor, "C_budget": C_budget,
        "expected_floor_ratio": expected_floor_ratio,
        "beta_mins": list(beta_mins),
        "budget_slope": slope, "budget_slope_r2": r2,
        "sigma_eff_cov": sig_cov if 'sig_cov' in dir() else float("nan"),
        "floor_over_bmin_mean": ratio_mean, "floor_over_bmin_cov": ratio_cov,
        "transfer_under_rows": under_rows,
        "s1_ok": s1_ok, "s2_ok": s2_ok, "s3_ok": s3_ok, "overall": overall,
        "rows": rows,
    }
    if out:
        os.makedirs(out, exist_ok=True)
        rec = dict(summary)
        rec.update({"target": target, "class_name": class_name})
        with open(os.path.join(out, "sweep_beta_min_5_2_b.json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  -> {os.path.join(out, 'sweep_beta_min_5_2_b.json')}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Sweep beta_min on one image to test Corollary 1 scaling "
                    "(reuses verify_5_2_b.verify).")
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--target", type=int, default=None,
                    help="target class (default: model top-1)")
    ap.add_argument("--grid", type=int, nargs=2, default=(12, 12),
                    metavar=("GH", "GW"))
    ap.add_argument("--sigma", type=float, default=11.0,
                    help="blur reference bandwidth")
    ap.add_argument("--beta-mins", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.04, 0.08],
                    help="beta_min grid; pick so the smallest clears "
                         "feasibility (Cor.3) and the largest stays below the "
                         "biggest real coefficient")
    ap.add_argument("--K", type=int, default=1, choices=[1])
    ap.add_argument("--c-lambda", type=float, default=0.30)
    ap.add_argument("--c-est", type=float, default=1.0)
    ap.add_argument("--c-budget", type=float, default=3.5)
    ap.add_argument("--n-pilot", type=int, default=512)
    ap.add_argument("--sigma-obs", type=float, default=0.0)
    ap.add_argument("--edge-band", type=float, default=0.25)
    ap.add_argument("--transfer-tol", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="print each per-beta_min verify() block in full")
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

    beta_mins = sorted(args.beta_mins)
    sweep(model, x, b, cell_ids, n_cells, target, class_names[target], device,
          beta_mins=beta_mins, K=args.K, c_lambda=args.c_lambda,
          c_est=args.c_est, C_budget=args.c_budget, n_pilot=args.n_pilot,
          sigma_obs=args.sigma_obs, edge_band=args.edge_band,
          transfer_tol=args.transfer_tol, seed=args.seed,
          batch_size=args.batch_size, out=args.out, verbose=args.verbose)


if __name__ == "__main__":
    main()