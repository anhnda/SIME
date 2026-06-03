"""Loop Section 5.2(B) over an image folder to test the budget rule
DISTRIBUTIONALLY -- the generalization claim a single image cannot make.

The single image (and the beta_min sweep on it) established that the budget rule
is OPERABLE and CONSERVATIVE on one church image: the pilot sigma_eff over-
estimates the run sigma_eff (B1), the floor round-trips, the certified set
extends to the realized floor. None of that is a population statement. The rule's
safety rests entirely on ONE claim being true ACROSS images, not on average but
in the TAIL:

    (G1) PILOT NEVER UNDER-BUDGETS. Corollary 1 + Appendix G promise the pilot
         proxy is UPPER-biased: sigma_eff(pilot) >= sigma_eff(run). If that holds
         only on average but fails on some images, those images get an N_pred too
         small to certify down to beta_min -- the dangerous, silent failure. The
         population claim is therefore about the WORST image, not the mean:
            #{images : pilot/run < 1 - tol}  should be 0 (or a controlled, tiny
            fraction), and we report the full distribution + the minimum ratio.

We also aggregate, as supporting (not load-bearing) evidence:

    (G2) ROUND-TRIP holds on every image (analytic; a failure is a code/regime
         bug on that image -- e.g. beta_min outside its resolvable spectrum).
    (G3) OVER-BUDGET FACTOR distribution: (pilot/run)^2, the slack the practitioner
         pays. Reported so the cost of conservativeness is visible across the set.
    (G4) VACUOUS / INFEASIBLE rate: images where beta_min certifies nothing or
         N_pred < s*log p_K. A high rate means beta_min is mis-set for this set,
         not that the rule failed.

WHAT THIS DOES AND DOES NOT ESTABLISH. This tests that the budget rule's
conservativeness GENERALIZES across natural images at a fixed beta_min -- the
missing population evidence for 5.2(B). It does NOT test that the detection floor
is the correct normalizer; that is the synthetic support-recovery collapse
(Section 5.1, Table 1), which is a separate and still-required result. A clean
G1 here means "spending N_pred certifies down to beta_min on these images, with
the slack in G3"; it does not certify the underlying floor theory.

Each image is scored at ITS OWN top-1 class unless --target is pinned. beta_min
is held fixed across the set; pick it so most images are feasible and non-vacuous
(the per-image warnings from verify() are suppressed unless --verbose; the
summary reports the vacuous/infeasible rate so a bad beta_min is visible).

No torch is run automatically. Execute yourself, e.g.:

    python loop_5_2_b.py --image-dir benchmark_50
    python loop_5_2_b.py --image-dir benchmark_50 --glob '*.JPEG' \
        --beta-min 0.02 --grid 12 12 --sigma 11 --n-pilot 512 --device cuda \
        --out runs/loop_b
"""
from __future__ import annotations

import argparse
import contextlib
import glob
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

# reuse the single-point verifier verbatim
from verify_5_2_b import (
    cell_id_map,
    p_K,
    verify,
)


def _pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else float("nan")


def loop(model, class_names, device, image_paths, target_pin,
         grid, sigma, beta_min, K=1, c_lambda=0.30, c_est=1.0, C_budget=3.5,
         n_pilot=512, sigma_obs=0.0, edge_band=0.25, transfer_tol=0.15,
         seed=0, batch_size=64, out=None, verbose=False):
    n_cells = grid[0] * grid[1]
    d = n_cells

    print("#" * 78)
    print(f"Section 5.2(B) loop: budget-rule generalization over {len(image_paths)} images")
    print("#" * 78)
    print(f"  grid cells d = {d}   K = {K}   p_K = {p_K(d, K)}")
    print(f"  beta_min     = {beta_min}   transfer_tol = {transfer_tol}")
    print(f"  target       = {'pinned ' + str(target_pin) if target_pin is not None else 'per-image top-1'}\n")

    rows = []
    for i, path in enumerate(image_paths):
        name = os.path.basename(path)
        try:
            x = load_image(path, device)
            _, _, H, W = x.shape
            if target_pin is not None:
                target = target_pin
            else:
                with torch.no_grad():
                    target = int(model(x).argmax(1).item())
            cell_ids = cell_id_map(H, W, tuple(grid)).to(device)
            b = blur_reference(x, sigma).to(device)

            sink = io.StringIO()
            ctx = contextlib.redirect_stdout(sink) if not verbose else \
                contextlib.nullcontext()
            with ctx:
                m = verify(model, x, b, cell_ids, n_cells, target,
                           class_names[target], device, beta_min=beta_min,
                           K=K, c_lambda=c_lambda, c_est=c_est,
                           C_budget=C_budget, n_pilot=n_pilot,
                           sigma_obs=sigma_obs, edge_band=edge_band,
                           transfer_tol=transfer_tol, seed=seed,
                           batch_size=batch_size, out=None)

            pilot_over_run = m["sigma_eff_pilot"] / (m["sigma_eff_run"] + 1e-12)
            N_feas = max(m["cert_count"], 1) * math.log(p_K(d, K))
            feasible = m["N_pred"] >= N_feas
            under = pilot_over_run < (1.0 - transfer_tol)
            rows.append({
                "image": name, "target": target,
                "class_name": class_names[target],
                "sigma_eff_pilot": m["sigma_eff_pilot"],
                "sigma_eff_run": m["sigma_eff_run"],
                "pilot_over_run": pilot_over_run,
                "over_budget_factor": m["over_budget_factor"],
                "N_pred": m["N_pred"],
                "floor_realized": m["floor_realized"],
                "floor_target": m["floor_target"],
                "round_trip_ok": m["round_trip_ok"],
                "cert_count": m["cert_count"],
                "beta_edge": m["beta_edge"],
                "edge_ok": m["edge_ok"],
                "vacuous": m["vacuous"],
                "feasible": feasible,
                "under_budget": under,
                "error": None,
            })
            flag = "UNDER" if under else ("vac" if m["vacuous"] else "ok")
            print(f"  [{i+1:>3}/{len(image_paths)}] {name:<28} "
                  f"cls={target:<4} pil/run={pilot_over_run:6.3f} "
                  f"N_pred={m['N_pred']:>7} #cert={m['cert_count']:>4} {flag}")
        except Exception as e:
            rows.append({"image": name, "error": repr(e)})
            print(f"  [{i+1:>3}/{len(image_paths)}] {name:<28} ERROR: {e!r}")

    ok_rows = [r for r in rows if r.get("error") is None]
    n_err = len(rows) - len(ok_rows)
    usable = [r for r in ok_rows if not r["vacuous"] and r["feasible"]]
    print()

    if not ok_rows:
        print("  no successfully processed images -- nothing to aggregate.")
        return {"rows": rows, "n_error": n_err}

    # ---- [G1] pilot never under-budgets (the load-bearing tail claim) ---- #
    print("[G1] pilot transfer holds ACROSS the set (tail claim, not mean)")
    ratios = [r["pilot_over_run"] for r in ok_rows]
    under_imgs = [r["image"] for r in ok_rows if r["under_budget"]]
    n_under = len(under_imgs)
    min_ratio = min(ratios)
    g1_ok = n_under == 0
    print(f"  images scored        = {len(ok_rows)}   (errors {n_err})")
    print(f"  pilot/run min        = {min_ratio:.3f}   "
          f"(threshold for under-budget = {1-transfer_tol:.2f})")
    print(f"  pilot/run median     = {_pct(ratios,50):.3f}   "
          f"p10 = {_pct(ratios,10):.3f}   p90 = {_pct(ratios,90):.3f}")
    print(f"  under-budget images  = {n_under} / {len(ok_rows)} "
          f"({100*n_under/len(ok_rows):.1f}%)")
    if g1_ok:
        print(f"  PASS: no image under-budgets -- pilot is conservative across "
              f"the set. Spending N_pred certifies down to beta_min on all "
              f"{len(ok_rows)} images.")
    else:
        print(f"  FAIL: {n_under} image(s) UNDER-BUDGET (pilot < run sigma_eff "
              f"by > {transfer_tol:.0%}): {under_imgs[:8]}"
              f"{' ...' if n_under > 8 else ''}")
        print(f"        These get an N_pred too small to certify down to "
              f"beta_min -- the silent failure. Enlarge n_pilot or cross-fit "
              f"the residual (App.G).")
    print()

    # ---- [G2] round-trip on every image (analytic sanity) ---------------- #
    print("[G2] floor round-trip holds on every image (analytic)")
    rt_fail = [r["image"] for r in ok_rows if not r["round_trip_ok"]]
    g2_ok = len(rt_fail) == 0
    if g2_ok:
        print(f"  PASS: realized floor round-trips to beta_min*C_floor/C_budget "
              f"on all {len(ok_rows)} images.")
    else:
        print(f"  CHECK: round-trip diverged on {len(rt_fail)} image(s): "
              f"{rt_fail[:8]} -- code/regime bug (beta_min outside resolvable "
              f"spectrum on those images).")
    print()

    # ---- [G3] over-budget factor distribution (cost of conservativeness) - #
    print("[G3] over-budget factor (pilot/run)^2 -- slack the practitioner pays")
    obf = [r["over_budget_factor"] for r in ok_rows]
    print(f"  median = {_pct(obf,50):.2f}x   p10 = {_pct(obf,10):.2f}x   "
          f"p90 = {_pct(obf,90):.2f}x   max = {max(obf):.2f}x")
    print(f"  (App.G: pilot is conservative -> these are UPPER estimates; the "
          f"practitioner may spend less than N_pred down to beta_min.)")
    print()

    # ---- [G4] vacuous / infeasible rate (is beta_min well-set?) ---------- #
    print("[G4] vacuous / infeasible rate (diagnoses beta_min for this set)")
    n_vac = sum(1 for r in ok_rows if r["vacuous"])
    n_infeas = sum(1 for r in ok_rows if not r["feasible"])
    print(f"  vacuous (0 certified)        = {n_vac} / {len(ok_rows)} "
          f"({100*n_vac/len(ok_rows):.1f}%)")
    print(f"  infeasible (N_pred<s*logp_K) = {n_infeas} / {len(ok_rows)} "
          f"({100*n_infeas/len(ok_rows):.1f}%)")
    print(f"  usable for tail claim        = {len(usable)} / {len(ok_rows)}")
    if n_vac + n_infeas > 0.2 * len(ok_rows):
        print(f"  NOTE: >20% of images vacuous/infeasible -- beta_min={beta_min} "
              f"may be mis-set for this set (too low -> huge N; too high -> "
              f"nothing resolves). This is a beta_min choice issue, not a rule "
              f"failure.")
    print()

    # ---- overall --------------------------------------------------------- #
    # G1 is load-bearing; G2 must hold (else code bug); G3/G4 are diagnostics.
    overall = g1_ok and g2_ok
    print("#" * 78)
    print(f"  LOOP OVERALL                 : {'PASS' if overall else 'FAIL'}")
    print(f"  pilot conservative all images: "
          f"{'yes' if g1_ok else 'NO (' + str(n_under) + ' under-budget)'}")
    print(f"  min pilot/run ratio          : {min_ratio:.3f}")
    print(f"  round-trip all images        : {'yes' if g2_ok else 'NO'}")
    print(f"  median over-budget factor    : {_pct(obf,50):.2f}x")
    print(f"  vacuous/infeasible           : {n_vac}/{n_infeas} of {len(ok_rows)}")
    print("#" * 78)
    print("  REMINDER: this is generalization of the budget rule's "
          "conservativeness,")
    print("  NOT confirmation of the detection-floor theory (that is the "
          "synthetic")
    print("  support-recovery collapse, Section 5.1 / Table 1 -- still "
          "required).")
    print("#" * 78)

    summary = {
        "n_images": len(image_paths), "n_scored": len(ok_rows),
        "n_error": n_err, "beta_min": beta_min, "d": d, "K": K,
        "transfer_tol": transfer_tol,
        "pilot_over_run_min": min_ratio,
        "pilot_over_run_median": _pct(ratios, 50),
        "pilot_over_run_p10": _pct(ratios, 10),
        "pilot_over_run_p90": _pct(ratios, 90),
        "n_under_budget": n_under, "under_budget_images": under_imgs,
        "n_round_trip_fail": len(rt_fail),
        "over_budget_factor_median": _pct(obf, 50),
        "over_budget_factor_max": max(obf),
        "n_vacuous": n_vac, "n_infeasible": n_infeas, "n_usable": len(usable),
        "g1_ok": g1_ok, "g2_ok": g2_ok, "overall": overall,
        "rows": rows,
    }
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "loop_5_2_b.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  -> {os.path.join(out, 'loop_5_2_b.json')}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Loop Section 5.2(B) over an image folder to test budget-rule "
                    "generalization (reuses verify_5_2_b.verify).")
    ap.add_argument("--image-dir", required=True,
                    help="folder of images (e.g. benchmark_50)")
    ap.add_argument("--glob", default="*.JPEG",
                    help="filename glob within --image-dir (default *.JPEG)")
    ap.add_argument("--target", type=int, default=None,
                    help="pin target class for ALL images (default: per-image "
                         "model top-1)")
    ap.add_argument("--grid", type=int, nargs=2, default=(12, 12),
                    metavar=("GH", "GW"))
    ap.add_argument("--sigma", type=float, default=11.0)
    ap.add_argument("--beta-min", type=float, default=0.02,
                    help="trust threshold, held fixed across the set")
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
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most this many images (debug)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="print each per-image verify() block in full")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.image_dir, args.glob)))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no images matching "
                         f"{os.path.join(args.image_dir, args.glob)}")

    device = args.device
    model = load_backbone(device)
    class_names = get_class_names()

    loop(model, class_names, device, paths, args.target,
         grid=tuple(args.grid), sigma=args.sigma, beta_min=args.beta_min,
         K=args.K, c_lambda=args.c_lambda, c_est=args.c_est,
         C_budget=args.c_budget, n_pilot=args.n_pilot, sigma_obs=args.sigma_obs,
         edge_band=args.edge_band, transfer_tol=args.transfer_tol,
         seed=args.seed, batch_size=args.batch_size, out=args.out,
         verbose=args.verbose)


if __name__ == "__main__":
    main()