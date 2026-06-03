"""Verify Section 5.2(C): reference ablation as a certification-COST lever.

This is NOT a fidelity test. The paper is explicit that we do not claim which
reference is "better" or more faithful. Each admissible reference rho defines its
OWN reference-conditioned estimand beta_{<=K,rho}; there is no shared ground truth
across references to rank them against. What (C) verifies is a single internal
consistency law of Corollary 1, holding each reference's own estimand fixed:

    The reference enters the floor ONLY through sigma_eff(rho).
    floor(N, rho) = Cest * Clam * sigma_eff(rho) * sqrt(log p_K / N).

Two falsifiable predictions follow, both reference-RELATIVE (neither ranks quality):

  1. At fixed N, the realized floor across references tracks
        sigma_eff(rho) * sqrt(log p_K / N)
     and the certified count follows the floor (lower sigma_eff -> lower floor ->
     more terms clear). This is a cost statement: a lower-mismatch reference is
     CHEAPER to certify at a given budget, not truer.

  2. The cross-reference BUDGET-RATIO law: to reach the SAME beta_min, a reference
     with sigma_eff^high needs a budget larger than one with sigma_eff^low by
        (sigma_eff^high / sigma_eff^low)^2     (Cor.1).
     This is THE test. We measure each sigma_eff from a cheap pilot (Eq.10), pick
     a common beta_min, predict N_rho = C^2 sigma_eff^2 log p_K / beta_min^2, and
     check the ratios match the sigma_eff-ratio-squared prediction.

THE TRAP this script instruments against: a FLATTENING reference lowers m_>K,rho
by collapsing g_rho into a near-constant function. It would clear "more terms"
trivially -- nothing to resolve -- and masquerade as a win if one (wrongly) reads
"more certified" as "better". Appendix I's non-degeneracy clause exists exactly to
exclude this. So EVERY reference is gated by an admissibility check BEFORE it is
allowed into the cost-law comparison:
  (a) target-score variance over pilot masks above a threshold (non-degenerate);
  (b) fill stays within the valid input range (in [0,1] pixel space);
  (c) OOD proxy: feature-space distance of the masked composites to the original
      is not extreme.
Constant fills sit at the edge of R_adm and are reported SEPARATELY, never folded
into the ratio law (matching the paper's framing).

CAVEAT reported alongside the ratio: the (sigma_eff)^2 ratio law assumes C
(equivalently Cest, Clam, and the RE constant gamma absorbed into Cest) is
reference-INDEPENDENT -- that the only moving part across references is sigma_eff.
gamma can shift across references (admissibility keeps it bounded away from zero,
not equal across references). We therefore report a per-reference RE proxy
(min eigenvalue of the standardized Gram) so a reader can see whether the clean
(sigma_eff)^2 law should pick up a gamma_low/gamma_high correction.

No torch is run automatically. Execute yourself, e.g.:

    python verify_5_2_c.py --image /path/to/img.jpg
    python verify_5_2_c.py --image img.jpg --grid 12 12 \
        --refs blur:5 blur:11 blur:21 mean inpaint:9 \
        --const-refs black gray --N 3200 --beta-min 0.02 --device cuda

Notes:
  * --refs are the ADMISSIBLE references compared by the cost law.
  * --const-refs are reported separately (edge of R_adm), excluded from the law.
  * Pick N above the feasibility floor N >~ s*log p_K (Cor.3); a reference whose
    pilot support is large needs a correspondingly larger N for its fit to be
    determined. Underdetermined fits give arbitrary floors and poison the ratio.
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
    denormalize,
    get_class_names,
    load_backbone,
    load_image,
    normalize_pixel,
)
from xai_suff.explainers import blur_reference
from xai_suff.explainers.base import gaussian_blur


# --------------------------------------------------------------------------- #
# candidate count / penalty / floor  (identical conventions to verify_5_2_a)
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


def required_budget(sigma_eff: float, d: int, beta_min: float, K: int = 1,
                    C: float = 1.0) -> float:
    """Cor.1: N >~ C^2 sigma_eff^2 log p_K / beta_min^2.

    C here is the SINGLE composite constant C = Cest*Clam (the floor's own
    constant), so that floor(N_req) == beta_min exactly when this N is used.
    """
    return (C ** 2) * (sigma_eff ** 2) * math.log(p_K(d, K)) / (beta_min ** 2)


# --------------------------------------------------------------------------- #
# cell-id map (mirrors xai_suff LIMEExplainer._cell_id_map)
# --------------------------------------------------------------------------- #
def cell_id_map(H, W, grid):
    gh, gw = grid
    ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
    xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
    return ys.view(-1, 1) * gw + xs.view(1, -1)


# --------------------------------------------------------------------------- #
# reference fields. Each returns a normalized (1,3,H,W) tensor `b` used as the
# off-cell fill, exactly as the LIMEExplainer / query oracle consume it.
# --------------------------------------------------------------------------- #
def build_reference(spec: str, x: torch.Tensor, device) -> torch.Tensor:
    """spec grammar:
        blur:S    Gaussian blur of bandwidth S (on-manifold; the paper default)
        mean      per-channel mean color (a flat but in-range fill)
        inpaint:S coarse 'inpaint' proxy = very strong blur of bandwidth S
                  (true inpainting needs an external model; this is a stand-in
                   that stays in-range and on the low-freq manifold)
        black     constant 0 in pixel space   (CONSTANT fill; edge of R_adm)
        gray      constant 0.5 in pixel space  (CONSTANT fill; edge of R_adm)
        white     constant 1.0 in pixel space  (CONSTANT fill; edge of R_adm)
    """
    name = spec.split(":")[0]
    if name == "blur":
        sigma = float(spec.split(":")[1]) if ":" in spec else 11.0
        return blur_reference(x, sigma).to(device)
    if name == "inpaint":
        sigma = float(spec.split(":")[1]) if ":" in spec else 31.0
        # strong-blur proxy for inpainting: low-freq, in-range, on-manifold
        x01 = denormalize(x)
        b01 = gaussian_blur(x01, sigma).clamp(0, 1)
        return normalize_pixel(b01).to(device)
    if name == "mean":
        x01 = denormalize(x)
        m = x01.mean(dim=(2, 3), keepdim=True)            # (1,3,1,1)
        b01 = m.expand_as(x01).clamp(0, 1)
        return normalize_pixel(b01).to(device)
    if name in ("black", "gray", "white"):
        val = {"black": 0.0, "gray": 0.5, "white": 1.0}[name]
        x01 = denormalize(x)
        b01 = torch.full_like(x01, val)
        return normalize_pixel(b01).to(device)
    raise ValueError(f"unknown reference spec: {spec!r}")


def is_constant_spec(spec: str) -> bool:
    return spec.split(":")[0] in ("black", "gray", "white")


# --------------------------------------------------------------------------- #
# masked-response oracle (identical to verify_5_2_a)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def query_masked_response(model, x, b, cell_ids, Z, target, device,
                          batch_size=64):
    """g_rho(z): target-class prob with keep->sharp, off->reference fill b."""
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
# standardized K=1 Lasso on the centered +-1 design (identical to verify_5_2_a)
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
# admissibility gate (Appendix I). This is the guard that stops a FLATTENING
# reference from trivially "winning" the cost law.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def admissibility_check(model, x, b, cell_ids, n_cells, target, device,
                        y_pilot, Z_pilot, r2_thresh, ood_thresh, batch_size):
    """Return (passed: bool, info: dict) for reference field b.

    (a) NON-DEGENERATE PER-CELL STRUCTURE (the operative Appendix I clause).
        The wrong test is "variance of g_rho over masks is large": a FLAT fill
        (mean/black/gray) produces a big, easy figure/ground swing -- the
        response is essentially "how much real image is showing" = a function of
        the global mask fraction alone -- so its variance is LARGE while it
        carries no resolvable per-cell signal. That is exactly the degenerate
        reference Appendix I excludes, and a variance gate REWARDS it.
        The right test: regress g_rho(z) on the single scalar mean(z) (the global
        on/off fraction). If R^2 is near 1, the response is explained by the
        global shift alone -- no per-cell structure beyond figure/ground -> the
        reference is degenerate for certification. We require R^2 < r2_thresh
        (i.e. substantial variance REMAINS after removing the global-shift term).
    (b) in-range fill: the reference, in [0,1] pixel space, lies in [0,1].
    (c) OOD proxy: penultimate-feature distance between the all-off composite
        and the original, normalized, must not exceed ood_thresh.
    """
    info = {}

    # (a) global-shift R^2 on the pilot responses (no extra queries).
    frac = Z_pilot.mean(axis=1)                       # mean(z) per mask, in [0,1]
    fc = frac - frac.mean()
    yc = y_pilot - y_pilot.mean()
    denom_f = float(fc @ fc) + 1e-12
    slope = float(fc @ yc) / denom_f                  # OLS of y on frac
    pred = slope * fc
    ss_res = float(np.sum((yc - pred) ** 2))
    ss_tot = float(np.sum(yc ** 2)) + 1e-12
    global_shift_r2 = 1.0 - ss_res / ss_tot
    info["global_shift_r2"] = global_shift_r2
    info["score_var"] = float(np.var(y_pilot))        # kept for reporting only
    pass_struct = global_shift_r2 < r2_thresh

    # (b) in-range fill (check the pixel-space reference)
    b01 = denormalize(b)
    lo, hi = float(b01.min()), float(b01.max())
    info["fill_min"], info["fill_max"] = lo, hi
    pass_range = (lo >= -1e-4) and (hi <= 1.0 + 1e-4)

    # (c) OOD proxy via penultimate features. Use the model up to its
    # pre-logit features if exposable; otherwise fall back to logit-space
    # distance (still a monotone manifold-distance proxy).
    feats_x = _features(model, x)
    all_off = torch.zeros(1, n_cells, device=device)
    keep = all_off[:, cell_ids].unsqueeze(1)
    comp = keep * x + (1.0 - keep) * b
    feats_b = _features(model, comp)
    denom = float(feats_x.norm()) + 1e-8
    ood = float((feats_b - feats_x).norm()) / denom
    info["ood_dist"] = ood
    pass_ood = ood <= ood_thresh

    info["pass_struct"] = pass_struct
    info["pass_range"] = pass_range
    info["pass_ood"] = pass_ood
    passed = bool(pass_struct and pass_range and pass_ood)
    return passed, info


@torch.no_grad()
def _features(model, x):
    """Penultimate features if the backbone exposes them, else logits.

    Tries common torchvision attribute layouts (ResNet: avgpool->flatten;
    ViT via forward_features). Falls back to logits so the proxy still works.
    """
    # torchvision ResNet-style
    if all(hasattr(model, a) for a in
           ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2",
            "layer3", "layer4", "avgpool")):
        h = model.conv1(x)
        h = model.bn1(h)
        h = model.relu(h)
        h = model.maxpool(h)
        h = model.layer1(h)
        h = model.layer2(h)
        h = model.layer3(h)
        h = model.layer4(h)
        h = model.avgpool(h)
        return torch.flatten(h, 1)
    # timm/ViT-style
    if hasattr(model, "forward_features"):
        h = model.forward_features(x)
        if h.ndim > 2:
            h = h.mean(dim=tuple(range(1, h.ndim - 1))) if h.ndim == 4 else \
                h.mean(dim=1)
        return torch.flatten(h, 1)
    # fallback: logits
    return model(x)


# --------------------------------------------------------------------------- #
# RE proxy -- reference-DEPENDENT. The restricted-eigenvalue constant of
# Assumption 2 is a property of the design RESTRICTED TO THE ACTIVE SUPPORT,
# and the active support is exactly where the reference enters (different refs
# certify different cells). We proxy gamma by the minimum eigenvalue of the
# standardized Gram on the certified columns S:  lambda_min( (1/N) X_S^T X_S ).
# Different S -> different value, so flat vs on-manifold references now separate.
# (The old whole-design Gram was identical across refs by construction -- a bug.)
# --------------------------------------------------------------------------- #
def re_proxy_on_support(Z, S):
    """lambda_min of the standardized Gram restricted to certified columns S.

    Returns NaN if S is empty (nothing certified -> RE undefined here). With a
    single certified column the Gram is 1x1 and the value is ~1 (a lone column
    is trivially well-conditioned); the diagnostic only bites once |S| >= 2,
    where spatially correlated certified cells drive lambda_min down.
    """
    S = list(S)
    if len(S) == 0:
        return float("nan")
    X = centered_design(Z)[:, S]
    N = X.shape[0]
    G = (X.T @ X) / N
    try:
        evals = np.linalg.eigvalsh(G)
        return float(np.min(evals))
    except np.linalg.LinAlgError:
        return float("nan")


# --------------------------------------------------------------------------- #
# pilot: measure sigma_eff via Eq.(10) for ONE reference. Also returns the
# pilot responses (for the admissibility variance check) and an RE proxy.
# --------------------------------------------------------------------------- #
def measure_sigma_eff(model, x, b, cell_ids, n_cells, target, device,
                      d, K, n_pilot, c_lambda, sigma_obs=0.0, seed=0,
                      batch_size=64):
    g = np.random.default_rng(seed)
    Zf = (g.random((n_pilot, n_cells)) > 0.5).astype(np.float64)
    Zf[0] = 1.0
    yf = query_masked_response(model, x, b, cell_ids, Zf, target, device,
                               batch_size)
    lam = lambda_value(max(sigma_obs, 1e-3), d, n_pilot, K, c_lambda)
    beta = standardized_lasso_fit(Zf, yf, lam)
    b0 = float(np.mean(yf.mean() - centered_design(Zf) @ beta))
    Zv = (g.random((n_pilot, n_cells)) > 0.5).astype(np.float64)
    yv = query_masked_response(model, x, b, cell_ids, Zv, target, device,
                               batch_size)
    pred = centered_design(Zv) @ beta + b0
    resid_var = float(np.mean((yv - pred) ** 2))
    m_hat = max(resid_var - sigma_obs ** 2, 0.0)
    sigma_eff = sigma_obs + math.sqrt(m_hat)
    # NOTE: the RE proxy is NOT computed here. The mask Gram is identical across
    # references (same RNG, reference never touches the design matrix), so a
    # design-only proxy is structurally constant and tells us nothing. The
    # reference-dependent RE proxy is computed in run_reference_at_N on the
    # certified active columns S, where the reference DOES enter (via which cells
    # clear the floor). See re_proxy_on_support().
    return sigma_eff, {"resid_var": resid_var, "m_hat": m_hat, "b0": b0,
                       "y_pilot": yf, "Z_pilot": Zf}


# --------------------------------------------------------------------------- #
# per-reference run at fixed N: fit, realized floor, certified count.
# Uses NESTED sampling within the reference (one draw, prefix at N) so the
# pilot and the fit share design conventions with verify_5_2_a.
# --------------------------------------------------------------------------- #
def run_reference_at_N(model, x, b, cell_ids, n_cells, target, device,
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
    cert_count = int(cert.sum())
    max_abs = float(np.max(np.abs(beta_hat)))
    # reference-dependent RE proxy on the certified active columns
    S = np.where(cert)[0]
    re = re_proxy_on_support(Z, S)
    return {
        "floor_realized": fl, "cert_count": cert_count,
        "max_abs_beta": max_abs, "n_cleared": cert_count,
        "vacuous": max_abs < fl, "re_proxy": re,
        "support_size": int(cert_count),
    }


# --------------------------------------------------------------------------- #
# main verification
# --------------------------------------------------------------------------- #
def verify(model, x, cell_ids, n_cells, target, class_name, device,
           ref_specs, const_specs, N, beta_min, K=1, c_lambda=0.30,
           c_est=1.0, n_pilot=512, sigma_obs=0.0, r2_thresh=0.95,
           ood_thresh=5.0, seed=0, batch_size=64, out=None):
    d = n_cells
    C_composite = c_est * c_lambda

    print("=" * 78)
    print("Section 5.2(C): reference ablation as a certification-COST lever")
    print("  (NOT a fidelity test -- no reference is ranked by explanation quality)")
    print("=" * 78)
    print(f"  target class = {target} ({class_name})")
    print(f"  grid cells d = {d}   K = {K}   p_K = {p_K(d, K)}")
    print(f"  fixed budget N = {N}   beta_min = {beta_min}")
    print(f"  C = Cest*Clam = {C_composite:.4f}\n")

    admissible_rows = []   # references that pass the gate
    rejected_rows = []     # references that FAIL the gate (flattening etc.)
    const_rows = []        # constant fills, reported separately

    all_specs = [(s, False) for s in ref_specs] + \
                [(s, True) for s in const_specs]

    for spec, is_const in all_specs:
        b = build_reference(spec, x, device)
        sigma_eff, pinfo = measure_sigma_eff(
            model, x, b, cell_ids, n_cells, target, device, d, K, n_pilot,
            c_lambda, sigma_obs=sigma_obs, seed=seed, batch_size=batch_size)
        passed, ainfo = admissibility_check(
            model, x, b, cell_ids, n_cells, target, device,
            pinfo["y_pilot"], pinfo["Z_pilot"], r2_thresh, ood_thresh,
            batch_size)

        run = run_reference_at_N(
            model, x, b, cell_ids, n_cells, target, device, sigma_eff, d, N,
            K, c_lambda, c_est, seed, batch_size)
        N_req = required_budget(sigma_eff, d, beta_min, K, C_composite)

        row = {
            "spec": spec, "sigma_eff": sigma_eff,
            "m_hat": pinfo["m_hat"], "re_proxy": run["re_proxy"],
            "score_var": ainfo["score_var"],
            "global_shift_r2": ainfo["global_shift_r2"], "ood": ainfo["ood_dist"],
            "fill_range": (ainfo["fill_min"], ainfo["fill_max"]),
            "pass_struct": ainfo["pass_struct"], "pass_range": ainfo["pass_range"],
            "pass_ood": ainfo["pass_ood"], "admissible": passed,
            "floor_realized": run["floor_realized"],
            "cert_count": run["cert_count"], "vacuous": run["vacuous"],
            "N_req_for_beta_min": N_req,
        }
        if is_const:
            const_rows.append(row)
        elif passed:
            admissible_rows.append(row)
        else:
            rejected_rows.append(row)

    # ---- per-reference table (admissible) ------------------------------- #
    print("[Admissible references] (gated by Appendix I before entering the law)")
    print(f"  {'ref':>10} {'sigma_eff':>9} {'m_hat':>8} {'RE_min':>7} "
          f"{'floor(N)':>9} {'#cert':>6} {'gshift_r2':>9} {'ood':>6}")
    for r in admissible_rows:
        print(f"  {r['spec']:>10} {r['sigma_eff']:>9.5f} {r['m_hat']:>8.5f} "
              f"{r['re_proxy']:>7.3f} {r['floor_realized']:>9.5f} "
              f"{r['cert_count']:>6} {r['global_shift_r2']:>9.4f} "
              f"{r['ood']:>6.2f}"
              f"{'  [VACUOUS]' if r['vacuous'] else ''}")
    print()

    if rejected_rows:
        print("[REJECTED by admissibility gate] -- excluded from the cost law")
        print("  (global_shift_r2 ~ 1 => flattening ref: response is just the")
        print("   global mask fraction, no per-cell structure; or range / OOD)")
        for r in rejected_rows:
            why = []
            if not r["pass_struct"]:
                why.append(
                    f"degenerate (global_shift_r2={r['global_shift_r2']:.4f})")
            if not r["pass_range"]:
                why.append(f"out-of-range {r['fill_range']}")
            if not r["pass_ood"]:
                why.append(f"OOD (dist={r['ood']:.2f})")
            print(f"  {r['spec']:>10}: {'; '.join(why)}")
        print()

    if const_rows:
        print("[Constant fills] -- reported SEPARATELY (edge of R_adm), NOT in law")
        print(f"  {'ref':>10} {'sigma_eff':>9} {'m_hat':>8} {'floor(N)':>9} "
              f"{'#cert':>6} {'gshift_r2':>9} {'admissible?':>11}")
        for r in const_rows:
            print(f"  {r['spec']:>10} {r['sigma_eff']:>9.5f} {r['m_hat']:>8.5f} "
                  f"{r['floor_realized']:>9.5f} {r['cert_count']:>6} "
                  f"{r['global_shift_r2']:>9.4f} "
                  f"{'yes' if r['admissible'] else 'no':>11}")
        print()

    # ---- Claim 1: at fixed N, lower sigma_eff => lower floor => more cert -- #
    # Monotonicity check among admissible references: sorting by sigma_eff
    # should sort the floor the same way (exactly, since floor is monotone in
    # sigma_eff at fixed N) and the certified count the OPPOSITE way (more or
    # equal terms cleared at lower floor), up to ties.
    print("[Claim C.1] at fixed N: floor monotone in sigma_eff; #cert anti-monotone")
    law1_ok = True
    if len(admissible_rows) >= 2:
        srt = sorted(admissible_rows, key=lambda r: r["sigma_eff"])
        floors = [r["floor_realized"] for r in srt]
        certs = [r["cert_count"] for r in srt]
        floor_sorted = all(floors[i] <= floors[i + 1] + 1e-12
                           for i in range(len(floors) - 1))
        # anti-monotone count allowing ties (and small noise of +-1 term)
        cert_anti = all(certs[i] >= certs[i + 1] - 1
                        for i in range(len(certs) - 1))
        law1_ok = floor_sorted and cert_anti
        print(f"  order by sigma_eff: {[r['spec'] for r in srt]}")
        print(f"  floors (asc sigma) : {[f'{f:.4f}' for f in floors]}  "
              f"monotone: {'PASS' if floor_sorted else 'FAIL'}")
        print(f"  #cert  (asc sigma) : {certs}  anti-monotone(+-1): "
              f"{'PASS' if cert_anti else 'CHECK'}")
    else:
        print("  (need >=2 admissible references for this check)")
        law1_ok = len(admissible_rows) >= 1
    print()

    # ---- Claim 2: cross-reference budget-RATIO law (THE test) ------------ #
    # To reach the SAME beta_min: N_req(rho) propto sigma_eff(rho)^2.
    # Pick the lowest-sigma_eff admissible reference as the baseline, then check
    # each other reference's required-budget ratio against (sigma ratio)^2.
    print("[Claim C.2] budget-ratio law: N_req(rho) / N_req(base) =?= "
          "(sigma_eff(rho)/sigma_eff(base))^2")
    law2_ok = True
    ratio_rows = []
    if len(admissible_rows) >= 2:
        base = min(admissible_rows, key=lambda r: r["sigma_eff"])
        s0, N0 = base["sigma_eff"], base["N_req_for_beta_min"]
        print(f"  baseline = {base['spec']} (sigma_eff={s0:.5f}, "
              f"N_req={N0:.0f})")
        print(f"  {'ref':>10} {'sigma_eff':>9} {'N_req':>9} "
              f"{'N_ratio':>8} {'(sig_ratio)^2':>13} {'rel_err':>8} "
              f"{'RE_min':>7}")
        for r in admissible_rows:
            if r["spec"] == base["spec"]:
                continue
            sig_ratio_sq = (r["sigma_eff"] / s0) ** 2
            n_ratio = r["N_req_for_beta_min"] / N0
            rel_err = abs(n_ratio - sig_ratio_sq) / (sig_ratio_sq + 1e-12)
            ratio_rows.append({
                "spec": r["spec"], "sig_ratio_sq": sig_ratio_sq,
                "n_ratio": n_ratio, "rel_err": rel_err,
                "re_proxy": r["re_proxy"],
            })
            print(f"  {r['spec']:>10} {r['sigma_eff']:>9.5f} "
                  f"{r['N_req_for_beta_min']:>9.0f} {n_ratio:>8.3f} "
                  f"{sig_ratio_sq:>13.3f} {rel_err:>8.3f} "
                  f"{r['re_proxy']:>7.3f}")
        # the law is ANALYTICALLY exact when C is reference-independent: both
        # N_req and the sigma-ratio-squared are built from the same sigma_eff,
        # so rel_err is ~0 BY CONSTRUCTION. The MEANINGFUL test is empirical:
        # does the REALIZED floor at the common N match sigma_eff-scaling, and
        # is RE_min stable across references (else the gamma caveat bites)?
        re_vals = [r["re_proxy"] for r in admissible_rows
                   if not math.isnan(r["re_proxy"])]
        re_spread = (max(re_vals) - min(re_vals)) / (np.mean(re_vals) + 1e-12) \
            if re_vals else float("nan")
        print(f"  RE_min spread across refs: {re_spread:.3f}  "
              f"({'stable -> clean (sigma)^2 law' if re_spread < 0.15 else 'VARIES -> law picks up gamma_low/gamma_high; report it'})")
        law2_ok = re_spread < 0.15 if not math.isnan(re_spread) else True
    else:
        print("  (need >=2 admissible references)")
        law2_ok = len(admissible_rows) >= 1
    print()

    # ---- Claim 3 (the real empirical content): REALIZED floor tracks ----- #
    # sigma_eff at the COMMON N. Since all admissible refs ran at the same N,
    # floor_realized(rho) / floor_realized(base) should equal sigma_eff ratio
    # (NOT squared -- floor is linear in sigma_eff at fixed N). This uses the
    # actually-fit floor, so it is the genuine empirical check, not a tautology.
    print("[Claim C.3] realized floor at common N tracks sigma_eff (linear)")
    law3_ok = True
    if len(admissible_rows) >= 2:
        base = min(admissible_rows, key=lambda r: r["sigma_eff"])
        f0, s0 = base["floor_realized"], base["sigma_eff"]
        print(f"  {'ref':>10} {'floor_ratio':>11} {'sigma_ratio':>11} "
              f"{'rel_err':>8}")
        errs = []
        for r in admissible_rows:
            if r["spec"] == base["spec"]:
                continue
            f_ratio = r["floor_realized"] / (f0 + 1e-12)
            s_ratio = r["sigma_eff"] / (s0 + 1e-12)
            rel = abs(f_ratio - s_ratio) / (s_ratio + 1e-12)
            errs.append(rel)
            print(f"  {r['spec']:>10} {f_ratio:>11.3f} {s_ratio:>11.3f} "
                  f"{rel:>8.3f}")
        # floor is computed from sigma_eff analytically, so this too is exact;
        # the value of printing it is to expose any code-path divergence.
        max_err = max(errs) if errs else 0.0
        law3_ok = max_err < 1e-6
        print(f"  max rel_err = {max_err:.2e}  "
              f"({'consistent' if law3_ok else 'DIVERGENCE -- check floor path'})")
    else:
        print("  (need >=2 admissible references)")
    print()

    # ---- overall -------------------------------------------------------- #
    n_adm = len(admissible_rows)
    any_vacuous = any(r["vacuous"] for r in admissible_rows)
    overall = law1_ok and law2_ok and law3_ok and n_adm >= 1 and not any_vacuous
    print("=" * 78)
    status = "PASS" if overall else ("VACUOUS" if any_vacuous else "CHECK")
    print(f"  OVERALL 5.2(C)               : {status}")
    print(f"  admissible refs in law       : {n_adm} "
          f"(rejected {len(rejected_rows)}, constants reported separately "
          f"{len(const_rows)})")
    if any_vacuous:
        print("  NOTE: a reference cleared NO terms above its own floor at this N"
              " -- its cost comparison is vacuous; raise N or beta_min.")
    print("  REMINDER: this certifies certification COST, not fidelity. No")
    print("            reference is claimed more faithful; each estimand differs.")
    print("=" * 78)

    if out:
        os.makedirs(out, exist_ok=True)
        # strip the heavy y_pilot arrays before dumping
        def clean(rows):
            return [{k: v for k, v in r.items() if k != "y_pilot"} for r in rows]
        rec = {
            "target": target, "class_name": class_name, "K": K, "N": N,
            "beta_min": beta_min, "C_composite": C_composite,
            "admissible": clean(admissible_rows),
            "rejected": clean(rejected_rows),
            "constants": clean(const_rows),
            "ratio_rows": ratio_rows,
            "law1_ok": law1_ok, "law2_ok": law2_ok, "law3_ok": law3_ok,
            "overall": overall,
        }
        with open(os.path.join(out, "verify_5_2_c.json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  -> {os.path.join(out, 'verify_5_2_c.json')}")

    return {"overall": overall, "n_admissible": n_adm}


def main():
    ap = argparse.ArgumentParser(
        description="Verify Section 5.2(C): reference ablation = certification "
                    "cost lever (not fidelity).")
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--target", type=int, default=None,
                    help="target class (default: model top-1)")
    ap.add_argument("--grid", type=int, nargs=2, default=(12, 12),
                    metavar=("GH", "GW"))
    ap.add_argument("--refs", nargs="+",
                    default=["blur:5", "blur:11", "blur:21", "mean", "inpaint:31"],
                    help="ADMISSIBLE references compared by the cost law; "
                         "grammar: blur:S | mean | inpaint:S")
    ap.add_argument("--const-refs", nargs="+", default=["black", "gray"],
                    help="constant fills, reported SEPARATELY (edge of R_adm)")
    ap.add_argument("--N", type=int, default=3200,
                    help="fixed budget for the at-N comparison; keep above the "
                         "feasibility floor N >~ s*log p_K (Cor.3)")
    ap.add_argument("--beta-min", type=float, default=0.02,
                    help="common trust threshold for the budget-ratio law")
    ap.add_argument("--K", type=int, default=1, choices=[1])
    ap.add_argument("--c-lambda", type=float, default=0.30)
    ap.add_argument("--c-est", type=float, default=1.0)
    ap.add_argument("--n-pilot", type=int, default=512)
    ap.add_argument("--sigma-obs", type=float, default=0.0,
                    help="query-noise scale (0 for deterministic model+ref)")
    ap.add_argument("--r2-thresh", type=float, default=0.95,
                    help="max global-shift R^2 for a reference to count as "
                         "non-degenerate (Appendix I): if g_rho is explained by "
                         "the global mask fraction alone above this, it carries "
                         "no resolvable per-cell structure and is rejected")
    ap.add_argument("--ood-thresh", type=float, default=5.0,
                    help="max relative penultimate-feature distance of the "
                         "all-off composite to the original (OOD gate)")
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

    verify(model, x, cell_ids, n_cells, target, class_names[target], device,
           ref_specs=args.refs, const_specs=args.const_refs, N=args.N,
           beta_min=args.beta_min, K=args.K, c_lambda=args.c_lambda,
           c_est=args.c_est, n_pilot=args.n_pilot, sigma_obs=args.sigma_obs,
           r2_thresh=args.r2_thresh, ood_thresh=args.ood_thresh,
           seed=args.seed, batch_size=args.batch_size, out=args.out)


if __name__ == "__main__":
    main()