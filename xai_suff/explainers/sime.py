"""SIME: Second-order Interaction Model Explanations.

Extends grid LIME to degree-2 interactions between grid cells.

Pipeline (matches the verified support-recovery theory):
  1. SCREEN  -- fit main effects; keep cells with non-negligible effect
                (hierarchy assumption: a true pair has detectable main effects,
                 so screening on mains is lossless for cooperative / non-XOR f).
  2. RECOVER -- LASSO + stability selection over [mains + candidate pairs]
                restricted to screened cells. Surviving pair coefficients are
                the estimated high-order interactions. Sample complexity
                N >~ C * s * log p / gamma^2 (verified ~ C in [2,3]).
  3. SOLVE   -- weighted re-fit on the selected support (mains + pairs) for
                clean, de-biased coefficients.

Per-cell main effects paint a blocky 2D map (as in LIME). Pairwise
interactions cannot be a pixel value, so they are returned in
extras["interactions"] as (cell_i, cell_j, strength) triples.

NOTE on standardization: the +/-0.5 main columns have std ~0.5 while the
product (pair) columns have std ~0.25. A single Lasso lambda cannot be correct
for both blocks at once -- the small-variance pair columns get over-penalized
and die. We therefore standardize all design columns to unit norm before Lasso
(the theory's RE/incoherence conditions assume normalized design), run
selection on the standardized design, then de-bias by refitting on the RAW
columns in Stage 3 so reported Delta values stay in probability units.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LassoLars

from .base import AttributionResult, Explainer, blur_reference


class SIMEExplainer(Explainer):
    name = "sime"

    def __init__(self, *args, grid=(12, 12), n_samples=2500, sigma=11.0,
                 kernel_width=0.25, seed=0, batch_size=64,
                 screen_quantile=0.6, max_active_cells=40,
                 stability_runs=20, stability_thresh=0.6,
                 lasso_C=1.0, leak_c=1.0, **kw):
        """
        lasso_C  : constant C in lambda = C (sigma + c sqrt(m)) sqrt(log p / N),
                   applied to a UNIT-NORM standardized design. Paper's verified
                   support-recovery transition is C in [2,3] for the raw scale;
                   on the standardized design C ~ 1.0 is the matching point.
        leak_c   : constant c on the reference-leakage term sqrt(m_hat). c=1 uses
                   the held-out residual energy directly; raise to be more
                   conservative against off-manifold references.
        """
        super().__init__(*args, **kw)
        self.grid = grid
        self.n_samples = n_samples
        self.sigma = sigma
        self.kernel_width = kernel_width
        self.seed = seed
        self.batch_size = batch_size
        self.screen_quantile = screen_quantile
        self.max_active_cells = max_active_cells
        self.stability_runs = stability_runs
        self.stability_thresh = stability_thresh
        self.lasso_C = lasso_C
        self.leak_c = leak_c

    # ---- identical cell map to LIME ----
    def _cell_id_map(self, H, W):
        gh, gw = self.grid
        ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
        xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
        return ys.view(-1, 1) * gw + xs.view(1, -1)

    # ---- query f on a batch of on/off cell vectors ----
    @torch.no_grad()
    def _query(self, Z, x, b, cell_ids, target):
        probs = np.zeros(Z.shape[0], dtype=np.float64)
        for start in range(0, Z.shape[0], self.batch_size):
            zb = Z[start:start + self.batch_size].to(self.device)
            keep = zb[:, cell_ids].unsqueeze(1)            # (B,1,H,W)
            comp = keep * x + (1 - keep) * b
            p = F.softmax(self.model(comp), dim=1)[:, target]
            probs[start:start + zb.shape[0]] = p.detach().cpu().numpy()
        return probs

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        _, _, H, W = x.shape
        b = blur_reference(x, self.sigma)

        cell_ids = self._cell_id_map(H, W).to(self.device)
        n_cells = self.grid[0] * self.grid[1]

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        # Rademacher-style on/off design (Bernoulli-1/2), all-on baseline first.
        Z = (torch.rand(self.n_samples, n_cells, generator=g) > 0.5).float()
        Z[0] = 1.0
        probs = self._query(Z, x, b, cell_ids, target)
        Znp = Z.cpu().numpy()

        # distance-based weights, same kernel as LIME (used in final solve).
        all_on = np.ones(n_cells)
        d = 1.0 - (Znp @ all_on) / (
            np.linalg.norm(Znp, axis=1) * np.linalg.norm(all_on) + 1e-12)
        weights = np.exp(-(d ** 2) / (self.kernel_width ** 2))

        # ---------- STAGE 1: screen main effects ----------
        main_coefs = _weighted_ridge(Znp, probs, weights, alpha=1.0)
        mag = np.abs(main_coefs)
        thr = np.quantile(mag, self.screen_quantile)
        active = np.where(mag >= thr)[0]
        if active.size > self.max_active_cells:
            active = active[np.argsort(mag[active])[::-1][:self.max_active_cells]]
        active = np.sort(active)
        pair_list = list(combinations(active.tolist(), 2))

        # ---------- STAGE 2: LASSO support recovery + stability selection ----------
        Zc = (Znp - 0.5)                       # +/-0.5 coding, zero-mean features
        main_block = Zc[:, active]             # (N, m')
        pair_block = np.empty((self.n_samples, len(pair_list)))
        for c, (i, j) in enumerate(pair_list):
            pair_block[:, c] = Zc[:, i] * Zc[:, j]
        X = np.hstack([main_block, pair_block])
        m_act = active.size
        p_dim = X.shape[1]
        y = probs - probs.mean()

        # standardize columns to unit norm so a single lambda is correct for both
        # the mains block (std ~0.5) and the pairs block (std ~0.25).
        col_scale = X.std(axis=0)
        col_scale[col_scale < 1e-12] = 1.0
        Xn = X / col_scale

        sigma_hat = max(np.std(y), 1e-6)

        # --- Reference-SNR penalty (Theorem 1): lambda = C (sigma + c sqrt(m)) sqrt(log p / N)
        #     m_hat = held-out higher-order residual energy (Algorithm 1 proxy),
        #     the reference-induced misspecification term. Estimated WITHOUT
        #     recovering any degree-2 coefficient: unexplained variance of the
        #     degree-2 fit on a fresh split. Computed on the standardized design.
        m_hat = self._residual_energy_proxy(Xn, y, sigma_hat)
        lam = (self.lasso_C * (sigma_hat + self.leak_c * np.sqrt(m_hat))
               * np.sqrt(np.log(p_dim) / self.n_samples))

        def _stability(lmbda):
            rng = np.random.default_rng(self.seed)
            sc = np.zeros(p_dim)
            for _ in range(self.stability_runs):
                idx = rng.choice(self.n_samples, self.n_samples, replace=True)
                mm = LassoLars(alpha=lmbda, fit_intercept=True, max_iter=4000)
                mm.fit(Xn[idx], y[idx])
                sc += (np.abs(mm.coef_) > 0).astype(float)
            return sc / self.stability_runs

        stab = _stability(lam)
        selected = stab >= self.stability_thresh
        n_pairs = int(np.sum(selected[m_act:]))

        # NO auto-relax. If nothing clears the floor, that is the floor reporting
        # the truth (interactions below it are not guaranteed recoverable -- raise N
        # or choose a lower-m reference). Relaxing lambda voids the recovery guarantee.

        # single-fit count + below-threshold candidates, reported as DIAGNOSTICS only.
        full = LassoLars(alpha=lam, fit_intercept=True, max_iter=4000).fit(Xn, y)
        n_full = int(np.sum(np.abs(full.coef_[m_act:]) > 0))
        below = []
        for gi in range(m_act, p_dim):
            if full.coef_[gi] != 0.0 and not selected[gi]:
                ci, cj = pair_list[gi - m_act]
                # report coef back on raw scale for interpretability
                below.append((int(ci), int(cj),
                              float(full.coef_[gi] / col_scale[gi]),
                              float(stab[gi])))
        below.sort(key=lambda t: -abs(t[2]))

        floor = lam  # detection floor ~ lambda (standardized scale), for reporting
        print(f"[sime] sigma(y)={sigma_hat:.4f} m_hat={m_hat:.5f} "
              f"leak_frac={self.leak_c*np.sqrt(m_hat)/(sigma_hat+1e-12):.2f} "
              f"lam={lam:.5f} single-fit pairs={n_full} stable pairs={n_pairs}")
        if n_pairs == 0:
            print("[sime] no interaction pairs cleared the floor. This is a valid "
                  "result: at this N and reference, no pair beats lambda="
                  f"{lam:.5f}. Raise n_samples or use a lower-m reference (blur/"
                  "inpaint) to lower the floor. (Did NOT relax lambda.)")

        # ---------- STAGE 3: re-solve on selected support (RAW columns) ----------
        sel_idx = np.where(selected)[0]
        if sel_idx.size:
            Xs = X[:, sel_idx]                 # raw, unscaled -> Delta in prob units
            beta = _weighted_ridge(Xs, y, weights, alpha=1.0)
        else:
            beta = np.zeros(0)

        main_effect = np.zeros(n_cells)
        interactions = []
        for k, gi in enumerate(sel_idx):
            coef = beta[k]
            if gi < m_act:
                main_effect[active[gi]] = coef
            else:
                ci, cj = pair_list[gi - m_act]
                interactions.append((int(ci), int(cj), float(coef)))
        interactions.sort(key=lambda t: -abs(t[2]))

        coef_t = torch.tensor(main_effect, dtype=torch.float32, device=self.device)
        attr = coef_t[cell_ids].cpu().numpy()

        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=float(self._probs(x)[0, target]),
            extras={
                "grid": self.grid,
                "n_samples": self.n_samples,
                "n_active_cells": int(m_act),
                "candidate_pairs": len(pair_list),
                "candidate_dim_p": int(p_dim),
                "lambda": float(lam),
                "sigma_hat": float(sigma_hat),
                "m_hat": float(m_hat),                 # reference residual energy
                "detection_floor": float(floor),
                "interactions": interactions,          # (cell_i, cell_j, strength) -- recovered
                "below_floor": below,                  # diagnostics: single-fit, NOT stability-confirmed
                "interaction_stability": {
                    f"{pair_list[gi - m_act][0]}-{pair_list[gi - m_act][1]}": float(stab[gi])
                    for gi in sel_idx if gi >= m_act
                },
            },
        )

    # ------------------------------------------------------------------ #
    # Algorithm 1 held-out residual-energy proxy for m_{>2,rho}.
    # m_hat = E_val[(y - g_hat_{<=2})^2] - sigma^2, clipped at 0.
    # Estimates higher-order residual energy WITHOUT recovering any
    # higher-order coefficient -- it is unexplained variance of the
    # degree-2 fit measured on a fresh split. Upward bias (finite-sample
    # error in g_hat) is the conservative direction for a detection floor.
    # Operates on the standardized design Xn passed in by explain().
    # ------------------------------------------------------------------ #
    def _residual_energy_proxy(self, X, y, sigma_hat):
        n = X.shape[0]
        rng = np.random.default_rng(self.seed + 1)
        perm = rng.permutation(n)
        n_tr = n // 2
        tr, va = perm[:n_tr], perm[n_tr:]
        Xtr = np.concatenate([X[tr], np.ones((tr.size, 1))], axis=1)
        A = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
        A[-1, -1] = 0.0
        coef = np.linalg.solve(A, Xtr.T @ y[tr])
        Xva = np.concatenate([X[va], np.ones((va.size, 1))], axis=1)
        resid = y[va] - Xva @ coef
        m_hat = float(np.mean(resid ** 2) - sigma_hat ** 2)
        return max(m_hat, 0.0)


def _weighted_ridge(Z, y, w, alpha=1.0):
    """Closed-form weighted ridge; returns per-feature coefficients (no intercept)."""
    n, d = Z.shape
    Zb = np.concatenate([Z, np.ones((n, 1))], axis=1)
    Wd = w[:, None]
    A = Zb.T @ (Wd * Zb)
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0
    A += reg
    b = Zb.T @ (w * y)
    sol = np.linalg.solve(A, b)
    return sol[:-1]