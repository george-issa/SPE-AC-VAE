"""
threshold_scan.py — Phase 1+3 diagnostic for the chi^2 < 1 anomaly.

Re-evaluates a trained finetune checkpoint's chi^2 under a sweep of
PCA-truncation thresholds v, and reports the per-mode variance ratio
r_k = <eta_k^2> / w_k where eta_k is the residual projected onto the
k-th eigenmode of the data covariance C.

The residuals (G_recon - G_input) are loaded from summary.pt — no
forward pass needed, no model load.

The whitener formula matches ChiSquaredLoss(covariance_estimator='pca_truncated')
in pretrain_losses.py: for each v, keep the top n eigenmodes covering
>= v * tr(C), build the truncated pseudo-inverse-square-root, multiply by
sqrt(L_tau / n_kept). At eta ~ N(0, C) this gives E[chi^2] = 1, independent
of v (assumes residuals are well-aligned with the kept modes).

Usage:
  python pretrain/threshold_scan.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process_real import load_covariance_v2  # type: ignore  # noqa: E402
from pretrain.synthetic_data import load_covariance_from_dqmc  # type: ignore  # noqa: E402


PLOT_RC = {
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 120,
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
}

V_GRID = (0.50, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 1.0)


def load_residuals_and_C(run_dir: str):
    """Load eta = G_recon - G_input (numpy, shape (N, L_tau)) and C (L_tau, L_tau)."""
    summary = torch.load(os.path.join(run_dir, "summary.pt"), weights_only=False)
    G_in   = summary["inputs"].numpy().astype(np.float64)   # (N, L_tau)
    G_rec  = summary["recon"].numpy().astype(np.float64)    # (N, L_tau)
    eta    = G_rec - G_in

    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)

    if params["DATA_SOURCE"] == "real":
        C = load_covariance_v2(params["DATA_PATH"], r1=0, r2=0)
    else:
        C = load_covariance_from_dqmc(params["DATA_PATH"])
    C = np.asarray(C, dtype=np.float64)

    return eta, C, params, summary


def chi2_pca_truncated(eta: np.ndarray, w: np.ndarray, V: np.ndarray, v: float):
    """Replicate ChiSquaredLoss('pca_truncated') chi^2 at threshold v.

    eta : (N, L_tau)   residuals
    w   : (L_tau,)     eigenvalues (clipped >= 0; ANY ORDER ok — sorted internally)
    V   : (L_tau, L_tau)  matching eigenvectors as columns

    Returns (chi2_mean, n_kept).
    """
    L_tau = w.shape[0]

    idx       = np.argsort(w)[::-1]
    w_sorted  = w[idx]
    V_sorted  = V[:, idx]

    cumsum = np.cumsum(w_sorted)
    n_kept = int(np.searchsorted(cumsum, v * cumsum[-1]) + 1)
    n_kept = min(n_kept, L_tau)

    w_trunc = w_sorted[:n_kept]
    V_trunc = V_sorted[:, :n_kept]

    eps = 1e-12
    inv_sqrt_C = (
        V_trunc @ np.diag(1.0 / np.sqrt(w_trunc + eps)) @ V_trunc.T
        * np.sqrt(L_tau / n_kept)
    )
    eta_white = eta @ inv_sqrt_C                  # (N, L_tau)
    chi2_per  = np.sum(eta_white ** 2, axis=1) / L_tau
    return float(chi2_per.mean()), n_kept


def threshold_scan(eta: np.ndarray, C: np.ndarray, v_grid=V_GRID):
    """Run the chi^2 vs v scan. Returns dict with the scan and the per-mode r_k."""
    w_raw, V = np.linalg.eigh(C)        # ascending
    w = np.maximum(w_raw, 0.0)

    # per-mode variance ratio r_k: project eta onto eigenbasis (columns of V),
    # report ratio of empirical to true variance, ordered by descending w_k.
    eta_modes = eta @ V                 # (N, L_tau), in ascending-w basis
    var_emp   = (eta_modes ** 2).mean(axis=0)
    eps = 1e-30
    r_ascending = var_emp / (w + eps)

    desc = np.argsort(w)[::-1]
    w_desc       = w[desc]
    r_desc       = r_ascending[desc]
    var_emp_desc = var_emp[desc]
    cum_var      = np.cumsum(w_desc) / np.sum(w_desc)

    chi2_vals, n_kept_vals = [], []
    for v in v_grid:
        c, n = chi2_pca_truncated(eta, w, V, v)
        chi2_vals.append(c)
        n_kept_vals.append(n)

    return {
        "v_grid":       np.asarray(v_grid),
        "chi2":         np.asarray(chi2_vals),
        "n_kept":       np.asarray(n_kept_vals),
        "L_tau":        int(C.shape[0]),
        "w_desc":       w_desc,
        "r_desc":       r_desc,
        "var_emp_desc": var_emp_desc,
        "cum_var":      cum_var,
    }


def plot_threshold_scan(res: dict, summary_chi2: float, out_dir: str, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_RC)

    os.makedirs(out_dir, exist_ok=True)

    v     = res["v_grid"]
    chi2  = res["chi2"]
    nkept = res["n_kept"]
    L_tau = res["L_tau"]

    # ------------------------------------------------------------------
    # Fig 1: chi^2(v) with calibration target at 1.0.
    # Under the new normalisation E[chi^2] = 1 independent of v at perfect
    # fit, so a flat curve sitting on the dotted target line is the
    # calibration check.
    # log(1-v) x-axis spreads the v -> 1 cluster naturally; v=1.0 maps to
    # 1-v=0 which has no log, so clamp to eps for plot only.
    # ------------------------------------------------------------------
    eps = 1e-5
    x_log = np.maximum(1.0 - v, eps)
    xticklabels = [f"{vv:g}" for vv in v]

    grey_bbox = dict(boxstyle="round,pad=0.28", fc="white",
                     ec="0.4", lw=0.7, alpha=0.95)
    blue_bbox = dict(boxstyle="round,pad=0.25", fc="white",
                     ec="C0", lw=0.9, alpha=0.95)

    # 3-level cycling stagger keeps adjacent labels from overlapping.
    Y_LEVELS = (14, 44, 74)
    def _yoff(i, sign=1):
        return sign * Y_LEVELS[i % 3]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(1.0, ls=":", color="black", lw=1.1, alpha=0.7,
               label=r"target  $\chi^2 = 1$")
    ax.plot(x_log, chi2, "o-", color="C0", lw=2.0, ms=6,
            label=r"$\chi^2_{\mathrm{eval}}(v)$")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(x_log)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=11)
    ax.minorticks_off()
    ax.set_xlabel(r"variance threshold $v$")
    ax.set_ylabel(r"$\chi^2_{\mathrm{eval}}$")
    ax.set_title(label + r"  --  calibration: $\chi^2$ vs threshold")
    ax.grid(alpha=0.3, which="major")
    ax.legend(loc="best")

    for i, (xl, cc, nk) in enumerate(zip(x_log, chi2, nkept)):
        ax.annotate(rf"$n={nk}$", xy=(xl, cc), xytext=(0, _yoff(i, +1)),
                    textcoords="offset points", fontsize=11, ha="center",
                    va="bottom", bbox=grey_bbox,
                    arrowprops=dict(arrowstyle="-", color="0.6",
                                    lw=0.6, shrinkA=0, shrinkB=4))
        ax.annotate(rf"${cc:.3f}$", xy=(xl, cc), xytext=(0, _yoff(i, -1)),
                    textcoords="offset points", fontsize=11, ha="center",
                    va="top", color="C0", bbox=blue_bbox,
                    arrowprops=dict(arrowstyle="-", color="C0",
                                    lw=0.6, shrinkA=0, shrinkB=4, alpha=0.5))
    y0, y1 = ax.get_ylim()
    yp = 0.22 * (y1 - y0)
    ax.set_ylim(y0 - yp, y1 + yp)

    fig.tight_layout()
    out_path_1 = os.path.join(out_dir, "chi2_vs_threshold.pdf")
    fig.savefig(out_path_1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: per-mode r_k (zoomed bulk + full log) and eigenvalue spectrum
    # ------------------------------------------------------------------
    k     = np.arange(1, L_tau + 1)
    K_BULK = 90       # modes shown in the zoomed-bulk panel
    Y_BULK_PAD = 0.10

    r_full  = res["r_desc"]
    r_bulk  = r_full[:K_BULK]
    bulk_lo = max(0.0, float(r_bulk.min()) - Y_BULK_PAD * (r_bulk.max() - r_bulk.min()))
    bulk_hi = float(r_bulk.max()) + Y_BULK_PAD * (r_bulk.max() - r_bulk.min())

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0),
                             gridspec_kw={"height_ratios": [1.1, 1.1, 0.9],
                                          "hspace": 0.32})

    # Panel 0 — zoomed bulk on linear y. This is where the chi^2<1 story lives.
    ax = axes[0]
    ax.plot(k[:K_BULK], r_bulk, "o-", color="C0", ms=3, lw=1.2,
            label=r"$r_k = \langle \eta_k^2\rangle / w_k$")
    ax.axhline(1.0, ls=":", color="black", lw=1.0, alpha=0.6,
               label=r"$r_k = 1$ (noise level)")
    # Extend xlim past K_BULK so the rightmost cluster line (v=1, k~99) lands
    # inside the panel. The data curve still only runs up to K_BULK.
    ax.set_xlim(0, max(K_BULK + 1, int(res["n_kept"].max()) + 5))
    ax.set_ylim(bulk_lo, bulk_hi)
    ax.set_xlabel(r"mode index $k$")
    ax.set_ylabel(r"$r_k$  (linear, bulk)")
    ax.set_title(label + r"  --  zoomed to modes $1\!\!-\!\!" + str(K_BULK) + r"$")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # Panels 1, 2 share x and span the full eigenspectrum.
    axes[2].sharex(axes[1])

    ax = axes[1]
    ax.plot(k, r_full, "o-", color="C0", ms=3, lw=1.2,
            label=r"$r_k$ (full range)")
    ax.axhline(1.0, ls=":", color="black", lw=1.0, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel(r"$r_k$  (log, all modes)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = axes[2]
    ax.semilogy(k, res["w_desc"], "s-", color="C2", ms=3, lw=1.2,
                label=r"eigenvalue $w_k$")
    ax.set_xlabel(r"mode index $k$ (sorted by descending $w_k$)")
    ax.set_ylabel(r"$w_k$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")

    # Cluster threshold lines {0.99, 0.995, 0.999, 0.9995, 1.0}.
    # Each gets a vertical line on all three panels; bottom panel carries the
    # v + chi^2 label rotated vertically near the lower edge.
    v_arr    = res["v_grid"]
    n_arr    = res["n_kept"]
    chi2_arr = res["chi2"]
    CLUSTER_V = (0.99, 0.995, 0.999, 0.9995, 1.0)

    for v_mark in CLUSTER_V:
        idx = int(np.where(np.isclose(v_arr, v_mark))[0][0])
        n_mark    = int(n_arr[idx])
        chi2_mark = float(chi2_arr[idx])
        for axx in axes:
            axx.axvline(n_mark, ls="--", color="grey", lw=0.8, alpha=0.55)
        label_text = rf"$v={v_mark:g}$, $\chi^2={chi2_mark:.3f}$"
        # Annotate on both the zoomed bulk panel and the eigenvalue panel.
        for axx in (axes[0], axes[2]):
            axx.annotate(
                label_text,
                xy=(n_mark, 0.05),
                xycoords=axx.get_xaxis_transform(),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=8.5, color="0.25",
                rotation=90, rotation_mode="anchor",
                ha="left", va="bottom",
            )

    out_path_2 = os.path.join(out_dir, "per_mode_variance_ratio.pdf")
    fig.savefig(out_path_2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path_1, out_path_2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="finetune output directory containing summary.pt and params.json")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    label = os.path.basename(run_dir.rstrip("/"))

    eta, C, params, summary = load_residuals_and_C(run_dir)
    print(f"Loaded run: {label}")
    print(f"  N samples:   {eta.shape[0]}")
    print(f"  L_tau:       {eta.shape[1]}")
    print(f"  C shape:     {C.shape}")
    print(f"  data path:   {params['DATA_PATH']}")

    res = threshold_scan(eta, C)

    summary_chi2 = float(summary.get("final_chi2", float("nan")))
    print()
    print(f"  baseline chi^2 (training-time, v=0.99 truncated): {summary_chi2:.4f}")
    print()
    print("  threshold scan (eval-time, fixed model):")
    print(f"    {'v':>8s}  {'n_kept':>7s}  {'chi^2':>10s}  {'chi^2/v':>10s}")
    for v, c, n in zip(res["v_grid"], res["chi2"], res["n_kept"]):
        print(f"    {v:>8.4f}  {n:>7d}  {c:>10.4f}  {c/v:>10.4f}")

    print()
    r = res["r_desc"]
    print(f"  per-mode variance ratio r_k (descending eigenvalue order):")
    print(f"    r_k  range:   [{r.min():.3e}, {r.max():.3e}]")
    print(f"    r_k  median:   {np.median(r):.4f}")
    print(f"    r_k  mean:     {r.mean():.4f}")
    bands = [(1, 10), (11, 30), (31, 60), (61, 90), (91, res["L_tau"])]
    print(f"    band-mean r_k (descending-w blocks):")
    for lo, hi in bands:
        block = r[lo - 1:hi]
        print(f"      modes {lo:>3d}..{hi:>3d}:  mean={block.mean():.4f}  median={np.median(block):.4f}")

    out_dir = os.path.join(run_dir, "diagnostics", "threshold_scan")
    p1, p2 = plot_threshold_scan(res, summary_chi2, out_dir, label.replace("_", r"\_"))
    print()
    print(f"  plots written:")
    print(f"    {p1}")
    print(f"    {p2}")

    np.savez(os.path.join(out_dir, "scan.npz"),
             v_grid=res["v_grid"], chi2=res["chi2"], n_kept=res["n_kept"],
             w_desc=res["w_desc"], r_desc=res["r_desc"],
             var_emp_desc=res["var_emp_desc"], cum_var=res["cum_var"])
    print(f"    {os.path.join(out_dir, 'scan.npz')}")


if __name__ == "__main__":
    main()
