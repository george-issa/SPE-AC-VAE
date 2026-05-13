"""
Aggregate the 5-seed 2-cell VAE campaign (per-cell raw covariance).

Loads every covpc run at this cell pair, dedupes by SEED (keeping the
most recent for each seed), and produces:

  out/comparison_multicell_seeds/
    latent_grid.pdf       — one panel per seed, shared axes, shows the
                            cell A and cell B clusters.
    spectra_per_cell.pdf  — 2 panels (cell A, cell B); each shows the 5
                            VAE medians coloured by seed, DEAC, MaxEnt.
    distances.pdf         — bar chart of ||z_A - z_B|| per seed.
    summary.md            — per-seed numbers + summary stats.

Usage:
    python pretrain/compare_multicell_seeds.py
"""

import os
import sys
import json
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pretrain.pretrain_losses import spectral_from_poles  # type: ignore


PROJ = "/Users/gissa/Documents/AC/SPE-AC-VAE"

# Covariance tag to aggregate. Each value matches a distinct run-dir
# pattern produced by run_finetune_multicell.py:
#   "covpc"   = per-cell raw covariance, 1e-12 floor only.
#   "covpclw" = per-cell Ledoit-Wolf shrinkage.
COV_TAG = "covpclw"

RUN_GLOB = (f"{PROJ}/out/"
            f"finetune_real-site_holstein_b10.00_2cell_A_B_"
            f"numpoles10_z2_{COV_TAG}-fresh-v2L-*")
OUT_DIR  = f"{PROJ}/out/comparison_multicell_seeds_{COV_TAG}"

OMEGA_MIN, OMEGA_MAX, OMEGA_N = -20.0, 20.0, 1000

# Only aggregate runs at this learning rate. Keeps mixed-lr sweeps from
# being silently averaged together.
FILTER_LR = 3e-4

PLOT_RC = {
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
}

# Cell B uses purple (#785EF0) rather than orange so it does not collide
# with DEAC's vermilion in side-by-side plots.
CELL_COLORS = ["#0072B2", "#785EF0", "#009E73", "#CC79A7"]


def _l2(omega, a, b):
    return float(np.sqrt(np.trapezoid((a - b) ** 2, omega)))


def _linf(a, b):
    return float(np.max(np.abs(a - b)))


def _maxent_path(beta, omega, n):
    return os.path.join(
        PROJ, "MaxEnt_benchmark", "out",
        f"anacont_real-site_holstein_b{beta:.2f}_w{omega:.2f}_n{n:.2f}",
        "summary_mean_fullcov.npz",
    )


def _load_run(run_dir):
    sp = os.path.join(run_dir, "summary.pt")
    if not os.path.exists(sp):
        return None
    s = torch.load(sp, weights_only=False)
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)
    omega = torch.linspace(OMEGA_MIN, OMEGA_MAX, OMEGA_N)
    with torch.no_grad():
        A_all = spectral_from_poles(s["poles"], s["residues"], omega).numpy()
    return {
        "run_dir":  run_dir,
        "seed":     params.get("SEED"),
        "params":   params,
        "cells":    s["cells"],
        "cell_ids": s["cell_ids"].numpy(),
        "mu":       s["mu"].numpy(),
        "var_mu":   s["var_mu_per_dim"].numpy(),
        "au":       int(s["active_units"]),
        "kl_total": float(s["kl_total_avg"]),
        "chi2_min": float(s["best_chi2"]),
        "chi2_final": float(s["final_chi2"]),
        "omega":    omega.numpy(),
        "A_all":    A_all,
        "ws_ref":   s["ws_ref"].numpy(),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(PLOT_RC)

    runs = []
    seen_seeds = set()
    for d in sorted(glob.glob(RUN_GLOB)):
        r = _load_run(d)
        if r is None or r["seed"] is None:
            continue
        if FILTER_LR is not None and float(r["params"].get("FINETUNE_LR", 0.0)) != FILTER_LR:
            continue
        if r["seed"] in seen_seeds:
            continue
        seen_seeds.add(r["seed"])
        runs.append(r)
    runs.sort(key=lambda r: r["seed"])
    print(f"Discovered {len(runs)} fixed-seed multi-cell runs "
          f"(seeds: {[r['seed'] for r in runs]})")
    if not runs:
        raise RuntimeError("No fixed-seed multi-cell runs found.")

    cells = runs[0]["cells"]
    omega = runs[0]["omega"]

    # Reference curves per cell
    A_de_per_cell = []
    A_me_per_cell = []
    for c in cells:
        A_de_per_cell.append(
            np.interp(omega, runs[0]["ws_ref"], c["dos_ref"].numpy())
        )
        me_path = _maxent_path(c["beta"], c["omega"], c["n"])
        if os.path.exists(me_path):
            d = dict(np.load(me_path, allow_pickle=True))
            A_me_per_cell.append(
                np.interp(omega, np.asarray(d["omega"]), np.asarray(d["A_opt"]))
            )
        else:
            A_me_per_cell.append(None)

    # Common axis limits for latent_grid: union of all seeds' mu values
    # with a small padding so clusters don't sit on the edges.
    all_mu = np.concatenate([r["mu"] for r in runs], axis=0)
    pad = 0.10 * (all_mu.max(axis=0) - all_mu.min(axis=0))
    xlim = (all_mu[:, 0].min() - pad[0], all_mu[:, 0].max() + pad[0])
    ylim = (all_mu[:, 1].min() - pad[1], all_mu[:, 1].max() + pad[1])

    # ---- latent_grid.pdf: 1 panel per seed, shared axes ----
    n_runs = len(runs)
    ncol = min(n_runs, 3)
    nrow = (n_runs + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.0 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    for i, r in enumerate(runs):
        ax = axes[i]
        for cid, c in enumerate(cells):
            mask = (r["cell_ids"] == cid)
            ax.scatter(r["mu"][mask, 0], r["mu"][mask, 1],
                       color=CELL_COLORS[cid], s=36,
                       edgecolor="k", alpha=0.85,
                       label=fr"cell {c['label']}" if i == 0 else None)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(
            rf"seed $= {r['seed']}$, AU $= {r['au']}/2$"
        )
        ax.grid(alpha=0.25)
    for j in range(n_runs, len(axes)):
        axes[j].axis("off")
    for ax in axes[:n_runs]:
        ax.set_xlabel(r"$\mu_z[0]$")
        ax.set_ylabel(r"$\mu_z[1]$")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        r"Multi-cell VAE: encoder latent positions per seed (shared axes)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "latent_grid.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- spectra_per_cell.pdf: 2 panels (cell A, cell B) ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    cmap = plt.cm.viridis
    cell_l2_me = [[None] * n_runs for _ in cells]   # cell-by-seed L^2 vs MaxEnt
    cell_l2_de = [[None] * n_runs for _ in cells]
    for cid, c in enumerate(cells):
        ax = axes[cid]
        for i, r in enumerate(runs):
            mask = (r["cell_ids"] == cid)
            A_med = np.median(r["A_all"][mask], axis=0)
            col = cmap(0.10 + 0.75 * i / max(1, n_runs - 1))
            l2_de = _l2(omega, A_med, A_de_per_cell[cid])
            cell_l2_de[cid][i] = l2_de
            if A_me_per_cell[cid] is not None:
                cell_l2_me[cid][i] = _l2(omega, A_med, A_me_per_cell[cid])
            label = fr"seed $={r['seed']}$ ($L^2_{{\rm DEAC}}={l2_de:.3f}$)"
            ax.plot(omega, A_med, color=col, lw=1.4, alpha=0.9, label=label)
        ax.plot(omega, A_de_per_cell[cid], color="#D55E00", lw=1.0, alpha=0.7,
                label="DEAC")
        if A_me_per_cell[cid] is not None:
            ax.plot(omega, A_me_per_cell[cid], color="#2C2C2C", lw=2.0, ls="--",
                    label="MaxEnt")
        ax.set_ylabel(r"$A(\omega)$")
        ax.set_xlim(-10, 10)
        ax.set_title(
            rf"cell {c['label']}: "
            rf"$(\beta, \Omega, n) = ({c['beta']:.1f}, {c['omega']:.2f}, "
            rf"{c['n']:.2f})$"
        )
        ax.legend(frameon=False, fontsize=8,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=2)
    axes[-1].set_xlabel(r"$\omega$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectra_per_cell.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- distances.pdf: ||z_A - z_B|| per seed ----
    dists = []
    for r in runs:
        mask_A = (r["cell_ids"] == 0)
        mask_B = (r["cell_ids"] == 1)
        z_A = r["mu"][mask_A].mean(axis=0)
        z_B = r["mu"][mask_B].mean(axis=0)
        dists.append(float(np.linalg.norm(z_A - z_B)))
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(n_runs)
    bars = ax.bar(xs, dists, color="#0072B2", alpha=0.7, edgecolor="k")
    for x, d in zip(xs, dists):
        ax.text(x, d + 0.02, f"{d:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([fr"seed {r['seed']}" for r in runs])
    ax.set_ylabel(r"$\|\bar z_A - \bar z_B\|$")
    ax.set_title(r"Inter-cluster latent distance, per seed")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "distances.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- summary.md ----
    md = []
    md.append("# Multi-seed 2-cell VAE on site-Holstein (per-cell raw covariance)\n")
    md.append(f"Cells: A $({cells[0]['beta']:.1f}, {cells[0]['omega']:.2f}, "
              f"{cells[0]['n']:.2f})$, B $({cells[1]['beta']:.1f}, "
              f"{cells[1]['omega']:.2f}, {cells[1]['n']:.2f})$. "
              f"`NUM_POLES=10`, `LATENT_DIM=2`, `BATCH_SIZE=10`, 1000 epochs.\n")
    md.append("## Per-seed numbers\n")
    md.append("| seed | AU | $\\|\\bar z_A - \\bar z_B\\|$ "
              "| $\\chi^2_{\\min}$ | $\\chi^2_{\\rm final}$ "
              "| cell A $L^2_{\\rm DEAC}$ | cell A $L^2_{\\rm ME}$ "
              "| cell B $L^2_{\\rm DEAC}$ |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(runs):
        au = f"{r['au']}/2"
        chi_min = f"{r['chi2_min']:.3f}"
        chi_fin = f"{r['chi2_final']:.3f}"
        l2_de_A = f"{cell_l2_de[0][i]:.3f}"
        l2_me_A = f"{cell_l2_me[0][i]:.3f}" if cell_l2_me[0][i] is not None else "—"
        l2_de_B = f"{cell_l2_de[1][i]:.3f}"
        md.append(f"| {r['seed']} | {au} | {dists[i]:.3f} "
                  f"| {chi_min} | {chi_fin} "
                  f"| {l2_de_A} | {l2_me_A} | {l2_de_B} |")
    a_l2_de_A = np.array(cell_l2_de[0])
    a_l2_de_B = np.array(cell_l2_de[1])
    a_l2_me_A = np.array([v for v in cell_l2_me[0] if v is not None])
    a_dist    = np.array(dists)
    a_chi     = np.array([r["chi2_min"] for r in runs])

    md.append("\n## Summary statistics\n")
    md.append("| metric | mean | median | std | min | max |")
    md.append("|---|---:|---:|---:|---:|---:|")
    def _row(label, a):
        return (f"| {label} | {a.mean():.4f} | {np.median(a):.4f} | "
                f"{a.std(ddof=1) if len(a)>1 else 0.0:.4f} | "
                f"{a.min():.4f} | {a.max():.4f} |")
    md.append(_row(r"$\|\bar z_A - \bar z_B\|$", a_dist))
    md.append(_row(r"$\chi^2_{\min}$", a_chi))
    md.append(_row(r"cell A $L^2$ vs DEAC", a_l2_de_A))
    md.append(_row(r"cell A $L^2$ vs MaxEnt", a_l2_me_A))
    md.append(_row(r"cell B $L^2$ vs DEAC", a_l2_de_B))

    md.append("\n## Run dirs")
    for r in runs:
        md.append(f"- seed {r['seed']}: `{os.path.basename(r['run_dir'])}`")

    md_text = "\n".join(md)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(md_text + "\n")
    print()
    print(md_text)
    print(f"\nPlots → {OUT_DIR}")


if __name__ == "__main__":
    main()
