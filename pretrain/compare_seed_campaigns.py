"""
Combined DecoderOnly + VAE (Lz=1) seed-sweep comparison on site-Holstein
(beta=10, Omega=1, n=1, LW + P=10, batch=5, 1000 epochs).

5 fixed seeds (42..46) per architecture. Tells us whether DEC > VAE or
VAE > DEC is genuine on this cell, or whether seed variance washes out
the architectural difference.

Usage:
    python pretrain/compare_seed_campaigns.py
"""

import os
import sys
import json
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_process_real import HolsteinJLD2Dataset  # type: ignore
from pretrain.pretrain_losses import spectral_from_poles  # type: ignore


PROJ        = "/Users/gissa/Documents/AC/SPE-AC-VAE"
DEC_GLOB    = f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_dec_covlw-fresh-vDEC-*"
VAE_GLOB    = f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_z1_covlw-fresh-v2L-*"
MAXENT_NPZ  = f"{PROJ}/MaxEnt_benchmark/out/anacont_real-site_holstein_b10.00_w1.00_n1.00/summary_mean_fullcov.npz"
OUT_DIR     = f"{PROJ}/out/comparison_seed_campaigns_lw_p10"

OMEGA_MIN, OMEGA_MAX, OMEGA_N = -20.0, 20.0, 1000

FILTER_BATCH_SIZE = 5
FILTER_NUM_POLES  = 10
FILTER_EPOCHS     = 1000

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


def _load_dec_run(run_dir):
    from decoder_only import DecoderOnly  # type: ignore
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)
    model = DecoderOnly(
        input_dim=params["INPUT_DIM"], num_poles=params["NUM_POLES"],
        beta=params["BETA"], N_nodes=params["N_NODES"],
        ph_symmetric=params.get("PH_SYMMETRIC", False),
    )
    state = torch.load(os.path.join(run_dir, "model", "best_model_finetune.pth"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, params


def _load_vae_run(run_dir):
    from model2_leaky import VariationalAutoEncoder2 as VAE  # type: ignore
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)
    model = VAE(
        input_dim=params["INPUT_DIM"], num_poles=params["NUM_POLES"],
        beta=params["BETA"], N_nodes=params["N_NODES"],
        latent_dim=params["LATENT_DIM"],
        ph_symmetric=params.get("PH_SYMMETRIC", False),
    )
    state = torch.load(os.path.join(run_dir, "model", "best_model_finetune.pth"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, params


def _spectrum(run_dir, loader_factory):
    if "_dec_" in os.path.basename(run_dir):
        model, params = _load_dec_run(run_dir)
    else:
        model, params = _load_vae_run(run_dir)
    ds, loader = loader_factory(params)
    omega = torch.linspace(OMEGA_MIN, OMEGA_MAX, OMEGA_N)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            B = batch.shape[0]
            x = batch.view(B, params["INPUT_DIM"])
            _, _, _, poles, residues, _ = model(x, deterministic=True)
            chunks.append(spectral_from_poles(poles, residues, omega).numpy())
    A = np.concatenate(chunks, axis=0)
    chi2 = np.load(os.path.join(run_dir, "losses", "chi2_losses_finetune.npy"))
    return {
        "params":   params,
        "omega":    omega.numpy(),
        "A_mean":   A.mean(axis=0),
        "chi2_min": float(np.min(chi2)),
        "chi2_final": float(chi2[-1]),
        "seed":     params.get("SEED"),
        "run_dir":  run_dir,
    }


def _make_loader_factory():
    cache = {}
    def factory(params):
        key = (params["DATA_PATH"], params["HOLSTEIN_N_IDX"],
               params["HOLSTEIN_OMEGA_IDX"], params["HOLSTEIN_BETA_IDX"],
               params["BATCH_SIZE"])
        if key not in cache:
            ds = HolsteinJLD2Dataset(
                params["DATA_PATH"], n_idx=params["HOLSTEIN_N_IDX"],
                omega_idx=params["HOLSTEIN_OMEGA_IDX"],
                beta_idx=params["HOLSTEIN_BETA_IDX"],
            )
            cache[key] = (ds, DataLoader(ds, batch_size=params["BATCH_SIZE"], shuffle=False))
        return cache[key]
    return factory


def _discover_fixed_seed_runs(glob_pattern, loader_factory):
    runs = []
    for d in sorted(glob.glob(glob_pattern)):
        try:
            with open(os.path.join(d, "params.json")) as f:
                params = json.load(f)
        except FileNotFoundError:
            continue
        seed = params.get("SEED")
        if (seed is None
            or params.get("BATCH_SIZE")        != FILTER_BATCH_SIZE
            or params.get("NUM_POLES")         != FILTER_NUM_POLES
            or params.get("FINETUNE_EPOCHS")   != FILTER_EPOCHS):
            continue
        runs.append(_spectrum(d, loader_factory))
    runs.sort(key=lambda r: r["seed"])
    return runs


def _l2(omega, a, b):
    return float(np.sqrt(np.trapezoid((a - b) ** 2, omega)))


def _linf(a, b):
    return float(np.max(np.abs(a - b)))


def _stats(arr):
    a = np.asarray(arr)
    return dict(mean=a.mean(), median=np.median(a),
                std=a.std(ddof=1) if len(a) > 1 else 0.0,
                min=a.min(), max=a.max(), n=len(a))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(PLOT_RC)

    factory = _make_loader_factory()
    dec = _discover_fixed_seed_runs(DEC_GLOB, factory)
    vae = _discover_fixed_seed_runs(VAE_GLOB, factory)
    print(f"Discovered {len(dec)} DEC fixed-seed runs, "
          f"{len(vae)} VAE fixed-seed runs.")
    if not dec or not vae:
        raise RuntimeError("Need fixed-seed runs from both architectures.")

    omega = dec[0]["omega"]
    d_me = dict(np.load(MAXENT_NPZ, allow_pickle=True))
    A_me_grid = np.interp(omega, d_me["omega"], d_me["A_opt"])

    # DEAC: pull from any DEC summary.pt
    A_de_grid = None
    for r in dec:
        sp = os.path.join(r["run_dir"], "summary.pt")
        if os.path.exists(sp):
            s = torch.load(sp, weights_only=False)
            if "dos_ref" in s and "ws_ref" in s:
                A_de_grid = np.interp(omega, s["ws_ref"].numpy(), s["dos_ref"].numpy())
                break

    # Stack per-arch
    A_dec = np.stack([r["A_mean"] for r in dec], axis=0)
    A_vae = np.stack([r["A_mean"] for r in vae], axis=0)
    dec_l2   = [_l2(omega, r["A_mean"], A_me_grid) for r in dec]
    vae_l2   = [_l2(omega, r["A_mean"], A_me_grid) for r in vae]
    dec_linf = [_linf(r["A_mean"], A_me_grid) for r in dec]
    vae_linf = [_linf(r["A_mean"], A_me_grid) for r in vae]
    dec_chi  = [r["chi2_min"] for r in dec]
    vae_chi  = [r["chi2_min"] for r in vae]

    # ---- Two-band overlay (DEC band + VAE band, both vs MaxEnt) ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08}, sharex=True,
    )
    DEC_C = "#0072B2"  # blue
    VAE_C = "#CC79A7"  # magenta
    ax_top.fill_between(omega, A_dec.min(0), A_dec.max(0), color=DEC_C, alpha=0.18,
                        label=fr"DEC envelope ($N={len(dec)}$)")
    ax_top.plot(omega, np.median(A_dec, 0), color=DEC_C, lw=1.6, label="DEC median")
    ax_top.fill_between(omega, A_vae.min(0), A_vae.max(0), color=VAE_C, alpha=0.18,
                        label=fr"VAE $L_z=1$ envelope ($N={len(vae)}$)")
    ax_top.plot(omega, np.median(A_vae, 0), color=VAE_C, lw=1.6, label="VAE median")
    if A_de_grid is not None:
        ax_top.plot(omega, A_de_grid, color="#D55E00", lw=1.0, alpha=0.85, label="DEAC")
    ax_top.plot(omega, A_me_grid, color="#2C2C2C", lw=2.4, ls="--", label="MaxEnt (basis)")
    ax_top.set_ylabel(r"$\langle A(\omega)\rangle$")
    ax_top.set_xlim(-10, 10)
    ax_top.set_title(
        rf"Site-Holstein $(\beta,\Omega,n)=(10,1,1)$, LW + $P=10$. "
        r"DEC vs VAE $L_z = 1$ seed bands"
    )
    ax_top.legend(frameon=False, fontsize=9, loc="upper right")
    ax_bot.fill_between(omega, A_dec.min(0) - A_me_grid, A_dec.max(0) - A_me_grid,
                        color=DEC_C, alpha=0.18)
    ax_bot.plot(omega, np.median(A_dec, 0) - A_me_grid, color=DEC_C, lw=1.4)
    ax_bot.fill_between(omega, A_vae.min(0) - A_me_grid, A_vae.max(0) - A_me_grid,
                        color=VAE_C, alpha=0.18)
    ax_bot.plot(omega, np.median(A_vae, 0) - A_me_grid, color=VAE_C, lw=1.4)
    if A_de_grid is not None:
        ax_bot.plot(omega, A_de_grid - A_me_grid, color="#D55E00", lw=0.9, alpha=0.85)
    ax_bot.axhline(0.0, color="#2C2C2C", ls="--", lw=1.0, alpha=0.6)
    ax_bot.set_xlabel(r"$\omega$")
    ax_bot.set_ylabel(r"$A - A_{\rm ME}$")
    ax_bot.set_xlim(-10, 10)
    fig.savefig(os.path.join(OUT_DIR, "spectral_bands_dec_vs_vae.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Box/strip plot of L^2 and chi^2 distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (label, dec_vals, vae_vals) in zip(axes, [
        (r"$\chi^2_{\min}$",      dec_chi,  vae_chi),
        (r"$L^2$ vs MaxEnt",      dec_l2,   vae_l2),
        (r"$L^\infty$ vs MaxEnt", dec_linf, vae_linf),
    ]):
        positions = [1, 2]
        bp = ax.boxplot([dec_vals, vae_vals], positions=positions, widths=0.5,
                        showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], [DEC_C, VAE_C]):
            patch.set_facecolor(c); patch.set_alpha(0.4)
        for k, vals, c in [(1, dec_vals, DEC_C), (2, vae_vals, VAE_C)]:
            ax.scatter(np.full(len(vals), k) + np.random.RandomState(k).uniform(-0.08, 0.08, len(vals)),
                       vals, color=c, edgecolor="k", s=46, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"DEC\n($N={len(dec_vals)}$)", f"VAE $L_z=1$\n($N={len(vae_vals)}$)"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-seed metric distributions ($N=5$ each, fixed seeds 42--46)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "metric_distributions.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Markdown summary ----
    md = []
    md.append("# DEC vs VAE $L_z=1$ seed-sweep comparison (site-Holstein, LW + P=10)\n")
    md.append(f"Cell: $(\\beta, \\Omega, n) = (10, 1, 1)$. "
              f"`BATCH_SIZE={FILTER_BATCH_SIZE}`, `NUM_POLES={FILTER_NUM_POLES}`, "
              f"`FINETUNE_EPOCHS={FILTER_EPOCHS}`. Fixed seeds 42--46 per arch.\n")

    def _fmt(s):
        return (f"mean = {s['mean']:.4f}, median = {s['median']:.4f}, "
                f"std = {s['std']:.4f}, range [{s['min']:.4f}, {s['max']:.4f}]")

    md.append("## Summary statistics\n")
    md.append("| arch | metric | mean | median | std | min | max |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for arch_label, vals_set in [
        ("DEC", [("$\\chi^2_{\\min}$", dec_chi), ("$L^2$ vs ME", dec_l2),
                 ("$L^\\infty$ vs ME", dec_linf)]),
        ("VAE $L_z=1$", [("$\\chi^2_{\\min}$", vae_chi), ("$L^2$ vs ME", vae_l2),
                         ("$L^\\infty$ vs ME", vae_linf)]),
    ]:
        for metric_label, vals in vals_set:
            s = _stats(vals)
            md.append(f"| {arch_label} | {metric_label} | "
                      f"{s['mean']:.4f} | {s['median']:.4f} | {s['std']:.4f} "
                      f"| {s['min']:.4f} | {s['max']:.4f} |")

    md.append("\n## Per-seed numbers — DEC")
    md.append("| seed | $\\chi^2_{\\min}$ | $L^2$ vs ME | $L^\\infty$ vs ME |")
    md.append("|---:|---:|---:|---:|")
    for r, l2, linf in zip(dec, dec_l2, dec_linf):
        md.append(f"| {r['seed']} | {r['chi2_min']:.4f} | {l2:.4f} | {linf:.4f} |")

    md.append("\n## Per-seed numbers — VAE $L_z=1$")
    md.append("| seed | $\\chi^2_{\\min}$ | $L^2$ vs ME | $L^\\infty$ vs ME |")
    md.append("|---:|---:|---:|---:|")
    for r, l2, linf in zip(vae, vae_l2, vae_linf):
        md.append(f"| {r['seed']} | {r['chi2_min']:.4f} | {l2:.4f} | {linf:.4f} |")

    if A_de_grid is not None:
        md.append(f"\nDEAC vs MaxEnt: $L^2 = {_l2(omega, A_de_grid, A_me_grid):.4f}$, "
                  f"$L^\\infty = {_linf(A_de_grid, A_me_grid):.4f}$.")

    md_text = "\n".join(md)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(md_text + "\n")
    print(md_text)
    print(f"\nPlots → {OUT_DIR}")


if __name__ == "__main__":
    main()
