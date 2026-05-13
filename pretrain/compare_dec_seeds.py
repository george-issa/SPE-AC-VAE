"""
Aggregate the 5-seed DecoderOnly campaign on site-Holstein
(beta=10, Omega=1, n=1, LW + P=10, batch=5).

Discovers all DEC runs at this cell and partitions them into
"fixed-seed" (the campaign) vs "random-seed" (the original vDEC-1, kept
as a reference point). Produces:

  out/comparison_dec_seed_sweep_lw_p10/
    spectral_bands.pdf       — median + min/max envelope of the 5 fixed-seed
                                spectra, MaxEnt and DEAC overlaid
    spectral_overlay.pdf     — per-seed curves coloured individually
    summary.md               — table of chi^2, L2, Linf per seed + summary stats

Usage:
    python pretrain/compare_dec_seeds.py
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
from decoder_only import DecoderOnly  # type: ignore


PROJ        = "/Users/gissa/Documents/AC/SPE-AC-VAE"
DEC_GLOB    = f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_dec_covlw-fresh-vDEC-*"
MAXENT_NPZ  = f"{PROJ}/MaxEnt_benchmark/out/anacont_real-site_holstein_b10.00_w1.00_n1.00/summary_mean_fullcov.npz"
OUT_DIR     = f"{PROJ}/out/comparison_dec_seed_sweep_lw_p10"

OMEGA_MIN, OMEGA_MAX, OMEGA_N = -20.0, 20.0, 1000

# Only fixed-seed runs that match this BATCH_SIZE / NUM_POLES / epochs set go
# into the campaign band. Random-seed runs and full-batch runs are excluded
# (they live in the same dir prefix but answer different questions).
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


def _load_run(run_dir):
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)
    model = DecoderOnly(
        input_dim=params["INPUT_DIM"], num_poles=params["NUM_POLES"],
        beta=params["BETA"], N_nodes=params["N_NODES"],
        ph_symmetric=params.get("PH_SYMMETRIC", False),
    )
    state = torch.load(
        os.path.join(run_dir, "model", "best_model_finetune.pth"),
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    ds = HolsteinJLD2Dataset(
        params["DATA_PATH"], n_idx=params["HOLSTEIN_N_IDX"],
        omega_idx=params["HOLSTEIN_OMEGA_IDX"], beta_idx=params["HOLSTEIN_BETA_IDX"],
    )
    loader = DataLoader(ds, batch_size=params["BATCH_SIZE"], shuffle=False)
    omega = torch.linspace(OMEGA_MIN, OMEGA_MAX, OMEGA_N)
    A_chunks = []
    with torch.no_grad():
        for batch in loader:
            B = batch.shape[0]
            x = batch.view(B, params["INPUT_DIM"])
            _, _, _, poles, residues, _ = model(x, deterministic=True)
            A_chunks.append(spectral_from_poles(poles, residues, omega).numpy())
    A = np.concatenate(A_chunks, axis=0)

    chi2 = np.load(os.path.join(run_dir, "losses", "chi2_losses_finetune.npy"))
    return {
        "params":   params,
        "omega":    omega.numpy(),
        "A_mean":   A.mean(axis=0),
        "chi2_min": float(np.min(chi2)),
        "chi2_final": float(chi2[-1]),
        "n_epochs": int(len(chi2)),
        "seed":     params.get("SEED"),
        "run_dir":  run_dir,
    }


def _load_maxent():
    d = dict(np.load(MAXENT_NPZ, allow_pickle=True))
    return np.asarray(d["omega"]), np.asarray(d["A_opt"])


def _load_deac_for_cell():
    # Pull DEAC from any DEC run's summary.pt (it's the in-file reference).
    for d in glob.glob(DEC_GLOB):
        sp = os.path.join(d, "summary.pt")
        if os.path.exists(sp):
            s = torch.load(sp, weights_only=False)
            if "dos_ref" in s and "ws_ref" in s:
                return s["ws_ref"].numpy(), s["dos_ref"].numpy()
    return None, None


def _l2(omega, a, b):
    return float(np.sqrt(np.trapezoid((a - b) ** 2, omega)))


def _linf(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(PLOT_RC)

    # Discover and partition runs
    fixed, random_seed_runs, other = [], [], []
    for d in sorted(glob.glob(DEC_GLOB)):
        try:
            r = _load_run(d)
        except FileNotFoundError:
            continue
        p = r["params"]
        seed = p.get("SEED")
        match = (p.get("BATCH_SIZE") == FILTER_BATCH_SIZE and
                 p.get("NUM_POLES")  == FILTER_NUM_POLES  and
                 p.get("FINETUNE_EPOCHS") == FILTER_EPOCHS)
        if not match:
            other.append(r)
            continue
        if seed is None:
            random_seed_runs.append(r)
        else:
            fixed.append(r)

    fixed.sort(key=lambda r: r["seed"])
    print(f"Discovered {len(fixed)} fixed-seed runs, "
          f"{len(random_seed_runs)} random-seed runs, "
          f"{len(other)} other (filtered by config).")

    if not fixed:
        raise RuntimeError("No fixed-seed runs to aggregate.")

    omega = fixed[0]["omega"]
    omega_me, A_me = _load_maxent()
    A_me_grid = np.interp(omega, omega_me, A_me)

    omega_de, A_de = _load_deac_for_cell()
    A_de_grid = np.interp(omega, omega_de, A_de) if A_de is not None else None

    # Stack spectra across seeds
    A_stack = np.stack([r["A_mean"] for r in fixed], axis=0)   # (n_seed, Nw)
    A_med   = np.median(A_stack, axis=0)
    A_lo    = A_stack.min(axis=0)
    A_hi    = A_stack.max(axis=0)

    # ---- Spectral overlay (per-seed) ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.5, 6.5),
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08}, sharex=True,
    )
    cmap = plt.cm.viridis
    for k, r in enumerate(fixed):
        c = cmap(0.10 + 0.75 * k / max(1, len(fixed) - 1))
        l2_me = _l2(omega, r["A_mean"], A_me_grid)
        ax_top.plot(omega, r["A_mean"], color=c, lw=1.4,
                    label=fr"DEC seed $={r['seed']}$ "
                          fr"($L^2_{{\rm ME}}={l2_me:.3f}$, "
                          fr"$\chi^2_{{\min}}={r['chi2_min']:.3f}$)")
        ax_bot.plot(omega, r["A_mean"] - A_me_grid, color=c, lw=1.0)
    if A_de_grid is not None:
        ax_top.plot(omega, A_de_grid, color="#D55E00", lw=1.0, alpha=0.85,
                    label="DEAC")
        ax_bot.plot(omega, A_de_grid - A_me_grid, color="#D55E00", lw=0.9,
                    alpha=0.85)
    ax_top.plot(omega, A_me_grid, color="#2C2C2C", lw=2.4, ls="--",
                label=r"MaxEnt (basis)")
    ax_top.set_ylabel(r"$\langle A(\omega)\rangle$")
    ax_top.set_xlim(-10, 10)
    ax_top.set_title(
        rf"Site-Holstein $(\beta,\Omega,n)=(10,1,1)$, LW + $P=10$. "
        rf"DecoderOnly seed sweep ($N={len(fixed)}$)"
    )
    ax_top.legend(frameon=False, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0)
    ax_bot.axhline(0.0, color="#2C2C2C", ls="--", lw=1.0, alpha=0.6)
    ax_bot.set_xlabel(r"$\omega$")
    ax_bot.set_ylabel(r"$A - A_{\rm ME}$")
    ax_bot.set_xlim(-10, 10)
    fig.savefig(os.path.join(OUT_DIR, "spectral_overlay.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Spectral band (median + min/max envelope) ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08}, sharex=True,
    )
    ax_top.fill_between(omega, A_lo, A_hi, color="#0072B2", alpha=0.20,
                        label=r"DEC seed envelope (min--max, $N=" + str(len(fixed)) + r"$)")
    ax_top.plot(omega, A_med, color="#0072B2", lw=1.6,
                label=r"DEC median")
    if A_de_grid is not None:
        ax_top.plot(omega, A_de_grid, color="#D55E00", lw=1.0, alpha=0.85,
                    label="DEAC")
    ax_top.plot(omega, A_me_grid, color="#2C2C2C", lw=2.4, ls="--",
                label=r"MaxEnt (basis)")
    ax_top.set_ylabel(r"$\langle A(\omega)\rangle$")
    ax_top.set_xlim(-10, 10)
    ax_top.set_title(
        rf"DecoderOnly seed-band on site-Holstein $(10,1,1)$ "
        rf"($N={len(fixed)}$ seeds)"
    )
    ax_top.legend(frameon=False, fontsize=9, loc="upper right")
    ax_bot.fill_between(omega, A_lo - A_me_grid, A_hi - A_me_grid,
                        color="#0072B2", alpha=0.20)
    ax_bot.plot(omega, A_med - A_me_grid, color="#0072B2", lw=1.4)
    if A_de_grid is not None:
        ax_bot.plot(omega, A_de_grid - A_me_grid, color="#D55E00", lw=0.9,
                    alpha=0.85)
    ax_bot.axhline(0.0, color="#2C2C2C", ls="--", lw=1.0, alpha=0.6)
    ax_bot.set_xlabel(r"$\omega$")
    ax_bot.set_ylabel(r"$A - A_{\rm ME}$")
    ax_bot.set_xlim(-10, 10)
    fig.savefig(os.path.join(OUT_DIR, "spectral_bands.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Markdown summary ----
    md = []
    md.append("# DecoderOnly seed sweep on site-Holstein (LW + P=10)\n")
    md.append("Cell: $(\\beta, \\Omega, n) = (10, 1, 1)$. "
              f"`BATCH_SIZE={FILTER_BATCH_SIZE}`, `NUM_POLES={FILTER_NUM_POLES}`, "
              f"`FINETUNE_EPOCHS={FILTER_EPOCHS}`. Architecture: DecoderOnly.\n")
    md.append("## Per-seed numbers\n")
    md.append("| seed | $\\chi^2_{\\min}$ | $\\chi^2_{\\rm final}$ "
              "| $L^2$ vs MaxEnt | $L^\\infty$ vs MaxEnt | run dir |")
    md.append("|---:|---:|---:|---:|---:|---|")
    rows_for_stats_l2   = []
    rows_for_stats_linf = []
    rows_for_stats_chi  = []
    for r in fixed:
        l2 = _l2(omega, r["A_mean"], A_me_grid)
        linf = _linf(r["A_mean"], A_me_grid)
        rows_for_stats_l2.append(l2)
        rows_for_stats_linf.append(linf)
        rows_for_stats_chi.append(r["chi2_min"])
        md.append(
            f"| {r['seed']} | {r['chi2_min']:.4f} | {r['chi2_final']:.4f} "
            f"| {l2:.4f} | {linf:.4f} | `{os.path.basename(r['run_dir'])}` |"
        )
    if random_seed_runs:
        md.append("\n## Random-seed reference (kept for context, not in stats)\n")
        md.append("| seed | $\\chi^2_{\\min}$ | $\\chi^2_{\\rm final}$ "
                  "| $L^2$ vs MaxEnt | $L^\\infty$ vs MaxEnt | run dir |")
        md.append("|---:|---:|---:|---:|---:|---|")
        for r in random_seed_runs:
            l2 = _l2(omega, r["A_mean"], A_me_grid)
            linf = _linf(r["A_mean"], A_me_grid)
            md.append(
                f"| (None) | {r['chi2_min']:.4f} | {r['chi2_final']:.4f} "
                f"| {l2:.4f} | {linf:.4f} | `{os.path.basename(r['run_dir'])}` |"
            )

    a_l2   = np.array(rows_for_stats_l2)
    a_linf = np.array(rows_for_stats_linf)
    a_chi  = np.array(rows_for_stats_chi)
    md.append("\n## Summary statistics over fixed seeds\n")
    md.append("| metric | mean | median | std | min | max |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for label, a in (("$\\chi^2_{\\min}$", a_chi),
                     ("$L^2$ vs MaxEnt",   a_l2),
                     ("$L^\\infty$ vs MaxEnt", a_linf)):
        md.append(f"| {label} | {a.mean():.4f} | {np.median(a):.4f} "
                  f"| {a.std(ddof=1):.4f} | {a.min():.4f} | {a.max():.4f} |")

    if A_de_grid is not None:
        l2_de   = _l2(omega, A_de_grid, A_me_grid)
        linf_de = _linf(A_de_grid, A_me_grid)
        md.append("\n## Reference baselines vs MaxEnt\n")
        md.append(f"- DEAC: $L^2 = {l2_de:.4f}$, $L^\\infty = {linf_de:.4f}$")

    md_text = "\n".join(md)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(md_text + "\n")
    print(md_text)
    print(f"\nPlots → {OUT_DIR}")


if __name__ == "__main__":
    main()
