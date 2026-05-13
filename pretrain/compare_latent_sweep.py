"""
Aggregate the LW + P=10 latent-dim sweep on site-Holstein (b=10, w=1, n=1)
into one comparison figure + a markdown summary table.

Reads each run's checkpoint, recomputes <A(omega)> with the encoder ON
(z = mu_z(x_i), per-sample averaged), and overlays all curves vs MaxEnt
(ana_cont, full cov) as the basis. Per project standard 11_Development_Standards
section "Cross-method comparison: MaxEnt as the residual basis".

Usage:
    python pretrain/compare_latent_sweep.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_process_real import HolsteinJLD2Dataset  # type: ignore
from pretrain.pretrain_losses import spectral_from_poles  # type: ignore


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

PROJ = "/Users/gissa/Documents/AC/SPE-AC-VAE"
RUNS = [
    (1, f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_z1_covlw-fresh-v2L-1"),
    (2, f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_z2_covlw-fresh-v2L-2"),
    (3, f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_z3_covlw-fresh-v2L-1"),
    (4, f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_z4_covlw-fresh-v2L-1"),
    (5, f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_z5_covlw-fresh-v2L-1"),
]
# DecoderOnly run (encoder-less). Lz tag is "dec" — labelled separately
# in plots / summary so the architectural difference is visible.
DEC_RUN = f"{PROJ}/out/finetune_real-site_holstein_b10.00_w1.00_n1.00_numpoles10_dec_covlw-fresh-vDEC-1"
MAXENT_NPZ = f"{PROJ}/MaxEnt_benchmark/out/anacont_real-site_holstein_b10.00_w1.00_n1.00/summary_mean_fullcov.npz"
OUT_DIR = f"{PROJ}/out/comparison_latent_sweep_lw_p10"

OMEGA_MIN, OMEGA_MAX, OMEGA_N = -20.0, 20.0, 1000


def _load_run(run_dir):
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)

    if params.get("ARCHITECTURE") == "DecoderOnly":
        from decoder_only import DecoderOnly  # type: ignore
        model = DecoderOnly(
            input_dim=params["INPUT_DIM"],
            num_poles=params["NUM_POLES"],
            beta=params["BETA"],
            N_nodes=params["N_NODES"],
            ph_symmetric=params.get("PH_SYMMETRIC", False),
        )
    else:
        from model2_leaky import VariationalAutoEncoder2 as VAE  # type: ignore
        model = VAE(
            input_dim=params["INPUT_DIM"],
            num_poles=params["NUM_POLES"],
            beta=params["BETA"],
            N_nodes=params["N_NODES"],
            latent_dim=params["LATENT_DIM"],
            ph_symmetric=params.get("PH_SYMMETRIC", False),
        )
    state = torch.load(
        os.path.join(run_dir, "model", "best_model_finetune.pth"),
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    ds = HolsteinJLD2Dataset(
        params["DATA_PATH"],
        n_idx=params["HOLSTEIN_N_IDX"],
        omega_idx=params["HOLSTEIN_OMEGA_IDX"],
        beta_idx=params["HOLSTEIN_BETA_IDX"],
    )
    loader = DataLoader(ds, batch_size=params["BATCH_SIZE"], shuffle=False)

    omega = torch.linspace(OMEGA_MIN, OMEGA_MAX, OMEGA_N)
    A_chunks = []
    mu_chunks = []
    sigma_chunks = []
    with torch.no_grad():
        for batch in loader:
            B = batch.shape[0]
            x = batch.view(B, params["INPUT_DIM"])
            # forward() with deterministic=True does z=mu for VAE and ignores
            # x entirely for DecoderOnly; both yield poles/residues directly.
            mu, logvar, _, poles, residues, _ = model(x, deterministic=True)
            sigma = torch.exp(0.5 * torch.clamp(logvar, -25, 25))
            A = spectral_from_poles(poles, residues, omega).numpy()
            A_chunks.append(A)
            mu_chunks.append(mu.numpy())
            sigma_chunks.append(sigma.numpy())
    A_all = np.concatenate(A_chunks, axis=0)         # (N, Nw)
    mu_all = np.concatenate(mu_chunks, axis=0)        # (N, L)
    sigma_all = np.concatenate(sigma_chunks, axis=0)  # (N, L)

    chi2 = np.load(os.path.join(run_dir, "losses", "chi2_losses_finetune.npy"))
    return {
        "params":   params,
        "omega":    omega.numpy(),
        "A_mean":   A_all.mean(axis=0),
        "A_per":    A_all,
        "mu":       mu_all,
        "sigma":    sigma_all,
        "chi2_min": float(np.min(chi2)),
        "var_mu":   mu_all.var(axis=0, ddof=0),
        "kl_total":
            0.5 * float(np.sum(
                mu_all ** 2 + sigma_all ** 2 - 1.0 - 2.0 * np.log(sigma_all + 1e-30)
            )) / mu_all.shape[0],
    }


def _load_maxent():
    d = dict(np.load(MAXENT_NPZ, allow_pickle=True))
    return d["omega"], d["A_opt"]


def _interp(x_src, y_src, x_tgt):
    return np.interp(x_tgt, x_src, y_src)


def _l2(omega, a, b):
    return float(np.sqrt(np.trapezoid((a - b) ** 2, omega)))


def _linf(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(PLOT_RC)

    rows = []
    for L, run_dir in RUNS:
        r = _load_run(run_dir)
        rows.append((L, run_dir, r))

    dec = _load_run(DEC_RUN) if os.path.isdir(DEC_RUN) else None

    omega = rows[0][2]["omega"]
    omega_me, A_me = _load_maxent()
    A_me_on_grid = _interp(omega_me, A_me, omega)

    # ---- spectral overlay: VAE Lz=1..5 + DecoderOnly + MaxEnt ----
    # Legend is parked outside the axes (right side); 7 entries with metric
    # parentheticals would otherwise sit on top of the data near omega = -3
    # (peak of L_z=1 at A ~ 0.21) and around the central pseudogap.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13.0, 6.5),
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08}, sharex=True,
    )
    cmap = plt.cm.plasma
    for k, (L, _, r) in enumerate(rows):
        c = cmap(0.10 + 0.75 * k / max(1, len(rows) - 1))
        au = int((r["var_mu"] > 1e-2).sum())
        l2_me = _l2(omega, r["A_mean"], A_me_on_grid)
        ax_top.plot(omega, r["A_mean"], color=c, lw=1.4,
                    label=fr"VAE $L_z={L}$ "
                          fr"(AU $={au}/{L}$, $L^2_{{\rm ME}}={l2_me:.3f}$, "
                          fr"$\chi^2_{{\min}}={r['chi2_min']:.3f}$)")
        ax_bot.plot(omega, r["A_mean"] - A_me_on_grid, color=c, lw=1.2)

    if dec is not None:
        l2_dec = _l2(omega, dec["A_mean"], A_me_on_grid)
        ax_top.plot(omega, dec["A_mean"], color="#1B7837", lw=2.0, ls="-",
                    label=fr"DecoderOnly "
                          fr"(no encoder, $L^2_{{\rm ME}}={l2_dec:.3f}$, "
                          fr"$\chi^2_{{\min}}={dec['chi2_min']:.3f}$)")
        ax_bot.plot(omega, dec["A_mean"] - A_me_on_grid,
                    color="#1B7837", lw=1.6)

    ax_top.plot(omega, A_me_on_grid, color="#2C2C2C", lw=2.4, ls="--",
                label=r"MaxEnt (ana\_cont, basis)")
    ax_top.set_ylabel(r"$\langle A(\omega)\rangle$")
    ax_top.set_xlim(-10, 10)
    ax_top.set_title(
        r"Site-Holstein $(\beta,\Omega,n)=(10,1,1)$, LW + $P=10$. "
        r"VAE $L_z\in\{1{-}5\}$ + DecoderOnly vs MaxEnt"
    )
    ax_top.legend(frameon=False, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0)

    ax_bot.axhline(0.0, color="#2C2C2C", ls="--", lw=1.0, alpha=0.6)
    ax_bot.set_xlabel(r"$\omega$")
    ax_bot.set_ylabel(r"$A - A_{\rm ME}$")
    ax_bot.set_xlim(-10, 10)

    fig.savefig(os.path.join(OUT_DIR, "spectral_avg_vs_Lz.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- mu_bar across dims, per run ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (L, _, r) in enumerate(rows):
        c = cmap(0.10 + 0.75 * k / max(1, len(rows) - 1))
        mu_bar = r["mu"].mean(axis=0)
        ax.scatter(np.full(L, L) + 0.06 * (np.arange(L) - (L-1)/2),
                   mu_bar, color=c, s=42, label=fr"$L_z = {L}$")
    ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5,
               label=r"prior $\mu = 0$")
    ax.set_xlabel(r"latent dim $L_z$ (point at $L_z + \epsilon\cdot d$)")
    ax.set_ylabel(r"$\bar\mu_d$ (encoder output, per dim)")
    ax.set_title(r"Encoder $\bar\mu$ per latent dim, across $L_z$ sweep")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_xticks([1, 2, 3, 4, 5])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mu_bar_per_dim_vs_Lz.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- markdown summary table ----
    md = []
    md.append("# Latent-dim sweep + DecoderOnly on site-Holstein (LW + P=10)\n")
    md.append("Cell: $(\\beta, \\Omega, n) = (10, 1, 1)$. "
              "Same hyperparameters across rows; only the architecture / "
              "`LATENT_DIM` differ. Single seed each (random).\n")
    md.append("| arch | $L_z$ | params | AU/$L_z$ | KL_total "
              "| $\\chi^2_{\\min}$ | $L^2$ vs MaxEnt | $L^\\infty$ vs MaxEnt "
              "| max $|\\bar\\mu_d|$ | min $\\bar\\sigma_d$ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for L, run_dir, r in rows:
        au = int((r["var_mu"] > 1e-2).sum())
        l2_me = _l2(omega, r["A_mean"], A_me_on_grid)
        linf_me = _linf(r["A_mean"], A_me_on_grid)
        mu_bar = r["mu"].mean(axis=0)
        sigma_bar = r["sigma"].mean(axis=0)
        n_params = r["params"].get("N_TRAINABLE_PARAMS", "—")
        md.append(
            f"| VAE | {L} | {n_params if n_params == '—' else f'{n_params:,}'} "
            f"| {au}/{L} | {r['kl_total']:.3f} "
            f"| {r['chi2_min']:.4f} "
            f"| {l2_me:.4f} | {linf_me:.4f} "
            f"| {np.max(np.abs(mu_bar)):.3f} | {np.min(sigma_bar):.2e} |"
        )
    if dec is not None:
        l2_dec = _l2(omega, dec["A_mean"], A_me_on_grid)
        linf_dec = _linf(dec["A_mean"], A_me_on_grid)
        md.append(
            f"| **DecoderOnly** | — | "
            f"{dec['params'].get('N_TRAINABLE_PARAMS'):,} "
            f"| 0/0 | 0.000 "
            f"| {dec['chi2_min']:.4f} "
            f"| {l2_dec:.4f} | {linf_dec:.4f} "
            f"| — | — |"
        )
    md.append("")
    md.append("Output dirs:")
    for L, run_dir, _ in rows:
        md.append(f"- VAE $L_z = {L}$: `{os.path.basename(run_dir)}`")
    if dec is not None:
        md.append(f"- DecoderOnly: `{os.path.basename(DEC_RUN)}`")
    md_text = "\n".join(md)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(md_text + "\n")
    print(md_text)
    print()
    print(f"Plots → {OUT_DIR}")


if __name__ == "__main__":
    main()
