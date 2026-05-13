"""
Analyse a multi-cell VAE run: AU diagnostic, latent-space scatter coloured
by cell, per-cell A(omega) bands vs the in-file DEAC reference (and MaxEnt
where available).

Usage:
    python pretrain/analyze_multicell.py <run_dir>
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pretrain.pretrain_losses import spectral_from_poles  # type: ignore


PROJ        = "/Users/gissa/Documents/AC/SPE-AC-VAE"
ANACONT_DIR = f"{PROJ}/MaxEnt_benchmark/out"

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


def _maxent_path(beta, omega, n):
    return os.path.join(
        ANACONT_DIR,
        f"anacont_real-site_holstein_b{beta:.2f}_w{omega:.2f}_n{n:.2f}",
        "summary_mean_fullcov.npz",
    )


def _l2(omega, a, b):
    return float(np.sqrt(np.trapezoid((a - b) ** 2, omega)))


def _linf(a, b):
    return float(np.max(np.abs(a - b)))


def analyze_run(run_dir):
    """Produce all multi-cell analysis plots + summary.md for a finished run.

    Called both from the CLI entry point and from `run_finetune_multicell.py`'s
    post-training hook so every multi-cell run gets a plots/ dir.
    """
    run_dir = os.path.abspath(run_dir)

    s = torch.load(os.path.join(run_dir, "summary.pt"), weights_only=False)
    poles    = s["poles"]      # (N, P) complex
    residues = s["residues"]
    cell_ids = s["cell_ids"].numpy()
    cells    = s["cells"]
    mu       = s["mu"].numpy()  # (N, L)
    var_mu   = s["var_mu_per_dim"].numpy()
    kl_per_dim = s["kl_per_dim"].numpy()
    n_active = int(s["active_units"])
    L_z      = int(s["latent_dim"])
    ws_ref   = s["ws_ref"].numpy()

    print(f"Multi-cell run: {os.path.basename(run_dir)}")
    print(f"  L_z = {L_z}    AU = {n_active}/{L_z}")
    print(f"  Var_x[mu] per dim: {var_mu}")
    print(f"  KL per dim:        {kl_per_dim}    KL_total = {kl_per_dim.sum():.4f}")
    for cid, c in enumerate(cells):
        mask = (cell_ids == cid)
        print(f"  cell {c['label']} (n={c['n']:.2f}, w={c['omega']:.2f}, "
              f"b={c['beta']:.1f}): {int(mask.sum())} samples, "
              f"mu_bar = {mu[mask].mean(axis=0)}")

    omega = torch.linspace(-20.0, 20.0, 1000)
    with torch.no_grad():
        A_all = spectral_from_poles(poles, residues, omega).numpy()  # (N, Nw)
    omega_np = omega.numpy()

    plt.rcParams.update(PLOT_RC)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Latent-space scatter ----
    fig, ax = plt.subplots(figsize=(6, 5))
    # Cell B is purple (#785EF0) rather than the Okabe-Ito orange so it
    # cannot be confused with DEAC's vermilion (#D55E00). Cells A/C/D keep
    # the canonical blue/green/magenta.
    cell_colors = ["#0072B2", "#785EF0", "#009E73", "#CC79A7"]
    if L_z == 1:
        for cid, c in enumerate(cells):
            mask = (cell_ids == cid)
            ax.scatter(mu[mask, 0], np.zeros(int(mask.sum())) + 0.05 * cid,
                       color=cell_colors[cid], s=44, edgecolor="k", alpha=0.8,
                       label=fr"cell {c['label']} ($n={c['n']:.2f}$, "
                             fr"$\Omega={c['omega']:.2f}$)")
        ax.set_xlabel(r"$\mu_z[0]$")
        ax.set_yticks([])
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    else:
        for cid, c in enumerate(cells):
            mask = (cell_ids == cid)
            ax.scatter(mu[mask, 0], mu[mask, 1],
                       color=cell_colors[cid], s=44, edgecolor="k", alpha=0.8,
                       label=fr"cell {c['label']} ($n={c['n']:.2f}$, "
                             fr"$\Omega={c['omega']:.2f}$)")
        ax.set_xlabel(r"$\mu_z[0]$")
        ax.set_ylabel(r"$\mu_z[1]$")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_title(rf"Encoder latent positions, AU $= {n_active}/{L_z}$")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    # Capture autoscaled limits so latent_path can render at the same
    # zoom — keeps the two PDFs visually comparable side-by-side.
    _scatter_xlim = ax.get_xlim()
    _scatter_ylim = ax.get_ylim()
    fig.savefig(os.path.join(plot_dir, "latent_scatter_by_cell.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Per-cell spectral panels ----
    n_cells = len(cells)
    fig, axes = plt.subplots(n_cells, 1, figsize=(9.5, 4.0 * n_cells), sharex=True)
    if n_cells == 1:
        axes = [axes]

    summary_lines = []
    summary_lines.append(f"# Multi-cell VAE analysis: {os.path.basename(run_dir)}\n")
    summary_lines.append(f"AU = {n_active}/{L_z}, KL_total = {kl_per_dim.sum():.4f}, "
                         f"chi2 best = {s['best_chi2']:.4f}\n")
    summary_lines.append("| cell | n | Omega | beta | N | mu_bar | "
                         "$L^2$ vs DEAC | $L^\\infty$ vs DEAC | $L^2$ vs MaxEnt |")
    summary_lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|")

    for cid, (ax, c) in enumerate(zip(axes, cells)):
        mask = (cell_ids == cid)
        A_c = A_all[mask]
        A_med = np.median(A_c, axis=0)
        A_lo  = A_c.min(axis=0)
        A_hi  = A_c.max(axis=0)

        # DEAC reference for this cell
        A_de = c["dos_ref"].numpy()
        A_de_grid = np.interp(omega_np, ws_ref, A_de)

        # MaxEnt — only if it exists for this cell
        me_path = _maxent_path(c["beta"], c["omega"], c["n"])
        A_me_grid = None
        if os.path.exists(me_path):
            d = dict(np.load(me_path, allow_pickle=True))
            A_me_grid = np.interp(omega_np, np.asarray(d["omega"]),
                                  np.asarray(d["A_opt"]))

        ax.fill_between(omega_np, A_lo, A_hi, color=cell_colors[cid], alpha=0.20,
                        label=fr"VAE band ($N={int(mask.sum())}$)")
        ax.plot(omega_np, A_med, color=cell_colors[cid], lw=1.6,
                label="VAE median")
        ax.plot(omega_np, A_de_grid, color="#D55E00", lw=1.0, alpha=0.85,
                label="DEAC")
        if A_me_grid is not None:
            ax.plot(omega_np, A_me_grid, color="#2C2C2C", lw=2.0, ls="--",
                    label="MaxEnt")
        ax.set_ylabel(r"$A(\omega)$")
        ax.set_title(
            rf"cell {c['label']}: $(\beta, \Omega, n) = "
            rf"({c['beta']:.1f}, {c['omega']:.2f}, {c['n']:.2f})$"
        )
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax.set_xlim(-10, 10)

        l2_de = _l2(omega_np, A_med, A_de_grid)
        linf_de = _linf(A_med, A_de_grid)
        l2_me = _l2(omega_np, A_med, A_me_grid) if A_me_grid is not None else None
        l2_me_str = f"{l2_me:.4f}" if l2_me is not None else "—"

        mu_c = mu[mask]
        mu_str = "[" + ", ".join(f"{x:+.3f}" for x in mu_c.mean(axis=0)) + "]"
        summary_lines.append(
            f"| {c['label']} | {c['n']:.2f} | {c['omega']:.2f} | {c['beta']:.1f} "
            f"| {int(mask.sum())} | {mu_str} "
            f"| {l2_de:.4f} | {linf_de:.4f} | {l2_me_str} |"
        )

    axes[-1].set_xlabel(r"$\omega$")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "spectral_per_cell.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- greens_per_cell.pdf: G(tau) data vs reconstruction, per cell ----
    G_input = s["inputs"].numpy()      # (N, L_tau)
    G_recon = s["recon"].numpy()       # (N, L_tau)
    L_tau   = G_input.shape[1]
    beta_run = float(s.get("beta", L_tau * 0.1))
    tau_grid = np.linspace(0.0, beta_run, L_tau, endpoint=False)

    fig, axes_g = plt.subplots(n_cells, 1, figsize=(9.5, 3.6 * n_cells),
                               sharex=True)
    if n_cells == 1:
        axes_g = [axes_g]
    for cid, (ax_g, c) in enumerate(zip(axes_g, cells)):
        mask = (cell_ids == cid)
        Gi_mean = G_input[mask].mean(axis=0)
        Gi_std  = G_input[mask].std(axis=0)
        Gr_mean = G_recon[mask].mean(axis=0)
        ax_g.fill_between(tau_grid, Gi_mean - Gi_std, Gi_mean + Gi_std,
                          color="#888888", alpha=0.25,
                          label=r"$G(\tau)$ data ($\pm 1\sigma$)")
        ax_g.plot(tau_grid, Gi_mean, color="#2C2C2C", lw=1.4,
                  label=r"$\langle G(\tau)\rangle$ data")
        ax_g.plot(tau_grid, Gr_mean, color=cell_colors[cid], lw=1.6,
                  ls="--", label=r"$\langle G(\tau)\rangle$ recon")
        ax_g.set_ylabel(r"$G(\tau)$")
        ax_g.set_title(
            rf"cell {c['label']}: $(\beta, \Omega, n) = ({c['beta']:.1f}, "
            rf"{c['omega']:.2f}, {c['n']:.2f})$"
        )
        ax_g.legend(frameon=False, fontsize=9, loc="best")
    axes_g[-1].set_xlabel(r"$\tau$")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "greens_per_cell.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Latent traversal: decode along z_A -> z_B (only meaningful if 2 cells) ----
    # Asks the question: as we move continuously through latent space from
    # cell A's cluster to cell B's, does A(omega) morph smoothly between
    # the two known spectra, or does it jump / produce nonsense? Smooth
    # morph -> the latent is a continuous coordinate, candidate for cube-
    # wide interpolation. Discontinuity -> just a discrete cluster label.
    if len(cells) == 2:
        # Need to reload the model for a forward pass at arbitrary z.
        from model2_leaky import VariationalAutoEncoder2 as VAE  # type: ignore
        import json as _json
        with open(os.path.join(run_dir, "params.json")) as f:
            params = _json.load(f)
        model = VAE(
            input_dim=params["INPUT_DIM"], num_poles=params["NUM_POLES"],
            beta=params["BETA"], N_nodes=params["N_NODES"],
            latent_dim=params["LATENT_DIM"],
            ph_symmetric=params.get("PH_SYMMETRIC", False),
        )
        model.load_state_dict(torch.load(
            os.path.join(run_dir, "model", "best_model_finetune.pth"),
            map_location="cpu", weights_only=True,
        ))
        model.eval()

        z_A = mu[cell_ids == 0].mean(axis=0)
        z_B = mu[cell_ids == 1].mean(axis=0)
        n_steps = 9
        ts = np.linspace(0.0, 1.0, n_steps)
        z_path = np.stack([(1 - t) * z_A + t * z_B for t in ts], axis=0)
        z_path_t = torch.tensor(z_path, dtype=torch.float32)
        with torch.no_grad():
            poles_p, residues_p = model.decode_poles_residues(z_path_t)
            A_path = spectral_from_poles(poles_p, residues_p, omega).numpy()

        # Cell-A and cell-B median spectra for context
        A_med_A = np.median(A_all[cell_ids == 0], axis=0)
        A_med_B = np.median(A_all[cell_ids == 1], axis=0)
        # Cell-A / B DEAC for context
        A_de_A_grid = np.interp(omega_np, ws_ref, cells[0]["dos_ref"].numpy())
        A_de_B_grid = np.interp(omega_np, ws_ref, cells[1]["dos_ref"].numpy())

        cmap = plt.cm.plasma

        # ---- latent_path.pdf: latent space with cluster scatter + traversal ----
        fig, ax_z = plt.subplots(figsize=(7.5, 5.5))
        # Draw order (back -> front):
        #   line (zorder=2) -> traversal X markers (3) -> cluster scatter (5)
        # so the cluster dots remain visible even where the t=0 and t=1
        # markers sit on top of them.
        ax_z.plot(z_path[:, 0], z_path[:, 1],
                  color="#888888", lw=1.0, ls="--", alpha=0.65, zorder=2)
        for k, t in enumerate(ts):
            ax_z.scatter(z_path[k, 0], z_path[k, 1],
                         color=cmap(0.10 + 0.75 * t), s=85,
                         marker="X", edgecolor="k", linewidths=1.0,
                         zorder=3)
        for cid, c in enumerate(cells):
            mask = (cell_ids == cid)
            ax_z.scatter(mu[mask, 0], mu[mask, 1],
                         color=cell_colors[cid], s=46,
                         edgecolor="k", alpha=0.85, zorder=5,
                         label=fr"cell {c['label']} ($n={c['n']:.2f}$, "
                               fr"$\Omega={c['omega']:.2f}$)")
        # Endpoint annotations: pulled toward the path interior so they
        # don't sit on the title (top) or x-axis label (bottom).
        ax_z.annotate(r"$\bar z_A$ ($t=0$)",
                      xy=(z_A[0], z_A[1]),
                      xytext=(z_A[0] + 0.10, z_A[1] - 0.10),
                      fontsize=10, ha="left", va="center")
        ax_z.annotate(r"$\bar z_B$ ($t=1$)",
                      xy=(z_B[0], z_B[1]),
                      xytext=(z_B[0] - 0.10, z_B[1] + 0.10),
                      fontsize=10, ha="right", va="center")
        ax_z.set_xlabel(r"$\mu_z[0]$")
        ax_z.set_ylabel(r"$\mu_z[1]$")
        ax_z.set_title(
            rf"Latent space + traversal path: "
            rf"AU $= {n_active}/{L_z}$, "
            rf"$\|\bar z_A - \bar z_B\| = {np.linalg.norm(z_A - z_B):.2f}$"
        )
        ax_z.legend(frameon=False, fontsize=9, loc="best")
        ax_z.grid(alpha=0.3)
        # Match latent_scatter_by_cell's autoscaled limits so the two
        # PDFs are viewable from the same perspective. Path markers and
        # endpoint annotations are inside the cluster span by construction
        # (z(t) is a convex combination of cluster centroids).
        ax_z.set_xlim(_scatter_xlim)
        ax_z.set_ylim(_scatter_ylim)
        sm_z = plt.cm.ScalarMappable(cmap=cmap,
                                     norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm_z.set_array([])
        plt.colorbar(sm_z, ax=ax_z,
                     label=r"$t$ along $z(t) = (1-t)\,\bar z_A + t\,\bar z_B$")
        fig.savefig(os.path.join(plot_dir, "latent_path.pdf"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- latent_traversal.pdf: decoded A(omega) along z(t) ----
        fig, ax_a = plt.subplots(figsize=(9.5, 5.0))
        for k, t in enumerate(ts):
            ax_a.plot(omega_np, A_path[k],
                      color=cmap(0.10 + 0.75 * t), lw=1.4, alpha=0.9)
        ax_a.plot(omega_np, A_med_A, color=cell_colors[0], lw=2.2, ls=":",
                  label=r"VAE median, cell A ($t=0$)")
        ax_a.plot(omega_np, A_med_B, color=cell_colors[1], lw=2.2, ls=":",
                  label=r"VAE median, cell B ($t=1$)")
        ax_a.plot(omega_np, A_de_A_grid, color=cell_colors[0], lw=1.0, alpha=0.45,
                  label="DEAC, cell A")
        ax_a.plot(omega_np, A_de_B_grid, color=cell_colors[1], lw=1.0, alpha=0.45,
                  label="DEAC, cell B")
        ax_a.set_xlabel(r"$\omega$")
        ax_a.set_ylabel(r"$A(\omega)$")
        ax_a.set_xlim(-10, 10)
        ax_a.set_title(
            rf"Decoded spectra along $z(t) = (1-t)\,\bar z_A + t\,\bar z_B$ "
            rf"({n_steps} steps)"
        )
        # Legend placed below the axes (outside the data area) in a 2-col
        # horizontal layout. lower-right and upper-right both overlap data
        # (DEAC cell B has a long tail past omega=5; cell B median peaks at
        # omega=4); below-axes is the only collision-free spot.
        ax_a.legend(frameon=False, fontsize=9,
                    loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    ncol=2)
        sm_a = plt.cm.ScalarMappable(cmap=cmap,
                                     norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm_a.set_array([])
        plt.colorbar(sm_a, ax=ax_a, label=r"$t$ (cell A $\to$ cell B)")
        fig.savefig(os.path.join(plot_dir, "latent_traversal.pdf"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Latent traversal: z_A = {z_A}, z_B = {z_B}, "
              f"||z_A - z_B|| = {np.linalg.norm(z_A - z_B):.3f}")

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(run_dir, "multicell_analysis.md"), "w") as f:
        f.write(summary_text + "\n")
    print()
    print(summary_text)
    print(f"\nPlots → {plot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    analyze_run(args.run_dir)


if __name__ == "__main__":
    main()
