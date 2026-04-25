"""
Visualization for pretraining and fine-tuning results.

Plots:
  1. G(tau) input vs reconstructed for selected samples
  2. A(omega) predicted vs target Gaussian for selected samples
  3. Training/validation loss curves with all components
  4. Pole/residue visualization in the complex plane

Usage:
    python pretrain/plot_results.py --eval_file VAE_Library/pretrain_synthetic_numpoles2-pretrained/pretrain_eval.pt
    python pretrain/plot_results.py --loss_dir VAE_Library/pretrain_synthetic_numpoles2-pretrained/losses --tag pretrain
    python pretrain/plot_results.py --summary_file VAE_Library/.../summary.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pretrain.pretrain_losses import spectral_from_poles  # type: ignore


# ---------------------------------------------------------------------------
# Color and style defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    
    "font.size": 12,        # General font size for labels and legends
    "axes.labelsize": 13,   # Font size for axis labels
    "xtick.labelsize": 12,   # Font size for x-axis tick labels
    "ytick.labelsize": 12,   # Font size for y-axis tick labels
    "legend.fontsize": 10,   # Font size for legend
    "figure.dpi": 120,
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"

})

COLORS = {
    "input": "#2196F3",
    "recon": "#F44336",
    "target": "#4CAF50",
    "pred": "#FF9800",
    "pole_re": "#9C27B0",
    "pole_im": "#009688",
}


# ---------------------------------------------------------------------------
# Parameter annotation helpers
# ---------------------------------------------------------------------------

def _fmt_val(v):
    """Format a loss weight for display: 0 → OFF, small/large → LaTeX sci notation."""
    if v == 0:
        return r"\mathrm{OFF}"
    if abs(v) < 0.01 or abs(v) >= 1000:
        s = f"{v:.2e}"
        mantissa, exp = s.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_int = int(exp)
        if mantissa in ("1", "-1"):
            return rf"10^{{{exp_int}}}" if mantissa == "1" else rf"-10^{{{exp_int}}}"
        return rf"{mantissa}\!\times\!10^{{{exp_int}}}"
    return f"{v:g}"


def _build_loss_annotation(fig, params):
    """Add two text lines to the figure bottom: run identity and loss weights."""
    def tex(s):
        return str(s).replace("_", r"\_")

    # --- Line 1: run identity ---
    parts = []
    if params.get("sID"):
        parts.append(rf"\texttt{{{tex(params['sID'])}}}")
    if params.get("MODEL_VERSION"):
        parts.append(rf"model~v{tex(params['MODEL_VERSION'])}")
    if params.get("SPECTRAL_TYPE") and params.get("DATA_SOURCE") != "real":
        s = tex(params["SPECTRAL_TYPE"])
        ns = params.get("NOISE_S")
        noise_str = rf",~$s={ns:.0e}$" if isinstance(ns, float) else ""
        parts.append(rf"{s}{noise_str}")
    lr = params.get("FINETUNE_LR", params.get("PRETRAIN_LR"))
    if lr is not None:
        parts.append(rf"LR$={_fmt_val(lr)}$")
    if params.get("FINETUNE_EPOCHS"):
        parts.append(rf"{params['FINETUNE_EPOCHS']}~ep")
    if params.get("BATCH_SIZE"):
        parts.append(rf"batch~{params['BATCH_SIZE']}")
    line1 = r"~$|$~".join(parts)

    # --- Line 2: active vs inactive losses ---
    loss_defs = [
        (r"\lambda_{\chi^2}",        params.get("LAMBDA_CHI2",  1.0)),
        (r"\alpha_\mathrm{KL}",      params.get("ALPHA_KL",     0.0)),
        (r"\eta_0",                  params.get("ETA0",          0.0)),
        (r"\eta_2",                  params.get("ETA2",          0.0)),
        (r"\eta_4",                  params.get("ETA4",          0.0)),
        (r"\lambda_\mathrm{smooth}", params.get("LAMBDA_SMOOTH", 0.0)),
        (r"\lambda_\mathrm{pos}",    params.get("LAMBDA_POS",    0.0)),
    ]
    active, inactive = [], []
    for label, val in loss_defs:
        if val > 0:
            active.append(rf"${label}={_fmt_val(val)}$")
        else:
            inactive.append(rf"${label}$")

    off_str = r"~$|$~".join(inactive) if inactive else r"none"
    line2 = (
        r"\textbf{Active:}~" + r"~$|$~".join(active)
        + r"~~~\textbf{Off:}~" + off_str
    )

    fig.text(0.5, -0.02, line1, ha="center", fontsize=9, style="italic")
    fig.text(0.5, -0.06, line2, ha="center", fontsize=9)


# ---------------------------------------------------------------------------
# G(tau) comparison plots
# ---------------------------------------------------------------------------

def plot_gtau_comparison(G_input, G_recon, beta, n_samples=6, noise_var=None, save_path=None):
    """Plot G(tau) input vs reconstructed for selected samples.

    Parameters
    ----------
    G_input  : tensor (N, L_tau)
    G_recon  : tensor (N, L_tau)
    beta     : float
    n_samples: int, number of samples to plot
    save_path: str or None
    """
    G_input = G_input.numpy() if isinstance(G_input, torch.Tensor) else G_input
    G_recon = G_recon.numpy() if isinstance(G_recon, torch.Tensor) else G_recon

    N, L_tau = G_input.shape
    taus = np.linspace(0, beta - beta / L_tau, L_tau)
    n_samples = min(n_samples, N)

    cols = min(3, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i in range(n_samples):
        ax = axes[i // cols, i % cols]
        ax.plot(taus, G_input[i], "o", color=COLORS["input"], markersize=2, alpha=0.6, label="Input")
        ax.plot(taus, G_recon[i], "-", color=COLORS["recon"], linewidth=1.5, label="Reconstructed")
        mse = np.mean((G_input[i] - G_recon[i]) ** 2)
        if noise_var is not None:
            chi2_est = mse / noise_var
            ax.set_title(rf"Sample {i+1}  (MSE$={mse:.2e}$,  $\hat{{\chi}}^2\approx{chi2_est:.2f}$)")
        else:
            ax.set_title(rf"Sample {i+1}  (MSE$={mse:.2e}$)")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$G(\tau)$")
        ax.legend(loc="upper right")

    # Hide unused subplots
    for i in range(n_samples, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(r"$G(\tau)$ Reconstruction", fontsize=16, y=1.02)
    note = (
        r"MSE $= \frac{1}{L_\tau}\sum_\tau(G_\mathrm{recon}-G_\mathrm{data})^2$"
        r"$\quad\hat{\chi}^2 = \mathrm{MSE}/\bar{\sigma}^2$"
        r"$\quad$ at $\chi^2\!\to\!1$: MSE $\to\bar{\sigma}^2$ (noise variance)"
    )
    fig.text(0.5, -0.01, note, ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# A(omega) comparison plots
# ---------------------------------------------------------------------------

def plot_spectral_comparison(poles, residues, mu_targets, sigma_targets,
                             n_samples=6, wmin=-10, wmax=10, Nw=1000, save_path=None):
    """Plot predicted A(omega) vs target Gaussian for selected samples.

    Parameters
    ----------
    poles         : complex tensor (N, P)
    residues      : complex tensor (N, P)
    mu_targets    : tensor (N,)
    sigma_targets : tensor (N,)
    n_samples     : int
    save_path     : str or None
    """
    if not isinstance(poles, torch.Tensor):
        poles = torch.tensor(poles)
    if not isinstance(residues, torch.Tensor):
        residues = torch.tensor(residues)
    if not isinstance(mu_targets, torch.Tensor):
        mu_targets = torch.tensor(mu_targets, dtype=torch.float32)
    if not isinstance(sigma_targets, torch.Tensor):
        sigma_targets = torch.tensor(sigma_targets, dtype=torch.float32)

    N = poles.shape[0]
    n_samples = min(n_samples, N)
    omegas = torch.linspace(wmin, wmax, Nw)

    with torch.no_grad():
        A_pred = spectral_from_poles(poles[:n_samples], residues[:n_samples], omegas)

    A_pred = A_pred.numpy()
    omegas_np = omegas.numpy()

    cols = min(3, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i in range(n_samples):
        ax = axes[i // cols, i % cols]

        # Target Gaussian
        mu_i = mu_targets[i].item()
        sig_i = sigma_targets[i].item()
        A_target = np.exp(-0.5 * ((omegas_np - mu_i) / sig_i) ** 2) / (sig_i * np.sqrt(2 * np.pi))

        ax.fill_between(omegas_np, A_target, alpha=0.2, color=COLORS["target"])
        ax.plot(omegas_np, A_target, "-", color=COLORS["target"], linewidth=1.5,
                label=rf"Target $\mathcal{{N}}({mu_i:.2f}, {sig_i:.2f})$")
        ax.plot(omegas_np, A_pred[i], "-", color=COLORS["pred"], linewidth=1.5, label="Predicted")

        # Mark pole positions
        eps = poles[i].real.numpy()
        gamma = -poles[i].imag.numpy()
        for p in range(len(eps)):
            ax.axvline(eps[p], color=COLORS["pole_re"], linestyle="--", alpha=0.4, linewidth=0.8)

        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$A(\omega)$")
        ax.set_xlim(wmin, wmax)
        ax.legend(loc="upper right")

    for i in range(n_samples, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(r"Spectral Function $A(\omega)$ Recovery", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(loss_dir, tag="pretrain", save_path=None, params=None):
    """Plot training and validation loss curves with all components.

    Handles both pretrain losses (chi2, spec, smooth, kl) and
    finetune losses (chi2, smooth, kl, neg_green, neg_second).

    Parameters
    ----------
    loss_dir  : str, directory containing .npy loss files
    tag       : str, loss file tag
    save_path : str or None
    params    : dict or None — run config for annotation; auto-loaded from
                params.json in parent of loss_dir if not provided
    """
    # Auto-load params from the output directory if not passed explicitly
    if params is None:
        candidate = os.path.join(os.path.dirname(loss_dir), "params.json")
        if os.path.exists(candidate):
            with open(candidate) as _f:
                params = json.load(_f)
    # Try loading all possible loss files
    possible_files = {
        "train": f"train_losses_{tag}.npy",
        "val": f"val_losses_{tag}.npy",
        "chi2": f"chi2_losses_{tag}.npy",
        "spec": f"spec_losses_{tag}.npy",
        "smooth": f"smooth_losses_{tag}.npy",
        "kl": f"kl_losses_{tag}.npy",
        "moment": f"moment_losses_{tag}.npy",
        "recon": f"recon_losses_{tag}.npy",
        "pos": f"pos_losses_{tag}.npy",
        "neg_green": f"neg_green_losses_{tag}.npy",
        "neg_second": f"neg_second_losses_{tag}.npy",
    }

    losses = {}
    for key, fname in possible_files.items():
        fpath = os.path.join(loss_dir, fname)
        if os.path.exists(fpath):
            losses[key] = np.load(fpath)

    if not losses:
        print(f"No loss files found in {loss_dir} with tag={tag}")
        return None

    # Detect if this is a finetune run (has neg_green/neg_second)
    has_penalties = "neg_green" in losses or "neg_second" in losses

    if has_penalties:
        # Finetune layout: 2x4 grid
        fig = plt.figure(figsize=(22, 8))
        gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    else:
        # Pretrain layout: 2x3 grid (to accommodate moment loss)
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top-left: Total train/val loss
    ax1 = fig.add_subplot(gs[0, 0])
    if "train" in losses:
        ax1.semilogy(losses["train"], label="Train", color="#1976D2", linewidth=1.5)
    if "val" in losses:
        ax1.semilogy(losses["val"], label="Val", color="#D32F2F", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-middle: Chi-squared loss
    ax2 = fig.add_subplot(gs[0, 1])
    if "chi2" in losses:
        ax2.semilogy(losses["chi2"], label=r"$\chi^2$", color="#7B1FA2", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"$\chi^2 / L_\tau$")
    ax2.set_title(r"$\chi^2$ Loss ($\chi^2/L_\tau \to 1$ is ideal)")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if has_penalties:
        # Top-right (0,2): Smoothness + KL
        ax3 = fig.add_subplot(gs[0, 2])
        if "smooth" in losses:
            ax3.semilogy(losses["smooth"], label="Smoothness", color="#F57C00", linewidth=1.5)
        if "kl" in losses:
            ax3.semilogy(np.maximum(losses["kl"], 1e-10), label="KL", color="#0097A7",
                         linewidth=1.5, linestyle="--")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.set_title("Smoothness + KL")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Top-far-right (0,3): Spectral positivity
        ax3b = fig.add_subplot(gs[0, 3])
        if "pos" in losses:
            ax3b.semilogy(np.maximum(losses["pos"], 1e-10), label=r"Positivity", color="#FF5722", linewidth=1.5)
        ax3b.set_xlabel("Epoch")
        ax3b.set_ylabel("Penalty")
        ax3b.set_title(r"Spectral Positivity $\int \mathrm{ReLU}(-A)^2 d\omega$")
        ax3b.legend()
        ax3b.grid(True, alpha=0.3)

        # Bottom-left (1,0): Negative Green's penalty
        ax4 = fig.add_subplot(gs[1, 0])
        if "neg_green" in losses:
            ax4.semilogy(losses["neg_green"], label=r"Neg $G(\tau)$", color="#E91E63", linewidth=1.5)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Penalty")
        ax4.set_title(r"Negative $G(\tau)$ Penalty")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Bottom (1,1): Negative second derivative penalty
        ax5 = fig.add_subplot(gs[1, 1])
        if "neg_second" in losses:
            ax5.semilogy(losses["neg_second"], label=r"Neg $G''(\tau)$", color="#795548", linewidth=1.5)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Penalty")
        ax5.set_title(r"Negative $G''(\tau)$ Penalty")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Bottom (1,2-3 spanning): All components overlaid
        ax6 = fig.add_subplot(gs[1, 2:])
        component_colors = {
            "chi2": ("#7B1FA2", r"$\chi^2$"),
            "smooth": ("#F57C00", "Smooth"),
            "pos": ("#FF5722", "Positivity"),
            "kl": ("#0097A7", "KL"),
            "neg_green": ("#E91E63", r"Neg $G$"),
            "neg_second": ("#795548", r"Neg $G''$"),
        }
        for key, (color, label) in component_colors.items():
            if key in losses:
                vals = np.maximum(losses[key], 1e-10)
                ax6.semilogy(vals, label=label, color=color, linewidth=1.2, alpha=0.8)
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Loss")
        ax6.set_title("All Components")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

    else:
        # Pretrain layout (no penalties): 2x3 grid
        # Bottom-left: Spectral MSE
        ax3 = fig.add_subplot(gs[1, 0])
        if "spec" in losses:
            ax3.semilogy(losses["spec"], label="Spectral MSE", color="#388E3C", linewidth=1.5)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Spectral MSE")
        ax3.set_title(r"Spectral MSE: $\int (A_{pred} - A_{target})^2 d\omega$")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom-middle: Smoothness and KL
        ax4 = fig.add_subplot(gs[1, 1])
        if "smooth" in losses:
            ax4.semilogy(losses["smooth"], label="Smoothness", color="#F57C00", linewidth=1.5)
        if "kl" in losses:
            ax4.semilogy(np.maximum(losses["kl"], 1e-10), label="KL", color="#0097A7",
                         linewidth=1.5, linestyle="--")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss")
        ax4.set_title("Smoothness + KL")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Top-right (0,2): Chi-squared (already plotted in ax2 for col 1; reuse col 2 top for chi2 detail)
        ax5 = fig.add_subplot(gs[0, 2])
        if "chi2" in losses:
            ax5.semilogy(losses["chi2"], label=r"$\chi^2$", color="#7B1FA2", linewidth=1.5)
        ax5.set_xlabel("Epoch")
        ax5.set_title(r"$\chi^2$ detail")
        ax5.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Bottom-right: Moment loss
        ax6 = fig.add_subplot(gs[1, 2])
        if "moment" in losses:
            ax6.semilogy(np.maximum(losses["moment"], 1e-10), label="Moment loss",
                         color="#E91E63", linewidth=1.5)
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Loss")
        ax6.set_title(r"Spectral Moment Loss ($M_1$, $M_2$)")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves ({tag})", fontsize=16, y=1.01)
    if params is not None:
        _build_loss_annotation(fig, params)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Pole/residue visualization
# ---------------------------------------------------------------------------

def plot_poles_residues(poles, residues, n_samples=6, save_path=None):
    """Plot poles and residues in the complex plane.

    Parameters
    ----------
    poles    : complex tensor (N, P)
    residues : complex tensor (N, P)
    n_samples: int
    save_path: str or None
    """
    poles = poles.numpy() if isinstance(poles, torch.Tensor) else poles
    residues = residues.numpy() if isinstance(residues, torch.Tensor) else residues

    N = poles.shape[0]
    n_samples = min(n_samples, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Poles in complex plane
    for i in range(n_samples):
        eps = poles[i].real
        gamma = poles[i].imag
        ax1.scatter(eps, gamma, s=40, alpha=0.7, label=f"Sample {i+1}" if i < 4 else None)

    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_xlabel(r"$\epsilon_p$ (Re)")
    ax1.set_ylabel(r"$\gamma_p$ (Im)")
    ax1.set_title("Poles (lower half-plane)")
    if n_samples <= 4:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residues
    for i in range(n_samples):
        a = residues[i].real
        b = residues[i].imag
        ax2.scatter(a, b, s=40, alpha=0.7, label=f"Sample {i+1}" if i < 4 else None)

    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel(r"$a_p$ (Re)")
    ax2.set_ylabel(r"$b_p$ (Im)")
    ax2.set_title("Residues")
    if n_samples <= 4:
        ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Predicted A(omega) from poles/residues (no target needed)
# ---------------------------------------------------------------------------

def plot_spectral_predicted(poles, residues, A_input=None, omega_input=None,
                            n_samples=6, wmin=-20, wmax=20, Nw=1000, save_path=None,
                            poles_from_avg=None, residues_from_avg=None):
    """Plot predicted A(omega) from poles/residues for selected samples.

    Each sample slot has two stacked panels:
      - Top: per-sample A_i(omega), overlaid with ground truth (if any) and
             the reference A(omega | <G(tau)>) computed from poles_from_avg /
             residues_from_avg (if provided).
      - Bottom: difference panel A_i(omega) - A(omega | <G>), annotated with
             the per-sample SC-L2 = int(diff^2) dw and SC-Linfty = max|diff|.

    Parameters
    ----------
    poles, residues          : complex tensor (N, P), per-sample VAE outputs
    A_input, omega_input     : ground-truth spectrum + grid (optional)
    poles_from_avg,
    residues_from_avg        : complex tensor (P,) — VAE output from <G>;
                               drives the overlay and the per-sample SC values
    n_samples, wmin, wmax, Nw, save_path : usual
    """
    if not isinstance(poles, torch.Tensor):
        poles = torch.tensor(poles)
    if not isinstance(residues, torch.Tensor):
        residues = torch.tensor(residues)

    N = poles.shape[0]
    n_samples = min(n_samples, N)
    omegas = torch.linspace(wmin, wmax, Nw)

    with torch.no_grad():
        A_pred = spectral_from_poles(poles[:n_samples], residues[:n_samples], omegas)

    A_pred = A_pred.numpy()
    omegas_np = omegas.numpy()

    A_ref = None
    if poles_from_avg is not None and residues_from_avg is not None:
        if not isinstance(poles_from_avg, torch.Tensor):
            poles_from_avg = torch.tensor(poles_from_avg)
        if not isinstance(residues_from_avg, torch.Tensor):
            residues_from_avg = torch.tensor(residues_from_avg)
        with torch.no_grad():
            A_ref = spectral_from_poles(
                poles_from_avg.unsqueeze(0),
                residues_from_avg.unsqueeze(0),
                omegas,
            ).squeeze(0).numpy()

    cols = min(3, n_samples)
    rows = (n_samples + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    outer = GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.32)

    for i in range(n_samples):
        r, c = i // cols, i % cols
        if A_ref is not None:
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, c],
                height_ratios=[3, 1], hspace=0.08,
            )
            ax_main = fig.add_subplot(inner[0])
            ax_diff = fig.add_subplot(inner[1], sharex=ax_main)
        else:
            ax_main = fig.add_subplot(outer[r, c])
            ax_diff = None

        # Ground truth
        if A_input is not None and omega_input is not None:
            ax_main.fill_between(omega_input, A_input, alpha=0.15, color=COLORS["target"])
            ax_main.plot(omega_input, A_input, "-", color=COLORS["target"], linewidth=1.5,
                         label="Input (exact)")

        # Per-sample prediction
        ax_main.plot(omegas_np, A_pred[i], "-", color=COLORS["pred"], linewidth=1.5,
                     label=r"Predicted ($z = \mu$)")

        # Reference: A from <G> (also the SC reference)
        if A_ref is not None:
            ax_main.plot(omegas_np, A_ref, "--", color="#8E24AA", linewidth=1.5,
                         alpha=0.9, label=r"$A(\omega\,|\,\langle G(\tau)\rangle)$")

        # Pole positions
        eps = poles[i].real.numpy()
        for p in range(len(eps)):
            ax_main.axvline(eps[p], color=COLORS["pole_re"], linestyle="--",
                            alpha=0.4, linewidth=0.8)

        ax_main.set_title(f"Sample {i+1}")
        ax_main.set_ylabel(r"$A(\omega)$")
        ax_main.set_xlim(wmin, wmax)
        ax_main.legend(loc="upper right", fontsize=8)
        if ax_diff is not None:
            plt.setp(ax_main.get_xticklabels(), visible=False)
        else:
            ax_main.set_xlabel(r"$\omega$")

        # Difference panel + per-sample SC annotation
        if ax_diff is not None:
            diff = A_pred[i] - A_ref
            sc_l2_i   = float(np.trapezoid(diff ** 2, omegas_np))
            sc_linf_i = float(np.max(np.abs(diff)))
            ax_diff.axhline(0, color="gray", lw=0.6, alpha=0.5)
            ax_diff.plot(omegas_np, diff, "-", color=COLORS["pred"], linewidth=1.0)
            ax_diff.fill_between(omegas_np, diff, alpha=0.20, color=COLORS["pred"])
            ax_diff.text(
                0.02, 0.92,
                rf"SC-$L^2$ = {sc_l2_i:.4f}     SC-$L^\infty$ = {sc_linf_i:.4f}",
                transform=ax_diff.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
            )
            ax_diff.set_xlabel(r"$\omega$")
            ax_diff.set_ylabel(r"$\Delta A$", fontsize=10)
            ax_diff.set_xlim(wmin, wmax)
            ax_diff.grid(True, alpha=0.3)

    fig.suptitle(
        r"Predicted $A(\omega) = -\frac{1}{\pi}\sum_p \mathrm{Im}"
        r"\!\left(\frac{r_p}{\omega - s_p}\right)$"
        "\n"
        r"Deterministic $z = \mu$; dashed verticals: pole positions $\epsilon_p$"
        + (r"; difference panel: $A_i(\omega) - A(\omega|\langle G\rangle)$"
           if A_ref is not None else ""),
        fontsize=13, y=1.02,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_spectral_average(poles, residues, A_input=None, omega_input=None,
                          wmin=-20, wmax=20, Nw=1000, suptitle=None, save_path=None,
                          poles_from_avg=None, residues_from_avg=None):
    """Plot the average predicted A(omega) with ground truth overlay and difference.

    Two panels:
      - Top: average predicted A(omega) vs ground truth with +/- 1 sigma band
      - Bottom: pointwise difference (predicted - input) with integrated error

    Parameters
    ----------
    poles       : complex tensor (N, P)
    residues    : complex tensor (N, P)
    A_input     : ndarray (N_omega,) or None, ground truth spectral function
    omega_input : ndarray (N_omega,) or None, omega grid for ground truth
    wmin, wmax  : float
    Nw          : int
    save_path   : str or None
    """
    if not isinstance(poles, torch.Tensor):
        poles = torch.tensor(poles)
    if not isinstance(residues, torch.Tensor):
        residues = torch.tensor(residues)

    omegas = torch.linspace(wmin, wmax, Nw)

    with torch.no_grad():
        A_all = spectral_from_poles(poles, residues, omegas)  # (N, Nw)

    A_all = A_all.numpy()
    omegas_np = omegas.numpy()

    A_mean = A_all.mean(axis=0)
    A_std = A_all.std(axis=0)

    A_from_avg = None
    if poles_from_avg is not None and residues_from_avg is not None:
        if not isinstance(poles_from_avg, torch.Tensor):
            poles_from_avg = torch.tensor(poles_from_avg)
        if not isinstance(residues_from_avg, torch.Tensor):
            residues_from_avg = torch.tensor(residues_from_avg)
        with torch.no_grad():
            A_from_avg = spectral_from_poles(
                poles_from_avg.unsqueeze(0), residues_from_avg.unsqueeze(0), omegas
            ).squeeze(0).numpy()

    # Average pole positions
    eps_mean = poles.real.mean(dim=0).numpy()

    has_input = A_input is not None and omega_input is not None

    if has_input:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))

    # --- Top panel: overlay ---
    if has_input:
        ax1.fill_between(omega_input, A_input, alpha=0.2, color=COLORS["target"])
        ax1.plot(omega_input, A_input, "-", color=COLORS["target"], linewidth=2,
                 label="Input (exact)")

    ax1.plot(omegas_np, A_mean, "-", color=COLORS["pred"], linewidth=2,
             label=r"$\langle A(\omega) \rangle$ (mean over dataset)")
    ax1.fill_between(omegas_np, A_mean - A_std, A_mean + A_std,
                     alpha=0.2, color=COLORS["pred"],
                     label=r"$\pm 1\sigma$ (variation across dataset samples)")

    if A_from_avg is not None:
        ax1.plot(omegas_np, A_from_avg, ":", color="#8E24AA", linewidth=2,
                 label=r"$A(\omega \mid \langle G(\tau)\rangle)$ (single forward pass)")

    # Average pole positions — one entry per pole bloats the legend at high
    # NUM_POLES, so the per-pole verticals are unlabeled and a single proxy
    # entry covers them all in red.
    for p in range(len(eps_mean)):
        ax1.axvline(eps_mean[p], color="tab:green", linestyle="--",
                    alpha=0.5, linewidth=1.0)
    if len(eps_mean) > 0:
        ax1.axvline(eps_mean[0], color="tab:green", linestyle="--",
                    alpha=0.5, linewidth=1.0, label=r"pole positions")

    ax1.set_ylabel(r"$A(\omega)$")
    ax1.set_title(r"Dataset-averaged $A(\omega)$: $\langle\cdot\rangle$ and $\pm 1\sigma$ over all samples")
    ax1.set_xlim(wmin, wmax)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: difference ---
    if has_input:
        # Interpolate ground truth onto prediction grid
        A_input_interp = np.interp(omegas_np, omega_input, A_input)
        diff = A_mean - A_input_interp

        ax2.plot(omegas_np, diff, "-", color="#D32F2F", linewidth=1.5,
                 label=r"$\langle A\rangle - A_\mathrm{in}$")
        ax2.fill_between(omegas_np, diff, alpha=0.2, color="#D32F2F")
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

        # Integrated absolute error
        dw = (wmax - wmin) / (Nw - 1)
        int_abs_err = np.sum(np.abs(diff)) * dw
        int_sq_err = np.sqrt(np.sum(diff ** 2) * dw)
        max_abs_err = np.max(np.abs(diff))

        title = (rf"$\int |{{\Delta A}}| d\omega = {int_abs_err:.4f}$, "
                 rf"$\sqrt{{\int {{\Delta A}}^2 d\omega}} = {int_sq_err:.4f}$, "
                 rf"$\max |{{\Delta A}}| = {max_abs_err:.4f}$")

        if A_from_avg is not None:
            diff_avg = A_from_avg - A_input_interp
            ax2.plot(omegas_np, diff_avg, ":", color="#8E24AA", linewidth=1.5,
                     label=r"$A(\langle G\rangle) - A_\mathrm{in}$")
            int_abs_avg = np.sum(np.abs(diff_avg)) * dw
            int_sq_avg  = np.sqrt(np.sum(diff_avg ** 2) * dw)
            title += (rf"   |   from $\langle G\rangle$: "
                      rf"$\int|\Delta A|d\omega = {int_abs_avg:.4f}$, "
                      rf"$\sqrt{{\int \Delta A^2 d\omega}} = {int_sq_avg:.4f}$")
            ax2.legend(fontsize=9, loc="upper right")

        ax2.set_xlabel(r"$\omega$")
        ax2.set_ylabel(r"$\Delta A(\omega)$")
        ax2.set_title(title)
        ax2.grid(True, alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# All-in-one plot from eval file
# ---------------------------------------------------------------------------

def plot_pretrain_eval(eval_file, output_dir=None):
    """Generate all plots from a pretrain_eval.pt file.

    Parameters
    ----------
    eval_file  : str, path to pretrain_eval.pt
    output_dir : str or None, directory to save plots (defaults to same dir as eval_file)
    """
    d = torch.load(eval_file, map_location="cpu", weights_only=False)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(eval_file), "plots")
    os.makedirs(output_dir, exist_ok=True)

    beta = d["beta"]
    G_input = d["G_input"]
    G_recon = d["G_recon"]
    poles = d["poles"]
    residues = d["residues"]

    # G(tau) comparison
    plot_gtau_comparison(G_input, G_recon, beta, n_samples=6,
                         save_path=os.path.join(output_dir, "gtau_comparison.pdf"))

    # A(omega) comparison (only if mu/sigma targets are available)
    if "mu_targets" in d and "sigma_targets" in d:
        plot_spectral_comparison(poles, residues, d["mu_targets"], d["sigma_targets"],
                                 n_samples=6,
                                 save_path=os.path.join(output_dir, "spectral_comparison.pdf"))

    # Poles/residues
    plot_poles_residues(poles, residues, n_samples=min(20, len(poles)),
                        save_path=os.path.join(output_dir, "poles_residues.pdf"))

    print(f"\nAll plots saved to {output_dir}/")


def plot_spectral_mc(A_mean, A_std, omega_eval_grid, A_input=None, omega_input=None,
                     n_mc=None, save_path=None):
    """Plot MC mean ± 1σ spectral function with optional ground truth overlay.

    Parameters
    ----------
    A_mean         : ndarray (N_test, Nw), mean A(omega) across MC samples
    A_std          : ndarray (N_test, Nw), std A(omega) across MC samples
    omega_eval_grid: ndarray (Nw,)
    A_input        : ndarray (N_omega,) or None, ground truth
    omega_input    : ndarray (N_omega,) or None
    save_path      : str or None
    """
    A_mean_avg = A_mean.mean(axis=0)  # average over test samples
    A_std_avg  = A_std.mean(axis=0)   # average uncertainty over test samples

    has_input = A_input is not None and omega_input is not None

    if has_input:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))

    if has_input:
        ax1.fill_between(omega_input, A_input, alpha=0.2, color=COLORS["target"])
        ax1.plot(omega_input, A_input, "-", color=COLORS["target"], linewidth=2,
                 label="Ground truth")

    n_mc_str = rf" ({n_mc} $z$-draws)" if n_mc else ""
    ax1.plot(omega_eval_grid, A_mean_avg, "-", color=COLORS["pred"], linewidth=2,
             label=rf"$\langle A(\omega) \rangle$ MC mean{n_mc_str}")
    ax1.fill_between(omega_eval_grid,
                     A_mean_avg - A_std_avg,
                     A_mean_avg + A_std_avg,
                     alpha=0.25, color=COLORS["pred"],
                     label=rf"$\pm 1\sigma$ (VAE posterior uncertainty{n_mc_str})")

    ax1.set_ylabel(r"$A(\omega)$")
    ax1.set_title(
        r"VAE Posterior Uncertainty: $z \sim \mathcal{N}(\mu,\sigma^2)$"
        + (rf" $\times\,{n_mc}$ draws" if n_mc else "")
        + r"; $\pm 1\sigma$ = epistemic spread per input"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if has_input:
        A_input_interp = np.interp(omega_eval_grid, omega_input, A_input)
        diff = A_mean_avg - A_input_interp
        ax2.plot(omega_eval_grid, diff, "-", color="#D32F2F", linewidth=1.5)
        ax2.fill_between(omega_eval_grid, diff, alpha=0.2, color="#D32F2F")
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        dw = omega_eval_grid[1] - omega_eval_grid[0]
        int_abs_err = np.sum(np.abs(diff)) * dw
        ax2.set_xlabel(r"$\omega$")
        ax2.set_ylabel(r"$\Delta A(\omega)$")
        ax2.set_title(rf"Residual: $\int|\Delta A|d\omega = {int_abs_err:.4f}$")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_finetune_eval(summary_file, output_dir=None):
    """Generate plots from a fine-tuning summary.pt file.

    Parameters
    ----------
    summary_file : str, path to summary.pt
    output_dir   : str or None
    """
    d = torch.load(summary_file, map_location="cpu", weights_only=False)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(summary_file), "plots")
    os.makedirs(output_dir, exist_ok=True)

    beta = d["beta"]
    G_input = d["inputs"]
    G_recon = d["recon"]
    poles = d["poles"]
    residues = d["residues"]

    # Load ground truth spectral function if path is saved
    A_input = None
    omega_input = None
    if "spectral_input_path" in d and d["spectral_input_path"] is not None:
        spec_path = d["spectral_input_path"]
        if os.path.exists(spec_path):
            A_input = np.loadtxt(spec_path, delimiter=",")
            # The spectral_input.csv files use omega = linspace(-5, 5, len(A))
            omega_input = np.linspace(-5, 5, len(A_input))
            print(f"Loaded ground truth spectral function from: {spec_path}")
        else:
            print(f"WARNING: spectral_input_path not found: {spec_path}")

    noise_var = d.get("noise_var", None)

    # --- Build physical model title from saved metadata or folder name ---
    import re as _re
    _title_parts = []

    # sim_name: from summary.pt (new runs) or parsed from the output folder path (old runs)
    sim_name = d.get("sim_name", None)
    if sim_name is None:
        # Fall back: extract sim name from the output folder name
        # Folder pattern: finetune_real-{sim_name}_numpoles{P}-{sID}
        _folder = os.path.basename(os.path.dirname(summary_file))
        _fm = _re.search(r"finetune_real-(.+?)_numpoles(\d+)-(.+)$", _folder)
        if _fm:
            sim_name  = _fm.group(1)
            if d.get("num_poles") is None:
                d["num_poles"] = int(_fm.group(2))
            if d.get("sID") is None:
                d["sID"] = _fm.group(3)

    if sim_name:
        _m_holstein = _re.search(
            r"w(?P<w>[\d.]+)_a(?P<a>[\d.]+)_b(?P<b>[\d.]+)_L(?P<L>\d+)",
            sim_name
        )
        _m_hubbard = _re.search(
            r"hubbard.*_U(?P<U>[\d.]+)_mu(?P<mu>[\d.]+)_L(?P<L>\d+)_b(?P<b>[\d.]+)",
            sim_name
        )
        if _m_holstein:
            _title_parts.append(
                rf"Bond-Holstein: $\omega={_m_holstein.group('w')}$, "
                rf"$\alpha={_m_holstein.group('a')}$, "
                rf"$\beta={_m_holstein.group('b')}$, "
                rf"$L={_m_holstein.group('L')}$"
            )
        elif _m_hubbard:
            _title_parts.append(
                rf"Hubbard: $U={_m_hubbard.group('U')}$, "
                rf"$\mu={_m_hubbard.group('mu')}$, "
                rf"$\beta={_m_hubbard.group('b')}$, "
                rf"$L={_m_hubbard.group('L')}$"
            )
        else:
            _title_parts.append(sim_name.replace("_", r"\_"))
    else:
        _title_parts.append(rf"$\beta={beta:.2f}$")

    num_poles = d.get("num_poles", None)
    sID_str   = d.get("sID", None)
    if num_poles:
        _title_parts.append(rf"{num_poles} poles")
    if sID_str:
        _sID_escaped = sID_str.replace("_", r"\_")
        _title_parts.append(rf"\texttt{{{_sID_escaped}}}")
    _spectral_suptitle = r"~$|$~".join(_title_parts)

    plot_gtau_comparison(G_input, G_recon, beta, n_samples=6, noise_var=noise_var,
                         save_path=os.path.join(output_dir, "gtau_comparison.pdf"))

    plot_poles_residues(poles, residues, n_samples=min(20, len(poles)),
                        save_path=os.path.join(output_dir, "poles_residues.pdf"))

    # Predicted A(omega) for individual samples (with ground truth overlay
    # and the SC-reference A(omega|<G>) overlay + per-sample diff panel)
    plot_spectral_predicted(poles, residues,
                            A_input=A_input, omega_input=omega_input,
                            n_samples=6,
                            poles_from_avg=d.get("poles_from_avg"),
                            residues_from_avg=d.get("residues_from_avg"),
                            save_path=os.path.join(output_dir, "spectral_predicted.pdf"))

    # Average A(omega) with ground truth overlay and difference
    plot_spectral_average(poles, residues,
                          A_input=A_input, omega_input=omega_input,
                          suptitle=_spectral_suptitle,
                          poles_from_avg=d.get("poles_from_avg"),
                          residues_from_avg=d.get("residues_from_avg"),
                          save_path=os.path.join(output_dir, "spectral_average.pdf"))

    # Average G(tau)
    fig, ax = plt.subplots(figsize=(8, 5))
    L_tau = G_input.shape[1]
    taus = np.linspace(0, beta - beta / L_tau, L_tau)
    ax.plot(taus, d["inputs_avg"].numpy(), "o", color=COLORS["input"],
            markersize=3, alpha=0.6, label="Input (avg)")
    ax.plot(taus, d["recon_avg"].numpy(), "-", color=COLORS["recon"],
            linewidth=2, label=r"$\langle$Reconstructed$\rangle$ (avg of recon)")
    if "recon_from_avg" in d and d["recon_from_avg"] is not None:
        ax.plot(taus, d["recon_from_avg"].numpy(), ":", color="#8E24AA",
                linewidth=2, label=r"Reconstructed from $\langle G\rangle$ (single pass)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G(\tau)$")
    ax.set_title(r"Average $G(\tau)$ Reconstruction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "gtau_average.pdf"), bbox_inches="tight")
    print(f"Saved: {os.path.join(output_dir, 'gtau_average.pdf')}")

    print(f"\nAll plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot pretraining/fine-tuning results")
    parser.add_argument("--eval_file", type=str, help="Path to pretrain_eval.pt")
    parser.add_argument("--summary_file", type=str, help="Path to summary.pt (fine-tuning)")
    parser.add_argument("--loss_dir", type=str, help="Directory with loss .npy files")
    parser.add_argument("--tag", type=str, default="pretrain", help="Loss file tag")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    if args.eval_file:
        plot_pretrain_eval(args.eval_file, args.output_dir)

    if args.summary_file:
        plot_finetune_eval(args.summary_file, args.output_dir)

    if args.loss_dir:
        save_path = None
        if args.output_dir is None:
            args.output_dir = os.path.join(os.path.dirname(args.loss_dir), "plots")
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"loss_curves_{args.tag}.pdf")
        plot_loss_curves(args.loss_dir, tag=args.tag, save_path=save_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
