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
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
# G(tau) comparison plots
# ---------------------------------------------------------------------------

def plot_gtau_comparison(G_input, G_recon, beta, n_samples=6, save_path=None):
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
        ax.set_title(f"Sample {i+1}  (MSE={mse:.2e})")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$G(\tau)$")
        ax.legend(loc="upper right")

    # Hide unused subplots
    for i in range(n_samples, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(r"$G(\tau)$ Reconstruction", fontsize=16, y=1.02)
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

def plot_loss_curves(loss_dir, tag="pretrain", save_path=None):
    """Plot training and validation loss curves with all components.

    Handles both pretrain losses (chi2, spec, smooth, kl) and
    finetune losses (chi2, smooth, kl, neg_green, neg_second).

    Parameters
    ----------
    loss_dir  : str, directory containing .npy loss files
    tag       : str, loss file tag
    save_path : str or None
    """
    # Try loading all possible loss files
    possible_files = {
        "train": f"train_losses_{tag}.npy",
        "val": f"val_losses_{tag}.npy",
        "chi2": f"chi2_losses_{tag}.npy",
        "spec": f"spec_losses_{tag}.npy",
        "smooth": f"smooth_losses_{tag}.npy",
        "kl": f"kl_losses_{tag}.npy",
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
        # Pretrain layout: 2x2 grid
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

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
        # Pretrain layout (no penalties)
        # Top-right or Bottom-left: Spectral MSE
        ax3 = fig.add_subplot(gs[1, 0])
        if "spec" in losses:
            ax3.semilogy(losses["spec"], label="Spectral MSE", color="#388E3C", linewidth=1.5)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Spectral MSE")
        ax3.set_title(r"Spectral MSE: $\int (A_{pred} - A_{target})^2 d\omega$")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom-right: Smoothness and KL
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

    fig.suptitle(f"Training Curves ({tag})", fontsize=16, y=1.01)
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
                            n_samples=6, wmin=-6, wmax=6, Nw=1000, save_path=None):
    """Plot predicted A(omega) from poles/residues for selected samples,
    overlaid with ground truth if available.

    Parameters
    ----------
    poles       : complex tensor (N, P)
    residues    : complex tensor (N, P)
    A_input     : ndarray (N_omega,) or None, ground truth spectral function
    omega_input : ndarray (N_omega,) or None, omega grid for ground truth
    n_samples   : int
    wmin, wmax  : float, omega range
    Nw          : int, number of omega grid points
    save_path   : str or None
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

    cols = min(3, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i in range(n_samples):
        ax = axes[i // cols, i % cols]

        # Ground truth
        if A_input is not None and omega_input is not None:
            ax.fill_between(omega_input, A_input, alpha=0.15, color=COLORS["target"])
            ax.plot(omega_input, A_input, "-", color=COLORS["target"], linewidth=1.5,
                    label="Input (exact)")

        ax.plot(omegas_np, A_pred[i], "-", color=COLORS["pred"], linewidth=1.5,
                label="Predicted")

        # Mark pole positions
        eps = poles[i].real.numpy()
        for p in range(len(eps)):
            ax.axvline(eps[p], color=COLORS["pole_re"], linestyle="--",
                       alpha=0.4, linewidth=0.8)

        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$A(\omega)$")
        ax.set_xlim(wmin, wmax)
        ax.legend(loc="upper right")

    for i in range(n_samples, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(r"Predicted Spectral Function $A(\omega)$", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_spectral_average(poles, residues, A_input=None, omega_input=None,
                          wmin=-6, wmax=6, Nw=1000, save_path=None):
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
             label=r"$\langle A(\omega) \rangle$ predicted")
    ax1.fill_between(omegas_np, A_mean - A_std, A_mean + A_std,
                     alpha=0.2, color=COLORS["pred"], label=r"$\pm 1\sigma$")

    for p in range(len(eps_mean)):
        ax1.axvline(eps_mean[p], color=COLORS["pole_re"], linestyle="--",
                    alpha=0.5, linewidth=1.0,
                    label=rf"$\langle \epsilon_{p+1} \rangle = {eps_mean[p]:.2f}$")

    ax1.set_ylabel(r"$A(\omega)$")
    ax1.set_title(r"Spectral Function Comparison")
    ax1.set_xlim(wmin, wmax)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: difference ---
    if has_input:
        # Interpolate ground truth onto prediction grid
        A_input_interp = np.interp(omegas_np, omega_input, A_input)
        diff = A_mean - A_input_interp

        ax2.plot(omegas_np, diff, "-", color="#D32F2F", linewidth=1.5)
        ax2.fill_between(omegas_np, diff, alpha=0.2, color="#D32F2F")
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

        # Integrated absolute error
        dw = (wmax - wmin) / (Nw - 1)
        int_abs_err = np.sum(np.abs(diff)) * dw
        int_sq_err = np.sqrt(np.sum(diff ** 2) * dw)
        max_abs_err = np.max(np.abs(diff))

        ax2.set_xlabel(r"$\omega$")
        ax2.set_ylabel(r"$\Delta A(\omega)$")
        ax2.set_title(
            rf"Difference (pred $-$ input): "
            rf"$\int |{{\Delta A}}| d\omega = {int_abs_err:.4f}$, "
            rf"$\sqrt{{\int {{\Delta A}}^2 d\omega}} = {int_sq_err:.4f}$, "
            rf"$\max |{{\Delta A}}| = {max_abs_err:.4f}$"
        )
        ax2.grid(True, alpha=0.3)

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
            # The spectral_input.csv files use omega = linspace(-6, 6, len(A))
            omega_input = np.linspace(-6, 6, len(A_input))
            print(f"Loaded ground truth spectral function from: {spec_path}")
        else:
            print(f"WARNING: spectral_input_path not found: {spec_path}")

    plot_gtau_comparison(G_input, G_recon, beta, n_samples=6,
                         save_path=os.path.join(output_dir, "gtau_comparison.pdf"))

    plot_poles_residues(poles, residues, n_samples=min(20, len(poles)),
                        save_path=os.path.join(output_dir, "poles_residues.pdf"))

    # Predicted A(omega) for individual samples (with ground truth overlay)
    plot_spectral_predicted(poles, residues,
                            A_input=A_input, omega_input=omega_input,
                            n_samples=6,
                            save_path=os.path.join(output_dir, "spectral_predicted.pdf"))

    # Average A(omega) with ground truth overlay and difference
    plot_spectral_average(poles, residues,
                          A_input=A_input, omega_input=omega_input,
                          save_path=os.path.join(output_dir, "spectral_average.pdf"))

    # Average G(tau)
    fig, ax = plt.subplots(figsize=(8, 5))
    L_tau = G_input.shape[1]
    taus = np.linspace(0, beta - beta / L_tau, L_tau)
    ax.plot(taus, d["inputs_avg"].numpy(), "o", color=COLORS["input"],
            markersize=3, alpha=0.6, label="Input (avg)")
    ax.plot(taus, d["recon_avg"].numpy(), "-", color=COLORS["recon"],
            linewidth=2, label="Reconstructed (avg)")
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
