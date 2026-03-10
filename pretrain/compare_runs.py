"""
Compare fresh vs pretrained fine-tuning runs on the same dataset.

Loads outputs from two completed fine-tuning runs and produces:
  1. chi2_overlay.pdf      — chi2 vs epoch, both runs on one axis
  2. chi2_curves.pdf       — side-by-side chi2 panels
  3. loss_components.pdf   — total, neg_green, neg_second, neg_fourth vs epoch
  4. gtau_comparison.pdf   — G(tau) input vs fresh/pretrained recon per sample
  5. spectral_comparison.pdf — A(omega) mean ± 1sigma MC per sample
  6. spectral_avg.pdf      — dataset-averaged A(omega) from both runs

Prints a summary metrics table to stdout.

Usage:
    python pretrain/compare_runs.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ==========================================================================
# CONFIGURATION
# ==========================================================================

MAIN_PATH = "/Users/georgeissa/Documents/AC/SPE-AC-VAE"
OUT_BASE  = os.path.join(MAIN_PATH, "out")

# Output directories of the two runs to compare
FRESH_DIR      = os.path.join(OUT_BASE, "finetune_gaussian_double_numpoles4_s1e-04_xi0.5-fresh-v2s-1")
PRETRAINED_DIR = os.path.join(OUT_BASE, "finetune_gaussian_double_numpoles4_s1e-04_xi0.5-pretrained-v2s")

TAG = "finetune"

# Samples to show in per-sample plots
SAMPLE_INDICES = [0, 1, 2, 3]

# Where to write comparison outputs
COMPARE_OUT_DIR = os.path.join(OUT_BASE, "comparison_fresh_vs_pretrained")

# ==========================================================================
# HELPERS
# ==========================================================================

def load_losses(run_dir, tag=TAG):
    """Load all finetune loss .npy arrays from run_dir/losses/."""
    loss_dir = os.path.join(run_dir, "losses")
    def _load(name):
        path = os.path.join(loss_dir, f"{name}_{tag}.npy")
        return np.load(path) if os.path.exists(path) else None
    return {
        "train":      _load("train_losses"),
        "val":        _load("val_losses"),
        "chi2":       _load("chi2_losses"),
        "smooth":     _load("smooth_losses"),
        "kl":         _load("kl_losses"),
        "neg_green":  _load("neg_green_losses"),
        "neg_second": _load("neg_second_losses"),
        "neg_fourth": _load("neg_fourth_losses"),
    }


def load_summary(run_dir):
    path = os.path.join(run_dir, "summary.pt")
    return torch.load(path, weights_only=False)

# ==========================================================================
# PLOTS
# ==========================================================================

def plot_chi2_overlay(fresh_losses, pre_losses, out_path):
    """chi2 vs epoch for both runs on one axis (log scale)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for losses, label, color in [
        (fresh_losses, "Fresh (random init)", "tab:blue"),
        (pre_losses,   "Pretrained",           "tab:orange"),
    ]:
        chi2 = losses["chi2"]
        ax.plot(np.arange(1, len(chi2) + 1), chi2, color=color, lw=1.5, label=label)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="Target χ²=1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("χ²")
    ax.set_title("Convergence: Fresh vs Pretrained")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_chi2_curves(fresh_losses, pre_losses, out_path):
    """Side-by-side chi2 panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, losses, label, color in zip(
        axes,
        [fresh_losses, pre_losses],
        ["Fresh (random init)", "Pretrained"],
        ["tab:blue", "tab:orange"],
    ):
        chi2 = losses["chi2"]
        ax.plot(np.arange(1, len(chi2) + 1), chi2, color=color, lw=1.5)
        ax.axhline(1.0, color="k", ls="--", lw=1, label="Target χ²=1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("χ²")
        ax.set_title(label)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("χ² vs Epoch", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_loss_components(fresh_losses, pre_losses, out_path):
    """Total loss + negativity penalties vs epoch for both runs."""
    keys   = ["val",       "neg_green",  "neg_second",  "neg_fourth"]
    labels = ["Total loss", "neg G(τ)",   "neg G''(τ)",  "neg G''''(τ)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, key, label in zip(axes, keys, labels):
        for losses, run_label, color in [
            (fresh_losses, "Fresh",      "tab:blue"),
            (pre_losses,   "Pretrained", "tab:orange"),
        ]:
            arr = losses[key]
            if arr is None:
                continue
            ax.plot(np.arange(1, len(arr) + 1), arr, color=color, lw=1.5, label=run_label)
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Loss Components: Fresh vs Pretrained", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gtau_comparison(fresh_summary, pre_summary, indices, out_path):
    """G(tau) input vs fresh recon vs pretrained recon for selected samples."""
    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)

    G_in  = fresh_summary["inputs"]   # same input for both runs
    G_rc_f = fresh_summary["recon"]
    G_rc_p = pre_summary["recon"]
    Ltau   = int(fresh_summary["Ltau"])
    beta   = float(fresh_summary["beta"])
    taus   = np.linspace(0, beta, Ltau)

    for row, idx in enumerate(indices):
        ax = axes[row, 0]
        ax.plot(taus, G_in[idx].numpy(),   "k-",  lw=1.5, label="G(τ) input")
        ax.plot(taus, G_rc_f[idx].numpy(), "b--", lw=1.2, label="Fresh recon")
        ax.plot(taus, G_rc_p[idx].numpy(), "r:",  lw=1.2, label="Pretrained recon")
        ax.set_ylabel("G(τ)")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("τ")
    fig.suptitle("G(τ) Reconstruction: Fresh vs Pretrained", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_spectral_comparison(fresh_summary, pre_summary, indices, out_path):
    """A(omega) mean ± 1sigma for selected samples, both runs."""
    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)

    omega    = fresh_summary["omega_eval_grid"].numpy()
    A_mean_f = fresh_summary["A_mean"].numpy()
    A_std_f  = fresh_summary["A_std"].numpy()
    A_mean_p = pre_summary["A_mean"].numpy()
    A_std_p  = pre_summary["A_std"].numpy()

    for row, idx in enumerate(indices):
        ax = axes[row, 0]
        ax.plot(omega, A_mean_f[idx], "b-",  lw=1.5, label="Fresh")
        ax.fill_between(omega,
                        A_mean_f[idx] - A_std_f[idx],
                        A_mean_f[idx] + A_std_f[idx],
                        color="blue", alpha=0.15)
        ax.plot(omega, A_mean_p[idx], "r--", lw=1.5, label="Pretrained")
        ax.fill_between(omega,
                        A_mean_p[idx] - A_std_p[idx],
                        A_mean_p[idx] + A_std_p[idx],
                        color="red", alpha=0.15)
        ax.set_ylabel("A(ω)")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("ω")
    fig.suptitle("A(ω) Spectral Function: Fresh vs Pretrained (mean ± 1σ MC)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_spectral_avg(fresh_summary, pre_summary, out_path):
    """Dataset-averaged A(omega) from both runs."""
    omega    = fresh_summary["omega_eval_grid"].numpy()
    A_avg_f  = fresh_summary["A_mean"].numpy().mean(axis=0)
    A_avg_p  = pre_summary["A_mean"].numpy().mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(omega, A_avg_f, "b-",  lw=1.5, label="Fresh (dataset avg)")
    ax.plot(omega, A_avg_p, "r--", lw=1.5, label="Pretrained (dataset avg)")
    ax.set_xlabel("ω")
    ax.set_ylabel("⟨A(ω)⟩")
    ax.set_title("Dataset-averaged A(ω): Fresh vs Pretrained")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary_table(fresh_losses, pre_losses):
    """Print final metrics comparison table."""
    def _final(arr):  return float(arr[-1])  if arr is not None and len(arr) > 0 else float("nan")
    def _min(arr):    return float(arr.min()) if arr is not None and len(arr) > 0 else float("nan")
    def _argmin(arr): return int(arr.argmin()) + 1 if arr is not None and len(arr) > 0 else -1

    print("\n" + "=" * 58)
    print(f"{'Metric':<32} {'Fresh':>10} {'Pretrained':>12}")
    print("=" * 58)
    print(f"{'Epochs trained':<32} {len(fresh_losses['chi2']):>10d} {len(pre_losses['chi2']):>12d}")
    print(f"{'Final χ²':<32} {_final(fresh_losses['chi2']):>10.4f} {_final(pre_losses['chi2']):>12.4f}")
    print(f"{'Min χ²':<32} {_min(fresh_losses['chi2']):>10.4f} {_min(pre_losses['chi2']):>12.4f}")
    print(f"{'Epoch of min χ²':<32} {_argmin(fresh_losses['chi2']):>10d} {_argmin(pre_losses['chi2']):>12d}")
    print(f"{'Final total loss':<32} {_final(fresh_losses['val']):>10.4f} {_final(pre_losses['val']):>12.4f}")
    print(f"{'Final neg G(τ)':<32} {_final(fresh_losses['neg_green']):>10.4f} {_final(pre_losses['neg_green']):>12.4f}")
    print(f"{'Final neg G\'\'(τ)':<32} {_final(fresh_losses['neg_second']):>10.4f} {_final(pre_losses['neg_second']):>12.4f}")
    print("=" * 58 + "\n")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    os.makedirs(COMPARE_OUT_DIR, exist_ok=True)

    print(f"Fresh run:      {FRESH_DIR}")
    print(f"Pretrained run: {PRETRAINED_DIR}")

    print("\nLoading fresh run outputs...")
    fresh_losses  = load_losses(FRESH_DIR)
    fresh_summary = load_summary(FRESH_DIR)

    print("Loading pretrained run outputs...")
    pre_losses  = load_losses(PRETRAINED_DIR)
    pre_summary = load_summary(PRETRAINED_DIR)

    print_summary_table(fresh_losses, pre_losses)

    print("Generating plots...")
    plot_chi2_overlay(
        fresh_losses, pre_losses,
        os.path.join(COMPARE_OUT_DIR, "chi2_overlay.pdf"))
    plot_chi2_curves(
        fresh_losses, pre_losses,
        os.path.join(COMPARE_OUT_DIR, "chi2_curves.pdf"))
    plot_loss_components(
        fresh_losses, pre_losses,
        os.path.join(COMPARE_OUT_DIR, "loss_components.pdf"))
    plot_gtau_comparison(
        fresh_summary, pre_summary, SAMPLE_INDICES,
        os.path.join(COMPARE_OUT_DIR, "gtau_comparison.pdf"))
    plot_spectral_comparison(
        fresh_summary, pre_summary, SAMPLE_INDICES,
        os.path.join(COMPARE_OUT_DIR, "spectral_comparison.pdf"))
    plot_spectral_avg(
        fresh_summary, pre_summary,
        os.path.join(COMPARE_OUT_DIR, "spectral_avg.pdf"))

    print(f"\nAll outputs saved to: {COMPARE_OUT_DIR}")


if __name__ == "__main__":
    main()
