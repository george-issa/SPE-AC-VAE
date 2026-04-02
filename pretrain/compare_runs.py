"""
Compare fresh vs pretrained fine-tuning runs on the same dataset.

Loads outputs from two completed fine-tuning runs and produces:
  1. chi2_overlay.pdf        — chi2 vs epoch, both runs on one axis
  2. chi2_curves.pdf         — side-by-side chi2 panels
  3. loss_components.pdf     — total, neg_green, neg_second vs epoch
  4. chi2_histogram.pdf      — per-sample min-chi2 distribution
  5. gtau_comparison.pdf     — G(tau) input vs fresh/pretrained recon per sample
  6. spectral_comparison.pdf — A(omega) per sample with ground truth overlay
  7. spectral_avg.pdf        — dataset-averaged A(omega) from both runs + ground truth

Prints a summary metrics table to stdout.

Usage:
    python pretrain/compare_runs.py
"""

import os
import sys
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# LaTeX rendering — matches plot_results.py style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
})

# ==========================================================================
# CONFIGURATION — update these to point at your completed runs
# ==========================================================================

MAIN_PATH = "/Users/georgeissa/Documents/AC/SPE-AC-VAE"
OUT_BASE  = os.path.join(MAIN_PATH, "out")

# Fresh run: run_finetune.py with LOAD_PRETRAIN=False
FRESH_DIR      = os.path.join(OUT_BASE, "finetune_gaussian_double_numpoles4_s1e-04_xi0.5-fresh-v2L-2")
# Pretrained run: run_pretrain_pipeline.py
PRETRAINED_DIR = os.path.join(OUT_BASE, "finetune_gaussian_double_numpoles4_s1e-04_xi0.5-pretrained-v2L")

TAG = "finetune"

# Samples to show in per-sample plots (must exist in both summaries)
SAMPLE_INDICES = [0, 1, 2, 3]

# Where to write comparison outputs — auto-named from the fresh run directory:
# strips the finetune_ prefix and the -fresh-... suffix, then appends the model version.
# e.g. finetune_gaussian_double_numpoles4_s1e-04_xi0.5-fresh-v2L-2
#   -> comparison_gaussian_double_numpoles4_s1e-04_xi0.5-v2L
_fresh_base = os.path.basename(FRESH_DIR)
_body    = re.sub(r"^finetune_", "", _fresh_base)          # drop leading "finetune_"
_body    = re.sub(r"-(fresh|pretrained).*$", "", _body)    # drop -fresh-... / -pretrained-...
_ver     = re.search(r"-(v\w+)", _fresh_base)              # extract model version, e.g. v2L
_ver_str = _ver.group(1) if _ver else "vX"
COMPARE_OUT_DIR = os.path.join(OUT_BASE, f"comparison_{_body}-{_ver_str}")

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


def load_ground_truth(summary):
    """Load ground truth A(omega) from the path stored in the summary."""
    spec_path = summary.get("spectral_input_path", None)
    if spec_path is None or not os.path.exists(spec_path):
        return None, None
    A_gt = np.loadtxt(spec_path, delimiter=",")
    omega_gt = np.linspace(-5, 5, len(A_gt))
    return omega_gt, A_gt


# ==========================================================================
# PLOTS
# ==========================================================================

def plot_chi2_overlay(fresh_losses, pre_losses, out_path):
    """chi2 vs epoch for both runs on one axis (log scale)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for losses, label, color in [
        (fresh_losses, "Fresh (random init)", "tab:blue"),
        (pre_losses,   "Pretrained + finetuned", "tab:orange"),
    ]:
        chi2 = losses["chi2"]
        if chi2 is None:
            continue
        ax.plot(np.arange(1, len(chi2) + 1), chi2, color=color, lw=1.5, label=label)
    ax.axhline(1.0, color="k", ls="--", lw=1, label=r"Target $\chi^2=1$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\chi^2$")
    ax.set_title(r"Convergence: Fresh vs Pretrained")
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
        ["Fresh (random init)", "Pretrained + finetuned"],
        ["tab:blue", "tab:orange"],
    ):
        chi2 = losses["chi2"]
        if chi2 is None:
            continue
        ax.plot(np.arange(1, len(chi2) + 1), chi2, color=color, lw=1.5)
        ax.axhline(1.0, color="k", ls="--", lw=1, label=r"$\chi^2=1$")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\chi^2$")
        ax.set_title(label)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(r"$\chi^2$ vs Epoch", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_loss_components(fresh_losses, pre_losses, out_path):
    """Total loss + negativity penalties vs epoch for both runs."""
    keys   = ["val",       "neg_green",   "neg_second"]
    labels = ["Total loss", r"neg $G(\tau)$", r"neg $G''(\tau)$"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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


def plot_chi2_histogram(fresh_losses, pre_losses, out_path):
    """
    Histogram of per-epoch chi2 values, comparing the two runs.
    Also marks the minimum chi2 achieved by each run.
    """
    chi2_f = fresh_losses["chi2"]
    chi2_p = pre_losses["chi2"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(chi2_f.min() if chi2_f is not None else 1,
            chi2_p.min() if chi2_p is not None else 1) * 0.9,
        max(chi2_f[-1]  if chi2_f is not None else 10,
            chi2_p[-1]  if chi2_p is not None else 10) * 1.1,
        50,
    )

    if chi2_f is not None:
        ax.hist(chi2_f, bins=bins, color="tab:blue", alpha=0.5, label="Fresh", density=True)
        ax.axvline(chi2_f.min(), color="tab:blue", ls="--", lw=1.5,
                   label=f"Fresh min = {chi2_f.min():.3f}")
    if chi2_p is not None:
        ax.hist(chi2_p, bins=bins, color="tab:orange", alpha=0.5, label="Pretrained", density=True)
        ax.axvline(chi2_p.min(), color="tab:orange", ls="--", lw=1.5,
                   label=f"Pretrained min = {chi2_p.min():.3f}")

    ax.axvline(1.0, color="k", ls="-", lw=1, label=r"Target $\chi^2=1$")
    ax.set_xlabel(r"$\chi^2$")
    ax.set_ylabel("Density")
    ax.set_title(r"Distribution of $\chi^2$ across epochs")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gtau_comparison(fresh_summary, pre_summary, indices, out_path):
    """G(tau) input vs fresh recon vs pretrained recon for selected samples."""
    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)

    G_in   = fresh_summary["inputs"]   # same input for both runs
    G_rc_f = fresh_summary["recon"]
    G_rc_p = pre_summary["recon"]
    Ltau   = int(fresh_summary["Ltau"])
    beta   = float(fresh_summary["beta"])
    taus   = np.linspace(0, beta, Ltau)

    for row, idx in enumerate(indices):
        ax = axes[row, 0]
        ax.plot(taus, G_in[idx].numpy(),   "k-",  lw=1.5, label=r"$G(\tau)$ input")
        ax.plot(taus, G_rc_f[idx].numpy(), "b--", lw=1.2, label="Fresh recon")
        ax.plot(taus, G_rc_p[idx].numpy(), "r:",  lw=1.2, label="Pretrained recon")
        ax.set_ylabel(r"$G(\tau)$")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel(r"$\tau$")
    fig.suptitle(r"$G(\tau)$ Reconstruction: Fresh vs Pretrained", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_spectral_comparison(fresh_summary, pre_summary, indices, out_path):
    """A(omega) per sample — both runs + ground truth overlay + difference panel.

    Layout: nested GridSpec so that main+diff panels within a sample are tight,
    while samples themselves are well separated.
    """
    n = len(indices)
    omega_f  = fresh_summary["omega_eval_grid"].numpy()
    A_mean_f = fresh_summary["A_mean"].numpy()
    A_std_f  = fresh_summary["A_std"].numpy()
    A_mean_p = pre_summary["A_mean"].numpy()
    A_std_p  = pre_summary["A_std"].numpy()

    omega_gt, A_gt = load_ground_truth(fresh_summary)
    has_gt = omega_gt is not None

    # Outer GridSpec: one row per sample, with generous vertical separation
    fig = plt.figure(figsize=(9, 5 * n))
    outer = GridSpec(n, 1, figure=fig, hspace=0.55)

    for row, idx in enumerate(indices):
        # Inner GridSpec: main (tall) + diff (short), kept tight together
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row],
            height_ratios=[3, 1], hspace=0.07,
        )
        ax_main = fig.add_subplot(inner[0])
        ax_diff = fig.add_subplot(inner[1], sharex=ax_main)

        # --- Main panel: spectral overlay ---
        ax_main.plot(omega_f, A_mean_f[idx], color="tab:blue",   lw=1.5, label=r"Fresh")
        if A_std_f[idx].max() > 0:
            ax_main.fill_between(omega_f,
                                 A_mean_f[idx] - A_std_f[idx],
                                 A_mean_f[idx] + A_std_f[idx],
                                 color="tab:blue", alpha=0.15)
        ax_main.plot(omega_f, A_mean_p[idx], color="tab:orange", lw=1.5, ls="--",
                     label=r"Pretrained")
        if A_std_p[idx].max() > 0:
            ax_main.fill_between(omega_f,
                                 A_mean_p[idx] - A_std_p[idx],
                                 A_mean_p[idx] + A_std_p[idx],
                                 color="tab:orange", alpha=0.15)
        if has_gt:
            ax_main.fill_between(omega_gt, A_gt, alpha=0.12, color="tab:green")
            ax_main.plot(omega_gt, A_gt, color="tab:green", lw=1.5, alpha=0.9,
                         label=r"Ground truth")
        ax_main.set_ylabel(r"$A(\omega)$")
        ax_main.set_title(rf"Sample {idx}", pad=4)
        ax_main.legend(fontsize=9, loc="upper right")
        ax_main.grid(True, alpha=0.3)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # --- Difference panel: A_pred - A_GT ---
        ax_diff.axhline(0, color="gray", lw=0.8, alpha=0.5)
        if has_gt:
            gt_interp = np.interp(omega_f, omega_gt, A_gt)
            diff_f = A_mean_f[idx] - gt_interp
            diff_p = A_mean_p[idx] - gt_interp
            ax_diff.plot(omega_f, diff_f, color="tab:blue",   lw=1.2)
            ax_diff.fill_between(omega_f, diff_f, alpha=0.20, color="tab:blue")
            ax_diff.plot(omega_f, diff_p, color="tab:orange", lw=1.2, ls="--")
            ax_diff.fill_between(omega_f, diff_p, alpha=0.20, color="tab:orange")
            dw = omega_f[1] - omega_f[0]
            err_f = np.sqrt(np.sum(diff_f ** 2) * dw)
            err_p = np.sqrt(np.sum(diff_p ** 2) * dw)
            ax_diff.set_ylabel(
                rf"$\|\Delta A\|_2$: F$={err_f:.3f}$, P$={err_p:.3f}$",
                fontsize=8,
            )
        else:
            ax_diff.set_ylabel(r"$\Delta A$", fontsize=9)
        ax_diff.grid(True, alpha=0.3)
        ax_diff.set_xlabel(r"$\omega$")

    fig.suptitle(
        r"$A(\omega)$: \textbf{Fresh} (blue) vs \textbf{Pretrained} (orange dashed)"
        r" vs \textbf{Ground truth} (green)"
        "\n"
        r"Shaded bottom: $\Delta A = A_{\mathrm{pred}} - A_{\mathrm{GT}}$",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.93)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_spectral_avg(fresh_summary, pre_summary, out_path):
    """Dataset-averaged A(omega) from both runs + ground truth.

    Single panel with norm values reported in the title.
    """
    omega_f = fresh_summary["omega_eval_grid"].numpy()
    A_avg_f = fresh_summary["A_mean"].numpy().mean(axis=0)
    A_avg_p = pre_summary["A_mean"].numpy().mean(axis=0)

    omega_gt, A_gt = load_ground_truth(fresh_summary)
    has_gt = omega_gt is not None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # --- Top: overlay ---
    if has_gt:
        ax1.fill_between(omega_gt, A_gt, alpha=0.12, color="tab:green")
        ax1.plot(omega_gt, A_gt, color="tab:green", lw=1.5, alpha=0.9,
                 label=r"Ground truth")
    ax1.plot(omega_f, A_avg_f, color="tab:blue",   lw=1.5,
             label=r"Fresh $\langle A(\omega)\rangle$")
    ax1.plot(omega_f, A_avg_p, color="tab:orange", lw=1.5, ls="--",
             label=r"Pretrained $\langle A(\omega)\rangle$")
    ax1.set_ylabel(r"$\langle A(\omega) \rangle$")
    ax1.set_title(r"Dataset-averaged $A(\omega)$: Fresh vs Pretrained", fontsize=11, pad=6)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Bottom: difference (pred - GT) ---
    ax2.axhline(0, color="gray", lw=0.8, alpha=0.5)
    if has_gt:
        gt_interp = np.interp(omega_f, omega_gt, A_gt)
        diff_f = A_avg_f - gt_interp
        diff_p = A_avg_p - gt_interp
        ax2.plot(omega_f, diff_f, color="tab:blue",   lw=1.2, label=r"Fresh $-$ GT")
        ax2.fill_between(omega_f, diff_f, alpha=0.20, color="tab:blue")
        ax2.plot(omega_f, diff_p, color="tab:orange", lw=1.2, ls="--",
                 label=r"Pretrained $-$ GT")
        ax2.fill_between(omega_f, diff_p, alpha=0.20, color="tab:orange")
        dw = omega_f[1] - omega_f[0]
        err_f = np.sqrt(np.sum(diff_f ** 2) * dw)
        err_p = np.sqrt(np.sum(diff_p ** 2) * dw)
        diff_title = (
            r"$\Delta A(\omega) = \langle A_{\mathrm{pred}}\rangle - A_{\mathrm{GT}}$"
            "\n"
            r"$\|\cdot\|_\mathrm{F}$ (Fresh)$= "  + rf"{err_f:.4f}$"
            r"$\quad \|\cdot\|_\mathrm{P}$ (Pretrained)$= " + rf"{err_p:.4f}$"
            r"$\quad \|\cdot\| = \sqrt{{\int (\Delta A)^2\,d\omega}}$"
        )
        ax2.set_title(diff_title, fontsize=9, pad=4)
        ax2.legend(fontsize=9, loc="upper right")
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta A(\omega)$")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary_table(fresh_losses, pre_losses):
    """Print final metrics comparison table."""
    def _final(arr):  return float(arr[-1])   if arr is not None and len(arr) > 0 else float("nan")
    def _min(arr):    return float(arr.min())  if arr is not None and len(arr) > 0 else float("nan")
    def _argmin(arr): return int(arr.argmin()) + 1 if arr is not None and len(arr) > 0 else -1

    print("\n" + "=" * 60)
    print(f"{'Metric':<34} {'Fresh':>10} {'Pretrained':>12}")
    print("=" * 60)
    def _len(arr): return len(arr) if arr is not None else 0
    print(f"{'Epochs trained':<34} {_len(fresh_losses['chi2']):>10d} {_len(pre_losses['chi2']):>12d}")
    print(f"{'Final chi2':<34} {_final(fresh_losses['chi2']):>10.4f} {_final(pre_losses['chi2']):>12.4f}")
    print(f"{'Min chi2':<34} {_min(fresh_losses['chi2']):>10.4f} {_min(pre_losses['chi2']):>12.4f}")
    print(f"{'Epoch of min chi2':<34} {_argmin(fresh_losses['chi2']):>10d} {_argmin(pre_losses['chi2']):>12d}")
    print(f"{'Final total loss':<34} {_final(fresh_losses['val']):>10.4f} {_final(pre_losses['val']):>12.4f}")
    print(f"{'Final neg G(tau)':<34} {_final(fresh_losses['neg_green']):>10.4f} {_final(pre_losses['neg_green']):>12.4f}")
    label_neg2 = "Final neg G''(tau)"
    print(f"{label_neg2:<34} {_final(fresh_losses['neg_second']):>10.4f} {_final(pre_losses['neg_second']):>12.4f}")
    print("=" * 60 + "\n")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    # Validate both run directories exist
    for label, path in [("Fresh", FRESH_DIR), ("Pretrained", PRETRAINED_DIR)]:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{label} run directory not found: {path}\n"
                f"Run the appropriate script first, then update the paths in compare_runs.py."
            )

    os.makedirs(COMPARE_OUT_DIR, exist_ok=True)

    print(f"Fresh run:      {FRESH_DIR}")
    print(f"Pretrained run: {PRETRAINED_DIR}")
    print(f"Output dir:     {COMPARE_OUT_DIR}\n")

    print("Loading fresh run outputs...")
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
    plot_chi2_histogram(
        fresh_losses, pre_losses,
        os.path.join(COMPARE_OUT_DIR, "chi2_histogram.pdf"))
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
