"""
Benchmark comparison: VAE-AC vs MaxEnt analytic continuation.

Loads results from:
  1. A completed VAE fine-tuning run (out/finetune_*/summary.pt)
  2. ana_cont MaxEnt run (MaxEnt_benchmark/out/anacont_*/summary_mean.npz)
  3. OmegaMaxEnt run    (MaxEnt_benchmark/out/omegamaxent_*/summary.npz)   [optional]

Produces (in COMPARE_OUT_DIR):
  spectral_avg.pdf         — averaged A(omega): VAE vs MaxEnt(s) + ground truth
  spectral_samples.pdf     — per-sample A(omega) panel [VAE only; MaxEnt for mean]
  gtau_comparison.pdf      — G(tau) input vs reconstructions
  metrics_table.txt        — L2 and chi2 summary printed to file
  metrics.json             — same data as JSON

Usage:
    python MaxEnt_benchmark/compare_vae_maxent.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# LaTeX rendering consistent with project style
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

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

# ============================================================================
# CONFIGURATION — update these paths after running the two MaxEnt scripts
# ============================================================================

MAIN_PATH = "/Users/georgeissa/Documents/AC/SPE-AC-VAE"
MB_PATH   = os.path.join(MAIN_PATH, "MaxEnt_benchmark")

# VAE run to compare against
# Set to None to skip any method
VAE_SUMMARY = os.path.join(
    MAIN_PATH, "out",
    "finetune_gaussian_double_numpoles6_s1e-04_xi0.5-fresh-v2L-2",
    "summary.pt"
)

# ana_cont MaxEnt summary (produced by run_anacont_maxent.py, MODE='mean')
ANACONT_SUMMARY = os.path.join(
    MB_PATH, "out",
    "anacont_gaussian_double_inputs-8_s1e-04",
    "summary_mean_fullcov.npz"
)

# OmegaMaxEnt summary (produced by run_omegamaxent.py) — set to None to skip
OMEGAMAXENT_SUMMARY = os.path.join(
    MB_PATH, "out",
    "omegamaxent_gaussian_double_inputs-8_s1e-04",
    "summary.npz"
)

# Samples to highlight in per-sample panels
SAMPLE_INDICES = [0, 1, 2, 3]

# Output directory
COMPARE_OUT_DIR = os.path.join(MB_PATH, "out", "comparison_vae_maxent")

# ============================================================================
# LOADERS
# ============================================================================

def load_ground_truth(spectral_input_path):
    """Load ground-truth A(omega) from spectral_input.csv."""
    if not os.path.exists(spectral_input_path):
        return None, None
    A_gt = np.loadtxt(spectral_input_path, delimiter=",")
    # omega grid fixed to [-5, 5] with len(A_gt) points (project convention)
    omega_gt = np.linspace(-5.0, 5.0, len(A_gt))
    return omega_gt, A_gt


def load_vae(summary_path):
    """Load VAE summary.pt. Returns dict with keys: omega, A_mean [N,Nw], etc."""
    if summary_path is None or not os.path.exists(summary_path):
        return None
    s = torch.load(summary_path, weights_only=False)
    return {
        "omega":               s["omega_eval_grid"].numpy(),
        "A_mean":              s["A_mean"].numpy(),           # (N, Nw)
        "A_avg":               s["A_mean"].numpy().mean(0),   # (Nw,)
        "G_input":             s["inputs"].numpy(),           # (N, L)
        "G_recon":             s["recon"].numpy(),            # (N, L)
        "beta":                float(s["beta"]),
        "Ltau":                int(s["Ltau"]),
        "spectral_input_path": str(s.get("spectral_input_path", "")),
    }


def load_anacont(npz_path):
    """Load ana_cont MaxEnt summary_mean.npz."""
    if npz_path is None or not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    return {
        "omega":               d["omega"],
        "A_opt":               d["A_opt"],         # (Nw,) mean G result
        "chi2":                float(d["chi2"]),
        "backtransform":       d["backtransform"],
        "G_mean":              d["G_mean"],
        "tau":                 d["tau"],
        "beta":                float(d["beta"]),
        "spectral_input_path": str(d["spectral_input_path"]),
    }


def load_omegamaxent(npz_path):
    """Load OmegaMaxEnt summary.npz."""
    if npz_path is None or not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    return {
        "omega":               d["omega"],
        "A_opt":               d["A_opt"],
        "G_mean":              d["G_mean"],
        "tau":                 d["tau"],
        "beta":                float(d["beta"]),
        "spectral_input_path": str(d["spectral_input_path"]),
    }


# ============================================================================
# METRIC HELPERS
# ============================================================================

def l2_error(omega, A_pred, A_gt_on_omega):
    """L2 norm of (A_pred - A_gt), integrated over omega."""
    diff = A_pred - A_gt_on_omega
    return float(np.sqrt(np.trapz(diff ** 2, omega)))


def linf_error(A_pred, A_gt_on_omega):
    """Max absolute deviation."""
    return float(np.max(np.abs(A_pred - A_gt_on_omega)))


def compute_chi2(G_recon, G_input, C_inv):
    """Covariance-whitened reconstruction error averaged over time points."""
    diff = G_recon - G_input          # (..., L)
    # chi2 = diff^T C^{-1} diff / L
    tmp = diff @ C_inv                # (..., L)
    return float(np.mean(np.sum(tmp * diff, axis=-1)))


# ============================================================================
# PLOTS
# ============================================================================

def plot_spectral_avg(vae, anacont, omegamaxent, omega_gt, A_gt, out_path):
    """Dataset-averaged A(omega): VAE + MaxEnt methods + ground truth."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Ground truth
    if A_gt is not None:
        ax1.fill_between(omega_gt, A_gt, alpha=0.12, color="tab:green")
        ax1.plot(omega_gt, A_gt, color="tab:green", lw=1.8, alpha=0.9,
                 label=r"Ground truth")

    # VAE
    if vae is not None:
        A_vae_avg = vae["A_avg"]
        omega_v   = vae["omega"]
        ax1.plot(omega_v, A_vae_avg, color="tab:blue", lw=1.8,
                 label=r"VAE-AC $\langle A(\omega)\rangle$")

    # ana_cont MaxEnt
    if anacont is not None:
        ax1.plot(anacont["omega"], anacont["A_opt"],
                 color="tab:orange", lw=1.8, ls="--",
                 label=r"MaxEnt (ana\_cont)")

    # OmegaMaxEnt
    if omegamaxent is not None:
        ax1.plot(omegamaxent["omega"], omegamaxent["A_opt"],
                 color="tab:red", lw=1.8, ls="-.",
                 label=r"MaxEnt (OmegaMaxEnt)")

    ax1.set_ylabel(r"$A(\omega)$")
    ax1.set_title(r"Spectral function $A(\omega)$: VAE vs MaxEnt", fontsize=12, pad=6)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Difference panel
    ax2.axhline(0, color="gray", lw=0.8, alpha=0.5)
    if A_gt is not None:
        if vae is not None:
            omega_v = vae["omega"]
            gt_interp = np.interp(omega_v, omega_gt, A_gt)
            diff_v    = vae["A_avg"] - gt_interp
            ax2.plot(omega_v, diff_v, color="tab:blue", lw=1.2, label="VAE")
            ax2.fill_between(omega_v, diff_v, alpha=0.20, color="tab:blue")
        if anacont is not None:
            omega_m = anacont["omega"]
            gt_interp_m = np.interp(omega_m, omega_gt, A_gt)
            diff_m = anacont["A_opt"] - gt_interp_m
            ax2.plot(omega_m, diff_m, color="tab:orange", lw=1.2, ls="--",
                     label="ana\_cont")
            ax2.fill_between(omega_m, diff_m, alpha=0.20, color="tab:orange")
        if omegamaxent is not None:
            omega_o = omegamaxent["omega"]
            gt_interp_o = np.interp(omega_o, omega_gt, A_gt)
            diff_o = omegamaxent["A_opt"] - gt_interp_o
            ax2.plot(omega_o, diff_o, color="tab:red", lw=1.2, ls="-.",
                     label="OmegaMaxEnt")
            ax2.fill_between(omega_o, diff_o, alpha=0.20, color="tab:red")
        ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylabel(r"$\Delta A(\omega)$")
    ax2.set_xlabel(r"$\omega$")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_spectral_samples(vae, anacont, omegamaxent, omega_gt, A_gt,
                          sample_indices, out_path):
    """Per-sample A(omega) panels: VAE per sample vs MaxEnt mean."""
    n = len(sample_indices)
    fig = plt.figure(figsize=(9, 5 * n))
    outer = GridSpec(n, 1, figure=fig, hspace=0.55)

    omega_gt_use = omega_gt if omega_gt is not None else None

    for row, idx in enumerate(sample_indices):
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row],
            height_ratios=[3, 1], hspace=0.07,
        )
        ax_main = fig.add_subplot(inner[0])
        ax_diff = fig.add_subplot(inner[1], sharex=ax_main)

        if A_gt is not None:
            ax_main.fill_between(omega_gt_use, A_gt, alpha=0.12, color="tab:green")
            ax_main.plot(omega_gt_use, A_gt, color="tab:green", lw=1.5, alpha=0.9,
                         label="Ground truth")

        if vae is not None and idx < vae["A_mean"].shape[0]:
            A_v   = vae["A_mean"][idx]
            omega_v = vae["omega"]
            ax_main.plot(omega_v, A_v, color="tab:blue", lw=1.5,
                         label=rf"VAE sample {idx}")

        if anacont is not None:
            ax_main.plot(anacont["omega"], anacont["A_opt"],
                         color="tab:orange", lw=1.5, ls="--",
                         label=r"MaxEnt (ana\_cont, mean)")

        if omegamaxent is not None:
            ax_main.plot(omegamaxent["omega"], omegamaxent["A_opt"],
                         color="tab:red", lw=1.5, ls="-.",
                         label=r"MaxEnt (OmegaMaxEnt, mean)")

        ax_main.set_ylabel(r"$A(\omega)$")
        ax_main.set_title(rf"Sample {idx}", pad=4)
        ax_main.legend(fontsize=9)
        ax_main.grid(True, alpha=0.3)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Diff panel
        ax_diff.axhline(0, color="gray", lw=0.8, alpha=0.5)
        if A_gt is not None:
            if vae is not None and idx < vae["A_mean"].shape[0]:
                omega_v  = vae["omega"]
                gt_i     = np.interp(omega_v, omega_gt_use, A_gt)
                diff_v   = vae["A_mean"][idx] - gt_i
                ax_diff.plot(omega_v, diff_v, color="tab:blue", lw=1.2)
                ax_diff.fill_between(omega_v, diff_v, alpha=0.20, color="tab:blue")
            if anacont is not None:
                omega_m  = anacont["omega"]
                gt_m     = np.interp(omega_m, omega_gt_use, A_gt)
                diff_m   = anacont["A_opt"] - gt_m
                ax_diff.plot(omega_m, diff_m, color="tab:orange", lw=1.2, ls="--")
                ax_diff.fill_between(omega_m, diff_m, alpha=0.20, color="tab:orange")
        ax_diff.set_ylabel(r"$\Delta A$", fontsize=9)
        ax_diff.set_xlabel(r"$\omega$")
        ax_diff.grid(True, alpha=0.3)

    fig.suptitle(
        r"\textbf{VAE} (blue, per sample) vs "
        r"\textbf{MaxEnt} (orange/red, mean $G(\tau)$)"
        "\n"
        r"Ground truth: green shaded",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.93)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gtau(vae, anacont, omegamaxent, sample_indices, out_path):
    """G(tau) input vs VAE reconstruction and MaxEnt back-transformed G(tau)."""
    if vae is None:
        return

    n    = len(sample_indices)
    beta = vae["beta"]
    L    = vae["Ltau"]
    taus = np.linspace(0, beta, L)

    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)

    for row, idx in enumerate(sample_indices):
        ax = axes[row, 0]
        ax.plot(taus, vae["G_input"][idx], "k-", lw=1.5, label=r"$G(\tau)$ input")
        ax.plot(taus, vae["G_recon"][idx], "b--", lw=1.2, label="VAE recon")
        if anacont is not None:
            ax.plot(anacont["tau"],
                    anacont["backtransform"],
                    color="tab:orange", lw=1.2, ls=":",
                    label=r"MaxEnt (ana\_cont) back-transform")
        ax.set_ylabel(r"$G(\tau)$")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel(r"$\tau$")
    fig.suptitle(r"$G(\tau)$: input vs reconstructions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_and_save_metrics(vae, anacont, omegamaxent, omega_gt, A_gt, out_dir):
    """Compute and save comparison metrics to text and JSON."""
    rows = []

    for label, method_data, omega_key, A_key in [
        ("VAE-AC (avg)",         vae,          "omega", "A_avg"),
        ("MaxEnt ana_cont",      anacont,      "omega", "A_opt"),
        ("MaxEnt OmegaMaxEnt",   omegamaxent,  "omega", "A_opt"),
    ]:
        if method_data is None:
            continue
        omega_m = method_data[omega_key]
        A_m     = method_data[A_key]
        norm    = float(np.trapz(A_m, omega_m))

        if A_gt is not None and omega_gt is not None:
            gt_interp = np.interp(omega_m, omega_gt, A_gt)
            l2  = l2_error(omega_m, A_m, gt_interp)
            linf = linf_error(A_m, gt_interp)
        else:
            l2  = float("nan")
            linf = float("nan")

        chi2_val = method_data.get("chi2", float("nan"))

        rows.append({
            "method":   label,
            "norm":     round(norm, 4),
            "L2_error": round(l2,   5),
            "Linf_error": round(linf, 5),
            "chi2":     round(chi2_val, 5) if not np.isnan(chi2_val) else None,
        })

    # Print table
    header = f"\n{'Method':<30} {'Norm':>8} {'L2 err':>10} {'Linf err':>10} {'chi2':>8}"
    print(header)
    print("=" * len(header.lstrip("\n")))
    for r in rows:
        chi2_str = f"{r['chi2']:.5f}" if r["chi2"] is not None else "   n/a"
        print(f"{r['method']:<30} {r['norm']:>8.4f} {r['L2_error']:>10.5f} "
              f"{r['Linf_error']:>10.5f} {chi2_str:>8}")
    print()

    # Save
    txt_path = os.path.join(out_dir, "metrics_table.txt")
    with open(txt_path, "w") as f:
        f.write(header + "\n")
        f.write("=" * len(header.lstrip("\n")) + "\n")
        for r in rows:
            chi2_str = f"{r['chi2']:.5f}" if r["chi2"] is not None else "   n/a"
            f.write(f"{r['method']:<30} {r['norm']:>8.4f} {r['L2_error']:>10.5f} "
                    f"{r['Linf_error']:>10.5f} {chi2_str:>8}\n")
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Metrics saved: {txt_path}")
    print(f"  Metrics saved: {json_path}")

    return rows


def plot_metrics_bar(rows, out_path):
    """Bar chart comparing Norm, L2 error, and L∞ error across methods."""
    if not rows:
        return

    methods  = [r["method"] for r in rows]
    norms    = [r["norm"]       for r in rows]
    l2s      = [r["L2_error"]   for r in rows]
    linfs    = [r["Linf_error"] for r in rows]

    n      = len(methods)
    x      = np.arange(n)
    colors = ["tab:blue", "tab:orange", "tab:red"][:n]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    panels = [
        (axes[0], norms,  r"Norm $\int A(\omega)\,d\omega$",  1.0),
        (axes[1], l2s,    r"$L^2$ error",                     None),
        (axes[2], linfs,  r"$L^\infty$ error",                None),
    ]

    for ax, values, ylabel, hline in panels:
        bars = ax.bar(x, values, color=colors, width=0.5, alpha=0.85,
                      edgecolor="black", linewidth=0.7)
        if hline is not None:
            ax.axhline(hline, color="gray", lw=1.2, ls="--", alpha=0.7,
                       label=f"ideal = {hline}")
            ax.legend(fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002 * max(values),
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ymax = max(values) * 1.18
        ax.set_ylim(0, ymax if ymax > 0 else 1)

    fig.suptitle(r"Method comparison metrics", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(COMPARE_OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all results
    # ------------------------------------------------------------------
    print("Loading VAE summary...")
    vae = load_vae(VAE_SUMMARY)
    if vae is None:
        print(f"  WARNING: VAE summary not found at {VAE_SUMMARY}")

    print("Loading ana_cont MaxEnt summary...")
    anacont = load_anacont(ANACONT_SUMMARY)
    if anacont is None:
        print(f"  WARNING: ana_cont summary not found at {ANACONT_SUMMARY}")

    print("Loading OmegaMaxEnt summary...")
    omegamaxent = load_omegamaxent(OMEGAMAXENT_SUMMARY)
    if omegamaxent is None:
        print(f"  Note: OmegaMaxEnt summary not found — skipping.")

    if vae is None and anacont is None:
        raise RuntimeError("No data loaded. Run the MaxEnt and/or VAE scripts first.")

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    spec_path = None
    if vae is not None:
        spec_path = vae["spectral_input_path"]
    elif anacont is not None:
        spec_path = anacont["spectral_input_path"]
    omega_gt, A_gt = load_ground_truth(spec_path)
    if A_gt is not None:
        print(f"Ground truth loaded: {len(A_gt)} omega points")
    else:
        print(f"WARNING: ground truth not found at {spec_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_spectral_avg(
        vae, anacont, omegamaxent, omega_gt, A_gt,
        os.path.join(COMPARE_OUT_DIR, "spectral_avg.pdf"),
    )
    plot_spectral_samples(
        vae, anacont, omegamaxent, omega_gt, A_gt,
        SAMPLE_INDICES,
        os.path.join(COMPARE_OUT_DIR, "spectral_samples.pdf"),
    )
    plot_gtau(
        vae, anacont, omegamaxent,
        SAMPLE_INDICES,
        os.path.join(COMPARE_OUT_DIR, "gtau_comparison.pdf"),
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\nComputing metrics...")
    rows = print_and_save_metrics(vae, anacont, omegamaxent, omega_gt, A_gt, COMPARE_OUT_DIR)
    plot_metrics_bar(
        rows,
        os.path.join(COMPARE_OUT_DIR, "metrics_bar.pdf"),
    )

    print(f"\nAll outputs saved to: {COMPARE_OUT_DIR}")


if __name__ == "__main__":
    main()
