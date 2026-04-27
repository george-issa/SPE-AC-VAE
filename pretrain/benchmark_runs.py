"""
benchmark_runs.py — supervisor-ready cross-run benchmark for finetune outputs.

For each run dir, reads summary.pt + params.json + losses/chi2_losses_finetune.npy
and reports an extended metrics block plus a cross-run markdown table:

  - chi^2 init / best (epoch) / final
  - active units AU/L, KL total
  - self-consistency SC-L2 and SC-Linf
  - tail-expansion coefficients c_n of <A(w)> ~ sum_n c_n / w^(n+1)
    (PH runs should have c_0 = c_2 = c_4 = ... = 0 exactly)
  - dataset-avg spectral asymmetry max|A(w)-A(-w)| / max|A|
  - peak positions of <A(w)> above half-max

With --plot OUTDIR, writes:
  spectral_avg_compare.pdf : <A(w)> overlay across runs
  asymmetry_compare.pdf    : A(w)-A(-w) overlay; flat zero for PH runs
  loss_curves_compare.pdf  : chi^2 per epoch overlay
  tail_coeffs_compare.pdf  : bar chart of c_1 and c_3 across runs

Usage:
  python pretrain/benchmark_runs.py <run_dir> [<run_dir> ...]
  python pretrain/benchmark_runs.py PH:<dir> base8:<dir> base4:<dir>
  python pretrain/benchmark_runs.py <runs...> --plot out/comparison_ph_vs_base/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrain.pretrain_losses import spectral_from_poles  # noqa: E402


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def parse_run_spec(spec: str):
    """'label:path' -> (label, path); 'path' -> (None, path).
    Falls back to bare-path if the prefix doesn't precede a real directory.
    """
    if ":" in spec:
        label, _, path = spec.partition(":")
        if os.path.isdir(path):
            return label, path
    return None, spec


def load_run(run_dir: str):
    summary = torch.load(os.path.join(run_dir, "summary.pt"), weights_only=False)
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)
    chi2_curve = np.load(os.path.join(run_dir, "losses", "chi2_losses_finetune.npy"))
    return {"summary": summary, "params": params, "chi2_curve": chi2_curve, "dir": run_dir}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def tail_coeffs(poles: torch.Tensor, residues: torch.Tensor, n_max: int = 4):
    """A(w) ~ sum_n c_n / w^(n+1), with c_n = -(1/pi) Im(sum_p r_p z_p^n).

    Returns a (..., n_max+1) tensor, one row per sample. Computed in float64
    so the PH-zero coefficients hit machine precision.
    """
    z = poles.to(torch.complex128)
    r = residues.to(torch.complex128)
    out = []
    z_pow = torch.ones_like(z)
    for _ in range(n_max + 1):
        out.append(-(1.0 / np.pi) * (r * z_pow).imag.sum(dim=-1))
        z_pow = z_pow * z
    return torch.stack(out, dim=-1)


def spectral_avg(poles: torch.Tensor, residues: torch.Tensor,
                 omega_max: float = 5.0, n_omega: int = 1001):
    """<A(w)> averaged over the per-sample posterior."""
    omegas = torch.linspace(-omega_max, omega_max, n_omega)
    A = spectral_from_poles(poles, residues, omegas)        # (B, M)
    return omegas, A.mean(dim=0)                            # (M,), (M,)


def spectral_asymmetry(omegas: torch.Tensor, A_mean: torch.Tensor):
    A_flip = torch.flip(A_mean, dims=[0])
    asym_abs = (A_mean - A_flip).abs().max().item()
    asym_rel = asym_abs / (A_mean.abs().max().item() + 1e-30)
    return asym_abs, asym_rel, (A_mean - A_flip)


def peak_positions(omegas: torch.Tensor, A_mean: torch.Tensor,
                   frac: float = 0.5, top_k: int = 4):
    """Local maxima of A(w) above frac*A.max(); returned in omega order."""
    A = A_mean.numpy()
    w = omegas.numpy()
    thresh = A.max() * frac
    peaks = [(float(w[i]), float(A[i]))
             for i in range(1, len(A) - 1)
             if A[i] > A[i - 1] and A[i] > A[i + 1] and A[i] > thresh]
    peaks.sort(key=lambda x: -x[1])
    return sorted(peaks[:top_k], key=lambda x: x[0])


def benchmark(run, label: Optional[str] = None):
    s, params, curve = run["summary"], run["params"], run["chi2_curve"]
    poles, residues = s["poles"], s["residues"]

    omegas, A_mean = spectral_avg(poles, residues)
    asym_abs, asym_rel, asym_curve = spectral_asymmetry(omegas, A_mean)
    c_per = tail_coeffs(poles, residues, n_max=4)        # (N, 5)
    peaks = peak_positions(omegas, A_mean)

    return {
        "label": label or os.path.basename(run["dir"].rstrip("/")),
        "dir": run["dir"],
        "params": {
            "PH_SYMMETRIC":    bool(params.get("PH_SYMMETRIC", False)),
            "NUM_POLES":       params.get("NUM_POLES"),
            "NUM_POLES_EFF":   params.get("NUM_POLES_EFFECTIVE", params.get("NUM_POLES")),
            "LATENT_DIM":      params.get("LATENT_DIM"),
            "MODEL_VERSION":   params.get("MODEL_VERSION"),
            "FINETUNE_EPOCHS": params.get("FINETUNE_EPOCHS"),
            "DATA_SOURCE":     params.get("DATA_SOURCE"),
        },
        "chi2_init":     float(curve[0]),
        "chi2_min":      float(s["best_chi2"]),
        "chi2_final":    float(s["final_chi2"]),
        "best_epoch":    int(s["best_chi2_epoch"]),
        "n_epochs":      int(s["n_epochs"]),
        "AU":            int(s["active_units"]),
        "L":             int(s["latent_dim"]),
        "KL_total":      float(s["kl_total_avg"]),
        "SC_L2":         float(s["self_consistency"]),
        "SC_Linf":       float(s["self_consistency_linfty"]),
        "asym_abs":      asym_abs,
        "asym_rel":      asym_rel,
        "tail_mean":     [float(x) for x in c_per.mean(dim=0).tolist()],
        "tail_std":      [float(x) for x in c_per.std(dim=0).tolist()],
        "peaks":         peaks,
        "omegas":        omegas.numpy(),
        "A_mean":        A_mean.numpy(),
        "asym_curve":    asym_curve.numpy(),
        "chi2_curve":    curve,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_extended(b):
    p = b["params"]
    print(f"\n=== {b['label']} ===")
    print(f"  dir:        {b['dir']}")
    print(f"  config:     PH={'on' if p['PH_SYMMETRIC'] else 'off'}  "
          f"P_free={p['NUM_POLES']}  P_eff={p['NUM_POLES_EFF']}  "
          f"L={p['LATENT_DIM']}  epochs={b['n_epochs']}  model={p['MODEL_VERSION']}")
    print(f"  chi^2:      init={b['chi2_init']:.3e}  "
          f"best={b['chi2_min']:.4f} @ ep {b['best_epoch']}  final={b['chi2_final']:.4f}")
    print(f"  AU/L:       {b['AU']}/{b['L']}    KL_total={b['KL_total']:.2f}")
    print(f"  SC-L2:      {b['SC_L2']:.4f}    SC-Linf={b['SC_Linf']:.4f}")
    print(f"  asymmetry:  max|A(w)-A(-w)|={b['asym_abs']:.2e}    rel={b['asym_rel']:.2e}")
    print(f"  tail coeffs (per-sample mean +/- std):")
    for n, (m, sd) in enumerate(zip(b["tail_mean"], b["tail_std"])):
        flag = "  <- 0 by PH" if (p["PH_SYMMETRIC"] and n % 2 == 0) else ""
        print(f"    c_{n} (1/w^{n+1}): {m:+.4e} +/- {sd:.2e}{flag}")
    if b["peaks"]:
        print(f"  peaks of <A(w)> above half-max:")
        for w, h in b["peaks"]:
            print(f"    omega = {w:+.3f}, A = {h:.4f}")


def print_markdown(blocks):
    cols = ["run", "PH", "P_free", "P_eff", "L",
            "chi^2_min", "chi^2_final", "AU/L", "KL",
            "SC-L2", "SC-Linf", "rel asym", "peaks (omega)"]
    print("\n## Cross-run comparison\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for b in blocks:
        p = b["params"]
        peak_str = ", ".join(f"{w:+.2f}" for w, _ in b["peaks"]) or "-"
        row = [
            b["label"],
            "on" if p["PH_SYMMETRIC"] else "off",
            str(p["NUM_POLES"]),
            str(p["NUM_POLES_EFF"]),
            str(p["LATENT_DIM"]),
            f"{b['chi2_min']:.4f}",
            f"{b['chi2_final']:.4f}",
            f"{b['AU']}/{b['L']}",
            f"{b['KL_total']:.1f}",
            f"{b['SC_L2']:.4f}",
            f"{b['SC_Linf']:.4f}",
            f"{b['asym_rel']:.1e}",
            peak_str,
        ]
        print("| " + " | ".join(row) + " |")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def make_plots(blocks, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    os.makedirs(out_dir, exist_ok=True)

    # 1. Spectral overlay
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in blocks:
        ax.plot(b["omegas"], b["A_mean"], lw=1.6, label=b["label"])
    ax.axvline(0, color="gray", alpha=0.4, lw=0.8)
    ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$\langle A(\omega) \rangle$")
    ax.set_title(r"Dataset-averaged spectral function")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "spectral_avg_compare.pdf")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

    # 2. Asymmetry residual A(w) - A(-w). Flat zero for PH; finite wobble for unconstrained.
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in blocks:
        ax.plot(b["omegas"], b["asym_curve"], lw=1.4, label=b["label"])
    ax.axhline(0, color="gray", alpha=0.5, lw=0.8)
    ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$\langle A(\omega)\rangle - \langle A(-\omega)\rangle$")
    ax.set_title(r"Spectral asymmetry residual (PH-symmetric runs are flat zero)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "asymmetry_compare.pdf")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

    # 3. Loss curves chi^2 vs epoch
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in blocks:
        ep = np.arange(1, len(b["chi2_curve"]) + 1)
        ax.plot(ep, b["chi2_curve"], lw=1.2, label=b["label"])
    ax.axhline(1.0, color="gray", alpha=0.5, ls="--", label=r"$\chi^2 = 1$")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\chi^2$")
    ax.set_title(r"Fine-tune $\chi^2$ vs epoch")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "loss_curves_compare.pdf")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

    # 4. Tail coefficients (only c_1, c_3 are physically informative; c_0/c_2/c_4 are 0 by PH)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    labels = [b["label"] for b in blocks]
    for ax, n, ylabel in [(axes[0], 1, r"$c_1$ (coef of $1/\omega^2$)"),
                          (axes[1], 3, r"$c_3$ (coef of $1/\omega^4$)")]:
        means = [b["tail_mean"][n] for b in blocks]
        stds  = [b["tail_std"][n]  for b in blocks]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.axhline(0, color="gray", alpha=0.5, lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    fig.suptitle(r"Tail-expansion coefficients (per-sample mean $\pm$ std)")
    fig.tight_layout()
    p = os.path.join(out_dir, "tail_coeffs_compare.pdf")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="run dirs, optionally prefixed 'label:path'")
    ap.add_argument("--plot", default=None,
                    help="write comparison plots to this directory")
    args = ap.parse_args()

    blocks: List[dict] = []
    for spec in args.runs:
        label, run_dir = parse_run_spec(spec)
        run = load_run(run_dir)
        b = benchmark(run, label=label)
        print_extended(b)
        blocks.append(b)

    print_markdown(blocks)

    if args.plot:
        make_plots(blocks, args.plot)


if __name__ == "__main__":
    main()
