"""
Encoder-audit: diagnostic for whether the trained encoder is doing anything,
and what the decoder's response surface to z looks like.

Three forward-pass-only tests on a finetune run dir (no retraining):

  Test 1  — encoder outputs over the dataset
            histogram of mu_z(x_i) and sigma_z(x_i) per latent dim, vs the
            prior N(0, 1). AU = #{d : Var_x[mu_d] > 0.01} is the standard
            (Burda et al.) collapse diagnostic; this view also reveals a
            non-zero constant offset, which AU misses (KL is then driven by
            the offset, not by per-sample work).

  Test 2  — decoder z-sweep
            sweep z over (a) the encoder's empirical operating range
            [mu_bar - 3 sigma_bar, mu_bar + 3 sigma_bar], and (b) the prior
            range [-3, +3]. Plot A_z(omega) for each. If the curves overlap,
            the decoder ignores z; if they vary smoothly along one feature,
            the latent is doing one well-defined thing.

  Test 3  — encoder bypass at inference
            <A(omega)> with the encoder ON (per-sample z = mu(x_i), and
            stochastic MC=50) vs BYPASS (z=0, z=mu_bar, z ~ p(z) MC=200).
            L2 between each pair quantifies how much the encoder contributes
            beyond a fixed scalar input.

Usage:
    python pretrain/encoder_audit.py <run_dir>

Outputs (under <run_dir>/plots/):
    encoder_outputs.pdf          — Test 1 histograms
    decoder_z_sweep.pdf          — Test 2 sweep curves
    encoder_bypass.pdf           — Test 3 <A(omega)> comparison
    encoder_audit_summary.txt    — numerical summary (in run_dir, not plots/)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_process_real import (  # type: ignore
    HolsteinJLD2Dataset,
    SmoQyV2Dataset,
)
from data_process import GreenFunctionDataset  # type: ignore
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


def _import_vae(model_version):
    if model_version == "2L":
        from model2_leaky import VariationalAutoEncoder2 as VAE  # type: ignore
    elif model_version == "2t":
        from model2_tanh import VariationalAutoEncoder2 as VAE  # type: ignore
    elif model_version == "2s":
        from model2_silu import VariationalAutoEncoder2 as VAE  # type: ignore
    elif model_version == "2sc":
        from model2_silu_sym import VariationalAutoEncoder2 as VAE  # type: ignore
    elif model_version in (2, "2"):
        from model2_copy import VariationalAutoEncoder2 as VAE  # type: ignore
    else:
        raise ValueError(f"Unknown MODEL_VERSION={model_version!r}")
    return VAE


def _load_dataset(params):
    src = params["DATA_SOURCE"]
    if src == "holstein_jld2":
        return HolsteinJLD2Dataset(
            params["DATA_PATH"],
            n_idx=params["HOLSTEIN_N_IDX"],
            omega_idx=params["HOLSTEIN_OMEGA_IDX"],
            beta_idx=params["HOLSTEIN_BETA_IDX"],
        )
    if src == "real":
        return SmoQyV2Dataset(params["DATA_PATH"], r1=0, r2=0)
    if src == "synthetic":
        return GreenFunctionDataset(file_path=params["DATA_PATH"])
    raise ValueError(f"Unknown DATA_SOURCE={src!r}")


def _l2(omega_np, a, b):
    return float(np.sqrt(np.trapezoid((a - b) ** 2, omega_np)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="cpu",
                        help="cpu / mps / cuda (default: cpu — diagnostics are tiny)")
    parser.add_argument("--n-mc", type=int, default=50,
                        help="MC samples for stochastic encoder-on inference")
    parser.add_argument("--n-prior", type=int, default=200,
                        help="MC samples for prior-bypass inference")
    parser.add_argument("--n-z-sweep", type=int, default=25,
                        help="number of curves in each decoder z-sweep panel")
    parser.add_argument("--wmin", type=float, default=-20.0)
    parser.add_argument("--wmax", type=float, default=20.0)
    parser.add_argument("--n-w", type=int, default=1000)
    parser.add_argument("--xlim", type=float, default=10.0,
                        help="abs(omega) limit for plot view (default 10)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)

    device = torch.device(args.device)
    VAE = _import_vae(params["MODEL_VERSION"])
    model = VAE(
        input_dim=params["INPUT_DIM"],
        num_poles=params["NUM_POLES"],
        beta=params["BETA"],
        N_nodes=params["N_NODES"],
        latent_dim=params["LATENT_DIM"],
        ph_symmetric=params.get("PH_SYMMETRIC", False),
    ).to(device)

    state = torch.load(
        os.path.join(run_dir, "model", "best_model_finetune.pth"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    ds = _load_dataset(params)
    loader = DataLoader(ds, batch_size=params["BATCH_SIZE"], shuffle=False)

    # ------------------------------------------------------------------ #
    # Test 1 — encoder outputs over the dataset
    # ------------------------------------------------------------------ #
    mu_chunks = []
    logvar_chunks = []
    with torch.no_grad():
        for batch in loader:
            B = batch.shape[0]
            batch = batch.view(B, params["INPUT_DIM"]).to(device)
            mu, logvar = model.encode(batch)
            mu_chunks.append(mu.cpu())
            logvar_chunks.append(logvar.cpu())
    mu_all = torch.cat(mu_chunks, dim=0).numpy()           # (N, L)
    logvar_all = torch.cat(logvar_chunks, dim=0).numpy()   # (N, L)
    sigma_all = np.exp(0.5 * np.clip(logvar_all, -25, 25))  # (N, L)

    N, L = mu_all.shape
    mu_bar = mu_all.mean(axis=0)
    mu_std = mu_all.std(axis=0)
    sigma_bar = sigma_all.mean(axis=0)
    sigma_std = sigma_all.std(axis=0)
    var_mu = mu_all.var(axis=0, ddof=0)

    var_q = sigma_all ** 2
    kl_per_dim_per_sample = 0.5 * (mu_all ** 2 + var_q - 1.0 - logvar_all)
    kl_per_dim = kl_per_dim_per_sample.mean(axis=0)
    kl_total = float(kl_per_dim.sum())
    au = int((var_mu > 1e-2).sum())

    # ------------------------------------------------------------------ #
    # Test 2 — decoder z-sweep
    # Empirical range: per-dim [mu_bar - 3 sigma_bar, mu_bar + 3 sigma_bar].
    # Prior range:     per-dim [-3, +3]. For L > 1 we sweep all dims along
    # the same fraction of their range (a 1-D path through z-space) — it
    # exercises the decoder along the most natural diagonal but does NOT
    # exhaustively probe higher-dim response surfaces. For L = 1 (the case
    # this script was written for) this is the full sweep.
    # ------------------------------------------------------------------ #
    omega = torch.linspace(args.wmin, args.wmax, args.n_w)
    omega_np = omega.numpy()
    Nz = int(args.n_z_sweep)

    sweeps = {}
    for label, lo, hi in (
        ("empirical", mu_bar - 3.0 * sigma_bar, mu_bar + 3.0 * sigma_bar),
        ("prior",     -3.0 * np.ones(L),         +3.0 * np.ones(L)),
    ):
        z_grid = np.linspace(lo, hi, Nz).astype(np.float32)   # (Nz, L)
        z_t = torch.tensor(z_grid, device=device)
        with torch.no_grad():
            poles_z, residues_z = model.decode_poles_residues(z_t)
            A_z = spectral_from_poles(poles_z, residues_z, omega).cpu().numpy()
        sweeps[label] = {"z_values": z_grid, "A": A_z}

    # ------------------------------------------------------------------ #
    # Test 3 — encoder bypass at inference
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        # Encoder ON, deterministic: z_i = mu(x_i), per-sample
        z_on_det = torch.tensor(mu_all, dtype=torch.float32, device=device)
        poles_on, res_on = model.decode_poles_residues(z_on_det)
        A_on = spectral_from_poles(poles_on, res_on, omega).cpu().numpy()  # (N, Nw)
        A_on_mean = A_on.mean(axis=0)

        # Encoder ON, stochastic MC: z_i ~ q(z|x_i)
        mu_t = torch.tensor(mu_all, dtype=torch.float32, device=device)
        sd_t = torch.tensor(sigma_all, dtype=torch.float32, device=device)
        A_on_stoch_acc = np.zeros((N, args.n_w), dtype=np.float64)
        for _ in range(args.n_mc):
            z_s = mu_t + torch.randn_like(mu_t) * sd_t
            poles_s, res_s = model.decode_poles_residues(z_s)
            A_on_stoch_acc += spectral_from_poles(poles_s, res_s, omega).cpu().numpy()
        A_on_stoch = A_on_stoch_acc / args.n_mc
        A_on_stoch_mean = A_on_stoch.mean(axis=0)

        # Bypass: z = 0 (single forward, no encoder)
        z_zero = torch.zeros(1, L, device=device)
        poles_z0, res_z0 = model.decode_poles_residues(z_zero)
        A_z0 = spectral_from_poles(poles_z0, res_z0, omega).cpu().numpy().squeeze()

        # Bypass: z = mu_bar (single forward at the encoder's empirical mean)
        z_mb = torch.tensor(mu_bar, dtype=torch.float32, device=device).unsqueeze(0)
        poles_mb, res_mb = model.decode_poles_residues(z_mb)
        A_mb = spectral_from_poles(poles_mb, res_mb, omega).cpu().numpy().squeeze()

        # Bypass: z ~ p(z) MC (no encoder, marginalize over the prior)
        z_prior = torch.randn(args.n_prior, L, device=device)
        poles_pr, res_pr = model.decode_poles_residues(z_prior)
        A_prior_samples = spectral_from_poles(poles_pr, res_pr, omega).cpu().numpy()
        A_prior_mean = A_prior_samples.mean(axis=0)

    diffs = {
        "encoder_ON_det vs encoder_ON_stoch": _l2(omega_np, A_on_mean, A_on_stoch_mean),
        "encoder_ON_det vs bypass_z=0":       _l2(omega_np, A_on_mean, A_z0),
        "encoder_ON_det vs bypass_z=mu_bar":  _l2(omega_np, A_on_mean, A_mb),
        "encoder_ON_det vs bypass_prior_MC":  _l2(omega_np, A_on_mean, A_prior_mean),
        "bypass_z=0 vs bypass_z=mu_bar":      _l2(omega_np, A_z0, A_mb),
    }

    # ------------------------------------------------------------------ #
    # Plots + summary
    # ------------------------------------------------------------------ #
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update(PLOT_RC)

    # Test 1 — encoder outputs
    fig, axes = plt.subplots(L, 2, figsize=(9, 2.8 * L), squeeze=False)
    for d in range(L):
        ax_mu = axes[d, 0]
        ax_sd = axes[d, 1]

        ax_mu.hist(mu_all[:, d], bins=20, color="C0", alpha=0.75, edgecolor="k")
        ax_mu.axvline(0.0, ls="--", color="k", alpha=0.5,
                      label=r"prior: $\mu = 0$")
        ax_mu.axvline(mu_bar[d], ls=":", color="C3", lw=1.6,
                      label=fr"$\bar\mu = {mu_bar[d]:+.3f}$ "
                            fr"(std $= {mu_std[d]:.2e}$)")
        ax_mu.set_xlabel(fr"$\mu_z[{d}]$")
        ax_mu.set_ylabel("count")
        ax_mu.legend(frameon=False, fontsize=9, loc="best")

        ax_sd.hist(sigma_all[:, d], bins=20, color="C1", alpha=0.75, edgecolor="k")
        ax_sd.axvline(1.0, ls="--", color="k", alpha=0.5,
                      label=r"prior: $\sigma = 1$")
        ax_sd.axvline(sigma_bar[d], ls=":", color="C3", lw=1.6,
                      label=fr"$\bar\sigma = {sigma_bar[d]:.3g}$ "
                            fr"(std $= {sigma_std[d]:.2e}$)")
        ax_sd.set_xlabel(fr"$\sigma_z[{d}]$")
        ax_sd.set_ylabel("count")
        ax_sd.legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle(
        fr"Encoder outputs over dataset ($N={N}$, $L_z={L}$). "
        fr"AU $= {au}/{L}$, KL$_{{\rm total}} = {kl_total:.3f}$"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "encoder_outputs.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Test 2 — decoder z-sweep
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    cmap = plt.cm.viridis
    for ax, label in zip(axes, ("empirical", "prior")):
        sweep = sweeps[label]
        z_vals = sweep["z_values"]
        A = sweep["A"]
        if L == 1:
            zlo, zhi = float(z_vals[0, 0]), float(z_vals[-1, 0])
        else:
            zlo, zhi = 0.0, float(Nz - 1)
        norm = plt.Normalize(vmin=zlo, vmax=zhi)

        for k in range(Nz):
            if L == 1:
                frac = (z_vals[k, 0] - zlo) / max(zhi - zlo, 1e-12)
            else:
                frac = k / max(Nz - 1, 1)
            ax.plot(omega_np, A[k], color=cmap(frac), lw=1.0, alpha=0.85)

        ax.plot(omega_np, A_on_mean, color="black", lw=1.8, ls="--",
                label=r"encoder-on $\langle A\rangle$")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$A(\omega)$")
        ax.set_xlim(-args.xlim, args.xlim)
        if label == "empirical":
            ax.set_title(
                fr"$z$-sweep, empirical: $z[0] \in [{zlo:+.2f}, {zhi:+.2f}]$"
                fr"\;($\bar\mu \pm 3\bar\sigma$)" if L == 1 else
                f"z-sweep, empirical: 1-D path of {Nz} pts"
            )
        else:
            ax.set_title(
                fr"$z$-sweep, prior: $z[0] \in [-3, +3]$"
                if L == 1 else
                f"z-sweep, prior: 1-D path of {Nz} pts"
            )
        ax.legend(frameon=False, loc="best")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb_label = r"$z$" if L == 1 else r"$z$ index"
        plt.colorbar(sm, ax=ax, label=cb_label)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "decoder_z_sweep.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Test 3 — encoder bypass comparison
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(omega_np, A_on_mean, color="black", lw=1.8, ls="-",
            label=r"encoder ON (det): $z_i = \mu_z(x_i)$")
    ax.plot(omega_np, A_on_stoch_mean, color="C0", lw=1.2, ls="--",
            label=fr"encoder ON (stoch, MC$={args.n_mc}$)")
    ax.plot(omega_np, A_z0, color="C3", lw=1.2,
            label=r"bypass: $z = 0$")
    if L == 1:
        bypass_mb_lbl = fr"bypass: $z = \bar\mu = {mu_bar[0]:+.3f}$"
    else:
        bypass_mb_lbl = r"bypass: $z = \bar\mu$"
    ax.plot(omega_np, A_mb, color="C2", lw=1.2, label=bypass_mb_lbl)
    ax.plot(omega_np, A_prior_mean, color="C4", lw=1.2,
            label=fr"bypass: $z \sim p(z)$ (MC$={args.n_prior}$)")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\langle A(\omega)\rangle$")
    ax.set_title("Encoder-bypass comparison")
    ax.set_xlim(-args.xlim, args.xlim)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "encoder_bypass.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Text summary
    lines = []
    lines.append("ENCODER AUDIT")
    lines.append("=" * 72)
    lines.append(f"run_dir : {run_dir}")
    lines.append(f"N samples = {N}    L_z = {L}    P = {params['NUM_POLES']}")
    lines.append(f"AU = {au}/{L}  (Var_x[mu] > 0.01)    KL_total = {kl_total:.4f}")
    lines.append("")
    lines.append("Test 1 — encoder outputs (per latent dim):")
    for d in range(L):
        lines.append(
            f"  dim {d}: "
            f"mu_bar = {mu_bar[d]:+.4f}  "
            f"std(mu) = {mu_std[d]:.2e}  "
            f"Var_x[mu] = {var_mu[d]:.4e}  ||  "
            f"sigma_bar = {sigma_bar[d]:.4e}  "
            f"std(sigma) = {sigma_std[d]:.2e}  ||  "
            f"KL = {kl_per_dim[d]:.4e}"
        )
    lines.append("")
    lines.append("Test 3 — encoder-bypass L2 differences "
                 "(integrated over omega in [{:g}, {:g}]):".format(args.wmin, args.wmax))
    for k, v in diffs.items():
        lines.append(f"  {k:42s} : {v:.4e}")
    lines.append("")
    lines.append("Plots:")
    lines.append(f"  {os.path.join(plot_dir, 'encoder_outputs.pdf')}")
    lines.append(f"  {os.path.join(plot_dir, 'decoder_z_sweep.pdf')}")
    lines.append(f"  {os.path.join(plot_dir, 'encoder_bypass.pdf')}")
    summary = "\n".join(lines)

    print(summary)
    with open(os.path.join(run_dir, "encoder_audit_summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
