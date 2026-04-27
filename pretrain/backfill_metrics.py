"""
Backfill SC-Linfty into existing summary.pt files, and regenerate
spectral_predicted.pdf with the new overlay/diff layout — without re-training.

Works purely from arrays already saved in summary.pt
(poles, residues, poles_from_avg, residues_from_avg).
AU is *not* backfilled here — it requires a model forward pass.

Usage:
    python pretrain/backfill_metrics.py                          # all runs in out/
    python pretrain/backfill_metrics.py --filter numpoles15      # filter by name
    python pretrain/backfill_metrics.py PATH/summary.pt [PATH ...]  # explicit
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain.pretrain_losses import spectral_from_poles  # type: ignore
from pretrain.plot_results import (  # type: ignore
    plot_spectral_predicted,
    plot_spectral_average,
)


SC_WMIN, SC_WMAX, SC_NW = -20.0, 20.0, 1000
AU_THRESHOLD = 1e-2


def _vae_class(model_version):
    """Dispatch MODEL_VERSION string -> model class (mirrors run_finetune.py)."""
    if model_version in (2, "2"):
        from model2_copy import VariationalAutoEncoder2 as C  # type: ignore
    elif model_version == "2t":
        from model2_tanh import VariationalAutoEncoder2 as C  # type: ignore
    elif model_version == "2s":
        from model2_silu import VariationalAutoEncoder2 as C  # type: ignore
    elif model_version == "2L":
        from model2_leaky import VariationalAutoEncoder2 as C  # type: ignore
    elif model_version == "2sc":
        from model2_silu_sym import VariationalAutoEncoder2 as C  # type: ignore
    else:
        raise ValueError(f"Unknown MODEL_VERSION={model_version!r}")
    return C


def compute_au_via_model(out_dir, summary, params):
    """Run the saved best-model encoder on the saved inputs to recover (mu, logvar)
    per sample, then compute the Burda active-units count. Returns dict to merge
    into summary.pt or None if prerequisites are missing.
    """
    ckpt_path = os.path.join(out_dir, "model", "best_model_finetune.pth")
    if not os.path.exists(ckpt_path):
        return None
    if "inputs" not in summary:
        return None

    model_version = params.get("MODEL_VERSION")
    if model_version is None:
        # Older runs may not have it — fall back to v2L (our default).
        model_version = "2L"

    VAE = _vae_class(model_version)
    input_dim = int(summary.get("Ltau", params.get("INPUT_DIM", 0)))
    num_poles = int(summary.get("num_poles", params.get("NUM_POLES", 0)))
    beta      = float(summary.get("beta", params.get("BETA", 0.0)))
    n_nodes   = int(params.get("N_NODES", 256))
    if not (input_dim and num_poles and beta):
        return None

    latent_dim_override = params.get("LATENT_DIM")  # may be None
    model = VAE(input_dim=input_dim, num_poles=num_poles, beta=beta, N_nodes=n_nodes,
                latent_dim=latent_dim_override)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    inputs = summary["inputs"]
    if inputs.dim() != 2:
        inputs = inputs.view(-1, input_dim)
    with torch.no_grad():
        # Forward pass, deterministic (z = mu) — only mu, logvar are needed for AU
        mu, logvar, _, _, _, _ = model(inputs, deterministic=True)
    mu_np     = mu.cpu().numpy()
    logvar_np = logvar.cpu().numpy()

    var_mu = mu_np.var(axis=0, ddof=0)
    sigma2 = np.exp(logvar_np)
    kl_per_dim = 0.5 * (mu_np ** 2 + sigma2 - 1.0 - logvar_np).mean(axis=0)
    n_active   = int((var_mu > AU_THRESHOLD).sum())
    latent_dim = int(var_mu.shape[0])
    return {
        "active_units":   n_active,
        "latent_dim":     latent_dim,
        "au_threshold":   AU_THRESHOLD,
        "var_mu_per_dim": var_mu,
        "kl_per_dim":     kl_per_dim,
        "kl_total_avg":   float(kl_per_dim.sum()),
    }


def discover_summaries(out_root, filter_str=None):
    paths = []
    for name in sorted(os.listdir(out_root)):
        if not name.startswith("finetune_"):
            continue
        if filter_str and filter_str not in name:
            continue
        p = os.path.join(out_root, name, "summary.pt")
        if os.path.exists(p):
            paths.append(p)
    return paths


def backfill_one(summary_path, replot=True, with_au=False):
    s = torch.load(summary_path, weights_only=False)
    out_dir = os.path.dirname(summary_path)
    folder  = os.path.basename(out_dir)

    needs = {
        "poles":             "poles" in s,
        "residues":          "residues" in s,
        "poles_from_avg":    "poles_from_avg" in s,
        "residues_from_avg": "residues_from_avg" in s,
    }
    if not all(needs.values()):
        missing = [k for k, v in needs.items() if not v]
        print(f"  [SKIP] {folder}: missing {missing}")
        return None

    omega_sc = torch.linspace(SC_WMIN, SC_WMAX, SC_NW)
    with torch.no_grad():
        A_samples  = spectral_from_poles(s["poles"], s["residues"], omega_sc)  # (N, Nw)
        A_from_avg = spectral_from_poles(
            s["poles_from_avg"].unsqueeze(0),
            s["residues_from_avg"].unsqueeze(0),
            omega_sc,
        ).squeeze(0)                                                            # (Nw,)
    diff = (A_samples - A_from_avg.unsqueeze(0)).numpy()
    sc_l2_per_sample   = np.trapezoid(diff ** 2, omega_sc.numpy(), axis=1)
    sc_linf_per_sample = np.max(np.abs(diff), axis=1)

    sc_l2_mean   = float(sc_l2_per_sample.mean())
    sc_l2_std    = float(sc_l2_per_sample.std())
    sc_linf_mean = float(sc_linf_per_sample.mean())
    sc_linf_std  = float(sc_linf_per_sample.std())

    # Update summary.pt — keep existing SC-L2 if already present (don't clobber
    # whatever was written by the live training run); only fill missing fields.
    updated_keys = []
    if "self_consistency" not in s:
        s["self_consistency"]            = sc_l2_mean
        s["self_consistency_std"]        = sc_l2_std
        s["self_consistency_per_sample"] = sc_l2_per_sample
        s["self_consistency_grid"]       = {"wmin": SC_WMIN, "wmax": SC_WMAX, "Nw": SC_NW}
        updated_keys.append("SC-L2")
    if "self_consistency_linfty" not in s:
        s["self_consistency_linfty"]            = sc_linf_mean
        s["self_consistency_linfty_std"]        = sc_linf_std
        s["self_consistency_linfty_per_sample"] = sc_linf_per_sample
        updated_keys.append("SC-Linfty")

    # Optional: backfill active-units by loading model + running encoder on inputs
    au_info = None
    if with_au and "active_units" not in s:
        params_path = os.path.join(out_dir, "params.json")
        params = {}
        if os.path.exists(params_path):
            import json as _json
            with open(params_path) as _f:
                params = _json.load(_f)
        au_info = compute_au_via_model(out_dir, s, params)
        if au_info is not None:
            s.update(au_info)
            updated_keys.append("AU")

    if updated_keys:
        torch.save(s, summary_path)

    # Regenerate per-folder spectral plots with the current plotting code
    plots_dir = os.path.join(out_dir, "plots")
    if replot and os.path.isdir(plots_dir):
        # Ground-truth A(omega) only present for synthetic data
        A_input = omega_input = None
        spec_path = s.get("spectral_input_path")
        if spec_path and os.path.exists(str(spec_path)):
            A_input = np.loadtxt(spec_path, delimiter=",")
            omega_input = np.linspace(-5.0, 5.0, len(A_input))
        plot_spectral_predicted(
            s["poles"], s["residues"],
            A_input=A_input, omega_input=omega_input,
            n_samples=6,
            poles_from_avg=s["poles_from_avg"],
            residues_from_avg=s["residues_from_avg"],
            save_path=os.path.join(plots_dir, "spectral_predicted.pdf"),
        )
        plot_spectral_average(
            s["poles"], s["residues"],
            A_input=A_input, omega_input=omega_input,
            poles_from_avg=s["poles_from_avg"],
            residues_from_avg=s["residues_from_avg"],
            save_path=os.path.join(plots_dir, "spectral_average.pdf"),
        )

    return {
        "folder":       folder,
        "num_poles":    int(s.get("num_poles", -1)),
        "n_epochs":     (int(s["n_epochs"]) if "n_epochs" in s else None),
        "sc_l2_mean":   sc_l2_mean,
        "sc_linf_mean": sc_linf_mean,
        "active_units": (int(s["active_units"]) if "active_units" in s else None),
        "latent_dim":   (int(s["latent_dim"])   if "latent_dim"   in s else None),
        "updated":      updated_keys,
    }


def main():
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*",
                        help="explicit summary.pt paths; if empty, walk out/")
    parser.add_argument("--out-root", default=os.path.join(main_path, "out"),
                        help="root for auto-discovery (default: out/)")
    parser.add_argument("--filter", default=None,
                        help="substring filter on folder name (auto-discovery only)")
    parser.add_argument("--no-replot", action="store_true",
                        help="skip regenerating spectral_predicted.pdf and spectral_average.pdf")
    parser.add_argument("--epochs", type=int, default=None,
                        help="keep only runs with this exact n_epochs "
                             "(applied after auto-discovery; ignored when paths are given)")
    parser.add_argument("--with-au", action="store_true",
                        help="backfill active-units by loading the saved model "
                             "checkpoint and running the encoder on saved inputs")
    args = parser.parse_args()

    if args.paths:
        paths = args.paths
    else:
        paths = discover_summaries(args.out_root, filter_str=args.filter)
        if args.epochs is not None:
            before = len(paths)
            kept = []
            for p in paths:
                try:
                    s = torch.load(p, weights_only=False, map_location="cpu")
                    if int(s.get("n_epochs", -1)) == args.epochs:
                        kept.append(p)
                except Exception:
                    pass
            paths = kept
            print(f"Filtered by n_epochs={args.epochs}: kept {len(paths)} of {before} runs")

    if not paths:
        print("No summary.pt files matched.")
        return

    print(f"Backfilling {len(paths)} run(s)...\n")
    rows = []
    for p in paths:
        r = backfill_one(p, replot=not args.no_replot, with_au=args.with_au)
        if r is None:
            continue
        rows.append(r)
        upd    = ", ".join(r["updated"]) if r["updated"] else "nothing new"
        au_str = (f"  AU={r['active_units']}/{r['latent_dim']}"
                  if r.get("active_units") is not None else "")
        print(f"  {r['folder']}")
        print(f"     num_poles={r['num_poles']:>3}  "
              f"SC-L2={r['sc_l2_mean']:.6f}  SC-Linfty={r['sc_linf_mean']:.6f}{au_str}  "
              f"[updated: {upd}]")

    if rows:
        print(f"\nDone — {len(rows)} run(s) processed.")


if __name__ == "__main__":
    main()
