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


def backfill_one(summary_path, replot=True):
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
        r = backfill_one(p, replot=not args.no_replot)
        if r is None:
            continue
        rows.append(r)
        upd = ", ".join(r["updated"]) if r["updated"] else "nothing new"
        print(f"  {r['folder']}")
        print(f"     num_poles={r['num_poles']:>3}  "
              f"SC-L2={r['sc_l2_mean']:.6f}  SC-Linfty={r['sc_linf_mean']:.6f}  "
              f"[updated: {upd}]")

    if rows:
        print(f"\nDone — {len(rows)} run(s) processed.")


if __name__ == "__main__":
    main()
