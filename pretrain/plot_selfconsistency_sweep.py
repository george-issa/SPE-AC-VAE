"""
Plot self-consistency (SC) vs number of poles, swept across fine-tuning runs.

SC is the per-run model-selection signal saved by run_finetune.py:
    SC = (1/N) sum_i  int |A_i(w | G_i) - A(w | <G>)|^2 dw

Walks `out/finetune_*/summary.pt`, groups runs by (data_source, sim_name,
init_tag, model_tag), and plots SC vs NUM_POLES for each group.

Outputs (in OUT_DIR):
  selfconsistency_sweep.pdf  — SC vs num_poles, one curve per (sim, init, model)
  selfconsistency_sweep.csv  — raw table of (folder, num_poles, SC, ...)

Usage:
    python pretrain/plot_selfconsistency_sweep.py
    python pretrain/plot_selfconsistency_sweep.py --filter hubbard
    python pretrain/plot_selfconsistency_sweep.py --out-root /path/to/out
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================================================
# CONFIG — toggles & defaults (override per-invocation via CLI args)
# ==========================================================================

# Filter runs by exact n_epochs (int) or None to keep every run.
EPOCHS_FILTER = 200

# Show the active-units panel. False -> 1x3 layout (SC-L² | SC-L^∞ | chi²).
# True  -> 2x2 layout (adds AU panel bottom-right).
SHOW_AU = False

# Substring filter on folder name (str) or None for no filter.
NAME_FILTER = None

# Default I/O. None for OUT_ROOT means "<project_root>/out".
# OUT_DIR=None means "write plot+csv into OUT_ROOT".
OUT_ROOT = None
OUT_DIR  = None

# ==========================================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
})


def parse_sid(sid):
    """Parse sID like 'fresh-v2L-3' into (init_tag, model_tag, run_id)."""
    if not isinstance(sid, str):
        return ("?", "?", "?")
    parts = sid.split("-")
    if len(parts) >= 3:
        return (parts[0], parts[1], "-".join(parts[2:]))
    return (sid, "?", "?")


def collect_runs(out_root, filter_str=None):
    """Walk out_root for finetune_*/summary.pt, return list of run dicts."""
    runs = []
    skipped_no_sc = 0
    skipped_load = 0
    if not os.path.isdir(out_root):
        raise FileNotFoundError(f"out-root not found: {out_root}")

    for name in sorted(os.listdir(out_root)):
        if not name.startswith("finetune_"):
            continue
        if filter_str is not None and filter_str not in name:
            continue
        summary_path = os.path.join(out_root, name, "summary.pt")
        if not os.path.exists(summary_path):
            continue
        try:
            s = torch.load(summary_path, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"  WARNING: failed to load {summary_path}: {e}")
            skipped_load += 1
            continue

        if "self_consistency" not in s:
            skipped_no_sc += 1
            continue

        init_tag, model_tag, run_id = parse_sid(s.get("sID", ""))
        sc_per_sample = s.get("self_consistency_per_sample")
        n_samples = (int(len(sc_per_sample))
                     if sc_per_sample is not None else None)

        runs.append({
            "folder":         name,
            "data_source":    s.get("data_source", "?"),
            "sim_name":       s.get("sim_name") or "synthetic",
            "init_tag":       init_tag,
            "model_tag":      model_tag,
            "run_id":         run_id,
            "num_poles":      int(s["num_poles"]) if "num_poles" in s else None,
            "n_epochs":       (int(s["n_epochs"]) if "n_epochs" in s else None),
            "sc_mean":        float(s["self_consistency"]),
            "sc_std":         float(s.get("self_consistency_std", float("nan"))),
            "sc_linf_mean":   (float(s["self_consistency_linfty"])
                               if "self_consistency_linfty" in s else None),
            "sc_linf_std":    float(s.get("self_consistency_linfty_std", float("nan"))),
            "n_samples":      n_samples,
            "best_chi2":      float(s.get("best_chi2",  float("nan"))),
            "final_chi2":     float(s.get("final_chi2", float("nan"))),
            "active_units":   (int(s["active_units"]) if "active_units" in s else None),
            "latent_dim":     (int(s["latent_dim"])   if "latent_dim"   in s else None),
        })

    print(f"Loaded {len(runs)} runs with SC "
          f"(skipped {skipped_no_sc} pre-SC runs, {skipped_load} load failures)")
    return runs


def write_csv(runs, csv_path):
    cols = ["folder", "data_source", "sim_name", "init_tag", "model_tag",
            "run_id", "num_poles", "n_epochs", "sc_mean", "sc_std",
            "sc_linf_mean", "sc_linf_std", "n_samples",
            "best_chi2", "final_chi2", "active_units", "latent_dim"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print(f"  Wrote: {csv_path}")


def _is_finite(v):
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    return True


def _draw_metric_panel(ax, ax_au, groups, value_key, std_key, cmap,
                       ylabel, draw_au, force_color=None, single_label=None):
    """Plot one metric variant on `ax` (and optionally AU on `ax_au`).

    Inputs are pre-filtered to one run per (group, num_poles) — the run with
    the best (lowest) SC-L² in that bucket — so each (group, num_poles) is a
    single marker; no within-rerun aggregation happens here.

    force_color / single_label: if set, all scatters use that color and the
    legend collapses to one entry (`single_label`) instead of one per group.
    """
    drew_any = False
    for i, (key, items) in enumerate(sorted(groups.items())):
        sim, init_tag, model_tag = key
        items_with = [d for d in items if _is_finite(d.get(value_key))]
        if not items_with:
            continue
        xs  = np.array([d["num_poles"] for d in items_with], dtype=float)
        ys  = np.array([d[value_key]   for d in items_with], dtype=float)
        ses = np.array([
            (d.get(std_key) / np.sqrt(d["n_samples"]))
            if (d["n_samples"] and _is_finite(d.get(std_key))) else np.nan
            for d in items_with
        ], dtype=float)

        if force_color is not None:
            color = force_color
            label = None  # single legend entry added below
        else:
            color = cmap(i % 10)
            sim_short = sim.replace("_", r"\_")
            label = rf"{sim_short} ({init_tag}, {model_tag})"

        ax.errorbar(xs, ys, yerr=ses, fmt="o", color=color, ms=6, lw=0,
                    elinewidth=1.0, capsize=2.5, alpha=0.9, label=label, zorder=3)
        drew_any = True

        if draw_au and ax_au is not None:
            au_pairs = [(d["num_poles"], d["active_units"])
                        for d in items_with if d.get("active_units") is not None]
            if au_pairs:
                xa = np.array([p[0] for p in au_pairs], dtype=float)
                ya = np.array([p[1] for p in au_pairs], dtype=float)
                # AU uses a single green everywhere to read as "AU" at a glance,
                # independent of the per-group color cycle on the SC axis.
                ax_au.plot(xa, ya, "s", mfc="none", mec="tab:green", ms=6,
                           mew=1.2, alpha=0.9, zorder=3)

    if force_color is not None and single_label is not None and drew_any:
        ax.plot([], [], "o", color=force_color, ms=6, label=single_label)

    ax.set_xlabel(r"Number of poles")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_sweep(runs, out_path, show_au=True):
    if not runs:
        print("No runs to plot.")
        return

    raw_groups = defaultdict(list)
    for r in runs:
        if r["num_poles"] is None:
            continue
        key = (r["sim_name"], r["init_tag"], r["model_tag"])
        raw_groups[key].append(r)

    # Per (group, num_poles) keep only the run with the best (lowest) SC-L².
    # Reruns at the same num_poles can be inconsistent and would otherwise
    # overshadow the trend across num_poles. The L^infty panel shows the
    # L^infty value of the same selected best-L² run.
    groups = defaultdict(list)
    for key, items in raw_groups.items():
        by_poles = defaultdict(list)
        for r in items:
            by_poles[r["num_poles"]].append(r)
        for poles, runs_at_p in by_poles.items():
            best = min(runs_at_p, key=lambda d: d["sc_mean"])
            groups[key].append(best)

    selected_runs = [r for items in groups.values() for r in items]
    has_au   = show_au and any(r.get("active_units") is not None for r in selected_runs)
    has_linf = any(r.get("sc_linf_mean") is not None for r in selected_runs)
    has_chi2 = any(_is_finite(r.get("best_chi2")) for r in selected_runs)

    cmap = plt.get_cmap("tab10")
    LINF_COLOR = "tab:red"
    CHI2_COLOR = "k"   # distinct from tab10 (used for L² groups) and from red (L^infty)
    AU_COLOR   = "tab:green"

    if has_linf:
        if has_au:
            # 2x2 layout (toggled on by --au): SC-L²  | SC-L^infty
            #                                  chi^2  | AU
            fig, axes = plt.subplots(2, 2, figsize=(11, 11), sharex=True)
            ax_l2, ax_li = axes[0, 0], axes[0, 1]
            ax_chi = axes[1, 0] if has_chi2 else None
            ax_au  = axes[1, 1]
            if not has_chi2:
                axes[1, 0].set_visible(False)
        else:
            # Default 1x3 layout: SC-L² | SC-L^infty | chi^2  (AU hidden)
            n_panels = 2 + (1 if has_chi2 else 0)
            fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5),
                                     sharex=True)
            if n_panels == 1:
                axes = [axes]
            ax_l2  = axes[0]
            ax_li  = axes[1]
            ax_chi = axes[2] if has_chi2 else None
            ax_au  = None

        _draw_metric_panel(
            ax_l2, None, groups, "sc_mean", "sc_std", cmap,
            ylabel=r"SC-$L^2$ $=\frac{1}{N}\sum_i\!\int |A_i - A(\!|\!\langle G\rangle)|^2\,d\omega$",
            draw_au=False,
        )
        _draw_metric_panel(
            ax_li, None, groups, "sc_linf_mean", "sc_linf_std", cmap,
            ylabel=r"SC-$L^\infty$ $=\frac{1}{N}\sum_i\, \max_\omega |A_i - A(\!|\!\langle G\rangle)|$",
            draw_au=False,
            force_color=LINF_COLOR,
            single_label=r"best-$L^2$ run per num\_poles",
        )
        ax_l2.set_title(r"$L^2$ self-consistency  (best-$L^2$ run per num\_poles)")
        ax_li.set_title(r"$L^\infty$ self-consistency  (same selected runs)")

        if ax_chi is not None:
            _draw_metric_panel(
                ax_chi, None, groups, "best_chi2", None, cmap,
                ylabel=r"$\chi^2$ (best across training)",
                draw_au=False,
                force_color=CHI2_COLOR,
                single_label=r"best-$L^2$ run per num\_poles",
            )
            ax_chi.axhline(1.0, color=CHI2_COLOR, ls="--", lw=1.0, alpha=0.5,
                           label=r"$\chi^2 = 1$ (ideal)")
            ax_chi.set_title(r"$\chi^2$ vs num\_poles  (same selected runs)")

        if ax_au is not None:
            # Plot active_units directly (one marker per group, all green) plus
            # a dashed line per group showing the latent_dim ceiling for context.
            for i, (key, items) in enumerate(sorted(groups.items())):
                items_with = [d for d in items
                              if d.get("active_units") is not None]
                if not items_with:
                    continue
                xs = np.array([d["num_poles"]   for d in items_with], dtype=float)
                ya = np.array([d["active_units"] for d in items_with], dtype=float)
                yd = np.array([d["latent_dim"]   for d in items_with], dtype=float)
                ax_au.plot(xs, ya, "s", mfc="none", mec=AU_COLOR, ms=8,
                           mew=1.5, alpha=0.95, zorder=3,
                           label=r"active units")
                ax_au.plot(xs, yd, "--", color="0.5", lw=1.0, alpha=0.6, zorder=2,
                           label=r"latent dim (ceiling)")
            ax_au.set_xlabel(r"Number of poles")
            ax_au.set_ylabel(r"Active units (Var$_x[\mu_i] > 10^{-2}$)")
            ax_au.set_title(r"Active units vs num\_poles  (same selected runs)")
            ax_au.set_ylim(bottom=0)
            ax_au.grid(True, alpha=0.3)
            # De-duplicate legend (line+marker repeat per group)
            h, l = ax_au.get_legend_handles_labels()
            seen = set()
            uniq = [(hi, li) for hi, li in zip(h, l)
                    if not (li in seen or seen.add(li))]
            if uniq:
                ax_au.legend(*zip(*uniq), loc="best", framealpha=0.9, fontsize=8)

        for ax in (ax_l2, ax_li):
            ax.legend(loc="best", framealpha=0.9, fontsize=8)
        if ax_chi is not None:
            ax_chi.legend(loc="best", framealpha=0.9, fontsize=8)

        fig.suptitle(r"Self-consistency vs number of poles", y=1.02)
    else:
        fig, ax_l2 = plt.subplots(figsize=(8, 5.5))
        ax_au_l2 = ax_l2.twinx() if has_au else None
        _draw_metric_panel(
            ax_l2, ax_au_l2, groups, "sc_mean", "sc_std", cmap,
            ylabel=r"SC-$L^2$ $=\frac{1}{N}\sum_i\!\int |A_i - A(\!|\!\langle G\rangle)|^2\,d\omega$",
            draw_au=True,
        )
        title = r"$L^2$ self-consistency (best-$L^2$ run per num\_poles)"
        if has_au:
            ax_au_l2.set_ylabel(r"Active units (Var$_x[\mu_i] > 10^{-2}$)")
            from matplotlib.lines import Line2D
            au_handle = Line2D([], [], color="tab:green", marker="s", mfc="none",
                               mec="tab:green", ls="", ms=6, label=r"active units")
            h, l = ax_l2.get_legend_handles_labels()
            ax_l2.legend(h + [au_handle], l + [au_handle.get_label()],
                         loc="best", framealpha=0.9)
            title += r"  (open $\Box$ = active units)"
        else:
            ax_l2.legend(loc="best", framealpha=0.9)
        ax_l2.set_title(title)

    if selected_runs:
        all_x = sorted({r["num_poles"] for r in selected_runs})
        for ax in fig.axes:
            ax.set_xticks(all_x)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_path}")
    print(f"  ({len(selected_runs)} run(s) shown; one per (group, num_poles), "
          f"selected by min SC-L²)")


def print_table(runs):
    if not runs:
        return
    runs_sorted = sorted(runs, key=lambda d: (
        d["sim_name"], d["init_tag"], d["model_tag"],
        d["num_poles"] if d["num_poles"] is not None else -1,
        d["run_id"],
    ))
    hdr = (f"\n{'sim':<35} {'init':<10} {'model':<6} {'run':>4} "
           f"{'poles':>5} {'SC-L2':>12} {'SC-Linf':>12} {'best_chi2':>10} {'AU':>9}")
    print(hdr)
    print("=" * len(hdr.lstrip("\n")))
    for r in runs_sorted:
        sc_linf = (f"{r['sc_linf_mean']:.6f}"
                   if r.get("sc_linf_mean") is not None else "        n/a")
        chi2    = (f"{r['best_chi2']:.4f}"
                   if not np.isnan(r['best_chi2']) else "     n/a")
        au_str  = (f"{r['active_units']}/{r['latent_dim']}"
                   if r.get("active_units") is not None else "      n/a")
        print(f"{r['sim_name'][:34]:<35} {r['init_tag']:<10} {r['model_tag']:<6} "
              f"{str(r['run_id']):>4} {r['num_poles']:>5} {r['sc_mean']:>12.6f} "
              f"{sc_linf:>12} {chi2:>10} {au_str:>9}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    main_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root_d = OUT_ROOT if OUT_ROOT is not None else os.path.join(main_path, "out")

    parser.add_argument("--out-root", default=out_root_d,
                        help=f"root containing finetune_*/ folders "
                             f"(default: {out_root_d})")
    parser.add_argument("--filter", default=NAME_FILTER,
                        help="only include folders whose name contains this "
                             f"substring (default: {NAME_FILTER!r})")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="where to write the plot+csv (default: --out-root)")
    parser.add_argument("--epochs", type=int, default=EPOCHS_FILTER,
                        help=f"keep only runs with this exact n_epochs; pass 0 to "
                             f"keep all (default: {EPOCHS_FILTER})")
    parser.add_argument("--au", action=argparse.BooleanOptionalAction,
                        default=SHOW_AU,
                        help=f"include the active-units panel and switch to 2x2 "
                             f"layout (use --no-au to force off; default: {SHOW_AU})")
    args = parser.parse_args()

    out_dir = args.out_dir or args.out_root
    os.makedirs(out_dir, exist_ok=True)

    runs = collect_runs(args.out_root, filter_str=args.filter)
    epochs_filter = args.epochs if (args.epochs is not None and args.epochs != 0) else None
    if epochs_filter is not None:
        before = len(runs)
        runs   = [r for r in runs if r["n_epochs"] == epochs_filter]
        print(f"Filtered to n_epochs={epochs_filter}: kept {len(runs)} of {before} runs")
    print_table(runs)

    write_csv(runs, os.path.join(out_dir, "selfconsistency_sweep.csv"))
    plot_sweep(runs, os.path.join(out_dir, "selfconsistency_sweep.pdf"),
               show_au=args.au)


if __name__ == "__main__":
    main()
