"""
Plot results from a completed fine-tuning run.

Usage
-----
    python plot_run.py <out_dir>

    # Example:
    python plot_run.py out/finetune_real-bond_holstein_..._numpoles6-fresh-v2L-real-1

If no argument is given, OUT_DIR below is used instead.
Plots are saved to <out_dir>/plots/.
"""

import os
import sys

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Fallback when no command-line argument is given.
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "out",
    "finetune_real-bond_holstein_square_w1.00_a0.8165_b2.00_L12-1_numpoles8-fresh-v2L-1"
)
# ─────────────────────────────────────────────────────────────────────────────

# Allow passing the output directory as a positional argument
if len(sys.argv) > 1:
    OUT_DIR = sys.argv[1]

# Resolve paths
SUMMARY_FILE = os.path.join(OUT_DIR, "summary.pt")
LOSS_DIR     = os.path.join(OUT_DIR, "losses")
PLOTS_DIR    = os.path.join(OUT_DIR, "plots")

# Validate
if not os.path.isdir(OUT_DIR):
    print(f"ERROR: output directory not found:\n  {OUT_DIR}")
    sys.exit(1)
if not os.path.exists(SUMMARY_FILE):
    print(f"ERROR: summary.pt not found in {OUT_DIR}")
    sys.exit(1)
if not os.path.isdir(LOSS_DIR):
    print(f"ERROR: losses/ directory not found in {OUT_DIR}")
    sys.exit(1)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Imports (after path check so errors are clear) ───────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from pretrain.plot_results import plot_finetune_eval, plot_loss_curves  # type: ignore

# ── Plot ─────────────────────────────────────────────────────────────────────
print(f"Plotting run: {OUT_DIR}")
print()

print("1/2  G(tau), A(omega), poles/residues ...")
plot_finetune_eval(SUMMARY_FILE, output_dir=PLOTS_DIR)

print("2/2  Loss curves ...")
plot_loss_curves(
    LOSS_DIR,
    tag="finetune",
    save_path=os.path.join(PLOTS_DIR, "loss_curves_finetune.pdf"),
)

print(f"\nDone. All plots saved to:\n  {PLOTS_DIR}/")
