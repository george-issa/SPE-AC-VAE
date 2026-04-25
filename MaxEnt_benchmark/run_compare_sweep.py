"""
Run MaxEnt_benchmark/compare_vae_maxent.py for every VAE finetune run that
matches a num_poles set + n_epochs filter. Writes one comparison_*/ folder per
VAE run, all sharing the same MaxEnt reference (MaxEnt depends only on G(tau),
not on the VAE pole count).

Each call overrides VAE_SUMMARY in compare_vae_maxent.py via the
SWEEP_VAE_SUMMARY env var.

Usage:
    python MaxEnt_benchmark/run_compare_sweep.py
"""

import os
import subprocess
import sys
import time

import torch

# ==========================================================================
# CONFIG — toggles at the top, edit and re-run
# ==========================================================================

POLES_SWEEP   = [5, 10, 15, 20, 25, 30]
EPOCHS_FILTER = 200          # only include VAE runs with this n_epochs
SIM_PREFIX    = "finetune_real-hubbard_square_U8"  # match this finetune-folder prefix

# Run sweep in parallel? compare_vae_maxent.py is light (few-second LaTeX renders),
# so parallelism is mostly a UX win — they finish nearly together.
PARALLEL = True

# ==========================================================================


def discover_vae_summaries(out_root):
    hits = []
    for d in sorted(os.listdir(out_root)):
        if not d.startswith(SIM_PREFIX):
            continue
        p = os.path.join(out_root, d, "summary.pt")
        if not os.path.exists(p):
            continue
        try:
            s = torch.load(p, weights_only=False, map_location="cpu")
        except Exception:
            continue
        if int(s.get("n_epochs", -1)) != EPOCHS_FILTER:
            continue
        np_ = int(s.get("num_poles", -1))
        if np_ in POLES_SWEEP:
            hits.append((np_, p))
    # One per num_poles (best by sc_mean if duplicates exist)
    by_p = {}
    for np_, p in hits:
        s = torch.load(p, weights_only=False, map_location="cpu")
        sc = float(s.get("self_consistency", float("inf")))
        cur = by_p.get(np_)
        if cur is None or sc < cur[1]:
            by_p[np_] = (p, sc)
    return [(np_, by_p[np_][0]) for np_ in sorted(by_p)]


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target    = os.path.join(proj_root, "MaxEnt_benchmark", "compare_vae_maxent.py")
    log_dir   = os.path.join(proj_root, "MaxEnt_benchmark", "out", "compare_sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    runs = discover_vae_summaries(os.path.join(proj_root, "out"))
    if not runs:
        print(f"No VAE summaries matched (prefix={SIM_PREFIX!r}, "
              f"n_epochs={EPOCHS_FILTER}, poles={POLES_SWEEP}).")
        return

    print(f"Comparison sweep: {len(runs)} VAE run(s)")
    for np_, p in runs:
        print(f"  num_poles={np_:>3}  →  {p}")
    print()

    t0 = time.time()
    failures = []

    if PARALLEL:
        procs = []
        for np_, vae_p in runs:
            log_path = os.path.join(log_dir, f"compare_p{np_:02d}.log")
            log_f    = open(log_path, "w")
            env      = {**os.environ,
                        "SWEEP_VAE_SUMMARY": vae_p,
                        "MPLBACKEND":        "Agg"}
            proc = subprocess.Popen(
                [sys.executable, target],
                env=env, cwd=proj_root,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            procs.append((np_, proc, log_f, log_path))
            print(f"[launched] poles={np_:>3}  pid={proc.pid:<6}  log={log_path}",
                  flush=True)
        print(f"\n{len(procs)} comparisons launched in parallel. Waiting...\n",
              flush=True)
        for np_, proc, log_f, log_path in procs:
            rc = proc.wait()
            log_f.close()
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  poles={np_:>3}  {status}  ({log_path})", flush=True)
            if rc != 0:
                failures.append(np_)
    else:
        for np_, vae_p in runs:
            print(f"\n[poles={np_}]", flush=True)
            env = {**os.environ, "SWEEP_VAE_SUMMARY": vae_p}
            rc = subprocess.run([sys.executable, target], env=env, cwd=proj_root).returncode
            if rc != 0:
                failures.append(np_)

    print(f"\nSweep complete: {len(runs)} comparisons in {(time.time()-t0):.1f} s")
    if failures:
        print(f"  Failed: {failures}")


if __name__ == "__main__":
    main()
