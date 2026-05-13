"""
Multi-seed 2-cell VAE on site-Holstein (per-cell raw covariance).

5 fixed seeds at the same config as the headline multi-cell run
(`fresh-v2L-1`). Tests whether the encoder-engagement and the specific
cluster geometry are seed-stable.

Parallel launch is safe now that `run_finetune_multicell.py` claims its
output dir atomically (mkdir + exist_ok=False retry loop).

Usage:
    python pretrain/run_seed_sweep_multicell.py
"""

import os
import subprocess
import sys
import time

SEEDS           = [42, 43, 44, 45, 46]
NUM_POLES       = 10
LATENT_DIM      = 2
BATCH_SIZE      = 10
FINETUNE_EPOCHS = 1000

# M5 Pro has 18 cores. 5 workers * 3 threads = 15, leaves headroom for OS.
THREADS_PER_RUN = 3


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target    = os.path.join(proj_root, "pretrain", "run_finetune_multicell.py")
    log_dir   = os.path.join(proj_root, "out", "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    n = len(SEEDS)
    t0 = time.time()
    print(f"Multi-cell seed sweep (parallel): {n} runs, "
          f"{THREADS_PER_RUN} threads each. "
          f"NUM_POLES={NUM_POLES}, LATENT_DIM={LATENT_DIM}, "
          f"BATCH_SIZE={BATCH_SIZE}, EPOCHS={FINETUNE_EPOCHS}", flush=True)

    procs = []
    for seed in SEEDS:
        log_path = os.path.join(log_dir, f"multicell_2cell_s{seed:02d}.log")
        log_f    = open(log_path, "w")
        env = {**os.environ,
               "SWEEP_NUM_POLES":       str(NUM_POLES),
               "SWEEP_LATENT_DIM":      str(LATENT_DIM),
               "SWEEP_BATCH_SIZE":      str(BATCH_SIZE),
               "SWEEP_FINETUNE_EPOCHS": str(FINETUNE_EPOCHS),
               "SWEEP_SEED":            str(seed),
               # Per-cell LW: pass through whatever is set on the parent
               # shell. Set SWEEP_COV_LW_PER_CELL=1 in the calling
               # environment to switch the sweep to per-cell LW shrinkage.
               "OMP_NUM_THREADS":       str(THREADS_PER_RUN),
               "MKL_NUM_THREADS":       str(THREADS_PER_RUN),
               "OPENBLAS_NUM_THREADS":  str(THREADS_PER_RUN),
               "VECLIB_MAXIMUM_THREADS": str(THREADS_PER_RUN),
               "NUMEXPR_NUM_THREADS":   str(THREADS_PER_RUN),
               "MPLBACKEND":            "Agg"}
        proc = subprocess.Popen(
            [sys.executable, target],
            env=env, cwd=proj_root,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs.append((seed, proc, log_f, log_path))
        print(f"  [launched] seed={seed:>2}  pid={proc.pid:<6}  log={log_path}",
              flush=True)

    failures = []
    for seed, proc, log_f, log_path in procs:
        rc = proc.wait()
        log_f.close()
        dt = (time.time() - t0) / 60
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  [done @ {dt:5.1f} min] seed={seed:>2}  {status}",
              flush=True)
        if rc != 0:
            failures.append(seed)

    print(f"\nSweep complete: {n} runs in {(time.time()-t0)/60:.1f} min")
    if failures:
        print(f"  Failed: {failures}")


if __name__ == "__main__":
    main()
