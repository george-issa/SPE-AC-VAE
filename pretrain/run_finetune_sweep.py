"""
Sweep pretrain/run_finetune.py over a list of NUM_POLES values, in parallel.

Reuses the configuration in run_finetune.py as-is; only NUM_POLES and
FINETUNE_EPOCHS are overridden via env vars (SWEEP_NUM_POLES, SWEEP_FINETUNE_EPOCHS).

All runs are launched concurrently as subprocesses, each with a thread cap
(OMP_NUM_THREADS, MKL_NUM_THREADS, etc.) so the processes share cores cleanly
instead of oversubscribing. Per-run stdout goes to out/sweep_logs/sweep_p{N}.log.

After the sweep completes, run pretrain/plot_selfconsistency_sweep.py to refresh
the SC / chi^2 sweep figure.

Usage:
    python pretrain/run_finetune_sweep.py
"""

import os
import subprocess
import sys
import time

POLES_SWEEP     = [2, 12, 14, 16, 18, 22, 24, 26, 28]
FINETUNE_EPOCHS = 200

# Force a fixed latent bottleneck (decoupled from num_poles) — None = default
# (= 4*num_poles - 2). Set to an int to override; the run folders gain a
# `_z{N}` suffix so they stay separate from default-latent runs.
LATENT_DIM_OVERRIDE = 2

# Particle-hole symmetric decoder: NUM_POLES is the *free* count (effective is
# 2*NUM_POLES). Run folders gain a `_ph` suffix so they stay separate from
# non-PH runs at the same NUM_POLES. Set to None to leave the choice to
# run_finetune.py's own PH_SYMMETRIC default.
PH_SYMMETRIC_OVERRIDE = None

# Per-process thread cap. M5 Pro: 18 physical cores (6 perf + 12 efficiency).
# 4 runs * 4 threads = 16 — fits comfortably; spare cores absorb anything else.
THREADS_PER_RUN = 4

# Run POLES_SWEEP in waves of BATCH_SIZE so we never exceed core count when
# len(POLES_SWEEP) > BATCH_SIZE. Set to len(POLES_SWEEP) for "all in parallel".
BATCH_SIZE = 4


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target    = os.path.join(proj_root, "pretrain", "run_finetune.py")
    log_dir   = os.path.join(proj_root, "out", "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    n       = len(POLES_SWEEP)
    batches = [POLES_SWEEP[i:i + BATCH_SIZE]
               for i in range(0, n, BATCH_SIZE)]
    t0      = time.time()

    print(f"Sweep: {n} runs in {len(batches)} batch(es) of up to {BATCH_SIZE} "
          f"({THREADS_PER_RUN} threads each).", flush=True)

    failures = []
    for bi, batch in enumerate(batches, 1):
        print(f"\n=== Batch {bi}/{len(batches)}: NUM_POLES={batch} ===",
              flush=True)
        procs = []
        for p in batch:
            log_path = os.path.join(log_dir, f"sweep_p{p:02d}.log")
            log_f    = open(log_path, "w")
            env = {**os.environ,
                   "SWEEP_NUM_POLES":       str(p),
                   "SWEEP_FINETUNE_EPOCHS": str(FINETUNE_EPOCHS),
                   "OMP_NUM_THREADS":       str(THREADS_PER_RUN),
                   "MKL_NUM_THREADS":       str(THREADS_PER_RUN),
                   "OPENBLAS_NUM_THREADS":  str(THREADS_PER_RUN),
                   "VECLIB_MAXIMUM_THREADS": str(THREADS_PER_RUN),
                   "NUMEXPR_NUM_THREADS":   str(THREADS_PER_RUN),
                   # Disable matplotlib display in subprocess (headless)
                   "MPLBACKEND":            "Agg"}
            if LATENT_DIM_OVERRIDE is not None:
                env["SWEEP_LATENT_DIM"] = str(LATENT_DIM_OVERRIDE)
            if PH_SYMMETRIC_OVERRIDE is not None:
                env["SWEEP_PH_SYMMETRIC"] = "1" if PH_SYMMETRIC_OVERRIDE else "0"
            proc = subprocess.Popen(
                [sys.executable, target],
                env=env, cwd=proj_root,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            procs.append((p, proc, log_f, log_path))
            print(f"  [launched] NUM_POLES={p:>3}  pid={proc.pid:<6}  log={log_path}",
                  flush=True)

        for p, proc, log_f, log_path in procs:
            rc = proc.wait()
            log_f.close()
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  [done @ {dt/60:5.1f} min] NUM_POLES={p:>3}  {status}",
                  flush=True)
            if rc != 0:
                failures.append(p)

    print(f"\nSweep complete: {n} runs in {(time.time()-t0)/60:.1f} min")
    if failures:
        print(f"  Failed: {failures}")


if __name__ == "__main__":
    main()
