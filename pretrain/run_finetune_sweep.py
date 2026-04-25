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

POLES_SWEEP     = [5, 10, 15, 20, 25, 30]
FINETUNE_EPOCHS = 200

# Per-process thread cap. M5 Pro: 18 physical cores (6 perf + 12 efficiency).
# 6 runs * 3 threads = 18 — one thread per physical core, no oversubscription.
THREADS_PER_RUN = 3


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target    = os.path.join(proj_root, "pretrain", "run_finetune.py")
    log_dir   = os.path.join(proj_root, "out", "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    n  = len(POLES_SWEEP)
    t0 = time.time()

    procs = []
    for p in POLES_SWEEP:
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
        proc = subprocess.Popen(
            [sys.executable, target],
            env=env, cwd=proj_root,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs.append((p, proc, log_f, log_path))
        print(f"[launched] NUM_POLES={p:>3}  pid={proc.pid:<6}  log={log_path}",
              flush=True)

    print(f"\n{n} runs launched in parallel ({THREADS_PER_RUN} threads each, "
          f"{n*THREADS_PER_RUN} total). Waiting...\n", flush=True)

    failures = []
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
