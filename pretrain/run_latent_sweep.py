"""
Sweep pretrain/run_finetune.py over a list of LATENT_DIM values, keeping
NUM_POLES fixed. Mirrors run_finetune_sweep.py but for the latent-bottleneck
characterization on site-Holstein at LW + P = 10.

Usage:
    python pretrain/run_latent_sweep.py
"""

import os
import subprocess
import sys
import time

LATENT_SWEEP    = [2, 3, 4, 5]
NUM_POLES       = 10
FINETUNE_EPOCHS = 1000

# Per-process thread cap. M5 Pro: 18 physical cores. 4 runs * 4 threads = 16.
THREADS_PER_RUN = 4
BATCH_SIZE      = 4   # waves of up to this many runs in parallel


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target    = os.path.join(proj_root, "pretrain", "run_finetune.py")
    log_dir   = os.path.join(proj_root, "out", "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    n       = len(LATENT_SWEEP)
    batches = [LATENT_SWEEP[i:i + BATCH_SIZE]
               for i in range(0, n, BATCH_SIZE)]
    t0      = time.time()

    print(f"Latent sweep: {n} runs in {len(batches)} batch(es) of up to "
          f"{BATCH_SIZE} ({THREADS_PER_RUN} threads each). "
          f"NUM_POLES={NUM_POLES}, FINETUNE_EPOCHS={FINETUNE_EPOCHS}.", flush=True)

    failures = []
    for bi, batch in enumerate(batches, 1):
        print(f"\n=== Batch {bi}/{len(batches)}: LATENT_DIM={batch} ===",
              flush=True)
        procs = []
        for L in batch:
            log_path = os.path.join(
                log_dir,
                f"latentsweep_z{L:02d}_lw_p{NUM_POLES:02d}.log",
            )
            log_f    = open(log_path, "w")
            env = {**os.environ,
                   "SWEEP_NUM_POLES":       str(NUM_POLES),
                   "SWEEP_LATENT_DIM":      str(L),
                   "SWEEP_FINETUNE_EPOCHS": str(FINETUNE_EPOCHS),
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
            procs.append((L, proc, log_f, log_path))
            print(f"  [launched] LATENT_DIM={L:>2}  pid={proc.pid:<6}  "
                  f"log={log_path}", flush=True)

        for L, proc, log_f, log_path in procs:
            rc = proc.wait()
            log_f.close()
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  [done @ {dt/60:5.1f} min] LATENT_DIM={L:>2}  {status}",
                  flush=True)
            if rc != 0:
                failures.append(L)

    print(f"\nSweep complete: {n} runs in {(time.time()-t0)/60:.1f} min")
    if failures:
        print(f"  Failed: {failures}")


if __name__ == "__main__":
    main()
