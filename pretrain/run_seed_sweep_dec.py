"""
Multi-seed DecoderOnly campaign on site-Holstein (LW + P=10, batch=5).

5 fixed seeds (42..46) at the exact configuration that produced vDEC-1
(the run that posted L^2_ME = 0.050). Tells us whether that result was a
basin-luck outlier or a representative member of the seed distribution.

Usage:
    python pretrain/run_seed_sweep_dec.py
"""

import os
import subprocess
import sys
import time

SEEDS           = [42, 43, 44, 45, 46]
NUM_POLES       = 10
BATCH_SIZE      = 5
FINETUNE_EPOCHS = 1000

THREADS_PER_RUN = 4
BATCH_PARALLEL  = 4   # 4 then 1 (M5 Pro: 18 cores, 4*4=16 < 18)


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target    = os.path.join(proj_root, "pretrain", "run_finetune_decoder_only.py")
    log_dir   = os.path.join(proj_root, "out", "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    n       = len(SEEDS)
    batches = [SEEDS[i:i + BATCH_PARALLEL]
               for i in range(0, n, BATCH_PARALLEL)]
    t0      = time.time()

    print(f"DEC seed sweep: {n} runs in {len(batches)} batch(es) of up to "
          f"{BATCH_PARALLEL} ({THREADS_PER_RUN} threads each). "
          f"NUM_POLES={NUM_POLES}, BATCH_SIZE={BATCH_SIZE}, "
          f"FINETUNE_EPOCHS={FINETUNE_EPOCHS}.", flush=True)

    failures = []
    for bi, batch in enumerate(batches, 1):
        print(f"\n=== Batch {bi}/{len(batches)}: SEEDS={batch} ===", flush=True)
        procs = []
        for seed in batch:
            log_path = os.path.join(log_dir, f"decseed_s{seed:02d}_lw_p{NUM_POLES:02d}.log")
            log_f    = open(log_path, "w")
            env = {**os.environ,
                   "SWEEP_NUM_POLES":       str(NUM_POLES),
                   "SWEEP_BATCH_SIZE":      str(BATCH_SIZE),
                   "SWEEP_FINETUNE_EPOCHS": str(FINETUNE_EPOCHS),
                   "SWEEP_SEED":            str(seed),
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
            print(f"  [launched] SEED={seed:>2}  pid={proc.pid:<6}  log={log_path}",
                  flush=True)

        for seed, proc, log_f, log_path in procs:
            rc = proc.wait()
            log_f.close()
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  [done @ {dt/60:5.1f} min] SEED={seed:>2}  {status}",
                  flush=True)
            if rc != 0:
                failures.append(seed)

    print(f"\nSweep complete: {n} runs in {(time.time()-t0)/60:.1f} min")
    if failures:
        print(f"  Failed: {failures}")


if __name__ == "__main__":
    main()
