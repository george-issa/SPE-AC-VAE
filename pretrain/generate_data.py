"""
CLI / editor script to generate synthetic single-Gaussian pretraining data.

Two ways to run
---------------
1. Edit the CONFIG section below and run with no arguments:
       python pretrain/generate_data.py

2. Override any knob from the command line (argparse defaults to CONFIG values):
       python pretrain/generate_data.py --N 5000 --mu_max 2.5 --seed 0
       python pretrain/generate_data.py --N 20000 --covariance_source path/to/Gbins.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pretrain.synthetic_data import generate_and_save  # type: ignore

# ===========================================================================
# CONFIG — edit these values to run without command-line arguments
# ===========================================================================

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset size
N = 1000

# Spectral parameter ranges
# mu    ~ Uniform(-MU_MAX, +MU_MAX)        : peak position
# sigma ~ LogUniform(SIGMA_MIN, SIGMA_MAX) : peak width
MU_MAX    = 0.5   # |mu| ≤ MU_MAX; beyond ~3 G(tau) is near-zero for beta=10
SIGMA_MIN = 3.0   # narrowest reliably-resolved peak
SIGMA_MAX = 6.0   # widest peak; keeps spectral weight inside integration domain

# Physical grid
BETA      = 10.0
DTAU      = 0.05

# Output directory
OUTPUT_DIR = os.path.join(
    MAIN_PATH, "Data", "datasets", "synthetic", "synthetic-gaussian-pretrain"
)

# Covariance source: path to a DQMC Gbins CSV so chi-squared uses real noise structure.
# Set to None to fall back to the signal covariance (last resort only).
COVARIANCE_SOURCE = os.path.join(
    MAIN_PATH, "Data", "datasets", "synthetic",
    "half-filled-gaussian_double", "inputs-8",
    "Gbins_s1e-04_xi0.5.csv"
)

# Integration method: False = GL quadrature (fast, recommended); True = adaptive (slow)
USE_QUADRATURE = False
N_GL      = 256
OMEGA_MAX = 20.0

SEED = 42

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Gaussian pretraining data. "
                    "Edit the CONFIG section at the top of this file to set defaults."
    )

    parser.add_argument("--N",          type=int,   default=N,
                        help=f"Number of samples (default: {N})")
    parser.add_argument("--mu_max",     type=float, default=MU_MAX,
                        help=f"Peak centers ~ Uniform(-mu_max, +mu_max) (default: {MU_MAX})")
    parser.add_argument("--sigma_min",  type=float, default=SIGMA_MIN,
                        help=f"Min peak width — LogUniform lower bound (default: {SIGMA_MIN})")
    parser.add_argument("--sigma_max",  type=float, default=SIGMA_MAX,
                        help=f"Max peak width — LogUniform upper bound (default: {SIGMA_MAX})")
    parser.add_argument("--beta",       type=float, default=BETA,
                        help=f"Inverse temperature (default: {BETA})")
    parser.add_argument("--dtau",       type=float, default=DTAU,
                        help=f"Imaginary time step (default: {DTAU})")
    parser.add_argument("--output_dir", type=str,   default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--covariance_source", type=str, default=COVARIANCE_SOURCE,
                        help="Path to DQMC Gbins CSV for covariance (default: CONFIG value)")
    parser.add_argument("--adaptive",   action="store_true", default=USE_QUADRATURE,
                        help="Use adaptive quadrature instead of GL (slower, more precise)")
    parser.add_argument("--N_gl",       type=int,   default=N_GL,
                        help=f"GL nodes per half-axis (default: {N_GL})")
    parser.add_argument("--omega_max",  type=float, default=OMEGA_MAX,
                        help=f"Integration cutoff (default: {OMEGA_MAX})")
    parser.add_argument("--seed",       type=int,   default=SEED,
                        help=f"Random seed (default: {SEED})")

    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Gaussian data generation")
    print("=" * 60)
    print(f"  N             = {args.N}")
    print(f"  mu_max        = {args.mu_max}  (Uniform)")
    print(f"  sigma_min     = {args.sigma_min}  (LogUniform)")
    print(f"  sigma_max     = {args.sigma_max}  (LogUniform)")
    print(f"  beta          = {args.beta},  dtau = {args.dtau}")
    print(f"  output_dir    = {args.output_dir}")
    print(f"  cov_source    = {args.covariance_source}")
    print(f"  quadrature    = {'adaptive' if args.adaptive else 'GL'}")
    print(f"  seed          = {args.seed}")
    print()

    generate_and_save(
        N=args.N,
        mu_max=args.mu_max,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        beta=args.beta,
        dtau=args.dtau,
        output_dir=args.output_dir,
        covariance_source=args.covariance_source,
        use_quadrature=args.adaptive,
        N_gl=args.N_gl,
        omega_max=args.omega_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
