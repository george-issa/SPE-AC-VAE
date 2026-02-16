"""
CLI script to generate synthetic single-Gaussian pretraining data.

Usage:
    python pretrain/generate_data.py --N 10000 --sigma_mu 1.0 --mu_sigma 3.0 --sigma_sigma 0.5

    # With DQMC covariance:
    python pretrain/generate_data.py --N 10000 --covariance_source Data/datasets/half-filled-gaussian-double/inputs-1/Gbins_s1e-05_xi0.5.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pretrain.synthetic_data import generate_and_save  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Gaussian pretraining data")

    parser.add_argument("--N", type=int, default=10000, help="Number of samples")
    parser.add_argument("--sigma_mu", type=float, default=0.5, help="Std of Normal for mu")
    parser.add_argument("--mu_sigma", type=float, default=3.0, help="Mean of InverseGamma for sigma (approx W/2)")
    parser.add_argument("--sigma_sigma", type=float, default=0.5, help="Std of InverseGamma for sigma")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--dtau", type=float, default=0.05, help="Imaginary time step")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--covariance_source", type=str, default=None,
                        help="Path to DQMC Gbins CSV for covariance (optional)")
    parser.add_argument("--fast", action="store_true",
                        help="Use GL quadrature instead of adaptive (faster, slightly less precise)")
    parser.add_argument("--N_gl", type=int, default=512, help="GL nodes per half-axis (for --fast)")
    parser.add_argument("--omega_max", type=float, default=20.0, help="Integration cutoff (for --fast)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.output_dir is None:
        main_path = os.path.join(os.path.dirname(__file__), "..")
        args.output_dir = os.path.join(main_path, "Data", "datasets", "synthetic-gaussian-pretrain")
        
    if args.covariance_source is None:
        args.covariance_source = os.path.join(main_path, "Data", "datasets", "half-filled-gaussian_double", "inputs-7", "Gbins_s1e-05_xi0.5.csv")

    generate_and_save(
        N=args.N,
        sigma_mu=args.sigma_mu,
        mu_sigma=args.mu_sigma,
        sigma_sigma=args.sigma_sigma,
        beta=args.beta,
        dtau=args.dtau,
        output_dir=args.output_dir,
        covariance_source=args.covariance_source,
        use_quadrature=not args.fast,
        N_gl=args.N_gl,
        omega_max=args.omega_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
