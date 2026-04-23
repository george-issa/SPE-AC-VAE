"""
MaxEnt analytic continuation via ana_cont (Kaufmann & Held, CPC 2022).

Operates on imaginary-time Green's function data from the DQMC dataset.
Two modes are available:

  MODE = 'mean'       — fit the dataset-averaged G(tau) using SEM errors.
                        Standard MaxEnt workflow: one spectrum per dataset.

  MODE = 'per_sample' — fit each individual G(tau) bin independently.
                        Directly comparable to VAE per-sample output.
                        Can be slow for large datasets; use N_SAMPLES to cap.

Outputs (in OUT_DIR):
  summary_mean.npz        (mean mode) omega, A_opt, chi2, backtransform
  summary_samples.npz     (per_sample mode) omega, A_all [N, Nw], chi2_all
  params.json             run configuration

Usage:
    python MaxEnt_benchmark/run_anacont_maxent.py
"""

import os
import sys
import json
import numpy as np

# Add project root and ana_cont to path
PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANACONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ana_cont")
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ANACONT_DIR)

import ana_cont.continuation as cont  # type: ignore

# ============================================================================
# CONFIGURATION
# ============================================================================

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data source: "synthetic" or "real" ---
DATA_SOURCE = "real"

# --------------------------------------------------------------------------
# Real QMC data (DATA_SOURCE = "real")
# *** Only this line should change when uploading a new simulation folder ***
# --------------------------------------------------------------------------
QMC_SIM_DIR = os.path.join(
    MAIN_PATH, "Data", "datasets", "real",
    "hubbard_square_U8.00_mu0.00_L4_b6.00-1"
)

# --------------------------------------------------------------------------
# Synthetic data (DATA_SOURCE = "synthetic")
# --------------------------------------------------------------------------
SPECTRAL_TYPE = "gaussian_double"
INPUT_ID      = "inputs-8"
NOISE_S       = 1e-04
NOISE_XI      = 0.5
DATA_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", "synthetic",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets", "synthetic",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    "spectral_input.csv"
)

# --- Physical parameters — auto-read from toml when DATA_SOURCE="real" ---
BETA    = 10.0
DTAU    = 0.05
N_BINS  = None   # if None, inferred from data

# --- Real-frequency grid for A(omega) ---
N_OMEGA   = 1000
OMEGA_MIN = -20.0
OMEGA_MAX =  20.0

# --- MaxEnt solver settings ---
ALPHA_START = 1e12
ALPHA_END   = 1e-2
PREBLUR     = False
BLUR_WIDTH  = 0.5

# --- Run mode ---
MODE = 'mean'       # 'mean' or 'per_sample'
N_SAMPLES = 50      # max samples in per_sample mode (None = all)

# --- Covariance mode (mean mode only) ---
# True  — pass full (L_tau x L_tau) covariance matrix; more information but
#         can be numerically unstable when many eigenvalues are near zero.
# False — use diagonal SEM errors only; stabler and the standard MaxEnt approach.
USE_FULL_COV = True

# --- Output (set automatically from data source) ---
if DATA_SOURCE == "real":
    _out_tag = f"anacont_real-{os.path.basename(QMC_SIM_DIR)}"
else:
    _out_tag = f"anacont_{SPECTRAL_TYPE}_{INPUT_ID}_s{NOISE_S:.0e}"
OUT_DIR = os.path.join(MAIN_PATH, "MaxEnt_benchmark", "out", _out_tag)

# ============================================================================
# HELPERS
# ============================================================================

def load_data(csv_path):
    """Load G(tau) bins from CSV. Returns ndarray (N_bins, L_tau)."""
    G_bins = np.loadtxt(csv_path, delimiter=",")
    if G_bins.ndim == 1:
        G_bins = G_bins[np.newaxis, :]
    return G_bins


def compute_covariance(G_bins):
    """Sample covariance C from bins. Returns (L_tau, L_tau)."""
    N = G_bins.shape[0]
    G_mean = G_bins.mean(axis=0, keepdims=True)
    dG = G_bins - G_mean
    return (dG.T @ dG) / (N - 1)


def regularize_cov(C, rcond=1e-6):
    """Regularize a covariance matrix by flooring small eigenvalues.

    Eigenvalues below rcond * max_eigenvalue are raised to that floor.
    This is the covariance-space analogue of SVD truncation: it prevents
    near-zero eigenvalues from producing astronomically large weights
    (E = 1/var) in the MaxEnt chi2.

    Parameters
    ----------
    C     : (L, L) symmetric positive semi-definite covariance matrix
    rcond : relative condition threshold (default 1e-6)

    Returns
    -------
    C_reg : (L, L) regularized covariance matrix
    """
    evals, evecs = np.linalg.eigh(C)
    floor = evals.max() * rcond
    n_floored = int((evals < floor).sum())
    if n_floored > 0:
        print(f"  Regularizing covariance: flooring {n_floored}/{len(evals)} "
              f"eigenvalue(s) below {floor:.3e} (rcond={rcond:.0e})")
    evals_reg = np.maximum(evals, floor)
    return evecs @ np.diag(evals_reg) @ evecs.T


def run_maxent(tau, omega, G_tau, beta, err=None, cov=None, preblur=False, blur_width=0.5):
    """Run ana_cont MaxEnt for a single G(tau) curve.

    Parameters
    ----------
    tau   : (L,) imaginary time points
    omega : (Nw,) real frequency grid
    G_tau : (L,) imaginary-time Green's function (real-valued)
    beta  : float, inverse temperature
    err   : (L,) standard deviation at each tau point  [diagonal mode]
    cov   : (L, L) covariance matrix                   [full-cov mode]
    Exactly one of err or cov must be provided.

    Returns
    -------
    A_opt        : (Nw,) spectral function
    chi2         : float, chi-squared of the fit
    backtransform: (L,) back-transformed G(tau) from A_opt
    """
    if (err is None) == (cov is None):
        raise ValueError("Provide exactly one of err (diagonal) or cov (full matrix).")

    # Flat default model normalized to sum rule ∫A(ω)dω = 1
    model = np.ones_like(omega)
    model /= np.trapezoid(model, omega)

    probl = cont.AnalyticContinuationProblem(
        im_axis=tau,
        re_axis=omega,
        im_data=G_tau,
        kernel_mode='time_fermionic',
        beta=beta,
    )

    solve_kwargs = dict(
        method='maxent_svd',
        alpha_determination='chi2kink',
        optimizer='newton',
        model=model,
        interactive=False,
        alpha_start=ALPHA_START,
        alpha_end=ALPHA_END,
        preblur=preblur,
        blur_width=blur_width,
    )
    if cov is not None:
        solve_kwargs['cov'] = cov
    else:
        solve_kwargs['stdev'] = err

    sol, _ = probl.solve(**solve_kwargs)

    A_opt         = sol.A_opt
    chi2          = float(sol.chi2)
    backtransform = sol.backtransform

    return A_opt, chi2, backtransform


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if DATA_SOURCE == "real":
        from data_process_real import extract_greens_bins_v2, read_model_params  # type: ignore
        print(f"Loading real QMC data from: {QMC_SIM_DIR}")
        G_bins, _params = extract_greens_bins_v2(QMC_SIM_DIR, r1=0, r2=0)
        beta  = _params["beta"]
        dtau  = _params["dtau"]
        L_tau = _params["L_tau"]
        spectral_input_path = None
        print(f"  beta={beta}, dtau={dtau}, L_tau={L_tau}")
    else:
        print(f"Loading synthetic data from: {DATA_PATH}")
        G_bins = load_data(DATA_PATH)
        beta   = BETA
        dtau   = DTAU
        L_tau  = G_bins.shape[1]
        spectral_input_path = SPECTRAL_INPUT_PATH

    N_bins = G_bins.shape[0]
    print(f"  {N_bins} bins x {L_tau} tau points")

    tau   = np.linspace(0.0, beta - dtau, L_tau)
    omega = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)

    # ------------------------------------------------------------------
    # Covariance and errors
    # ------------------------------------------------------------------
    print("Computing covariance...")
    C = compute_covariance(G_bins)
    var_diag = np.diag(C)           # variance of each tau bin
    sem_diag = np.sqrt(var_diag / N_bins)   # standard error of the mean

    # ------------------------------------------------------------------
    # Save run params
    # ------------------------------------------------------------------
    params = {
        "DATA_SOURCE":          DATA_SOURCE,
        "DATA_PATH":            QMC_SIM_DIR if DATA_SOURCE == "real" else DATA_PATH,
        "SPECTRAL_INPUT_PATH":  spectral_input_path,
        "SPECTRAL_TYPE":        SPECTRAL_TYPE,
        "INPUT_ID":             INPUT_ID,
        "NOISE_S":              NOISE_S,
        "BETA":                 beta,
        "DTAU":                 dtau,
        "N_BINS":               N_bins,
        "L_TAU":                L_tau,
        "N_OMEGA":              N_OMEGA,
        "OMEGA_MIN":            OMEGA_MIN,
        "OMEGA_MAX":            OMEGA_MAX,
        "ALPHA_START":          ALPHA_START,
        "ALPHA_END":            ALPHA_END,
        "PREBLUR":              PREBLUR,
        "BLUR_WIDTH":           BLUR_WIDTH,
        "MODE":                 MODE,
        "N_SAMPLES":            N_SAMPLES,
    }
    with open(os.path.join(OUT_DIR, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # ------------------------------------------------------------------
    # Mode: MEAN — MaxEnt on dataset-averaged G(tau) with SEM errors
    # ------------------------------------------------------------------
    if MODE == 'mean':
        print("\n--- Mode: MEAN ---")
        G_mean    = G_bins.mean(axis=0)
        cov_mean  = C / N_bins              # covariance of the mean: C_ij / N
        cov_mean  = regularize_cov(cov_mean)  # floor near-zero eigenvalues

        print(f"G_mean(tau=0) = {G_mean[0]:.6f}")
        print(f"Mean SEM (diag) = {sem_diag.mean():.3e}, Max SEM = {sem_diag.max():.3e}")
        print(f"Passing full ({L_tau}x{L_tau}) covariance matrix to MaxEnt...")

        print("Running MaxEnt (this may take ~30 s)...")
        A_opt, chi2, backtransform = run_maxent(
            tau, omega, G_mean, beta, cov=cov_mean,
            preblur=PREBLUR, blur_width=BLUR_WIDTH,
        )

        # Normalise A (should already integrate to ~1 by sum rule enforcement)
        norm = np.trapezoid(A_opt, omega)
        print(f"  chi2 = {chi2:.4f},  ∫A dω = {norm:.4f}")

        np.savez(
            os.path.join(OUT_DIR, "summary_mean_fullcov.npz"),
            omega=omega,
            A_opt=A_opt,
            chi2=chi2,
            backtransform=backtransform,
            G_mean=G_mean,
            tau=tau,
            beta=beta,
            sem_err=sem_diag,
            cov_mean=cov_mean,
            spectral_input_path=spectral_input_path if spectral_input_path else "",
        )
        print(f"  Saved: {OUT_DIR}/summary_mean_fullcov.npz")

    # ------------------------------------------------------------------
    # Mode: PER_SAMPLE — MaxEnt on each individual G(tau) bin
    # ------------------------------------------------------------------
    elif MODE == 'per_sample':
        print("\n--- Mode: PER_SAMPLE ---")
        n_run = N_bins if N_SAMPLES is None else min(N_SAMPLES, N_bins)
        err   = np.sqrt(var_diag)  # std dev of the bins (not SEM)

        print(f"Processing {n_run}/{N_bins} samples...")
        A_all    = np.zeros((n_run, N_OMEGA))
        chi2_all = np.zeros(n_run)
        bt_all   = np.zeros((n_run, L_tau))

        for i in range(n_run):
            if i % 10 == 0:
                print(f"  sample {i+1}/{n_run} ...", flush=True)
            A_opt, chi2, bt = run_maxent(
                tau, omega, G_bins[i], beta, err=err,
                preblur=PREBLUR, blur_width=BLUR_WIDTH,
            )
            A_all[i]    = A_opt
            chi2_all[i] = chi2
            bt_all[i]   = bt

        print(f"\n  chi2: mean={chi2_all.mean():.4f}, min={chi2_all.min():.4f}, max={chi2_all.max():.4f}")
        norm_all = np.trapezoid(A_all, omega, axis=1)
        print(f"  ∫A dω: mean={norm_all.mean():.4f}")

        np.savez(
            os.path.join(OUT_DIR, "summary_samples.npz"),
            omega=omega,
            A_all=A_all,
            A_mean=A_all.mean(axis=0),
            chi2_all=chi2_all,
            backtransform_all=bt_all,
            G_bins=G_bins[:n_run],
            tau=tau,
            beta=beta,
            bin_err=err,
            spectral_input_path=spectral_input_path if spectral_input_path else "",
        )
        print(f"  Saved: {OUT_DIR}/summary_samples.npz")

    else:
        raise ValueError(f"Unknown MODE={MODE!r}. Use 'mean' or 'per_sample'.")

    print(f"\nDone. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
