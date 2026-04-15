"""
MaxEnt analytic continuation via OmegaMaxEnt (Bergeron & Tremblay, PRE 2016).

Uses the Matsubara-frequency path (standard 3-column format):
  column 1: ωn = (2n+1)π/β  (fermionic Matsubara frequencies)
  column 2: Re[G(iωn)]       (standard sign convention: G_std = -G_ours)
  column 3: Im[G(iωn)]       (Im < 0 for standard fermionic Green function)

G_ours(τ) > 0 (project convention) → G_std = -G_ours.
DFT: G_std(iωn) = -dtau * Σ_j G_ours(τ_j) exp(iωn τ_j)

Errors in Matsubara space are estimated bin-by-bin from the DFT of
each G_bins sample, then taking the std across samples.

Prerequisites:
    Build OmegaMaxEnt (with ARMA_DONT_USE_WRAPPER, no -ffast-math):
        cd MaxEnt_benchmark/OmegaMaxEnt
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DDOWNLOAD_ARMADILLO=1 -Wno-dev
        make -j4

Outputs (in OUT_DIR/):
    OmegaMaxEnt_run/   working directory with all input/output files
    summary.npz        omega, A_opt for downstream compare_vae_maxent.py
    params.json        run configuration

Usage:
    python MaxEnt_benchmark/run_omegamaxent.py
"""

import os
import sys
import json
import subprocess
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

# ============================================================================
# CONFIGURATION
# ============================================================================

MAIN_PATH = "/Users/georgeissa/Documents/AC/SPE-AC-VAE"

OMEGAMAXENT_BIN = os.path.join(
    MAIN_PATH, "MaxEnt_benchmark", "OmegaMaxEnt", "build", "OmegaMaxEnt"
)

# --- Dataset ---
SPECTRAL_TYPE = "gaussian_double"
INPUT_ID      = "inputs-8"
NOISE_S       = 1e-04
NOISE_XI      = 0.5
DATA_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    "spectral_input.csv"
)

# --- Physical parameters ---
BETA = 10.0
DTAU = 0.05      # dtau of the original data

# --- Matsubara frequency settings ---
# Include Matsubara frequencies up to this cutoff (energy units).
# DFT is reliable for ωn ≪ π/dtau = 62.8; use 15 for good conditioning.
MATSUBARA_CUTOFF = 50.0

# --- Spectral grid hints for OmegaMaxEnt ---
SPECTRAL_WIDTH  = 5.0   # half-width of main spectral range (symmetric)
SPECTRAL_CENTER = 0.0   # center of spectral function (half-filled → 0)
FREQ_STEP       = 0.1   # step size at frequency grid origin

# --- Output ---
OUT_DIR = os.path.join(MAIN_PATH, "MaxEnt_benchmark", "out",
                       f"omegamaxent_{SPECTRAL_TYPE}_{INPUT_ID}_s{NOISE_S:.0e}")

# ============================================================================
# HELPERS
# ============================================================================

def load_data(csv_path):
    G_bins = np.loadtxt(csv_path, delimiter=",")
    if G_bins.ndim == 1:
        G_bins = G_bins[np.newaxis, :]
    return G_bins


def gtau_to_matsubara(G_bins, beta, dtau):
    """Convert G(τ) bins to Matsubara Green's function G_std(iωn).

    Uses the discrete Fourier transform:
        G_std(iωn) = -dtau * Σ_j G_ours(τ_j) exp(iωn τ_j)
    where G_ours = -G_std (our sign convention has G_ours > 0).

    Returns
    -------
    wn       : (Nn,) fermionic Matsubara frequencies (2n+1)π/β
    G_wn     : (N_bins, Nn) complex G_std(iωn) for each bin
    """
    N_bins, N_tau = G_bins.shape
    tau = np.arange(N_tau) * dtau                      # [0, dtau, ..., β-dtau]

    # Determine Matsubara indices n such that ωn = (2n+1)π/β ≤ MATSUBARA_CUTOFF
    n_max = int(np.floor((MATSUBARA_CUTOFF * beta / np.pi - 1.0) / 2.0))
    n_arr = np.arange(0, n_max + 1)                    # shape (Nn,)
    wn    = (2 * n_arr + 1) * np.pi / beta             # shape (Nn,)

    # DFT: G_std(iωn) = -dtau * Σ_j G_ours(τ_j) exp(i ωn τ_j)
    # outer product: (Nn, N_tau)
    phase  = np.exp(1j * np.outer(wn, tau))            # (Nn, N_tau)
    G_wn   = -dtau * (phase @ G_bins.T).T              # (N_bins, Nn)
    return wn, G_wn


def write_data_file(path, wn, G_wn_mean):
    """Write 3-column Matsubara data file: [ωn, Re[G], Im[G]]."""
    data = np.column_stack([wn, G_wn_mean.real, G_wn_mean.imag])
    np.savetxt(path, data, fmt="%.15e")


def write_error_file(path, wn, G_wn_err):
    """Write 3-column error file: [ωn, σ_Re, σ_Im]."""
    data = np.column_stack([wn, G_wn_err.real, G_wn_err.imag])
    np.savetxt(path, data, fmt="%.15e")


def write_params_file(path, data_fname, error_fname):
    """Write OmegaMaxEnt_input_params.dat for Matsubara fermionic data."""
    params_text = f"""data file: {data_fname}

OPTIONAL PREPROCESSING TIME PARAMETERS

DATA PARAMETERS
bosonic data (yes/[no]):
imaginary time data (yes/[no]):
temperature (in energy units, k_B=1):
finite value at infinite frequency (yes/[no]):
value at infinite frequency:
norm of spectral function:
1st moment:
1st moment error:
2nd moment:
2nd moment error:
3rd moment:
3rd moment error:
truncation frequency:

INPUT FILES PARAMETERS
input directory:
Re(G) column in data file (default: 2):
Im(G) column in data file (default: 3):
error file: {error_fname}
Re(G) column in error file (default: 2):
Im(G) column in error file (default: 3):
re-re covariance file:
im-im covariance file:
re-im covariance file:
column of G(tau) in data file (default: 2):
column of G(tau) error in error file (default: 2):
imaginary time covariance file:
added noise relative error (s1 s2 ...) (default: 0):

FREQUENCY GRID PARAMETERS
Matsubara frequency cutoff (in energy units, k_B=1): {MATSUBARA_CUTOFF}
spectral function width: {SPECTRAL_WIDTH}
spectral function center: {SPECTRAL_CENTER}
real frequency grid origin:
real frequency step: {FREQ_STEP}
real frequency grid file:
use non uniform grid in main spectral range (yes/[no]): yes
use parameterized real frequency grid (yes/[no]):
grid parameters (w_0 dw_0 w_1 dw_1 ... w_{{N-1}} dw_{{N-1}} w_N):
output real frequency grid parameters (w_min dw w_max): -8.0 0.032 8.0

COMPUTATION OPTIONS
evaluate moments (yes/[no]):
maximum moment:
default model center (default: 1st moment):
default model half width (default: standard deviation):
default model shape parameter (default: 2):
default model file:
initial spectral function file:
compute Pade result (yes/[no]):
number of frequencies for Pade:
imaginary part of frequency in Pade:

PREPROCESSING EXECUTION OPTIONS
preprocess only (yes/[no]):
display preprocessing figures (yes/[no]): no
display advanced preprocessing figures (yes/[no]): no
print other parameters (yes/[no]):


OPTIONAL MINIMIZATION TIME PARAMETERS

OUTPUT FILES PARAMETERS
output directory:
output file names suffix:
maximum alpha for which results are saved:
minimum alpha for which results are saved:
spectral function sample frequencies (w_1 w_2 ... w_N):

COMPUTATION PARAMETERS
initial value of alpha:
minimum value of alpha:
maximum optimal alpha:
minimum optimal alpha:

MINIMIZATION EXECUTION OPTIONS
number of values of alpha computed in one execution:
initialize maxent (yes/[no]):
initialize preprocessing (yes/[no]):
interactive mode ([yes]/no): no

DISPLAY OPTIONS
print results at each value of alpha (yes/[no]):
show optimal alpha figures ([yes]/no): no
show lowest alpha figures ([yes]/no): no
show alpha dependant curves ([yes]/no): no
reference spectral function file:
"""
    with open(path, "w") as f:
        f.write(params_text)


def parse_optimal_spectrum(result_dir):
    """Parse omega, A(omega) from OmegaMaxEnt final-result directory."""
    for fname in ["optimal_spectral_function.dat", "A_alpha_min.dat"]:
        path = os.path.join(result_dir, fname)
        if os.path.exists(path):
            print(f"  Parsing: {fname}")
            data = np.loadtxt(path)
            return data[:, 0], data[:, 1]
    raise FileNotFoundError(
        f"No spectral function output found in:\n  {result_dir}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not os.path.exists(OMEGAMAXENT_BIN):
        raise FileNotFoundError(
            f"OmegaMaxEnt binary not found: {OMEGAMAXENT_BIN}\n"
            f"Build it:\n"
            f"  cd MaxEnt_benchmark/OmegaMaxEnt && mkdir -p build && cd build\n"
            f"  cmake .. -DCMAKE_BUILD_TYPE=Release -DDOWNLOAD_ARMADILLO=1 -Wno-dev\n"
            f"  make -j4"
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from: {DATA_PATH}")
    G_bins = load_data(DATA_PATH)
    N_bins, L_tau = G_bins.shape
    T = 1.0 / BETA
    print(f"  {N_bins} bins × {L_tau} τ points  (τ ∈ [0, β-Δτ])")
    print(f"  β = {BETA}, T = {T:.4f}")

    # ------------------------------------------------------------------
    # DFT to Matsubara space (using all bins for error estimation)
    # ------------------------------------------------------------------
    print(f"\nComputing Matsubara Green's function (cutoff = {MATSUBARA_CUTOFF})...")
    wn, G_wn_bins = gtau_to_matsubara(G_bins, BETA, DTAU)
    Nn = len(wn)
    print(f"  {Nn} Matsubara frequencies: ω_0 = {wn[0]:.4f}, ω_max = {wn[-1]:.4f}")

    G_wn_mean    = G_wn_bins.mean(axis=0)
    # Compute SEM separately for real and imaginary parts; combining them into
    # a complex array for the write functions.
    G_wn_std_re  = G_wn_bins.real.std(axis=0, ddof=1) / np.sqrt(N_bins)
    G_wn_std_im  = G_wn_bins.imag.std(axis=0, ddof=1) / np.sqrt(N_bins)
    G_wn_std     = G_wn_std_re + 1j * G_wn_std_im

    print(f"  Im[G(iω_0)] = {G_wn_mean[0].imag:.6f}  (expected < 0 for G_std)")
    print(f"  Mean SEM_Im = {G_wn_std_im.mean():.3e}")

    # ------------------------------------------------------------------
    # Set up OmegaMaxEnt run directory
    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUT_DIR, "OmegaMaxEnt_run")
    os.makedirs(run_dir, exist_ok=True)

    data_fname  = "Gwn_mean.dat"
    error_fname = "Gwn_mean_err.dat"

    write_data_file( os.path.join(run_dir, data_fname),  wn, G_wn_mean)
    write_error_file(os.path.join(run_dir, error_fname), wn, G_wn_std)
    write_params_file(
        os.path.join(run_dir, "OmegaMaxEnt_input_params.dat"),
        data_fname, error_fname
    )

    params = {
        "DATA_PATH":            DATA_PATH,
        "SPECTRAL_INPUT_PATH":  SPECTRAL_INPUT_PATH,
        "SPECTRAL_TYPE":        SPECTRAL_TYPE,
        "INPUT_ID":             INPUT_ID,
        "NOISE_S":              NOISE_S,
        "BETA":                 BETA,
        "DTAU":                 DTAU,
        "TEMPERATURE":          T,
        "N_BINS":               N_bins,
        "L_TAU":                L_tau,
        "N_MATSUBARA":          Nn,
        "MATSUBARA_CUTOFF":     MATSUBARA_CUTOFF,
        "SPECTRAL_WIDTH":       SPECTRAL_WIDTH,
        "SPECTRAL_CENTER":      SPECTRAL_CENTER,
    }
    with open(os.path.join(OUT_DIR, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # ------------------------------------------------------------------
    # Run OmegaMaxEnt
    # ------------------------------------------------------------------
    print(f"\nRunning OmegaMaxEnt in: {run_dir}")
    print("  This may take 2–10 minutes...")

    proc = subprocess.run(
        [OMEGAMAXENT_BIN],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=1800,    # 30-minute hard cap
    )

    log_path = os.path.join(OUT_DIR, "omegamaxent.log")
    with open(log_path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(proc.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(proc.stderr)
    print(f"  Log written to: {log_path}")

    if proc.returncode != 0:
        print(f"  OmegaMaxEnt exited with code {proc.returncode}")
        print("  Last 20 lines of stdout:")
        for line in proc.stdout.strip().splitlines()[-20:]:
            print(f"    {line}")
        raise RuntimeError("OmegaMaxEnt failed. See log for details.")

    print(f"  OmegaMaxEnt finished (exit code 0)")

    # ------------------------------------------------------------------
    # Parse output
    # ------------------------------------------------------------------
    result_dir = os.path.join(run_dir, "OmegaMaxEnt_final_result")
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(
            f"No output directory: {result_dir}\n"
            f"Check log: {log_path}"
        )

    omega, A_opt = parse_optimal_spectrum(result_dir)
    norm = float(np.trapz(A_opt, omega))
    print(f"\n  ∫A dω = {norm:.4f}  (fermionic sum rule: expected ~1.0)")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(OUT_DIR, "summary.npz"),
        omega=omega,
        A_opt=A_opt,
        wn=wn,
        G_wn_mean=G_wn_mean,
        G_wn_std=G_wn_std,
        beta=BETA,
        spectral_input_path=SPECTRAL_INPUT_PATH,
    )
    print(f"  Saved: {OUT_DIR}/summary.npz")
    print(f"\nDone. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
