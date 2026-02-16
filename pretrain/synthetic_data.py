"""
Synthetic single-Gaussian data generation for VAE pretraining.

Implements the protocol from vae_pretraining_protocol.pdf:
  1. Sample mu_n ~ N(0, sigma_mu), sigma_n ~ InverseGamma(alpha, beta)
  2. Compute clean G(tau) via numerical integration of K(tau, omega) * A(omega)
  3. Add DQMC-like noise via sqrt(C) @ R, where C is the covariance matrix

Provides both a pre-generate-and-save workflow and an on-the-fly PyTorch Dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import integrate, special, stats


# ---------------------------------------------------------------------------
# Fermionic kernel (numerically stable form, matching SmoQySynthAC)
# ---------------------------------------------------------------------------

def kernel_tau_fermi(omega, tau, beta):
    """Numerically stable fermionic imaginary-time kernel.

    K(omega, tau) = 1 / (exp(tau * omega) + exp((tau - beta) * omega))

    Parameters
    ----------
    omega : ndarray, shape (N_omega,)
    tau   : ndarray, shape (L_tau,)
    beta  : float

    Returns
    -------
    K : ndarray, shape (L_tau, N_omega)
    """
    omega = np.asarray(omega, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    # Broadcast: tau[:, None] (L_tau, 1), omega[None, :] (1, N_omega)
    arg1 = tau[:, None] * omega[None, :]           # (L_tau, N_omega)
    arg2 = (tau[:, None] - beta) * omega[None, :]  # (L_tau, N_omega)
    # Clamp for numerical safety
    arg1 = np.clip(arg1, -500, 500)
    arg2 = np.clip(arg2, -500, 500)
    K = 1.0 / (np.exp(arg1) + np.exp(arg2))
    return K


# ---------------------------------------------------------------------------
# Parameter sampling (from the PDF protocol)
# ---------------------------------------------------------------------------

def sample_gaussian_params(N, sigma_mu, mu_sigma, sigma_sigma, rng=None):
    """Sample (mu, sigma) for N Gaussian spectral functions.

    mu_n     ~ N(0, sigma_mu)
    sigma_n  ~ InverseGamma(alpha_sigma, beta_sigma)

    where alpha_sigma = 2 + mu_sigma^2 / sigma_sigma^2
          beta_sigma  = mu_sigma * (1 + mu_sigma^2 / sigma_sigma^2)

    This gives InverseGamma with mean = mu_sigma, std = sigma_sigma.

    Parameters
    ----------
    N            : int, number of samples
    sigma_mu     : float, std of the Normal for mu
    mu_sigma     : float, mean of the InverseGamma for sigma
    sigma_sigma  : float, std of the InverseGamma for sigma
    rng          : numpy Generator or None

    Returns
    -------
    mus    : ndarray (N,)
    sigmas : ndarray (N,)
    """
    if rng is None:
        rng = np.random.default_rng()

    mus = rng.normal(0.0, sigma_mu, size=N)

    alpha_sigma = 2.0 + mu_sigma**2 / sigma_sigma**2
    beta_sigma = mu_sigma * (1.0 + mu_sigma**2 / sigma_sigma**2)

    # scipy InvGamma: pdf(x, a) = x^(-a-1) * exp(-1/x) / Gamma(a), with scale = beta
    sigmas = stats.invgamma.rvs(a=alpha_sigma, scale=beta_sigma, size=N, random_state=rng)

    return mus, sigmas


# ---------------------------------------------------------------------------
# Clean G(tau) computation via adaptive quadrature (high precision)
# ---------------------------------------------------------------------------

def _gaussian_pdf(omega, mu, sigma):
    """Gaussian probability density function."""
    return np.exp(-0.5 * ((omega - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def compute_clean_greens_quadrature(mus, sigmas, taus, beta, tol=1e-12):
    """Compute clean G(tau) for each (mu, sigma) via adaptive quadrature.

    G(tau) = integral_{-inf}^{+inf} K(omega, tau) * A(omega) d_omega

    Uses scipy.integrate.quad_vec with split at omega=0 for stability.
    This matches the SmoQySynthAC approach.

    Parameters
    ----------
    mus    : ndarray (N,)
    sigmas : ndarray (N,)
    taus   : ndarray (L_tau,)
    beta   : float
    tol    : float, absolute tolerance for integration

    Returns
    -------
    G_hat : ndarray (N, L_tau)
    """
    N = len(mus)
    L_tau = len(taus)
    G_hat = np.zeros((N, L_tau), dtype=np.float64)

    for n in range(N):
        mu_n, sigma_n = mus[n], sigmas[n]

        def integrand(omega):
            K = kernel_tau_fermi(np.atleast_1d(omega), taus, beta)  # (L_tau, 1)
            A = _gaussian_pdf(omega, mu_n, sigma_n)
            return (K * A).squeeze()  # (L_tau,)

        # Split integral at 0 for stability (matching Julia quadgk with bounds (-Inf, 0, +Inf))
        result_neg, _ = integrate.quad_vec(integrand, -np.inf, 0.0, epsabs=tol)
        result_pos, _ = integrate.quad_vec(integrand, 0.0, np.inf, epsabs=tol)
        G_hat[n] = result_neg + result_pos

    return G_hat


# ---------------------------------------------------------------------------
# Clean G(tau) computation via Gauss-Legendre quadrature (fast, for on-the-fly)
# ---------------------------------------------------------------------------

def _build_gl_kernel(taus, beta, N_gl=512, omega_max=20.0):
    """Build Gauss-Legendre quadrature kernel for fast G(tau) computation.

    Maps GL nodes from [-1, 1] to [-omega_max, 0] and [0, omega_max],
    split at 0 for numerical stability.

    Returns
    -------
    omega_nodes : ndarray (2*N_gl,) - quadrature nodes in omega space
    gl_weights  : ndarray (2*N_gl,) - quadrature weights (including Jacobian)
    K_matrix    : ndarray (L_tau, 2*N_gl) - kernel matrix K(tau, omega) * w
    """
    nodes, weights = special.roots_legendre(N_gl)

    # Negative half: [-omega_max, 0]
    omega_neg = 0.5 * omega_max * (nodes - 1.0)   # maps [-1,1] -> [-omega_max, 0]
    w_neg = 0.5 * omega_max * weights

    # Positive half: [0, omega_max]
    omega_pos = 0.5 * omega_max * (nodes + 1.0)   # maps [-1,1] -> [0, omega_max]
    w_pos = 0.5 * omega_max * weights

    omega_nodes = np.concatenate([omega_neg, omega_pos])
    gl_weights = np.concatenate([w_neg, w_pos])

    K = kernel_tau_fermi(omega_nodes, taus, beta)  # (L_tau, 2*N_gl)

    return omega_nodes, gl_weights, K


def compute_clean_greens_gl(mus, sigmas, taus, beta, N_gl=512, omega_max=20.0):
    """Compute clean G(tau) via Gauss-Legendre quadrature (vectorized).

    Fast alternative to adaptive quadrature for large batch generation.

    Parameters
    ----------
    mus       : ndarray (N,)
    sigmas    : ndarray (N,)
    taus      : ndarray (L_tau,)
    beta      : float
    N_gl      : int, number of GL nodes per half-axis
    omega_max : float, integration cutoff

    Returns
    -------
    G_hat : ndarray (N, L_tau)
    """
    omega_nodes, gl_weights, K = _build_gl_kernel(taus, beta, N_gl, omega_max)

    # K: (L_tau, 2*N_gl), omega_nodes: (2*N_gl,), gl_weights: (2*N_gl,)
    N = len(mus)
    L_tau = len(taus)

    # Evaluate Gaussian PDFs at all omega nodes for all samples
    # mus: (N,), sigmas: (N,), omega_nodes: (M,) -> A: (N, M)
    diff = omega_nodes[None, :] - mus[:, None]  # (N, M)
    A = np.exp(-0.5 * (diff / sigmas[:, None]) ** 2) / (sigmas[:, None] * np.sqrt(2.0 * np.pi))

    # Weighted A: (N, M) * (M,) -> (N, M)
    A_weighted = A * gl_weights[None, :]

    # G_hat = K @ A_weighted^T -> (L_tau, N) -> transpose -> (N, L_tau)
    G_hat = (K @ A_weighted.T).T

    return G_hat


# ---------------------------------------------------------------------------
# Covariance matrix computation and noise addition
# ---------------------------------------------------------------------------

def compute_covariance_from_bins(G_bins):
    """Compute sample covariance matrix from DQMC bins.

    C_{l',l} = (1/(N-1)) sum_n (G_{l',n} - G_bar_{l'}) (G_{l,n} - G_bar_l)

    Parameters
    ----------
    G_bins : ndarray (N, L_tau) — N bins of Green's function data

    Returns
    -------
    C : ndarray (L_tau, L_tau)
    """
    N = G_bins.shape[0]
    G_mean = G_bins.mean(axis=0, keepdims=True)  # (1, L_tau)
    dG = G_bins - G_mean  # (N, L_tau)
    C = (dG.T @ dG) / (N - 1)  # (L_tau, L_tau)
    return C


def load_covariance_from_dqmc(csv_path):
    """Load DQMC bins from CSV and compute covariance matrix.

    Parameters
    ----------
    csv_path : str, path to Gbins CSV file (N rows x L_tau columns)

    Returns
    -------
    C : ndarray (L_tau, L_tau)
    """
    G_bins = np.loadtxt(csv_path, delimiter=",")
    return compute_covariance_from_bins(G_bins)


def cholesky_sqrt(C, eps=1e-10):
    """Compute Cholesky factor of covariance matrix (regularized).

    Parameters
    ----------
    C   : ndarray (L_tau, L_tau)
    eps : float, regularization

    Returns
    -------
    L : ndarray (L_tau, L_tau) lower-triangular such that L @ L^T = C + eps*I
    """
    L_tau = C.shape[0]
    C_reg = C + eps * np.eye(L_tau)
    return np.linalg.cholesky(C_reg)


def add_noise(G_hat, sqrt_C, rng=None):
    """Add DQMC-like noise to clean Green's functions.

    G_tilde^(n) = G_hat^(n) + sqrt_C @ R^(n), R ~ N(0, 1)

    Parameters
    ----------
    G_hat  : ndarray (N, L_tau)
    sqrt_C : ndarray (L_tau, L_tau) — Cholesky factor of covariance
    rng    : numpy Generator or None

    Returns
    -------
    G_tilde : ndarray (N, L_tau)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L_tau = G_hat.shape
    R = rng.standard_normal((L_tau, N))  # (L_tau, N)
    noise = sqrt_C @ R  # (L_tau, N)
    G_tilde = G_hat + noise.T  # (N, L_tau)
    return G_tilde


# ---------------------------------------------------------------------------
# Generate and save dataset
# ---------------------------------------------------------------------------

def generate_and_save(
    N,
    sigma_mu,
    mu_sigma,
    sigma_sigma,
    beta,
    dtau,
    output_dir,
    covariance_source=None,
    use_quadrature=True,
    N_gl=512,
    omega_max=20.0,
    seed=None,
):
    """Generate synthetic Gaussian pretraining data and save to disk.

    Parameters
    ----------
    N                 : int, number of samples
    sigma_mu          : float, std of Normal for mu
    mu_sigma          : float, mean of InverseGamma for sigma
    sigma_sigma       : float, std of InverseGamma for sigma
    beta              : float, inverse temperature
    dtau              : float, imaginary time step
    output_dir        : str, directory to save files
    covariance_source : str or None
        If str: path to existing DQMC Gbins CSV to compute covariance from.
        If None: compute covariance from the generated clean G(tau) curves.
    use_quadrature    : bool, if True use adaptive quadrature (slow, precise);
                        if False use GL quadrature (fast)
    N_gl              : int, GL nodes per half-axis (if use_quadrature=False)
    omega_max         : float, integration cutoff (if use_quadrature=False)
    seed              : int or None, random seed
    """
    rng = np.random.default_rng(seed)

    L_tau = int(round(beta / dtau))
    taus = np.linspace(0.0, beta - dtau, L_tau)

    print(f"Generating {N} synthetic Gaussian samples (beta={beta}, L_tau={L_tau})...")

    # 1. Sample parameters
    mus, sigmas = sample_gaussian_params(N, sigma_mu, mu_sigma, sigma_sigma, rng=rng)
    print(f"  mu range: [{mus.min():.3f}, {mus.max():.3f}]")
    print(f"  sigma range: [{sigmas.min():.3f}, {sigmas.max():.3f}]")

    # 2. Compute clean G(tau)
    if use_quadrature:
        print("  Computing G(tau) via adaptive quadrature (this may take a while)...")
        G_hat = compute_clean_greens_quadrature(mus, sigmas, taus, beta)
    else:
        print(f"  Computing G(tau) via GL quadrature (N_gl={N_gl}, omega_max={omega_max})...")
        G_hat = compute_clean_greens_gl(mus, sigmas, taus, beta, N_gl, omega_max)

    print(f"  G(tau=0) range: [{G_hat[:, 0].min():.4f}, {G_hat[:, 0].max():.4f}]")

    # 3. Covariance matrix
    if covariance_source is not None:
        print(f"  Loading covariance from: {covariance_source}")
        C = load_covariance_from_dqmc(covariance_source)
    else:
        print("  Computing covariance from generated clean G(tau)...")
        C = compute_covariance_from_bins(G_hat)

    sqrt_C = cholesky_sqrt(C)

    # 4. Add noise
    print("  Adding noise...")
    G_tilde = add_noise(G_hat, sqrt_C, rng=rng)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(os.path.join(output_dir, "Gbins_synthetic.csv"), G_tilde, delimiter=",")
    np.savetxt(os.path.join(output_dir, "Ghat_clean.csv"), G_hat, delimiter=",")
    np.savetxt(os.path.join(output_dir, "params.csv"),
               np.column_stack([mus, sigmas]), delimiter=",",
               header="mu,sigma", comments="")
    np.save(os.path.join(output_dir, "covariance.npy"), C)

    print(f"  Saved to {output_dir}/")
    print(f"    Gbins_synthetic.csv  ({G_tilde.shape})")
    print(f"    Ghat_clean.csv       ({G_hat.shape})")
    print(f"    params.csv           ({N}, 2)")
    print(f"    covariance.npy       ({C.shape})")

    return G_tilde, G_hat, mus, sigmas, C


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class SyntheticGaussianDataset(Dataset):
    """On-the-fly synthetic Gaussian dataset for pretraining.

    Each __getitem__ call samples a fresh (mu, sigma), computes clean G(tau),
    and adds noise. This provides effectively infinite data variety.

    Parameters
    ----------
    N_samples    : int, virtual dataset size (controls epoch length)
    sigma_mu     : float
    mu_sigma     : float
    sigma_sigma  : float
    beta         : float
    dtau         : float
    sqrt_C       : ndarray (L_tau, L_tau), Cholesky factor of covariance
    N_gl         : int, GL quadrature nodes per half-axis
    omega_max    : float, integration cutoff
    """

    def __init__(self, N_samples, sigma_mu, mu_sigma, sigma_sigma,
                 beta, dtau, sqrt_C, N_gl=512, omega_max=20.0):
        self.N_samples = N_samples
        self.sigma_mu = sigma_mu
        self.mu_sigma = mu_sigma
        self.sigma_sigma = sigma_sigma
        self.beta = beta
        self.dtau = dtau
        self.L_tau = int(round(beta / dtau))
        self.taus = np.linspace(0.0, beta - dtau, self.L_tau)
        self.sqrt_C = sqrt_C  # (L_tau, L_tau)

        # Precompute GL kernel for fast G(tau) computation
        self._omega_nodes, self._gl_weights, self._K = _build_gl_kernel(
            self.taus, beta, N_gl, omega_max
        )

        # InverseGamma parameters
        self._alpha_sigma = 2.0 + mu_sigma**2 / sigma_sigma**2
        self._beta_sigma = mu_sigma * (1.0 + mu_sigma**2 / sigma_sigma**2)

    def __len__(self):
        return self.N_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng()

        # Sample mu, sigma
        mu = rng.normal(0.0, self.sigma_mu)
        sigma = stats.invgamma.rvs(a=self._alpha_sigma, scale=self._beta_sigma,
                                   random_state=rng)

        # Compute Gaussian PDF at quadrature nodes
        A = _gaussian_pdf(self._omega_nodes, mu, sigma)

        # G_hat = K @ (A * weights)
        G_hat = self._K @ (A * self._gl_weights)  # (L_tau,)

        # Add noise
        R = rng.standard_normal(self.L_tau)
        noise = self.sqrt_C @ R
        G_tilde = G_hat + noise

        return (
            torch.tensor(G_tilde, dtype=torch.float32),
            torch.tensor(mu, dtype=torch.float32),
            torch.tensor(sigma, dtype=torch.float32),
        )


class SavedSyntheticDataset(Dataset):
    """Load pre-generated synthetic dataset from CSV files.

    Parameters
    ----------
    data_dir : str, directory containing Gbins_synthetic.csv and params.csv
    """

    def __init__(self, data_dir):
        gbins_path = os.path.join(data_dir, "Gbins_synthetic.csv")
        params_path = os.path.join(data_dir, "params.csv")

        self.data = np.loadtxt(gbins_path, delimiter=",").astype(np.float32)
        params = np.loadtxt(params_path, delimiter=",", skiprows=1).astype(np.float32)
        self.mus = params[:, 0]
        self.sigmas = params[:, 1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.mus[idx], dtype=torch.float32),
            torch.tensor(self.sigmas[idx], dtype=torch.float32),
        )
