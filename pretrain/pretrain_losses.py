"""
Loss functions for VAE pretraining and fine-tuning on DQMC Green's function data.

Classes
-------
  KLDivergenceLoss             — KL[ q(z|x) || p(z) ] for diagonal Gaussian
  SpectralMSELoss              — MSE between predicted A(omega) and known Gaussian
  ChiSquaredLoss               — G(tau) reconstruction with full covariance whitening
  SpectralSmoothnessLoss       — second-derivative regularizer on A(omega)
  SpectralPositivityLoss       — penalizes negative values in A(omega)
  SpectralMomentLoss           — spectral moment matching loss
  NegativeGreenPenalty         — variance-weighted penalty on G(tau) < 0
  NegativeSecondDerivativePenalty  — variance-weighted penalty on G''(tau) < 0
  NegativeFourthDerivativePenalty  — variance-weighted penalty on G''''(tau) < 0

Functions
---------
  spectral_from_poles     — compute A(omega) from poles/residues
  pretrain_total_loss     — combined loss for Stage 1 pretraining
  finetune_total_loss     — combined loss for Stage 2 fine-tuning
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import special

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Utility: spectral function from poles/residues
# ---------------------------------------------------------------------------

def spectral_from_poles(poles, residues, omegas):
    """Compute A(omega) = -(1/pi) sum_p Im(residue_p / (omega - pole_p)).

    Parameters
    ----------
    poles    : complex tensor (B, P)
    residues : complex tensor (B, P)
    omegas   : real tensor (M,)

    Returns
    -------
    A : real tensor (B, M)
    """
    w = omegas.unsqueeze(0).unsqueeze(0)   # (1, 1, M)
    p = poles.unsqueeze(2)                  # (B, P, 1)
    r = residues.unsqueeze(2)               # (B, P, 1)
    A = -(1.0 / torch.pi) * torch.sum(torch.imag(r / (w - p)), dim=1)  # (B, M)
    return A


# ---------------------------------------------------------------------------
# KLDivergenceLoss
# ---------------------------------------------------------------------------

class KLDivergenceLoss(nn.Module):
    """KL divergence: KL[ q(z|x) || p(z) ] for a diagonal Gaussian posterior.

    KL = -0.5 * sum_i (1 + log σ²_i - μ²_i - σ²_i) / B

    Logvar is clamped to [-50, 50] for numerical stability (matches Ben's VAE2).
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        B = mu.shape[0]
        logvar_clamped = torch.clamp(logvar, min=-25, max=25)
        loss = -0.5 * torch.sum(
            1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
        ) / B
        return loss


# ---------------------------------------------------------------------------
# SpectralMSELoss
# ---------------------------------------------------------------------------

class SpectralMSELoss(nn.Module):
    """MSE loss between predicted spectral function and known Gaussian target.

    From the PDF protocol (page 2):
    L[p] = (1/N) sum_n [ integral_{-T}^{T} dt cosh(t) (A_pred(sinh t) - A_target(sinh t|mu_n, sigma_n))^2 ]

    where omega = sinh(t), T = arcsinh(4*W), evaluated via Gauss-Legendre quadrature.

    Parameters
    ----------
    W      : float, approximate bandwidth (integration extends to +-4W in omega space)
    N_gl   : int, number of Gauss-Legendre nodes
    """

    def __init__(self, W=6.0, N_gl=128):
        super().__init__()

        T = np.arcsinh(4.0 * W)
        nodes_ref, weights_ref = special.roots_legendre(N_gl)
        t_nodes = T * nodes_ref
        t_weights = T * weights_ref
        omega_nodes = np.sinh(t_nodes)
        cosh_nodes = np.cosh(t_nodes)
        eff_weights = t_weights * cosh_nodes

        self.register_buffer("omega_nodes", torch.tensor(omega_nodes, dtype=torch.float32))
        self.register_buffer("eff_weights", torch.tensor(eff_weights, dtype=torch.float32))

    def forward(self, poles, residues, mu_targets, sigma_targets):
        B = poles.shape[0]
        A_pred = spectral_from_poles(poles, residues, self.omega_nodes)  # (B, N_gl)
        omega = self.omega_nodes.unsqueeze(0)
        mu = mu_targets.unsqueeze(1)
        sigma = sigma_targets.unsqueeze(1)
        A_target = torch.exp(-0.5 * ((omega - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        diff_sq = (A_pred - A_target) ** 2
        integral = torch.sum(diff_sq * self.eff_weights.unsqueeze(0), dim=1)
        return integral.mean()


# ---------------------------------------------------------------------------
# ChiSquaredLoss
# ---------------------------------------------------------------------------

class ChiSquaredLoss(nn.Module):
    """Chi-squared loss via whitening (inverse square root) of the covariance matrix.

    Matches the implementation in cohensbw/SPE-VAE-AC.  At perfect fit
    (G_recon → G_hat, dG = G_hat - G_tilde = -eta where eta ~ N(0, C)):

        E[loss] = variance_threshold  ≈ 1

    so training loss converging to ~1 is the calibrated signal that the model
    is reconstructing the clean Green's function at the level of the DQMC noise.

    Implementation
    --------------
    1. Eigendecompose C = V diag(w) V^T  (eigh, ascending order)
    2. Clip w ≥ 0  (remove numerical negatives from finite-sample covariance)
    3. Sort descending; keep the top n_components that explain
       ≥ variance_threshold of total variance
    4. Rescale retained eigenvalues to preserve Tr(C):
           w_trunc ← w_trunc * sum(w) / sum(w_trunc)
    5. Build the whitening matrix with the sqrt(Ltau/n_components) normalisation
       factor that makes E[loss] = variance_threshold at perfect fit:
           inv_sqrt_C = V_trunc diag(1/√w_trunc) V_trunc^T  *  √(Ltau/n_components)
    6. Forward:
           dG_white = dG @ inv_sqrt_C
           loss = mean_batch( ||dG_white||² / Ltau )

    Parameters
    ----------
    C                  : ndarray (L_tau, L_tau), DQMC covariance matrix
    variance_threshold : float, fraction of total variance retained (default 0.999)
    """

    def __init__(self, C, variance_threshold=0.99):
        super().__init__()

        C_np = np.array(C, dtype=np.float64)
        Ltau = C_np.shape[0]

        w, V = np.linalg.eigh(C_np)
        w = np.maximum(w, 0.0)
        sum_w = float(np.sum(w))

        idx = np.argsort(w)[::-1]
        w_sorted = w[idx]
        V_sorted = V[:, idx]

        cumsum = np.cumsum(w_sorted)
        n_components = int(np.searchsorted(cumsum, variance_threshold * cumsum[-1]) + 1)
        n_components = min(n_components, Ltau)

        w_trunc = w_sorted[:n_components].copy()
        V_trunc = V_sorted[:, :n_components].copy()
        w_trunc *= sum_w / float(np.sum(w_trunc))

        eps = 1e-12
        inv_sqrt_C = (
            V_trunc @ np.diag(1.0 / np.sqrt(w_trunc + eps)) @ V_trunc.T
            * np.sqrt(Ltau / n_components)
        )

        self.Ltau = Ltau
        self.n_components = n_components
        print(f"  ChiSquaredLoss: {n_components}/{Ltau} components kept "
              f"({100 * variance_threshold:.1f}% variance threshold), "
              f"E[loss] → {variance_threshold:.4f} at perfect fit")
        self.register_buffer("inv_sqrt_C", torch.tensor(inv_sqrt_C, dtype=torch.float32))

    def forward(self, G_pred, G_input):
        dG = G_pred - G_input
        dG_white = dG @ self.inv_sqrt_C
        loss = torch.mean(torch.sum(dG_white ** 2, dim=1) / self.Ltau)
        return loss


# ---------------------------------------------------------------------------
# SpectralSmoothnessLoss
# ---------------------------------------------------------------------------

class SpectralSmoothnessLoss(nn.Module):
    """Smoothness regularizer on predicted spectral function.

    L_smooth = integral |d^2 A / d omega^2|^2 d omega

    Computed via second finite differences on a uniform omega grid.

    Parameters
    ----------
    Nw   : int, number of omega grid points
    wmin : float, lower bound of omega grid
    wmax : float, upper bound of omega grid
    """

    def __init__(self, Nw=500, wmin=-8.0, wmax=8.0):
        super().__init__()
        omegas = torch.linspace(wmin, wmax, Nw, dtype=torch.float32)
        self.register_buffer("omegas", omegas)
        self.dw = (wmax - wmin) / (Nw - 1)

    def forward(self, poles, residues):
        A = spectral_from_poles(poles, residues, self.omegas)
        d2A = (A[:, 2:] - 2.0 * A[:, 1:-1] + A[:, :-2]) / (self.dw ** 2)
        loss = torch.sum(d2A ** 2, dim=1).mean() * self.dw
        return loss


# ---------------------------------------------------------------------------
# SpectralPositivityLoss
# ---------------------------------------------------------------------------

class SpectralPositivityLoss(nn.Module):
    """Penalizes negative values in the predicted spectral function.

    L_pos = (1/B) sum_batch  integral ReLU(-A(omega))^2 d omega

    Evaluated via Gauss-Legendre quadrature with sinh-arcsinh variable
    transformation: omega = sinh(t), d omega = cosh(t) dt, t in [-T, T]
    where T = arcsinh(4*W). This clusters grid points near omega=0 where
    spectral features are sharp and is consistent with SpectralMSELoss.

    Parameters
    ----------
    W    : float, approximate bandwidth (grid covers +- 4W in omega)
    N_gl : int, number of Gauss-Legendre nodes
    """

    def __init__(self, W=6.0, N_gl=128):
        super().__init__()
        T = np.arcsinh(4.0 * W)
        nodes_ref, weights_ref = special.roots_legendre(N_gl)
        t_nodes     = T * nodes_ref
        t_weights   = T * weights_ref
        omega_nodes = np.sinh(t_nodes)
        eff_weights = t_weights * np.cosh(t_nodes)
        self.register_buffer("omega_nodes", torch.tensor(omega_nodes, dtype=torch.float32))
        self.register_buffer("eff_weights", torch.tensor(eff_weights, dtype=torch.float32))

    def forward(self, poles, residues):
        A   = spectral_from_poles(poles, residues, self.omega_nodes)  # (B, N_gl)
        neg = torch.relu(-A) ** 2
        return (neg * self.eff_weights.unsqueeze(0)).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# SpectralMomentLoss
# ---------------------------------------------------------------------------

class SpectralMomentLoss(nn.Module):
    """Penalizes mismatch of first two spectral moments with known Gaussian targets.

    M1 = integral(omega * A(omega) d_omega)   -> target: mu
    M2 = integral(omega^2 * A(omega) d_omega) -> target: mu^2 + sigma^2

    Parameters
    ----------
    W    : float, approximate bandwidth
    N_gl : int, number of Gauss-Legendre nodes
    """

    def __init__(self, W=6.0, N_gl=128):
        super().__init__()
        T = np.arcsinh(4.0 * W)
        nodes_ref, weights_ref = special.roots_legendre(N_gl)
        t_nodes = T * nodes_ref
        t_weights = T * weights_ref
        omega_nodes = np.sinh(t_nodes)
        eff_weights = t_weights * np.cosh(t_nodes)
        self.register_buffer("omega_nodes", torch.tensor(omega_nodes, dtype=torch.float32))
        self.register_buffer("eff_weights", torch.tensor(eff_weights, dtype=torch.float32))

    def forward(self, poles, residues, mu_targets, sigma_targets):
        A = spectral_from_poles(poles, residues, self.omega_nodes)
        w = self.omega_nodes.unsqueeze(0)
        ew = self.eff_weights.unsqueeze(0)
        M1 = torch.sum(w * A * ew, dim=1)
        M2 = torch.sum(w ** 2 * A * ew, dim=1)
        M1_target = mu_targets
        M2_target = mu_targets ** 2 + sigma_targets ** 2
        return ((M1 - M1_target) ** 2 + (M2 - M2_target) ** 2).mean()


# ---------------------------------------------------------------------------
# NegativeGreenPenalty
# ---------------------------------------------------------------------------

class NegativeGreenPenalty(nn.Module):
    """Variance-weighted penalty on G(tau) < 0.

    penalty = mean_batch[ sum_tau( relu(-G)^2 / var0 ) / Ltau ]

    var0 = diag(C) — per-tau DQMC noise variance. Down-weights noisy tau points
    and up-weights reliable ones. Matches Ben's cohensbw/SPE-VAE-AC implementation.

    Parameters
    ----------
    C : ndarray (Ltau, Ltau), DQMC covariance matrix
    """

    def __init__(self, C):
        super().__init__()
        var0 = np.diag(np.array(C, dtype=np.float64)).astype(np.float32)
        self.register_buffer("var0", torch.tensor(var0))

    def forward(self, Gtau_reconstructed):
        B, Ltau = Gtau_reconstructed.shape
        neg0 = torch.relu(-Gtau_reconstructed)
        return torch.mean(torch.sum(neg0 ** 2 / self.var0, dim=1) / Ltau)


# ---------------------------------------------------------------------------
# NegativeSecondDerivativePenalty
# ---------------------------------------------------------------------------

class NegativeSecondDerivativePenalty(nn.Module):
    """Variance-weighted penalty on G''(tau) < 0.

    G''[i] = G[i] - 2*G[i+1] + G[i+2]  (discrete second finite difference)
    penalty = mean_batch[ sum_i( relu(-G'')^2 / var2 ) / (Ltau-2) ]

    var2[i] = diag(C)[i] + 4*diag(C)[i+1] + diag(C)[i+2]
    Matches Ben's cohensbw/SPE-VAE-AC implementation.

    Parameters
    ----------
    C : ndarray (Ltau, Ltau), DQMC covariance matrix
    """

    def __init__(self, C):
        super().__init__()
        var0 = np.diag(np.array(C, dtype=np.float64))
        var2 = (var0[:-2] + 4 * var0[1:-1] + var0[2:]).astype(np.float32)
        self.register_buffer("var2", torch.tensor(var2))

    def forward(self, Gtau_reconstructed):
        B, Ltau = Gtau_reconstructed.shape
        dG2 = (Gtau_reconstructed[:, 2:]
               - 2 * Gtau_reconstructed[:, 1:-1]
               + Gtau_reconstructed[:, :-2])
        neg2 = torch.relu(-dG2)
        return torch.mean(torch.sum(neg2 ** 2 / self.var2, dim=1) / (Ltau - 2))


# ---------------------------------------------------------------------------
# NegativeFourthDerivativePenalty
# ---------------------------------------------------------------------------

class NegativeFourthDerivativePenalty(nn.Module):
    """Variance-weighted penalty on G''''(tau) < 0.

    G''''[i] = G[i] - 4*G[i+1] + 6*G[i+2] - 4*G[i+3] + G[i+4]
    penalty = mean_batch[ sum_i( relu(-G'''')^2 / var4 ) / (Ltau-4) ]

    var4[i] = diag(C)[i] + 16*diag(C)[i+1] + 36*diag(C)[i+2]
            + 16*diag(C)[i+3] + diag(C)[i+4]
    Matches Ben's cohensbw/SPE-VAE-AC implementation.

    Parameters
    ----------
    C : ndarray (Ltau, Ltau), DQMC covariance matrix
    """

    def __init__(self, C):
        super().__init__()
        var0 = np.diag(np.array(C, dtype=np.float64))
        var4 = (
            var0[:-4] + 16 * var0[1:-3] + 36 * var0[2:-2]
            + 16 * var0[3:-1] + var0[4:]
        ).astype(np.float32)
        self.register_buffer("var4", torch.tensor(var4))

    def forward(self, Gtau_reconstructed):
        B, Ltau = Gtau_reconstructed.shape
        dG4 = (
            Gtau_reconstructed[:, 4:]
            - 4 * Gtau_reconstructed[:, 3:-1]
            + 6 * Gtau_reconstructed[:, 2:-2]
            - 4 * Gtau_reconstructed[:, 1:-3]
            + Gtau_reconstructed[:, :-4]
        )
        neg4 = torch.relu(-dG4)
        return torch.mean(torch.sum(neg4 ** 2 / self.var4, dim=1) / (Ltau - 4))


# ---------------------------------------------------------------------------
# Combined pretraining loss (Stage 1)
# ---------------------------------------------------------------------------

def pretrain_total_loss(
    G_recon,
    G_input,
    mu_vae,
    logvar_vae,
    poles,
    residues,
    mu_targets,
    sigma_targets,
    spectral_mse_fn,
    chi2_fn,
    smoothness_fn,
    lambda_chi2=1.0,
    lambda_spec=1.0,
    lambda_smooth=0.01,
    alpha_kl=0.0,
    moment_fn=None,
    lambda_moment=0.0,
):
    """Combine all pretraining losses.

    L = lambda_chi2 * L_chi2
      + lambda_spec * L_spectral_mse
      + lambda_smooth * L_smoothness
      + alpha_kl * L_KL
      + lambda_moment * L_moment

    Returns
    -------
    total      : scalar tensor
    chi2_val   : scalar tensor
    spec_val   : scalar tensor
    smooth_val : scalar tensor
    kl_val     : scalar tensor
    moment_val : scalar tensor
    """
    chi2_val = chi2_fn(G_recon, G_input)
    spec_val = spectral_mse_fn(poles, residues, mu_targets, sigma_targets)
    smooth_val = smoothness_fn(poles, residues)

    # KL divergence (inline, logvar clamped for stability)
    B = mu_vae.shape[0]
    logvar_clamped = torch.clamp(logvar_vae, min=-25, max=25)
    kl_val = -0.5 * torch.sum(
        1 + logvar_clamped - mu_vae.pow(2) - logvar_clamped.exp()
    ) / B

    if moment_fn is not None and lambda_moment > 0.0:
        moment_val = moment_fn(poles, residues, mu_targets, sigma_targets)
    else:
        moment_val = torch.tensor(0.0, device=G_recon.device)

    total = (
        lambda_chi2 * chi2_val
        + lambda_spec * spec_val
        + lambda_smooth * smooth_val
        + alpha_kl * kl_val
        + lambda_moment * moment_val
    )

    return total, chi2_val, spec_val, smooth_val, kl_val, moment_val


# ---------------------------------------------------------------------------
# Combined fine-tuning loss (Stage 2)
# ---------------------------------------------------------------------------

def finetune_total_loss(
    G_recon,
    G_input,
    mu_vae,
    logvar_vae,
    poles,
    residues,
    kl_fn,
    chi2_fn,
    smoothness_fn,
    neg_green_fn,
    neg_second_fn,
    neg_fourth_fn,
    positivity_fn=None,
    lambda_chi2=1.0,
    lambda_smooth=0.0,
    lambda_pos=0.0,
    alpha_kl=0.0,
    eta0=0.0,
    eta2=0.0,
    eta4=0.0,
):
    """Combine all fine-tuning losses.

    L = lambda_chi2 * chi2
      + lambda_smooth * smoothness
      + lambda_pos   * positivity
      + alpha_kl     * KL
      + eta0         * neg_green_penalty        (variance-weighted via diag(C))
      + eta2         * neg_second_deriv_penalty (variance-weighted via diag(C))
      + eta4         * neg_fourth_deriv_penalty (variance-weighted via diag(C))

    Parameters
    ----------
    kl_fn         : KLDivergenceLoss instance
    chi2_fn       : ChiSquaredLoss instance
    smoothness_fn : SpectralSmoothnessLoss instance
    neg_green_fn  : NegativeGreenPenalty instance
    neg_second_fn : NegativeSecondDerivativePenalty instance
    neg_fourth_fn : NegativeFourthDerivativePenalty instance
    positivity_fn : SpectralPositivityLoss instance (optional)

    Returns
    -------
    total          : scalar tensor
    chi2_val       : scalar tensor
    smooth_val     : scalar tensor
    pos_val        : scalar tensor
    kl_val         : scalar tensor
    neg_green_val  : scalar tensor
    neg_second_val : scalar tensor
    neg_fourth_val : scalar tensor
    """
    chi2_val = chi2_fn(G_recon, G_input)
    smooth_val = smoothness_fn(poles, residues)
    kl_val = kl_fn(mu_vae, logvar_vae)
    neg_green_val = neg_green_fn(G_recon)
    neg_second_val = neg_second_fn(G_recon)
    neg_fourth_val = neg_fourth_fn(G_recon)

    if positivity_fn is not None:
        pos_val = positivity_fn(poles, residues)
    else:
        pos_val = torch.tensor(0.0, device=G_recon.device)

    total = (
        lambda_chi2  * chi2_val
        + lambda_smooth * smooth_val
        + lambda_pos    * pos_val
        + alpha_kl      * kl_val
        + eta0          * neg_green_val
        + eta2          * neg_second_val
        + eta4          * neg_fourth_val
    )

    return total, chi2_val, smooth_val, pos_val, kl_val, neg_green_val, neg_second_val, neg_fourth_val
