"""
Loss functions for VAE pretraining on synthetic Gaussian data.

Implements:
  1. SpectralMSELoss — MSE between predicted A(omega) and known Gaussian,
     evaluated via Gauss-Legendre quadrature with sinh transform (from PDF protocol)
  2. ChiSquaredLoss — G(tau) reconstruction loss with full covariance matrix
  3. SpectralSmoothnessLoss — second-derivative regularizer on A(omega)
  4. pretrain_total_loss — combines all losses
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import special

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loss2 import kl_divergence_loss  # type: ignore


# ---------------------------------------------------------------------------
# Spectral function from poles/residues (inline, for arbitrary omega points)
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
    # omegas: (M,) -> (1, 1, M)
    # poles: (B, P) -> (B, P, 1)
    # residues: (B, P) -> (B, P, 1)
    w = omegas.unsqueeze(0).unsqueeze(0)        # (1, 1, M)
    p = poles.unsqueeze(2)                       # (B, P, 1)
    r = residues.unsqueeze(2)                    # (B, P, 1)

    A = -(1.0 / torch.pi) * torch.sum(torch.imag(r / (w - p)), dim=1)  # (B, M)
    return A


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

        # GL nodes and weights on [-1, 1]
        nodes_ref, weights_ref = special.roots_legendre(N_gl)

        # Map to [-T, T]: t_j = T * nodes_ref_j, w_j = T * weights_ref_j
        t_nodes = T * nodes_ref                     # (N_gl,)
        t_weights = T * weights_ref                  # (N_gl,)

        # Precompute omega = sinh(t) and cosh(t)
        omega_nodes = np.sinh(t_nodes)               # (N_gl,)
        cosh_nodes = np.cosh(t_nodes)                # (N_gl,)

        # Effective weights: w_j * cosh(t_j) for the transformed integral
        eff_weights = t_weights * cosh_nodes          # (N_gl,)

        self.register_buffer("omega_nodes", torch.tensor(omega_nodes, dtype=torch.float32))
        self.register_buffer("eff_weights", torch.tensor(eff_weights, dtype=torch.float32))

    def forward(self, poles, residues, mu_targets, sigma_targets):
        """
        Parameters
        ----------
        poles         : complex tensor (B, P)
        residues      : complex tensor (B, P)
        mu_targets    : float tensor (B,)
        sigma_targets : float tensor (B,)

        Returns
        -------
        loss : scalar tensor
        """
        B = poles.shape[0]

        # Predicted A(omega) at quadrature nodes: (B, N_gl)
        A_pred = spectral_from_poles(poles, residues, self.omega_nodes)

        # Target Gaussian PDF at quadrature nodes: (B, N_gl)
        omega = self.omega_nodes.unsqueeze(0)            # (1, N_gl)
        mu = mu_targets.unsqueeze(1)                     # (B, 1)
        sigma = sigma_targets.unsqueeze(1)               # (B, 1)
        A_target = torch.exp(-0.5 * ((omega - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )

        # Integrand: (A_pred - A_target)^2 * eff_weights
        diff_sq = (A_pred - A_target) ** 2               # (B, N_gl)
        integral = torch.sum(diff_sq * self.eff_weights.unsqueeze(0), dim=1)  # (B,)

        loss = integral.mean()
        return loss


# ---------------------------------------------------------------------------
# ChiSquaredLoss
# ---------------------------------------------------------------------------

class ChiSquaredLoss(nn.Module):
    """Chi-squared loss with full covariance matrix.

    L_chi2 = (1/(B * L_tau)) sum_batch (dG^T @ C_inv @ dG)

    Parameters
    ----------
    C : ndarray or tensor (L_tau, L_tau), covariance matrix
    eps : float, regularization for inversion
    """

    def __init__(self, C, cond_cap=1e6):
        super().__init__()

        if isinstance(C, np.ndarray):
            C = torch.tensor(C, dtype=torch.float64)

        # Eigenvalue-clipped inversion to cap condition number.
        # This prevents ill-conditioned directions from dominating chi^2,
        # analogous to the std clamping in loss2.mse_loss.
        eigvals, eigvecs = torch.linalg.eigh(C)
        min_eigval = eigvals.max() / cond_cap
        eigvals_clipped = torch.clamp(eigvals, min=min_eigval.item())
        C_inv = eigvecs @ torch.diag(1.0 / eigvals_clipped) @ eigvecs.T
        C_inv = C_inv.to(torch.float32)

        self.register_buffer("C_inv", C_inv)

    def forward(self, G_pred, G_input):
        """
        Parameters
        ----------
        G_pred  : tensor (B, L_tau)
        G_input : tensor (B, L_tau)

        Returns
        -------
        loss : scalar tensor
        """
        B, L_tau = G_pred.shape
        dG = G_pred - G_input  # (B, L_tau)

        # (B, L_tau) @ (L_tau, L_tau) -> (B, L_tau), then dot with dG
        chi2 = torch.sum(dG * (dG @ self.C_inv), dim=1)  # (B,)

        loss = chi2.mean() / L_tau
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
        """
        Parameters
        ----------
        poles    : complex tensor (B, P)
        residues : complex tensor (B, P)

        Returns
        -------
        loss : scalar tensor
        """
        # A(omega): (B, Nw)
        A = spectral_from_poles(poles, residues, self.omegas)

        # Second finite difference: A''_j = (A_{j+1} - 2*A_j + A_{j-1}) / dw^2
        d2A = (A[:, 2:] - 2.0 * A[:, 1:-1] + A[:, :-2]) / (self.dw ** 2)

        # Integral: sum |d2A|^2 * dw
        loss = torch.sum(d2A ** 2, dim=1).mean() * self.dw
        return loss


# ---------------------------------------------------------------------------
# SpectralPositivityLoss
# ---------------------------------------------------------------------------

class SpectralPositivityLoss(nn.Module):
    """Penalizes negative values in the predicted spectral function.

    L_pos = (1/B) sum_batch  integral ReLU(-A(omega))^2 d omega

    Evaluated on a uniform omega grid via finite sum.

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
        """
        Parameters
        ----------
        poles    : complex tensor (B, P)
        residues : complex tensor (B, P)

        Returns
        -------
        loss : scalar tensor
        """
        # A(omega): (B, Nw)
        A = spectral_from_poles(poles, residues, self.omegas)

        # Penalize negative values: integral ReLU(-A)^2 dw
        loss = torch.sum(torch.relu(-A) ** 2, dim=1).mean() * self.dw
        return loss


# ---------------------------------------------------------------------------
# Combined pretraining loss
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
):
    """Combine all pretraining losses.

    L = lambda_chi2 * L_chi2
      + lambda_spec * L_spectral_mse
      + lambda_smooth * L_smoothness
      + alpha_kl * L_KL

    Returns
    -------
    total     : scalar tensor
    chi2_val  : scalar tensor
    spec_val  : scalar tensor
    smooth_val: scalar tensor
    kl_val    : scalar tensor
    """
    chi2_val = chi2_fn(G_recon, G_input)
    spec_val = spectral_mse_fn(poles, residues, mu_targets, sigma_targets)
    smooth_val = smoothness_fn(poles, residues)
    kl_val = kl_divergence_loss(mu_vae, logvar_vae)

    total = (
        lambda_chi2 * chi2_val
        + lambda_spec * spec_val
        + lambda_smooth * smooth_val
        + alpha_kl * kl_val
    )

    return total, chi2_val, spec_val, smooth_val, kl_val

