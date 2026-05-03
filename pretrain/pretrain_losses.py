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

def ledoit_wolf_shrinkage(G_bins):
    """Ledoit-Wolf shrinkage covariance with scaled-identity target.

    C_LW = rho * mu * I + (1 - rho) * S, where S = (Xc^T Xc) / N is the ML
    sample covariance, mu = tr(S)/p, and rho is the closed-form optimal
    intensity (Ledoit & Wolf 2004 eqs. 14-15 with F = mu * I).

    Pulls the tail eigenvalues away from zero, fixing the Marchenko-Pastur
    blowup that prevents using the untruncated whitener directly.

    Parameters
    ----------
    G_bins : ndarray (N, p), raw bins (centred internally)

    Returns
    -------
    C_LW : ndarray (p, p)
    rho  : float in [0, 1]
    """
    X = np.asarray(G_bins, dtype=np.float64)
    N, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / N
    mu = float(np.trace(S)) / p
    F = mu * np.eye(p)

    d_sq = float(np.sum((S - F) ** 2))

    # b_bar^2 = (1/N^2) sum_n ||x_n x_n^T - S||_F^2
    #         = (1/N^2) sum_n ||x_n||^4  -  (1/N) ||S||_F^2
    x_norms_sq = np.sum(Xc ** 2, axis=1)
    sum_x4     = float(np.sum(x_norms_sq ** 2))
    S_fro_sq   = float(np.sum(S ** 2))
    b_sq = sum_x4 / (N * N) - S_fro_sq / N
    b_sq = min(max(b_sq, 0.0), d_sq)

    rho = b_sq / d_sq if d_sq > 0 else 0.0
    rho = float(min(max(rho, 0.0), 1.0))

    C_LW = rho * F + (1.0 - rho) * S
    return C_LW, rho


class ChiSquaredLoss(nn.Module):
    """Chi-squared loss via whitening (inverse square root) of the covariance matrix.

    Matches the implementation in cohensbw/SPE-VAE-AC.  At perfect fit
    (G_recon → G_hat, dG = G_hat - G_tilde = -eta where eta ~ N(0, C)):

        E[loss] ≈ 1

    so training loss converging to ~1 is the calibrated signal that the model
    is reconstructing the clean Green's function at the level of the DQMC noise.

    Two whitening modes
    -------------------
    `covariance_estimator="pca_truncated"` (default — backwards compatible):
       Sample C is poorly conditioned (Marchenko-Pastur regime), so keep only
       the top eigendirections explaining `variance_threshold` of the trace.
       The truncated pseudo-inverse-square-root is multiplied by
       sqrt(Ltau/n_kept) so that E[loss] = 1 at perfect fit, independent of
       the threshold (assumes residuals are well-aligned with the kept modes).

    `covariance_estimator="full"`:
       No truncation. Builds inv_sqrt_C from the full eigendecomposition.
       Only safe when C has been regularised upstream (e.g. via Ledoit-Wolf
       shrinkage); otherwise tiny tail eigenvalues blow up the whitener.
       At perfect fit E[loss] = 1.

    Parameters
    ----------
    C                    : ndarray (L_tau, L_tau), covariance matrix
    covariance_estimator : "pca_truncated" | "full"
    variance_threshold   : float, fraction of variance retained (PCA mode only)
    """

    def __init__(self, C, covariance_estimator="pca_truncated", variance_threshold=0.99):
        super().__init__()

        C_np = np.array(C, dtype=np.float64)
        Ltau = C_np.shape[0]

        w, V = np.linalg.eigh(C_np)
        w = np.maximum(w, 0.0)

        if covariance_estimator == "pca_truncated":
            idx = np.argsort(w)[::-1]
            w_sorted = w[idx]
            V_sorted = V[:, idx]

            cumsum = np.cumsum(w_sorted)
            n_components = int(np.searchsorted(cumsum, variance_threshold * cumsum[-1]) + 1)
            n_components = min(n_components, Ltau)

            w_trunc = w_sorted[:n_components]
            V_trunc = V_sorted[:, :n_components]

            eps = 1e-12
            inv_sqrt_C = (
                V_trunc @ np.diag(1.0 / np.sqrt(w_trunc + eps)) @ V_trunc.T
                * np.sqrt(Ltau / n_components)
            )
            print(f"  ChiSquaredLoss (pca_truncated): {n_components}/{Ltau} components kept "
                  f"({100 * variance_threshold:.1f}% variance threshold), "
                  f"E[loss] → 1.0 at perfect fit")
        elif covariance_estimator == "full":
            eps = 1e-12
            n_components = Ltau
            inv_sqrt_C = V @ np.diag(1.0 / np.sqrt(w + eps)) @ V.T
            w_min = float(w.min())
            cond = float(w.max() / max(w_min, eps))
            print(f"  ChiSquaredLoss (full): no truncation, eigenvalue range "
                  f"[{w_min:.3e}, {float(w.max()):.3e}] (cond ≈ {cond:.2e}), "
                  f"E[loss] → 1.0 at perfect fit")
        else:
            raise ValueError(
                f"Unknown covariance_estimator={covariance_estimator!r}. "
                f"Expected 'pca_truncated' or 'full'."
            )

        self.Ltau = Ltau
        self.n_components = n_components
        self.covariance_estimator = covariance_estimator
        self.register_buffer("inv_sqrt_C", torch.tensor(inv_sqrt_C, dtype=torch.float32))

    def forward(self, G_pred, G_input):
        dG = G_pred - G_input
        dG_white = dG @ self.inv_sqrt_C
        loss = torch.mean(torch.sum(dG_white ** 2, dim=1) / self.Ltau)
        return loss


class Chi2FloorTransform(nn.Module):
    """Smooth-target ('floored') wrapper around the chi^2 scalar.

    Pass-through during warmup. Once an epoch's mean chi^2 first drops below
    `warmup_threshold`, irrevocably switches to a pseudo-Huber penalty
    centred at `target` (= 1.0):

        L(chi^2) = sqrt((chi^2 - target)^2 + delta^2) - delta

    Zero at target, |chi^2 - target| far from it, quadratic within +/- delta.
    Crucially the gradient pushes chi^2 *up* toward 1 when chi^2 < 1 — the
    diagnostic point of this wrapper for the chi^2 < 1 anomaly.

    Stateful: the trainer must call `notify_epoch_end(epoch_chi2_mean)` once
    per epoch with the raw (untransformed) chi^2 mean. The switch is one-way.

    The transform applies only to the value entering the gradient; raw chi^2
    is still returned by ChiSquaredLoss for logging and for the threshold test.
    """

    def __init__(self, target=1.0, delta=0.1, warmup_threshold=5.0):
        super().__init__()
        self.target = float(target)
        self.delta = float(delta)
        self.warmup_threshold = float(warmup_threshold)
        self.register_buffer("warmed_up",    torch.tensor(False))
        self.register_buffer("switch_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("epoch_count",  torch.tensor(0,  dtype=torch.long))

    def forward(self, chi2):
        if not bool(self.warmed_up.item()):
            return chi2
        d = self.delta
        return torch.sqrt((chi2 - self.target) ** 2 + d * d) - d

    def notify_epoch_end(self, epoch_chi2_mean):
        self.epoch_count += 1
        if bool(self.warmed_up.item()):
            return
        if float(epoch_chi2_mean) < self.warmup_threshold:
            self.warmed_up.fill_(True)
            self.switch_epoch.fill_(int(self.epoch_count.item()))
            print(f"  Chi2FloorTransform: warmup complete at epoch {int(self.switch_epoch.item())} "
                  f"(epoch chi^2 = {float(epoch_chi2_mean):.4f} < {self.warmup_threshold}); "
                  f"switching to floored loss with target={self.target}, delta={self.delta}")


class Chi2OneSidedBarrier(nn.Module):
    """One-sided barrier on chi^2 < 1; raw chi^2 above target.

        L(chi^2) = chi^2 + lambda * ReLU(target - chi^2)^2

    Above target the barrier is exactly zero, so the model sees the raw chi^2
    gradient and continues pulling chi^2 down toward the noise floor. Below
    target the barrier ramps up and adds a restoring force.

    Equilibrium (where dL/d(chi^2) = 0): chi^2* = target - 1/(2 * lambda_).
    Pick lambda_ large enough that chi^2* sits arbitrarily close to target:
        lambda_ = 10  -> chi^2* = target - 0.05
        lambda_ = 50  -> chi^2* = target - 0.01
        lambda_ = 500 -> chi^2* = target - 0.001

    Crucially, unlike the symmetric pseudo-Huber floor, the chi^2 gradient is
    nowhere flat — it stays at +1 above target and only drops as the barrier
    catches up below target. The other loss components cannot dominate near
    the target because chi^2 is still actively being minimised there.

    No warmup needed: at large chi^2 the barrier is identically zero, so the
    transform reduces to chi^2 itself during the descent phase.

    Stateless. notify_epoch_end is a no-op so the trainer's per-epoch hook
    can call it uniformly across transforms.
    """

    def __init__(self, lambda_=50.0, target=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)
        self.target = float(target)

    def forward(self, chi2):
        below = torch.relu(self.target - chi2)
        return chi2 + self.lambda_ * below * below

    def notify_epoch_end(self, epoch_chi2_mean):
        return


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
    chi2_transform=None,
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

    # chi2_transform reshapes only the gradient-side chi^2; chi2_val (raw) is
    # what we return for logging and what the trainer feeds back as the
    # warmup-threshold signal.
    chi2_for_total = chi2_transform(chi2_val) if chi2_transform is not None else chi2_val

    total = (
        lambda_chi2  * chi2_for_total
        + lambda_smooth * smooth_val
        + lambda_pos    * pos_val
        + alpha_kl      * kl_val
        + eta0          * neg_green_val
        + eta2          * neg_second_val
        + eta4          * neg_fourth_val
    )

    return total, chi2_val, smooth_val, pos_val, kl_val, neg_green_val, neg_second_val, neg_fourth_val
