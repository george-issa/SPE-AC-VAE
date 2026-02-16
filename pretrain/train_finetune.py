"""
Training loop for Stage 2: unsupervised fine-tuning on DQMC Green's function data.

Mirrors train_pretrain.py in structure but:
  - Batches are plain G(tau) tensors (no mu/sigma targets)
  - Uses chi-squared (covariance-weighted) reconstruction instead of MSE/std
  - Adds smoothness regularizer on predicted A(omega) from poles/residues
  - Adds positivity penalty on predicted A(omega) (penalizes negative values)
  - Keeps negative Green's penalty and negative second derivative penalty
  - Supports gradient clipping and early stopping

Loss:
  L = lambda_chi2 * L_chi2
    + lambda_smooth * L_smoothness
    + lambda_pos * L_positivity
    + alpha * L_KL
    + eta0 * L_neg_green
    + eta2 * L_neg_second_deriv
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loss2 import kl_divergence_loss, negative_green_penalty, negative_second_derivative_penalty  # type: ignore


def finetune_total_loss(
    G_recon,
    G_input,
    mu_vae,
    logvar_vae,
    poles,
    residues,
    std,
    chi2_fn,
    smoothness_fn,
    positivity_fn=None,
    lambda_chi2=1.0,
    lambda_smooth=0.01,
    lambda_pos=0.0,
    alpha_kl=0.0,
    eta0=1.0,
    eta2=1.0,
):
    """Combine all unsupervised fine-tuning losses.

    L = lambda_chi2 * chi2
      + lambda_smooth * smoothness
      + lambda_pos * positivity
      + alpha_kl * KL
      + eta0 * neg_green_penalty
      + eta2 * neg_second_derivative_penalty

    Returns
    -------
    total      : scalar tensor
    chi2_val   : scalar tensor
    smooth_val : scalar tensor
    pos_val    : scalar tensor
    kl_val     : scalar tensor
    neg_green_val    : scalar tensor
    neg_second_val   : scalar tensor
    """
    chi2_val = chi2_fn(G_recon, G_input)
    smooth_val = smoothness_fn(poles, residues)
    kl_val = kl_divergence_loss(mu_vae, logvar_vae)
    neg_green_val = negative_green_penalty(G_recon, std)
    neg_second_val = negative_second_derivative_penalty(G_recon, std)

    if positivity_fn is not None:
        pos_val = positivity_fn(poles, residues)
    else:
        pos_val = torch.tensor(0.0, device=G_recon.device)

    total = (
        lambda_chi2 * chi2_val
        + lambda_smooth * smooth_val
        + lambda_pos * pos_val
        + alpha_kl * kl_val
        + eta0 * neg_green_val
        + eta2 * neg_second_val
    )

    return total, chi2_val, smooth_val, pos_val, kl_val, neg_green_val, neg_second_val


def train_finetune(
    model,
    optimizer,
    scheduler,
    num_epochs,
    input_dim,
    train_loader,
    val_loader,
    std,
    chi2_fn,
    smoothness_fn,
    positivity_fn,
    lambda_chi2,
    lambda_smooth,
    lambda_pos,
    alpha_kl,
    eta0,
    eta2,
    device,
    out_dir,
    tag="finetune",
    patience=10,
    deterministic=False,
    grad_clip_norm=5.0,
):
    """Train the VAE with unsupervised losses on DQMC Green's function data.

    Parameters
    ----------
    model          : VariationalAutoEncoder2
    optimizer      : torch optimizer
    scheduler      : LR scheduler (CosineAnnealingWarmRestarts or ReduceLROnPlateau)
    num_epochs     : int
    input_dim      : int (L_tau)
    train_loader   : DataLoader yielding G(tau) batches (B, L_tau)
    val_loader     : DataLoader yielding G(tau) batches (B, L_tau)
    std            : tensor (L_tau,), sample std of the dataset (for neg_green, neg_second penalties)
    chi2_fn        : ChiSquaredLoss (covariance-weighted reconstruction loss)
    smoothness_fn  : SpectralSmoothnessLoss
    positivity_fn  : SpectralPositivityLoss (penalizes negative A(omega))
    lambda_chi2    : float, weight for chi-squared loss
    lambda_smooth  : float, weight for smoothness regularizer
    lambda_pos     : float, weight for spectral positivity penalty
    alpha_kl       : float, weight for KL divergence
    eta0           : float, weight for negative Green's penalty
    eta2           : float, weight for negative second derivative penalty
    device         : torch device
    out_dir        : str, output directory (must contain model/ and losses/ subdirs)
    tag            : str, identifier for saved files
    patience       : int, early stopping patience
    deterministic  : bool, if True use z=mu (no VAE sampling)
    grad_clip_norm : float, max gradient norm for clipping

    Returns
    -------
    best_val_loss : float
    best_epoch    : int
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    train_losses, val_losses = [], []
    chi2_losses, smooth_losses, pos_losses, kl_losses = [], [], [], []
    neg_green_losses, neg_second_losses = [], []
    best_val_loss = float("inf")
    counter = 0
    best_epoch = -1

    use_cosine = isinstance(scheduler, CosineAnnealingWarmRestarts)

    for epoch in range(num_epochs):

        print(f"{'-'*75} {tag} Epoch {epoch + 1} {'-'*50}")

        # ----- Training -----
        model.train()
        train_loss = 0.0
        epoch_chi2 = 0.0
        epoch_smooth = 0.0
        epoch_pos = 0.0
        epoch_kl = 0.0
        epoch_neg_green = 0.0
        epoch_neg_second = 0.0
        n_samples = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for batch_idx, batch in loop:
            B = batch.shape[0]
            batch = batch.view(B, input_dim).to(device)

            # Forward pass
            mu, logvar, z, poles, residues, Gtau_reconstructed = model(
                batch, deterministic=deterministic
            )

            # Total loss
            loss, chi2_val, smooth_val, pos_val, kl_val, neg_green_val, neg_second_val = finetune_total_loss(
                Gtau_reconstructed, batch,
                mu, logvar,
                poles, residues,
                std,
                chi2_fn, smoothness_fn, positivity_fn,
                lambda_chi2, lambda_smooth, lambda_pos, alpha_kl, eta0, eta2,
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            # Cosine annealing steps per batch
            if use_cosine:
                scheduler.step(epoch + batch_idx / len(train_loader))

            train_loss += loss.item() * B
            epoch_chi2 += chi2_val.item() * B
            epoch_smooth += smooth_val.item() * B
            epoch_pos += pos_val.item() * B
            epoch_kl += kl_val.item() * B
            epoch_neg_green += neg_green_val.item() * B
            epoch_neg_second += neg_second_val.item() * B
            n_samples += B

            loop.set_description(f"{tag} Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(
                lr=optimizer.param_groups[0]["lr"],
                chi2=chi2_val.item(),
                smooth=smooth_val.item(),
                pos=pos_val.item(),
                kl=kl_val.item(),
                total=loss.item(),
            )

        train_loss /= n_samples
        train_losses.append(train_loss)
        chi2_losses.append(epoch_chi2 / n_samples)
        smooth_losses.append(epoch_smooth / n_samples)
        pos_losses.append(epoch_pos / n_samples)
        kl_losses.append(epoch_kl / n_samples)
        neg_green_losses.append(epoch_neg_green / n_samples)
        neg_second_losses.append(epoch_neg_second / n_samples)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                B = batch.shape[0]
                batch = batch.view(B, input_dim).to(device)

                mu, logvar, z, poles, residues, Gtau_reconstructed = model(
                    batch, deterministic=deterministic
                )

                loss, _, _, _, _, _, _ = finetune_total_loss(
                    Gtau_reconstructed, batch,
                    mu, logvar,
                    poles, residues,
                    std,
                    chi2_fn, smoothness_fn, positivity_fn,
                    lambda_chi2, lambda_smooth, lambda_pos, alpha_kl, eta0, eta2,
                )

                val_loss += loss.item() * B
                n_val += B

        val_loss /= n_val
        val_losses.append(val_loss)

        # ReduceLROnPlateau steps per epoch with val_loss
        if not use_cosine:
            scheduler.step(val_loss)

        print(
            f"{tag} Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e} "
            f"| chi2={chi2_losses[-1]:.4e} smooth={smooth_losses[-1]:.4e} "
            f"pos={pos_losses[-1]:.4e} kl={kl_losses[-1]:.4e} "
            f"neg_green={neg_green_losses[-1]:.4e} neg_second={neg_second_losses[-1]:.4e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), f"{out_dir}/model/best_model_{tag}.pth")
            print(f"  Best {tag} model saved at epoch {epoch+1} (val_loss={best_val_loss:.4e})")
        else:
            counter += 1
            if counter >= patience:
                print(f"  {tag} early stopping triggered at epoch {epoch+1}!")
                break

        # Save loss curves every epoch
        np.save(f"{out_dir}/losses/train_losses_{tag}.npy", np.array(train_losses))
        np.save(f"{out_dir}/losses/val_losses_{tag}.npy", np.array(val_losses))
        np.save(f"{out_dir}/losses/chi2_losses_{tag}.npy", np.array(chi2_losses))
        np.save(f"{out_dir}/losses/smooth_losses_{tag}.npy", np.array(smooth_losses))
        np.save(f"{out_dir}/losses/pos_losses_{tag}.npy", np.array(pos_losses))
        np.save(f"{out_dir}/losses/kl_losses_{tag}.npy", np.array(kl_losses))
        np.save(f"{out_dir}/losses/neg_green_losses_{tag}.npy", np.array(neg_green_losses))
        np.save(f"{out_dir}/losses/neg_second_losses_{tag}.npy", np.array(neg_second_losses))

    print(f"{tag} training completed. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")
    return best_val_loss, best_epoch
