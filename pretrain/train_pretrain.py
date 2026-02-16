"""
Training loop for VAE pretraining on synthetic Gaussian data.

Similar to train_func.py but handles (G_tilde, mu, sigma) batches
and uses the pretraining loss functions.
"""

import numpy as np
import torch
from tqdm import tqdm

from pretrain.pretrain_losses import pretrain_total_loss  # type: ignore


def train_pretrain(
    model,
    optimizer,
    scheduler,
    num_epochs,
    input_dim,
    train_loader,
    val_loader,
    spectral_mse_fn,
    chi2_fn,
    smoothness_fn,
    lambda_chi2,
    lambda_spec,
    lambda_smooth,
    alpha_kl,
    device,
    out_dir,
    tag="pretrain",
    patience=10,
    deterministic=False,
    grad_clip_norm=5.0,
):
    """Train the VAE with pretraining losses on synthetic data.

    Parameters
    ----------
    model          : VariationalAutoEncoder2_FT
    optimizer      : torch optimizer
    scheduler      : LR scheduler
    num_epochs     : int
    input_dim      : int (L_tau)
    train_loader   : DataLoader yielding (G_tilde, mu, sigma) batches
    val_loader     : DataLoader yielding (G_tilde, mu, sigma) batches
    spectral_mse_fn: SpectralMSELoss
    chi2_fn        : ChiSquaredLoss
    smoothness_fn  : SpectralSmoothnessLoss
    lambda_chi2    : float
    lambda_spec    : float
    lambda_smooth  : float
    alpha_kl       : float
    device         : torch device
    out_dir        : str
    tag            : str
    patience       : int, early stopping patience
    deterministic  : bool, if True use z=mu (no VAE sampling); use when alpha_kl=0
    grad_clip_norm : float, max gradient norm for clipping

    Returns
    -------
    best_val_loss : float
    best_epoch    : int
    """
    train_losses, val_losses = [], []
    chi2_losses, spec_losses, smooth_losses, kl_losses = [], [], [], []
    best_val_loss = float("inf")
    counter = 0
    best_epoch = -1

    for epoch in range(num_epochs):

        print(f"{'-'*75} {tag} Epoch {epoch + 1} {'-'*50}")

        # ----- Training -----
        model.train()
        train_loss = 0.0
        epoch_chi2 = 0.0
        epoch_spec = 0.0
        epoch_smooth = 0.0
        epoch_kl = 0.0
        n_samples = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for _, batch in loop:
            G_tilde, mu_targets, sigma_targets = batch
            B = G_tilde.shape[0]

            G_tilde = G_tilde.view(B, input_dim).to(device)
            mu_targets = mu_targets.to(device)
            sigma_targets = sigma_targets.to(device)

            # Forward pass
            mu_vae, logvar_vae, z, poles, residues, G_recon = model(G_tilde, deterministic=deterministic)

            # Combined loss
            loss, chi2_val, spec_val, smooth_val, kl_val = pretrain_total_loss(
                G_recon, G_tilde,
                mu_vae, logvar_vae,
                poles, residues,
                mu_targets, sigma_targets,
                spectral_mse_fn, chi2_fn, smoothness_fn,
                lambda_chi2, lambda_spec, lambda_smooth, alpha_kl,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            train_loss += loss.item() * B
            epoch_chi2 += chi2_val.item() * B
            epoch_spec += spec_val.item() * B
            epoch_smooth += smooth_val.item() * B
            epoch_kl += kl_val.item() * B
            n_samples += B

            loop.set_description(f"{tag} Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(
                lr=optimizer.param_groups[0]["lr"],
                chi2=chi2_val.item(),
                spec=spec_val.item(),
                smooth=smooth_val.item(),
                kl=kl_val.item(),
                total=loss.item(),
            )

        train_loss /= n_samples
        train_losses.append(train_loss)
        chi2_losses.append(epoch_chi2 / n_samples)
        spec_losses.append(epoch_spec / n_samples)
        smooth_losses.append(epoch_smooth / n_samples)
        kl_losses.append(epoch_kl / n_samples)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                G_tilde, mu_targets, sigma_targets = batch
                B = G_tilde.shape[0]

                G_tilde = G_tilde.view(B, input_dim).to(device)
                mu_targets = mu_targets.to(device)
                sigma_targets = sigma_targets.to(device)

                mu_vae, logvar_vae, z, poles, residues, G_recon = model(G_tilde, deterministic=deterministic)

                loss, _, _, _, _ = pretrain_total_loss(
                    G_recon, G_tilde,
                    mu_vae, logvar_vae,
                    poles, residues,
                    mu_targets, sigma_targets,
                    spectral_mse_fn, chi2_fn, smoothness_fn,
                    lambda_chi2, lambda_spec, lambda_smooth, alpha_kl,
                )

                val_loss += loss.item() * B
                n_val += B

        val_loss /= n_val
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"{tag} Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e} "
            f"| chi2={chi2_losses[-1]:.4e} spec={spec_losses[-1]:.4e} "
            f"smooth={smooth_losses[-1]:.4e} kl={kl_losses[-1]:.4e}"
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

        # Save loss curves
        np.save(f"{out_dir}/losses/train_losses_{tag}.npy", np.array(train_losses))
        np.save(f"{out_dir}/losses/val_losses_{tag}.npy", np.array(val_losses))
        np.save(f"{out_dir}/losses/chi2_losses_{tag}.npy", np.array(chi2_losses))
        np.save(f"{out_dir}/losses/spec_losses_{tag}.npy", np.array(spec_losses))
        np.save(f"{out_dir}/losses/smooth_losses_{tag}.npy", np.array(smooth_losses))
        np.save(f"{out_dir}/losses/kl_losses_{tag}.npy", np.array(kl_losses))

    print(f"{tag} training completed. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")
    return best_val_loss, best_epoch
