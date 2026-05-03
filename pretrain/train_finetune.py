"""
Training loop for Stage 2: unsupervised fine-tuning on DQMC Green's function data.

Loss:
  L = lambda_chi2  * chi2                     (covariance-whitened reconstruction)
    + lambda_smooth * smoothness              (second-derivative regularizer on A(omega))
    + lambda_pos    * positivity              (penalizes negative A(omega))
    + alpha_kl      * KL                     (latent prior regularization)
    + eta0          * neg_green_penalty      (G(tau) >= 0, variance-weighted)
    + eta2          * neg_second_penalty     (G''(tau) >= 0, variance-weighted)
    + eta4          * neg_fourth_penalty     (G''''(tau) >= 0, variance-weighted)

All negativity penalties use variance weighting from diag(C) — the per-tau
DQMC noise variance — matching Ben's cohensbw/SPE-VAE-AC implementation.
"""

import numpy as np
import torch
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pretrain.pretrain_losses import finetune_total_loss  # type: ignore


def train_finetune(
    model,
    optimizer,
    scheduler,
    num_epochs,
    input_dim,
    train_loader,
    val_loader,
    kl_fn,
    chi2_fn,
    smoothness_fn,
    neg_green_fn,
    neg_second_fn,
    neg_fourth_fn,
    positivity_fn,
    lambda_chi2,
    lambda_smooth,
    lambda_pos,
    alpha_kl,
    eta0,
    eta2,
    eta4,
    device,
    out_dir,
    tag="finetune",
    patience=10,
    deterministic=False,
    grad_clip_norm=5.0,
    kl_anneal_epochs=50,
    chi2_transform=None,
):
    """Train the VAE with unsupervised losses on DQMC Green's function data.

    Parameters
    ----------
    model         : VariationalAutoEncoder2
    optimizer     : torch optimizer
    scheduler     : LR scheduler (ReduceLROnPlateau or similar)
    num_epochs    : int
    input_dim     : int (L_tau)
    train_loader  : DataLoader — full dataset (no held-out split)
    val_loader    : DataLoader — same as train_loader; scheduler monitors this
    kl_fn         : KLDivergenceLoss
    chi2_fn       : ChiSquaredLoss
    smoothness_fn : SpectralSmoothnessLoss
    neg_green_fn  : NegativeGreenPenalty   (variance-weighted, diag(C))
    neg_second_fn : NegativeSecondDerivativePenalty
    neg_fourth_fn : NegativeFourthDerivativePenalty
    positivity_fn : SpectralPositivityLoss (or None)
    lambda_chi2   : float
    lambda_smooth : float
    lambda_pos    : float
    alpha_kl      : float
    eta0          : float — weight for G(tau) < 0 penalty
    eta2          : float — weight for G''(tau) < 0 penalty
    eta4          : float — weight for G''''(tau) < 0 penalty
    device        : torch.device
    out_dir       : str, must contain model/ and losses/ subdirectories
    tag           : str, identifier for saved files
    patience      : int, early-stopping patience (set > num_epochs to disable)
    deterministic : bool, if True use z = mu (no sampling)
    grad_clip_norm: float, max gradient norm for clipping
    kl_anneal_epochs : int, ramp KL weight from 0 to alpha_kl over this many epochs

    Returns
    -------
    best_val_loss : float
    best_epoch    : int
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    train_losses, val_losses = [], []
    chi2_losses, smooth_losses, pos_losses, kl_losses = [], [], [], []
    neg_green_losses, neg_second_losses, neg_fourth_losses = [], [], []

    best_val_loss = float("inf")
    counter = 0
    best_epoch = -1

    # Running EMA of successful-step loss, used to catch anomalous batches
    # before their gradients corrupt the weights. Persists across epochs.
    ema_loss = None
    EMA_BETA = 0.95
    LOSS_RATIO_THRESHOLD = 100.0   # skip if loss > 100 * ema_loss
    ABS_LOSS_FLOOR = 100.0         # floor so early batches are not over-sensitive

    for epoch in range(num_epochs):

        print(f"{'-'*75} {tag} Epoch {epoch + 1} {'-'*50}")

        # KL annealing: ramp from 0 → alpha_kl over kl_anneal_epochs
        kl_weight = alpha_kl * min(1.0, (epoch + 1) / max(1, kl_anneal_epochs))

        # ----- Training -----
        model.train()
        train_loss = 0.0
        epoch_chi2 = epoch_smooth = epoch_pos = epoch_kl = 0.0
        epoch_neg_green = epoch_neg_second = epoch_neg_fourth = 0.0
        n_samples = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for batch_idx, batch in loop:
            B = batch.shape[0]
            batch = batch.view(B, input_dim).to(device)

            mu, logvar, z, poles, residues, Gtau_reconstructed = model(
                batch, deterministic=deterministic
            )

            loss, chi2_val, smooth_val, pos_val, kl_val, neg_green_val, neg_second_val, neg_fourth_val = finetune_total_loss(
                Gtau_reconstructed, batch,
                mu, logvar,
                poles, residues,
                kl_fn, chi2_fn, smoothness_fn,
                neg_green_fn, neg_second_fn, neg_fourth_fn,
                positivity_fn,
                lambda_chi2, lambda_smooth, lambda_pos, kl_weight,
                eta0, eta2, eta4,
                chi2_transform=chi2_transform,
            )

            optimizer.zero_grad()

            # Pre-backward: anomaly guard on the loss value itself.
            loss_val = loss.item()
            ema_threshold = (
                max(ABS_LOSS_FLOOR, LOSS_RATIO_THRESHOLD * ema_loss)
                if ema_loss is not None else float("inf")
            )
            if not torch.isfinite(loss) or loss_val > ema_threshold:
                ema_str = f"{ema_loss:.3e}" if ema_loss is not None else "N/A"
                print(f"  WARNING: loss anomaly ({loss_val:.3e}, ema={ema_str}) at epoch {epoch+1}, skipping update")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            # Post-backward: catch non-finite gradients before they hit the optimizer state.
            if not torch.isfinite(grad_norm):
                print(f"  WARNING: non-finite grad norm ({grad_norm}) at epoch {epoch+1}, skipping update")
                optimizer.zero_grad()
                continue

            optimizer.step()

            # Update EMA only on accepted steps so a bad batch cannot pull the threshold up.
            ema_loss = loss_val if ema_loss is None else EMA_BETA * ema_loss + (1 - EMA_BETA) * loss_val

            train_loss      += loss.item()          * B
            epoch_chi2      += chi2_val.item()      * B
            epoch_smooth    += smooth_val.item()    * B
            epoch_pos       += pos_val.item()       * B
            epoch_kl        += kl_val.item()        * B
            epoch_neg_green += neg_green_val.item() * B
            epoch_neg_second+= neg_second_val.item()* B
            epoch_neg_fourth+= neg_fourth_val.item()* B
            n_samples += B

            loop.set_description(f"{tag} Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(
                lr=optimizer.param_groups[0]["lr"],
                chi2=chi2_val.item(),
                kl=kl_val.item(),
                total=loss.item(),
            )

        if n_samples == 0:
            print(f"  All batches skipped at epoch {epoch+1} — model diverged. Stopping; best checkpoint preserved.")
            break

        train_loss       /= n_samples
        train_losses.append(train_loss)
        chi2_losses.append(epoch_chi2       / n_samples)
        smooth_losses.append(epoch_smooth   / n_samples)
        pos_losses.append(epoch_pos         / n_samples)
        kl_losses.append(epoch_kl           / n_samples)
        neg_green_losses.append(epoch_neg_green  / n_samples)
        neg_second_losses.append(epoch_neg_second/ n_samples)
        neg_fourth_losses.append(epoch_neg_fourth/ n_samples)

        if chi2_transform is not None:
            chi2_transform.notify_epoch_end(chi2_losses[-1])

        # ----- Validation (eval mode on val_loader = train_loader) -----
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                B = batch.shape[0]
                batch = batch.view(B, input_dim).to(device)

                # Validation always uses z = mu (no reparameterization noise) so
                # val_loss is a clean signal for the scheduler / early stopping.
                mu, logvar, z, poles, residues, Gtau_reconstructed = model(
                    batch, deterministic=True
                )

                loss, _, _, _, _, _, _, _ = finetune_total_loss(
                    Gtau_reconstructed, batch,
                    mu, logvar,
                    poles, residues,
                    kl_fn, chi2_fn, smoothness_fn,
                    neg_green_fn, neg_second_fn, neg_fourth_fn,
                    positivity_fn,
                    lambda_chi2, lambda_smooth, lambda_pos, alpha_kl,
                    eta0, eta2, eta4,
                    chi2_transform=chi2_transform,
                )

                val_loss += loss.item() * B
                n_val += B

        val_loss /= n_val
        val_losses.append(val_loss)

        # Scheduler step (ReduceLROnPlateau monitors val/train loss)
        if scheduler is None:
            pass
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(
            f"{tag} Train: {train_loss:.4e}  Val: {val_loss:.4e} "
            f"| chi2={chi2_losses[-1]:.4e}  kl={kl_losses[-1]:.4e} "
            f"| smooth={smooth_losses[-1]:.4e}  pos={pos_losses[-1]:.4e} "
            f"| neg_G={neg_green_losses[-1]:.4e}  neg_G''={neg_second_losses[-1]:.4e} "
            f"neg_G''''={neg_fourth_losses[-1]:.4e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), f"{out_dir}/model/best_model_{tag}.pth")
            print(f"  Best model saved at epoch {epoch+1} (val_loss={best_val_loss:.4e})")
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1}.")
                break

        # Save loss curves
        np.save(f"{out_dir}/losses/train_losses_{tag}.npy",     np.array(train_losses))
        np.save(f"{out_dir}/losses/val_losses_{tag}.npy",       np.array(val_losses))
        np.save(f"{out_dir}/losses/chi2_losses_{tag}.npy",      np.array(chi2_losses))
        np.save(f"{out_dir}/losses/smooth_losses_{tag}.npy",    np.array(smooth_losses))
        np.save(f"{out_dir}/losses/pos_losses_{tag}.npy",       np.array(pos_losses))
        np.save(f"{out_dir}/losses/kl_losses_{tag}.npy",        np.array(kl_losses))
        np.save(f"{out_dir}/losses/neg_green_losses_{tag}.npy", np.array(neg_green_losses))
        np.save(f"{out_dir}/losses/neg_second_losses_{tag}.npy",np.array(neg_second_losses))
        np.save(f"{out_dir}/losses/neg_fourth_losses_{tag}.npy",np.array(neg_fourth_losses))

    print(f"{tag} training done. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")
    return best_val_loss, best_epoch
