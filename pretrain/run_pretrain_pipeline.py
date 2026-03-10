"""
Main pretraining pipeline orchestrator.

Stage 1: Pretrain VAE on synthetic single-Gaussian spectral functions.
         The covariance matrix is loaded from the Stage 2 target DQMC dataset
         so that the pretraining chi-squared loss uses the same noise structure
         the model will see during fine-tuning.

Stage 2 (optional): Fine-tune on any target DQMC dataset using chi-squared
         loss and variance-weighted negativity penalties.

To switch target datasets, change SPECTRAL_TYPE / INPUT_ID / NOISE_S / NOISE_XI
in the STAGE 2 section below. The covariance (and thus Stage 1 noise) updates
automatically.

Usage:
    python pretrain/run_pretrain_pipeline.py
"""

import os
import sys
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model2_silu import VariationalAutoEncoder2  # type: ignore
from pretrain.plot_results import (  # type: ignore
    plot_pretrain_eval, plot_loss_curves, plot_finetune_eval,
)
from pretrain.synthetic_data import (  # type: ignore
    SavedSyntheticDataset,
    SyntheticGaussianDataset,
    generate_and_save,
    load_covariance_from_dqmc,
    cholesky_sqrt,
    compute_covariance_from_bins,
)
from pretrain.pretrain_losses import (  # type: ignore
    KLDivergenceLoss,
    SpectralMSELoss,
    ChiSquaredLoss,
    SpectralSmoothnessLoss,
    SpectralPositivityLoss,
    SpectralMomentLoss,
    NegativeGreenPenalty,
    NegativeSecondDerivativePenalty,
    NegativeFourthDerivativePenalty,
    finetune_total_loss,
    spectral_from_poles,
)
from pretrain.train_pretrain import train_pretrain  # type: ignore
from pretrain.train_finetune import train_finetune  # type: ignore

from data_process import GreenFunctionDataset  # type: ignore
from utils import MakeOutPath  # type: ignore

# ==========================================================================
# CONFIGURATION
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_PATH = "/Users/georgeissa/Documents/AC/SPE-AC-VAE"

# --- Model ---
NUM_POLES = 4
BETA = 10.0
DTAU = 0.05
INPUT_DIM = int(BETA / DTAU)  # Must match greens input dimension
N_NODES = 512
BATCH_SIZE = 32

# --------------------------------------------------------------------------
# STAGE 2 — Target DQMC dataset
# Change these to switch between gaussian, gaussian_double, semicircle, etc.
# The covariance loaded from this dataset is also used in Stage 1.
# --------------------------------------------------------------------------
SPECTRAL_TYPE = "gaussian_double"
NOISE_S = 1e-04
NOISE_XI = 0.5
INPUT_ID = "inputs-8"

DQMC_DATA_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    "spectral_input.csv"
)

# --------------------------------------------------------------------------
# STAGE 1 — Pretraining (always synthetic single-Gaussian data)
# --------------------------------------------------------------------------

# Synthetic data generation
N_SAMPLES = 1000             # Total synthetic samples to generate
MU_MAX = 0.5                 # Peak centers: Uniform(-MU_MAX, +MU_MAX)
SIGMA_MIN = 0.3              # Peak widths: LogUniform(SIGMA_MIN, SIGMA_MAX)
SIGMA_MAX = 3.0
GENERATE_FRESH = True        # If True, generate new data; if False, load from disk
USE_ON_THE_FLY = False       # If True, use on-the-fly dataset (infinite variety)
SYNTHETIC_DATA_DIR = os.path.join(MAIN_PATH, "Data", "datasets", "synthetic-gaussian-pretrain")

# Pretraining hyperparameters
PRE_EPOCHS = 400             # Train for all epochs — no early stopping (weight initialisation goal)
PRE_LR = 1e-3
PRE_LAMBDA_SPEC = 1.0
PRE_LAMBDA_MOMENT = 1.0
PRE_LAMBDA_SMOOTH = 1e-3
PRE_LAMBDA_CHI2 = 0.1
PRE_ALPHA_KL = 0.0
PRE_PATIENCE = PRE_EPOCHS + 1  # Early stopping disabled
PRE_LR_FACTOR = 0.5
PRE_LR_PATIENCE = 20
PRE_LR_MIN = 1e-6

# Spectral loss parameters
SPECTRAL_W = 6.0
SPECTRAL_N_GL = 512
SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -8.0
SMOOTHNESS_WMAX = 8.0

# --------------------------------------------------------------------------
# STAGE 2 — Fine-tuning hyperparameters
# --------------------------------------------------------------------------

DO_FINETUNE = True
FINETUNE_EPOCHS = 200
FINETUNE_LR = 5e-4
FINETUNE_PATIENCE = FINETUNE_EPOCHS + 1  # Disabled — train for all epochs
FINETUNE_KL_ANNEAL_EPOCHS = 75
FINETUNE_LAMBDA_CHI2 = 1.0
FINETUNE_LAMBDA_SMOOTH = 0.0
FINETUNE_LAMBDA_POS = 0.0
FINETUNE_ALPHA_KL = 0.0
FINETUNE_ETA0 = 1.0          # G(tau) >= 0 penalty — variance-weighted via diag(C)
FINETUNE_ETA2 = 1.0          # G''(tau) >= 0 penalty
FINETUNE_ETA4 = 1.0          # G''''(tau) >= 0 penalty
FINETUNE_LR_FACTOR = 0.25
FINETUNE_LR_PATIENCE = 10
FINETUNE_LR_MIN = 1e-6
sID = "pretrained-v2s"

# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Model: NUM_POLES={NUM_POLES}, INPUT_DIM={INPUT_DIM}")
    print(f"Target dataset: {DQMC_DATA_PATH}")

    # ------------------------------------------------------------------
    # Load covariance once — shared by Stage 1 and Stage 2
    # ------------------------------------------------------------------
    print(f"\nLoading covariance from: {DQMC_DATA_PATH}")
    C = load_covariance_from_dqmc(DQMC_DATA_PATH)
    sqrt_C = cholesky_sqrt(C)

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    model = VariationalAutoEncoder2(
        input_dim=INPUT_DIM,
        num_poles=NUM_POLES,
        beta=BETA,
        N_nodes=N_NODES,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # STAGE 1: Pretraining on synthetic Gaussians
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: PRETRAINING ON SYNTHETIC GAUSSIAN DATA")
    print("=" * 80 + "\n")

    # --- Generate / load synthetic data ---
    if GENERATE_FRESH and not USE_ON_THE_FLY:
        print("Generating fresh synthetic dataset...")
        generate_and_save(
            N=N_SAMPLES,
            mu_max=MU_MAX,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            beta=BETA,
            dtau=DTAU,
            output_dir=SYNTHETIC_DATA_DIR,
            covariance_source=DQMC_DATA_PATH,
            use_quadrature=False,
            N_gl=256,
            omega_max=20.0,
            seed=42,
        )

    if USE_ON_THE_FLY:
        print("Using on-the-fly synthetic dataset")
        dataset = SyntheticGaussianDataset(
            N_samples=N_SAMPLES,
            mu_max=MU_MAX,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            beta=BETA,
            dtau=DTAU,
            sqrt_C=sqrt_C,
        )
    else:
        print(f"Loading saved synthetic dataset from {SYNTHETIC_DATA_DIR}")
        dataset = SavedSyntheticDataset(SYNTHETIC_DATA_DIR)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = train_loader  # Same data; ReduceLROnPlateau monitors eval-mode training loss
    print(f"Dataset: {len(dataset)} samples, {len(train_loader)} batches/epoch")

    # --- Loss modules (using shared C) ---
    spectral_mse_fn = SpectralMSELoss(W=SPECTRAL_W, N_gl=SPECTRAL_N_GL).to(DEVICE)
    chi2_fn = ChiSquaredLoss(C).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    moment_fn = SpectralMomentLoss(W=SPECTRAL_W, N_gl=SPECTRAL_N_GL).to(DEVICE)

    # --- Output directory ---
    pre_out_dir = os.path.join(
        MAIN_PATH, "out",
        f"pretrain_synthetic_numpoles{NUM_POLES}"
    )
    MakeOutPath(pre_out_dir)

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=PRE_LR, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=PRE_LR_FACTOR,
        patience=PRE_LR_PATIENCE, min_lr=PRE_LR_MIN
    )

    # --- Train ---
    best_val_loss, best_epoch = train_pretrain(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=PRE_EPOCHS,
        input_dim=INPUT_DIM,
        train_loader=train_loader,
        val_loader=val_loader,
        spectral_mse_fn=spectral_mse_fn,
        chi2_fn=chi2_fn,
        smoothness_fn=smoothness_fn,
        lambda_chi2=PRE_LAMBDA_CHI2,
        lambda_spec=PRE_LAMBDA_SPEC,
        lambda_smooth=PRE_LAMBDA_SMOOTH,
        alpha_kl=PRE_ALPHA_KL,
        device=DEVICE,
        out_dir=pre_out_dir,
        tag="pretrain",
        patience=PRE_PATIENCE,
        deterministic=False,
        moment_fn=moment_fn,
        lambda_moment=PRE_LAMBDA_MOMENT,
    )

    print(f"\nPretraining done. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")

    # Load best pretrained model
    model.load_state_dict(torch.load(f"{pre_out_dir}/model/best_model_pretrain.pth", weights_only=True))

    # ------------------------------------------------------------------
    # Evaluate pretraining
    # ------------------------------------------------------------------
    print("\n--- Pretraining evaluation ---")
    model.eval()

    eval_loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=False)
    with torch.no_grad():
        batch = next(iter(eval_loader))
        G_tilde, mu_targets, sigma_targets = batch
        G_tilde = G_tilde.to(DEVICE)
        mu_vae, logvar_vae, z, poles, residues, G_recon = model(G_tilde)

    torch.save({
        "G_input": G_tilde.cpu(),
        "G_recon": G_recon.cpu(),
        "poles": poles.cpu(),
        "residues": residues.cpu(),
        "mu_targets": mu_targets,
        "sigma_targets": sigma_targets,
        "beta": BETA,
        "Ltau": INPUT_DIM,
    }, f"{pre_out_dir}/pretrain_eval.pt")
    print(f"Pretraining evaluation saved to {pre_out_dir}/pretrain_eval.pt")

    plot_pretrain_eval(f"{pre_out_dir}/pretrain_eval.pt")
    plot_loss_curves(f"{pre_out_dir}/losses", tag="pretrain",
                     save_path=f"{pre_out_dir}/plots/loss_curves_pretrain.pdf")

    # ------------------------------------------------------------------
    # STAGE 2: Fine-tuning on target dataset (optional)
    # ------------------------------------------------------------------
    if not DO_FINETUNE:
        print("\nSkipping fine-tuning (DO_FINETUNE=False).")
        return

    print("\n" + "=" * 80)
    print(f"STAGE 2: FINE-TUNING ON {SPECTRAL_TYPE.upper()} DATA")
    print("=" * 80 + "\n")

    tag = "finetune"

    ft_out_dir = os.path.join(
        MAIN_PATH, "out",
        f"finetune_{SPECTRAL_TYPE}_numpoles{NUM_POLES}_s{NOISE_S:.0e}_xi{NOISE_XI}-{sID}"
    )
    MakeOutPath(ft_out_dir)

    # --- Load full dataset — train on all samples, no held-out split ---
    print(f"Loading dataset from: {DQMC_DATA_PATH}")
    ft_dataset = GreenFunctionDataset(file_path=DQMC_DATA_PATH)
    ft_train_loader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    ft_val_loader = ft_train_loader  # Scheduler monitors eval-mode training loss
    print(f"  Full dataset: {len(ft_dataset)} samples, {len(ft_train_loader)} batches/epoch")

    # --- Build fine-tuning loss modules (same C as Stage 1) ---
    ft_kl_fn         = KLDivergenceLoss().to(DEVICE)
    ft_chi2_fn       = ChiSquaredLoss(C).to(DEVICE)
    ft_smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    ft_positivity_fn = SpectralPositivityLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    ft_neg_green_fn  = NegativeGreenPenalty(C).to(DEVICE)
    ft_neg_second_fn = NegativeSecondDerivativePenalty(C).to(DEVICE)
    ft_neg_fourth_fn = NegativeFourthDerivativePenalty(C).to(DEVICE)

    # --- Optimizer ---
    ft_optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.0)
    ft_scheduler = ReduceLROnPlateau(
        ft_optimizer, mode="min", factor=FINETUNE_LR_FACTOR,
        patience=FINETUNE_LR_PATIENCE, min_lr=FINETUNE_LR_MIN
    )

    # --- Fine-tune ---
    print(f"\nLoss weights: lambda_chi2={FINETUNE_LAMBDA_CHI2}, lambda_smooth={FINETUNE_LAMBDA_SMOOTH}, "
          f"lambda_pos={FINETUNE_LAMBDA_POS}, alpha_kl={FINETUNE_ALPHA_KL}, "
          f"eta0={FINETUNE_ETA0}, eta2={FINETUNE_ETA2}, eta4={FINETUNE_ETA4}")
    print(f"Scheduler: ReduceLROnPlateau(factor={FINETUNE_LR_FACTOR}, patience={FINETUNE_LR_PATIENCE})")
    print(f"Training on full dataset ({len(ft_dataset)} samples), early stopping disabled.")

    best_ft_val, best_ft_epoch = train_finetune(
        model=model,
        optimizer=ft_optimizer,
        scheduler=ft_scheduler,
        num_epochs=FINETUNE_EPOCHS,
        input_dim=INPUT_DIM,
        train_loader=ft_train_loader,
        val_loader=ft_val_loader,
        kl_fn=ft_kl_fn,
        chi2_fn=ft_chi2_fn,
        smoothness_fn=ft_smoothness_fn,
        neg_green_fn=ft_neg_green_fn,
        neg_second_fn=ft_neg_second_fn,
        neg_fourth_fn=ft_neg_fourth_fn,
        positivity_fn=ft_positivity_fn,
        lambda_chi2=FINETUNE_LAMBDA_CHI2,
        lambda_smooth=FINETUNE_LAMBDA_SMOOTH,
        lambda_pos=FINETUNE_LAMBDA_POS,
        alpha_kl=FINETUNE_ALPHA_KL,
        eta0=FINETUNE_ETA0,
        eta2=FINETUNE_ETA2,
        eta4=FINETUNE_ETA4,
        device=DEVICE,
        out_dir=ft_out_dir,
        tag=tag,
        patience=FINETUNE_PATIENCE,
        kl_anneal_epochs=FINETUNE_KL_ANNEAL_EPOCHS,
    )

    print(f"\nFine-tuning done. Best val loss: {best_ft_val:.4e} at epoch {best_ft_epoch+1}")

    # Load best fine-tuned model
    model.load_state_dict(torch.load(f"{ft_out_dir}/model/best_model_{tag}.pth", weights_only=True))

    # ------------------------------------------------------------------
    # Full-dataset evaluation
    # ------------------------------------------------------------------
    print("\n--- Full-dataset evaluation ---")
    model.eval()
    eval_loss = 0.0
    G_input_all, G_recon_all = [], []
    poles_all, residues_all = [], []

    with torch.no_grad():
        for batch in ft_train_loader:
            B = batch.shape[0]
            batch = batch.view(B, INPUT_DIM).to(DEVICE)

            mu_vae, logvar_vae, z, poles, residues, G_recon = model(batch)

            loss, _, _, _, _, _, _, _ = finetune_total_loss(
                G_recon, batch,
                mu_vae, logvar_vae,
                poles, residues,
                ft_kl_fn, ft_chi2_fn, ft_smoothness_fn,
                ft_neg_green_fn, ft_neg_second_fn, ft_neg_fourth_fn,
                ft_positivity_fn,
                FINETUNE_LAMBDA_CHI2, FINETUNE_LAMBDA_SMOOTH, FINETUNE_LAMBDA_POS,
                FINETUNE_ALPHA_KL, FINETUNE_ETA0, FINETUNE_ETA2, FINETUNE_ETA4,
            )
            eval_loss += loss.item() * B

            G_input_all.append(batch.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())

    eval_loss /= len(ft_dataset)
    print(f"Final Loss: {eval_loss:.4e}")

    G_input_all  = torch.cat(G_input_all,  dim=0)
    G_recon_all  = torch.cat(G_recon_all,  dim=0)
    poles_all    = torch.cat(poles_all,    dim=0)
    residues_all = torch.cat(residues_all, dim=0)

    # Multi-sample MC evaluation
    print("\n--- Multi-sample MC evaluation (N_MC=50) ---")
    N_MC = 50
    omega_eval_grid = torch.linspace(SMOOTHNESS_WMIN, SMOOTHNESS_WMAX, SMOOTHNESS_NW)
    omega_eval_grid_dev = omega_eval_grid.to(DEVICE)
    G_input_tensor = G_input_all.to(DEVICE)

    spectral_samples = []
    with torch.no_grad():
        for _ in range(N_MC):
            _, _, _, poles_mc, residues_mc, _ = model(G_input_tensor, deterministic=False)
            A = spectral_from_poles(poles_mc, residues_mc, omega_eval_grid_dev)
            spectral_samples.append(A.cpu())

    A_samples = torch.stack(spectral_samples, dim=0)
    A_mean = A_samples.mean(0)
    A_std  = A_samples.std(0)
    print(f"MC spectral mean shape: {A_mean.shape}, std shape: {A_std.shape}")

    torch.save({
        "inputs":       G_input_all,
        "recon":        G_recon_all,
        "poles":        poles_all,
        "residues":     residues_all,
        "inputs_avg":   G_input_all.mean(dim=0),
        "recon_avg":    G_recon_all.mean(dim=0),
        "poles_avg":    poles_all.mean(dim=0),
        "residues_avg": residues_all.mean(dim=0),
        "A_mean":       A_mean,
        "A_std":        A_std,
        "omega_eval_grid":      omega_eval_grid,
        "beta":                 BETA,
        "Ltau":                 INPUT_DIM,
        "spectral_input_path":  SPECTRAL_INPUT_PATH,
        "noise_var":    float(np.mean(np.diag(np.array(C)))),
        "n_mc":         N_MC,
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved to {ft_out_dir}/summary.pt")
    print(f"Poles mean: {poles_all.mean(dim=0)}")
    print(f"Residues mean: {residues_all.mean(dim=0)}")

    plot_finetune_eval(f"{ft_out_dir}/summary.pt")
    plot_loss_curves(f"{ft_out_dir}/losses", tag=tag,
                     save_path=f"{ft_out_dir}/plots/loss_curves_{tag}.pdf")


if __name__ == "__main__":
    main()
