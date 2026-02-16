"""
Standalone Stage 2: fine-tune a pretrained VAE on DQMC Green's function data.

Loads a pretrained checkpoint and trains unsupervised on a target dataset
(e.g., 1000 samples of G(tau) from the same spectral function).

Loss:
  L = lambda_chi2 * chi2  (covariance-weighted reconstruction)
    + lambda_smooth * smoothness  (on predicted A(omega))
    + alpha * KL
    + eta0 * neg_green_penalty
    + eta2 * neg_second_derivative_penalty

Usage:
    python pretrain/run_finetune.py
"""

import os
import sys
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model2 import VariationalAutoEncoder2  # type: ignore
from pretrain.plot_results import (  # type: ignore
    plot_finetune_eval, plot_loss_curves,
)
from pretrain.synthetic_data import (  # type: ignore
    load_covariance_from_dqmc,
)
from pretrain.pretrain_losses import (  # type: ignore
    ChiSquaredLoss,
    SpectralSmoothnessLoss,
    SpectralPositivityLoss,
)
from pretrain.train_finetune import train_finetune  # type: ignore

from data_process import GreenFunctionDataset  # type: ignore
from utils import MakeOutPath, STD, TrainTestSplit, LoadData, PrintDataSize  # type: ignore

# ==========================================================================
# CONFIGURATION
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_PATH = "/Users/georgeissa/Documents/AC"

# --- Model (must match pretraining) ---
NUM_POLES = 4
LATENT_DIM = 4 * NUM_POLES - 2
HIDDEN_DIM1 = 8 * LATENT_DIM
HIDDEN_DIM2 = LATENT_DIM
HIDDEN_DIM3 = 2 * LATENT_DIM
HIDDEN_DIM4 = 4 * LATENT_DIM
BETA = 10.0
DTAU = 0.05
INPUT_DIM = int(BETA / DTAU)  # 200
N_NODES = 256
BATCH_SIZE = 32

# --- Pretrained checkpoint ---
PRETRAIN_ID = "pretrained"
PRETRAIN_DIR = os.path.join(
    MAIN_PATH, "VAE_Library",
    f"pretrain_synthetic_numpoles{NUM_POLES}-{PRETRAIN_ID}"
)
PRETRAIN_CHECKPOINT = os.path.join(PRETRAIN_DIR, "model", "best_model_pretrain.pth")

# --- Target dataset ---
SPECTRAL_TYPE = "gaussian_double"
NOISE_S = 1e-05
NOISE_XI = 0.5
INPUT_ID = "inputs-7"  # Which inputs folder to use
DATA_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    f"Gbins_s{NOISE_S:.0e}_xi{NOISE_XI}.csv"
)

# --- Covariance source (same dataset used for chi2 loss) ---
COVARIANCE_SOURCE = DATA_PATH  # Use the same Gbins file

# --- Ground truth spectral function (for evaluation plots) ---
SPECTRAL_INPUT_PATH = os.path.join(
    MAIN_PATH, "Data", "datasets",
    f"half-filled-{SPECTRAL_TYPE}", INPUT_ID,
    "spectral_input.csv"
)

# --- Fine-tuning hyperparameters ---
FINETUNE_EPOCHS = 200
FINETUNE_LR = 3e-3
FINETUNE_PATIENCE = 15
sID = "pretrained"

# Loss weights
LAMBDA_CHI2 = 1.0         # Covariance-weighted reconstruction
LAMBDA_SMOOTH = 1e-3      # Smoothness on predicted A(omega)
LAMBDA_POS = 1.0          # Spectral positivity penalty (penalizes negative A(omega))
ALPHA_KL = 0.001          # KL divergence
ETA0 = 1.0                # Negative Green's penalty
ETA2 = 1.0                # Negative second derivative penalty

# Cosine annealing warm restarts
T_0 = 50                  # Period of first restart (epochs)
T_MULT = 2                # Factor by which T_0 increases after each restart

# Spectral smoothness parameters
SMOOTHNESS_NW = 500
SMOOTHNESS_WMIN = -8.0
SMOOTHNESS_WMAX = 8.0


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Model: NUM_POLES={NUM_POLES}, LATENT_DIM={LATENT_DIM}, INPUT_DIM={INPUT_DIM}")

    # ------------------------------------------------------------------
    # Initialize model and load pretrained weights
    # ------------------------------------------------------------------
    model = VariationalAutoEncoder2(
        input_dim=INPUT_DIM,
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        latent_dim=LATENT_DIM,
        hidden_dim3=HIDDEN_DIM3,
        hidden_dim4=HIDDEN_DIM4,
        num_poles=NUM_POLES,
        beta=BETA,
        N_nodes=N_NODES,
    ).to(DEVICE)

    # Load pretrained checkpoint
    if os.path.exists(PRETRAIN_CHECKPOINT):
        print(f"\nLoading pretrained model from: {PRETRAIN_CHECKPOINT}")
        model.load_state_dict(torch.load(PRETRAIN_CHECKPOINT, map_location=DEVICE, weights_only=True))
        print("Pretrained weights loaded successfully.")
    else:
        print(f"\nWARNING: Pretrained checkpoint not found at {PRETRAIN_CHECKPOINT}")
        print("Starting from scratch (random initialization).\n")

    # ------------------------------------------------------------------
    # STAGE 2: Fine-tuning on target dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STAGE 2: FINE-TUNING ON {SPECTRAL_TYPE.upper()} DATA")
    print("=" * 80 + "\n")

    tag = "finetune"

    # --- Output directory ---
    ft_out_dir = os.path.join(
        MAIN_PATH, "VAE_Library",
        f"finetune_{SPECTRAL_TYPE}_numpoles{NUM_POLES}_s{NOISE_S:.0e}_xi{NOISE_XI}-{sID}"
    )
    MakeOutPath(ft_out_dir)

    # --- Load dataset ---
    print(f"Loading dataset from: {DATA_PATH}")
    ft_dataset = GreenFunctionDataset(file_path=DATA_PATH)
    std = STD(ft_dataset)

    train_dataset, val_dataset, test_dataset = TrainTestSplit(
        ft_dataset, train_ratio=0.8, val_ratio=0.1
    )
    train_loader, val_loader, test_loader = LoadData(
        train_dataset, val_dataset, test_dataset,
        batch_size=BATCH_SIZE, shuffle=True
    )
    PrintDataSize(ft_dataset, train_dataset, val_dataset, test_dataset, train_loader)

    # --- Load covariance and build loss modules ---
    print(f"Loading covariance from: {COVARIANCE_SOURCE}")
    C = load_covariance_from_dqmc(COVARIANCE_SOURCE)

    chi2_fn = ChiSquaredLoss(C).to(DEVICE)
    smoothness_fn = SpectralSmoothnessLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)
    positivity_fn = SpectralPositivityLoss(
        Nw=SMOOTHNESS_NW, wmin=SMOOTHNESS_WMIN, wmax=SMOOTHNESS_WMAX
    ).to(DEVICE)

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT)

    # --- Train ---
    print(f"\nLoss weights: lambda_chi2={LAMBDA_CHI2}, lambda_smooth={LAMBDA_SMOOTH}, "
          f"lambda_pos={LAMBDA_POS}, alpha_kl={ALPHA_KL}, eta0={ETA0}, eta2={ETA2}")
    print(f"Scheduler: CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_MULT})")

    best_val_loss, best_epoch = train_finetune(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=FINETUNE_EPOCHS,
        input_dim=INPUT_DIM,
        train_loader=train_loader,
        val_loader=val_loader,
        std=std,
        chi2_fn=chi2_fn,
        smoothness_fn=smoothness_fn,
        positivity_fn=positivity_fn,
        lambda_chi2=LAMBDA_CHI2,
        lambda_smooth=LAMBDA_SMOOTH,
        lambda_pos=LAMBDA_POS,
        alpha_kl=ALPHA_KL,
        eta0=ETA0,
        eta2=ETA2,
        device=DEVICE,
        out_dir=ft_out_dir,
        tag=tag,
        patience=FINETUNE_PATIENCE,
    )

    print(f"\nFine-tuning done. Best val loss: {best_val_loss:.4e} at epoch {best_epoch+1}")

    # Load best fine-tuned model
    model.load_state_dict(torch.load(f"{ft_out_dir}/model/best_model_{tag}.pth", weights_only=True))

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\n--- Test evaluation ---")
    model.eval()
    test_loss = 0.0
    G_input_all, G_recon_all = [], []
    poles_all, residues_all = [], []

    with torch.no_grad():
        for batch in test_loader:
            B = batch.shape[0]
            batch = batch.view(B, INPUT_DIM).to(DEVICE)

            mu, logvar, z, poles, residues, G_recon = model(batch)

            from pretrain.train_finetune import finetune_total_loss  # type: ignore
            loss, _, _, _, _, _, _ = finetune_total_loss(
                G_recon, batch,
                mu, logvar,
                poles, residues,
                std,
                chi2_fn, smoothness_fn, positivity_fn,
                LAMBDA_CHI2, LAMBDA_SMOOTH, LAMBDA_POS, ALPHA_KL, ETA0, ETA2,
            )
            test_loss += loss.item() * B

            G_input_all.append(batch.cpu())
            G_recon_all.append(G_recon.cpu())
            poles_all.append(poles.cpu())
            residues_all.append(residues.cpu())

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4e}")

    G_input_all = torch.cat(G_input_all, dim=0)
    G_recon_all = torch.cat(G_recon_all, dim=0)
    poles_all = torch.cat(poles_all, dim=0)
    residues_all = torch.cat(residues_all, dim=0)

    torch.save({
        "inputs": G_input_all,
        "recon": G_recon_all,
        "poles": poles_all,
        "residues": residues_all,
        "inputs_avg": G_input_all.mean(dim=0),
        "recon_avg": G_recon_all.mean(dim=0),
        "poles_avg": poles_all.mean(dim=0),
        "residues_avg": residues_all.mean(dim=0),
        "beta": BETA,
        "Ltau": INPUT_DIM,
        "spectral_input_path": SPECTRAL_INPUT_PATH,
    }, f"{ft_out_dir}/summary.pt")

    print(f"Summary saved to {ft_out_dir}/summary.pt")
    print(f"Poles mean: {poles_all.mean(dim=0)}")
    print(f"Residues mean: {residues_all.mean(dim=0)}")

    # --- Plot fine-tuning results ---
    print("\n--- Generating fine-tuning plots ---")
    plot_finetune_eval(f"{ft_out_dir}/summary.pt")
    plot_loss_curves(f"{ft_out_dir}/losses", tag=tag,
                     save_path=f"{ft_out_dir}/plots/loss_curves_{tag}.pdf")


if __name__ == "__main__":
    main()
